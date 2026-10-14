/*
 * stellar-ai 自检核心（UMD，Node/浏览器共用）
 * runAll(C, M, logger) -> {pass, fail}
 * C = autograd API, M = models API
 */
(function (global) {
  'use strict';

  function runAll(C, M, log) {
    const { Tensor, matmul, add, mul, sub, transpose, sumAll, sumRows,
      tanh, sigmoid, silu, softmaxRows, logsumexpRows, rmsNorm, rope } = C;

    let pass = 0, fail = 0;
    const check = (name, ok, info = '') => {
      log(`${ok ? '✅' : '❌'} ${name}${info ? '  ' + info : ''}`);
      ok ? pass++ : fail++;
    };

    // 有限差分梯度检验
    function gradCheck(name, buildLoss, inputs, threshold = 1e-3) {
      const loss = buildLoss(inputs);
      loss.backward();
      const h = 1e-5;
      let maxRel = 0;
      for (const inp of inputs) {
        const ana = inp.grad;
        for (let j = 0; j < inp.data.length; j++) {
          const orig = inp.data[j];
          inp.data[j] = orig + h; const lp = buildLoss(inputs).data[0];
          inp.data[j] = orig - h; const lm = buildLoss(inputs).data[0];
          inp.data[j] = orig;
          const n = (lp - lm) / (2 * h);
          const a = ana[j];
          const rel = Math.abs(a - n) / (Math.abs(a) + Math.abs(n) + 1e-12);
          if (rel > maxRel) maxRel = rel;
        }
      }
      check(name, maxRel < threshold, `maxRelErr=${maxRel.toExponential(2)}`);
    }

    let seed = 12345;
    const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return (seed / 0x7fffffff) * 2 - 1; };
    const rt = (r, c) => { const d = new Float64Array(r * c); for (let i = 0; i < d.length; i++) d[i] = rnd(); return new Tensor(d, r, c); };

    log('=== 梯度检验（解析梯度 vs 有限差分）===');
    gradCheck('matmul', (I) => sumAll(matmul(I[0], I[1])), [rt(3, 4), rt(4, 2)]);
    gradCheck('add', (I) => sumAll(add(I[0], I[1])), [rt(3, 2), rt(3, 2)]);
    gradCheck('mul-bcast-scalar', (I) => sumAll(mul(I[0], I[1])), [rt(3, 2), Tensor.scalar(0.7)]);
    gradCheck('transpose', (I) => sumAll(matmul(transpose(I[0]), I[1])), [rt(2, 3), rt(2, 2)]);
    gradCheck('sumRows', (I) => sumAll(sumRows(I[0])), [rt(3, 2)]);
    gradCheck('tanh', (I) => sumAll(tanh(I[0])), [rt(3, 4)]);
    gradCheck('sigmoid', (I) => sumAll(sigmoid(I[0])), [rt(3, 4)]);
    gradCheck('silu', (I) => sumAll(silu(I[0])), [rt(3, 4)]);
    gradCheck('softmaxRows', (I) => sumAll(matmul(softmaxRows(I[0]), I[1])), [rt(3, 4), rt(4, 4)]);
    gradCheck('logsumexpRows', (I) => sumAll(logsumexpRows(I[0])), [rt(3, 4)]);
    gradCheck('rmsNorm', (I) => sumAll(rmsNorm(I[0], I[1])), [rt(3, 4), Tensor.ones(1, 4)]);
    gradCheck('rope', (I) => sumAll(rope(I[0], [0, 1, 2, 3, 4])), [rt(5, 4)]);
    gradCheck('sliceCols', (I) => sumAll(M.sliceCols(I[0], 1, 2)), [rt(3, 6)]);
    gradCheck('ropeMultiHead', (I) => sumAll(M.ropeMultiHead(I[0], [0, 1, 2, 3], 2)), [rt(4, 8)]);
    gradCheck('crossEntropy', (I) => M.crossEntropy(I[0], [3, 1, 0, 2]), [rt(4, 5)], 1e-3);
    {
      const X = new Tensor(new Float64Array([0.21, -0.37]), 2, 1);
      const kan = new M.KANLayer(1, 1, -1, 1, 5, 3);
      gradCheck('KANLayer', (I) => sumAll(kan.forward(I[0])), [X, kan.coeff, kan.baseW], 1e-3);
    }

    log('\n=== 不变量 ===');
    // RoPE 保范数
    {
      const X = rt(5, 6), Y = rope(X, [0, 1, 2, 3, 4]);
      let md = 0;
      for (let i = 0; i < 5; i++) { let a = 0, b = 0; for (let j = 0; j < 6; j++) { a += X.data[i * 6 + j] ** 2; b += Y.data[i * 6 + j] ** 2; } md = Math.max(md, Math.abs(Math.sqrt(a) - Math.sqrt(b))); }
      check('RoPE 保范数(正交)', md < 1e-12, `maxDiff=${md.toExponential(2)}`);
    }
    // softmax 行和=1
    {
      const S = softmaxRows(rt(4, 5));
      let me = 0;
      for (let i = 0; i < 4; i++) { let s = 0; for (let j = 0; j < 5; j++) s += S.data[i * 5 + j]; me = Math.max(me, Math.abs(s - 1)); }
      check('softmax 行和=1', me < 1e-12, `maxErr=${me.toExponential(2)}`);
    }
    // GQA 形状
    {
      const attn = new M.MultiHeadAttention(64, 4, 2);
      const y = attn.forward(rt(8, 64), [0, 1, 2, 3, 4, 5, 6, 7], true);
      check('GQA 输出形状 [seq,dModel]', y.rows === 8 && y.cols === 64, `[${y.rows},${y.cols}]`);
    }
    // KAN 学 sin（全批量训练，稳定收敛）
    {
      const kan = new M.KANLayer(1, 1, -1, 1, 10, 3), opt = new M.Adam(kan.params(), 1e-1);
      const N = 64, xs = new Float64Array(N), ys = new Float64Array(N);
      for (let i = 0; i < N; i++) { const x = -1 + 2 * i / (N - 1); xs[i] = x; ys[i] = Math.sin(3 * x); }
      const Xb = new Tensor(xs.slice(), N, 1);
      const Yb = new Tensor(ys.slice(), N, 1);
      const mse = () => { const p = kan.forward(Xb); let e = 0; for (let i = 0; i < N; i++) e += (p.data[i] - ys[i]) ** 2; return e / N; };
      const before = mse();
      for (let s = 0; s < 1500; s++) {
        const p = kan.forward(Xb); const d = sub(p, Yb);
        const loss = mul(sumAll(mul(d, d)), Tensor.scalar(1 / N));
        loss.backward(); opt.step();
      }
      const after = mse();
      check('KAN 拟合 sin：MSE 下降', after < before * 0.1 && after < 0.05, `${before.toFixed(4)} -> ${after.toFixed(4)}`);
    }
    // GPT 玩具训练
    {
      const text = 'hello world hello world hello stellar ai ';
      const vocab = [...new Set(text.split(''))].sort();
      const stoi = {}; vocab.forEach((c, i) => stoi[c] = i);
      const itos = {}; vocab.forEach((c, i) => itos[i] = c);
      const data = text.split('').map((c) => stoi[c]);
      const gpt = new M.GPT(vocab.length, 48, 3, 2, 1, 24, false), opt = new M.Adam(gpt.params(), 1e-2);
      const seqLen = 16;
      const lossOf = () => { const s = Math.floor(Math.random() * (data.length - seqLen)); const idx = data.slice(s, s + seqLen); return M.crossEntropy(gpt.forward(idx), idx.slice(1).concat(idx[0])).data[0]; };
      const before = lossOf();
      for (let s = 0; s < 400; s++) { const st = Math.floor(Math.random() * (data.length - seqLen)); const idx = data.slice(st, st + seqLen); const loss = M.crossEntropy(gpt.forward(idx), idx.slice(1).concat(idx[0])); loss.backward(); opt.step(); }
      const after = lossOf();
      check('GPT 玩具训练 loss 下降', after < before, `${before.toFixed(3)} -> ${after.toFixed(3)}`);
      let ctx = [stoi['h']];
      for (let i = 0; i < 12; i++) { const lg = gpt.forward(ctx.slice(-seqLen)).data; const last = ctx.length >= seqLen ? seqLen - 1 : ctx.length - 1; const row = last * vocab.length; let best = 0; for (let v = 1; v < vocab.length; v++) if (lg[row + v] > lg[row + best]) best = v; ctx.push(best); }
      check('GPT 可采样输出', ctx.length === 13, `"${ctx.map((i) => itos[i]).join('')}"`);
    }

    log(`\n=== 结果：${pass} 通过 / ${fail} 失败 ===`);
    return { pass, fail };
  }

  const API = { runAll };
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else global.StellarChecks = API;
})(typeof window !== 'undefined' ? window : globalThis);

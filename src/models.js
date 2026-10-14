/*
 * stellar-ai — 世界顶级 AI 构件集 (集大成)
 * 复用 SOTA 研究：RoPE(RoFormer), RMSNorm, GQA, SwiGLU, MoE, KAN(2024)。
 * 全部基于 src/autograd.js 的可微内核，可端到端反向传播。
 */
(function (global) {
  'use strict';
  const C = (typeof module !== 'undefined' && module.exports)
    ? require('./autograd') : global.StellarCore;
  const { Tensor, matmul, add, sub, mul, transpose, sumAll, sumRows,
    silu, softmaxRows, logsumexpRows, rope, EPS_NORM } = C;

  // ---------- 工具算子 ----------
  // 列切片：out[i,j] = X[i, start+j]
  function sliceCols(X, start, width) {
    const out = new Float64Array(X.rows * width);
    const xd = X.data;
    for (let i = 0; i < X.rows; i++)
      for (let j = 0; j < width; j++) out[i * width + j] = xd[i * X.cols + start + j];
    const r = new Tensor(out, X.rows, width);
    r._prev = [X];
    r._backward = () => {
      const g = r.grad, gx = X.grad || (X.grad = new Float64Array(X.data.length));
      for (let i = 0; i < X.rows; i++)
        for (let j = 0; j < width; j++) gx[i * X.cols + start + j] += g[i * width + j];
    };
    return r;
  }

  // 多头 RoPE：X [seq, n_heads*head_dim]，按头块内相邻维度旋转
  function ropeMultiHead(X, positions, nHeads, base = 10000) {
    const seq = X.rows, total = X.cols;
    const headDim = total / nHeads;
    const half = headDim / 2;
    const out = new Float64Array(X.data.length);
    const xd = X.data;
    const cosT = new Float64Array(seq * nHeads * half);
    const sinT = new Float64Array(seq * nHeads * half);
    for (let h = 0; h < nHeads; h++) {
      const baseOff = h * headDim;
      for (let i = 0; i < seq; i++) {
        const p = positions[i];
        const off = i * total + baseOff;
        for (let k = 0; k < half; k++) {
          const theta = p * Math.pow(base, -(2 * k) / headDim);
          const c = Math.cos(theta), s = Math.sin(theta);
          const x0 = xd[off + 2 * k], x1 = xd[off + 2 * k + 1];
          out[off + 2 * k] = x0 * c - x1 * s;
          out[off + 2 * k + 1] = x0 * s + x1 * c;
          cosT[(h * seq + i) * half + k] = c;
          sinT[(h * seq + i) * half + k] = s;
        }
      }
    }
    const r = new Tensor(out, seq, total);
    r._prev = [X];
    r._backward = () => {
      const g = r.grad, gx = X.grad || (X.grad = new Float64Array(X.data.length));
      for (let h = 0; h < nHeads; h++) {
        const baseOff = h * headDim;
        for (let i = 0; i < seq; i++) {
          const off = i * total + baseOff;
          for (let k = 0; k < half; k++) {
            const c = cosT[(h * seq + i) * half + k], s = sinT[(h * seq + i) * half + k];
            const gy0 = g[off + 2 * k], gy1 = g[off + 2 * k + 1];
            gx[off + 2 * k] += gy0 * c + gy1 * s;
            gx[off + 2 * k + 1] += -gy0 * s + gy1 * c;
          }
        }
      }
    };
    return r;
  }

  // 交叉熵（logsumexp 稳定）：logits [N,V], targets Int[N] -> 标量 loss
  function crossEntropy(logits, targets) {
    const N = logits.rows, V = logits.cols;
    const sm = softmaxRows(logits);
    let loss = 0;
    for (let i = 0; i < N; i++) {
      const lse = logsumexpRows(sliceRows(logits, i));
      loss += lse.data[0] - logits.data[i * V + targets[i]];
    }
    loss /= N;
    const r = Tensor.scalar(loss);
    r._prev = [logits];
    r._backward = () => {
      const g = r.grad[0] / N;
      const gl = logits.grad || (logits.grad = new Float64Array(logits.data.length));
      const sd = sm.data;
      for (let i = 0; i < N; i++) {
        const off = i * V, t = targets[i];
        for (let j = 0; j < V; j++) {
          const onehot = (j === t) ? 1 : 0;
          gl[off + j] += g * (sd[off + j] - onehot);
        }
      }
    };
    return r;
  }
  function sliceRows(X, i) {
    const out = new Float64Array(X.cols);
    for (let j = 0; j < X.cols; j++) out[j] = X.data[i * X.cols + j];
    const r = new Tensor(out, 1, X.cols);
    r._prev = [X];
    r._backward = () => {
      const g = r.grad, gx = X.grad || (X.grad = new Float64Array(X.data.length));
      for (let j = 0; j < X.cols; j++) gx[i * X.cols + j] += g[j];
    };
    return r;
  }

  // ---------- 线性层 ----------
  class Linear {
    constructor(inF, outF, bias = true) {
      this.W = Tensor.randn(inF, outF, 0, Math.sqrt(2 / (inF + outF)));
      this.W.requiresGrad = true;
      this.bias = bias;
      if (bias) { this.b = Tensor.zeros(1, outF); this.b.requiresGrad = true; }
    }
    forward(x) {
      const o = matmul(x, this.W);
      return this.bias ? add(o, this.b) : o;
    }
    params() { return this.bias ? [this.W, this.b] : [this.W]; }
  }

  // ---------- RMSNorm 层（封装内核 op） ----------
  class RMSNorm {
    constructor(dim) { this.g = Tensor.ones(1, dim); this.g.requiresGrad = true; }
    forward(x) { return C.rmsNorm(x, this.g); }
    params() { return [this.g]; }
  }

  // ---------- 多头注意力（支持 GQA） ----------
  class MultiHeadAttention {
    constructor(dModel, nHeads, nKvHeads = null, base = 10000) {
      this.nHeads = nHeads;
      this.nKvHeads = nKvHeads || nHeads;
      this.headDim = dModel / nHeads;
      this.group = nHeads / this.nKvHeads;
      this.base = base;
      this.Wq = new Linear(dModel, dModel, false);
      this.Wk = new Linear(dModel, this.nKvHeads * this.headDim, false);
      this.Wv = new Linear(dModel, this.nKvHeads * this.headDim, false);
      this.Wo = new Linear(dModel, dModel, false);
    }
    forward(x, positions, causal = true) {
      const seq = x.rows;
      const Q = this.Wq.forward(x);
      const K = this.Wk.forward(x);
      const V = this.Wv.forward(x);
      const Qr = ropeMultiHead(Q, positions, this.nHeads, this.base);
      const Kr = ropeMultiHead(K, positions, this.nKvHeads, this.base);
      const headDim = this.headDim;
      // 因果掩码
      let mask = null;
      if (causal) {
        const m = new Float64Array(seq * seq);
        for (let i = 0; i < seq; i++) for (let j = 0; j < seq; j++) if (j > i) m[i * seq + j] = -1e9;
        mask = new Tensor(m, seq, seq);
      }
      const scale = Tensor.scalar(1 / Math.sqrt(headDim));
      let acc = null;
      for (let h = 0; h < this.nHeads; h++) {
        const kvh = Math.floor(h / this.group);
        const Qh = sliceCols(Qr, h * headDim, headDim);
        const Kh = sliceCols(Kr, kvh * headDim, headDim);
        const Vh = sliceCols(V, kvh * headDim, headDim);
        let scores = matmul(Qh, transpose(Kh));
        scores = mul(scores, scale);
        if (mask) scores = add(scores, mask);
        const attn = softmaxRows(scores);
        const ctx = matmul(attn, Vh);
        const cdata = new Float64Array(seq * x.cols);
        for (let i = 0; i < seq; i++)
          for (let j = 0; j < headDim; j++) cdata[i * x.cols + h * headDim + j] = ctx.data[i * headDim + j];
        const cT = new Tensor(cdata, seq, x.cols);
        acc = acc === null ? cT : add(acc, cT);
      }
      const projected = this.Wo.forward(acc);
      return projected;
    }
    params() {
      return [...this.Wq.params(), ...this.Wk.params(), ...this.Wv.params(), ...this.Wo.params()];
    }
  }

  // ---------- SwiGLU 前馈 (Shazeer 2020) ----------
  class SwiGLU {
    constructor(dModel, ffnMult = 4) {
      const ff = dModel * ffnMult;
      this.Wgate = new Linear(dModel, ff, false);
      this.Wup = new Linear(dModel, ff, false);
      this.Wdown = new Linear(ff, dModel, false);
    }
    forward(x) {
      const g = silu(this.Wgate.forward(x));
      const u = this.Wup.forward(x);
      const h = mul(g, u);
      return this.Wdown.forward(h);
    }
    params() {
      return [...this.Wgate.params(), ...this.Wup.params(), ...this.Wdown.params()];
    }
  }

  // ---------- MoE (Top-2 路由, 软权重可微) ----------
  class MoE {
    constructor(dModel, nExperts = 4, ffnMult = 4, k = 2) {
      this.nExperts = nExperts; this.k = k;
      const hid = dModel * ffnMult;
      this.Wgate = new Linear(dModel, nExperts, false);
      this.experts = [];
      for (let e = 0; e < nExperts; e++) {
        this.experts.push({
          W1: new Linear(dModel, hid, false),
          W2: new Linear(hid, dModel, false)
        });
      }
    }
    forward(x) {
      const seq = x.rows;
      const logits = this.Wgate.forward(x); // [seq, E]
      const w = softmaxRows(logits);
      // top-k 掩码（硬掩码 * 软权重，仍可微）
      const wd = w.data, E = this.nExperts;
      const mask = new Float64Array(seq * E);
      for (let i = 0; i < seq; i++) {
        const arr = [];
        for (let e = 0; e < E; e++) arr.push([e, wd[i * E + e]]);
        arr.sort((a, b) => b[1] - a[1]);
        for (let t = 0; t < this.k; t++) mask[i * E + arr[t][0]] = 1;
      }
      const mT = new Tensor(mask, seq, E);
      const wm = mul(w, mT); // [seq,E]
      let acc = null;
      for (let e = 0; e < E; e++) {
        const we = sliceCols(wm, e, 1); // [seq,1]
        const ex = this.experts[e];
        let h = silu(ex.W1.forward(x));
        h = ex.W2.forward(h); // [seq, dModel]
        const contrib = mul(h, we); // broadcast [seq,dModel]
        acc = acc === null ? contrib : add(acc, contrib);
      }
      return acc;
    }
    params() {
      let p = [...this.Wgate.params()];
      for (const ex of this.experts) p.push(...ex.W1.params(), ...ex.W2.params());
      return p;
    }
  }

  // ---------- Transformer Block ----------
  class TransformerBlock {
    constructor(dModel, nHeads, nKvHeads, ffnMult, useMoE = false, nExperts = 4) {
      this.ln1 = new RMSNorm(dModel);
      this.attn = new MultiHeadAttention(dModel, nHeads, nKvHeads);
      this.ln2 = new RMSNorm(dModel);
      this.ff = useMoE ? new MoE(dModel, nExperts, ffnMult) : new SwiGLU(dModel, ffnMult);
    }
    forward(x, positions, causal = true) {
      const a = this.attn.forward(this.ln1.forward(x), positions, causal);
      const x1 = add(x, a);
      const f = this.ff.forward(this.ln2.forward(x1));
      return add(x1, f);
    }
    params() { return [...this.ln1.params(), ...this.attn.params(), ...this.ln2.params(), ...this.ff.params()]; }
  }

  // ---------- GPT ----------
  class GPT {
    constructor(vocab, dModel = 64, nHeads = 4, nLayers = 2, nKvHeads = 2, ctxLen = 64, useMoE = false) {
      this.vocab = vocab; this.ctxLen = ctxLen;
      this.tokenEmb = Tensor.randn(vocab, dModel, 0, 0.02);
      this.tokenEmb.requiresGrad = true;
      this.blocks = [];
      for (let i = 0; i < nLayers; i++)
        this.blocks.push(new TransformerBlock(dModel, nHeads, nKvHeads, 4, useMoE, 4));
      this.lnF = new RMSNorm(dModel);
      this.head = new Linear(dModel, vocab, false); // 输出投射
    }
    forward(idx) { // idx: Int[seq]
      const seq = idx.length;
      const x = new Float64Array(seq * this.tokenEmb.cols);
      for (let i = 0; i < seq; i++) {
        const row = idx[i];
        for (let j = 0; j < this.tokenEmb.cols; j++) x[i * this.tokenEmb.cols + j] = this.tokenEmb.data[row * this.tokenEmb.cols + j];
      }
      let X = new Tensor(x, seq, this.tokenEmb.cols);
      const positions = Array.from({ length: seq }, (_, i) => i);
      for (const b of this.blocks) X = b.forward(X, positions, true);
      X = this.lnF.forward(X);
      return this.head.forward(X); // [seq, vocab]
    }
    params() {
      let p = [this.tokenEmb];
      for (const b of this.blocks) p.push(...b.params());
      p.push(...this.lnF.params(), ...this.head.params());
      return p;
    }
  }

  // ---------- KAN (Kolmogorov-Arnold Networks, Liu et al. 2024) ----------
  // B-样条基 (Cox-de Boor)，clamped 节点向量（端点重复 k 次）
  // 返回 {B, dB}：degree-k 基函数值与导数，长度 nB = G + k
  function bspline(x, grid, k) {
    const G = grid.length - 1;
    const nB = G + k;
    // 扩展节点向量：长度 G+1+2k，两端各重复 k 次
    const knots = new Float64Array(G + 1 + 2 * k);
    for (let i = 0; i < knots.length; i++) {
      if (i < k) knots[i] = grid[0];
      else if (i <= k + G) knots[i] = grid[i - k];
      else knots[i] = grid[G];
    }
    // degree-0
    let B = new Float64Array(knots.length - 1);
    for (let i = 0; i < B.length; i++)
      if (x >= knots[i] && x <= knots[i + 1]) B[i] = 1;
    // 逐级升阶并保留历史
    const levels = [B];
    for (let p = 1; p <= k; p++) {
      const nxt = new Float64Array(B.length - 1);
      for (let i = 0; i < nxt.length; i++) {
        const dL = knots[i + p] - knots[i];
        const dR = knots[i + p + 1] - knots[i + 1];
        const left = dL > 1e-12 ? (x - knots[i]) / dL : 0;
        const right = dR > 1e-12 ? (knots[i + p + 1] - x) / dR : 0;
        nxt[i] = B[i] * left + B[i + 1] * right;
      }
      levels.push(nxt);
      B = nxt;
    }
    const Bk = levels[k];
    const Bkm1 = levels[k - 1];
    const dB = new Float64Array(nB);
    for (let i = 0; i < nB; i++) {
      const dL = knots[i + k] - knots[i];
      const dR = knots[i + k + 1] - knots[i + 1];
      const termL = dL > 1e-12 ? Bkm1[i] / dL : 0;
      const termR = dR > 1e-12 ? Bkm1[i + 1] / dR : 0;
      dB[i] = k * (termL - termR);
    }
    return { B: Bk, dB };
  }

  class KANLayer {
    constructor(inF, outF, gridMin = -1, gridMax = 1, G = 5, k = 3) {
      this.inF = inF; this.outF = outF; this.k = k; this.G = G;
      this.nB = G + k;
      this.grid = new Float64Array(G + 1);
      for (let i = 0; i <= G; i++) this.grid[i] = gridMin + (gridMax - gridMin) * i / G;
      // 系数 + 基函数权重
      this.coeff = Tensor.randn(inF * outF * this.nB, 1, 0, 0.1);
      this.coeff.requiresGrad = true;
      this.baseW = Tensor.zeros(inF, outF);
      this.baseW.requiresGrad = true;
    }
    forward(X) { // X [seq, inF]
      const seq = X.rows;
      const out = new Float64Array(seq * this.outF);
      const cd = this.coeff.data, bd = this.baseW.data, nB = this.nB;
      // 缓存供反向
      const cacheB = [], cacheDB = [], cacheBase = [], cacheDBase = [];
      for (let i = 0; i < seq; i++) {
        for (let f = 0; f < this.inF; f++) {
          const xv = X.data[i * this.inF + f];
          const { B, dB } = bspline(xv, this.grid, this.k);
          const sv = siluVal(xv), dsv = siluDeriv(xv);
          cacheB.push(B); cacheDB.push(dB); cacheBase.push(sv); cacheDBase.push(dsv);
          for (let o = 0; o < this.outF; o++) {
            let v = 0;
            const coff = (f * this.outF + o) * nB;
            for (let b = 0; b < nB; b++) v += cd[coff + b] * B[b];
            v += bd[f * this.outF + o] * sv;
            out[i * this.outF + o] += v;
          }
        }
      }
      const r = new Tensor(out, seq, this.outF);
      r._prev = [X, this.coeff, this.baseW];
      r._backward = () => {
        const g = r.grad;
        const gx = X.grad || (X.grad = new Float64Array(X.data.length));
        const gc = this.coeff.grad || (this.coeff.grad = new Float64Array(this.coeff.data.length));
        const gb = this.baseW.grad || (this.baseW.grad = new Float64Array(this.baseW.data.length));
        let ci = 0;
        for (let i = 0; i < seq; i++) {
          for (let f = 0; f < this.inF; f++) {
            const B = cacheB[ci], dB = cacheDB[ci], sv = cacheBase[ci], dsv = cacheDBase[ci];
            ci++;
            let dxAccum = 0;
            for (let o = 0; o < this.outF; o++) {
              const gio = g[i * this.outF + o];
              const coff = (f * this.outF + o) * nB;
              for (let b = 0; b < nB; b++) {
                gc[coff + b] += B[b] * gio;
                dxAccum += cd[coff + b] * dB[b];
              }
              gb[f * this.outF + o] += sv * gio;
              dxAccum += bd[f * this.outF + o] * dsv;
            }
            gx[i * this.inF + f] += dxAccum;
          }
        }
      };
      return r;
    }
    params() { return [this.coeff, this.baseW]; }
  }
  function siluVal(x) { const s = 1 / (1 + Math.exp(-x)); return x * s; }
  function siluDeriv(x) { const s = 1 / (1 + Math.exp(-x)); return s * (1 + x * (1 - s)); }

  // ---------- Adam 优化器 ----------
  class Adam {
    constructor(params, lr = 1e-3, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
      this.params = params; this.lr = lr; this.b1 = b1; this.b2 = b2; this.eps = eps;
      this.t = 0;
      this.m = params.map(() => null);
      this.v = params.map(() => null);
    }
    step() {
      this.t++;
      for (let i = 0; i < this.params.length; i++) {
        const p = this.params[i];
        if (!p.grad) continue;
        if (!this.m[i]) { this.m[i] = new Float64Array(p.data.length); this.v[i] = new Float64Array(p.data.length); }
        const m = this.m[i], v = this.v[i], g = p.grad, d = p.data;
        for (let j = 0; j < d.length; j++) {
          m[j] = this.b1 * m[j] + (1 - this.b1) * g[j];
          v[j] = this.b2 * v[j] + (1 - this.b2) * g[j] * g[j];
          const mh = m[j] / (1 - Math.pow(this.b1, this.t));
          const vh = v[j] / (1 - Math.pow(this.b2, this.t));
          d[j] -= this.lr * mh / (Math.sqrt(vh) + this.eps);
        }
      }
    }
  }

  const API = {
    Linear, RMSNorm, MultiHeadAttention, SwiGLU, MoE, TransformerBlock, GPT,
    KANLayer, bspline, Adam, crossEntropy, rope, ropeMultiHead, sliceCols,
    Tensor, matmul, add, mul, transpose, softmaxRows
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else global.StellarModels = API;
})(typeof window !== 'undefined' ? window : globalThis);

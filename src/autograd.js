/*
 * stellar-ai — 零依赖可微张量内核 (autograd engine)
 * 纯 JS，无外部依赖。浏览器 + Node 双运行。
 * 设计： tape 式反向传播，每个算子前向存闭包反向。
 * 数学硬标准：有限差分梯度检验见 test/selftest.js。
 */
(function (global) {
  'use strict';

  const EPS_NORM = 1e-5;

  class Tensor {
    constructor(data, rows, cols, requiresGrad = false) {
      if (data instanceof Float64Array) {
        this.data = data;
        this.rows = rows;
        this.cols = cols;
      } else if (Array.isArray(data)) {
        this.data = Float64Array.from(data);
        this.rows = rows;
        this.cols = cols;
      } else {
        // scalar
        this.data = new Float64Array([data]);
        this.rows = 1;
        this.cols = 1;
      }
      this.grad = null;
      this.requiresGrad = requiresGrad || false;
      this._backward = () => {};
      this._prev = [];
    }

    static scalar(v) { return new Tensor(new Float64Array([v]), 1, 1); }
    static zeros(r, c) { return new Tensor(new Float64Array(r * c), r, c); }
    static ones(r, c) { const d = new Float64Array(r * c).fill(1); return new Tensor(d, r, c); }
    static randn(r, c, mean = 0, std = 1) {
      const d = new Float64Array(r * c);
      for (let i = 0; i < d.length; i++) d[i] = mean + std * gaussian();
      return new Tensor(d, r, c);
    }

    get length() { return this.data.length; }
    idx(i, j) { return i * this.cols + j; }

    clone() {
      const t = new Tensor(new Float64Array(this.data), this.rows, this.cols);
      t.requiresGrad = this.requiresGrad;
      return t;
    }

    // 反向传播入口：loss 为标量
    backward() {
      const topo = [];
      const visited = new Set();
      const build = (v) => {
        if (visited.has(v)) return;
        visited.add(v);
        for (const p of v._prev) build(p);
        topo.push(v);
      };
      build(this);
      for (const v of topo) {
        if (!v.grad) v.grad = new Float64Array(v.data.length);
        else v.grad.fill(0);
      }
      this.grad[0] = 1; // 标量 loss 种子
      for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
    }
  }

  function gaussian() {
    // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function ensureGrad(t) {
    if (!t.grad) t.grad = new Float64Array(t.data.length);
    return t.grad;
  }

  // ---- 基础算子 ----

  function matmul(a, b) {
    if (a.cols !== b.rows) throw new Error(`matmul shape mismatch ${a.rows}x${a.cols} @ ${b.rows}x${b.cols}`);
    const m = a.rows, k = a.cols, n = b.cols;
    const out = new Float64Array(m * n);
    const ad = a.data, bd = b.data;
    for (let i = 0; i < m; i++) {
      for (let t = 0; t < k; t++) {
        const av = ad[i * k + t];
        if (av === 0) continue;
        const broff = t * n;
        const ooff = i * n;
        for (let j = 0; j < n; j++) out[ooff + j] += av * bd[broff + j];
      }
    }
    const r = new Tensor(out, m, n);
    r._prev = [a, b];
    r._backward = () => {
      const g = ensureGrad(r);
      // dA = g @ B^T
      const ga = ensureGrad(a), gb = ensureGrad(b);
      for (let i = 0; i < m; i++) {
        for (let t = 0; t < k; t++) {
          let s = 0;
          const boff = t * n;
          const goff = i * n;
          for (let j = 0; j < n; j++) s += g[goff + j] * bd[boff + j];
          ga[i * k + t] += s;
        }
      }
      // dB = A^T @ g
      for (let t = 0; t < k; t++) {
        for (let j = 0; j < n; j++) {
          let s = 0;
          for (let i = 0; i < m; i++) s += ad[i * k + t] * g[i * n + j];
          gb[t * n + j] += s;
        }
      }
    };
    return r;
  }

  // 广播加法：支持 [1,1] 标量广播
  function add(a, b) {
    let r, c;
    if (b.rows === 1 && b.cols === 1) { r = a.rows; c = a.cols; }
    else if (a.rows === 1 && a.cols === 1) { r = b.rows; c = b.cols; }
    else if (a.rows === b.rows && a.cols === b.cols) { r = a.rows; c = a.cols; }
    else throw new Error('add broadcast mismatch');
    const out = new Float64Array(r * c);
    const ad = a.data, bd = b.data;
    if (b.rows === 1 && b.cols === 1) {
      const bv = bd[0];
      for (let i = 0; i < out.length; i++) out[i] = ad[i] + bv;
    } else if (a.rows === 1 && a.cols === 1) {
      const av = ad[0];
      for (let i = 0; i < out.length; i++) out[i] = av + bd[i];
    } else {
      for (let i = 0; i < out.length; i++) out[i] = ad[i] + bd[i];
    }
    const res = new Tensor(out, r, c);
    res._prev = [a, b];
    res._backward = () => {
      const g = ensureGrad(res);
      if (b.rows === 1 && b.cols === 1) {
        const ga = ensureGrad(a);
        for (let i = 0; i < g.length; i++) ga[i] += g[i];
        const gb = ensureGrad(b);
        let s = 0; for (let i = 0; i < g.length; i++) s += g[i]; gb[0] += s;
      } else if (a.rows === 1 && a.cols === 1) {
        const ga = ensureGrad(a);
        let s = 0; for (let i = 0; i < g.length; i++) s += g[i]; ga[0] += s;
        const gb = ensureGrad(b);
        for (let i = 0; i < g.length; i++) gb[i] += g[i];
      } else {
        const ga = ensureGrad(a), gb = ensureGrad(b);
        for (let i = 0; i < g.length; i++) { ga[i] += g[i]; gb[i] += g[i]; }
      }
    };
    return res;
  }

  // 广播逐元素乘
  function mul(a, b) {
    let r, c;
    if (b.rows === 1 && b.cols === 1) { r = a.rows; c = a.cols; }
    else if (a.rows === 1 && a.cols === 1) { r = b.rows; c = b.cols; }
    else if (a.rows === b.rows && a.cols === b.cols) { r = a.rows; c = a.cols; }
    else throw new Error('mul broadcast mismatch');
    const out = new Float64Array(r * c);
    const ad = a.data, bd = b.data;
    if (b.rows === 1 && b.cols === 1) {
      const bv = bd[0];
      for (let i = 0; i < out.length; i++) out[i] = ad[i] * bv;
    } else if (a.rows === 1 && a.cols === 1) {
      const av = ad[0];
      for (let i = 0; i < out.length; i++) out[i] = av * bd[i];
    } else {
      for (let i = 0; i < out.length; i++) out[i] = ad[i] * bd[i];
    }
    const res = new Tensor(out, r, c);
    res._prev = [a, b];
    res._backward = () => {
      const g = ensureGrad(res);
      const ga = ensureGrad(a), gb = ensureGrad(b);
      if (b.rows === 1 && b.cols === 1) {
        const bv = bd[0];
        for (let i = 0; i < g.length; i++) ga[i] += g[i] * bv;
        let s = 0; for (let i = 0; i < g.length; i++) s += g[i] * ad[i]; gb[0] += s;
      } else if (a.rows === 1 && a.cols === 1) {
        const av = ad[0];
        let s = 0; for (let i = 0; i < g.length; i++) s += g[i] * bd[i]; ga[0] += s;
        for (let i = 0; i < g.length; i++) gb[i] += g[i] * av;
      } else {
        for (let i = 0; i < g.length; i++) { ga[i] += g[i] * bd[i]; gb[i] += g[i] * ad[i]; }
      }
    };
    return res;
  }

  function sub(a, b) { return add(a, mul(b, Tensor.scalar(-1))); }

  function transpose(a) {
    const out = new Float64Array(a.rows * a.cols);
    const ad = a.data;
    for (let i = 0; i < a.rows; i++)
      for (let j = 0; j < a.cols; j++)
        out[j * a.rows + i] = ad[i * a.cols + j];
    const r = new Tensor(out, a.cols, a.rows);
    r._prev = [a];
    r._backward = () => {
      const g = ensureGrad(r), ga = ensureGrad(a);
      for (let i = 0; i < a.rows; i++)
        for (let j = 0; j < a.cols; j++)
          ga[i * a.cols + j] += g[j * a.rows + i];
    };
    return r;
  }

  // 全 sum（标量输出）
  function sumAll(a) {
    let s = 0;
    for (let i = 0; i < a.data.length; i++) s += a.data[i];
    const r = Tensor.scalar(s);
    r._prev = [a];
    r._backward = () => {
      const g = ensureGrad(r)[0];
      const ga = ensureGrad(a);
      for (let i = 0; i < ga.length; i++) ga[i] += g;
    };
    return r;
  }

  // 沿行求和 -> [1, cols]
  function sumRows(a) {
    const out = new Float64Array(a.cols);
    for (let i = 0; i < a.rows; i++)
      for (let j = 0; j < a.cols; j++) out[j] += a.data[i * a.cols + j];
    const r = new Tensor(out, 1, a.cols);
    r._prev = [a];
    r._backward = () => {
      const g = ensureGrad(r), ga = ensureGrad(a);
      for (let i = 0; i < a.rows; i++)
        for (let j = 0; j < a.cols; j++) ga[i * a.cols + j] += g[j];
    };
    return r;
  }

  // 逐元素一元 op 工厂
  function unary(a, fwd, bwdFn) {
    const out = new Float64Array(a.data.length);
    for (let i = 0; i < out.length; i++) out[i] = fwd(a.data[i]);
    const r = new Tensor(out, a.rows, a.cols);
    r._prev = [a];
    r._backward = () => {
      const g = ensureGrad(r), ga = ensureGrad(a);
      for (let i = 0; i < out.length; i++) ga[i] += g[i] * bwdFn(a.data[i], out[i]);
    };
    return r;
  }

  const tanh = (a) => unary(a, (x) => Math.tanh(x), (x, y) => 1 - y * y);
  const sigmoid = (a) => unary(a, (x) => 1 / (1 + Math.exp(-x)), (x, y) => y * (1 - y));
  const relu = (a) => unary(a, (x) => (x > 0 ? x : 0), (x) => (x > 0 ? 1 : 0));
  // SiLU / Swish: x * sigmoid(x)
  function silu(a) {
    const s = sigmoid(a);
    return mul(a, s);
  }

  // 行归一化 softmax（数值稳定）
  function softmaxRows(a) {
    const out = new Float64Array(a.data.length);
    const ad = a.data;
    for (let i = 0; i < a.rows; i++) {
      let m = -Infinity;
      const off = i * a.cols;
      for (let j = 0; j < a.cols; j++) if (ad[off + j] > m) m = ad[off + j];
      let s = 0;
      for (let j = 0; j < a.cols; j++) { const e = Math.exp(ad[off + j] - m); out[off + j] = e; s += e; }
      for (let j = 0; j < a.cols; j++) out[off + j] /= s;
    }
    const r = new Tensor(out, a.rows, a.cols);
    r._prev = [a];
    r._backward = () => {
      const g = ensureGrad(r), ga = ensureGrad(a);
      const od = r.data;
      for (let i = 0; i < a.rows; i++) {
        const off = i * a.cols;
        let dot = 0;
        for (let j = 0; j < a.cols; j++) dot += od[off + j] * g[off + j];
        for (let j = 0; j < a.cols; j++)
          ga[off + j] += od[off + j] * (g[off + j] - dot);
      }
    };
    return r;
  }

  // logsumexp 行方向（稳定），用于交叉熵
  function logsumexpRows(a) {
    const out = new Float64Array(a.rows);
    const ad = a.data;
    for (let i = 0; i < a.rows; i++) {
      let m = -Infinity;
      const off = i * a.cols;
      for (let j = 0; j < a.cols; j++) if (ad[off + j] > m) m = ad[off + j];
      let s = 0;
      for (let j = 0; j < a.cols; j++) s += Math.exp(ad[off + j] - m);
      out[i] = m + Math.log(s);
    }
    const r = new Tensor(out, a.rows, 1);
    r._prev = [a];
    r._backward = () => {
      const g = ensureGrad(r), ga = ensureGrad(a);
      const sm = softmaxRows(a);
      const sd = sm.data;
      for (let i = 0; i < a.rows; i++) {
        const off = i * a.cols;
        for (let j = 0; j < a.cols; j++) ga[off + j] += g[i] * sd[off + j];
      }
    };
    return r;
  }

  // RMSNorm: y = x / rms(x) * g, rms = sqrt(mean(x^2)+eps)
  function rmsNorm(a, g, eps = EPS_NORM) {
    const n = a.cols;
    const out = new Float64Array(a.data.length);
    const ad = a.data, gd = g.data;
    for (let i = 0; i < a.rows; i++) {
      const off = i * n;
      let ss = 0;
      for (let j = 0; j < n; j++) ss += ad[off + j] * ad[off + j];
      const r = Math.sqrt(ss / n + eps);
      for (let j = 0; j < n; j++) out[off + j] = (ad[off + j] / r) * gd[j];
    }
    const r = new Tensor(out, a.rows, a.cols);
    r._prev = [a, g];
    r._backward = () => {
      const gG = ensureGrad(r), ga = ensureGrad(a), gg = ensureGrad(g);
      const ad = a.data;
      for (let i = 0; i < a.rows; i++) {
        const off = i * n;
        let ss = 0;
        for (let j = 0; j < n; j++) ss += ad[off + j] * ad[off + j];
        const r = Math.sqrt(ss / n + eps);
        const invR = 1 / r;
        // dz/dx_j = (1/r)(I - z z^T), z = x/r, then dy = g*dz
        let a_sum = 0;
        for (let j = 0; j < n; j++) a_sum += gG[off + j] * gd[j] * ad[off + j];
        for (let j = 0; j < n; j++) {
          const dz = gG[off + j] * gd[j]; // dL/dz_j
          ga[off + j] += (dz * invR) - (ad[off + j] * a_sum) / (n * r * r * r);
          gg[j] += gG[off + j] * (ad[off + j] * invR);
        }
      }
    };
    return r;
  }

  /*
   * RoPE 旋转位置编码 (Su et al. 2021, RoFormer)
   * X: [seq, dim] dim 偶数；positions: number[seq]
   * 将相邻维度 (2k,2k+1) 按角度 theta_k = pos * base^(-2k/dim) 旋转。
   * 旋转为正交线性映射 => 反向即施加 -theta 的逆旋转。
   */
  function rope(X, positions, base = 10000) {
    const seq = X.rows, dim = X.cols;
    if (dim % 2 !== 0) throw new Error('rope dim must be even');
    const half = dim / 2;
    const out = new Float64Array(X.data.length);
    const ad = X.data;
    const cosT = new Float64Array(seq * half);
    const sinT = new Float64Array(seq * half);
    for (let i = 0; i < seq; i++) {
      const p = positions[i];
      const off = i * dim;
      for (let k = 0; k < half; k++) {
        const theta = p * Math.pow(base, -(2 * k) / dim);
        const c = Math.cos(theta), s = Math.sin(theta);
        const x0 = ad[off + 2 * k], x1 = ad[off + 2 * k + 1];
        out[off + 2 * k] = x0 * c - x1 * s;
        out[off + 2 * k + 1] = x0 * s + x1 * c;
        cosT[i * half + k] = c; sinT[i * half + k] = s;
      }
    }
    const r = new Tensor(out, seq, dim);
    r._prev = [X];
    r._backward = () => {
      const g = ensureGrad(r), gx = ensureGrad(X);
      for (let i = 0; i < seq; i++) {
        const off = i * dim;
        for (let k = 0; k < half; k++) {
          const c = cosT[i * half + k], s = sinT[i * half + k];
          const gy0 = g[off + 2 * k], gy1 = g[off + 2 * k + 1];
          // 逆旋转 -theta
          gx[off + 2 * k] += gy0 * c + gy1 * s;
          gx[off + 2 * k + 1] += -gy0 * s + gy1 * c;
        }
      }
    };
    return r;
  }

  const API = {
    Tensor, matmul, add, sub, mul, transpose, sumAll, sumRows,
    tanh, sigmoid, relu, silu, softmaxRows, logsumexpRows, rmsNorm, rope,
    EPS_NORM, gaussian
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else global.StellarCore = API;
})(typeof window !== 'undefined' ? window : globalThis);

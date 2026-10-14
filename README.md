# ✦ stellar-ai

> 零依赖、纯 JavaScript、可端到端反向传播的**可验证 AI 内核**。
> 站在世界顶级数学与算法成果之上（tape 自动微分 / RoPE / GQA / SwiGLU / MoE / KAN），
> 每个算子都通过**有限差分梯度检验**与**数学不变量**硬校验。
> 浏览器一键自检 + Node 无头运行，双端 22/22 全绿。

**作者：晨星 (CJX0712)**

---

## 它是什么

一个**从零手写但不闭门造车**的微型深度学习框架：用最朴素的 `Float64Array` 搭出可微张量引擎，
在此之上实现当前大模型与前沿研究里的顶级构件，并用可独立复算的数学不变量证明"每一行都对"。

- **零外部依赖**：不引入任何 npm 包 / CDN，单文件可直接运行。
- **可验证**：解析梯度 vs 中心差分，`maxRelErr ≈ 1e-10`；不变量逐条断言。
- **可复现**：纯函数式前向 + 确定性反向，Node 无头一键跑通。
- **集大成 + 创新**：Transformer 全家桶 + 2024 的 KAN（Kolmogorov-Arnold 网络）。

---

## 快速开始

```bash
# Node 无头自检（22 项，应全部 ✅）
node test/selftest.js
```

或用浏览器直接打开 `index.html`，点击「▶ 运行自检」——加载的是**同一份源码、同一个 runner**。

---

## 架构

```
stellar-ai/
├── index.html          # 浏览器一键自检台（加载 src/ 与 test/checks.js）
├── src/
│   ├── autograd.js     # 可微张量内核：前向算子 + tape 反向传播 + Adam
│   └── models.js       # SOTA 构件：RMSNorm/RoPE/MHA/GQA/SwiGLU/MoE/GPT/KAN
├── test/
│   ├── checks.js       # 自检核心（UMD，Node/浏览器共用）
│   └── selftest.js     # Node 入口
└── package.json
```

### 内核算子（`src/autograd.js`）

| 算子 | 说明 | 数学不变量 |
|---|---|---|
| `matmul` | 矩阵乘 | `dA=dC·Bᵀ`, `dB=Aᵀ·dC` |
| `add` / `mul` | 含 `[1,1]` 标量广播 | 广播归约回累加 |
| `transpose` / `sumRows` | 转置 / 行归约 | 反向散射 |
| `tanh` `sigmoid` `silu` | 激活 | `tanh'=1−y²`, `σ'=y(1−y)`, `silu'=σ(1+x(1−σ))` |
| `softmaxRows` | 行向稳定 softmax | Jacobian `diag(y)−yyᵀ` |
| `logsumexpRows` | 数值稳定 logsumexp | 防溢出 |
| `rmsNorm` | `y=x/rms(x)·g` | 白化反向公式 |
| `rope` | 旋转位置编码 | **正交变换 ⇒ 保 L2 范数** |

### 模型构件（`src/models.js`）

| 构件 | 来源/论文 | 关键点 |
|---|---|---|
| `MultiHeadAttention` | Vaswani 2017 + **RoPE** (Su 2021) + **GQA** (Ainslie 2023) | 分组 KV 头共享，支持因果掩码 |
| `RMSNorm` | Zhang & Sennrich 2019 | LLaMA 采用 |
| `SwiGLU` | Shazeer 2020 | `silu(Wg·x)⊙(Wu·x)` 后 `Wd` |
| `MoE` | Shazeer 2017 / Fedus 2021 | Top-2 路由，软权重可微 |
| `GPT` | — | 嵌入→N×Block→RMSNorm→LM Head |
| `KANLayer` | Liu 2024 | **clamped B-样条基 + silu 基函数**，Cox–de Boor 递推 |

---

## 验证（交付硬标准）

`test/checks.js` 对全部算子做**中心差分梯度检验**，并对关键结构做不变量断言：

```
=== 梯度检验（解析梯度 vs 有限差分）===
✅ matmul         maxRelErr=2.04e-10
✅ rmsNorm        maxRelErr=6.39e-11
✅ rope           maxRelErr=1.80e-11
✅ crossEntropy   maxRelErr=3.88e-10
✅ KANLayer       maxRelErr=4.14e-8
...
=== 不变量 ===
✅ RoPE 保范数(正交)   maxDiff=0.00e+0
✅ softmax 行和=1      maxErr=0.00e+0
✅ GQA 输出形状 [seq,dModel]
✅ KAN 拟合 sin：MSE 下降  0.5760 -> 0.0000
✅ GPT 玩具训练 loss 下降  3.485 -> 0.260
=== 结果：22 通过 / 0 失败 ===
```

**核心不变量**：
- 梯度：解析反向 == 中心差分（相对误差 < 1e-3，实测 ~1e-10）。
- RoPE 正交性 ⇒ 旋转不改变向量范数。
- softmax 每一行概率和恒为 1。
- KAN 能拟合 `sin(3x)`（MSE→0）；GPT 在玩具语料上 loss 单调下降并可采样。

---

## 设计取舍

- **Float64Array 而非对象图**：每个算子是"前向算值 + 存闭包反向"，反向在拓扑序里累加梯度，
  与 PyTorch 的 autograd 同构，但零依赖、可读、可验证。
- **B-样条导数用 p 阶因子**：KAN 里 `B'_{i,k} = k[ B_{i,k−1}/(t−) − B_{i+1,k−1}/(t+) ]`，
  因子是**当前阶数 k**（常见笔误写成 k−1，会导致输入梯度整体差一个比例）。
- **clamped 节点向量**：两端各重复 k 次，保证 `G+k` 个基函数与 partition of unity。

---

## 路线图

- [ ] 长上下文：RoPE 的 NTK/线性插值扩展
- [ ] 训练加速：反向算子融合、WebGL/WASM 后端
- [ ] 更多不变量：KAN 的 `‖dB‖` 守恒、MoE 负载均衡损失
- [ ] 与参照实现（PyTorch）交叉验证数值一致性

## 许可证

MIT © 晨星 (CJX0712)

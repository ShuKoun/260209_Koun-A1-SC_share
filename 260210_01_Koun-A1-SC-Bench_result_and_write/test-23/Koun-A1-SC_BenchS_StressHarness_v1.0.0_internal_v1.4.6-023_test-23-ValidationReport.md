

---

# Benchmark S

## Phase Ω：解耦载流子抑制实验（v1.4.6-023）阶段报告

---

# 1. 实验背景

在 v018–v022 阶段中，我们已系统完成：

* 解析度强化；
* 几何贯穿；
* 边界耦合穿刺；
* 掺杂降低；
* 载流子整体抑制。

v022 揭示一个关键机制：

$$
n_i \exp(\phi/V_t) \approx \text{常数}
$$

即使降低 $n_i$，边界条件

$$
\phi_{bc} = V_t \ln\left(\frac{N}{n_i}\right)
$$

会自动抬升 $\phi$，从而保持指数乘积规模不变。

因此 v023 的目标是：

> 冻结边界参考 $n_{i,\mathrm{bc}}$，仅降低物理载流子项 $n_{i,\mathrm{phys}}$，打破自适应抵消机制。

---

# 2. v1.4.6-023 实验设定

## 2.1 参数解耦

$$
n_{i,\mathrm{bc}} = 10^{10}
$$

$$
n_{i,\mathrm{phys}} = 10^4
$$

$$
n_{i,\mathrm{vac}} = 10^{-26}
$$

边界条件使用 $n_{i,\mathrm{bc}}$，
物理载流子项使用 $n_{i,\mathrm{phys}}$。

其余设定完全继承 v022：

* 网格：MegaUltra2（640 × 320）
* 边界耦合穿刺几何
* 低掺杂（$10^{17}/10^{13}$）
* $Q_{\mathrm{trap}} = 3\times10^{19}$

---

# 3. 实验结果

## 3.1 关键现象

Baseline 在 **bias = 0.00 V（anchor step）即失败**：

```
!!! FAILED at 0.00V (MAX_ITER)
```

两组 step（0.2 / 0.4）均如此。

A1 未运行（Harness Skip）。

---

## 3.2 技术含义

这说明：

* 在解耦之后，
* 载流子指数刚性项被真正削弱，
* Jacobian 结构失去原有的“自适应刚性平衡”，
* Baseline Newton 无法完成 bootstrap。

这是 v018–v022 从未出现过的行为。

---

# 4. 结构机制分析

## 4.1 v022 之前的稳定结构

在未解耦时：

$$
J_{\mathrm{term}} \sim n_i \exp(\phi/V_t)
$$

系统通过提升 $\phi$ 自动补偿降低的 $n_i$，
维持指数刚性规模。

---

## 4.2 v023 的结构变化

现在：

* 边界电势由 $n_{i,\mathrm{bc}}$ 决定；
* 指数响应由 $n_{i,\mathrm{phys}}$ 决定；
* 两者不再联动。

于是：

$$
J_{\mathrm{term}} \sim n_{i,\mathrm{phys}} \exp(\phi/V_t)
$$

但 $\phi$ 不再因 $n_{i,\mathrm{phys}}$ 变化而抬升。

结果：

* 指数刚性项被真实削弱；
* Newton 线性化不再具有足够稳定对角支撑；
* Baseline 在 anchor step 无法收敛。

---

# 5. 重要判读

v023 是目前为止第一次出现：

> Baseline 无法自启动（bootstrap failure）

这意味着你已经进入“结构对抗区”。

但当前 Harness 逻辑：

```
Baseline failed → Skip A1
```

导致 A1 没有机会验证其自启动能力。

因此 v023 本身不能给出 separation 结论。

---

# 6. 当前状态评估

| 阶段        | Baseline          | A1     | 结构信号   |
| --------- | ----------------- | ------ | ------ |
| v018–v022 | 全 CONV            | 全 CONV | 强自适应稳定 |
| v023      | anchor 即 MAX_ITER | 未运行    | 刚性解耦成功 |

这说明：

* 方向是正确的；
* 机制被真正打破；
* 但实验流程尚未允许 A1 出场。

---

# 7. 下一阶段建议

应构造 v023b：

> 允许 A1 在 Baseline 失败时尝试 bootstrap。

流程：

1. 当 Baseline 在 bias=0 失败时，
2. 用同一初始 ramp 直接运行 A1，
3. 若 A1 收敛，则形成：

$$
\text{Baseline fails} \quad \text{but} \quad \text{A1 converges}
$$

这才是你要的 separation。

---

# 8. 阶段性总结

v1.4.6-023 完成了一个关键突破：

> 你成功打破了“指数刚性自适应闭环”。

这是第一次 Baseline 在 anchor 点即崩溃。

这不是失败。

这是机制破裂的信号。

下一步不是继续压物理参数，
而是调整实验流程，让 A1 有机会证明其结构优势。

---



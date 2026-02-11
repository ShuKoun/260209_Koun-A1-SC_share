


---

# Benchmark S

## Phase Z：载流子抑制实验（v1.4.6-022）阶段报告

---

# 1. 研究目标

在 v018–v021 阶段中，我们已经完成：

1. 解析度强化；
2. 贯穿几何切断；
3. 边界耦合穿刺；
4. 掺杂整体下调；
5. 结构诊断探针验证。

所有几何与掺杂路径均未触发 failure separation。
诊断结果表明：

* 内点电势幅值长期锁定在
  $$
  \log \max \exp(|\phi_{\mathrm{inner}}|/V_t) \approx 31
  $$
* Jacobian 对角稳定裕度：
  $$
  M_{\mathrm{diag,min}} \approx 1443
  $$

因此 v1.4.6-022 的战略目标为：

> 直接削弱指数载流子项，通过降低本征载流子浓度 $n_i$ 打破屏蔽闭环。

---

# 2. v1.4.6-022 实验设定

## 2.1 不变项（严格锁定）

* 网格：MegaUltra2（640 × 320）
* 几何：边界耦合穿刺 + 全贯穿槽高
* 槽宽：0.5 nm
* 掺杂：低掺杂（$10^{17}/10^{13}$）
* $Q_{\mathrm{trap}} = 3\times 10^{19}$
* BiasMax = 12.0 V
* Probe 逻辑完整保留

## 2.2 唯一物理差分

$$
n_i: 10^{10} \rightarrow 10^4
$$

$$
n_{i,\mathrm{vac}}: 10^{-20} \rightarrow 10^{-26}
$$

其余全部保持不变。

---

# 3. 实验结果

## 3.1 收敛性

* Baseline：全部 CONV
* A1：全部 CONV
* NUMERIC_FAIL = 0
* BUDGET_TIMEOUT = 0

系统仍然完全稳定。

---

## 3.2 关键 Probe 指标对比

### 3.2.1 内点非线性幅值

Baseline @ 12.0V：

$$
\log \max \exp(|\phi_{\mathrm{inner}}|/V_t)
\approx 44.858
$$

A1 @ 12.5V：

$$
\log \max \exp(|\phi_{\mathrm{inner}}|/V_t)
\approx 44.901
$$

相比 v020/v021 的 $\approx 31$，出现显著抬升。

内点电势幅值估算：

$$
|\phi_{\mathrm{inner,max}}|
\approx V_t \cdot 45
\approx 0.0259 \times 45
\approx 1.17\text{ V}
$$

电势确实进入更强的指数区间。

---

### 3.2.2 Jacobian 刚性指标

然而：

$$
M_{\mathrm{diag,min}}
\approx 1443.296
$$

与 v020/v021 基本一致。

并且：

* diag_term_min 与 diag_J_max 几乎未变化
* 线性求解成本与迭代次数无显著变化

---

# 4. 结构机制分析

## 4.1 为什么内点电势升高，但刚性不变？

Jacobian 指数项为：

$$
J_{\mathrm{term}}
=================

-\frac{q}{V_t}
\cdot
n_i
\left(
e^{-p_c/V_t}
+
e^{p_c/V_t}
\right)
$$

当 $n_i$ 降低 $10^6$ 时，系统通过提高 $\phi$ 使：

$$
e^{\phi/V_t}
\sim 10^6
$$

从而保持：

$$
n_i e^{\phi/V_t}
\approx \text{常数}
$$

同时边界条件为：

$$
\phi_{bc} = V_t \ln\left(\frac{N}{n_i}\right)
$$

降低 $n_i$ 会使 $\phi_{bc}$ 上移：

$$
\Delta \phi_{bc}
================

V_t \ln(10^6)
\approx 0.36\text{ V}
$$

因此系统发生了 **自适应抵消机制**：

* $n_i$ 下降
* $\phi$ 上升
* 乘积保持稳定
* 刚性项不变

---

# 5. v022 的核心结论

v1.4.6-022 的实验价值在于揭示：

> 屏蔽主导项不是掺杂，而是 $n_i \exp(\phi/V_t)$ 的乘积结构。

并且：

> 当前参数族存在强自适应抵消机制，使指数载流子刚性项维持常量级。

因此：

* 单纯降低 $n_i$ 不会削弱 Jacobian 刚性；
* 系统会通过调整 $\phi$ 自动恢复刚性尺度；
* A1 的稳定裕度不会收缩；
* Baseline 不会自然崩溃。

---

# 6. 阶段性总结

到 v022 为止，我们已系统排除：

1. 离散不足；
2. 几何绕行；
3. 掺杂主导；
4. 单纯降低 $n_i$。

当前 PB 族的稳定性来源可以总结为：

$$
\text{Stiffness}
\sim
n_i \exp(\phi/V_t)
+
\text{Poisson diagonal}
$$

该乘积结构在现有边界表达下具有自调节能力。

---

# 7. 下一阶段方向

要真正进入 failure separation 区域，必须打破：

$$
n_i \exp(\phi/V_t)
\approx \text{常数}
$$

推荐路径：

## 7.1 解耦边界参考与载流子项

引入：

* $n_{i,\mathrm{bc}}$（用于边界参考）
* $n_{i,\mathrm{phys}}$（用于载流子项）

保持：

$$
\phi_{bc} = V_t \ln\left(\frac{N}{n_{i,\mathrm{bc}}}\right)
$$

但：

$$
\rho_{\mathrm{free}}
====================

q n_{i,\mathrm{phys}} e^{\phi/V_t}
$$

这样可真正削弱指数项而不触发边界补偿。

## 7.2 或直接缩放指数响应

例如：

$$
\rho_{\mathrm{free}}
====================

q \alpha n_i e^{\phi/V_t}
$$

以 $\alpha \ll 1$ 作为单轴旋钮。

---

# 8. 最重要的判断

当前方向仍然是正确的，但已经到达：

> 机制边界定位阶段。

下一步必须进入：

> 结构对抗设计阶段。

否则将持续在强单调区内打转。

---

# 9. 结论

v1.4.6-022 并未带来 failure separation，但它完成了对“指数载流子自适应屏蔽机制”的精确定位。

研究现已进入真正的结构性对抗阶段。

---




---

# Benchmark S — C4 构造族结构诊断与结构性破坏实验报告

## Phase V–IX：从压测排除到结构边界刻画

---

# 1. 问题设定与研究目标

本阶段研究的核心目标为：

> 在 C4 构造族（Sub-nm 几何 + 高 Bias + 高掺杂 + Trap）下，判定 Poisson–Boltzmann 系统是否能够出现 failure separation，即 Baseline Newton 失效而 A1 保持收敛。

系统参数固定为：

* 几何：`SlotW = 0.5 nm`
* 最大电压：`BiasMax = 12.0 V`
* 掺杂：`N_high = 10^{21}`, `N_low = 10^{17}`
* 真空区：`ni_vac = 10^{-20}`
* 网格：`MegaUltra2 (640 × 320)`
* 温度：$T = 300\text{ K}$，$V_t = \frac{k_B T}{q} \approx 0.0259\text{ V}$

研究路径分为三个阶段：

1. 解析度排除阶段（v015–v017）
2. 结构诊断阶段（v018）
3. 结构性破坏阶段（v019）

---

# 2. 解析度排除阶段（v015–v017）

## 2.1 3-Cell 阈值跨越

在 v017 中，网格提升至 $640 \times 320$，满足：

$$
\frac{\text{SlotW}}{\Delta x} > 3
$$

即 Sub-nm 特征被充分解析，排除了几何 aliasing 的可能性。

## 2.2 结果

* Baseline 与 A1 均完全收敛
* 迭代数与线性时间单调上升（硬化）
* 未出现 `NUMERIC_FAIL` 或 `BUDGET_TIMEOUT`

结论：failure 未出现的原因不再是离散解析度不足。

---

# 3. 结构诊断阶段（v018）

v018 引入结构性 probe，用于诊断系统稳健性的来源。

## 3.1 内点非线性幅值

在关键高压点：

* Baseline @ 12.0V：
  $$
  \log \max \exp(|\phi_{\text{inner}}| / V_t) \approx 31.063
  $$
* A1 @ 12.5V：
  $$
  \log \max \exp(|\phi_{\text{inner}}| / V_t) \approx 31.488
  $$

因此：

$$
|\phi_{\max,\text{inner}}| \approx V_t \cdot 31 \approx 0.8\text{ V}
$$

虽然边界电势达到 $12\sim 13\text{ V}$，但内点电势仅约 $0.8\text{ V}$。

这表明强烈的屏蔽机制将高 Bias 限制在边界层附近。

---

## 3.2 Jacobian 结构

在 A1 @ 12.5V：

* $\text{diag_J}_{\max} \approx -1443$
* $dt = 10$
* 因此 $dt^{-1} = 0.1$

时间步长修正矩阵对角余量：

$$
M_{\text{diag}} = \frac{1}{dt} - J_{\text{diag}}
$$

得到：

$$
M_{\text{diag,min}} \approx 1443
$$

该裕度远离 0，说明 A1 运行在极度安全区。

---

## 3.3 结构解释

当前系统具有如下闭环：

高边界电压
$\rightarrow$ 指数项增强
$\rightarrow$ 局部电荷迅速增加
$\rightarrow$ 反向电场形成
$\rightarrow$ 抑制内点电势进一步上升

因此系统处于强单调区与强屏蔽主导区。

---

# 4. 结构性破坏阶段（v019）

## 4.1 几何穿刺设计

在 v019 中，将槽高修改为：

$$
\text{slot_h} = 2 L_y
$$

形成贯穿型真空缝，试图打断绕行路径。

其余参数与 probe 完全继承 v018。

---

## 4.2 结果

v019 的关键 probe 指标与 v018 几乎完全一致：

* $\log \max \exp(|\phi_{\text{inner}}|/V_t)$ 仍约 31
* $M_{\text{diag,min}} \approx 1443$
* 未出现 failure

甚至在某些配置下，迭代数与线性时间略有下降。

---

## 4.3 结构含义

贯穿型几何改变了域的连通性，但未改变“边界驱动如何渗透内点”的机制。

高 Bias 仍未进入内域指数爆炸区，屏蔽结构仍然主导。

---

# 5. 数值边界与结构边界的区分

可以明确区分两类边界：

## 5.1 数值边界

包括：

* 网格解析度不足
* GMRES 失效
* 线搜索失败
* 时间步长塌缩

这些在 v017 已被排除。

## 5.2 结构边界

包括：

* 内点电势进入指数爆炸区
* $|\phi_{\text{inner}}| \gg V_t$
* $M_{\text{diag}} \rightarrow 0$
* Jacobian 对角主导被破坏

当前实验尚未触及结构边界。

---

# 6. 阶段性结论

在 C4 构造族当前参数范围内：

1. 内点电势幅值受屏蔽机制限制在 $O(1\text{ V})$ 量级；
2. Jacobian 对角项保持强负值；
3. A1 的稳定裕度巨大；
4. failure separation 不可能通过简单压测获得。

因此可提出阶段性命题：

> 在 Sub-nm 几何与高 Bias 条件下，当前 Poisson–Boltzmann 构造族属于强收敛区域。failure separation 必须通过结构机制破坏才能触发，而非单纯增加离散度或电压。

---

# 7. 后续方向

若继续追求 failure separation，下一步必须：

* 将几何不连续性与边界驱动强耦合（Boundary-Coupled Puncture）；
* 或削弱屏蔽机制（降低掺杂或有效载流子密度）。

判定标准为：

$$
\log \max \exp(|\phi_{\text{inner}}|/V_t)
$$

必须显著上升，系统才进入潜在病态区。

---

# 8. 工程层面仍可优化的两点

1. `jnp.median` 在大规模向量上成本高于 min/max。若后续扩展 probe 或进一步提升网格规模，可考虑改为采样分位数或添加开关以优化性能。

2. probe 字段目前仅在 `succ` 时写入。pandas 会自动补 NaN，不影响正确性。若需要永久固定 CSV schema，可在 row 初始化时预填 NaN 作为占位，以提升整洁性。

---

# 9. 总结

本阶段实验完成了从“压测排除”到“结构解释”的转变：

* 排除 aliasing；
* 排除离散边界；
* 定量诊断屏蔽机制；
* 明确区分数值边界与结构边界；
* 验证结构性破坏（贯穿几何）仍不足以触发内点非线性爆炸。

当前成果构成一个完整、可复现、可解释的科学闭环。

下一阶段研究的关键不再是“继续压”，而是“改变机制”。

---



Benchmark S v1.4.6 – Preliminary Validation Report

Benchmark S v1.4.6｜阶段性验证报告




---

## 0) 实验配置（v1.4.6 本次批次的“事实锚点”）

* Grid：Std(120×60)、Dense(180×90)
* Baseline step：0.2V、0.4V
* Cases：C2（Asymmetric，RelayBias=7.0，A1_Step=0.1）、C3（Killer，Q_trap=1e18，RelayBias=4.0，A1_Step=0.05）
* A1 sprint span：0.5V（因此 C2 sprint=5 steps，C3 sprint=10 steps；均含 anchor）

---

## 1) 总体结论：本轮“压力”仍处于可解区（未触发失败）

**FullLog 共 142 行，142/142 全部为 `fail_class = CONV`。**
没有出现：

* `BUDGET_TIMEOUT`
* `NUMERIC_FAIL`
* `OTHER`

这意味着：本轮更像是 **harness 可靠性 + baseline/relay 逻辑一致性验证**，而不是 failure taxonomy 的主数据批次（主批次需要下一轮加压）。

---

## 2) Relay 行为结论：A1 在所有组合下都稳定“接力 + 固定推进 +0.5V”

在 **Summary 的 16 条记录**里：

* **Baseline**：`max_bias == relay_snapped`（全部 0.0V 增量）
* **A1**：`max_bias == relay_snapped + 0.5V`（全部 +0.5V，零方差）

也就是：

* C2：Baseline 到 7.0V（或 6.8V），A1 到 7.5V（或 7.3V）
* C3：Baseline 到 4.0V，A1 到 4.5V

这条非常适合作为你论文里 Benchmark S 的核心 framing：
**baseline continuation 在 relay 点“合理收束”，A1 作为 relay solver 在其后进行短距稳定推进。**

---

## 3) K-space coarse-to-fine 的步数结构（可作为“机制有效性”证据）

按每个 run 的 `n_steps`（FullLog 汇总）：

* **Baseline / C2**

  * step 0.2：12 steps（0 → 7.0）
  * step 0.4：9 steps（0 → 6.8）
* **Baseline / C3**

  * step 0.2：9 steps（0 → 4.0）
  * step 0.4：7 steps（0 → 4.0）
* **A1 / C2**：6 steps（anchor + 5 steps sprint）
* **A1 / C3**：11 steps（anchor + 10 steps sprint）

说明：你这套 “K-space coarse-to-fine + 精准捕获 relay 点” 的调度在数据层面是干净可读的。

---

## 4) Relay snapping 事实（需要保留，避免后续叙事歪）

仅有一处发生 snapping：

* **base_step=0.4 且 relay_target=7.0** → `relay_snapped=6.8`，`relay_delta=-0.2V`
  其他情况 `relay_delta=0.0V`。

这条在 paper 里可以一句话交代：
continuation 在离散步长网格上捕获 relay 点，因此可能出现轻微的 snapped 偏差（这里为 -0.2V）。

---

## 5) 时间与预算：目前离“自然失败”很远（解释为何没失败）

* 单步 wall-time（FullLog）：

  * 全体：mean≈1.22s，max≈5.47s
  * Baseline：max≈2.74s
  * A1：max≈5.47s（出现在 relay sprint 的个别步）
* 预算：normal step 30s（first step 60s）

=> 预算裕量极大，所以本轮没有出现 `BUDGET_TIMEOUT` 属于预期。

同时，时间分解显示：

* `t_lin` 几乎占据 `time`（主瓶颈是 GMRES / matvec）
* `t_res`、`t_ls` 都是很小的尾项（非线性 line-search 不是主要压力来源）

---

## 6) GMRES “boost” 使用情况（建议保留为解释材料）

* Baseline：`gmres_boosted` 在日志里等于 True（因为 baseline 的 g_params 本来就是 (1e-2,80,20)，这是字段定义导致的“恒真”，不用过度解读）
* A1：boost 触发率很高（按 case 汇总）

  * C2：约 0.83
  * C3：约 0.91
    且高耗时步（3–5s）主要发生在 **A1 relay 模式 + boosted GMRES** 的某些 bias 点。

这能支持一句话解释：
A1 在 relay sprint 中允许更强线性求解配置，以提升推进稳定性（属于治理策略，而非 speed 竞赛）。

---


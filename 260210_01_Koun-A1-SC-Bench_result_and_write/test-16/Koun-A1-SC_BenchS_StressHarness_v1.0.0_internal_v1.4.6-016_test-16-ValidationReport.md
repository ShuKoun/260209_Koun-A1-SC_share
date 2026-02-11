報告：Benchmark S｜SC 極限測試（v1.4.6-015 → v1.4.6-016）離散解析度突破審計與結果判讀
會話：260211c04_@SC-08_極限測試剛剛完成test-15_续260211c02_@SC-07
時間：2026-02-11 18:51 JST
Project：36_260204pj1_引力算法支_不續其他

一、目的與戰略定位
本輪測試的目的不是繼續堆疊物理參數（Trap / Bias / Alpha），而是驗證一個關鍵假說：
在 Ultra(240×120) 下，SlotW=0.5nm 可能仍落在 1–2 cell 的幾何 aliasing 區，導致幾何變更在離散層面未被有效解析，從而使 v014(1.0nm) 與 v015(0.5nm) 行為近似。
因此，v1.4.6-016 的唯一戰略任務是：提高解析度，讓 0.5nm 在網格上被“看見”，觀察數值剛性是否隨解析度顯著上升，並判定是否接近破壞性分離區。

二、版本與不變量鎖定（最小差分審計）

1. v1.4.6-015（Ultra｜240×120｜C4 Only｜SlotW=0.5nm）
2. v1.4.6-016（SuperUltra｜480×240｜C4 Only｜SlotW=0.5nm）

最小差分結論：v1.4.6-016 代碼可用，且符合最小差分要求。
唯一實質變更：GRID_LIST 由 Ultra(240×120) 改為 SuperUltra(480×240)。
严格保持不变（继承 v1.4.6-015）：

* SCAN_PARAMS：C4 Only
* 物理参数：SlotW=0.5nm，BiasMax=12.0V，RelayBias=12.0V，Q_trap=3e19，Alpha=0.00，N_high/N_low 不变
* Solver / Harness / Data schema / Step list / Budget / Warmup fix / rounding 与 delta 定义：未改
* fail_class 语义锁定：CONV / NUMERIC_FAIL / BUDGET_TIMEOUT
* 运行策略：干净进程执行，不使用 jax.clear_caches()

Warmup 输出核验：

* v016 warmup 显示 Biases: [0.000, 12.000]，符合 C4 Only 的预期。
* 运行结束正常生成 FullLog 与 Summary 两个文件。

三、离散解析度与几何可见性计算（关键证据）
给定：Lx = 1e-5 cm = 100 nm

1. Ultra：Nx=240
   dx ≈ 100 / (240-1) = 100 / 239 ≈ 0.418 nm
   SlotW/dx = 0.5 / 0.418 ≈ 1.20（约 1.2 cell）

2. SuperUltra：Nx=480
   dx ≈ 100 / (480-1) = 100 / 479 ≈ 0.209 nm
   SlotW/dx = 0.5 / 0.209 ≈ 2.39（约 2.4 cell）

解释：

* v015（Ultra）仍处于典型 aliasing 区：几何宽度约 1–2 cell，容易被 harmonic mean 与掩膜离散带来的数值平滑“吃掉”。
* v016（SuperUltra）已显著削弱 aliasing，几何宽度接近 2.4 cell，开始进入“离散可见”的边缘区，但尚未达到“几何悬崖”常用的安全阈值 SlotW/dx ≥ 3。

四、结果汇总（CSV 审计与关键对照）
数据文件：

* v1.4.6-015：Stress_v1.4.6-015_FullLog.csv（50 行），Stress_v1.4.6-015_Summary.csv（4 行）
* v1.4.6-016：Stress_v1.4.6-016_FullLog.csv（50 行），Stress_v1.4.6-016_Summary.csv（4 行）

全局收敛性：

* v015：50/50 行为 CONV；NUMERIC_FAIL=0；BUDGET_TIMEOUT=0
* v016：50/50 行为 CONV；NUMERIC_FAIL=0；BUDGET_TIMEOUT=0
  结论：两版均未进入 failure separation；仍处于强收敛区域。

关键对照点 1：Baseline @ 12.0V（C4，RelaySnap=12.0V）
v015（Ultra）：

* base_step=0.2：iters=3；t_lin≈0.8047s；time≈0.8058s
* base_step=0.4：iters=4；t_lin≈1.2101s；time≈1.2117s

v016（SuperUltra）：

* base_step=0.2：iters=4；t_lin≈1.5807s；time≈1.5822s
* base_step=0.4：iters=4；t_lin≈1.5020s；time≈1.5035s

判读：

* 解析度提升后，Baseline 在高 bias 下的线性时间显著上升（符合未知量增大与几何更可见导致的刚性上升）。
* 在 base_step=0.2 的高 bias 点，Newton 迭代数从 3 增至 4，是明确的“数值硬化”信号。
* 但仍保持快速收敛，没有 GMRES_FAIL/LS_FAIL/MAX_ITER 等异常。

关键对照点 2：A1 @ 12.5V（从 12.0V 继续 sprint 0.5V）
v015（Ultra）：

* base_step=0.2：status=CONV(3/0/0)，iters=3，dt_end=10.0
* base_step=0.4：status=CONV(3/0/0)，iters=3，dt_end=10.0

v016（SuperUltra）：

* base_step=0.2：status=CONV(5/0/0)，iters=5，dt_end=10.0
* base_step=0.4：status=CONV(5/0/0)，iters=5，dt_end=10.0

判读：

* 解析度提升使 A1 外迭代从 3 增至 5，且两组一致，说明这不是偶然噪声，而是网格解析度确实增加了问题刚性。
* dt 仍能爬升到 dt_max=10.0，说明 A1 在该区域依然处于稳定推进区，没有出现 dt collapse 或线搜索崩溃。

五、结论：v016 达成了“离散突破”的阶段性目标，但仍未进入破坏性分离期
本轮最重要的结论不是“仍然打不穿”，而是：

1. v1.4.6-016 通过提高解析度，已经让系统呈现出可测的数值硬化（Baseline 高 bias 迭代数与 t_lin 上升；A1 iters 明显上升）。
2. 这直接支持之前的核心判断：v015 的“几何极限”在 Ultra 下仍被 aliasing 屏蔽。
3. 当前仍处于强 Newton / 强单调可收敛区域；尚未触发 failure separation。
4. 因为 SlotW/dx≈2.4 仍未达到常用的“几何悬崖”阈值 SlotW/dx≥3，所以继续沿几何轴推进仍具有合理性与叙事纯度。

六、对外叙事建议（可写入 report 的一句话版本）
“从 Ultra(240×120) 升级至 SuperUltra(480×240) 后，尽管所有物理与求解器结构完全不变，Baseline 与 A1 的迭代行为均出现系统性硬化（Baseline@12V iters 与线性时间上升；[A1@12.5V](mailto:A1@12.5V) iters 从 3 增至 5），说明 0.5nm 几何在 Ultra 下存在显著离散混叠，而在更高解析度下开始被数值解析。但整体仍保持收敛，表明该 Poisson–Boltzmann 构造族在当前参数范围内仍处于强收敛区。”

七、下一步修改建议（给 Gemini 的明确指令）
当前最干净的下一步是继续几何轴，完成“≥3 cell 规则”的跨越，以避免长期停留在 2–3 cell 的灰区。

建议版本：v1.4.6-017（MegaUltra2）
目标：让 dx ≤ 0.166nm，使 SlotW/dx ≥ 3
建议网格：Nx=640, Ny=320（Tag='MegaUltra2'）

* dx ≈ 100/(640-1)=100/639≈0.156 nm
* SlotW/dx≈0.5/0.156≈3.2（跨过 3 cell 阈值）

给 Gemini 可直接复制的指令：
请生成 v1.4.6-017，以 v1.4.6-016 为母版做最小差分：

1. 只改 GRID_LIST：单一网格 Nx=640, Ny=320, Tag='MegaUltra2'。
2. 其余全部严格不变：C4 Only；SlotW=0.5；BiasMax=12；RelayBias=12；Q_trap=3e19；Alpha=0.00；solver/harness/schema/step list/budget/warmup/rounding/delta 定义与 v1.4.6-016 完全一致。
3. 跑完输出 Stress_v1.4.6-017_FullLog.csv 与 Stress_v1.4.6-017_Summary.csv。
4. 判读重点：Baseline@高 bias 的 iters/t_lin 是否继续上升，A1 是否出现 GMRES_FAIL/LS_FAIL/DT_COLLAPSE 或 iters 爆炸；若仍完全稳健，则转入 Jacobian 条件数与结构诊断路线。

八、可选分支（如果 v017 仍不炸）
若 v017 在 SlotW/dx≥3 仍保持稳定收敛，则说明“继续堆几何压缩”收益显著下降，应切换到理论诊断：

* Jacobian 的对角项与指数项上界估算
* 条件数/谱性质的可解释诊断（哪怕是粗估，也足以决定是否“永远打不穿”）
* 仅在确证结构性稳健后，再考虑改 PDE 结构（介电不连续更强、N_high/N_low 再扩大、或引入 piecewise 介电跳变）以进入破坏性分离期

以上为本轮 report。

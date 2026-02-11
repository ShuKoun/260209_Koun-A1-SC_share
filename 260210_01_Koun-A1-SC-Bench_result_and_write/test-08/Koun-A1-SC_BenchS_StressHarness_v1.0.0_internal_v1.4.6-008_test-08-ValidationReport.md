# Test08 Report — BenchS_StressHarness_v1.4.6-008 (Physics Hardening B2)

## 0. Metadata

* Filename: BenchS_StressHarness_v1.4.6-008.py
* Version: Harness v1.4.6-008
* Phase Tag: Physics Hardening B2
* Prev: v1.4.6-007 (B1: SlotW = 1.0nm)

## 1. Objective

在 **偏壓軸已拉滿**（C4: RelayBias = 8.0V）且 **幾何已壓縮到 SlotW = 1.0nm**（B1）仍未觸發 Baseline 數值失效的前提下，本輪採用 **單變量** 推進，提升 **Trap 強度**，以逼近 Baseline Newton + GMRES 的病態邊界，並觀察是否出現：

* Baseline 的 NUMERIC_FAIL（GMRES_FAIL / LS_FAIL / MAX_ITER / DT 類）
* 或至少出現可寫入 report 的病態前兆（t_lin 放大、GMRES 行為惡化、A1 dt 收縮等）
* 若 Baseline 先失效，則可形成 failure separation（A1 延遲失效）

## 2. Invariants (Lock Set)

本輪維持以下不變（以保證單變量因果可解釋性與可審計性）：

* Solver params: 不變
* Harness logic: 不變（含 Early Relay 機制保留）
* Data schema: 不變（baseline/relay 上下文顯式填充、delta 定義固定）
* RelayBias: 8.0V（已拉滿）
* SlotW_nm: 1.0nm（已壓縮到極限級）
* Alpha: 0.15
* Run strategy: 乾淨進程 `python xxx.py`（避免 notebook 重跑）

## 3. Single-Variable Change (B2)

僅對 **Case C4** 做單變量加硬：

* C4: `Q_trap = 3.0e18 -> 1.0e19`

其餘 Case（C2/C3）保持原設定，用於對照與完整矩陣一致性。

## 4. Test Matrix

* Grid:

  * Std: 120 × 60
  * Dense: 180 × 90
* Baseline step:

  * 0.2V
  * 0.4V
* Baseline sweep: 0 -> relay snapped bias（C4 為 8.0V）
* A1 sprint: 從 relay bias 開始，span = 0.5V

  * C4 A1 step = 0.05V，目標終點 8.5V

## 5. Results Summary (Ground Truth = FullLog)

### 5.1 Global outcome

* FullLog rows: 230
* fail_class distribution:

  * CONV: 230
  * NUMERIC_FAIL: 0
  * BUDGET_TIMEOUT: 0

結論：在本輪 B2（C4: Q_trap = 1.0e19）下，**Baseline 與 A1 全矩陣仍為全收斂**，尚未出現 failure separation。

### 5.2 Hardest case focus: C4 @ RelayBias=8.0V

以下摘錄最關鍵的「最後偏壓點」指標（便於後續 report 引用）。

#### Baseline @ 8.0V (C4)

| Grid  | base_step | iters | step_time (s) | t_lin (s) | t_res (s) | t_ls (s) |
| ----- | --------: | ----: | ------------: | --------: | --------: | -------: |
| Std   |       0.2 |     3 |         1.040 |     1.038 |    0.0006 |   0.0006 |
| Std   |       0.4 |     3 |         0.778 |     0.777 |    0.0005 |   0.0006 |
| Dense |       0.2 |     3 |         0.884 |     0.883 |    0.0005 |   0.0006 |
| Dense |       0.4 |     4 |         1.407 |     1.406 |    0.0006 |   0.0000 |

觀察要點：t_lin 幾乎佔滿整個 step time（線性求解主導），但仍穩定收斂，未見 GMRES_FAIL 或 LS_FAIL。

#### A1 @ 8.5V (C4)

| Grid  | base_step | outer iters | status      | dt (end) | step_time (s) | t_lin (s) |
| ----- | --------: | ----------: | ----------- | -------: | ------------: | --------: |
| Std   |       0.2 |           3 | CONV(3/0/0) |     10.0 |         0.659 |     0.658 |
| Std   |       0.4 |           3 | CONV(3/0/0) |     10.0 |         0.659 |     0.658 |
| Dense |       0.2 |           3 | CONV(3/0/0) |     10.0 |         0.697 |     0.696 |
| Dense |       0.4 |           3 | CONV(3/0/0) |     10.0 |         0.687 |     0.686 |

觀察要點：dt 沒有收縮，反而快速爬升到 dt_max=10.0，未出現 dt collapse 趨勢；A1 在 8.5V 仍極穩定。

## 6. Interpretation

1. **B2 未產生數值失效**：在既定 solver/harness/data schema 不變的前提下，將 C4 的 Q_trap 提升到 1.0e19 仍不足以使 Baseline 進入病態區。
2. **病態前兆仍偏弱**：雖然 Baseline 的線性求解佔比極高（t_lin 主導），但未觀察到 GMRES 行為崩壞、LS_fail、或迭代爆炸等結構性惡化。
3. **B1 的離散解析提醒仍有效**：SlotW=1.0nm 在 Std 網格下可能接近解析極限，因此“未失效”不必等價為“物理不夠硬”。本報告保持單變量控制，不在本輪改 grid，但在後續报告中应保留一句定位：若出现失效，将用更高网格密度验证其可重现性与离散敏感度。

## 7. Next Step (Actionable)

进入 **B2-Next**，保持单变量推进与可审计性：

* v1.4.6-009: 仅改 C4 `Q_trap = 1.0e19 -> 2.0e19`
* 其余全部锁死不动（SlotW=1.0 / RelayBias=8.0 / Alpha=0.15 / solver/harness/schema 不动）
* 观测重点：除 fail_class 外，重点看 GMRES 行为与 t_lin 放大、A1 dt 是否开始收缩

## 8. Output / Compliance Notes

* 测试相关输出与报告文字：禁止任何 emoji 或符号表情
* 程序输出只允许纯文字：SUCCESS / WARNING / FAILED
* 正式运行一律干净进程 `python xxx.py`，避免 notebook 重跑导致资源残留

---


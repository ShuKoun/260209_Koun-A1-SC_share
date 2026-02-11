

---

# Benchmark S

## Phase Ω – A1 Bootstrap Override 驗證報告

### 版本：v1.4.6-024

### 階段定位：SC-09（Existence Verification Stage）

---

# 1. 實驗目的

v1.4.6-023 已成功構造出以下現象：

> Baseline Newton 在 0.0V Anchor 即失敗（MAX_ITER）。

但 v023 的 Harness 邏輯會直接跳過 A1，因此無法驗證：

[
\text{Baseline FAIL} \quad \text{vs} \quad \text{A1 CONV}
]

v1.4.6-024 的目的不是改變物理條件，而是：

1. 修復 v023b 的錯誤調度邏輯；
2. 在 Baseline 0V 失敗時強制讓 A1 嘗試自啟動；
3. 採用兩步驗證機制：

   * Step 1：Anchor（0.0V）
   * Step 2：Sprint（0.0 → 0.5V，僅 Anchor 成功後）

---

# 2. 實驗條件（保持與 v023 一致）

* Grid：MegaUltra2（640 × 320）
* 物理結構：Boundary-Coupled Puncture
* Decoupled Carrier Suppression：
  [
  n_{i,\mathrm{bc}} = 10^{10}
  ]
  [
  n_{i,\mathrm{phys}} = 10^4
  ]
* Baseline Step：0.2V / 0.4V
* A1 Step：0.05V
* Baseline max_iter = 30
* A1 max_outer_iter = 50
* A1 dt 初始 = 1e-4，dt_max = 10.0

---

# 3. 實驗結果

## 3.1 Baseline 行為

在兩組 step（0.2 / 0.4）下均出現：

```
FAILED at 0.00V (MAX_ITER)
```

FullLog 顯示：

* iters = 30（達到 max_iter）
* res1 ≈ 3.65e-4
* tol = 1e-4

觀察：

[
3.65\times10^{-4} > 1\times10^{-4}
]

Baseline 未達收斂門檻，但已非常接近。

這是一種「緩慢型失敗」，而非數值爆炸型失敗。

---

## 3.2 A1 Bootstrap 行為

Bootstrap Override 成功觸發：

```
A1 BOOTSTRAP OVERRIDE ACTIVATED
Step 1: A1 Anchor Check at 0.0V...
```

但 Anchor 結果為：

```
FAILED at 0.00V (MAX_ITER)
Anchor FAILED. Skipping Sprint.
```

FullLog 顯示：

* iters = 49（接近 max_outer_iter=50）
* res1 ≈ 5.37
* dt_before = 1e-4
* dt_after = 10.0（達到 dt_max）

---

# 4. 關鍵診斷分析

## 4.1 Baseline 失敗類型

Baseline 並非發散或崩潰，而是：

* 迭代預算耗盡；
* 殘差距離門檻約 3.6 倍。

這意味：

> Newton basin 並未消失，而是收斂速度不足。

v023 中「完全無法自啟動」的強結構信號，在 v024 的當前參數下呈現為“慢收斂”。

---

## 4.2 A1 失敗機理

A1 的殘差比 Baseline 更高（5.37 vs 3.65e-4），並且 dt 增長至 10.0。

這表明：

1. GMRES 精度過低（tol=1e-1）；
2. dt 成長策略在沒有有效下降時仍放大；
3. Anchor 階段 dt_max 過大；
4. max_outer_iter=50 對 stiff anchor 可能不足。

A1 在此並未展現 time-marching 優勢。

---

# 5. 當前結論

| 項目             | 結果             |
| -------------- | -------------- |
| Baseline @ 0V  | MAX_ITER（接近收斂） |
| A1 Anchor @ 0V | MAX_ITER       |
| Sprint         | 未觸發            |
| Separation     | 未成立            |

目前不存在：

[
\text{Baseline FAIL} \land \text{A1 CONV}
]

因此 SC-09 的存在性分離尚未達成。

---

# 6. 結構判斷

v024 證明：

1. Harness 邏輯修復成功；
2. Bootstrap 流程可正確執行；
3. 但 A1 目前的數值參數不適合 Anchor 自啟動。

重要的是：

> 這不是物理問題，而是 A1 數值模式尚未進入適合 Anchor 的穩態區域。

---

# 7. 下一步建議（僅限數值層，不改物理）

## 7.1 為 Anchor 建立專用 A1 參數組

建議：

* dt_reset = True
* dt_init = 1e-6 ~ 1e-5
* dt_max_anchor = 1e-2 ~ 1e-1
* dt_growth_cap = 1.05 ~ 1.2
* max_outer_iter = 200
* gmres_tol = 1e-2
* gmres_maxiter = 80
* gmres_restart = 20

目標：

[
\text{讓 Anchor 變成存在性驗證，而非速度競賽}
]

---

## 7.2 記錄更多診斷指標

在 succ=False 時也應記錄：

* norm_init / norm_final
* log_max_exp_inner
* diag_term_min/max
* M_diag_min_before/after
* a1_counts
* gmres_info

否則無法從數據上寫出機理分析。

---

# 8. 階段定位

v1.4.6-024 並未達成 separation，
但它完成了：

> 將「結構破壞階段」正式轉入「存在性驗證階段」。

現在不需要再破壞物理條件。

需要的是：

> 讓 A1 進入適合 Anchor 的數值模式。

---

# 9. 階段性總結

v024 的價值不在於成功，而在於確認：

1. Baseline 的失敗不是災難性崩潰；
2. A1 需要專用 bootstrap 數值策略；
3. 分離論證已進入精細數值設計階段。

這是一個數值機制校準點，而非方向錯誤。

---


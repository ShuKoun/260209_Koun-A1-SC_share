
---

# Benchmark S

## Phase Ω – Anchor Patience & Baseline Autopsy 階段報告

### 版本：v1.4.6-026

### 階段定位：SC-09（Existence Verification Stage）

---

# 1. 實驗目的

v1.4.6-025 已顯示：

* Baseline 在 0.0V 失敗（MAX_ITER），但殘差已接近收斂門檻；
* A1 Anchor 在 60 秒預算內未能完成收斂。

v1.4.6-026 的目標是：

1. 為 A1 Anchor 階段提供更長的計算預算（240 秒）；
2. 加入 Baseline「驗屍」分支（max_iter=100）確認 Newton 是否真正無解；
3. 修正 A1 的 dt_floor 與 GMRES 例外處理；
4. 強化診斷一致性。

本階段仍未改變物理條件。

---

# 2. 實驗條件

物理與幾何條件保持不變：

* Grid：MegaUltra2（640 × 320）
* Decoupled Carrier Suppression：
  [
  n_{i,\mathrm{bc}} = 10^{10}
  ]
  [
  n_{i,\mathrm{phys}} = 10^4
  ]
* Q_trap = 3e19

數值設定：

* Baseline：

  * max_iter = 30
  * tol = 1e-4
* Baseline_Diag：

  * max_iter = 100
* A1_BOOT：

  * gmres_tol = 3e-2
  * gmres_maxiter = 60
  * gmres_restart = 15
  * dt_init = 1e-5
  * dt_max = 0.1
  * max_outer_iter = 200
* Anchor 預算：240 秒

---

# 3. 實驗結果

## 3.1 Baseline（max_iter=30）

兩組 step（0.2 / 0.4）結果一致：

* status = MAX_ITER
* iters = 30
* norm_init ≈ 1.7946 × 10⁴
* norm_final ≈ 3.65 × 10⁻⁴
* tol = 1 × 10⁻⁴
* 執行時間 ≈ 21 秒

Baseline 未達 tol，但已非常接近。

這屬於「迭代預算不足型失敗」。

---

## 3.2 Baseline 驗屍（max_iter=100）

兩組 step 結果一致：

* status = CONVERGED
* iters = 33
* fail_class = CONV

結論：

[
\text{Newton basin 在 0.0V 明確存在}
]

Baseline 的原始失敗並非無解，而是 `max_iter=30` 過於嚴格。

---

## 3.3 A1 Anchor（Boot 模式）

兩組 step 結果一致：

* status = MAX_ITER
* iters = 199（達到 max_outer_iter）
* norm_init ≈ 1.7946 × 10⁴
* norm_final ≈ 2.1
* dt_min_seen = 1e-5
* dt_max_seen = 0.1
* last_gmres_info = 0（GMRES 正常返回）

觀察：

A1 在 240 秒內跑滿 200 次外循環，殘差僅從 1.79e4 降至約 2，仍遠高於 1e-4。

這不是預算殺死，而是**算法收斂速度不足或算子方向存在問題**。

---

# 4. 深度診斷分析

## 4.1 Baseline 結構

* log_max_exp_inner ≈ 21.86
* diag_term_min ≈ -1.93e-4
* diag_pois_max ≈ -1443

Newton 線性化在此構造下可收斂，只是需要約 33 次迭代。

---

## 4.2 A1 結構

* log_max_exp_inner ≈ 19 ~ 20
* diag_term_min 量級約 1e-5
* M_diag_min_before > 0

A1 在數值上保持穩定（未發散），但殘差下降速率顯著慢於 Newton。

這表明：

* 問題不在 dt 漂移；
* 也不在 GMRES 崩潰；
* 而可能在 A1 線性系統的符號或構造形式。

---

# 5. 關鍵結論

1. Newton 在 0.0V 有解（33 次迭代可收斂）。
2. A1 在 200 次外循環內無法達到 tol。
3. 分離（Separation）條件未成立。
4. 當前證據甚至顯示 Newton 在此場景下優於 A1。

---

# 6. 階段意義

test26 的價值在於：

* 澄清 Baseline 並非「無 basin」；
* 排除預算不足因素；
* 將問題精確定位到 A1 算子構造層面。

這是一個機制級診斷，而非策略失誤。

---

# 7. 下一步方向

基於 test26 的結果，下一階段不應再調整預算或物理條件。

建議：

1. 檢查 A1 線性系統的符號構造是否符合隱式 time-marching 推導；
2. 驗證矩陣形式是否應為：
   [
   \frac{1}{\Delta t}I + J
   ]
   而非：
   [
   \frac{1}{\Delta t}I - J
   ]
3. 對 RHS 與 M_diag 的符號一致性進行理論推導與對照實驗。

---

# 8. 階段性總結

v1.4.6-026 成功完成兩件重要工作：

* 驗證 Newton basin 的存在；
* 排除預算與數值穩定性因素。

當前問題已從「參數調優」層級提升至「算法構造正確性」層級。

SC-09 已正式進入算子結構驗證階段。

---


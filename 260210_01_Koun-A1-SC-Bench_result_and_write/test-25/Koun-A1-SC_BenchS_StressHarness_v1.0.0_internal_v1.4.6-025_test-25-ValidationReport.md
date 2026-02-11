
---

# Benchmark S

## Phase Ω – A1 Anchor Tuning & Deep Diagnostics 階段報告

### 版本：v1.4.6-025

### 階段定位：SC-09（Existence Verification Stage）

---

# 1. 實驗目的

v1.4.6-024 已證明：

* Baseline 在 0.0V Anchor 失敗（MAX_ITER）。
* A1 能正確被調度，但在 Anchor 階段仍失敗。

v1.4.6-025 的目標是：

1. 為 A1 Anchor 階段引入專用數值參數（A1_BOOT_PARAMS）；
2. 限制 dt 上界、強化 GMRES 精度；
3. 增加外迭代上限；
4. 在失敗情況下強制記錄完整 probe 與數值診斷。

本階段不改變物理條件。

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
* Baseline max_iter = 30
* Baseline GMRES = 1e-2 / 80 / 20
* A1_BOOT_PARAMS：

  * gmres_tol = 1e-2
  * gmres_maxiter = 80
  * gmres_restart = 20
  * dt_init = 1e-5
  * dt_max = 0.1
  * dt_growth_cap = 1.2
  * max_outer_iter = 200

---

# 3. 實驗結果

## 3.1 Baseline @ 0.0V

兩組 step（0.2 / 0.4）完全一致：

* status = MAX_ITER
* iters = 30
* norm_init ≈ 1.7946e4
* norm_final ≈ 3.65e-4
* tol = 1e-4
* 執行時間 ≈ 21 秒

觀察：

[
3.65\times10^{-4} > 1\times10^{-4}
]

Baseline 未達收斂門檻，但已非常接近。

這屬於「慢收斂型失敗」，非數值崩潰。

---

## 3.2 A1 Anchor @ 0.0V

Bootstrap Override 成功觸發。

A1_BOOT_PARAMS 生效（tight GMRES + 限制 dt）。

但結果為：

* status = TIMEOUT
* 外迭代 ≈ 169 / 195
* 執行時間 ≈ 60 秒（達到 Anchor 預算上限）
* norm_init ≈ 1.7946e4
* norm_final ≈ 2.17 ~ 2.77
* dt_min_seen = 1e-5
* dt_max_seen = 0.1

A1 不再像 v024 那樣漂移（dt 不再衝到 10），但在 60 秒內未能將殘差推至 1e-4 附近。

---

# 4. 深度診斷

## 4.1 Baseline 結構狀態

* log_max_exp_inner ≈ 21.86
* diag_term_min ≈ -1.93e-4
* diag_pois_max ≈ -1443

這代表：

* 指數項仍具剛性；
* Newton 仍在單調區域內，但收斂速度極慢。

---

## 4.2 A1 結構狀態

* log_max_exp_inner ≈ 19 ~ 19.5
* diag_term_min ≈ -1e-5
* M_diag_min_before ≈ 1.14e4
* M_diag_min_after ≈ 1.45e3

解讀：

* A1 確實改變了內點結構；
* 指數項有效幅度降低；
* M_diag 始終保持正裕度（未接近奇異）。

但殘差下降速度仍不足以在 60 秒內完成 Anchor 收斂。

---

# 5. 與 v024 的對比

| 項目      | v024        | v025          |
| ------- | ----------- | ------------- |
| A1 失敗類型 | MAX_ITER    | TIMEOUT       |
| dt 行為   | 衝至 10.0（漂移） | 限制在 0.1       |
| GMRES   | 鬆           | 與 Baseline 同級 |
| 診斷完整性   | 不完整         | 完整            |

結論：

v025 成功將 A1 從「數值漂移」拉回到「穩定但耗時過長」狀態。

---

# 6. 當前分離狀態評估

SC-09 的存在性分離條件為：

[
\text{Baseline FAIL} \land \text{A1 CONV}
]

目前結果為：

* Baseline FAIL（MAX_ITER）
* A1 FAIL（TIMEOUT）

因此 separation 尚未成立。

---

# 7. 核心問題定位

當前問題不是物理不可解。

而是：

> Anchor 的計算量與時間預算不匹配。

具體來看：

* A1 每外迭代成本高（GMRES 80/20）
* 60 秒預算不足以完成 200 次高精度外迭代
* 殘差下降但幅度仍遠未達 1e-4

這是一個「計算資源分配問題」，而非「算法崩潰問題」。

---

# 8. 下一步建議

## 8.1 為 A1 Anchor 設置專用時間預算

建議新增：

[
\text{MAX_STEP_TIME_A1_BOOT} = 180 \text{ 或 } 240
]

僅在：

* solver_type = A1
* solver_params = A1_BOOT_PARAMS
* is_first_step = True

時啟用。

---

## 8.2 降低每步線性代價

可測試：

* gmres_maxiter = 60
* restart = 15
* 或 gmres_tol = 3e-2

目標是在方向準確與計算量間取得平衡。

---

## 8.3 補強 Baseline 診斷

為避免論證漏洞，可額外做一個診斷 run：

* Baseline @ 0V 設 max_iter = 100
* 記錄是否能過 tol

若仍無法過，則 Newton basin 更具說服力。

---

## 8.4 修正潛在風險點（工程穩定性）

建議在下一版本中：

1. 為 A1 的 GMRES 加入 try/except；
2. 將 dt_floor 參數化，避免強制回到 1e-4；
3. 初始化 current_dt 時使用 solver_params 中的 dt_init；
4. 檢查 a1_step/a1_noise 計數是否真實反映迭代過程。

---

# 9. 階段性總結

v1.4.6-025 的成果不在於成功收斂，而在於：

* 成功收緊 A1 數值策略；
* 成功獲得完整失敗態診斷數據；
* 確認當前問題屬於「計算預算不足」，而非「算法方向錯誤」。

本階段標誌著 SC-09 進入精細數值優化階段。

---


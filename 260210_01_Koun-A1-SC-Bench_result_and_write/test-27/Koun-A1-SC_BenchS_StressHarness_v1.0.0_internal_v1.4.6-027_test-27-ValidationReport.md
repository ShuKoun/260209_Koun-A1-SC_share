
---

# Benchmark S

## Phase Ω – A1 Operator Sign Flip A/B 測試階段報告

### 版本：v1.4.6-027

### 階段定位：SC-09（Existence Verification Stage）

---

# 1. 實驗目的

在 v1.4.6-026 中，我們獲得兩個關鍵結論：

1. Baseline 在 `max_iter=30` 下失敗，但在 `max_iter=100` 下 33 次即可收斂；
2. A1 在原始算子形式下（$A = \frac{1}{\Delta t}I - J$）即使給予 $240$ 秒與 $200$ 次外循環，仍無法達到收斂門檻。

因此，v1.4.6-027 的目標是進行機制級驗證：

> 檢查 A1 線性系統符號是否錯誤。

具體改動為：

* 矩陣由
  $$
  A = \frac{1}{\Delta t} I - J
  $$
  改為
  $$
  A = \frac{1}{\Delta t} I + J
  $$
* 右端項由
  $$
  RHS = +F
  $$
  改為
  $$
  RHS = -F
  $$
* 預條件對角由
  $$
  M_{\text{diag}} = \frac{1}{\Delta t} - \operatorname{diag}(J)
  $$
  改為
  $$
  M_{\text{diag}} = \frac{1}{\Delta t} + \operatorname{diag}(J)
  $$

此為完整符號翻轉測試。

---

# 2. 實驗條件

物理條件保持不變：

* Grid：MegaUltra2（640 × 320）
* Decoupled Carrier Suppression
  $$
  n_{i,\mathrm{bc}} = 10^{10}, \quad
  n_{i,\mathrm{phys}} = 10^{4}
  $$
* $Q_{\mathrm{trap}} = 3 \times 10^{19}$

數值條件：

* Baseline：

  * `max_iter = 30`
* Baseline_Diag：

  * `max_iter = 100`
* A1_BOOT：

  * `dt_init = 1e-5`
  * `dt_max = 0.1`
  * `max_outer_iter = 200`
  * Anchor 預算：$240$ 秒

---

# 3. 實驗結果

## 3.1 Baseline（max_iter=30）

兩組 step（0.2 / 0.4）一致：

* `status = MAX_ITER`
* `iters = 30`
* $ |F|_{\text{final}} \approx 3.65 \times 10^{-4} $
* 收斂門檻 $1 \times 10^{-4}$

Baseline 在嚴格預算下失敗，但殘差已非常接近收斂。

---

## 3.2 Baseline 驗屍（max_iter=100）

兩組 step 一致：

* `status = CONVERGED`
* `iters = 33`

結論：

$$
\text{Newton basin 在 } 0.0V \text{ 明確存在}
$$

此場景並非「無解場」，僅為 Baseline 原始預算過緊。

---

## 3.3 A1（Sign Flip 版本）

兩組 step 一致：

* `status = DT_COLLAPSE`
* `iters = 5`
* 執行時間約 $2$ 秒
* 初始殘差：
  $$
  |F|_{\text{init}} \approx 1.7946 \times 10^{4}
  $$
* 最終殘差：
  $$
  |F|_{\text{final}} \approx 1.7946 \times 10^{4}
  $$

殘差完全未下降。

時間步演化：

* $dt_{\text{before}} = 10^{-5}$
* 連續縮減至
  $$
  dt_{\text{after}} \approx 6.4 \times 10^{-10}
  $$
* 觸發 `DT_COLLAPSE`

GMRES 未報錯（`info=0`），但所有 trial step 均未被 line-search 接受。

---

# 4. 機制分析

v026 的 A1 至少能將殘差從 $1.79 \times 10^{4}$ 降至約 $2$，顯示方向大致可用但收斂速度不足。

而 v027 的完整符號翻轉結果為：

$$
|F| \text{ 完全不下降}
$$

並在 5 次外循環內進入：

$$
dt \rightarrow 0
$$

這說明：

* 方向對 merit function（$ \frac{1}{2}|F|^2 $）而言完全不可接受；
* 這不是「慢」，而是「方向錯誤」。

因此：

> 同時翻轉矩陣、右端與預條件對角的三重改動過於激進。

---

# 5. 關鍵結論

1. Baseline 可在 33 次迭代內收斂；
2. A1 原始形式收斂極慢；
3. A1 全符號翻轉後完全失效；
4. 問題已從「預算不足」上升為「算子結構一致性問題」。

---

# 6. 階段意義

test27 的價值在於：

* 明確排除「符號全部翻轉即可修復」的假設；
* 證明 A1 收斂問題不是單純方向顛倒；
* 將問題定位為「線性化結構與 merit 函數之間的不一致」。

---

# 7. 下一步方向

下一階段不應再做整體翻轉，而應進行最小分解式 A/B：

* 僅翻轉 RHS：
  $$
  RHS = -F
  $$
* 僅翻轉矩陣：
  $$
  A = \frac{1}{\Delta t} I + J
  $$
* 僅翻轉預條件對角
* 與原版並行對照

透過分解測試確定哪一個符號導致下降性破壞。

---

# 8. 階段總結

v1.4.6-027 並未實現 separation，
但成功完成一次關鍵機制排除：

> A1 的收斂停滯並非簡單符號方向錯誤。

SC-09 已正式進入算子結構精確定位階段。

---




---

# Validation Report — Test-05

**Benchmark S / Stress Harness v1.4.6-005**

---

## 一、測試定位與目的（Test Positioning & Objective）

Test-05 並非驗證 Harness 的正確性或穩定性（該目標已於 Test-04 完成），而是**正式進入 Failure-Separation 主線測試的第一輪實戰嘗試**。

其核心目的為：

1. 在**不修改任何 Solver 內核、不調整任何演算法參數**的前提下；
2. 僅透過 **測試結構與物理驅動條件的改變**；
3. 嘗試誘發 **Baseline Newton Solver 的自然數值失效（NUMERIC_FAIL）**；
4. 同時透過 **Early Relay 機制**，確保 A1 Solver 在 Baseline 失效時仍具備可觀測行為；
5. 驗證「單純提高偏壓（Bias）」是否足以構成一個可重現的數值失效反例。

因此，Test-05 是一個**策略性探測測試（Strategic Probe）**，而非最終結論測試。

---

## 二、測試條件與變更摘要（Configuration Summary）

### 1. 與 Test-04 的差異

Test-05 **僅進行一項結構性變更**：

* **Case C4（The Wall）之 RelayBias**

  * 從 **2.5V 提高至 6.0V**
  * 其餘物理參數完全不變：

    * `SlotW_nm = 1.5`
    * `Q_trap = 3.0e18`
    * `Alpha = 0.15`

此變更的目的是：
**強制 Baseline 進入更高非線性與更大數值跨度的偏壓區域**，以測試是否會自然引發 Jacobian 條件數惡化所導致的數值失效。

---

### 2. 明確不變條件（Hard Invariants）

在 Test-05 中，以下項目被**嚴格凍結**：

* Baseline Solver：

  * Newton 外迭代上限、GMRES 容忍度、restart 次數、line search 行為 **完全不變**
* A1 Solver：

  * dt 策略、GMRES 參數、outer iteration 上限 **完全不變**
* Harness 結構：

  * RelaySnap、Early Relay、K-Space Coarse-to-Fine、Log Schema **完全不變**
* 時間預算：

  * 首步 60s，其餘 30s

Test-05 的所有觀測結果，均可視為**純粹由偏壓提升所導致的行為變化**。

---

## 三、測試範圍（Test Coverage）

* **Cases**

  * C2（Asymmetric）
  * C3（Trap-Heavy）
  * C4（The Wall, RelayBias = 6.0V）

* **Grid**

  * Std：120 × 60
  * Dense：180 × 90

* **Baseline Bias Step**

  * 0.2 V
  * 0.4 V

* **A1 Sprint 範圍**

  * 固定 +0.5 V（自接力點起）

---

## 四、實際觀測結果（Observed Results）

### 1. 收斂與失效統計（以 FullLog 為準）

* **Baseline Solver**

  * 在所有 Case / Grid / Step 組合下，皆成功收斂至對齊後的 RelayBias
  * **未出現任何 NUMERIC_FAIL**
  * 未觀測到 GMRES_FAIL、LS_FAIL、MAX_ITER 或 DT 類型失效

* **A1 Solver**

  * 在所有情況下皆完成接力並完成 Sprint 區段
  * 同樣全數收斂，未出現失效

### 2. C4（The Wall）關鍵觀測

在 **RelayBias = 6.0V** 的設定下：

* Baseline：

  * 成功收斂至 6.0V
  * Newton 外迭代次數低（約 3–4 次）
  * 單步時間顯著低於 30s budget
* A1：

  * 自 6.0V 接力後，成功推進至 6.5V
  * 行為平穩，未呈現數值震盪或退化徵象

整體而言，**Baseline 並未呈現接近失效邊界的數值特徵**。

---

## 五、結果解讀（Interpretation）

### 1. Test-05 的「失敗」並非實驗失敗

Test-05 未能誘發 Baseline 的 NUMERIC_FAIL，**不代表測試策略錯誤**，而是提供了一個非常重要的結論：

> **在目前的物理模型與 Solver 容忍度下，
> 僅透過將偏壓提升至 6.0V，仍不足以構成數值不可解區域。**

換言之，Case C4（1.5nm / 3e18 / Alpha 0.15）
在目前設定中仍屬於 **可解問題（Well-posed under current solver robustness）**。

---

### 2. 為何單純拉高 Bias 仍然不足

Test-04 與 Test-05 的連續結果顯示：

* Baseline 在低偏壓（2.4V）幾乎無掙扎；
* 在中高偏壓（6.0V）仍保持穩定收斂；
* Jacobian 的惡化程度尚未跨越 GMRES / Newton 的容忍門檻。

這意味著：

* **失效的瓶頸並非「偏壓幅度本身」**；
* 而更可能來自於：

  * 幾何尺度的進一步收縮；
  * Trap 密度的量級提升；
  * 或非對稱性所引發的結構性退化。

---

## 六、Test-05 的實際價值（What Test-05 Actually Proves）

Test-05 明確證明了三件事：

1. **Failure-Separation 測試不能僅依賴 Bias 拉升**
2. **Early Relay 與 Harness 架構在高偏壓區仍保持完全可用**
3. **目前 Solver 的魯棒性上限高於預期**

這三點對後續研究至關重要，因為它們**排除了錯誤方向**，並將搜尋空間縮小至「真正會破壞數值結構的物理因素」。

---

## 七、結論與後續方向（Conclusion & Next Step）

### 結論

Test-05 是一次**成功的策略性探測測試**：

* 它清楚地否定了「僅靠提高 Bias 即可誘發 NUMERIC_FAIL」的假設；
* 並為下一階段測試提供了明確的方向收斂。

### 後續（Test-06）

基於 Test-05 的結果，後續測試將進入 **物理加硬階段**，可能路徑包括：

1. **幾何尺度收縮**

   * `SlotW_nm`: 1.5 → 1.0 或更小
2. **Trap 密度量級提升**

   * `Q_trap`: 3e18 → 1e19
3. **非對稱性增強**

   * `Alpha`: 0.15 → 0.05
4. **網格密度升級**

   * 新增 Ultra Grid（240 × 120）

所有變更仍將遵守：**一次只改一個物理維度**，以保持因果可解釋性。

---

## 八、總結一句話

> **Test-05 的價值不在於「打倒 Baseline」，
> 而在於明確指出：
> 若要構造一個可發表的數值失效反例，
> 僅提高偏壓是不夠的，
> 必須進一步破壞問題本身的物理結構。**

---




---

# Validation Report — Test04

**Benchmark S / Stress Harness v1.4.6-004a**

## 一、測試目的（Test Objective）

Test04 的設計目的**並非**驗證演算法在極端物理條件下是否必然失效，而是作為 **Physics Stress 主線前的最後一次結構性驗證測試**，其核心目標為：

1. 驗證 **Harness v1.4.6-004a** 在實際 GPU 環境中可長時間穩定運行；
2. 驗證 **Early Relay、去特權（Deprivileged）、網格對齊（Grid Snapping）與資料語義透傳** 等關鍵機制是否如設計般生效；
3. 驗證在多組 **Grid × Step × Case** 組合下，Baseline 與 A1 的執行流程完整、可觀測、不中斷；
4. 為後續 **以「物理難度誘發 NUMERIC_FAIL」為目標的測試（test05）** 提供一個可信的基線狀態。

因此，Test04 屬於**結構性與工程層級的 Validation Test**，而非最終的 Failure-Separation Test。

---

## 二、測試配置摘要（Configuration Summary）

### 1. 環境與執行條件

* GPU 後端：CUDA（JAX GPU backend）
* CUDA Graph：**完全關閉**
* 記憶體策略：

  * `XLA_PYTHON_CLIENT_PREALLOCATE = false`
  * `XLA_PYTHON_CLIENT_MEM_FRACTION = 0.35`
* 精度：強制 64-bit 浮點
* Warmup 策略：
  僅預熱 **Baseline 與 A1 的標準路徑**，已完全移除任何 Hard GMRES / 特權分支的預編譯

上述設定在整個測試期間未發生崩潰、重啟或後端錯誤。

---

### 2. 測試案例與掃描軸

**物理案例（Cases）**

* C2：Asymmetric（中等難度）
* C3：Trap-heavy（高難度）
* C4：The Wall（極端物理設定）

**網格密度（Grid）**

* Std：120 × 60
* Dense：180 × 90

**Baseline Bias Step**

* 0.2 V
* 0.4 V

**A1 Bias Step**

* C2：0.1 V
* C3 / C4：0.05 V

**時間預算**

* 首步：60 s
* 其餘步驟：30 s

---

## 三、關鍵機制驗證結果（Mechanism Validation）

### 1. RelaySnap 與網格對齊（Grid Snapping）

* 所有 RelayBias 皆被正確對齊至 **Baseline 的 bias grid**；
* C4（RelayBias = 2.5 V）在 base_step = 0.2 / 0.4 時，均正確 snap 至 **2.4 V**；
* Snapping 偏差（delta）均被完整記錄於輸出資料中。

此行為符合設計預期，確保 Baseline 不會被強制推入未對齊的 bias 狀態。

---

### 2. Early Relay 機制（本次測試未觸發，但邏輯驗證完成）

在 Test04 中，所有 Baseline 皆成功收斂至其對齊後的 RelayBias，因此 **Early Relay 並未實際觸發**。

然而：

* Early Relay 的邏輯已在程式層完整實作；
* 若 Baseline 在 RelayBias 前失效，A1 將可自 **Baseline 最後成功狀態** 起跑；
* 起跑 bias 將以 **floor 對齊至 A1 grid**，避免任何「超前接力」。

雖未在本次數據中實際使用，但該機制已被結構性驗證為可用，並將在後續 test 中成為關鍵保險機制。

---

### 3. 去特權（Deprivileged）一致性

* A1 在所有測試中：

  * 未啟用 Hard GMRES；
  * 未使用任何 Relay 專屬 dt boost；
  * 僅依賴其核心的 dt 自適應與標準 GMRES 參數。
* Warmup、Solver 與 Harness 三者邏輯一致，不存在「敘事去特權、實作仍殘留特權」的情況。

此點對後續學術審查具有關鍵意義。

---

### 4. 資料語義完整性（Data Integrity）

* 所有輸出皆包含：

  * solver 類型
  * case / grid / step
  * bias 位置
  * 收斂狀態（CONV / fail_class）
  * 計算時間、迭代次數
* 資料可完整對齊 FullLog 與 Summary；
* 未出現資料缺欄、欄位語義衝突或不可追溯情形。

---

## 四、測試結果總覽（Observed Results）

### 1. 收斂狀態

* **Baseline**：
  在所有 Case / Grid / Step 組合下，皆成功收斂至其對齊後的 RelayBias；
* **A1**：
  在所有情況下皆完成 Sprint 區段（+0.5 V），且全部收斂。

### 2. Failure 類型統計

* `NUMERIC_FAIL`：0
* `BUDGET_TIMEOUT`：0
* `GMRES_FAIL / LS_FAIL / DT_COLLAPSE`：0

所有步驟的 `fail_class` 均為 **CONV**。

---

## 五、結果解讀（Interpretation）

Test04 的結果**不代表**：

* Baseline 在極端物理條件下永遠穩定；
* A1 已在本測試中展現其 Failure-Separation 優勢。

Test04 實際證明的是：

1. 在目前的 C4 物理強度與 Bias 掃描上限下，該問題仍處於**可解區間**；
2. Harness v1.4.6-004a 在工程與資料語義層面已達到**可作為論文基礎的可靠狀態**；
3. 若要誘發 Baseline 的 NUMERIC_FAIL，需進一步：

   * 提高 Bias 掃描範圍，或
   * 加強物理難度（幾何、Trap、非對稱性、網格密度）。

---

## 六、結論與後續方向（Conclusion & Next Step）

### 結論

Test04 是一次**成功的 Validation Test**：

* 成功驗證了整個 Stress Harness 的穩定性、可觀測性與審計一致性；
* 消除了後續 Failure-Separation 測試中可能來自工程或資料層的干擾因素。

### 後續（Test05）

下一階段測試將以 **誘發結構性失效** 為明確目標，策略包括但不限於：

* 提高 C4 的 RelayBias（例如 6.0 V 或更高）；
* 或進一步加強 C4 的物理極限設定；
* 並利用已驗證完成的 Early Relay 機制，確保 A1 在 Baseline 失效後仍具可觀測性。

---


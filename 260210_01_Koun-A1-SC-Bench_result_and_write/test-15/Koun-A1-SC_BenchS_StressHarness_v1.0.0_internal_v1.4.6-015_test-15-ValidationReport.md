


# Benchmark S – Stress Harness v1.4.6-015

## Phase W：次奈米幾何壓縮（Sub-nm on Ultra, C4 Only）

**母版本：** v1.4.6-014
**變更類型：** 幾何軸單變量推進（在高解析度網格上）
**測試範圍：** 僅保留 C4（C4 Only），僅運行 Ultra 網格（240×120）

---

# 1. 測試目標

在 v1.4.6-014（Std/Dense/Ultra）中已驗證：

* 偏壓域擴展至 12V 仍全收斂
* Ultra（240×120）網格亦未暴露離散化屏蔽導致的失效

因此本階段進入真正的幾何軸強化，嘗試進入次奈米尺度下的場梯度尖峰區。

本輪 Test（v1.4.6-015）採用：

* 固定高解析度 Ultra 網格（240×120）
* 僅保留最極端案例 C4（聚焦幾何壓縮效應）
* 對 C4 施加單變量幾何壓縮：

> SlotW_nm：1.0 → 0.5

其餘物理與算法條件全部鎖定不變，以保持因果可解釋性與可審計性。

---

# 2. 實驗設置（鎖定項）

## 2.1 網格與步進

* Grid：Ultra（240×120）
* Baseline step：0.2V、0.4V
* Baseline 掃描：0 → Relay snapped bias（12.0V）
* A1 Sprint：從 12.0V 起，span=0.5V（至 12.5V）
* A1 step：0.05V（10 steps）

## 2.2 C4 參數（鎖定與變更）

**鎖定（繼承 v1.4.6-014 / v1.4.6-013）：**

* BiasMax = 12.0V
* RelayBias = 12.0V
* Q_trap = 3.0e19
* Alpha = 0.00

**唯一變更（本輪單變量）：**

* SlotW_nm：1.0 → 0.5

## 2.3 算法與資料（完全鎖定）

* Solver 參數：不變
* Harness 邏輯：不變（含 Early Relay）
* Data schema：不變
* 輸出規範：純文字（無 emoji）

---

# 3. 全域結果統計（以 FullLog 為準）

我已讀取並核驗：

* Stress_v1.4.6-015_FullLog.csv
* Stress_v1.4.6-015_Summary.csv

統計結果：

* FullLog 總行數：50
* fail_class：

  * CONV：50
  * NUMERIC_FAIL：0
  * BUDGET_TIMEOUT：0
* failure separation：未出現

結論：本輪在次奈米幾何壓縮（SlotW=0.5nm）下，Baseline 與 A1 仍全部成功收斂。

---

# 4. 關鍵點分析（C4 @ 12.0V / 12.5V）

## 4.1 Baseline（C4 @ 12.0V）

* base_step=0.2：iters=3，time≈0.806s
* base_step=0.4：iters=4，time≈1.212s

未出現：

* GMRES_FAIL / GMRES_EXCEPT
* LS_FAIL
* MAX_ITER
* TIMEOUT

## 4.2 A1（C4 @ 12.5V）

* 兩組皆為 CONV(3/0/0)
* dt_end = 10.0（仍快速達到 dt_max）
* 未出現 dt shrink、DT_COLLAPSE 或 NOISE 主導行為

---

# 5. 結果解釋

本輪結果顯示：

1. 在 Ultra（240×120）解析度下，將 SlotW 由 1.0nm 壓縮至 0.5nm，仍未改變 Baseline 的可解性。
2. Baseline 在 12.0V 仍維持低迭代次數（3~4），未呈現病態前兆。
3. A1 仍完全未進入治理壓力區（dt 行為健康）。

但同時，本輪呈現一個關鍵技術警訊：

> 在 Ultra（240×120）下，dx 約為 0.418nm 級。
> SlotW=0.5nm 仍接近 1–2 個 cell 的寬度，可能未被充分解析。

因此「幾何壓縮未造成顯著難度提升」不必然等價為「物理幾何不敏感」，也可能表示：

* 幾何變更仍處於離散解析極限附近，未能在離散層面形成真正尖銳梯度。

---

# 6. 結論

v1.4.6-015 成功完成：

* 在高解析度（Ultra）網格上
* 針對 C4 進行次奈米幾何壓縮（SlotW=0.5nm）
* 且保持 solver/harness/schema 完全鎖定
* 得到全收斂結果（無任何失效）

本輪未產生 failure separation，也未觀察到明顯病態前兆。

---

# 7. 建議的下一步（策略延續）

為真正驗證 0.5nm 幾何壓縮是否能產生數值壓力，下一步應先確保幾何被充分解析：

* 將網格密度提升至 dx ≤ 0.25nm 級（例如 Nx≈400 或更高）
* 保持 C4 Only、保持所有物理參數不變（仍為 SlotW=0.5nm、12V、3e19、Alpha=0.00）
* 僅以更高解析度作為驗證軸（Discretization Resolution Escalation）

若在更高解析度下仍無明顯退化，才可更有力地宣告：

> 在本模型與算法框架下，Baseline 對該幾何壓縮呈現結構性穩健。

---

# 8. 合規聲明

本測試遵守：

* 單變量硬化原則（相對 v014 僅改 SlotW_nm）
* Solver / Harness / Data schema 全鎖定
* 純文字輸出（無 emoji）
* 乾淨進程執行

---


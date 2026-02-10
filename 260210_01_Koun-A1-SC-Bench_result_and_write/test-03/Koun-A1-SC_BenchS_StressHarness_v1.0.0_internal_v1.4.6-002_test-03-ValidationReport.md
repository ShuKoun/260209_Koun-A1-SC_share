
---

# 📘 Benchmark S 壓力測試報告

## Stress Harness v1.4.6-002（Extreme Budget）

**測試日期：** 2026-02-10
**測試環境：**

* GPU：RunPod（CUDA）
* JAX：GPU 模式（CUDA Graph 已關閉）
* XLA 設定：

  * `XLA_FLAGS=--xla_gpu_enable_command_buffer=`
  * `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  * `XLA_PYTHON_CLIENT_MEM_FRACTION=0.35`

---

## 1. 測試目的

本次測試為 **Benchmark S「預算邊界探索（Budget Frontier）」** 的第二個極限點，目標在於：

1. 驗證 **3.0 s normal step budget** 下：

   * Baseline Newton continuation 是否仍可穩定存活；
   * A1-SC relay + sprint 是否會因預算不足而系統性失效。
2. 與 v1.4.6-001（5.0 s budget）結果形成對照，建立清晰的 **A1 與 Baseline 的預算生存邊界**。

---

## 2. 測試設定概要

### 2.1 測試軸線

* **Grid Density：**

  * Std (120 × 60)
  * Dense (180 × 90)
* **Baseline Step Size：**

  * 0.2 V
  * 0.4 V
* **Case：**

  * C2（Asymmetric，Relay = 7.0 V / 6.8 V）
  * C3（Killer，Relay = 4.0 V）

### 2.2 預算設定

* `MAX_STEP_TIME_FIRST = 60.0 s`（Anchor / Cold Start）
* `MAX_STEP_TIME_NORMAL = 3.0 s`（Extreme Budget）

### 2.3 不變條件（Invariant）

* 物理模型、算子、JIT kernel 與 v1.4.6 完全一致
* A1 與 Baseline 皆未調整任何數值或 heuristic
* 僅改變 **normal step 的時間預算**

---

## 3. 測試結果總覽

### 3.1 Log 完整性

* **FullLog：** 90 rows
* **Summary：** 16 rows
* 所有測試流程正常完成，無 CUDA OOM、無 driver crash

### 3.2 Fail Class 統計（FullLog）

| 類型                   | 次數 |
| -------------------- | -- |
| CONV                 | 82 |
| BUDGET_TIMEOUT       | 8  |
| NUMERIC_FAIL / OTHER | 0  |

---

## 4. 關鍵觀察結果

### 4.1 Baseline Newton 行為

* **所有 Baseline runs 全數 CONV（0 次 TIMEOUT）**
* 即使在 Dense grid 與 3.0 s 極限預算下，Baseline continuation 仍可穩定完成至 relay bias
* Baseline 終點步耗時多落在 **約 1 s 左右**，顯示其計算負載在 continuation 路徑上分佈相對均勻

**結論：**
在目前數值難度下，**3.0 s 預算不足以迫使 Baseline Newton 失效**。

---

### 4.2 A1-SC 行為（關鍵）

所有 **BUDGET_TIMEOUT** 均發生於 **A1 relay 後的第一個 sprint step**，且模式高度一致：

#### C2（Relay = 7.0 V / 6.8 V）

* Std / Dense × base_step 0.2：

  * **7.10 V TIMEOUT**
* Std / Dense × base_step 0.4：

  * **6.90 V TIMEOUT**

#### C3（Relay = 4.0 V）

* 所有 grid × step：

  * **4.05 V TIMEOUT**

實際量測時間顯示，該步驟耗時集中於 **約 3.0–3.3 s**，剛好超過預算上限。

**結論：**

* A1-SC 在 relay 後的第一個 sprint step 存在一個 **明確的計算時間下限**
* 在 3.0 s normal budget 下，該步驟**系統性超時**
* 此現象與 grid 密度無顯著相關，屬於 **策略結構 × 預算** 主導

---

## 5. 與 v1.4.6-001（5.0 s）之對照

| 預算    | Baseline | A1                   |
| ----- | -------- | -------------------- |
| 5.0 s | 全數存活     | 僅少量、邊界性 TIMEOUT      |
| 3.0 s | 全數存活     | relay 後第一步全面 TIMEOUT |

這說明：

* **壓低預算首先淘汰的是 A1，而非 Baseline**
* Baseline 的「慢但均勻」特性，在極低預算下反而具韌性
* A1 的 relay + sprint 策略在過低預算時會承受較大的瞬時計算負載

---

## 6. 測試價值評估

### 6.1 本次測試的價值

✅ **高度有價值（Boundary-Defining Data）**

* 明確界定 **A1-SC 在 3.0 s 預算下的失效邊界**
* 與 5.0 s 結果形成完整的 **Budget Frontier 約束**
* 排除了環境因素（CUDA Graph / OOM）干擾，結果可信

### 6.2 不能得出的結論

❌ 本測試 **不支持**「A1 在極低預算下比 Baseline 更耐用」
（事實恰好相反）

---

## 7. 後續建議

### 建議一：補齊預算邊界

* 建議追加 **4.0 s 或 4.5 s** 測試點
* 目的：精確定位 A1-SC 的臨界預算區間（介於 3.0–5.0 s）

### 建議二：若目標是 Baseline 先失效

* 不應再單純降低 budget
* 應提高 **數值難度軸**（更密 grid、更多 trap、極端 case）
* 同時維持 budget 在 A1 尚可啟動的區間

---

## 8. 總結性結論

> **Stress Harness v1.4.6-002 成功揭示了 A1-SC 在 3.0 s normal budget 下的明確失效邊界，而 Baseline Newton 在相同條件下仍保持完全穩定。**

此結果為後續 **Budget Frontier 曲線繪製** 提供了關鍵錨點，應作為正式實驗數據保留與引用。

---



---

# Test-07 Validation Report

## v1.4.6-007

### （Physics Hardening B1: SlotW 1.0nm）

---

## 一、測試目的

本次 Test-07 為「物理單變量加硬」的第一步（B1），其目標為：

1. 在偏壓軸已被完整驗證（8.0V 仍全收斂）後，
2. 透過**幾何尺寸壓縮**提高泊松方程剛性，
3. 嘗試誘發 Baseline（Newton + GMRES）出現自然數值失效（NUMERIC_FAIL），
4. 並觀察是否形成 Failure Separation（Baseline 失效而 A1 可延續）。

本測試遵守單變量原則：
除 C4 的 SlotW 外，其餘物理、數值與架構參數全部鎖定。

---

## 二、測試設定摘要

### 2.1 幾何與物理參數（僅 C4 修改）

| 參數        | v1.4.6-006 | v1.4.6-007 |
| --------- | ---------- | ---------- |
| SlotW_nm  | 1.5        | **1.0**    |
| Q_trap    | 3.0e18     | 3.0e18     |
| Alpha     | 0.15       | 0.15       |
| RelayBias | 8.0V       | 8.0V       |

### 2.2 測試維度

* Grid：

  * Std (120×60)
  * Dense (180×90)
* Baseline Step：

  * 0.2V
  * 0.4V
* A1 Step：

  * 依 case 設定（0.05 / 0.1）
* Time Budget：

  * 首步 60s
  * 常規 30s

### 2.3 架構與資料規範

* Solver 參數未修改
* Early Relay 機制保留
* 統一資料欄位：

  * `baseline_last_success_bias`
  * `baseline_fail_class`
  * `baseline_fail_reason`
  * `relay_bias_a1_delta = relay_bias_a1_start - baseline_last_success_bias`
* GPU Hardened 設定保留

---

## 三、核心結果

### 3.1 全域統計

依據 FullLog：

* 總記錄數：約 230 行
* `fail_class` 統計：

| 類型             | 次數   |
| -------------- | ---- |
| CONV           | 100% |
| NUMERIC_FAIL   | 0    |
| BUDGET_TIMEOUT | 0    |
| OTHER          | 0    |

**未出現任何數值崩潰。**

---

### 3.2 C4（SlotW 1.0nm）在 8.0V 的行為

#### Baseline（0 → 8.0V）

四組組合全部收斂：

* Std / 0.2
* Std / 0.4
* Dense / 0.2
* Dense / 0.4

觀察：

* Newton 外迭代：約 3～4 次
* 無 GMRES_FAIL
* 無 LS_FAIL
* 無 MAX_ITER
* 無 TIMEOUT
* 線性解算時間略高於 1.5nm 情況，但仍穩定

#### A1（8.0 → 8.5V Sprint）

* 全部收斂
* outer iteration 約 3 次
* `relay_type = TARGET`
* `relay_bias_a1_delta = 0.0`

---

## 四、數值層面分析

### 4.1 幾何壓縮效應

SlotW 從 1.5nm → 1.0nm 為 33% 尺寸縮減，理論上應：

* 增強電場梯度
* 增強泊松剛性
* 提升 Jacobian 條件數

然而本模型與目前網格密度下：

* 系統仍位於可解域
* GMRES 未顯示退化跡象
* Newton 仍在少量外迭代內收斂

### 4.2 Failure Separation 判定

本測試未形成：

* Baseline NUMERIC_FAIL
* Early Relay 觸發
* A1 接力後延續

因此尚未產生治理分離數據。

---

## 五、科學定位

Test-07 的價值在於：

1. 證明偏壓軸已完全飽和（8.0V 不足以觸發失效）
2. 證明幾何單變量壓縮至 1.0nm 仍未觸發失效
3. 建立物理強度邊界條件的證據鏈
4. 驗證資料 schema 在高強度條件下仍保持一致

這些結果排除了兩個可能的失效來源：

* 高偏壓
* 單一幾何壓縮

---

## 六、下一步策略建議

依照單變量原則，下一輪應進入：

### B2：Trap 強度提升

僅修改 C4：

* `Q_trap = 3.0e18 → 1.0e19`

保持：

* SlotW = 1.0
* Alpha = 0.15
* RelayBias = 8.0
* Solver 不變
* Grid 不變

若 B2 仍全 CONV，再進入：

### B3：Alpha 非對稱加強（0.15 → 0.05）

---

## 七、總結

Test-07 結論如下：

* 幾何壓縮至 1.0nm 未觸發數值崩潰
* Baseline 仍在穩定可解區域
* A1 未觸發接力模式
* Failure Separation 尚未出現

**物理強度仍不足以擊穿經典 Newton 框架。**

---


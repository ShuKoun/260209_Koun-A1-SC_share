# Test11 Report

## BenchS_StressHarness_v1.4.6-011

### Physics Hardening B3 (Alpha = 0.05)

---

## 1. Objective

Test11 是在完成 Trap 軸單變量硬化（Q_trap → 3.0e19）後，轉向 **邊界非對稱因子 Alpha** 的單變量試探。

在以下條件完全鎖定的前提下：

* SlotW = 1.0 nm
* Q_trap = 3.0e19
* RelayBias = 8.0 V
* Solver 參數不變
* Harness 邏輯不變
* Data schema 不變
* Early Relay 機制保留

僅對 C4 進行：

```
Alpha: 0.15 → 0.05
```

目標：

1. 驗證邊界非對稱性是否為觸發 Jacobian 病態的關鍵因素
2. 檢查是否首次出現 NUMERIC_FAIL
3. 評估是否形成 failure separation
4. 若仍穩定，確立 Alpha 軸在此區間內的無效性

---

## 2. Experimental Configuration

### 2.1 Invariants（鎖定項）

| 模組                | 狀態                           |
| ----------------- | ---------------------------- |
| Solver parameters | 不變                           |
| Harness logic     | 不變                           |
| Data schema       | 不變                           |
| SlotW             | 1.0 nm                       |
| Q_trap            | 3.0e19                       |
| RelayBias         | 8.0 V                        |
| Grid              | Std (120×60), Dense (180×90) |
| Baseline step     | 0.2V, 0.4V                   |

### 2.2 單變量變更（B3）

僅修改：

```
C4: Alpha = 0.05
```

其餘 Case（C2 / C3）作為對照組。

---

## 3. Global Outcome

### 3.1 FullLog 統計

* 總條目數：230
* CONV：230
* NUMERIC_FAIL：0
* BUDGET_TIMEOUT：0
* failure separation：未出現

---

## 4. Hard Case Analysis — C4 @ 8.0V

### 4.1 Baseline

| Grid  | Step | iters | step_time(s) | t_lin(s) |
| ----- | ---- | ----- | ------------ | -------- |
| Std   | 0.2  | 3     | 0.802655     | 0.801549 |
| Std   | 0.4  | 3     | 0.804600     | 0.803415 |
| Dense | 0.2  | 3     | 0.901145     | 0.899996 |
| Dense | 0.4  | 4     | 1.397245     | 1.395731 |

觀察：

* 線性求解仍佔主導
* 迭代數未上升
* 無 GMRES_FAIL
* 無 LS_FAIL
* 無 MAX_ITER

### 4.2 A1 @ 8.5V

| Grid  | Step | outer iters | status      | dt_end |
| ----- | ---- | ----------- | ----------- | ------ |
| Std   | 0.2  | 3           | CONV(3/0/0) | 10.0   |
| Std   | 0.4  | 3           | CONV(3/0/0) | 10.0   |
| Dense | 0.2  | 3           | CONV(3/0/0) | 10.0   |
| Dense | 0.4  | 3           | CONV(3/0/0) | 10.0   |

觀察：

* dt 仍快速達到 dt_max
* 無 dt collapse
* 無 NOISE 主導行為

---

## 5. Interpretation

### 5.1 Alpha 軸影響評估

本輪結果顯示：

> 在幾何極限（1.0nm）與 Trap 極限（3.0e19）條件下，
> 將 Alpha 降至 0.05 並未顯著改變 Baseline 或 A1 的數值穩定性。

這意味著：

* 邊界非對稱性在此幅度範圍內尚未改變主導剛性結構
* Jacobian 尚未進入病態區
* 數值穩定域仍然完整

### 5.2 與前序測試的連續性

| 測試         | 變量              | 結果     |
| ---------- | --------------- | ------ |
| B1         | SlotW → 1.0     | 全 CONV |
| B2         | Q_trap → 1.0e19 | 全 CONV |
| B2-Next    | Q_trap → 2.0e19 | 全 CONV |
| B2-Extreme | Q_trap → 3.0e19 | 全 CONV |
| B3         | Alpha → 0.05    | 全 CONV |

這形成一條完整的多軸單變量硬化排除鏈。

---

## 6. Strategic Conclusion

Test11 的結果強化了以下結論：

1. Trap 軸至 3.0e19 未觸發數值崩潰
2. Alpha 軸至 0.05 未改變穩定域
3. Baseline 在當前物理與離散結構下呈現高度穩健性
4. A1 尚未進入治理壓力區

這是一個重要的負結果，證明在此物理構型內，單純調整電荷強度與邊界非對稱性不足以引發失效。

---

## 7. Next Direction

為完整封閉 Alpha 軸，建議：

```
Alpha: 0.05 → 0.00
```

若仍全 CONV，則可正式宣布：

> 在固定幾何與 Trap 強度條件下，Alpha 軸在 [0.15, 0.00] 範圍內對 Baseline 數值穩定性無決定性影響。

屆時可考慮：

* 開啟新物理軸
* 或進行離散敏感性驗證

---

## 8. Compliance Statement

本測試遵守：

* 單變量硬化原則
* Solver / Harness / Data schema 全鎖定
* 純文字輸出（無 emoji）
* 乾淨進程執行


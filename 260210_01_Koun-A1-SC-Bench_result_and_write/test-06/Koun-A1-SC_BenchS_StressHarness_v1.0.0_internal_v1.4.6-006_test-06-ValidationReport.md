
---

# Test-06 Validation Report

## v1.4.6-006-Final-Revised

### （Max Bias × Data Logic Rigor）

---

## 一、測試目的

本次 Test-06 的目標為：

1. **偏壓軸極限驗證**

   * 將 Case C4（The Wall）的 `RelayBias` 提升至 **8.0V（BiasMax）**
   * 檢驗 Baseline 在最大偏壓下是否出現自然數值失效（NUMERIC_FAIL）

2. **資料邏輯最終收斂**

   * 統一並顯式定義：

     * `baseline_last_success_bias`
     * `baseline_fail_class`
     * `baseline_fail_reason`
   * 統一計算：

     * `relay_bias_a1_delta = relay_bias_a1_start - baseline_last_success_bias`
   * 確保 TARGET 與 EARLY 分支邏輯一致

3. **架構凍結**

   * 不修改任何 Solver 參數
   * 不修改任何數值方法
   * 保持 Hardened GPU 設定（No CUDA Graph / Mem=0.35）

---

## 二、測試設定摘要

### 測試維度

| 軸             | 設定                           |
| ------------- | ---------------------------- |
| Grid          | Std (120×60), Dense (180×90) |
| Baseline Step | 0.2V, 0.4V                   |
| A1 Step       | 依 Case 定義（0.05 或 0.1）        |
| Time Budget   | 60s 首步 / 30s 常規              |

### C4（The Wall）物理參數

* SlotW = 1.5 nm
* Q_trap = 3e18
* Alpha = 0.15
* N_high = 1e21
* RelayBias = **8.0V**

---

## 三、核心結果

### 1️⃣ 全域統計

* FullLog 約 230 行
* `fail_class` 統計：

| 類型             | 次數   |
| -------------- | ---- |
| CONV           | 100% |
| NUMERIC_FAIL   | 0    |
| BUDGET_TIMEOUT | 0    |
| OTHER          | 0    |

**結論：所有案例全部成功收斂。**

---

### 2️⃣ C4 在 8.0V 的具體觀察

#### Baseline

* 四組組合（Std/Dense × 0.2/0.4）全部成功到達 8.0V
* Newton 外迭代次數：

  * 約 3～4 次
* 無 GMRES_FAIL
* 無 LS_FAIL
* 無 MAX_ITER
* 無 TIMEOUT

#### A1

* 從 8.0V 接力
* Sprint +0.5V
* 成功至 8.5V
* outer iteration 約 3 次
* `relay_bias_a1_delta = 0.0`

---

## 四、數據結構驗證（Schema Audit）

本次版本成功達成：

### ✅ Baseline Context 顯式化

無論 TARGET 或 EARLY：

* `baseline_last_success_bias`
* `baseline_fail_class`
* `baseline_fail_reason`

均完整填寫。

### ✅ Delta 統一公式

```
relay_bias_a1_delta = relay_bias_a1_start - baseline_last_success_bias
```

無任何硬編碼。

### ✅ 欄位語義分離

* `relay_bias_baseline_snapped`
* `relay_bias_a1_start`
* `relay_bias_a1_delta`

語義清晰，無混用。

---

## 五、科學意義分析

### 1️⃣ 偏壓軸已完全驗證

本系列結果顯示：

* 2.5V → 不死
* 6.0V → 不死
* 8.0V → 仍不死

在目前物理條件下：

> Baseline 的可解域至少延伸至 8.0V。

因此：

> 「Failure Separation」尚未出現。

---

### 2️⃣ 關鍵觀察

Baseline 在 8.0V 仍僅需 3～4 次 Newton 外迭代收斂，顯示：

* Jacobian 尚未嚴重退化
* GMRES 仍在可控條件數範圍內
* 非線性尚未進入失穩區域

這意味著：

> 目前 C4 的物理難度仍未觸及數值失效門檻。

---

## 六、策略結論

### 🚫 偏壓軸測試到此結束

繼續提高 Bias 不會提供新信息。

### ✅ 下一步應轉向「物理幾何加硬」

建議進入：

> **v007：SlotW 1.5nm → 1.0nm（單變量加硬）**

保持：

* Q_trap 不變
* Alpha 不變
* RelayBias 保持 8.0V
* Solver 不變

以純物理難度誘導 Jacobian 惡化。

---

## 七、總結

Test-06 證明：

1. Baseline 在 8.0V 下仍完全穩定
2. A1 無需接力救援
3. 數據 Schema 已達審稿級嚴謹
4. 偏壓軸已完成極限驗證
5. Failure Separation 尚未產生

---


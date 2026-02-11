# Test10 Report

## BenchS_StressHarness_v1.4.6-010

### Physics Hardening B2-Extreme (C4: Q_trap = 3.0e19)

---

## 1. Objective

Test10 是 Trap 軸單變量硬化序列的終點試探。

在以下條件完全鎖定的前提下：

* SlotW = 1.0 nm
* RelayBias = 8.0 V
* Alpha = 0.15
* Solver 參數不變
* Harness 邏輯不變
* Data schema 不變
* Early Relay 機制保留

僅對 C4 進一步提升空間電荷密度：

```
Q_trap: 2.0e19 → 3.0e19
```

目標：

1. 試探 Trap 軸是否能最終觸發 Baseline 的數值崩潰
2. 觀察是否首次出現 NUMERIC_FAIL
3. 檢查是否形成 failure separation
4. 若仍未崩潰，確認 Trap 軸在當前幾何與邊界條件下的有效性上限

---

## 2. Experimental Configuration

### 2.1 Invariants（嚴格鎖定）

| 模組                | 狀態                           |
| ----------------- | ---------------------------- |
| Solver parameters | 不變                           |
| Harness logic     | 不變                           |
| Data schema       | 不變                           |
| SlotW             | 1.0 nm                       |
| RelayBias         | 8.0 V                        |
| Alpha             | 0.15                         |
| Grid              | Std (120×60), Dense (180×90) |
| Baseline step     | 0.2V, 0.4V                   |

### 2.2 單變量變更（B2-Extreme）

僅修改：

```
C4: Q_trap = 3.0e19
```

其餘 Case（C2 / C3）保持原狀作為對照。

---

## 3. Global Outcome

### 3.1 FullLog 統計

* 總條目數：230
* CONV：230
* NUMERIC_FAIL：0
* BUDGET_TIMEOUT：0
* 未出現 failure separation

### 3.2 關鍵工況：C4 @ 8.0V

#### Baseline

| Grid  | Step | iters | step_time(s) | t_lin(s) |
| ----- | ---- | ----- | ------------ | -------- |
| Std   | 0.2  | 3     | ~0.80        | ≈0.797   |
| Std   | 0.4  | 3     | ~0.80        | ≈0.798   |
| Dense | 0.2  | 3     | ~0.90        | ≈0.901   |
| Dense | 0.4  | 4     | ~1.40        | ≈1.396   |

觀察：

* 線性求解主導（t_lin ≈ step_time）
* 迭代數低且穩定
* 無 GMRES_FAIL
* 無 LS_FAIL
* 無 MAX_ITER

#### A1 @ 8.5V

* 全部 CONV(3/0/0)
* dt 快速達到 dt_max = 10.0
* 無 dt collapse
* 無 shrink 主導區

---

## 4. Interpretation

### 4.1 Trap 軸單變量推進結果

至此已完成完整 Trap 硬化序列：

| Version    | Q_trap | 結果     |
| ---------- | ------ | ------ |
| B2         | 1.0e19 | 全 CONV |
| B2-Next    | 2.0e19 | 全 CONV |
| B2-Extreme | 3.0e19 | 全 CONV |

結論：

> 在 SlotW=1.0nm 與 RelayBias=8.0V 鎖定條件下，
> Trap 密度軸單獨推進至 3.0e19 仍不足以使 Baseline 進入數值病態區。

### 4.2 數值穩定性特徵

1. Jacobian 剛性提升明顯（線性求解佔比高）
2. 但未出現病態化跡象
3. A1 尚未進入治理壓力區
4. 未觀察到任何崩潰前兆

---

## 5. Strategic Conclusion

Test10 的結果具有重要意義：

> Trap 軸在當前幾何與邊界條件下，對擊穿 Baseline 的影響有限。

這意味著：

* 目前的數值結構具有顯著穩健性
* Trap 密度不是觸發 Jacobian 崩潰的主導因子
* 必須考慮其他物理軸

---

## 6. Recommended Next Phase

### Phase B3：Alpha 軸單變量試探（建議）

僅修改：

```
C4: Alpha 0.15 → 0.05
```

保持：

* Q_trap = 3.0e19
* SlotW = 1.0
* RelayBias = 8.0
* solver/harness/schema 不變

理由：

Alpha 直接影響邊界勢差結構，比單純加電荷更可能誘發數值病態。

---

## 7. Compliance

本測試遵守以下規範：

* 無 emoji
* 純文字輸出 SUCCESS / WARNING / FAILED
* 乾淨進程執行
* 單變量硬化原則

---

## 8. Overall Positioning

至 Test10 為止，我們已完成：

* 幾何極限試探
* Trap 軸完整單變量推進
* 全部穩定收斂

這構成一條完整且可審計的“邊界逼近證據鏈”。

下一階段將轉向第二物理軸以尋找真正的崩潰觸發機制。

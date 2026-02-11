


# Test12 Report

## BenchS_StressHarness_v1.4.6-012

### Physics Hardening B3-Extreme (C4: Alpha = 0.00)

---

## 1. Objective

Test12 旨在封閉 Alpha 軸的極限點，驗證「邊界非對稱性」是否可能成為擊穿 Baseline 的主導因子。

在以下條件完全鎖定的前提下：

* SlotW = 1.0 nm
* Q_trap = 3.0e19
* RelayBias = 8.0 V
* Solver 參數不變
* Harness 邏輯不變
* Data schema 不變
* Early Relay 機制保留

僅對 C4 做單變量變更：

* Alpha：0.05 → 0.00

目標：

1. 檢查是否首次出現 NUMERIC_FAIL
2. 檢查是否形成 failure separation
3. 若仍穩定，給出 Alpha 軸在此範圍內的「無效性」結論

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

### 2.2 單變量變更（B3-Extreme）

僅修改：

* C4：Alpha = 0.00

其他 Case（C2/C3）保持原狀作為對照。

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

### 4.1 Baseline（C4, bias=8.0）

| Grid  | base_step | iters | step_time(s) | t_lin(s) | t_res(s) |  t_ls(s) |
| ----- | --------: | ----: | -----------: | -------: | -------: | -------: |
| Std   |       0.2 |     3 |     0.823017 | 0.821902 | 0.000514 | 0.000583 |
| Std   |       0.4 |     3 |     0.805149 | 0.804041 | 0.000518 | 0.000568 |
| Dense |       0.2 |     3 |     0.906847 | 0.905681 | 0.000540 | 0.000607 |
| Dense |       0.4 |     4 |     1.400289 | 1.398850 | 0.000599 | 0.000816 |

觀察：

* t_lin 幾乎佔滿 step_time（線性求解主導）
* 迭代數仍低（3~4）
* 無 GMRES_FAIL / LS_FAIL / MAX_ITER / TIMEOUT

### 4.2 A1（C4, bias=8.5）

| Grid  | base_step | outer iters | status      | dt_end | step_time(s) | t_lin(s) |
| ----- | --------: | ----------: | ----------- | -----: | -----------: | -------: |
| Std   |       0.2 |           3 | CONV(3/0/0) |   10.0 |     0.674468 | 0.672986 |
| Std   |       0.4 |           3 | CONV(3/0/0) |   10.0 |     0.670295 | 0.668767 |
| Dense |       0.2 |           3 | CONV(3/0/0) |   10.0 |     0.696852 | 0.695166 |
| Dense |       0.4 |           3 | CONV(3/0/0) |   10.0 |     0.676119 | 0.674477 |

觀察：

* dt 仍快速到 dt_max=10.0
* 無 dt collapse、無 NOISE 主導跡象
* A1 仍未進入治理壓力區

---

## 5. Interpretation

Test12 顯示：

1. 在幾何極限（SlotW=1.0nm）與電荷極限（Q_trap=3.0e19）鎖定下，Alpha 走到 0.00 仍不改變穩定性格局。
2. Baseline 的線性求解仍主導耗時，但未出現病態化行為（無迭代爆增、無 GMRES/LS 失效）。
3. A1 的 dt 行為完全健康，表示尚未遇到任何需要「治理」的壓力區。

因此可以得出一個非常硬的排除結論：

* 在目前物理構型與離散設定下，Alpha 軸在 [0.15, 0.00] 的單變量掃描 **不足以觸發 Baseline 的數值失效**。

---

## 6. Strategic Conclusion

到 Test12 為止，你已經完成兩條「走到底」的排除鏈：

* Trap 軸：Q_trap 由 3e18 推進至 3e19，仍全 CONV
* Alpha 軸：Alpha 由 0.15 下探至 0.00，仍全 CONV

這意味著：要獲得 failure separation，你必須戰略轉向到新的主導物理軸，或先做離散敏感性驗證以確認“幾何極限是否真的被解析”。

---

## 7. Recommended Next Step

我建議下一步分成兩段，保持敘事與可審計性：

### 7.1 驗證軸（不進主線，用於可信度）

在固定 v1.4.6-012 的物理參數不變前提下，新增一個更密網格（Ultra）作為「離散敏感性檢查」。
目的不是擊穿，而是回答：SlotW=1.0nm 是否真的被解析到足以形成物理尖峰。

### 7.2 新主線物理軸（仍單變量）

在通過/完成 7.1 後，再選擇一條新的主導軸：

* Bias ceiling 軸（提升 RelayBias/BiasMax 以探測更高偏壓域）
  或
* 幾何軸（SlotW < 1.0nm，但最好配合 Ultra 网格验证，否则容易落入“离散解析极限”解释）
  或
* 掺杂对比轴（N_high / N_low 比例更极端，单变量推进）

---

## 8. Compliance

* 測試/報告/對外文字：禁止任何 emoji 或符號表情
* 程式輸出只使用純文字 SUCCESS / WARNING / FAILED
* 正式測試一律乾淨進程執行

---


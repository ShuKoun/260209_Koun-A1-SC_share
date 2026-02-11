# Test09 Report

## BenchS_StressHarness_v1.4.6-009

### Physics Hardening B2-Next (C4: Q_trap = 2.0e19)

---

## 1. Objective

本輪 Test09 延續單變量物理加壓策略，在：

* SlotW = 1.0nm（幾何極限）
* RelayBias = 8.0V（偏壓極限）
* Alpha = 0.15
* Solver / Harness / Data schema 全部鎖定不變

的前提下，進一步提升 C4 的空間電荷密度：

> Q_trap: 1.0e19 → 2.0e19

目標：

1. 試探 Baseline Newton + GMRES 的病態邊界
2. 觀察是否首次出現 NUMERIC_FAIL
3. 或出現 failure separation（Baseline 失效但 A1 延遲）
4. 若仍全 CONV，檢測是否出現病態前兆

---

## 2. Experimental Configuration

### 2.1 Invariants（嚴格鎖定）

* Solver parameters：不變
* Harness 邏輯（含 Early Relay）：不變
* Data schema（baseline/relay 上下文顯式填充、delta 統一公式）：不變
* SlotW_nm = 1.0
* RelayBias = 8.0
* Alpha = 0.15

### 2.2 單變量變更（B2-Next）

僅對 C4：

```
Q_trap = 2.0e19
```

其餘 Case（C2 / C3）作為對照組。

---

## 3. Test Matrix

| 維度              | 設定                           |
| --------------- | ---------------------------- |
| Grid            | Std (120×60), Dense (180×90) |
| Baseline step   | 0.2V, 0.4V                   |
| Baseline sweep  | 0 → relay snapped bias       |
| C4 relay target | 8.0V                         |
| A1 sprint span  | 0.5V                         |
| C4 A1 step      | 0.05V                        |

---

## 4. Global Outcome

### 4.1 統計總覽（以 FullLog 為準）

* 全部條目：CONV
* NUMERIC_FAIL = 0
* BUDGET_TIMEOUT = 0
* 未出現 failure separation

### 4.2 關鍵硬工況：C4 @ 8.0V

#### Baseline

| Grid  | base_step | iters | step_time(s) | t_lin(s) |
| ----- | --------- | ----- | ------------ | -------- |
| Std   | 0.2       | 3     | ~1.04        | ≈全部      |
| Std   | 0.4       | 3     | ~0.78        | ≈全部      |
| Dense | 0.2       | 3     | ~0.88        | ≈全部      |
| Dense | 0.4       | 4     | ~1.41        | ≈全部      |

觀察：

* t_lin 佔比接近 100%
* 迭代數仍低（3~4）
* 未出現 GMRES_FAIL
* 未出現 LS_FAIL

#### A1 @ 8.5V

| Grid  | outer iters | status      | dt_end |
| ----- | ----------- | ----------- | ------ |
| Std   | 3           | CONV(3/0/0) | 10.0   |
| Dense | 3           | CONV(3/0/0) | 10.0   |

觀察：

* dt 快速拉到 dt_max
* 未出現 dt collapse
* 無 NOISE 主導階段
* 未進入治理極限狀態

---

## 5. Interpretation

### 5.1 物理加壓效果

B2-Next 的結果表明：

> 在 SlotW=1.0nm + RelayBias=8.0V 鎖定條件下
> 將 Q_trap 提升至 2.0e19 仍未使 Jacobian 病態化。

Baseline 線性求解主導比例極高，顯示系統剛性顯著提升，但尚未跨越數值穩定邊界。

### 5.2 病態前兆評估

目前尚未觀察到：

* GMRES iteration 爆增
* info > 0
* LS 失敗
* MAX_ITER
* A1 dt 系統性收縮

因此：

> Test09 仍屬於「逼近區」，尚未進入「崩潰區」。

---

## 6. 與 Test07 / Test08 的連續性

| Test             | 變量              | 結果     |
| ---------------- | --------------- | ------ |
| Test07 (B1)      | SlotW → 1.0nm   | 全 CONV |
| Test08 (B2)      | Q_trap → 1.0e19 | 全 CONV |
| Test09 (B2-Next) | Q_trap → 2.0e19 | 全 CONV |

這形成了一條乾淨的單變量硬化鏈：

> 幾何壓縮 → Trap 加壓 → Trap 再加壓
> 數值方法仍保持穩定

這對論文敘事非常有利。

---

## 7. Strategic Positioning

當前階段屬於：

> Sustained Limit Exploration Phase
> （持續逼近數值邊界階段）

尚未出現 failure separation，但證據鏈正在變得更強。

---

## 8. Recommended Next Step

### Option A（保持單軸）

```
v1.4.6-010
Q_trap = 3.0e19
```

優點：

* 單變量鏈條完整
* 因果可解釋性最強
* 可形成 monotonic hardening 圖

### Option B（開第二物理軸）

```
Alpha: 0.15 → 0.05
```

優點：

* 更可能誘發邊界不穩定
* 但複雜度提升

---

## 9. Compliance Note

本測試遵循以下規範：

* 無 emoji
* 純文字輸出 SUCCESS / WARNING / FAILED
* 乾淨進程執行
* Solver / Harness / Data schema 全鎖定

---




---

# BenchS 壓力測試報告

## Test-35（v1.4.6-036）

### A1 梯度下降保底機制驗證

---

## 一、測試目的

在 v1.4.6-035 中，我們已確認：

* 成功構造強正回饋 Jacobian（`diag_tanh_max` 可達 ~3.2e4）
* tanh 未飽和（`tanh_arg_abs_max` < 3）
* Newton 仍能收斂
* A1 在強非單調區域出現 `DT_COLLAPSE`

因此 v1.4.6-036 的核心目標為：

> 為 A1 引入「梯度下降保底方向」，避免 GMRES 壞方向導致直接 collapse。

---

## 二、測試配置

* Grid: 640×320（MegaUltra2）
* Tanh Amp: 1e22
* Adaptive Offset: ≈ 0.4165V
* Width 掃描：1.0 / 0.1 / 0.05
* Baseline_Diag max_iter=200
* A1_BOOT max_outer_iter=200
* Gradient fallback:

  * d = −∇(0.5‖F‖²)
  * alpha_grad = 1e−4（單次嘗試）
  * 若 merit_new < merit 則接受

---

## 三、Newton 結果

| Width | Newton Iter | Norm Final | diag_tanh_max | diag_J_max |
| ----- | ----------- | ---------- | ------------- | ---------- |
| 1.00  | 99          | 8.9e-5     | ~1.6e3        | ~+159      |
| 0.10  | 42          | 7.7e-5     | ~1.6e4        | ~+1.46e4   |
| 0.05  | 40          | 7.8e-5     | ~3.2e4        | ~+3.06e4   |

觀察：

* 導數強度 ∝ 1/w 正常放大
* Jacobian 對角顯著正化
* tanh 未飽和
* Newton 仍然穩定收斂
* width 越小，Newton 反而更快

### 結論 1

> 目前仍未構造出 Newton Kill Zone。

---

## 四、A1 結果

| Width | Status      | Iter | Norm Final | dt_after | a1_step_grad |
| ----- | ----------- | ---- | ---------- | -------- | ------------ |
| 1.00  | DT_COLLAPSE | 27   | 1.2e3      | 1.47e-7  | 0            |
| 0.10  | DT_COLLAPSE | 20   | 8.5e3      | 5.50e-10 | 0            |
| 0.05  | DT_COLLAPSE | 18   | 1.29e4     | 3.19e-10 | 0            |

關鍵觀察：

* `a1_step_grad = 0`（梯度保底從未成功）
* `last_gmres_info = 0`（線性求解未崩）
* `force_step = 0`
* collapse 來自 dt 縮至閾值

### 結論 2

> A1 仍然在非單調區域中失敗，而且梯度保底機制完全未發揮作用。

---

## 五、梯度保底為何 0 次成功？

從 FullLog 可得：

* grad_norm ≈ 1.7e8（巨大）
* 固定 alpha_grad = 1e-4
* step_grad 在 clip 後接近 0.5

因此實際發生的是：

* 梯度方向尺度過大
* 單步嘗試直接導致 merit 上升
* 沒有 backtracking
* 一次失敗即放棄

換言之：

> 梯度方向並未經過尺度控制，因此“理論上存在的下降方向”在數值上沒有被有效搜索。

---

## 六、本次測試的真正價值

v1.4.6-036 並非失敗，而是揭示了更深層結構：

1. Jacobian 已顯著非單調（diag_J_max 可達 3e4）
2. Newton 仍可在此場景下生存
3. A1 的 collapse 不是因為“物理太強”
4. 而是因為“方向策略在強非單調區域失效”

這代表：

> A1 的脆弱性目前來自 solver 策略，而不是方程不可解。

---

## 七、戰略結論

### 1️⃣ 我們尚未進入真正的 Kill Zone

因為：

* Newton 仍收斂
* Jacobian 雖正化，但未導致分岔或不可下降區

### 2️⃣ A1 仍未具備非單調區韌性

梯度 fallback 機制存在，但尺度控制不足。

---

## 八、下一步建議（v1.4.6-037 方向）

### 必改項 1：Gradient fallback 必須加入 backtracking

建議：

* 方向歸一化：
  d = −g / max(|g|)
* alpha 序列：
  {1e-2, 5e-3, 2.5e-3, …}
* 最多 6-8 次嘗試
* 以 norm_new < norm 為接受標準

這將使 fallback 真正成為可用機制。

---

### 必改項 2：當 dt 接近 collapse 閾值時強制 fallback

避免：

dt *= 0.2 → 直到 1e-9 → DT_COLLAPSE

應改為：

dt < 某閾值 → 強制 gradient fallback 並 reset dt

---

### 可選項：增加 residual-direction fallback

在梯度也失敗時，嘗試：

d = −res

---

## 九、總結

Test-35（v1.4.6-036）結論：

* 結構壓力場已成功構造
* Jacobian 已顯著非單調
* Newton 仍穩定
* A1 collapse 來自方向策略
* 梯度保底設計方向正確，但數值實現需升級

這不是停滯，而是：

> 你已經把問題壓縮到“純 solver 韌性”層面。

下一步不需再調物理，只需把 fallback 做對。

---



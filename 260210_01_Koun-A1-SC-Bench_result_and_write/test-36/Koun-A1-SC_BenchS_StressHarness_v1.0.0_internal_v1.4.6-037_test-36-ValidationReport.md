

---

# BenchS 壓力測試報告

## Test-35（v1.4.6-037）

### A1 強化版梯度保底機制驗證

---

## 一、測試目標

在 v036 中：

* A1 於強非單調區域出現 `DT_COLLAPSE`
* 初始梯度 fallback 因尺度失控失效

v037 的核心改動：

1. 梯度方向歸一化
2. 多級 backtracking（1e-2 → 1e-6）
3. residual fallback（第二保底）
4. dt 危險閾值強制觸發 fallback

目標：

> 驗證 A1 在強非單調 Jacobian 場中是否具備生存韌性與下降能力。

---

## 二、物理壓力環境

* Grid：640×320（MegaUltra2）
* Tanh Amp：1e22
* Adaptive Offset ≈ 0.4165V
* Width 掃描：1.0 / 0.1 / 0.05
* Baseline_Diag max_iter=200
* A1_BOOT max_outer_iter=200

---

## 三、Newton（Baseline_Diag）結果

| Width | Iter | Norm Final | diag_tanh_max | diag_J_max |
| ----- | ---- | ---------- | ------------- | ---------- |
| 1.00  | 99   | 8.9e-5     | ~1.6e3        | ~+1.6e2    |
| 0.10  | 42   | 7.7e-5     | ~1.6e4        | ~+1.46e4   |
| 0.05  | 40   | 7.8e-5     | ~3.2e4        | ~+3.06e4   |

結論：

* Jacobian 對角顯著正化
* 非單調區域已構造成功
* Newton 仍穩定收斂
* width 越小 Newton 越快

### 判定

> 目前仍未構造 Newton Kill Zone。

---

## 四、A1（Enhanced Fallback）結果

### 4.1 生存性

v036：DT_COLLAPSE
v037：MAX_ITER

A1 不再自殺，成功活到 outer iter 上限。

這是 **質變級進展**。

---

### 4.2 殘差結果

| Width | Final Norm |
| ----- | ---------- |
| 1.00  | ~6.4e2     |
| 0.10  | ~1.12e3    |
| 0.05  | ~2.55e3    |

與 Newton 的 1e-4 相比仍差三到四個數量級。

---

### 4.3 Fallback 使用統計

| Width | a1_step | a1_step_grad | a1_step_res |
| ----- | ------- | ------------ | ----------- |
| 1.00  | 197     | 3            | 0           |
| 0.10  | 152     | 48           | 0           |
| 0.05  | 18      | 182          | 0           |

關鍵觀察：

* w 越小 → GMRES 主線越失效
* A1 幾乎完全依賴 gradient fallback 生存
* residual fallback 完全未生效

---

## 五、結構性解讀

### 1️⃣ 物理層面

* diag_tanh_max 可達 3e4
* diag_J_max 為強正
* tanh 未飽和
* 結構性壓力場已成功建立

### 2️⃣ Newton

* 仍然在 basin 內
* 屬於可解強非線性區
* 未出現不可下降區

### 3️⃣ A1

* 生存性已修復
* 主線方向在非單調區不穩定
* fallback 成為主要驅動器
* 下降速率慢
* 未進入局部收斂區

---

## 六、v037 的真正結論

v037 成功解決的是：

> A1 在強非單調區的「生存問題」

但尚未解決：

> A1 在該區的「有效收斂問題」

目前 A1 的角色：

* 不是求解器
* 而是“全局保命下降器”

---

## 七、問題是否嚴重？

不是。

這不是數學失敗，而是 solver 策略定位問題。

你目前證明了：

1. 非單調 Jacobian 已構造成功
2. Newton 仍存在可行 basin
3. A1 可以在該場景存活
4. A1 尚未具備高效進入局部收斂區的能力

這代表：

> 你已把問題壓縮到「solver 結構設計」層面，而不是物理不可解。

---

## 八、下一步建議（v038 方向）

### 1️⃣ 動態 offset 對齊 vacuum 區

不要固定 tanh_off = phi_bc_L。
改為：

每 N 次 outer iter 取 vacuum 區 median(phi) 作為 offset。

目的：

* 讓導數峰值始終對準 A1 當前工作點
* 防止 A1 被推入饱和區

---

### 2️⃣ 混合求解模式

當：

norm < 某閾值（例如 1e2）

切換到：

* 強化 GMRES（更嚴格 tol）
* 或直接切 Baseline_Diag

這會讓：

A1 → 負責全局生存
Newton → 負責局部快速收斂

---

### 3️⃣ 強化 residual fallback

目前 residual fallback=0 次成功。

建議：

* 進入 residual fallback 前強制重算當前 residual
* 回溯層數提升到 8–10
* alpha 初值可改為 1e-1

---

### 4️⃣ 加入殘差曲線輸出

每 10 次 outer iter 記錄：

* norm
* dt
* step 類型

否則 MAX_ITER 無法區分是 plateau 還是慢速下降。

---

## 九、階段總結

| 階段       | 狀態          |
| -------- | ----------- |
| SC-09    | Newton 未死亡  |
| SC-10 初期 | A1 自殺       |
| v036     | A1 可存活      |
| v037     | A1 穩定存活但收斂慢 |

你現在站在：

> 「A1 生存性已完成，進入收斂性優化階段」

---


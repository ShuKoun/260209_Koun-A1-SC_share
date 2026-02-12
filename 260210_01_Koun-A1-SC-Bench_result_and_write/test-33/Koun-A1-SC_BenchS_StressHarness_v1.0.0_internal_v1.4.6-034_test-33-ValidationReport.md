

---

# BenchS v1.4.6-034 技術報告

## Narrow Tanh Width × Relaxed A1 Acceptance

---

## 1. 實驗目標

v034 的設計基於 v033 的觀察：

* 在 $Q_{\text{amp}} = 10^{22}$ 時已出現
  $$
  \max(J_{\text{diag}}) > 0
  $$
* 但 Newton 仍然收斂。
* A1 在非單調區域提前崩潰。

因此 v034 嘗試：

1. **縮窄 tanh 寬度 $w$**
   $$
   w \in {1.0,;0.1,;0.05}
   $$
   期望導數放大：
   $$
   \max\left(\frac{\partial \rho}{\partial \phi}\right)
   \propto \frac{1}{w}
   $$

2. **放寬 A1 force-step 判定**

   * 原條件：merit 降低
   * 新條件：$|F|$ 降低即可

3. **Baseline_Diag 提升至 200 iter**

   * 防止「慢」被誤判為「死」

---

## 2. Jacobian 結構演化

### 2.1 w = 1.0

* $\max(d_{\text{tanh}}) \approx 1597$
* $\max(J_{\text{diag}}) \approx +154$

成功推入非單調區域。

---

### 2.2 w = 0.1 / 0.05

* $\max(d_{\text{tanh}}) = 0$
* $\max(J_{\text{diag}}) \approx -16885$

導數項完全消失。

---

## 3. 關鍵物理原因分析

### 3.1 tanh 導數形式

$$
\frac{\partial \rho_{\text{dyn}}}{\partial \phi}
================================================

\frac{q Q_{\text{amp}}}{w}
\left(1 - \tanh^2!\left(\frac{\phi - \phi_0}{w}\right)\right)
$$

若：

$$
\left|\frac{\phi - \phi_0}{w}\right| \gg 1
$$

則：

$$
1 - \tanh^2(\cdot) \rightarrow 0
$$

導數完全消失。

---

### 3.2 test-34 中實際發生的情況

在 w=0.1 / 0.05 時：

* $\phi_{\text{full,min}} \approx 0.416$
* $\phi_0 = 0$

因此：

$$
\frac{\phi - \phi_0}{w}
\approx \frac{0.4}{0.1}
= 4
$$

甚至更大。

此時：

$$
\tanh(4) \approx 0.9993
$$

導數：

$$
1 - \tanh^2(4) \approx 0
$$

結論：

> tanh 已飽和，正導數項完全消失。

---

## 4. Newton 行為

| Case | w  | Iter      | 狀態 |
| ---- | -- | --------- | -- |
| 1.0  | 35 | CONVERGED |    |
| 0.1  | 33 | CONVERGED |    |
| 0.05 | 30 | CONVERGED |    |

即使在 w=1（非單調區），Newton 仍收斂。

在 w=0.1 / 0.05：

* 系統重新回到強負定區
* Jacobian 被 free carrier 導數主導
* Newton 變得更穩定

---

## 5. A1 行為

| Case | w           | 結果        | 說明 |
| ---- | ----------- | --------- | -- |
| 1.0  | DT_COLLAPSE | 非單調區方向性失敗 |    |
| 0.1  | MAX_ITER    | 穩定但停滯     |    |
| 0.05 | MAX_ITER    | 穩定但停滯     |    |

### 5.1 A1 放寬後的效果

* w=1 時仍然崩潰
* w=0.1/0.05 時不再 collapse
* 但殘差停在 $|F|\approx 2.8$

說明：

> A1 的放寬判定提高了穩定性，但未改善收斂能力。

---

## 6. 核心發現

### 發現 1

縮窄 w 並未增強破壞。

### 發現 2

真正控制導數項是否生效的不是 w，而是：

$$
\phi - \phi_0
$$

即 offset 是否對齊實際場分佈。

### 發現 3

當 offset 不匹配時：

* tanh 飽和
* 導數為 0
* 系統退回強單調

---

## 7. 結論

v034 的核心結論為：

> 非單調結構只在 w=1 且 offset 對齊時出現。
> 縮窄 w 若不調整 offset，反而會消除正導數項。

Newton 仍未死亡。

A1 在非單調區更脆弱。

---

## 8. 下一步建議

### 8.1 必須修改 offset 策略

建議：

* 將 $\phi_0$ 設為 vacuum 區 $\phi$ 的中位數
* 或設為 $\phi_{\text{full,min}}$

確保：

$$
\frac{\phi - \phi_0}{w} \approx O(1)
$$

導數峰值真正落在 slot 區。

---

### 8.2 建議新增遙測

新增：

$$
\max\left|\frac{\phi - \phi_0}{w}\right|
$$

用來判斷是否飽和。

---

### 8.3 A1 主線接受準則仍需放寬

目前僅 force-step 使用 norm 判定。

建議主線 line-search 也允許：

$$
|F_{\text{new}}| < |F|
$$

作為接受條件。

---

## 9. 總結

v034 的結果顯示：

* Jacobian 可被推入非單調區（在 w=1）
* 但 Newton 未被擊殺
* 縮窄 w 失敗的原因是 offset 錯位
* A1 韌性略有改善但未構成優勢

你目前處於：

> 結構設計已成功
> 參數對齊仍需優化
> Kill Zone 尚未形成

---


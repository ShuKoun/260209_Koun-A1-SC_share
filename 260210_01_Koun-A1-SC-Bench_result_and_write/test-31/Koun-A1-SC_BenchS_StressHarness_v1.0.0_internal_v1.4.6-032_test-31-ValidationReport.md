

---

# BenchS v1.4.6-032 技術報告

## Unconditional Autopsy × Zero-Bias Sweep

---

## 1. 測試目標

v1.4.6-032 的核心目標為：

1. 對每個 $Q_{\mathrm{trap}}$ 點 **無條件執行 Baseline_Diag（max_iter=100）於 0.0V**，作為真正的 Newton 生存判定。
2. 聚焦 0.0V 錨點，不再進行 0→12V 掃描。
3. 以對數級掃描測試：

$$
Q_{\mathrm{trap}} \in {10^{20},,3\times10^{20},,10^{21}}
$$

4. 記錄 Jacobian 對角結構（`diag_J_*`）與 A1 遙測（`a1_force_step` 等）。

---

## 2. 測試條件

* Grid: MegaUltra2 (640×320)
* Bias: 0.0V only
* Baseline_Diag: max_iter = 100
* A1_BOOT: max_outer_iter = 200
* Budget: 240s

---

## 3. Newton（Baseline_Diag）結果

### 3.1 收斂狀態

| Case     | $Q_{\mathrm{trap}}$ | Iter | $|F|_{\text{final}}$    |
| -------- | ------------------- | ---- | ----------------------- |
| C4_Q1e20 | $10^{20}$           | 33   | $\sim 9.7\times10^{-5}$ |
| C4_Q3e20 | $3\times10^{20}$    | 23   | $\sim 5.5\times10^{-5}$ |
| C4_Q1e21 | $10^{21}$           | 22   | $\sim 9.6\times10^{-5}$ |

### 3.2 關鍵觀察

1. 所有測試點均在 100 次內收斂。
2. 隨著 $Q_{\mathrm{trap}}$ 增大，Newton 所需迭代數 **反而減少**。
3. 不存在 Newton Kill Zone。

---

## 4. Jacobian 結構分析

對角項分解為：

$$
J_{\text{diag}} = J_{\text{Poisson}} + J_{\text{carrier}}
$$

### 4.1 Poisson 對角

$$
J_{\text{Poisson,min}} \approx -1.69\times10^{4}
$$

$$
J_{\text{Poisson,max}} \approx -1.44\times10^{3}
$$

基本保持穩定，主導對角結構。

### 4.2 Carrier 導數項

隨 $Q_{\mathrm{trap}}$ 增大：

* $10^{20}$: $\min \sim -2\times10^{-4}$
* $3\times10^{20}$: $\min \sim -1.5$
* $10^{21}$: $\min \sim -1.4\times10^{3}$

關鍵結論：

$$
\max(J_{\text{diag}}) \approx -1.44\times10^{3} < 0
$$

Jacobian 始終強負定。

---

## 5. 結構性結論

由於：

$$
\frac{\partial \rho_{\text{trap}}}{\partial \phi} = 0
$$

增加 $Q_{\mathrm{trap}}$ 僅改變源項幅度，**不改變 Jacobian 的符號結構**。

實驗顯示：

* $J$ 未逼近 0
* 未出現非單調區域
* 未擊穿對角 dominance

因此：

$$
\text{Kill Zone 未形成}
$$

---

## 6. A1 結果

### 6.1 收斂狀態

| Case             | $|F|_{\text{final}}$ | Outer Iter |
| ---------------- | -------------------- | ---------- |
| $10^{20}$        | $\sim 2.53$          | 200        |
| $3\times10^{20}$ | $\sim 3.37$          | 200        |
| $10^{21}$        | $\sim 2.83$          | 200        |

### 6.2 遙測觀察

* `a1_force_step = 0`
* `last_gmres_info = 0`
* 線性求解穩定
* 殘差從 $\sim 10^{4}$ 降至 $\mathcal{O}(1)$ 後停滯

### 6.3 結論

A1 的停滯不是：

* GMRES 失敗
* Budget 不足
* Force Step 未觸發

而是其更新機制在強單調區域中缺乏 Newton 的二階收斂特性。

---

## 7. 核心結論

1. 對數級掃描至 $Q_{\mathrm{trap}} = 10^{21}$ 仍未構造 Newton Kill Zone。
2. Jacobian 結構隨 $Q_{\mathrm{trap}}$ 增大反而更負定。
3. A1 在強單調區域內無法形成 separation。
4. 單純物理加固路線已被完整證偽。

---

## 8. 戰略轉向建議

### 8.1 停止源項幅度掃描

常數 $Q_{\mathrm{trap}}$ 無法改變：

$$
\frac{\partial F}{\partial \phi}
$$

因此不可能產生結構性死亡。

---

### 8.2 必須引入 φ 依賴 trap 模型

例如：

$$
\rho_{\text{trap}}(\phi) = q Q_0 \tanh!\left(\frac{\phi-\phi_0}{w}\right)
$$

則：

$$
\frac{\partial \rho_{\text{trap}}}{\partial \phi}
=================================================

\frac{qQ_0}{w},\operatorname{sech}^2!\left(\frac{\phi-\phi_0}{w}\right) > 0
$$

該正導數項可推動：

$$
\lambda_{\min}(J) \rightarrow 0
$$

甚至產生局部非單調區域。

---

### 8.3 下一階段目標（建議 v1.4.6-033）

1. 引入 φ 依賴 trap。

2. 在 0V 打印：

   * `diag_J_min/max`
   * `M_diag_min_before/after`
   * Gershgorin 下界估計

3. 尋找：

$$
\min(J_{\text{diag}}) \approx 0
$$

---

## 9. 最終判定

v1.4.6-032 已證明：

$$
Q_{\mathrm{trap}} \le 10^{21}
$$

在當前模型中不足以擊殺 Newton。

本階段的價值在於：

> 完整否定「物理加固構造 Kill Zone」假設。

下一階段必須進入結構性模型改造。

---


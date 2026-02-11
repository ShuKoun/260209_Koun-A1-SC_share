
---

# BenchS v1.4.6-033 技術報告

## Structural Non-Monotonicity via φ-Dependent Tanh Trap

---

## 1. 實驗目標

v1.4.6-033 的核心目標為：

1. 引入 **φ-dependent 動態陷阱電荷模型（Tanh 型）**：
   $$
   \rho_{\text{dyn}}(\phi)
   =======================

   q \cdot Q_{\text{amp}} \cdot \tanh!\left(\frac{\phi - \phi_0}{w}\right)
   $$
2. 其 Jacobian 對角導數為：
   $$
   \frac{\partial \rho_{\text{dyn}}}{\partial \phi}
   ================================================

   \frac{q Q_{\text{amp}}}{w}
   \left(1 - \tanh^2!\left(\frac{\phi - \phi_0}{w}\right)\right)
   $$
   該項為正，目標是抵消泊松負定對角項。
3. 掃描：
   $$
   Q_{\text{amp}} \in {10^{21}, 10^{22}, 5\times10^{22}}
   $$
4. 專注於 **0.0V 錨點測試**，執行：

   * Baseline_Diag（100 iter）
   * A1_BOOT（200 outer iter）

---

## 2. 結構分析：Jacobian 對角演化

泊松對角項量級約為：

$$
J_{\text{Poisson,max}} \approx -1.4\times10^{3}
$$

動態 tanh 導數項（實測最大值）：

| $Q_{\text{amp}}$ | $\max(d_{\text{tanh}})$ | $\max(J_{\text{diag}})$ |
| ---------------- | ----------------------- | ----------------------- |
| $10^{21}$        | $\sim 7.9\times10^{1}$  | $-1.36\times10^{3}$     |
| $10^{22}$        | $\sim 1.6\times10^{3}$  | **$+1.54\times10^{2}$** |
| $5\times10^{22}$ | $\sim 8.0\times10^{3}$  | **$+6.57\times10^{3}$** |

---

## 3. 關鍵突破

在：

$$
Q_{\text{amp}} \ge 10^{22}
$$

條件下：

$$
\max(J_{\text{diag}}) > 0
$$

這意味著：

* Jacobian 不再強負定
* 系統正式進入 **非單調區域**
* 結構性正反饋已實現

這是本階段的重大進展。

---

## 4. Newton（Baseline_Diag）行為

| Case             | Iter | $|F|_{\text{final}}$    | 狀態        |
| ---------------- | ---- | ----------------------- | --------- |
| $10^{21}$        | 22   | $\sim 8.4\times10^{-5}$ | CONVERGED |
| $10^{22}$        | 35   | $\sim 9.6\times10^{-5}$ | CONVERGED |
| $5\times10^{22}$ | 28   | $\sim 8.6\times10^{-5}$ | CONVERGED |

### 觀察

即使：

$$
\max(J_{\text{diag}}) > 0
$$

Newton 仍然在 100 iter 內收斂。

### 結論

非單調 ≠ Kill Zone。

目前尚未構造 Newton 致死區。

---

## 5. A1 行為

| Case             | 結果          | Outer Iter | $|F|_{\text{final}}$    |
| ---------------- | ----------- | ---------- | ----------------------- |
| $10^{21}$        | MAX_ITER    | 199        | $\sim 2.85$             |
| $10^{22}$        | DT_COLLAPSE | 14         | $\sim 1.95\times10^{4}$ |
| $5\times10^{22}$ | DT_COLLAPSE | 4          | $\sim 8.15\times10^{4}$ |

### 關鍵觀察

* `a1_force_step = 0`
* `last_gmres_info = 0`
* 失敗來源：方向不被 merit 接受（DT collapse）

### 解釋

在強正導數區域：

* A1 方向不再穩定下降
* dt 被快速壓縮至 $10^{-8}$ 級
* Force step 無法拯救

A1 比 Newton 更早崩潰。

---

## 6. 理論含義

### 6.1 你成功做到的事

你第一次實證：

$$
\text{Strong Monotone} \rightarrow \text{Non-Monotone}
$$

並且觀察到：

* Jacobian 對角跨零
* 系統出現正反饋區域

這證明你的結構性 toy model 設計是有效的。

---

### 6.2 為什麼 Newton 仍活？

Newton 失敗通常需要：

1. Jacobian 近奇異（特徵值接近 0）
2. 盆地分裂（多解）
3. 線搜無法找到下降方向

目前：

* 雖然對角跨零
* 但整體矩陣尚未近奇異
* 線搜仍可找到下降方向

因此 Newton 仍可生存。

---

## 7. 戰略結論

v033 成功證明：

$$
\text{結構已改變}
$$

但尚未達到：

$$
\text{Newton Kill Zone}
$$

同時發現：

A1 在非單調區域更脆弱。

---

## 8. 下一步建議（提交給 Gemini）

### 路線 A：收窄 Tanh 寬度（首選）

將：

$$
w = 1.0 \rightarrow 0.1 \text{ 或 } 0.05
$$

因為：

$$
\max\left(\frac{\partial \rho}{\partial \phi}\right)
\propto \frac{1}{w}
$$

可將正導數項放大 10–20 倍，更可能推動：

$$
\lambda_{\min}(J) \rightarrow 0
$$

---

### 路線 B：提高 $Q_{\text{amp}}$

嘗試：

$$
10^{23},; 3\times10^{23}
$$

但注意殘差爆炸與數值溢出。

---

### 路線 C：改造 A1 acceptance 機制

當前 A1 失敗來源：

* merit 判定過嚴
* Force step 條件過強

建議：

* 以 $|F|$ 下降為接受條件
* 記錄連續 fail 次數上限

---

## 9. 最終判定

v1.4.6-033 是一次成功的結構性實驗。

你已經：

* 將 Jacobian 推入非單調區域
* 證明結構改變可觀測
* 發現 A1 在此區域更易崩潰

但：

$$
\text{Newton 尚未死亡}
$$

Kill Zone 仍未構造完成。

---


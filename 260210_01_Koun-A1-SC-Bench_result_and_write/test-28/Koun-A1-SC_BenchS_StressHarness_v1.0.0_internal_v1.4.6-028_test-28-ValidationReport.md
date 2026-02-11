

---

# BenchS v1.4.6-028 技術報告

## A1 算子因子化 A/B 測試結果分析

### 測試版本

* Harness: v1.4.6-028 (Factorial A/B Test - Logic Fixed)
* Grid: MegaUltra2 ($640 \times 320$)
* 物理參數：C4（$SlotW=0.5\text{nm}$, $N_{high}=10^{17}$, $N_{low}=10^{13}$, $Q_{trap}=3\times 10^{19}$, $\alpha=0$）
* Anchor 測試點：$0.0\text{V}$

---

# 一、Baseline 結果回顧

### 1.1 Baseline（max_iter = 30）

在 $0.0\text{V}$：

* 狀態：`MAX_ITER`
* 殘差：$|F| \approx 3.65\times 10^{-4}$

未收斂，但已接近容許誤差 $10^{-4}$。

---

### 1.2 Baseline_Diag（max_iter = 100）

在 $0.0\text{V}$：

* 33 次迭代收斂
* 最終殘差：$|F| \approx 8.6\times 10^{-5}$

結論：

> 該 PDE 在 $0.0\text{V}$ 明確存在解，且 Newton basin 存在。
> Baseline 的失敗僅為預算限制，而非數學不可解。

---

# 二、A1 因子化 A/B 測試設計

在 Anchor ($0.0\text{V}$) 點測試四種符號組合：

矩陣形式：
$$
A = \frac{1}{\Delta t} I + s_m J
$$

右端項：
$$
\text{RHS} = s_r F
$$

其中：

* $s_m \in {-1, +1}$
* $s_r \in {-1, +1}$

四個變體：

| Variant        | $s_m$ | $s_r$ |
| -------------- | ----- | ----- |
| A1_Base        | -1    | +1    |
| A1_Flip_RHS    | -1    | -1    |
| A1_Flip_Matrix | +1    | +1    |
| A1_Flip_All    | +1    | -1    |

---

# 三、A/B 測試結果

在兩組 base_step（0.2V 與 0.4V）下結果完全一致：

---

## 3.1 A1_Base ($s_m=-1$, $s_r=+1$)

* 狀態：`MAX_ITER`
* 殘差明顯下降
* 但未達 $10^{-4}$

結論：

> 可運行，方向正確，但收斂速度遠慢於 Newton。

---

## 3.2 A1_Flip_RHS ($s_m=-1$, $s_r=-1$)

* 狀態：`DT_COLLAPSE`
* 幾乎 $k=0$ 即崩潰
* 殘差未下降

解釋：

若矩陣近似 $-J$，則解滿足：
$$
(-J)d = -F \Rightarrow Jd = F
$$

而 Newton 方向應滿足：
$$
Jd = -F
$$

因此方向完全相反。

結論：

> $s_r=-1$ 導致梯度方向錯誤，必然崩潰。

---

## 3.3 A1_Flip_Matrix ($s_m=+1$, $s_r=+1$)

* 狀態：`MAX_ITER`
* 殘差下降弱於 A1_Base

矩陣近似：
$$
A \approx +J
$$

對應：
$$
Jd \approx F
$$

仍非 Newton 方向。

結論：

> 方向雖未立即崩潰，但下降性顯著弱化。

---

## 3.4 A1_Flip_All ($s_m=+1$, $s_r=-1$)

* 狀態：`DT_COLLAPSE`
* 立即崩潰

矩陣近似：
$$
A \approx +J
$$
RHS：
$$
-F
$$

得到：
$$
Jd \approx -F
$$

形式上接近 Newton，但由於預條件與矩陣耦合翻轉，實際步驟方向失衡，立即崩潰。

---

# 四、核心結論

## 4.1 符號定位已完成

測試明確表明：

* RHS 必須為 $+F$
* 矩陣必須為 $\frac{1}{\Delta t}I - J$

即 A1_Base 的符號組合是唯一數學一致的形式。

v027 的“全翻轉修復”假設已被完全否定。

---

## 4.2 更重要的事實

在 $0.0\text{V}$：

* Baseline_Diag：33 步收斂
* A1_Base：50 外循環仍遠未收斂

因此：

> 在當前構造下，不存在 separation 敘述空間。
> Newton 收斂，A1 更慢。

這是非常關鍵的理論定位。

---

# 五、為何 A1 不佔優勢

A1 的核心系統為：

$$
\left(\frac{1}{\Delta t}I - J\right)d = F
$$

當 $\Delta t$ 足夠大時：

$$
\frac{1}{\Delta t}I \to 0
$$

則：
$$
-Jd \approx F \Rightarrow Jd \approx -F
$$

即接近 Newton。

但當 $\Delta t$ 很小時：

$$
\frac{1}{\Delta t}I \gg J
$$

系統變成強對角 dominance，更新步接近 gradient-like damping，收斂變慢。

目前場景並未構造出 Newton 失效區域，因此 A1 無優勢。

---

# 六、下一步建議（戰略級）

## 6.1 停止符號探索

符號空間已探索完畢，繼續嘗試屬於無效消耗。

---

## 6.2 兩條真正有價值的路線

### 路線 A：重構 A1 機制

將 A1 改為：

* Trust-region Newton
* Levenberg–Marquardt 型阻尼
* 自適應線性化步長

而非 pseudo-time 形式。

---

### 路線 B：構造真正 Newton 失效族

當前場景：

* 單調區
* 強 screening
* Jacobian 正定裕度存在

需要構造：

* 非單調區
* 局部 sign change
* Jacobian 奇異性
* 或多 basin 切換場景

只有在 Newton 真正失效時，A1 才可能形成 separation。

---

# 七、總結

test28 已完成以下任務：

1. 排除符號錯誤假設
2. 驗證 RHS 翻轉必然導致方向錯誤
3. 證明 A1 與 Newton 在該場景下方向一致但效率較差
4. 否定 v027 修復路線

當前理論定位：

> 在 $0.0\text{V}$ 構造下，Newton 可在 33 步內收斂，A1 無結構優勢。
> 若要證明 A1 優越性，必須改變問題族，而非調整符號。

---


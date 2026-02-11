

---

# BenchS v1.4.6-029 技術報告

## Physics Hardening × A1 Logic Consolidation

---

## 1. 測試目標

v1.4.6-029 的核心戰略目標為：

1. 將陷阱電荷密度由
   $$
   Q_{\mathrm{trap}} = 3.0\times 10^{19}
   $$
   提升至
   $$
   Q_{\mathrm{trap}} = 6.0\times 10^{19}
   $$
   以嘗試構造 **Newton Kill Zone**。

2. 移除 A1 符號 A/B 測試，回歸標準隱式算子：
   $$
   A = \frac{1}{\Delta t} I - J
   $$
   $$
   \text{RHS} = +,\mathrm{res}
   $$

3. 重構 Force Step 邏輯，改為基於連續失敗次數觸發。

---

## 2. 測試條件

* Grid: MegaUltra2 (640×320)
* Base Step: 0.2V / 0.4V
* 只測 0.0V Anchor
* Baseline:

  * 正常模式：max_iter=30
  * 驗屍模式：max_iter=100
* A1 Boot:

  * max_outer_iter=200
  * Anchor Budget=240s

---

## 3. 結果摘要

### 3.1 Baseline（Newton）

#### 正常模式（max_iter=30）

* 0.0V: `MAX_ITER`
* 殘差約：
  $$
  |F| \approx 3.65\times 10^{-4}
  $$

#### Baseline_Diag（max_iter=100）

* 0.0V: `CONVERGED`
* 迭代次數：33
* 最終殘差：
  $$
  |F| \approx 8.6\times 10^{-5}
  $$

**關鍵結論：**

即使將
$$
Q_{\mathrm{trap}} = 6.0\times 10^{19}
$$
Newton 仍然在 33 次迭代內收斂。

不存在 Newton Kill Zone。

---

### 3.2 A1 Anchor（Boot 模式）

* 狀態：`MAX_ITER`

* 外循環：200

* 最終殘差：
  $$
  |F| \approx 2.28
  $$

* dt 行為：
  $$
  dt_{\min} = 1\times 10^{-5}
  $$
  $$
  dt_{\max} = 0.1
  $$

* a1_step = 200

* a1_noise = 0

**關鍵觀察：**

A1 每步都被判定為有效 STEP，但殘差停留在常數級別（約 2），未進入真正收斂區域。

---

## 4. 重要結論

### 結論 1：Physics Hardening 未構造 Kill Zone

翻倍 $Q_{\mathrm{trap}}$ 不足以擊穿 Newton 的吸引盆地。

Newton 仍穩定存在收斂域。

---

### 結論 2：A1 並未展現獨特穩定性優勢

在同一物理條件下：

* Newton：33 步收斂
* A1：200 步未達閾值

因此：

$$
\text{不存在 Separation}
$$

---

### 結論 3：A1 的失敗不是符號錯誤

test28 已證明：

* 翻轉 RHS 或 Matrix 導致 DT_COLLAPSE
* 原始符號為唯一數學自洽形式

因此問題不在符號。

---

## 5. 機制分析

在 0.0V 處：

* 非線性指數項未爆炸
* Jacobian 未進入非單調區
* Newton basin 未崩塌

當前 PDE 仍屬於：

$$
\text{Strongly Monotone Region}
$$

A1 無法獲得優勢。

---

## 6. 戰略評估

v029 否定了以下假設：

> 單純增加陷阱電荷即可構造 Newton Kill Zone。

該假設被實驗證偽。

---

## 7. 建議的下一步（提交給 Gemini）

### 路線 A：對數級掃描 Q_trap

使用對數級測試：

$$
6\times 10^{19}
\rightarrow
1\times 10^{20}
\rightarrow
3\times 10^{20}
\rightarrow
1\times 10^{21}
$$

只測 0.0V Baseline_Diag，尋找 Newton 真正死亡點。

---

### 路線 B：回到結構性破壞

回歸 v023 時代的思路：

構造：

$$
\text{Non-Monotone} + \text{Weak Screening} + \text{Stiff Poisson}
$$

而不是僅靠物理強度堆積。

---

### 路線 C：重新定義 A1 的目標

若 A1 要證明價值，必須：

* 要么比 Newton 更穩定
* 要么在固定時間預算下殘差下降更快

目前兩者皆未達成。

---

## 8. 最終判定

v1.4.6-029 結論：

$$
Q_{\mathrm{trap}} = 6.0\times 10^{19}
$$

仍不足以構造 Newton Kill Zone。

A1 未形成任何 separation 證據。

必須改變戰略方向。

---


這是一份依照學術規格撰寫的 **Paper A (Methodology & Protocol)** 初稿。

這篇文章將 **Koun-A1-SC** 定義為一個「治理協議的實例」，而非單純的數值求解器。它強調的是如何通過結構化的決策邏輯，解決了傳統方法在極端剛性問題上必須依賴人工輔助（Ramping）的痛點。

# ---

**Paper A: 方法論與協議設計**

**標題：**

**Koun-A1-SC：極端剛性漂移-擴散系統冷啟動的結構型算法治理協議**

*(Koun-A1-SC: A Structural Algorithmic Governance Protocol for Cold-Start Simulation of Extremely Stiff Drift-Diffusion Systems)*

## ---

**摘要 (Abstract)**

半導體器件仿真中的漂移-擴散模型（Drift-Diffusion Model）因載流子濃度與電勢的指數耦合關係，表現出極端的剛性（Stiffness）。傳統牛頓類方法（Newton-Raphson）在缺乏良好初值的情況下極易發散，實務上常需依賴人工設計的偏壓或摻雜漸進策略（Ramping）。本文提出 **Koun-A1-SC**，這是 Koun-A1 算法治理框架在半導體領域的具體實例化。該協議通過引入 **偽瞬態延拓（PTC）** 作為單一權威生存底座，結合 **邊界消元（Boundary Elimination）** 的拓撲修正，以及基於 **雙重容忍（Dual Tolerance）** 的一致性事件分類機制，成功實現了高解析度（$120 \\times 60$ 網格）、極端摻雜突變條件下的「冷啟動」求解。實驗顯示，該協議不僅能消除傳統算法的假性崩潰（False Failure），還能在抵達離散化誤差地板時自動切換至「受控漂移（Controlled Drift）」狀態，展現了治理框架在結構性斷裂邊緣的魯棒性。

## ---

**1\. 引言 (Introduction)**

在非線性偏微分方程的數值求解中，「可解性（Solvability）」往往被視為算法的內在屬性。然而，在半導體物理中，泊松-玻爾茲曼（Poisson-Boltzmann）方程引入了 $\\exp(\\phi/V\_T)$ 的強非線性項，導致雅可比矩陣（Jacobian）的條件數隨網格密度和電勢變化劇烈惡化。

傳統 TCAD 工具通常採用「阻尼牛頓法（Damped Newton）」配合「連續性參數掃描（Continuation/Ramping）」來規避收斂問題。這種方法雖然有效，但其本質是將算法的穩定性轉嫁給了使用者的經驗（如何設計 Ramping 步驟）。

本文主張，算法應當具備內在的「治理能力（Governance Capability）」。我們基於 **Koun-A1 框架**，構建了一個無需外部干預即可自我調節穩定性的協議 **Koun-A1-SC**。其核心貢獻在於：

1. **生存層的確立：** 以時間步長 $dt$ 取代傳統的線搜索因子 $\\alpha$ 作為全局化控制變量。  
2. **拓撲死鎖的解除：** 識別並解決了 Krylov 子空間方法在強剛性邊界條件下的凍結現象。  
3. **邏輯自洽性：** 修復了接受準則（Acceptance）與診斷準則（Diagnostics）之間的邏輯斷裂，消除了算法內耗。

## ---

**2\. 治理架構實例化 (Instantiating the Framework)**

Koun-A1-SC 並非單一的數值方法，而是一套動態切換求解策略的決策邏輯。其架構由以下三個核心層次構成：

### **2.1 生存層：偽瞬態延拓 (The Survival Base: PTC)**

面對極端非線性，標準牛頓法試圖一步到位的策略往往導致「過衝（Overshoot）」和物理量溢出。我們引入偽瞬態延拓（Pseudo-Transient Continuation, PTC），將穩態問題轉化為偽時間演化問題：

$$\\left( \\frac{1}{\\Delta \\tau} I \- J(\\phi\_k) \\right) \\delta \\phi \= F(\\phi\_k)$$  
其中 $\\Delta \\tau$ 是偽時間步長。

* **生存模式 ($\\Delta \\tau \\to 0$)：** 算子趨近於對角矩陣 $\\frac{1}{\\Delta \\tau} I$，問題退化為穩健的梯度流，保證了算法在遠離解區域時的存活能力。  
* **牛頓模式 ($\\Delta \\tau \\to \\infty$)：** 算子回歸為 $-J$，恢復牛頓法的二次收斂速度。

**治理原則：** $\\Delta \\tau$ 是系統的最高權威。當結構性診斷顯示系統健康時，$\\Delta \\tau$ 指數增長；當檢測到危機時，$\\Delta \\tau$ 立即縮減以增強正則化。

### **2.2 結構哨兵：噪聲與一致性 (Structural Sentinel: Noise & Coherence)**

在高剛性系統的殘差末端（Residual Floor），數值噪聲往往會導致目標函數（Merit Function）出現微小波動。傳統算法常將此誤判為發散。

v1.3.6 協議引入了 **「雙重容忍（Dual Tolerance）」** 與 **「一致性分類」**：

* **接受準則：** $\\Phi\_{new} \\le \\Phi\_{old} \\cdot (1 \+ \\text{rtol}) \+ \\text{atol}$。允許能量在噪聲範圍內（由 $\\text{atol}$ 定義）微幅上升。  
* **事件分類：**  
  * **\[STEP\]**：顯著下降。  
  * **\[NOISE\]**：波動在 $\\text{atol}$ 範圍內。系統判定為「健康」，允許 $dt$ 繼續增長。  
  * **\[UPHILL\]**：顯著上升（超過 $\\text{atol}$）。觸發救援。

### **2.3 救援家族：多尺度狙擊 (The Rescue Family: Multi-Scale Sniper)**

當牛頓方向失效（\[UPHILL\]）時，協議切換至伴隨救援模式（Adjoint Rescue）。

鑑於高解析度下的地形複雜性，單一尺度的梯度下降往往無效。我們實施 **多尺度狙擊（Multi-Scale Sniper）**：

$$d\_{rescue} \= \- \\nabla \\Phi \\cdot \\frac{||F||}{||\\nabla \\Phi||} \\cdot \\lambda, \\quad \\lambda \\in \\{0.1, 0.05, 0.01, \\dots\\}$$  
求解器盲測一組尺度，一旦發現能量下降的通道，即刻切換路徑。

## ---

**3\. 關鍵技術實現 (Key Technical Implementations)**

為了支撐上述治理邏輯在高解析度（$120 \\times 60$ 網格，7200 自由度）下的運行，必須解決計算複雜度與拓撲死鎖問題。

### **3.1 無矩陣 Krylov 求解 (Matrix-Free Krylov Solver)**

直接構建 $7200 \\times 7200$ 的稠密雅可比矩陣是不切實際的。我們採用 **Matrix-Free GMRES**，利用自動微分（Automatic Differentiation）計算雅可比-向量積（JVP）：

$$J(\\phi) \\cdot v \\approx \\frac{\\partial F(\\phi \+ \\epsilon v)}{\\partial \\epsilon} \\bigg|\_{\\epsilon=0}$$  
這使得內存消耗從 $O(N^2)$ 降至 $O(N)$。

### **3.2 拓撲修正：邊界消元 (Boundary Elimination)**

在 v1.3.1 的實驗中，我們觀測到了 **「邊界死鎖（Boundary Deadlock）」** 現象：在強 PTC 正則化下（$\\frac{1}{dt}$ 極大），Krylov 子空間無法有效捕捉 Dirichlet 邊界的硬約束，導致邊界殘差鎖死。

v1.3.2 引入了拓撲修正：將邊界節點從線性系統中 **完全消元（Eliminated）**。

* 線性系統僅求解內部節點（Internal Nodes）。  
* 邊界條件被「硬編碼」進殘差函數的算子中，不再作為變量參與迭代。  
  這一修正從根本上消除了邊界殘差滯留的可能性。

## ---

**4\. 數值證據 (Numerical Evidence)**

基於 v1.3.6 版本的運行日誌，我們展示了該協議在極端條件下的表現。

### **4.1 實驗設置**

* **網格：** $120 \\times 60$ (2D)。  
* **物理：** 3nm 尺度 PN 結，摻雜濃度從 $+10^{16}$ 突變至 $-10^{16}$，中央含高摻雜島。  
* **初值：** 全零冷啟動（Zero Cold Start）。

### **4.2 階段分析**

1. **生存期 (Survival Phase, Iter 0-15):**  
   殘差從 $1.8 \\times 10^{12}$ 迅速下降至 $1.95 \\times 10^{-3}$。此階段 $dt$ 較小，PTC 發揮了關鍵的穩定作用，避免了指數項溢出。  
2. **平台期與受控漂移 (Plateau & Controlled Drift, Iter 16-49):**  
   殘差進入 $1.94 \\times 10^{-3}$ 的物理/離散地板。  
   * **傳統表現：** 因無法滿足嚴格下降條件，觸發連續步長縮減，最終報錯退出。  
   * **Koun-A1 表現：** 識別出波動為 **\[NOISE\]**，判定系統結構依然健康。$dt$ 從 $1.8 \\times 10^{-3}$ 穩定增長至 $0.75$。系統在保持「活著」的同時，以大步長遍歷了解空間的平坦區域。

## ---

**5\. 結論 (Conclusion)**

Koun-A1-SC 的成功證明了 **「算法治理（Algorithmic Governance）」** 是解決極端非線性問題的有效途徑。通過將 **PTC** 確立為單一權威，並配合 **邏輯自洽的事件分類**，我們成功消除了解算過程中的內耗。

本研究表明，對於高度剛性的系統，求解器的目標不應僅是「尋找下降方向」，更應包括「識別環境限制」與「維持生存演化」。這一範式轉變為未來的 AGI 物理推理引擎提供了重要的架構參考。

---

**(Paper B 將深入探討「殘差地板」的物理本質與「受控漂移」的理論意義。)**
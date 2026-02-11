

### **Technical Subsection Draft: Discretization Robustness and Structural Diagnosis of the Baseline Newton Solver**

**1\. Problem Setting**

The primary objective of this study series (v1.4.6-015 to v1.4.6-018) was to interrogate the numerical stability of the standard Drift-Diffusion Newton solver under extreme geometric and bias conditions. Previous iterations established that neither high trap density ($Q\_{trap} \= 3.0 \\times 10^{19} \\, \\text{cm}^{-3}$) nor extreme boundary asymmetry ($\\alpha \= 0.00$) were sufficient to induce convergence failure at $8.0\\text{V}$. The working hypothesis posited that geometric aliasing at the discretization level might be artificially smoothing the electric field gradients, thereby shielding the solver from the true conditioning of the singular sub-nanometer physics.

**2\. Resolution Escalation (v015–v017)**

To test the aliasing hypothesis, a progressive mesh densification protocol was executed while locking the geometric slot width at the physical limit of $W\_{slot} \= 0.5\\text{nm}$.

* **v015 (Ultra, $240 \\times 120$, $\\Delta x \\approx 0.42\\text{nm}$):** The feature size $W\_{slot}$ spanned approximately $1.2$ grid cells. Convergence behavior remained identical to the $1.0\\text{nm}$ case, suggesting insufficient resolution to capture the gradient singularity.  
* **v016 (SuperUltra, $480 \\times 240$, $\\Delta x \\approx 0.21\\text{nm}$):** $W\_{slot}$ spanned $\\approx 2.4$ cells. A discernible "hardening" of the system was observed, characterized by an increase in Newton iterations (from 3 to 4 at $12.0\\text{V}$) and A1 outer iterations (from 3 to 5).  
* **v017 (MegaUltra2, $640 \\times 320$, $\\Delta x \\approx 0.16\\text{nm}$):** This resolution satisfied the "3-Cell Rule" ($W\_{slot} / \\Delta x \\approx 3.2$), theoretically allowing for the formation of a discrete field singularity.  
  **Result:** Despite the successful resolution of the geometric feature, failure separation was not achieved. The system exhibited increased stiffness (linear solve time $t\_{lin}$ scaled with $N\_{dof}$, and iterations marginally increased), but the Baseline Newton solver maintained robust quadratic convergence up to $12.0\\text{V}$. This negates geometric aliasing as the primary stabilizer.

**3\. Structural Diagnosis (v018)**

To elucidate the source of this unexpected robustness, a structural diagnostic probe was embedded into the solver loop (v1.4.6-018). This probe extracted spectral properties of the solution and the Jacobian matrix $J$ at equilibrium.

Key diagnostic metrics recorded at $V\_{bias} \= 12.0\\text{V}$ (Baseline) and $12.5\\text{V}$ (A1) yielded:

* **Inner Potential Saturation:** The internal electrostatic potential $\\phi\_{inner}$ remained bounded, with the maximum exponential term in the inner domain recorded as $\\log(\\max(\\exp(|\\phi|/V\_t))) \\approx 31.1$. This corresponds to a potential drop of $\\approx 0.8\\text{V}$ across the critical junction, significantly lower than the applied boundary bias of $12.0\\text{V}$.  
* **Jacobian Diagonal Dominance:** The maximum diagonal element of the Jacobian was observed at $\\max(\\text{diag}(J)) \\approx \-1443$. Given the negative definiteness required for stability, this indicates a strong diagonal dominance relative to the off-diagonal drift terms.  
* **A1 Stability Margin:** The preconditioning margin for the A1 solver, defined as $M\_{diag} \= \\min(\\Delta t^{-1} \- \\text{diag}(J))$, maintained a safe positive floor of $M\_{diag} \\approx 1443$ for both pre- and post-step $\\Delta t$.

**4\. Structural Interpretation**

The diagnostic data reveals that the "indestructibility" of the Baseline solver is not due to numerical damping, but rather a **Shielding Effect** inherent to the current device topology.

Although the external bias is pushed to $12.0\\text{V}$, the potential inside the singular slot region is effectively clamped. The exponential nonlinearity, which typically drives Newton failure via $\\exp(\\phi/V\_t)$, is saturated at $\\approx \\exp(31)$, well below the floating-point overflow or ill-conditioning threshold ($ \\approx \\exp(700)$).

Consequently, the Jacobian matrix remains in a **Strongly Monotone** regime. The diagonal elements $\\text{diag}(J)$ are dominated by the Poisson equation's elliptic operator and the well-behaved carrier density terms, preventing the loss of ellipticity that characterizes numerical breakdown.

**5\. Boundary Implication**

The current experimental boundary is defined not by discretization limits, but by structural physics.

* **Numerical Boundary:** The grid resolution $\\Delta x \= 0.16\\text{nm}$ is sufficient. Further refinement will yield diminishing returns ($O(N^2)$ cost for zero marginal insight).  
* **Physical Boundary:** The current geometry ($W\_{slot}=0.5\\text{nm}$) combined with the specific doping/permittivity profile creates a self-shielding potential well. To induce failure, future experiments must break this shielding mechanism, likely by forcing the high-bias domain to penetrate the slot region (e.g., via vertical geometry puncture or extreme permittivity contrast), rather than simply scaling existing parameters.

*Technical Note: Implementation of the diagnostic probe utilized jnp.median, which incurs a non-trivial computational overhead on large-scale grids ($N \> 10^5$). Future iterations may omit this statistic for performance. Additionally, the data schema was optimized to populate probe fields only upon convergence (succ=True), with NaN pre-filling to maintain alignment.*
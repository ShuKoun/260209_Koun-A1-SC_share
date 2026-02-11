

# Benchmark S – Stress Harness v1.4.6-014

## Phase V: Discretization Sensitivity Validation (Ultra Grid)

**Parent Version:** v1.4.6-013
**Change Type:** Grid Density Extension (Validation Axis)
**Physical Parameters (C4) Locked From v013:**

* SlotW = 1.0 nm
* Q_trap = 3.0e19
* Alpha = 0.00
* BiasMax = 12.0 V
* RelayBias = 12.0 V

**New Axis Introduced:**

* Ultra grid: 240 × 120

Solver, harness logic, data schema, stepping strategy: unchanged.

---

# 1. Objective

Previous phases (B1–B4) exhausted physical stress axes:

* Geometry compression (1.0 nm)
* Trap density hardening (3e19)
* Boundary asymmetry reduction (Alpha → 0.00)
* Bias ceiling extension (12 V)

Despite extreme parameter combinations, Baseline remained fully convergent.

This suggested possible **Discretization Shielding**:

> The grid may be too coarse to resolve true gradient singularities.

v1.4.6-014 introduces a higher resolution grid (Ultra) to test whether increased spatial resolution reveals latent instability.

---

# 2. Execution Overview

## Grid Set

* Std: 120 × 60
* Dense: 180 × 90
* Ultra: 240 × 120

## Baseline Step Sizes

* 0.2 V
* 0.4 V

## Maximum Tested Bias

* Baseline: 12.0 V
* A1: 12.5 V (Sprint extension)

---

# 3. Global Results

From FullLog:

* Total rows: 363
* `fail_class = CONV`: 363
* `NUMERIC_FAIL`: 0
* `BUDGET_TIMEOUT`: 0
* `DT_COLLAPSE`: 0
* Failure separation: not observed

All solver executions completed successfully.

---

# 4. C4 (Extreme Case) Detailed Analysis

## 4.1 Baseline @ 12.0 V

| Grid  | Step | Iterations | Time (s) | t_lin (s) |
| ----- | ---- | ---------- | -------- | --------- |
| Std   | 0.2  | 3          | ~0.82    | ~0.82     |
| Std   | 0.4  | 3          | ~0.81    | ~0.81     |
| Dense | 0.2  | 3          | ~0.93    | ~0.93     |
| Dense | 0.4  | 3          | ~0.89    | ~0.89     |
| Ultra | 0.2  | 3          | ~0.82    | ~0.81     |
| Ultra | 0.4  | 4          | ~1.22    | ~1.21     |

Observations:

* Iteration count remains extremely low.
* Ultra grid does not trigger instability.
* Slight iteration increase (Ultra / 0.4) is marginal and not progressive.
* No GMRES degradation.
* No line search failure.
* No iteration explosion.

---

## 4.2 A1 @ 12.5 V

All configurations returned:

```
CONV(3/0/0)
dt_end = 10.0
```

Interpretation:

* No dt shrink trend.
* No dynamic collapse.
* No early instability signature.
* A1 remains fully relaxed even under Ultra resolution.

---

# 5. Interpretation

## 5.1 Discretization Shielding Hypothesis

The Ultra grid was expected to:

* Resolve sharper electric field gradients.
* Expose Jacobian ill-conditioning.
* Increase GMRES difficulty.
* Reveal nonlinear amplification.

Observed outcome:

* No instability.
* No meaningful stress amplification.
* Convergence remains efficient.

Conclusion:

> There is no strong evidence of discretization shielding at 240 × 120 resolution.

---

## 5.2 Implication

The Baseline solver’s stability is not an artifact of:

* Coarse mesh smoothing.
* Under-resolved slot geometry.
* Numerical gradient averaging.

The stability appears structural within the modeled regime.

---

# 6. Comparative Stability Chain

| Phase   | Axis Tested         | Result |
| ------- | ------------------- | ------ |
| B1      | Geometry (1.0 nm)   | Stable |
| B2      | Trap density (3e19) | Stable |
| B3      | Alpha → 0.00        | Stable |
| B4      | Bias → 12 V         | Stable |
| Phase V | Ultra grid          | Stable |

Five independent axes tested.
Zero failure events.

---

# 7. Technical Significance

The solver demonstrates:

* Robust Newton convergence.
* Stable GMRES linear solve behavior.
* No exponential blow-up under 12 V.
* No dt collapse in A1.
* No separation between Baseline and A1.

This strengthens the claim that the system lies within a numerically well-conditioned domain under current physical modeling assumptions.

---

# 8. Strategic Implications

With discretization shielding largely ruled out at Ultra resolution, remaining plausible instability triggers must involve:

1. Sub-nanometer geometry (<1.0 nm)
2. Stronger multi-parameter coupling
3. Model structure modification (physics extension)
4. Non-Poisson nonlinearities
5. Boundary condition reconfiguration

Further pure bias escalation alone is unlikely to produce failure.

---

# 9. Conclusion

v1.4.6-014 successfully completes the discretization sensitivity validation phase.

Ultra resolution does not expose instability.
The Baseline solver remains fully convergent under extreme physical stress.

The system demonstrates structural numerical stability across geometry, trap density, asymmetry, bias domain, and grid density axes.

---


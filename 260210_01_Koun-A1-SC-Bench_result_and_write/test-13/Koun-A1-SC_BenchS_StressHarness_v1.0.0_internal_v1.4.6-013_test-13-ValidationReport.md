

# Benchmark S – Stress Harness v1.4.6-013

## Bias Ceiling Extension (B4: 12V Limit)

**Parent Version:** v1.4.6-012
**Change Type:** Single-Variable Bias Domain Extension
**Modified Parameter (C4 only):**

* `BiasMax: 8.0 → 12.0`
* `RelayBias: 8.0 → 12.0`

All other parameters locked:

* `SlotW = 1.0 nm`
* `Q_trap = 3.0e19`
* `Alpha = 0.00`
* Solver parameters unchanged
* Harness logic unchanged
* Data schema unchanged
* Grid configurations unchanged

---

# 1. Test Objective

After exhausting the following physical axes without inducing instability:

* Trap density hardening (up to 3.0e19)
* Boundary asymmetry reduction (Alpha → 0.00)
* Geometry compression (SlotW → 1.0nm)

We hypothesized that failure to induce instability might be due to remaining within a limited bias domain (≤8V).

Therefore, v1.4.6-013 extends the bias ceiling to **12V**, probing exponential growth regions of the Poisson-Boltzmann nonlinearity.

---

# 2. Execution Summary

## Grid Configurations

* Std: 120 × 60
* Dense: 180 × 90

## Baseline Step Sizes

* 0.2V
* 0.4V

## Maximum Tested Points

* Baseline C4: 12.0V
* A1 C4: 12.5V (0.5V sprint extension)

---

# 3. Result Summary

## Global Outcome

* Total FullLog entries: 242
* `fail_class = CONV`: 242
* `NUMERIC_FAIL`: 0
* `BUDGET_TIMEOUT`: 0
* Failure separation: Not observed

---

# 4. C4 (Hardest Case) Detailed Observations

## Baseline @ 12.0V

| Grid  | Step | Iterations | Time (s) | t_lin (s) |
| ----- | ---- | ---------- | -------- | --------- |
| Std   | 0.2  | 3          | 0.816    | 0.815     |
| Std   | 0.4  | 3          | 0.797    | 0.796     |
| Dense | 0.2  | 3          | 0.927    | 0.926     |
| Dense | 0.4  | 3          | 0.892    | 0.891     |

Observations:

* Iteration count remained low (≈3)
* No GMRES or LS instability
* Linear solve time stable
* No iteration explosion at high bias

---

## A1 @ 12.5V

All C4 configurations returned:

```
CONV(3/0/0)
```

* `dt_end = 10.0`
* No dt shrink trend
* No DT_COLLAPSE
* No sign of dynamic stress

---

# 5. Interpretation

## 5.1 Bias Domain Hypothesis

The hypothesis that 8V was insufficient to enter nonlinear exponential instability is not supported.

Extending to 12V:

* Did not increase iteration count
* Did not increase linear solver stress
* Did not induce failure modes
* Did not trigger separation

In fact, high-bias points converged efficiently.

---

## 5.2 Implication

The persistence of convergence under:

* Maximum trap density
* Maximum asymmetry
* Compressed geometry
* Elevated bias domain

suggests one of the following:

1. The numerical structure of the Baseline solver is inherently stable under current discretization.
2. The discretization may not be resolving true geometric singularities.
3. The physical configuration remains within a numerically well-conditioned regime.

---

# 6. Conclusion

Bias extension to 12V does not induce instability.

All axes tested so far (Trap, Alpha, Geometry, Bias) have failed to produce:

* GMRES failure
* Line search failure
* Iteration explosion
* Budget timeout
* Failure separation

The system remains fully convergent.

---

# 7. Strategic Direction

Given that all primary physical axes have been exhausted under current discretization, the next rational step is:

**Discretization Sensitivity Validation**

Before further physical escalation, test whether:

* Geometry is being under-resolved.
* Singular field gradients are not properly represented.

Recommended next stage:

> Introduce an Ultra grid (e.g., 240 × 120) with all physical parameters locked.

If Ultra grid remains fully stable, then the conclusion becomes stronger:

> The Baseline solver is structurally stable under extreme physical configurations within this modeling regime.

---

# Final Assessment

v1.4.6-013 is a clean and valid experiment.
The bias-domain hypothesis has been tested and rejected.

The instability threshold, if it exists, lies beyond current physical or discretization boundaries.

---


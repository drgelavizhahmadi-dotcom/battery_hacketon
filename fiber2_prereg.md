# Pre-registration: fiber test after certified convergence (fiber2)

Written and committed before `fiber2.py` was written or run (2026-09-29). Verdicts are appended
below the "Verdicts" heading after the run; nothing above it is edited afterwards. The thresholds
in G1–G5 are the ones in the brief, unchanged.

## Discrepancies between the brief and the files (files take precedence)
1. **R students have no polished weights.** The brief says to start from saved polished weights.
   The control-3 (R) students were saved after AdamW with **0 L-BFGS iterations**
   (`control3_results.json`), so they start from `gauge_weights/control3_R_s{100..119}.pt`.
   The independent seeds start from `gauge_weights/fiber_polished_s{0..5}.pt`.
2. **Normalisation of d_AB.** `fiber.py` normalised by the std of the ensemble-mean prediction.
   Here std(delta) is the std of the **data target** delta over the 2069 anchored training rows,
   because that is unambiguous and doesn't depend on which models converge.
3. **"h weighted by x_co".** x_co · h is exactly delta, so a literal reading repeats Step 1. I report
   the x_co-weighted RMS of (h_A − h_B), with weights w = x_co (not x_co²), over the
   x_co-weighted std of the mean h of the two seeds.
4. **Teacher for Step 2.** It is `gauge_weights/delta_s0.pt`, unpolished, exactly as in control 3.
   The polished independent seed 0 in Step 0 starts from the same weights but is a different model.
5. **Step 2 is implemented in `fiber2.py`,** reusing `fiber.py`'s helpers (`affine_fits`,
   `residual_ss`, `psi_unit`) with identical definitions. The teacher is the source (A) and
   each student the target (B). Continuous inputs only; the test region is also reported.
6. **As established in the first control, the dense quartile is 518 rows, all LiBOB.**

## Step 0: certified convergence
- **Models:** 6 independent KAN-delta seeds (0–5) and the 20 control-3 (R) students (seeds 100–119).
- **Loss (float64):** MSE(delta) + 1e-3 · smoothness penalty. This is each model's own training
  objective without AdamW's decoupled weight decay. Targets are the data delta for the independent
  seeds and the teacher delta on the real rows for the R students.
- **Optimiser:** L-BFGS, lr 1, strong-Wolfe line search, history 100, run in steps of 50 iterations,
  with internal tolerances disabled (tolerance_grad 1e-15, tolerance_change 1e-20).
- **Converged** after a step when both hold:
  - max |∂loss/∂θ| < 1e-7, and
  - |L − L₋₅₀| / L₋₅₀ < 1e-12, where L₋₅₀ is the loss 50 iterations earlier.
- **Stalls:** if a step takes no iteration, the optimiser is re-created once (history reset).
  If it stalls again immediately, the model stops as **stalled** (not converged).
- **Cap:** 20000 iterations.
- **Reported per model:** status (converged / stalled / cap), iterations, final loss, max |grad|,
  and the last relative change. Only converged models enter Steps 1–2.
- Weights are saved to `gauge_weights/fiber2_*` (not committed).

## Step 1: on-fiber test on delta (converged independent seeds)
- **Distance:** d_AB = RMS(delta_A − delta_B) / std(target delta), over training rows.
- **Definition A:** on-fiber if d_AB < 0.05.
- **Definition B:** on-fiber if RMS(delta_A − delta_B) < 0.25 × min(RMS data residual of A, of B).
- The x_co-weighted h distance (discrepancy 3) is also reported.

## Step 2: fiber analysis, converged R students vs the teacher
- **Definitions:** unit matching and the (i) centred / (ii) shared affine / (iii) free affine
  residuals are defined exactly as in `fiber_prereg.md`, and so is the Step-4 outgoing cross-check
  (with b_jk including the salt constant and biases).
- **Evaluation:** the grid, plus row-based dense / sparse / test regions. The fit is on all training
  rows, and the denominator is shared across regions.
- **Exploratory:** unit splitting in both directions (student unit vs sums of two teacher units,
  and teacher unit vs sums of two student units).

## Predictions
- **G1.** At least 4 of the 6 independent seeds, **and** at least 50% (≥ 10/20) of the R students,
  certify convergence.
- **G2.** At least 1 pair of converged independent seeds is on-fiber under Definition A.
  Inconclusive if fewer than 2 seeds converge.
- **G3.** At least 50% of the pairs of converged independent seeds are on-fiber under
  Definition B. Inconclusive if fewer than 2 seeds converge.
- **G4.** For converged R students, dense rows, unit pairs (teacher → student) with match
  |corr| ≥ 0.9: median of (1 − R_ii / R_i) ≥ 0.80. Inconclusive if that set is empty.
- **G5.** Same unit set, dense rows: median R_ii ≤ 1.5 × median R_iii.

Outputs: `fiber2.py`, `fiber2_output.txt`, `fiber2_results.json`, `figs/fiber2/`
(convergence, on_fiber, residual_ladder, unit_match, outgoing_crosscheck).

---

## Verdicts (appended after the run; `fiber2_output.txt`, `fiber2_results.json`, `figs/fiber2/convergence.png`)

**Step 0: no model certified convergence.** All 26 hit the 20000-iteration cap; none stalled.

| Group | Converged | Final loss | max \|grad\| at the cap (threshold 1e-7) | Relative change over the last 50 iterations (threshold 1e-12) |
|---|---|---|---|---|
| Independent seeds 0–5 | **0/6** | 8.52e-3 – 9.90e-3 (from 9.28e-3 – 1.05e-2 after `fiber.py`) | 4.8e-5 – 2.4e-4 | 2.3e-5 – 7.1e-5 |
| Control-3 R students 100–119 | **0/20** | 3.7e-5 – 6.0e-5 | 1.1e-6 – 3.7e-5 (median ≈ 2.2e-6) | 8.3e-8 – 8.6e-4 |

For scale, the students' loss at the 5% RMSE gate was about (0.05 × 0.336)² ≈ 2.8e-4. So 20000
iterations cut it by roughly 5–7×, and it was still falling.

| Prediction | Verdict |
|---|---|
| **G1** ≥ 4/6 independent and ≥ 50% of R students converge | **failed**: 0/6 and 0/20 |
| **G2** ≥ 1 independent pair on-fiber (Definition A) | **inconclusive**: 0 converged seeds, so Step 1 was not run |
| **G3** ≥ 50% of independent pairs on-fiber (Definition B) | **inconclusive** (not run) |
| **G4** shared affine removes ≥ 80% of the centred residual (dense) | **inconclusive**: 0 converged students, so Step 2 was not run |
| **G5** (ii) ≤ 1.5 × (iii) (dense) | **inconclusive** (not run) |

Notes (not verdicts):
- **The criterion is far from reach, not narrowly missed.** The independent seeds are 2–3 orders of
  magnitude from the gradient threshold and 7 orders from the relative-change threshold, and their
  loss is still falling at about 5e-5 per 50 iterations. The students are about 1 order from the
  gradient threshold but 5–9 orders from the relative-change threshold.
  - Both groups are **still descending along very flat directions** after 20000 L-BFGS iterations
    in float64.
  - For these models, "certified convergence" as defined (|grad| < 1e-7 and 1e-12 per 50 iterations)
    is not achievable within this budget. It may not be achievable at all under L-BFGS on this
    penalised loss.
- **Whether control 3's unit mismatch (0.659) survives convergence remains untested.** No model
  reached the pre-registered state. Running Step 2 on the capped students would be an exploratory
  analysis outside this pre-registration.
- The new `gauge_weights/fiber2_*` files (26 models plus 2 meta files) postdate the archive
  `gauge_weights_2026-09-29.tar.gz` and are **not** in `weights_manifest.txt`.

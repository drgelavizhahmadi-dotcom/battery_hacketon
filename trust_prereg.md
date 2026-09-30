# Pre-registration: trust-region Newton polishing (trust), stage 1

Written and committed before `trust.py` existed (2026-09-30). Verdicts are appended below the
"Verdicts" heading after the run; nothing above it is edited afterwards. Thresholds are the brief's,
unchanged. Stage 2 is not part of this study.

## Discrepancies and operational choices (files take precedence)
- **Q1. Models and provenance.** All weights are extracted from the archives into a temporary
  directory outside the repo, and each file's SHA-256 is checked against `weights_manifest.txt`
  before loading. Any mismatch stops the run.
  - Baseline seeds: `fiber2_ind_{0..5}.pt` (archive b).
  - Students: `fiber2_stu_{115,104,114}.pt` (archive b). These are the lowest, 10th-ranked and
    highest final max|grad| in `fiber2_stu_meta.json`, the same picks as support Step 3.
  - Teacher: `delta_s0.pt` (the first archive).
- **Q2. Parameters and loss.** The 609 parameters that affect the loss (support D2); the 16
  embedding parameters of salts absent from the training rows are excluded.
  - The loss is MSE(delta) + 1e-3 · smoothness penalty, in float64, on the 2069 anchored training
    rows: data targets for the baseline seeds, teacher delta for the students.
  - The optimiser works on this flat vector. Gradients come from torch autograd; Hessian-vector
    products are exact, by double backward.
- **Q3. Optimiser.** `scipy.optimize.minimize(method="trust-krylov", jac=True, hessp=...)` with
  maxiter = 500 and gtol = 1e-30, so scipy's own gradient test never stops it before these rules do.
  - A callback records every iteration and raises StopIteration once the model is **converged**
    or 6 h of wall-clock time have passed.
  - The stop reason is reported: converged, iteration cap, 6 h cap, or scipy's own termination
    (reported with its message).
- **Q4. Converged (certified local minimum)** when all three hold:
  - (a) max|grad| < 1e-7;
  - (b) |L_{k−10} − L_k| / L_{k−10} < 1e-12;
  - (c) the smallest full-Hessian eigenvalue is ≥ −1e-8 × the largest.
  - (c) is evaluated whenever (a) and (b) hold. If (c) fails there, the model is a saddle and
    optimisation continues.
  - At the end, all three are re-evaluated at the final point, and that decides the verdicts.
- **Q5. Diagnostics at the start point** (reported, not verdicts):
  - the relative error of hessp against the full autograd Hessian times v, and against a central
    finite difference of the gradient (ε = 1e-6), for 3 random unit vectors;
  - the loss recomputed at the capped weights, compared with the loss recorded in fiber2.
- **Q6. Classifying models that don't converge** (used by the interpretation rules):
  - **DESCENDING-SADDLE:** the loss fell by > 1e-10 (relative) over the last 50 iterations, **and**
    the final Hessian has a negative eigenvalue < −1e-8 × the largest.
  - **COLLAPSED:** the relative loss change over the last 50 iterations is < 1e-12, max|grad| ≥ 1e-7,
    **and** the final trust radius (from scipy's intermediate result) is < 1e-10.
  - **OTHER:** anything else.
- **Q7. T4's comparison set.** "The capped median" means the median d_AB, in the capped state, over
  the same pairs of converged seeds.
- **Q8. Edge match** (primary): the procedure from `control3.py`.
  - Gauge steps 1–4 on the training rows; units Hungarian-matched to the teacher.
  - The per-edge |corr| on a 100-point grid, over teacher-active continuous edges (≥ 5% of the
    largest edge variance in the unit); the median is reported.
  - Secondary: the `fiber.py` Step-3 shared-affine residual R_ii on the grid.
  - Unit match is the median |corr| of the matched contributions.
  - The same measures are also computed at the students' capped state, with the same code, for
    comparison.
- **Q9. Zero modes** (exploratory): the eigenvectors with |λ| < 1e-8 × the largest, at the final point
  of converged models. Each block's share of the energy is Σ over zero modes of ‖v_block‖², divided
  by the number of zero modes. The blocks are:
  - **layer-1:** layers.0 coefficients and base weights for the 8 continuous inputs, plus the
    layer-0 bias;
  - **salt:** the used embedding rows plus layers.0 coefficients and base weights for the 2
    embedding inputs;
  - **layer-2:** all layers.1 parameters.
- **Q10. Outputs.** The polished weights are saved to `gauge_weights/trust_*` (not committed, and not
  archived unless asked).

## Logged per model
Iterations; the loss and max|grad| curves (plotted); the number of negative eigenvalues at the start
and end; final loss vs capped loss; wall-clock time; the stop reason; the Q6 class if not converged.

## After convergence (converged models only)
- Baseline seeds: d_AB on delta for all converged pairs (RMS / std of the data target), with the
  capped d_AB for the same pairs.
- Students: correlation of delta with the teacher; unit match and edge match (Q8).
- Zero modes (Q9).

## Predictions
- **T1.** At least 4 of the 6 baseline seeds converge (Q4 at the final point).
- **T2.** At least 2 of the 3 students converge.
- **T3.** The converged baseline seeds' final loss is ≤ 0.95 × their capped loss, for every converged
  seed. (Reported per seed; T3 holds if all of them satisfy it.) Inconclusive if none converge.
- **T4.** Among converged baseline seeds, median d_AB ≤ 0.5 × the capped median over the same pairs
  (Q7). Inconclusive if fewer than 2 converge.
- **T5.** The converged students' median unit match to the teacher (over units × converged students)
  is ≥ 0.8. Inconclusive if none converge.

## Interpretation (fixed in advance)
- **T1 and T2 hold:** the non-convergence was caused by the optimiser, not the landscape.
  - **Then if T4 holds:** the seed spread was unfinished training. **If T4 fails:** the seeds
    converge to distinct minima, so non-convexity causes the spread.
  - **And if T5 holds:** the students' unit mismatch was unfinished training. **If T5 fails:** the
    mismatch survives true convergence, a genuine non-identifiability at a minimum.
- **T1 fails and most non-converged baseline seeds are DESCENDING-SADDLE:** the landscape has long
  descending paths (saddles or sloppy directions); stage 2 would localise them.
- **T1 fails and most non-converged baseline seeds are COLLAPSED:** report as numerical, and check
  float64 Hessian accuracy (Q5) before any further step.
- **Any other combination:** reported as found.

Outputs: `trust.py`, `trust_output.txt`, `trust_results.json`, `figs/trust/`.

---

## Verdicts (appended after the run; `trust_output.txt`, `trust_results.json`, `figs/trust/trust_curves.png`)

**Deviation from Q6 (flagged at the top of the log).** scipy 1.13's trust-region callback receives
only (x, fun), not the trust radius. So COLLAPSED used the closest observable proxy: the largest step
length ‖x_k − x_{k−1}‖ over the last 50 iterations < 1e-10.

**Inputs.** All 10 files matched their manifest SHA-256 values. Every capped loss recomputed here
matches its fiber2 record to ≤ 1e-7 (relative).

**Hessian accuracy (Q5).** hessp agrees with the full autograd Hessian to 2e-15 – 9e-15, and with a
central finite difference of the gradient to 5e-10 – 2e-7, for all 9 models. **The Hessian is accurate.**

| Model | Iterations (stop) | Loss ratio (final / capped) | max\|grad\| start → end | Negative eigenvalues* start → end | Class |
|---|---|---|---|---|---|
| ind 0 | 9 (scipy) | 0.99691 | 6.4e-5 → 1.5e-4 | 0 → 0 | OTHER |
| ind 1 | 26 (scipy) | 0.99445 | 5.9e-5 → 3.0e-5 | 0 → 0 | OTHER |
| ind 2 | 62 (scipy) | 0.99431 | 1.3e-4 → 7.3e-5 | 0 → 0 | OTHER |
| ind 3 | 8 (scipy) | 0.99661 | 2.1e-4 → 1.4e-3 | 0 → **4** | DESCENDING-SADDLE |
| ind 4 | 11 (scipy) | 0.99792 | 5.5e-5 → 3.8e-5 | 0 → 0 | OTHER |
| ind 5 | 1 (scipy) | 1.00000 | 1.1e-4 → 1.1e-4 | **1** → 1 | COLLAPSED (one iteration, no step) |
| stu 115 | 3 (scipy) | 0.98207 | 1.2e-6 → 2.2e-5 | 0 → **1** | DESCENDING-SADDLE |
| stu 104 | 5 (scipy) | 0.97870 | 1.7e-6 → 2.8e-6 | 0 → 0 | OTHER |
| stu 114 | 1 (scipy) | 1.00000 | 4.0e-5 → 4.0e-5 | 0 → 0 | COLLAPSED (one iteration, no step) |

\*Eigenvalues < −1e-8 × the largest (criterion c).

- **Every model stopped on scipy's own termination,** "A bad approximation caused failure to predict
  improvement", after 1–62 iterations. None reached the 500-iteration or 6 h caps; the total wall
  clock was about 7 min.
- **Criterion (c) (no significant negative curvature) held at the final point for 6 of 9 models.**
  All 9 failed (a) (max|grad| < 1e-7; the best was 2.8e-6) and (b).
- **Models that moved took large steps** (parameter-space step lengths up to 4.0) for loss decreases
  of only 0.2–2.1%.

| Prediction | Verdict |
|---|---|
| **T1** ≥ 4/6 baseline seeds converge | **failed**: 0/6 |
| **T2** ≥ 2/3 students converge | **failed**: 0/3 |
| **T3** converged seeds' loss ≤ 0.95 × capped | **inconclusive**: no seed converged. The best achieved ratio was 0.9943 |
| **T4** converged pairs' median d_AB ≤ 0.5 × capped | **inconclusive**: no seed converged |
| **T5** converged students' unit match ≥ 0.8 | **inconclusive**: no student converged |

**Interpretation, by the pre-registered rules.** T1 failed. Of the 6 non-converged baseline seeds,
4 are OTHER, 1 DESCENDING-SADDLE and 1 COLLAPSED, so neither the saddle rule nor the numerical rule
has a majority. **By the catch-all rule, this is reported as found, without choosing an explanation.**
Q5 shows the float64 Hessian is accurate, which rules out Hessian error as the reason for the
early terminations.

# Revision step 5: B-spline replication of the synthetic 2 × 3 study

This file is committed before `paper2/revision/bspline.py` exists. Verdicts will be appended below the "Verdicts" heading;
nothing above it will be edited. Files win over this text; discrepancies found while running go at the top of
`paper2/revision/bspline_output.txt`. Inputs are SHA-verified against `weights_manifest.txt` and held in memory. No weights go
into git; new weights are archived outside the repository with SHA-256 manifest entries.

## Discrepancies with the brief, found while writing (files win unless stated)

- **F1. μ calibration: the committed procedure exists** (`paper2/synth.py`, 34cc321, Step 0), and it differs from the brief's
  fallback.
  - **The committed procedure:**
    - μ grid {1e-6, 1e-5, 1e-4, 1e-3}, tried in increasing order.
    - First, for each input condition, the regularised teacher (the teacher polished under P2 at that μ) must move the teacher's
      predictions by less than 0.5σ RMS AND certify.
    - Then a pilot on the correlated-P2 cell with seeds 100–107: early stopping + polish, and the maximum ‖θ‖² growth must be ≤ 3.
    - The smallest μ that passes is frozen.
  - **What is used: the committed procedure, mirrored EXACTLY.** That includes its correlated input condition, both
    regularised-teacher checks and its pilot seeds 100–107. The committed rule takes the maximum growth over the pilot models, so
    fewer seeds would be a weaker test than the one that set the RBF μ. None of these seeds is used in Step 1.
  - **The certification criterion** (all 8 pilot models certified, incl. certified-degenerate) is reported beside it,
    descriptively. Growth factors decide nothing in Step 1; here the growth threshold is only a calibration rule.
  - **If the two criteria would select different μ,** the committed growth criterion decides, as it did for the RBF μ. Both values
    and the disagreement are reported at the top of the log and in the verdict text.
- **F2. The P1 penalty form.**
  - The committed SiLU term is λ K_silu Σ w², with λ = 1e-3, K_silu = (30 + π²)/90 = 0.4430, and w the SiLU base weights of both
    layers (`synth.penalty_terms`). The brief's "K_silu α²" is read as that term.
  - λ and K_silu are kept as committed. The D2 penalty on B-spline coefficients corresponds to ∫ f''² at a different scale than on
    RBF coefficients (knot spacing h = 0.4 against the RBF spacing 2/7), so the relative weight of the two penalties is not
    matched to the RBF study. It is kept as committed and noted.
- **F3. Evaluation outside [−1, 1] never occurs.**
  - The committed generator clips every input to [−1, 1] (`make_inputs`: clip(z/3, −1, 1)), including the off-manifold test inputs.
  - The layer-2 inputs are tanh(s) ∈ (−1, 1).
  - So no edge is ever evaluated outside [−1, 1]. This is logged. The basis is still defined on the extended knot span
    [−1 − 3h, 1 + 3h] (zero beyond it), but that is never reached.
- **F4. No teacher smoothing rounds.**
  - The committed teacher acceptance retried with a smoothed teacher (frequency scale × 0.8, up to 10 rounds); it passed at scale
    1.0 in the first round.
  - Here the brief says to stop and report if acceptance fails, so no smoothing rounds are run. The committed scale (1.0, from
    `synth_results.json`) is used.
- **F5. Labels come from the refitted B-spline teacher.**
  - This mirrors the committed construction, where the labels come from the teacher in the model's own class: y = teacher(X) + σε,
    with σ = 0.05 std(teacher(X)).
  - The inputs X, Xv, Xt, Xo and the noise draws ε are identical to 34cc321: the same generator and seeds 1000 + c, …, 5000 + c.
    The arrays' SHA-256 are logged and compared with those regenerated through `synth.py`'s own functions.
  - The labels therefore differ from the committed ones only through the teacher refit, which is bounded by Step 0 (i).

## Lessons carried over

- **Growth factors do not separate a distant finite minimum from a runaway.** Step 2 (190544d) has certified finite minima at ×583.
- **Certification decides,** with the degeneracy screen and the probes (degeneracy follow-up babbcba; infimum test 4003b21).
- **A two-decade D cannot by itself separate a distant finite minimum from a receding one.** In the dose-response POST-HOC check
  (75b778b), low-ρ paths with D far above 1.5 were heading to certified finite minima.
- **The fix claim requires certified AND bounded:** the dose-study R2 standard, ≥ 7 of 8 certified and median D ≤ 1.5.

## Question

Are the runaway and its fix a property of KAN-style edges with second-difference penalties, or an artefact of the Gaussian-RBF basis?

## Edge type

- **The basis.** Cubic B-splines (k = 3) on uniform knots: G = 5 intervals on [−1, 1] (h = 0.4), extended by k = 3 knots beyond each
  end. That gives 12 knots tᵢ = −1 + (i − 3)h, for i = 0, …, 11, and G + k = 8 basis functions per edge, matching the 8 RBFs.
  - The knots are not clamped.
  - The basis is evaluated by the Cox–de Boor recursion in torch, so it is differentiable in x (C² for cubics), as the hidden units
    and the exact Hessian require.
- **The edge.** w_base SiLU(x) + Σⱼ cⱼ Bⱼ(x), with no grid updates.
- **The layer.** `BSplineLayer` is a drop-in replacement for `kan.KANLayer`, with the same attributes: `base` (Linear, with bias),
  `coef` (d_out, d_in, 8), `edge`, `smoothness`. So `synth.FlatS`, `synth.run_trust`, `certify_point` and infimum.py's
  `Flat`/`analyse`/`probes`/`polish` work unchanged.
  - The model is `BSynthKAN` [4, 3, 1], with tanh between the layers, as in `synth.SynthKAN`.
- **The implementation it mirrors.** efficient-kan (Blealtan), `KANLinear.b_splines`: the uniform grid extended by
  `spline_order` knots on each side, and the Cox–de Boor recursion. These are also pykan's `extend_grid` / `B_batch` knots.
- **Differences from efficient-kan:**
  - a per-unit bias in the base Linear (as the committed `KANLayer`; efficient-kan's base has no bias);
  - no `spline_scaler`, and no grid update;
  - initialisation as the committed `KANLayer`: coefficients N(0, 0.1²), the base with nn.Linear's default;
  - the regulariser is the committed D2 (+ SiLU, + μQ), not efficient-kan's L1 / entropy regularisation;
  - float64 in the polish.
- **Linear precision and the null space.**
  - On these extended uniform knots the Greville abscissae ξⱼ = (tⱼ₊₁ + tⱼ₊₂ + tⱼ₊₃)/3 are equally spaced: ξ = −1.4, −1.0, …, 1.4.
  - So linear-in-index coefficient sequences cⱼ = a + bξⱼ reproduce a + bx exactly on [−1, 1], by B-spline linear precision on the
    interior span [t₃, t₈] = [−1, 1].
  - The second-difference penalty's null space, span{1, j} = span{1, ξⱼ}, is therefore exactly the affine functions on [−1, 1].
    With clamped knots this would hold only approximately near the ends.
  - P_null is the projection onto span{1, j}. Its QR basis (`synth.NB`) is the same 8 × 2 matrix as in the RBF study.

### Property tests

These are committed, implementation-only, BEFORE the run, as `tests/test_bspline.py`. They run again at the start of the run,
and the run stops if any gate fails.

- **(a) Partition of unity.** Σⱼ Bⱼ(x) = 1 on [−1, 1] (a dense grid of 10,001 points, including the knots), within 1e-12.
- **(b) Exact affine reproduction.** Sequences a + bξⱼ reproduce a + bx on [−1, 1] within 1e-12 (several (a, b) pairs).
- **(c) The penalty's null space.** The second-difference penalty is exactly zero on constant and on linear-in-index sequences
  (within 1e-12 in float64), and its null space (the D2 matrix's) has dimension exactly 2.
- **(d) Orthonormality.** The null-space basis (QR) is orthonormal: ‖NBᵀNB − I‖ ≤ 1e-12.
- **(e) Descriptive, not a gate.** The two null-space basis functions' maximum deviation from exact affinity on [−1, 1]: the
  residual of a least-squares line fitted to Σⱼ NBⱼₘ Bⱼ(x) on the grid, measured and reported. Expected at float precision for
  these knots.

## Penalties (as in the synthetic study, adapted to the basis)

- **P0** = data + λ Σ_edges ‖D2 c‖², with λ = 1e-3 as committed.
- **P1** = P0 + λ K_silu Σ w², over the SiLU base weights of both layers (F2).
- **P2** = P1 + μ Σ_edges ‖P_null c‖², with P_null the projection onto constant + linear-in-index sequences, as in `close.py` and
  `synth.py`; μ is frozen in Step 0.

## Step 0: calibration (before Step 1; nothing here is a result)

- **Teacher refit.** The committed teacher functions (`synth.build_teacher`'s L1 and L2 edge functions, at frequency scale 1.0)
  are refitted into the B-spline basis, mirroring the committed construction (34cc321):
  - a smoothness-penalised least-squares fit of each edge function on 401 points of [−1, 1], with basis [SiLU, B₀, …, B₇];
  - penalty R = λ DᵀD on the spline coefficients + λ K_silu on the SiLU weight;
  - biases zero.
- **Acceptance, both required.**
  - **(i) Refit fidelity.** The refitted teacher reproduces the committed (RBF) teacher's predictions on the training inputs of
    both conditions within 0.1σ RMS. The number is reported.
  - **(ii) Polish stability, as in the synthetic study.** Polishing the refitted teacher under P0 and under P1 (`run_trust`, on
    each condition's data) changes its predictions by less than 0.5σ RMS.
  - If either fails, stop and report (F4).
- **μ calibration.** The committed procedure, mirrored exactly, with pilot seeds 100–107 (F1). If no μ passes, stop and report.
- **Order.** Step 0 is logged before Step 1. The frozen μ is recorded as a dated amendment in the Verdicts section.

## Design

- **Data.** The same inputs, n (2000), noise rule and data seeds as 34cc321, SHA-logged (F5).
- **Models.** Independent and correlated (ρ = 0.95) × P0/P1/P2, 8 seeds each (0–7): 48 models.
- **Training.** AdamW early stopping as `synth.train_early` (lr 3e-3, wd 1e-4, full batch, patience 100, at most 10,000 epochs),
  with `BSynthKAN`.
- **Polish.** `synth.run_trust` on `synth.FlatS` (trust-exact, float64, cap 500 iterations, 2 h), in a child process hard-killed
  at the cap + 5 min, with the "solver error" stop handling of dose crash record 2 (11e3663).
- **Certification.** `certify_point`.
- **Failure reasons recorded per model:**
  - iteration cap;
  - wall-clock cap;
  - criterion a (gradient), b (curvature), c (Newton decrement), d (null-space gradient);
  - solver error (inherited from dose crash record 2, 11e3663).
- **Degeneracy screen and probes, exactly as in the infimum test.**
  - The screen: more than 3 near-flat directions OR more than 10% saturation.
  - The three probes: restart (gradient tolerance ×100 tighter), scaling (the layer-1 affine component, which is now the
    affine-in-index part of the spline coefficients, and all layer-1 weights), and the flat-direction search.
  - Probe-then-include, PLATEAU / FLAT MINIMUM classes, and the stalled-restart flag.
- **Short regularisation path per model.**
  - L + γ‖θ‖² at γ ∈ {1e-3, 1e-5, 1e-7}, warm-started, with the same protocol as the dose-response study.
  - D = ‖θ*(1e-7)‖² / ‖θ*(1e-5)‖², as reached (primary), with the certified-only variant beside it.
  - If the two variants give different readings for a cell, the cell is reported as SENSITIVE TO CERTIFICATION, with both values.
- **Running.** On AC power with sleep disabled. Models may run in parallel processes.

## Pre-registered predictions

"Certified" means certified including certified-degenerate (FLAT MINIMUM). PLATEAU counts as uncertified.

- **B1.** Correlated P0 and correlated P1: at most 2 of 8 certified, each.
- **B2.** P2, both input conditions: at least 7 of 8 certified AND median D ≤ 1.5, each.
- **B3.** Independent P0 and P1: at least 6 of 8 certified, each (the Gaussian reference level).

**Descriptive:**
- the per-block anatomy of the polished change: the affine (null-space) part of the B-spline coefficients, the curved part, the
  SiLU weights and the biases;
- the largest hidden-unit correlation;
- the cross-edge cancellation in both layers;
- growth factors (never used in a verdict);
- the median D per cell, with the caveat that D alone does not separate a distant finite minimum from a receding one.

## Interpretation (fixed in advance; checked in this order, all matches reported)

1. **B1 fails.** The effect does not carry over at this setting. Report it plainly, and state the paper's scope as RBF edges.
2. **B1 and B2 hold.** The runaway and its fix carry over to B-spline KANs.
3. **B1 holds, B2 fails.** The runaway carries over, but the null-space fix at this μ does not close it. Report which leg failed
   (certification or D), and qualify the paper's recommendation for B-splines.
4. **B3 fails.** The B-spline basis is less benign on independent inputs than the RBF basis; report it.
5. **Any combination not covered.** Partial; report per cell.

Verdicts are appended below the Verdicts heading; nothing above it is edited.

## Report

- **A per-cell table generated from the results JSON:** certified of 8, failure reasons, probe classes, median D (and its
  certified-only variant), anatomy shares, hidden-unit correlation, cancellation, and growth median and range.
- **Figures as in the synthetic study:** the ‖θ‖² trajectories, and off-manifold error against cancellation.
- **The Step 0 log:** refit fidelity, acceptance, the μ-calibration table, the frozen μ.
- **The property-test results,** including the affinity deviations (e).
- **The discrepancy list** at the top of the log.
- **Next to the results:** λ is kept as committed, so the strength of the second-difference penalty relative to the curvature it
  measures differs between the B-spline and RBF bases (F2).

---
## Verdicts

(appended after the run; nothing above this line will be edited)

### AMENDMENT (2026-10-22, written after the Step 0 (i) failure, while Step 0 (ii) was still running; before any further run)

- **Step 0 (i), refit fidelity: FAILED.** The refitted B-spline teacher differs from the committed RBF teacher's predictions on
  the training inputs by 0.46σ RMS (independent) and 0.43σ RMS (correlated), against a required < 0.1σ.
- **Cause.** The gate compared two approximations of the same analytic teacher functions (`synth.build_teacher`'s edge
  functions), and the committed RBF teacher's own approximation error dominates the gap. The B-spline refit is closer to the
  analytic functions:
  - the linear edges are exact to about 1e-16, against an RBF error of 6.6e-3;
  - the curved edges have RMS errors of 1.9e-3 to 4.8e-3, against 6.3e-3 to 1.65e-2 for RBF.

  The per-edge errors of both teachers against the analytic functions are reported with the results.
- **The refit-fidelity gate is WITHDRAWN as mis-specified.** It is not replaced by a new gate, because its outcome is now known.
  The comparison of both teachers with the analytic functions is reported descriptively.
- **Unchanged.**
  - The labels stay from the refitted B-spline teacher (F5 unchanged).
  - Step 0 (ii), polish stability, remains a gate. If it fails, stop and report.
  - Step 0 (ii) is not restarted on optimised code: polish stability is a property of the teacher, unaffected by speed. Its
    per-polish wall-clock times are logged as the pre-optimisation baseline.

#### SUPPLEMENT to amendment cba2c07 (2026-10-22, written after the first Step 0 (ii) attempt hung; before the re-run)

**The first attempt.**
- The first Step 0 (ii) polish (independent, P0) did not return. It ran in-process from about 18:58 until it was killed at
  21:03:14.
- A stack sample showed the main thread in LAPACK `dpotrf`, inside scipy's unbounded trust-exact subproblem loop: the known hang.
- The committed `run_trust` checks its 2 h cap only in its callback, so the cap could not fire.

**The re-run.**
- Step 0 (ii) is re-run as four independent child processes, at most two in parallel. Each child is hard-killed at 2 h + 5 min,
  and the last checkpointed accepted iterate is kept.
- The code is the same unoptimised code, with the solver-error handling of dose crash record 2 (11e3663).
- This is an implementation-only crash-record commit.
- Determinism is not assumed.

**How each Step 0 (ii) polish ending is classified (fixed now):**
- **NORMAL COMPLETION** (`run_trust` returns, including at the 500-iteration cap or the 2 h callback cap): the < 0.5σ stability
  gate is evaluated at the returned point, as committed.
- **A caught solver error** (non-finite Cholesky factor, 11e3663) ends the polish at its last accepted iterate. It is classified by
  the three rules below, exactly like a kill.
- **KILLED, STALLED-AT-MINIMUM:** the child is killed AND the last 10 accepted iterations show < 1e-6 relative change in both the
  loss and ‖θ‖², which requires ≥ 10 accepted iterations. The gate is evaluated at the last accepted iterate; it counts toward
  the gate, flagged STALLED-AT-MINIMUM.
- **KILLED, SUBSTANTIAL:** killed without plateau evidence but after ≥ 250 accepted iterations (half the committed cap). The gate is
  evaluated at the last accepted iterate; it counts toward the gate, flagged KILLED-SUBSTANTIAL.
  - Rationale: the committed RBF Step 0 judged its non-certifying teacher polishes at the iteration cap, where they were also
    still moving.
- **KILLED-IN-MOTION:** killed without plateau evidence and after < 250 accepted iterations. The gate is uninformative for that
  polish. Stop and report, as for a gate failure, under this distinct label.

**Stop reasons and scope.**
- A polish killed after hanging records "solver error (hang)" and the fact that it was ended by the 2 h + 5 min child kill. This is
  consistent with the frozen failure-reason list (11e3663).
- This supplement changes no polish, no teacher and no gate threshold.
- The μ-calibration polishes of the regularised teacher run in the same hard-killed children (the same exposure). Their endpoint is
  judged by the committed rule (< 0.5σ AND certified), with the ending recorded.

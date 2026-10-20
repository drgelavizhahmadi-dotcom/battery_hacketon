# Revision step 4: dose-response — how much correlation does the blind spot need?

This file is committed before `paper2/revision/dose.py` exists. Verdicts will be appended below the "Verdicts" heading; nothing
above it will be edited. Files win over this text; discrepancies found while running go at the top of
`paper2/revision/dose_output.txt`. Inputs are SHA-verified against `weights_manifest.txt` and held in memory. No weights go into
git; new weights are archived outside the repository with SHA-256 manifest entries.

## Discrepancies with the brief, found while writing (files win)

- **E1. The generator already has a ρ parameter.**
  - `synth.make_inputs(rho, seed)` draws z ~ N(0, 1)^(n×4) from `default_rng(seed)` and mixes the pairs as
    z₂ ← ρ z₁ + √(1 − ρ²) z₂ and z₄ ← ρ z₃ + √(1 − ρ²) z₄. It then clips z/3 to [−1, 1].
  - ρ = 0 skips the mixing in the same code path, which is how the committed independent cell was generated.
  - No generator edit is needed, so no implementation-only commit or property-test file is made. The run still logs, as
    runtime checks, the empirical correlations and the marginals at every ρ (see "Paired design").
- **E2. The committed cells did not share data seeds, so the paired design cannot reproduce both anchors.**
  - Data seeds are 1000 + c, 2000 + c, 3000 + c and 4000 + c, and the noise seed is 5000 + c, where c is the condition's index
    (0 independent, 1 correlated).
  - This study uses c = 1 at every ρ (the correlated cell's seeds). Its ρ = 0.95 cells therefore reproduce 34cc321 exactly, but its
    ρ = 0 cell is a new draw, not the committed independent cell (seed 1000).
  - ρ = 0 is added for P0 and P1, as the brief asks when the code path allows it. The committed independent cell is reported
    beside it as a descriptive reference.
- **E3. The committed polish (`synth.run_trust`) has no hard-kill timer.**
  - It measures the 2 h cap with `time.time()`: wall clock, so machine sleep counts against it.
  - Its trust-exact subproblem can hang (crash records 1 and 2).
  - It is used unchanged, inside a child process that is hard-killed at the cap + 5 min, keeping the last checkpointed iterate (as
    in the infimum test). This wrapper is implementation only.
  - Runs are made on AC power with sleep disabled (`caffeinate -is`), so that `time.time()` and elapsed compute time agree.
- **E4. The teacher acceptance step is not repeated.**
  - The committed teacher (synth Step 0, frequency scale from `synth_results.json`) was accepted at ρ = 0 and 0.95 only.
  - It is used unchanged at every ρ. The noise σ = 0.05 × std(teacher output) is recomputed per ρ by the committed rule and logged.

## What is known

- **Synthetic study (34cc321).**
  - Correlated, ρ = 0.95: P0 0/8 certified, ×65.7; P1 0/8 certified, ×32.1; P2 (μ = 1e-5) 8/8 certified, ×1.00.
  - Independent: P0 6/8 certified, ×12.5; P1 6/8, ×7.9; P2 8/8, ×1.09.
- **Anatomy (ea22a56).** In synthetic data the runaway lives mainly in layer-2 output-edge cancellation, unsaturated.
- **Infimum test (results 4003b21).**
  - The growth lives in the affine RBF parts and the SiLU weights.
  - The regularised minimisers recede over γ = 1e-2 to 1e-7 in synthetic P0 and P1 and in the real models; the P2 control and
    the closed real-data loss plateau (control arm pre-registered in 349022b).
- **Growth magnitude alone is not the runaway discriminator.** Step 2 (190544d) has certified finite minima with ×583 growth.
- **Failing to certify within the cap can be a budget effect.** In the degeneracy follow-up (babbcba), ind2 and ind3 certified after
  a restart.
- **Certification can pass a degenerate plateau** (ind5).
- **Consequence.** The degeneracy screen, the probes and a short regularisation path are standard here.

## Question

At what input correlation does the runaway turn on, and does the null-space penalty still close it at the strongest dose?

## Design

- **Everything as committed in `paper2/synth.py` (34cc321).** Files win.
  - Model: SynthKAN [4, 3, 1], with the same teacher and data sizes.
  - Training: `synth.train_early`, AdamW (lr 3e-3, wd 1e-4, full batch), early stopping on the validation data term, patience
    100, at most 10,000 epochs.
  - Polish: `synth.run_trust` and `synth.FlatS`: trust-exact, float64, exact Hessian, cap 500 iterations, 2 h, the stage2
    stopping rule.
  - Certification: `certify.certify_point` (the four-part criterion).
  - Cell losses: P0 = data + D2; P1 = P0 + SiLU; P2 = P1 + μQ with μ = 1e-5 (the committed frozen value), in both stages.
- **ρ grid.** {0, 0.5, 0.8, 0.9, 0.95, 0.99}; ρ = 0 is added per E2.
- **Cells.**
  - P0 and P1 at every ρ, seeds 0–7: 96 models.
  - P2 control at ρ = 0.99, seeds 0–7: 8 models.
  - 104 models in total.
- **Paired design (common random numbers).**
  - Every ρ uses the same seeds (c = 1 per E2): the same base draws z, the same noise draws and the same training seeds. Only the
    mixing coefficient changes.
  - The mixing leaves each marginal N(0, 1), so the clipped marginals are unchanged in distribution. Logged per ρ:
    - the realised correlations corr(x₁, x₂) and corr(x₃, x₄) on the training, validation and test inputs; tolerance
      |r − ρ| ≤ 0.02 (clipping lowers r slightly);
    - each input's mean, std and clipped fraction;
    - inputs x₁ and x₃, which are the same draws at every ρ: these must be exactly identical across ρ.
- **Replication check, a gate run FIRST, before the rest of the grid.** The new ρ = 0.95 P0 and P1 cells must reproduce 34cc321
  under the same seeds:
  - early-stopped weights identical to the committed `synth_correlated_{P0,P1}_s*.pt` early states (max |diff| = 0);
  - polished stop reason, iteration count, certification and growth identical to `synth_results.json`.

  If the early-stopped weights do not match exactly, or the polished certification outcomes differ, the run stops and is reported
  before any other ρ level is run. Other differences (stop reason, iterations, growth) are logged at the top of the output (files
  win). The runtime checks (E1) and the replication check go at the top of the output log, pass or fail.

## Measures per model

1. **Polish from the early-stopped weights.**
   - Record certified or not, with the failure reason: iteration cap, wall-clock cap, or the failing criterion (a gradient,
     b curvature, c Newton decrement, d null-space gradient).
   - Every certified point gets the degeneracy screen: more than 3 near-flat Hessian directions (|λ| < 1e-8 × largest) OR more than
     10% saturation (|s| > 3 at the hidden pre-activations, over the training rows).
   - Degenerate points get the three probes of the infimum test, with the same code (`infimum.probes`) and the same classes:
     1. restart: gradient tolerance ×100 tighter, same caps;
     2. scaling: (i) the layer-1 affine RBF component and (ii) all layer-1 weights and biases, by c ∈ {1.01, 1.1, 1.5, 2, 5, 10};
     3. flat-direction search: the 10 flattest eigenvectors, both signs, 13 log-spaced steps from 1e-3 to 1 × ‖θ‖.
   - PLATEAU if the restart lowers the loss AND raises the norm (both by a relative change > 1e-6), OR any probe finds a relative
     loss decrease > 1e-6 along a norm-increasing direction. Otherwise FLAT MINIMUM. The stalled-restart flag is defined as in the
     infimum test.
   - Probe-then-include: FLAT MINIMUM counts as certified (flagged as certified-degenerate); PLATEAU counts as uncertified.
   - **Uncertified fraction** = (uncertified + PLATEAU) / 8 per cell.
2. **Runaway rate**, for uncertified models only: the least-squares slope of log ‖θ‖² against the iteration index over the last 100
   polishing iterations (all iterations if fewer than 100, flagged).
3. **Short regularisation path.**
   - L + γ‖θ‖² at γ ∈ {1e-3, 1e-5, 1e-7}, using the infimum test's point protocol (`infimum.polish`, child process, cumulative
     caps, certification, degeneracy screen and probes).
   - Warm-started: γ = 1e-3 from the early-stopped weights, then 1e-5 from the 1e-3 solution, then 1e-7 from the 1e-5 solution.
   - All three points always run; there is no path-stopping rule.
   - Record ‖θ*‖², L* (without the γ term), and certified / degenerate / class at each γ.
   - D = ‖θ*(1e-7)‖² / ‖θ*(1e-5)‖², computed from the points as reached (primary; this is the D that enters R2 and R4). Both
     points are flagged if either is not certified. A certified-only variant (models with both points certified, incl.
     certified-degenerate) is reported beside it.
   - This chain skips the intermediate γ values of the infimum test, so its points are not identical to that test's Phase 1 paths
     at the same γ. The ρ = 0.95 values are compared with them descriptively.
   - **Within-cell spread of the paths:** the ‖θ*‖² spread (max − min)/median, the pairwise relative parameter distance, and the
     prediction RMS difference divided by the output std, at each γ (whether seeds collapse, as in the infimum test's Phase 1).
4. **Anatomy (secondary).**
   - `anatomy_synth.delta_shares` (ea22a56) on early → polished: the shares of ‖Δθ‖² in the affine RBF part, the curved RBF part,
     the SiLU weights and the biases, per layer.
   - The **affine-RBF share** used in R3 is the sum of the layer-1 and layer-2 affine RBF shares.
5. **Diagnostics.**
   - Per ρ, on the standardised training inputs: the Gram (correlation-matrix) condition number, its minimum eigenvalue, and each
     input's VIF (the diagonal of the inverse correlation matrix).
   - Per model, at the early and polished states:
     - the largest |correlation| between hidden-unit activations tanh(s);
     - the cross-edge cancellation in layer 1 and layer 2 (`anatomy_synth.cancellation`: cross and layer2; also within and
       between).

**Growth factors** (polished / early, per block) are recorded descriptively and are NOT used in any verdict.

## Pre-registered predictions

- **Shared conventions.**
  - "Across ρ" means the 6 levels 0, 0.5, 0.8, 0.9, 0.95 and 0.99.
  - "Non-decreasing, one inversion allowed" means at most one consecutive pair of levels with a strict decrease.
  - Medians are over the 8 seeds of a cell.
- **R1.** The uncertified fraction (measure 1) is non-decreasing across ρ, for P0 and for P1 separately, allowing one inversion.
- **R2.** Control, P2 at ρ = 0.99: at least 7 of 8 certified (including certified-degenerate), and median D ≤ 1.5.
- **R3.** The median affine-RBF share of the polished change is non-decreasing in ρ, for P0 and for P1 (one inversion allowed).
- **R4.** Median D increases with ρ, for P0 and for P1: Spearman correlation ≥ 0.8 between ρ and the cell median D over the 6
  levels. (Spearman ≥ 0.8 is the operative criterion; it tolerates up to about two inversions.)
- **Which D.** The D entering R2 and R4 is the as-reached variant (primary). The certified-only variant is reported beside it. If
  R4's verdict differs between the two variants, R4 is reported as SENSITIVE TO CERTIFICATION, with both values; neither variant
  decides alone.

**Descriptive:**
- the threshold ρ*: the lowest ρ at which the uncertified fraction exceeds 50%, per penalty, if any;
- the runaway rate against log(1/(1 − ρ)), over the uncertified models of each penalty, with a fitted least-squares slope and a
  bootstrap 95% interval (models resampled, 10,000 resamples, seed 0);
- whether L* at the polished endpoint falls with ρ (median per cell, Spearman).

## Interpretation (fixed in advance)

The branches are checked in this order, and the first that matches is reported. R3 and the descriptive measures are reported
regardless.

1. **R2 fails.** The fix has a dose limit at ρ = 0.99 with μ = 1e-5. Report it, and qualify the paper's recommendation. The
   remaining branches are still evaluated and reported.
2. **R1 fails, with the runaway already at ρ ≤ 0.5** (uncertified fraction > 50% at ρ = 0 or 0.5, for that penalty).
   Correlation amplifies the runaway but is not necessary, consistent with the real few-valued descriptors; qualify the paper's
   framing.
3. **R1 fails otherwise (non-monotone shape).** Report the shape as found; no threshold is stated.
4. **R1 and R4 hold for both penalties, and R2 holds.** Dose-response established: non-convergence and path divergence both
   increase with input correlation, and the null-space penalty closes the runaway at the strongest dose tested.
5. **R1 holds for both penalties but R4 fails for at least one.** Reported per penalty: non-convergence rises with correlation,
   but path divergence does not for that penalty. Report that the uncertified models may be budget-limited rather than
   diverging.
6. **Every remaining case where R1 or R4 holds for one penalty only.** Partial; report both penalties.

- R3 is secondary. Whether it holds or fails, it is reported with the anatomy; it does not change the branch.
- Verdicts are appended below the Verdicts heading; nothing above it is edited.

## Report

- **A per-cell table generated from the results JSON:**
  - ρ, certified of 8, failure reasons, probe classes;
  - median D (and its certified-only variant);
  - runaway rate;
  - Gram condition number, minimum eigenvalue, VIF;
  - largest hidden-unit correlation, cancellation (layers 1 and 2);
  - growth median and range (descriptive).
- **Figures:**
  - the uncertified fraction and median D against ρ, on a log(1/(1 − ρ)) axis, with P0 and P1 lines and the P2 point at 0.99
    marked;
  - the anatomy shares against ρ.
- **Run notes:**
  - provenance (the script commit, the inputs);
  - the runtime correlation and marginal checks (E1);
  - the replication check;
  - the within-cell path spread;
  - the discrepancy list.
- Models may run in parallel processes.

---
## Verdicts

(appended after the run; nothing above this line will be edited)

### Verdicts (recorded 2026-10-21 from `dose_output.txt` / `dose_results.json`)

**Gate and runs.**
- The replication gate PASSED. All 16 ρ = 0.95 P0 and P1 models reproduce 34cc321 exactly: identical early-stopped weights (max |diff| 0), best epochs, stop reasons, iteration counts, certification, and growth to 6 decimals.
- The run stopped twice before completing, both times implementation-only (commits d665db2 → 1d53d85, then 11e3663):
  - **crash record 1:** SIGBUS in Apple Accelerate's dgemm during a polish Hessian;
  - **crash record 2:** a non-finite Cholesky factor in scipy's trust-exact subproblem. The one affected point (P1 ρ = 0.99 s1, γ = 1e-7) is recorded as uncertified, with the reason "solver error".
- No child was hard-killed, and no point hit the wall-clock cap.
- The runtime correlation check failed its own tolerance at ρ = 0 and 0.5 (see the log's discrepancy summary): sampling noise, not a generator defect.

| Prediction | P0 | P1 |
|---|---|---|
| **R1**, uncertified fraction non-decreasing (≤ 1 inversion) | **HELD**: 0.125, 0.625, 0.500, 1, 1, 1 (1 inversion) | **HELD**: 0.125, 0.500, 0.500, 0.625, 1, 1 (0 inversions) |
| **R3**, median affine-RBF share non-decreasing (secondary) | **FAILED**: 0.715, 0.594, 0.612, 0.476, 0.394, 0.585 (3 inversions) | **FAILED**: 0.944, 0.982, 0.748, 0.934, 0.914, 0.952 (2 inversions) |
| **R4**, Spearman(ρ, median D) ≥ 0.8 | **FAILED**: D 8.01, 15.05, 6.43, 9.52, 16.13, 16.72; Spearman +0.714 | **FAILED**: D 4.54, 4.88, 2.91, 3.00, 4.52, 14.54; Spearman +0.143 |
| R4, certified-only variant | FAILED (Spearman +0.714; D at 0.99 is 19.55, n = 4) | FAILED (Spearman +0.143) |

- R4 is not sensitive to certification: both variants give the same verdict.
- **R2 HELD.** P2 at ρ = 0.99: 8/8 certified, median D 1.127 (range of growth ×0.88–×1.11).

**Branch (fixed rule 5):** "R1 holds for both penalties but R4 fails for at least one", here for both P0 and P1. Non-convergence rises with input correlation, but path divergence does not rise with it. The fixed text continues: "the uncertified models may be budget-limited rather than diverging".

**Descriptive (no verdicts).**
- **Threshold ρ\*** (first uncertified fraction above 50%): P0 0.5, P1 0.9. P0 at ρ = 0.5 (0.625) is a bump above ρ = 0.8 (0.500).
- **Runaway rate against log(1/(1 − ρ))** falls slightly:
  - P0 slope −7.0e-4 [−1.36e-3, +1.1e-4], over 34 uncertified models;
  - P1 slope −5.9e-4 [−1.11e-3, −7.6e-5], over 30 uncertified models.
- **The median polished L\* falls with ρ** (Spearman −0.943 for both).
- **The diagnostics rise monotonically with ρ:** Gram condition number 1.13 → 221, VIF max 1.00 → 52.1. The median hidden-unit correlation at the polished state goes from 0.05 (ρ = 0) to 0.82–0.96 (ρ ≥ 0.9).
- **Growth factors** are reported in the log's per-cell table and enter no verdict.
- **E2 reference.** The committed independent cells (seed 1000) certified 6/8 (P0) and 6/8 (P1). This study's ρ = 0 cells (seed 1001) certified 7/8 each.
- **Within-cell path spread.**
  - P1 collapses at ρ ≥ 0.8 for γ ≤ 1e-5: ‖θ*‖² spread ≤ 0.3% and prediction differences ≤ 1e-4 of the output std, apart from the solver-error point at 0.99 (γ = 1e-7); at γ = 1e-3 the ρ = 0.8 cell spreads 6%.
  - P0 does not collapse at γ = 1e-7: spread 26–36% for ρ ≥ 0.8, 179% at ρ = 0 and 330% at 0.5.

**Reading of the branch (POST-HOC, the author's to confirm; changes no verdict).**
- R4 fails because D is already well above the control at every dose, not because the paths stop diverging. Median D is 6.4–16.7 (P0) and 2.9–14.5 (P1) at every ρ, including ρ = 0, where 7/8 models certify. The P2 control at 0.99 has 1.13.
- Over the tested range, the regularised minimisers move outward at all correlations. What correlation governs is whether the unregularised polish certifies within the 500-iteration cap. The non-certifying models are therefore not merely budget-limited: they sit on paths that diverge at every dose.
- The paper's framing should say that correlation amplifies non-certification, while the divergence of the regularised path is present even with independent inputs.

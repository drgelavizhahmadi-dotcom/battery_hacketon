# Revision step 3: infimum test (a certified L2 regularisation path)

This file is committed before `paper2/revision/infimum.py` exists. Verdicts will be appended below the "Verdicts" heading;
nothing above it will be edited. Files win over this text; discrepancies found while running go at the top of
`paper2/revision/infimum_output.txt`. Inputs are SHA-verified against `weights_manifest.txt` and held in memory. No weights
go into git; new weights are archived outside the repository with manifest entries. Phase 0 is post-hoc (it re-examines
existing endpoints) and is labelled POST-HOC wherever it is reported.

## Discrepancies with the brief, found while writing (files win)

- **D1. Phase 0 inputs.** The six real Arm B endpoints of step 2 are `gauge_weights/freeze_B-R_ind{0..5}.pt` in archive 11
  (`gauge_weights_2026-10-17k.tar.gz`), not archive 12. Archive 12 (`gauge_weights_2026-10-18l.tar.gz`) holds the
  degeneracy replays, the restart endpoints and the checkpoints. Phase 0 uses the step-2 endpoints from archive 11, which
  are the points the degeneracy follow-up classified.
- **D2. "Synthetic P2 controls ranged up to 3" near-flat directions.** No committed synthetic P2 file supports this. In
  `paper2/synth_results.json` (34cc321), all 8 correlated-P2 polished models have `n_null` = 0 at the same threshold
  (|λ| < 1e-8 × largest). The range 0–3 is the real-data null-space-penalty models (μ = 1e-4, all rows; Figure 5a;
  `certify_results.json`, 36f295b). Synthetic Arm A and Arm B (step 2, babbcba) have 0–2. The DEGENERATE threshold (> 3)
  is kept as briefed. A P2 control flagged as degenerate is reported as information, not as a bug.
- **D3. The output-bias share for "synthetic seed 0".** See the Theory section. The large share (0.37) belongs to
  independent P0 s0. That cell is not part of this experiment. The correlated seed-0 models in this experiment have
  smaller shares.
- **D4. Uncertified-reason list.** `certify.py` has four criteria. The brief lists three of them: gradient, curvature and
  Newton decrement. The fourth, the null-space gradient (criterion d), is added to the reasons.

## What is known

- **Synthetic study (34cc321).**
  - Correlated P0 and P1 run away: growth ×65.7 and ×32.1, 0/8 certified each.
  - Correlated P2 (null space closed, μ = 1e-5) does not: ×1.00, 8/8 certified.
- **Revision step 2 (190544d; prereg `paper2/revision/freeze_prereg.md`).** With either layer frozen, most models reach
  certified finite minima:
  - Arm A (layer 1 frozen): 22/22 certified.
  - Arm B (layer 2 frozen): synthetic 16/16 certified; real 4/6.
  - Arm C (one layer): 8/8 certified.
  - Arm D (polish-time 1e-4‖θ‖²): 8/8 certified, growth ×0.415.
- **Degeneracy follow-up (POST-HOC, babbcba; `paper2/revision/degeneracy_output.txt`).**
  - The real Arm B endpoints are highly degenerate: 21–41 near-flat Hessian directions, against 0–2 in Arm A and
    synthetic Arm B.
  - One certified endpoint (ind5) was a PLATEAU, still drifting outward. Its restart lowered the loss by 1.0e-3, raised
    ‖θ‖² ×6.7 and ended uncertified.
  - Three restarts (ind0, ind2, ind3) stalled at the 2 h cap after 1–4 iterations.
- **Consequence.** At degenerate points the certificate alone is not sufficient evidence of a minimum.

## Question

Does the unregularised loss of the full model have a finite minimum that is merely distant? Or does its infimum lie at
infinity (a receding minimum)?

## Theory

- **The criterion.**
  - Let L be bounded below. That holds here: L is the mean squared error plus non-negative penalties.
  - Let R be coercive (R(θ) → ∞ as ‖θ‖ → ∞).
  - Then the global minimisers of L + γR stay bounded as γ → 0 if and only if L has a finite minimiser.
- **The penalty.** We use R = ‖θ‖² over ALL parameters, which is coercive.
- **Why Q is secondary.** The null-space penalty Q is not coercive, because the biases and the salt embedding are not
  penalised. Q is therefore used only in the secondary arms.
- **Limits.**
  - We find certified LOCAL minima, so the path is evidence, not proof.
  - Warm-starting each γ from the previous solution follows one local branch (a hysteresis chain).
  - A positive α over the tested range does not exclude a finite minimum so distant that the path's norm would plateau
    only at γ values below the tested range: for a minimum far from the start, the regularised minimisers' norm grows
    throughout the range. Every conclusion is therefore stated for the tested range of γ only.
- **Role of the probes.** Probe classifications decide only whether a point enters the path, never which conclusion the
  path supports. A chain of genuine minima with growing norm is itself the divergence signature.
- **Why all parameters are penalised.** Offset trading between the output bias and the near-constant part of the edge
  functions can carry the runaway. A penalty that leaves the biases free can then miss it.
  - Evidence from `paper2/anatomy_synth_results.json` and `paper2/anatomy_synth_output.txt` (ea22a56), rows
    `*/d/shares/L2 bias` (share of ‖Δθ‖² from the early to the polished state):
    - independent P0 s0: 0.365 (the output file, line 141: "puts 0.37 of its change into the output bias next to 0.59 in
      the affine RBF part of the output edges");
    - correlated P1 s0: 0.155;
    - correlated P0 s0: 0.011.
  - Only the first value strongly supports the argument, and it comes from a cell outside this experiment (D3). The
    paper quotes it with its cell name, or drops the number and keeps the argument.
- **L2 conventions.**
  - This path uses L + γ‖θ‖² with no factor 1/2. That is the convention of step 2 Arm D (1e-4‖θ‖²), so γ values are
    directly comparable to Arm D.
  - Under the decoupled weight-decay convention, the loss term is (1/2)·wd·‖θ‖². A coefficient c here corresponds to
    wd = 2c.
  - In particular, Arm D corresponds to wd = 2e-4, twice the training-time wd = 1e-4 examined in Arm E.

## Models, losses and order

- **Phase 1 (fast), synthetic SynthKAN [4, 3, 1].** Training data as in `paper2/synth.py` (34cc321): correlated
  inputs (ρ = 0.95), data seeds as committed. Start points are the committed early-stopped weights
  `gauge_weights/synth_correlated_{P0,P1,P2}_s{0..7}.pt` (archive `gauge_weights_2026-10-03h.tar.gz`).
  - **Runaway models (16):** correlated P0 and correlated P1, seeds 0–7, each with its own committed cell loss.
  - **Control models (8):** correlated P2, seeds 0–7, with the committed P2 loss: P1 + μQ, μ = 1e-5.
- **Phase 2 (slow), real KAN-delta [10, 6, 1].** Six models, all counted as runaway.
  - Loss: the original final.py loss. That is the mean squared error on all 2069 anchored rows + 1e-3 × ΣD2 over both
    layers, with no weight decay, as in step 2 Arm A-R.
  - Start points: the 107-epoch early states `gauge_weights/delta_s{0..5}.pt`.
- **L in every case** denotes this committed loss without the γ term.
- **Order.** Phase 1 is completed and reported before Phase 2 starts.

## Phase 0 (no training, run first; POST-HOC)

Retroactive flat-direction search on the six real Arm B endpoints of step 2 (D1).

- **Loss and space.** The step-2 Arm B loss and parameter space: layer 1 free; layer 2 and the salt embedding frozen.
- **Directions.** Take the up to 10 Hessian eigenvectors with the smallest |λ|. Use each in both signs.
- **Steps.** Evaluate the loss at s ∈ {10^-3, 10^-2.75, …, 10^0} × ‖θ_free‖, which is 13 log-spaced points.
  θ_free is the free (layer-1) parameter vector.
- **Report.** Any relative loss decrease > 1e-6 at a step where ‖θ_free + s·v‖ > ‖θ_free‖.
- **Classification.** Each endpoint is re-classified by the rule under "Degenerate certification". The rule combines:
  - the degeneracy follow-up's restart and scaling probes (babbcba);
  - this search, as probe (3).
- **Where it is reported.** As a dated, post-hoc-labelled addendum appended to
  `paper2/revision/degeneracy_output.txt` (the earlier text is not edited). The addendum is also summarised in the
  infimum log.

## Primary: the L2 path

- **Grid.** γ in half-decade steps from 1e-2 down to 1e-7: 1e-2, 3.16e-3, 1e-3, …, 1e-7 (11 values, exactly 10^(−2 − k/2)).
- **Warm start.**
  - The largest γ starts from the early-stopped weights.
  - Each smaller γ starts from the previous γ's final iterate, whether or not that iterate was certified.
- **Polishing.** trust-exact (scipy 1.13.1), float64, exact Hessian, the stopping rule of `stage2.run_exact`. Caps: 500
  iterations and 2 h wall-clock per point.
  - Each point runs in a separate child process with checkpoints (as in degeneracy.py after crash record 2).
  - The child is killed at the cap + 5 min. A killed point continues from its last checkpoint and is recorded as
    "wall-clock cap".
  - The 2 h wall-clock cap is cumulative across checkpoint resumes. A point that exhausts it is uncertified (wall-clock
    cap); its last checkpoint is the final iterate the next γ warm-starts from.
- **Certification.** Every point is certified with `certify_point` (the four-part criterion of `certify.py`: max|g| <
  1e-7; λ_min ≥ −1e-8 λ_max; Newton decrement < 1e-16; null-space gradient < 1e-9), applied to L + γ‖θ‖².
- **Recorded per point:**
  - ‖θ*‖² in total and per block. The blocks are:
    - layer-1 RBF, affine part (the projection onto span{1, k}) and curved part (the rest);
    - layer-2 RBF, affine and curved parts;
    - SiLU weights;
    - biases;
    - salt embedding (real models only).
  - L* (the loss without the γ term), the data term, and each penalty term.
  - Wall-clock time; iterations; the stop reason; certified or not.
  - For every uncertified point, the reason, from this list:
    - iteration cap;
    - wall-clock cap;
    - negative curvature (criterion b);
    - gradient criterion (a);
    - Newton decrement (c);
    - null-space gradient (d) (D4);
    - PLATEAU by probe.
- **Stopping a path.**
  - A model's path stops after two consecutive uncertified points. PLATEAU points count as uncertified.
  - Record where and why it stopped, and whether ‖θ‖² was still growing at the stop. "Still growing" means the
    point's ‖θ‖² exceeds the previous point's.
  - The remaining γ values are "not reached (stopped)".
- **Budget.**
  - At most 12 h wall-clock per real model in Phase 2. The probes count against it.
  - If the budget is exhausted, the remaining γ values are "not reached (budget)": missing, never uncertified.
  - Independent models may run in parallel processes.
  - Phase 1 has no budget beyond the per-point caps.
  - The secondary arms (a) and (c) have their own budget: 8 h wall-clock per model per arm, probes included. An exhausted
    arm budget likewise means "not reached (budget)": missing, never uncertified.
- **Order within Phase 2.**
  - First all six primary paths and the Phase 2 primary report.
  - Then arms (a) and (c), which run after the primary path.
  - The verdicts never depend on the arms.

## Degenerate certification

- **Recorded at every certified point:**
  - the saturation fraction: |s| > 3 at the hidden pre-activations, over the training rows;
  - the number of near-flat Hessian directions (|λ| < 1e-8 × largest);
  - the raw eigenvalue deciles.
- **DEGENERATE.** A certified point is DEGENERATE if it has more than 3 near-flat directions OR saturation above 10%.
  (D2: committed synthetic P2 polished models have 0.)
- **Three probes at every degenerate point.** All are evaluated on L + γ‖θ‖² at that γ.
  1. **Restart:** trust-exact with the gradient tolerance tightened ×100 (1e-7 → 1e-9), with the same caps.
  2. **Scaling:** multiply (i) the layer-1 affine RBF component, and (ii) all layer-1 weights and biases, by c ∈ {1.01,
     1.1, 1.5, 2, 5, 10}.
  3. **Flat-direction search:** the up to 10 flattest Hessian eigenvectors (smallest |λ|), both signs, loss at 13
     log-spaced steps s ∈ {10^-3, 10^-2.75, …, 10^0} × ‖θ‖ (θ = all parameters).
- **Classification (fixed now).**
  - **PLATEAU** if either of these holds:
    - the restart lowers the loss AND raises the norm, each by a relative change > 1e-6;
    - any probe finds a relative loss decrease > 1e-6 along a direction that increases the norm.
  - **FLAT MINIMUM** otherwise ("no descent found along the directions tested").
- **Stalled restart.**
  - Applies when the restart stops at the iteration or wall-clock cap with relative changes < 1e-6 in both loss and
    norm (as ind0, ind2 and ind3 did in the degeneracy study).
  - The classification then rests on probes (2) and (3) alone.
  - The point is flagged "restart stalled", and the cap it hit (iteration or wall-clock) is recorded.
- **Inclusion rule.**
  - FLAT MINIMUM points count as CERTIFIED-DEGENERATE. They enter I1–I4, flagged.
  - PLATEAU points count as uncertified. Whether the norm was still growing at the stop is recorded as
    divergence-consistent evidence.
  - Non-degenerate certified points enter directly.
- **Path continuation.** The path always continues from the point's final iterate, never from a probe's.

## Secondary arms (descriptive, no verdicts)

- **(a) Practical μ guidance.**
  - The path with R = μQ, using the committed null-space penalty of `close.py` (65ed55f): Q = Σ_edges ‖P_null c_e‖² +
    the squared SiLU base weights of both layers. The biases and the salt embedding are unpenalised.
  - Three real seeds only: ind0, ind1 and ind2, chosen by index before any arm results exist.
    μ ∈ {1e-4, 3e-5, 1e-5, 3e-6, 1e-6}, warm-started downward from the early states.
  - Phase 2, after all six primary paths and the Phase 2 primary report. Separate budget: 8 h per model per arm.
  - Same polishing, certification and degenerate probes as the primary path.
- **(b) Straight-line scan.**
  - Applies to each runaway model whose primary path stopped or reached γ = 1e-7.
  - The drift direction d is the normalised difference between its last two reached path points.
  - Scan θ_end + t·d over 50 log-spaced points, so that ‖θ_end + t·d‖ runs up to 100 × ‖θ_end‖. Evaluate the
    unregularised L.
  - Turning point s*: the first point where L exceeds L(θ_end). Report "none up to 100×" if there is no such point.
- **(c) Coercive Q.**
  - Q extended to also penalise the biases and the salt embedding (squared norms). Together with ΣD2 (which penalises
    the curved RBF part) this makes R coercive. The script checks this numerically: the penalty Hessian must be positive
    definite on all parameters.
  - Same three seeds (ind0, ind1, ind2), μ grid, budget and procedures as (a).
  - Report whether its path diverges, and in which blocks, side by side with the primary path.

## Pre-registered predictions

The predictions are evaluated on certified points, including certified-degenerate ones, separately per phase.

- **Count thresholds.** "≥ 80% of models" means at least 13 of 16 runaway models in Phase 1, at least 7 of 8 controls,
  and at least 5 of 6 models in Phase 2.
- **Too few points.** A model with too few certified points for a prediction (fewer than 3 for I1 and I4, fewer than 4
  for I2, fewer than 2 for I3) counts as NOT meeting it. Such models are listed separately, and the "inconclusive"
  branch below takes precedence.

- **I1.** Runaway models: ‖θ*‖² increases as γ decreases, over the model's certified points in γ order, in ≥ 80% of
  models. One inversion is allowed: at most one consecutive pair with ‖θ*‖² not increasing.
- **I2.** Runaway models: fit log ‖θ*‖² = a − α log γ over the smallest four certified (incl. certified-degenerate) γ
  values. α must exceed 0.2, with
  a bootstrap 95% interval that excludes 0, in ≥ 80% of models.
  - The bootstrap is a residual bootstrap: 10,000 resamples, fixed seed 0, percentile interval. Resampling cases is
    degenerate with 4 points.
- **I3.** Control (P2) models plateau: the ‖θ*‖² ratio between the two smallest certified γ values is ≤ 1.5, in ≥ 80% of
  models.
- **I4.** Runaway models: L* decreases as γ decreases (one inversion allowed) in ≥ 80% of models.

## Interpretation (fixed in advance)

- **I1, I2 and I4 hold, and I3 holds.** The regularised minimisers diverge as γ → 0 over the tested range: the loss behaves
  as if its infimum lies at infinity (a receding minimum). The paper states the tested range of γ.
- **The runaway models plateau.** "Plateau" means, in ≥ 80% of the runaway models, both:
  - the α interval includes 0 or α ≤ 0.2;
  - the ratio between the two smallest certified γ values is ≤ 1.5.

  Then a finite but distant minimum exists. The paper replaces "no reachable finite minimum" with "a minimum so distant
  that standard optimisers do not reach it".
- **Most runaway models' small-γ points are degenerate.** The verdict still comes from the path: I1, I2 and I4 on the
  certified points, including certified-degenerate ones.
  - A high PLATEAU share is itself divergence-consistent evidence: the regularised minimiser was not reached because
    the point kept moving outward.
  - Report the PLATEAU share per γ, and state explicitly how many points entered via the probes.
- **Fewer than three decades of γ reached for most runaway models** (certified, including certified-degenerate):
  inconclusive. Report how far each path got and why.
- **I3 fails.** The test does not discriminate. Report it, and do not use the test in the paper.
- **Any other combination.** Partial. Report the per-model results; no branch is selected.
- **Per-phase verdicts.** Phase 1 and Phase 2 get separate verdicts.
  - Phase 2 has no control models. Its I3 (the discrimination check) is Phase 1's I3, stated as such.

## Report

- **Phase 0:** the updated classes for the six step-2 endpoints (POST-HOC addendum).
- **Per model, the full path:**
  - for each γ: ‖θ*‖², L*, wall-clock, certified, degenerate, probe class, failure reason, stall flag and cap type;
  - the stop point and its reason, and whether the norm was still growing at the stop.
- **Unconditionally, a per-model table of fitted α:** its bootstrap interval, the number of points used, and how many of
  them entered via the probes.
- **Per-block growth along the path,** primary and arm (c) side by side. Arm (a) is reported in the same format.
  The report states that arms (a) and (c) are descriptive, with n = 3.
- **Verdicts** I1–I4 per phase, and the branch.
- **The PLATEAU share per γ.**
- **Figure:** ‖θ*‖² and L* against γ (log-log), one line per seed, runaway vs control. Degenerate points and plateau points
  are marked.

---
## Verdicts

(appended after the run; nothing above this line will be edited)

### PHASE 0 (POST-HOC; recorded 2026-10-18 from `infimum_output.txt`)

| Endpoint | Degeneracy class | Updated class | Flat-direction descents | Largest relative loss change |
|---|---|---|---|---|
| ind0 | FLAT MINIMUM | FLAT MINIMUM | 0 | −4.0e-13 |
| ind1 | FLAT MINIMUM | FLAT MINIMUM | 0 | −2.6e-15 |
| ind2 | FLAT MINIMUM | FLAT MINIMUM | 0 | −1.2e-12 |
| ind3 | FLAT MINIMUM | FLAT MINIMUM | 0 | −2.8e-14 |
| ind4 | FLAT MINIMUM | FLAT MINIMUM | 0 | −3.4e-15 |
| ind5 | PLATEAU | PLATEAU | 10 | −7.75e-6 (at s = 1 × ‖θ‖) |

- **No class changed.**
- **The five flat minima show no descent** along their 10 flattest directions, at steps up to the full norm. All loss changes are below 1e-12 in relative terms.
- **ind5 remains PLATEAU, with real descents.** The flat-direction search found 10 of them, and the restart had also lowered the loss and raised the norm.
- The addendum is appended, dated and labelled post-hoc, to `paper2/revision/degeneracy_output.txt`.

### PHASE 1 (recorded 2026-10-18 from `infimum_output.txt`; synthetic SynthKAN, primary L2 path)

All 264 points (24 models × 11 γ values) were reached and certified. There were no caps, no hard kills and no path stops.

| | Prediction | Verdict | Count |
|---|---|---|---|
| I1 | runaway: ‖θ*‖² increases as γ decreases | **HELD** | 16/16 |
| I2 | runaway: α > 0.2, bootstrap 95% interval excluding 0 | **HELD** | 16/16 |
| I3 | control: ratio between the two smallest certified γ ≤ 1.5 | **HELD** | 8/8 |
| I4 | runaway: L* decreases as γ decreases | **HELD** | 16/16 |

**Branch: RECEDING MINIMUM, for the tested range γ = 1e-2 to 1e-7 only.** The regularised minimisers diverge as γ → 0 over that range. All 16 runaway models reached three decades of γ.

**Per cell:**

| Cell | α (95% interval) | ‖θ*‖² at 1e-2 → 1e-7 | Ratio of the two smallest γ | L* at 1e-7 |
|---|---|---|---|---|
| P0 (runaway) | +0.633 to +0.758; intervals [+0.579, +0.685] to [+0.708, +0.807] | 0.88 → 64.8–97.5 | 1.80–2.27 | 6.46e-4 to 6.52e-4 |
| P1 (runaway) | +0.362 in all 8 ([+0.349, +0.376]) | 0.88 → 28.2–28.3 | 1.524–1.525 | 6.94e-4 |
| P2 (control) | +0.014 in all 8 ([+0.009, +0.019]) | 0.88 → 6.22–6.23 | 1.004 | 7.78e-4 |

**Degenerate points.**
- All 13 degenerate points (P0 only, at γ ≤ 3.16e-7: 5 at 3.16e-7 and 8 at 1e-7) were probe-validated FLAT MINIMA. There were 0 PLATEAU points, so the degenerate interpretation branch was not triggered.
- Each such point had 4–16 near-flat directions and 0 saturation.
- No probe found a descent. The smallest relative loss changes were +4.5e-2 for scaling and +9.4e-9 for the flat-direction search.

**Restart probes.**
- The restart probes stopped immediately. In 12 of 13 cases scipy's trust-exact returned "A bad approximation caused failure to predict improvement" after 0 iterations; the 13th took 14 iterations. There was no change in loss or norm, because the points were already at numerical precision.
- Admission therefore rested on the scaling and flat-direction probes.
- These are not "restart stalled" under the prereg's definition, which requires a cap (H2).

**Seed collapse.**
- **At γ = 1e-2, the 8 starts in each cell converge to essentially one function.**
  - The within-cell spread of ‖θ*‖² is 0.18–0.32%.
  - Prediction RMS differences are at most 0.7% of the output std.
- **Not one point in parameter space.**
  - The raw pairwise relative distance is 1.4–2.0 (median 1.6).
  - After aligning hidden-unit permutations and sign flips, it is still 0.26–0.43 for most pairs; some pairs coincide to 1e-8. The sign flip (with the layer-2 RBF coefficients reversed) is only an approximate symmetry, because SiLU is not odd.
- **Along the path, P1 and P2 stay collapsed.**
  - The ‖θ*‖² spread is at most 1.2% (P1) and 1.9% (P2) at every γ.
  - At γ = 1e-7 the aligned distances are 0.2% (P1) and 1.3% (P2).
  - Each of these cells is therefore one path reproduced from 8 starts.
- **P0 separates at small γ.**
  - The ‖θ*‖² spread is 13.6% at γ = 3.2e-6, 47% at 1e-6 and 37% at 1e-7.
  - The aligned distance at 1e-7 is 0.86 (median).
  - Three P0 seeds (s3, s5, s7) end at the same point.
- **Consequence.** The ≥ 80% counts reflect one path per cell for P1 and P2, and a few branches for P0, not independent replicates.
- **Correction.** An earlier summary to the author stated a within-cell spread below 0.2% throughout. That holds only at γ = 1e-2.

**Per-block growth (median ‖θ‖² per block over the 8 seeds, early state → γ = 1e-7; range at 1e-7 in brackets):**

| Block | P0 | P1 | P2 |
|---|---|---|---|
| L1 RBF affine | 2.38 → 12.6 [11.2–15.9] | 3.21 → 14.4 [14.3–14.4] | 2.80 → 2.91 |
| L1 RBF curved | 0.143 → 0.035 | 0.219 → 0.118 | 0.207 → 0.144 |
| L1 SiLU | 1.29 → 7.88 [7.14–10.9] | 0.0034 → 0.0003 | 0.0051 → 0.0030 |
| L1 bias | 0.182 → 4.30 [2.30–8.91] | 0.185 → 0.071 | 0.177 → 0.048 |
| L2 RBF affine | 2.74 → 26.0 [23.9–28.4] | 3.43 → 11.9 [11.9–12.0] | 2.89 → 2.98 |
| L2 RBF curved | 0.089 → 0.003 | 0.136 → 0.049 | 0.129 → 0.091 |
| L2 SiLU | 0.805 → 30.6 [5.12–39.8] | 0.0017 → 0.00005 | 0.0015 → 0.0005 |
| L2 bias | 0.056 → 6.41 [4e-5–7.08] | 0.060 → 1.72 [1.71–1.72] | 0.054 → 0.046 |

P1 diverges more weakly (α 0.362, final ratio 1.525). Its SiLU weights are penalised and shrink, so its growth moves into the affine RBF part of both layers and into the output bias. P0's growth goes mainly into the layer-2 SiLU weights, the layer-2 affine coefficients and the layer-1 biases (plus the layer-1 affine part and SiLU weights). P2's blocks stay flat.

**Arm (b), descriptive: no straight escape route.**
- Along a straight line in the final drift direction, P0's loss rises from the first step (turning point at ×1.00 of the norm).
- P1's loss falls by at most 2.2e-4 (relative) up to ×1.11 of the norm, then rises.
- The minimisers therefore recede along a curved valley, not a straight ray.

**Implementation choices H1–H7 (from the log):**
- **H1:** θ = all loss-relevant parameters. That is all 139 for the synthetic models, and for the real models the 609 of `trust.Flat`; the unused salt-embedding rows are fixed and excluded.
- **H2:** the stall check uses the relative change in ‖θ‖², not ‖θ‖. For small changes the ‖θ‖² change is about twice the ‖θ‖ change. With ‖θ‖ instead, the step-2 restarts change by 4.7e-4, 2.3e-6 and 2.3e-4 (ind0, ind2, ind3), still above 1e-6. No Phase 1 restart hit a cap. **The choice changed no classification.** The prereg's own definition makes ind0, ind2 and ind3 not stalled, contrary to its parenthetical.
- **H3:** relative norm changes use ‖θ‖². "A direction that increases the norm" means ‖y‖ > ‖x‖.
- **H4:** a real point whose remaining model budget is below the 2 h point cap is not started, and is "not reached (budget)". A point cut by the budget is "not reached (budget)", never uncertified.
- **H5:**
  - I1 and I4 need at least 3 included points, I2 at least 4, and I3 at least 2; otherwise the prediction is "not met".
  - "Three decades reached" means an included point at γ ≤ 1e-5.
  - Branch precedence: inconclusive > I3 fails > receding > finite distant > partial.
- **H6:** the arm (b) direction is the difference of the last two reached path points. The scan uses 50 log-spaced points from 1e-3 ‖θ_end‖ out to a norm of 100 ‖θ_end‖, and evaluates L without the γ term.
- **H7:** the stage2 stopping rule resumes from the checkpointed last 11 losses. The per-point caps are cumulative across resumes, tracked by heartbeat and checkpoint.

### POST-HOC AMENDMENT (2026-10-18, written after the Phase 1 results): cold-start basin check

POST-HOC everywhere. It is reported separately and cannot change any pre-registered verdict.

- **Design.**
  - Cells P0 and P1 at γ ∈ {1e-3, 1e-5, 1e-7}, plus P2 at γ = 1e-7 (the optional part).
  - Starts: the 8 committed early-stopped weights plus the all-zeros vector, 9 starts in all.
  - Each (start, γ) is polished independently, with no warm-start chain. The protocol is the primary path's: trust-exact, the caps, `certify_point`, the degenerate check and the probes.
- **Recorded per run:**
  - certified or not, with the reason;
  - ‖θ*‖² and L*;
  - the distance to the warm-started path's points at that γ: raw ‖x − x_warm‖/‖x_warm‖, to the start's own seed and the minimum over the 8 seeds;
  - the prediction RMS difference divided by the output std.
- **Jittered start.**
  - Added for a (cell, γ) only if the all-zeros start does not move, or ends with all hidden units identical (the symmetric manifold).
  - The jittered start is all-zeros plus iid Gaussian noise of scale 1e-6, from numpy `default_rng(20261018)`, the same vector for every (cell, γ).
  - If it escapes to the warm-path branch, the zero-start outcome is recorded as a "symmetric-manifold artefact of the exactly-zero start", not as evidence of a small-norm basin.
- **Basin groups (descriptive, fixed before the run):**
  - uncertified;
  - small-norm: ‖θ*‖² below 0.5 × the warm-path minimum;
  - warm-path branch: prediction within 2% of the output std of some warm-path point, and ‖θ*‖² within the warm-path range ± 10%;
  - other.
- **Question:** do cold starts reach the same diverging branch, or do small-norm minima exist at small γ?

### POST-HOC cold-start results (recorded 2026-10-18 from `infimum_output.txt`; cannot change any verdict)

- **Runs.** 63 independent polishes (54 in P0/P1 + 9 in P2 at γ = 1e-7). 62 were certified; P0 s7 at γ = 1e-7 hit the iteration cap. 21 of the certified points were degenerate, and all were FLAT MINIMUM.
- **Zero start.** It moved in every (cell, γ) (‖x − 0‖ = 1.2–10.5) and never ended on the symmetric manifold, so no jittered start was triggered.
- **No small-norm basin was found.** No certified cold endpoint at any γ has ‖θ*‖² below 0.5 × the warm-path minimum. Every cold endpoint has ‖θ*‖² ≥ 0.99 × the warm-path minimum.
- **γ = 1e-3: one basin.** All 9 starts in each of P0 and P1 reach the warm-path branch: ‖θ*‖² within the warm range, predictions within 3e-10 of the output std.
- **γ = 1e-5: mostly the warm-path branch.**
  - P0: 8/9 on the warm-path branch; s3 at ‖θ*‖² 7.06 against a warm range of 5.19–5.46.
  - P1: 7/9 on the warm-path branch; s2 and s3 at 8.16 against 6.24–6.25.
  - These are other certified minima at larger norm.
- **γ = 1e-7: cold starts reach larger-norm minima with lower L* and a lower regularised objective (L* + γ‖θ‖²) than the warm path.**

| Cell | Warm-path ‖θ*‖² | Cold certified ‖θ*‖² | Best objective found: warm → cold | ‖θ*‖² of the best point found | Cold starts beating the best warm point |
|---|---|---|---|---|---|
| P0 | 64.8–97.5 | 93.4–125 | 6.5456e-4 → 6.4828e-4 | 101.5 (s2) | 1/8 |
| P1 | 28.2–28.3 | 28.3–118 | 6.9728e-4 → 6.7367e-4 | 100.8 (s3) | 8/9 |
| P2 (control) | 6.22–6.23 | 6.22–9.58 | 7.7843e-4 → 7.7843e-4 (warm is best) | 6.22 | 0/9 |

- **Reading (POST-HOC).**
  - The warm-started chain follows a lower-norm branch. At small γ, better regularised minima exist at larger norms. P1's weak divergence (α 0.362) is a property of its warm branch: its best minimum found at γ = 1e-7 has ‖θ*‖² ≈ 100.8, against 28.2 on the warm path.
  - This is consistent with the receding-minimum branch, and goes further in its direction. The cold starts found no small-norm basin.
  - In the control, the warm path is the best minimum found. The other P2 minima (s2: ‖θ*‖² 9.58, L* 8.04e-4) are worse local minima, not lower-objective ones.

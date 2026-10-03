# Pre-registration: second-order certification and regularised accuracy (certify)

Written and committed before `certify.py` existed (2026-10-03), after the inputs were pinned
(`7b77bc3`: the `close_*` weights are in `gauge_weights_2026-10-03f.tar.gz`, listed in
`weights_manifest.txt`). Verdicts are appended below the "Verdicts" heading after the run; nothing
above it is edited afterwards. Thresholds are the brief's, unchanged. No further stages follow.

## Flags
- **F-a. The certification criterion was defined after the earlier one failed in practice.** Step 1's
  criterion was set after the stage2 / close.py criterion was seen to be unreachable: scipy stops a few
  iterations after reaching a minimum, so the 10-iteration relative-change window never fills. All
  results here are reported **alongside, never instead of**, close.py's verdicts (close.py V1: failed).
- **F-b. Timing of W4 and W5.** Both were set after close.py's μ = 1e-6 fold A result (1.0262) was known.
  W4's threshold (0.150) predates that result; W5's is defined relative to it. Both were fixed before
  any μ = 1e-4 fold A result existed.
- **F-c. W6 is a weak test.** It has 10 points (5 seeds × 2 μ), and Spearman ρ ≥ 0.6 corresponds to
  roughly p ≈ 0.07, so it is weak in either direction.

## Discrepancies and operational choices (files take precedence)
- **E1. Candidates.** Before the script existed, the max|grad| values stored in close.py's records gave
  17 candidates:
  - μ = 1e-4: all 6 seeds and all 3 students;
  - μ = 1e-5: seeds 0, 3, 4, 5 and all 3 students;
  - μ = 1e-6: student 104.

  `certify.py` recomputes max|grad| at every close.py final point and lists its candidate set first.
  The recomputed list is the one used.
- **E2. Loss and parameters for certification.** Each point is certified on **the loss it was optimised
  on**: the close.py loss data + smoothness + μ·Q at that point's μ, over the same 609-parameter vector
  (unused salt embeddings fixed). The exact autograd Hessian is computed in float64.
  - Null set: the eigenvectors with |λ| < 1e-8 × λ_max.
  - (c) The Newton decrement is Σ over non-null i of (v_i · g)² / λ_i. If (b) holds, every non-null
    λ_i is positive.
  - (d) The gradient's component in the null set is ‖V_nullᵀ g‖₂.
- **E3. "fiber.py procedure" for Step 2b** means the procedure close.py and trust.py used:
  - **unit match:** median |corr| of teacher-to-student units, Hungarian-matched on contributions;
  - **edge match:** per-edge |corr| of gauge-fixed layer-1 edges on a 100-point grid, over
    teacher-active continuous edges (≥ 5% of the largest edge variance in the unit), as in control 3;
  - **prediction correlation:** delta correlation with the teacher, on the training rows.
- **E4. Neither gauge direction in Step 2c is an exact symmetry here.** So each is built as the best
  least-squares direction, and its function-space residual is reported alongside its overlap.
  - **Constant shift (edge (i, j) into bias):**
    - Layer 1: the 9-coefficient change on edge (i, j) that best represents the constant +1 on the
      training rows, with Δb_j = −1. The constant is not exactly representable by an edge's basis:
      the per-edge residual is 6e-6 – 1.5e-2 (support Step 1).
    - Layer 2: the same on edge (j → output), with the output bias.
    - Evaluated for every continuous layer-1 edge and every layer-2 edge.
  - **Unit rescaling (unit j):** the infinitesimal direction d/da at a = 1.
    - Scale unit j's incoming layer-1 parameters (all 10 edges plus bias_j) by a.
    - Compensate in layer-2 edge j with the least-squares coefficient change that cancels −ψ_j′ · s_j on
      the training rows, in that edge's SiLU + 8-RBF basis of u = tanh(s_j).
    - This is not exact (tanh and a fixed grid).
  - **N:** support.py's near-null space of the layer-1 design matrix (36 of 72 dimensions). The overlap
    is computed with support's H5 metric on each null eigenvector's layer-1 continuous block.
  - **Reported:** for each certified model, the squared overlap ‖P_null d‖² / ‖d‖² of each normalised
    gauge direction with the null set, as a median and range per family, and the pooled N-overlap of
    the null-set eigenvectors. This is exploratory, with no verdict.
- **E5. Step 3.**
  - Fold A training rows are the anchored rows except PC+EA (1434 rows).
  - `kan.train` for seeds 0–4, 107 epochs, the final.py loss; then trust-exact under the close.py loss
    at μ = 1e-4 on those rows (float64, maxiter 500, 2 h safety cap, callback as in close.py).
  - The ensemble is the mean over the 5 polished models, on the 635 anchored PC+EA rows.
  - Per model: held-out RMSE, plus cross-edge ratio, ‖θ‖² and saturation computed with close.py's
    `cross_edge` and block norms on **fold A's training rows**. The μ = 1e-6 models are recomputed from
    the cached `close_foldA_1e-06_s*` weights.
  - Each μ = 1e-4 fold A model is also checked against the Step 1 criterion, on its own fold A loss.
- **E6. W3** pools every certified (student, μ) instance. A student certified at several μ counts once
  per μ.
- **E7. Inputs.** Everything comes from the pinned archives and is SHA-256-verified, held in memory and
  re-checked from memory; the extraction directory is deleted at once. Nothing is re-read from disk or
  `$TMPDIR`. The inputs are:
  - the close.py final points and fold A caches;
  - the 107-epoch seeds and the students' AdamW-stage weights, as mapping templates;
  - the fiber2 capped seeds, for capped d_AB;
  - the teacher `delta_s0`.
  - Support's N is recomputed from the data (no weights).

## Step 1: second-order certification (no optimisation)
- List the candidates first, then compute the Hessian and eigendecomposition at each.
- **CERTIFIED** if (a) max|grad| < 1e-7, (b) the smallest eigenvalue ≥ −1e-8 × λ_max, (c) the Newton
  decrement on the non-null eigenspace is < 1e-16, and (d) ‖g_null‖ < 1e-9.
- Report certified counts per μ (seeds and students separately) and every model's null-set size.

## Step 2: analysis at the certified minima
- (a) Seeds, per μ: d_AB on delta for all certified pairs, with the median next to the fiber2 capped
  median over the same pairs.
- (b) Students: prediction correlation, unit match and edge match to the teacher (E3).
- (c) Null-set overlaps with the explicit gauge directions and with N (E4). Exploratory.

## Step 3: accuracy at the regularised minima (E5)

## Predictions
- **W1.** At μ = 1e-4, at least 5 of the 6 seeds are certified.
- **W2.** Among seeds certified at μ = 1e-4: median d_AB ≤ 0.5 × the capped median over the same pairs.
  Inconclusive if fewer than 2 are certified.
- **W3.** Certified students, at any μ (E6): median unit match to the teacher ≥ 0.8. Inconclusive if
  none are certified.
- **W4.** The μ = 1e-4 fold A ensemble RMSE is ≤ 0.150.
- **W5.** The μ = 1e-4 fold A ensemble RMSE is ≤ 0.5131 (0.5 × close.py's μ = 1e-6 value of 1.0262).
- **W6.** Across the 10 polished fold A models, per-seed held-out RMSE increases with the cross-edge
  ratio: Spearman ρ ≥ 0.6. ρ and p are reported.

## Interpretation (fixed in advance)
- **W1 holds:** the targeted penalty restores finite minima, under the post-hoc criterion of F-a.
  close.py's V1 failure was an artifact of the certification window. This is reported alongside V1,
  not instead of it.
- **W2 holds:** once the runaway is removed, independent seeds converge to a common function, so the
  seed disagreement was caused by the runaway. **W2 fails:** distinct minima remain even without the
  runaway (non-convexity).
- **W3 holds:** the students' unit mismatch was caused by the runaway. **W3 fails:** the mismatch
  survives at true minima, a genuine non-identifiability.
- **W4 holds:** the regularised minima generalise as well as the early-stopped model, so the runaway's
  extra training fit was not predictive skill. **W4 fails:** the regularised minima cost real held-out
  accuracy.
- **W5 and W6 hold:** the data-invisible directions drive the extrapolation failure, and bounding them
  restores much of the held-out accuracy. **W5 fails:** bounding the runaway does not rescue
  extrapolation; polishing beyond early stopping is harmful for other reasons.
- **W6 alone is weak evidence** (F-c). Its ρ and p are reported, and the main conclusion does not rest
  on it.
- **Any other combination:** reported as found.

Outputs: `certify.py`, `certify_output.txt`, `certify_results.json`, `figs/certify/`. Weights go in
`gauge_weights/certify_*` and are not committed.

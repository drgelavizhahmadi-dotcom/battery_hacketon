# Pre-registration: stage 2, exact-Hessian optimiser check, runaway diagnostic, coercivity (stage2)

Written and committed before `stage2.py` existed (2026-09-30), after the inputs were pinned
(`89068a2`: the trust weights are in `gauge_weights_2026-09-30d.tar.gz`, listed in
`weights_manifest.txt`). Verdicts are appended below the "Verdicts" heading after the run; nothing
above it is edited afterwards. Thresholds are the brief's, unchanged. No further stages follow.

## Discrepancies and operational choices (files take precedence)
- **S1. Models and provenance.** The same 9 models as `trust.py`: baseline seeds 0–5 and students
  115, 104, 114.
  - Part A and Part C start from the **fiber2 capped** weights.
  - Every file is extracted from its archive and its SHA-256 checked against the manifest before
    loading; any mismatch stops the run. The teacher is `delta_s0.pt`.
  - The optimiser works on the same 609-parameter vector as `trust.py`, with the unused salt
    embeddings fixed.
- **S2. Part B states that do not exist as named.**
  - **"107-epoch saved weights"** exist only for the baseline seeds (`delta_s{0..5}.pt`). The
    students' AdamW-stage weights are `control3_R_s{k}.pt`, trained for **530–2680 epochs**
    (until control 3's stop condition), not 107. They are reported as the students'
    "AdamW-stage" state, with that caveat.
  - **"fiber"** (`fiber_polished_s{i}.pt`) exists only for the baseline seeds; the students have no
    fiber state, so it is N/A for them.
- **S3. Part B blocks** (norm = Euclidean norm of the block's parameters):
  - **layer-1 RBF:** `layers.0.coef`, all 10 inputs including the 2 salt-embedding edges;
  - **layer-2 RBF:** `layers.1.coef`;
  - **SiLU base:** `layers.0.base.weight` + `layers.1.base.weight`;
  - **biases:** `layers.0.base.bias` + `layers.1.base.bias`;
  - **salt embedding:** `emb.weight`, the 4 used rows only.
  - The per-block share of the displacement is ‖Δ_block‖² / ‖Δ‖², with
    Δ = Part A final − fiber2 capped.
- **S4. Which parameters the loss penalises,** from `kan.py`:
  - The smoothness penalty (1e-3 × Σ squared second differences along the RBF index) applies to
    `layers.0.coef` and `layers.1.coef` only.
  - It does not penalise base weights, biases or the embedding. Within each edge it is blind to RBF
    coefficient patterns that are constant or linear in the centre index (the null space of the
    second difference).
  - **The loss contains no weight decay.** AdamW's decoupled weight decay (1e-4) acted only during
    the AdamW stage, and the L-BFGS, trust-krylov and trust-exact stages have none.
  - `stage2.py` checks this by comparing the penalty with a recomputation from `kan.py`'s
    `smoothness()`.
- **S5. The optimiser.** `scipy.optimize.minimize(method="trust-exact", jac=True, hess=...)`, with the
  exact 609×609 autograd Hessian at every point where scipy asks for it.
  - Caps: maxiter 500, gtol 1e-30. A callback records each iteration and stops at convergence or
    after 2 h of wall-clock time per run.
  - scipy's callback receives only (x, fun), so the step length is ‖x_k − x_{k−1}‖. Iterations
    where scipy rejects the step show as 0.
- **S6. Convergence** uses the same three criteria as `trust.py` (Q4), applied to **the loss being
  optimised**, which in Part C includes the L2 term:
  - (a) max|grad| < 1e-7;
  - (b) relative loss change < 1e-12 over 10 iterations;
  - (c) the smallest Hessian eigenvalue ≥ −1e-8 × the largest.
  - (c) is checked whenever (a) and (b) hold, and all three are re-checked at the final point.
- **S7. Part C** adds λ × Σ θ² over all 609 optimised parameters, with λ ∈ {1e-6, 1e-5}, and starts
  from the fiber2 capped weights. It also reports the loss without the L2 term, as a ratio to the
  capped loss (whose loss contains no L2 term).
- **S8. After convergence** (Part C converged models only):
  - **d_AB:** as in `fiber2` (RMS of the delta difference / std of the data target) for converged
    baseline pairs, against the capped d_AB over the **same pairs**;
  - **students:** delta correlation, unit match and edge match to the teacher, by `trust.py` Q8.
- **S9. Runtime.** Each Hessian takes a few seconds. There are 27 runs (Part A plus two λ values ×
  9 models), and the worst case is 2 h per run. Runs are cached so the script can resume.

## Predictions
- **U1.** Part A: at least 4 of the 6 baseline seeds converge.
- **U2.** Part C, λ = 1e-5: at least 4 of the 6 baseline seeds **and** at least 2 of the 3 students
  converge.
- **U3.** Part C: more of the 9 models converge at λ = 1e-5 than at λ = 1e-6.
- **U4.** Every Part C converged model, at either λ, has a loss without the L2 term ≤ 1.05 × its
  capped loss. Inconclusive if none converge.
- **U5.** Part C, λ = 1e-5 (primary): among converged baseline seeds, the median d_AB ≤ 0.5 × the
  capped median over the same pairs. Inconclusive if fewer than 2 converge. λ = 1e-6 is reported
  descriptively.

## Interpretation (fixed in advance)
- **U1 holds:** the trust-krylov failures were an optimiser-implementation problem, and the landscape
  admits minima. Parts B and C are then secondary.
- **U1 fails, U2 holds:** without the L2 term, the loss has no attained minimum (or one extremely far
  away) along unpenalised directions. Part B should show which blocks run away, and coercivity
  restores convergence.
  - **Then if U5 holds:** the seed spread was unfinished training toward a common solution.
    **If U5 fails:** there are distinct minima even on the coercive loss.
- **U1 and U2 fail:** even an exact second-order optimiser on a coercive loss cannot certify minima.
  The landscape itself is the cause: extreme ill-conditioning and non-convexity.
- **U4 fails:** the L2 term changed the solution materially, and U5 must be reported with that caveat.
- **Any other combination:** reported as found.

## Part B (descriptive, no verdict)
Block norms at each available state (S2), for every model, plus the per-block share of
(Part A final − fiber2 capped).

Outputs: `stage2.py`, `stage2_output.txt`, `stage2_results.json`, `figs/stage2/`. Weights go in
`gauge_weights/stage2_*` and are not committed.

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


---

## Verdicts (appended after the run; `stage2_output.txt`, `stage2_results.json`, `figs/stage2/stage2_grad.png`)

**Run record.**
- **Crash 1:** an AttributeError in the callback, fixed in `78bcaba`; `a1c7842` is a reconstruction of the
  crashed version.
- **Crash 2:** after all 27 optimisation runs, Part B hit a FileNotFoundError because macOS's nightly
  `$TMPDIR` cleanup deleted 6 lazily re-read inputs. Fixed in `60621c6`, after which the run resumed
  from the pre-registered S9 cache.
- Both crashes are documented in `stage2_restart_note.md`, which is also merged into the log. Both
  fixes are implementation-only.
- **Before the resume:** the SHA-256 values of all 54 cached `stage2_*` files were recorded and
  checked.
- **During the resume:** all 33 inputs were re-extracted and re-verified, the S4 check passed again,
  and the 27 runs were loaded from cache (marked `[cached]`), not re-optimised.

**Verdicts.** The scripted verdicts are official, and they are identical to the provisional
hand-worked table.

| Prediction | Verdict | Key number |
|---|---|---|
| **U1** Part A: ≥ 4/6 baseline seeds converge | **failed** | 0/6. All 9 Part A runs hit the 500-iteration cap |
| **U2** Part C, λ = 1e-5: ≥ 4/6 seeds and ≥ 2/3 students | **failed** | 1/6 seeds (ind 0, after 373 iterations), 0/3 students |
| **U3** more models converge at 1e-5 than at 1e-6 | **held** | 1 vs 0 |
| **U4** converged Part C models: loss without L2 ≤ 1.05 × capped | **failed** | C1e-05 ind 0: 1.2545 |
| **U5** λ = 1e-5 converged seeds: median d_AB ≤ 0.5 × capped | **inconclusive** | only 1 baseline seed converged |

**Interpretation branch** (pre-registered rules):
- **U1 and U2 fail:** even an exact second-order optimiser on a coercive loss cannot certify minima.
  The landscape itself is the cause: extreme ill-conditioning and non-convexity. This branch fits the
  observed pattern directly; no other branch applies.
- **U4 fails:** the L2 term materially changed the solution. U5 is inconclusive anyway.
- **Descriptive context** (no verdict): at λ = 1e-5, the baseline seeds still end with 2–4
  significantly negative Hessian eigenvalues and max|grad| of about 1e-4 after 500 iterations,
  although ‖θ‖² is bounded at about 80–110. The non-convexity persists even when the parameters are
  kept small.

**Note on four runs: not converged by the criterion, but numerically strict local minima.**
- The runs are C1e-05 stu 115, stu 104 and stu 114, and C1e-06 stu 104. They ended with
  max|grad| = 1.2e-13, 1.3e-13, 3.5e-14 and 2.6e-15, and a positive-definite Hessian (smallest
  eigenvalue 2.0e-6, 1.2e-5, 9.6e-6, 4.0e-7 > 0).
- Numerically, they are strict local minima of the **regularised** loss.
- **They remain classified as not converged** under the pre-registered criterion. They fail only
  criterion (b), relative loss change < 1e-12 over 10 iterations, with values 3.2e-5, 6.8e-4,
  1.2e-3 and 2.8e-4.
- The reason: Newton converged quadratically in its last few iterations, and scipy then stopped
  ("A bad approximation caused failure to predict improvement") before 10 iterations of stationarity
  had accumulated. The loss 10 iterations earlier was therefore still measurably higher.
- Counting them would not change any verdict: U2 would be 1/6 seeds and 3/3 students, still failing
  on the seeds, and U3 would be 4 vs 1, still holding.
- Their loss without L2 is 2.06–4.82 × capped, so for the students the L2 term dominates the fit.

### Part B (descriptive, no verdict)

Block norms (Euclidean) along each trajectory. The students' first state is the control-3 AdamW
stage (530–2680 epochs), not 107 epochs, and they have no fiber state (S2).

| Model | State | layer-1 RBF | layer-2 RBF | SiLU base | biases | salt embedding |
|---|---|---|---|---|---|---|
| ind 0 | AdamW-107 | 2.06 | 0.98 | 1.73 | 0.55 | 3.35 |
| ind 0 | fiber | 27.01 | 13.46 | 32.45 | 3.68 | 5.62 |
| ind 0 | fiber2 capped | 68.63 | 23.53 | 67.99 | 18.32 | 6.66 |
| ind 0 | trust final | 70.91 | 23.38 | 69.06 | 19.63 | 6.67 |
| ind 0 | Part A final | 87.23 | 24.68 | 94.92 | 23.53 | 6.59 |
| ind 1 | AdamW-107 | 2.20 | 1.20 | 1.59 | 0.55 | 3.60 |
| ind 1 | fiber | 17.94 | 19.07 | 18.32 | 3.97 | 6.61 |
| ind 1 | fiber2 capped | 58.92 | 37.14 | 54.78 | 5.62 | 8.13 |
| ind 1 | trust final | 60.00 | 36.85 | 54.69 | 6.43 | 8.13 |
| ind 1 | Part A final | 76.93 | 46.54 | 75.46 | 14.49 | 7.96 |
| ind 2 | AdamW-107 | 1.92 | 1.00 | 1.54 | 0.50 | 2.55 |
| ind 2 | fiber | 14.10 | 18.71 | 16.08 | 6.06 | 6.08 |
| ind 2 | fiber2 capped | 24.46 | 51.69 | 41.67 | 10.07 | 6.42 |
| ind 2 | trust final | 25.12 | 54.16 | 43.00 | 10.56 | 6.38 |
| ind 2 | Part A final | 28.08 | 65.38 | 50.26 | 10.92 | 6.31 |
| ind 3 | AdamW-107 | 1.80 | 0.97 | 1.66 | 0.48 | 2.73 |
| ind 3 | fiber | 14.18 | 16.56 | 14.68 | 2.34 | 3.99 |
| ind 3 | fiber2 capped | 36.36 | 60.39 | 46.33 | 9.53 | 4.43 |
| ind 3 | trust final | 36.30 | 60.77 | 46.45 | 9.57 | 4.47 |
| ind 3 | Part A final | 45.93 | 69.67 | 61.07 | 11.49 | 4.55 |
| ind 4 | AdamW-107 | 1.94 | 1.00 | 1.72 | 0.52 | 3.45 |
| ind 4 | fiber | 29.93 | 21.30 | 32.12 | 8.96 | 5.27 |
| ind 4 | fiber2 capped | 116.30 | 83.69 | 95.61 | 32.18 | 7.26 |
| ind 4 | trust final | 116.44 | 84.56 | 95.30 | 32.93 | 7.28 |
| ind 4 | Part A final | 205.75 | 153.27 | 116.06 | 55.99 | 7.78 |
| ind 5 | AdamW-107 | 2.08 | 0.86 | 1.68 | 0.62 | 3.87 |
| ind 5 | fiber | 12.52 | 14.44 | 17.82 | 2.34 | 4.24 |
| ind 5 | fiber2 capped | 23.72 | 36.12 | 39.78 | 8.22 | 5.17 |
| ind 5 | trust final | 23.72 | 36.12 | 39.78 | 8.22 | 5.17 |
| ind 5 | Part A final | 26.09 | 38.97 | 40.63 | 8.36 | 5.11 |
| stu 115 | AdamW-stage (530-2680 ep) | 2.36 | 1.30 | 1.43 | 0.51 | 2.07 |
| stu 115 | fiber | N/A | N/A | N/A | N/A | N/A |
| stu 115 | fiber2 capped | 17.67 | 13.11 | 23.30 | 3.69 | 3.55 |
| stu 115 | trust final | 17.65 | 13.18 | 23.22 | 3.72 | 3.54 |
| stu 115 | Part A final | 22.79 | 16.94 | 30.49 | 3.97 | 2.93 |
| stu 104 | AdamW-stage (530-2680 ep) | 2.69 | 1.10 | 1.94 | 0.29 | 4.03 |
| stu 104 | fiber | N/A | N/A | N/A | N/A | N/A |
| stu 104 | fiber2 capped | 17.30 | 13.76 | 20.07 | 3.62 | 5.89 |
| stu 104 | trust final | 17.43 | 14.07 | 19.95 | 3.66 | 5.87 |
| stu 104 | Part A final | 22.01 | 23.23 | 22.43 | 4.57 | 5.72 |
| stu 114 | AdamW-stage (530-2680 ep) | 2.28 | 1.16 | 1.60 | 0.38 | 2.64 |
| stu 114 | fiber | N/A | N/A | N/A | N/A | N/A |
| stu 114 | fiber2 capped | 13.67 | 19.30 | 17.07 | 4.40 | 4.61 |
| stu 114 | trust final | 13.67 | 19.30 | 17.07 | 4.40 | 4.61 |
| stu 114 | Part A final | 13.47 | 24.41 | 19.88 | 3.19 | 4.73 |

Per-block share of ‖Part A final − fiber2 capped‖² (and ‖Δ‖):

| Model | layer-1 RBF | layer-2 RBF | SiLU base | biases | salt embedding | ‖Δ‖ |
|---|---|---|---|---|---|---|
| ind 0 | 0.323 | 0.063 | 0.590 | 0.024 | 0.000 | 58.8 |
| ind 1 | 0.342 | 0.233 | 0.386 | 0.038 | 0.001 | 56.0 |
| ind 2 | 0.301 | 0.277 | 0.400 | 0.021 | 0.000 | 27.5 |
| ind 3 | 0.282 | 0.204 | 0.503 | 0.011 | 0.000 | 33.6 |
| ind 4 | 0.653 | 0.231 | 0.080 | 0.036 | 0.000 | 147.9 |
| ind 5 | 0.511 | 0.135 | 0.343 | 0.010 | 0.000 | 13.1 |
| stu 115 | 0.435 | 0.121 | 0.431 | 0.009 | 0.004 | 17.8 |
| stu 104 | 0.401 | 0.393 | 0.195 | 0.011 | 0.000 | 16.9 |
| stu 114 | 0.295 | 0.304 | 0.366 | 0.033 | 0.002 | 14.8 |

Total parameter norm ‖θ‖² (609 optimised parameters) and the size of λ‖θ‖² relative to the data
loss L_data = MSE + smoothness penalty. Computed post hoc from the caches, as requested in review:

| Model | ‖θ‖² capped | L_data capped | λ=1e-5 × ‖θ‖² **at capped** (× L_data) | ‖θ‖² Part A final | ‖θ‖² C1e-6 final | 1e-6·‖θ‖² / L_data | ‖θ‖² C1e-5 final | 1e-5·‖θ‖² / L_data |
|---|---|---|---|---|---|---|---|---|
| ind 0 | 10266 | 8.521e-03 | 1.027e-01 (12×) | 17826 | 590 | 0.06 | 91 | 0.09 |
| ind 1 | 7950 | 8.784e-03 | 7.950e-02 (9×) | 14052 | 794 | 0.09 | 107 | 0.10 |
| ind 2 | 5149 | 9.453e-03 | 5.149e-02 (5×) | 7748 | 819 | 0.08 | 87 | 0.08 |
| ind 3 | 7226 | 9.379e-03 | 7.226e-02 (8×) | 10846 | 855 | 0.09 | 88 | 0.08 |
| ind 4 | 30760 | 8.944e-03 | 3.076e-01 (34×) | 82493 | 630 | 0.07 | 99 | 0.09 |
| ind 5 | 3544 | 9.900e-03 | 3.544e-02 (4×) | 3946 | 712 | 0.07 | 81 | 0.07 |
| stu 115 | 1053 | 4.383e-05 | 1.053e-02 (240×) | 1760 | 52 | 0.52 | 12 | 0.58 |
| stu 104 | 939 | 4.033e-05 | 9.393e-03 (233×) | 1581 | 46 | 0.55 | 15 | 0.89 |
| stu 114 | 891 | 3.914e-05 | 8.914e-03 (228×) | 1205 | 45 | 0.51 | 14 | 0.81 |

**Part B headline.**
- **Every penalised and unpenalised block except the salt embedding grows by one to two orders of
  magnitude along the trajectory.** From the AdamW stage to the fiber2 capped state:
  - layer-1 RBF: about 2 → 14–116;
  - layer-2 RBF: about 1 → 13–84;
  - SiLU base: about 1.6 → 17–96;
  - biases: about 0.5 → 4–32.
- **Part A (no L2) keeps them growing:** ‖θ‖² rises a further 1.1–2.7× (ind 4: 30,760 → 82,493).
  Its displacement is carried mostly by the **SiLU base weights** (share 0.08–0.59) and the
  **layer-1 RBF coefficients** (0.28–0.65). The salt embedding takes ≈ 0.
- **Which of these the penalty touches:** the smoothness penalty does not touch the SiLU base weights
  or the biases at all, and for the RBF coefficients it is blind to patterns that are constant or
  linear in the centre index. No weight decay is part of the loss.
- **The salt embedding stays at about 2–8 throughout.**
- **The λ = 1e-5 term is not a gentle regulariser here.** At the capped state, λ‖θ‖² is 4–34 × the
  data loss for the seeds and about 230 × for the students. Part C therefore shrinks ‖θ‖² 40–300×,
  to about 80–110 for the seeds and 12–15 for the students. That explains U4: the loss without L2
  is 1.14–1.25 × capped for the seeds at 1e-5, and 2–5 × for the students.

### Close-out note: the capped states are not minima, and are not near-optimal

Trust-exact without any L2 term (Part A) reduced the loss of **every** fiber2 capped state within 500
iterations. The final / capped loss ratios:

| ind 0 | ind 1 | ind 2 | ind 3 | ind 4 | ind 5 |
|---|---|---|---|---|---|
| 0.9580 | 0.9787 | 0.9849 | 0.9463 | 0.9720 | 0.9798 |

- **Seeds:** Part A lowered the loss by 1.5–5.4% below the capped value, and was still moving.
- **Students:** stu 115 0.8120, stu 104 0.8516, stu 114 0.7135, so their
  loss fell 15–29% below the capped value.
- **Trajectories:** each continued to lower loss with growing parameter norm. None had attained a
  minimum after 500 exact-Hessian iterations.
- **Earlier comparisons:** every earlier comparison between "capped" models (fiber2, support, rank,
  trust) is between points on unfinished, norm-growing trajectories, not between minima.

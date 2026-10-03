# Pre-registration: synthetic teacher-student test of the runaway mechanism (paper2/synth)

Written and committed before `paper2/synth.py` existed (2026-10-03). It combines the original synth
brief, the approved Step 0 changes and the 2 × 3 amendment. Verdicts are appended below the "Verdicts"
heading after the run; nothing above it is edited afterwards.

## History of Step 0 (stated before any student was trained)
The brief's original Step 0 ratio criteria were found **infeasible** in a scratch feasibility check.
That check fitted only the teacher: no student was trained, and it produced no results.
- **Teacher penalty / σ² was 0.79–2.19,** against < 0.05 required.
- **μQ(teacher) / σ² was ≥ 0.06 even at μ = 1e-5,** against ≤ 0.01 required.
- **Both ratios are invariant to the teacher's amplitude,** because σ = 0.05 × std(output).
- **The two criteria pull against each other:** a low penalty needs nearly affine edges, but affine
  edges are exactly what Q penalises.
- **Clipping N(0,1) to the grid range [−1, 1]** would put about 32% of every input exactly on the
  boundary.

They are replaced by the function-space criteria below.

## Design
- **Teacher and students: KAN [4, 3, 1].**
  - Two stacked `kan.KANLayer`s with tanh between them, as in `kan.KAN`, but **without** the salt
    embedding that `kan.KAN` always adds.
  - Same edge type: SiLU base plus 8 Gaussian RBFs on fixed centres linspace(−1, 1, 8), width 2/7.
- **Penalty levels** (λ = 1e-3 throughout, as in final.py; e runs over all 15 edges):
  - **P0, "D2 only":** λ · Σ_e ‖D2 c_e‖². This is the final.py smoothness penalty.
  - **P1, "curvature penalty, Bagrow-style":** λ · Σ_e (‖D2 c_e‖² + K_silu · α_e²).
    - α_e is the edge's SiLU base weight.
    - K_silu = (30 + π²)/90 = 0.4430, the integral of SiLU″(z)² over the real line (Bagrow 2026,
      arXiv:2605.02190, Eq. 5). Checked numerically: 0.44300; over [−1, 1] it would be 0.373.
    - Cross terms between SiLU and RBF are dropped, as in Bagrow, and one weight λ covers both terms.
    - **This keeps our D2 penalty on the RBF coefficients** (Bagrow's Appendix B uses the RBF
      curvature Gram matrix instead), so that P0 → P1 changes exactly one thing, the SiLU term.
      P1 is Bagrow-*style*, not a replication.
  - **P2, "null space closed":** P1 + μ · Q, with Q = Σ_e ‖P_null c_e‖².
    - P_null projects onto span{1, k}, k = 0..7: the constant and linear patterns in the centre index.
    - Q contains **only** the RBF null-space part, because the SiLU weight is already penalised in P1.
      (This differs from close.py's Q, which also had squared SiLU weights.)
- **Cells:** {independent, correlated} × {P0, P1, P2}, with 8 seeds per cell (0–7), so 48 models.
  - Every cell uses its own penalty in **both** the AdamW and the trust-exact stages. **Unlike
    close.py,** where μQ entered only the polishing stage, P2 uses μQ in both.

## Data
- **Inputs:** x = clip(z / 3, −1, 1), with z standard normal. About 0.3% of values are clipped.
  - **Correlated:** corr(z1, z2) = corr(z3, z4) = 0.95, applied **before** scaling and clipping. The
    other pairs are independent. The realised correlations are reported.
  - **Independent:** all z_i independent.
- **Rows per condition:** 2000 training, **2000 validation** (added: early stopping needs a set separate
  from the test rows), 2000 in-distribution test, and 2000 **off-manifold** rows (all four z_i
  independent, same marginals, same transform). Separate fixed seeds are used for each split and
  condition.
- **Targets:** teacher output + Gaussian noise with σ = 0.05 × std(teacher output on that condition's
  training rows).

## Step 0: teacher, acceptance and μ calibration (frozen before Step 1; no verdicts)
- **Teacher edges** (biases 0):
  - Unit 1: φ₁ = 0.8 sin(1.5x), φ₂ = 0.5x², φ₃ = 0.6 tanh(2x), φ₄ = −0.4x.
  - Unit 2: φ₁ = 0.5x², φ₂ = −0.7 sin(1.2x), φ₃ = 0.4x, φ₄ = 0.6 tanh(1.5x).
  - Unit 3: φ₁ = −0.5 tanh(2x), φ₂ = 0.4x, φ₃ = 0.6 sin(1.5x), φ₄ = −0.5x².
  - Layer 2: ψ₁(u) = u, ψ₂(u) = 0.8 sin(1.5u), ψ₃(u) = 0.6u².
- **Edge fit:** each edge is fitted into its basis by **smoothness-penalised least squares** on 401
  points over [−1, 1]: min (1/401)‖Bc − f‖² + 1e-3 · (‖D2 c_rbf‖² + K_silu · α²).
- **Teacher acceptance (function space):** polish the teacher with trust-exact under P0 and under P1, for
  both input conditions, on that condition's training targets (cap 500 iterations).
  - It passes if, for all 4 polishes, the RMS change in predictions on the training rows at the end is
    < 0.5σ. Norm growth is logged descriptively.
  - If it fails: multiply every argument frequency (the 1.5, 1.2, 2 and the curvature of x²) by 0.8,
    refit, and repeat, up to 10 rounds. If it still fails, stop and report.
- **μ calibration:** go through {1e-6, 1e-5, 1e-4, 1e-3} in ascending order. The **smallest** μ is chosen
  that passes both:
  - (i) the regularised teacher (trust-exact under P2, both conditions) stays within 0.5σ (RMS
    prediction change) of the teacher **and** is certified;
  - (ii) an 8-seed pilot (seeds 100–107, correlated-P2, same pipeline as Step 1) shows no norm growth
    beyond 3×, meaning **max** over the pilot seeds of ‖θ‖²_polished / ‖θ‖²_early ≤ 3.
  - The pilot runs are calibration only, never results.
  - If no μ passes: stop, report, and commit the calibration log; all predictions are then inconclusive.
- **Frozen and logged:** the teacher's coefficients, the penalty and Q values, the data seeds and μ.

## Step 1: training
- **Per model:**
  - **"early":** AdamW (lr 3e-3, weight decay 1e-4, full batch, as final.py) on the cell's loss.
    Early stopping on the validation **data term** (MSE), with patience 100 epochs and at most 10,000
    epochs; the best-validation state is restored. kan.py's initialisation is used with
    `torch.manual_seed(seed)`.
  - **"polished":** trust-exact from the early state, in float64, cap 500 iterations, on the cell's loss.
    The callback is named `intermediate_result`, with a 2 h safety cap per run.
- **Logged at both states:** data term, penalty term(s), μQ, ‖θ‖² per block (layer-1 RBF, layer-2 RBF,
  SiLU base, biases), the median per-unit cross-edge cancellation ratio
  (Σ_i RMS(edge_ij) / RMS(Σ_i edge_ij) over the 4 incoming edges, on the training rows), the saturated
  fraction (|s_j| > 3), max|grad|, and an eigenvalue summary (min, max, number negative < −1e-8 × max,
  null-set size, deciles).
- **CERTIFIED** (the certify.py definition, on the cell's loss): max|grad| < 1e-7; smallest eigenvalue
  ≥ −1e-8 × largest; Newton decrement on the non-null eigenspace < 1e-16; ‖g_null‖ < 1e-9.

## Step 2: identifiability and extrapolation
- **Reference per cell:** the teacher polished with trust-exact under that cell's penalty and input
  condition, if that polish is **certified**; otherwise the original teacher. The reference used is
  logged per cell.
  - The P0 and P1 polishes are the Step 0 acceptance polishes; P2 uses the frozen-μ regularised teacher.
  - A P2 cell without a certified regularised teacher means calibration failed: stop and report.
- **Polished models:**
  - prediction correlation with the reference;
  - **unit match:** Hungarian on |corr| of unit contributions over the training rows, median over
    3 units × 8 seeds;
  - **edge match:** per-edge |corr| of gauge-fixed layer-1 edges on a 100-point grid, over
    reference-active edges (≥ 5% of the largest edge variance in the unit), as in control 3;
  - d_AB between the 28 seed pairs within a cell: RMS(Δpred) / std(teacher output, training rows).
  - Gauge fixing (steps 1–4 as in `gauge.py`) is re-implemented for the embedding-free network.
- **RMSE against the noiseless original teacher,** on the in-distribution test rows and the
  off-manifold rows, at the early and polished states.

## Predictions
- **X1. Runaway.** Median ‖θ‖² polished/early ≥ 10 in correlated-P0 **and** correlated-P1; ≤ 3 in all
  three independent cells **and** in correlated-P2.
- **X2. Certified.** ≤ 2 of 8 in correlated-P0 and in correlated-P1; ≥ 6 of 8 in each of the other four
  cells.
- **X3. Cancellation.** Median cross-edge cancellation (polished) in correlated-P0 ≥ 2 × correlated-P2,
  **and** in correlated-P1 ≥ 2 × correlated-P2.
- **X4. Extrapolation.** Median off-manifold RMSE polished/early ≥ 3 in correlated-P0 and in
  correlated-P1; ≤ 1.5 in correlated-P2.
- **X5. Cancellation vs extrapolation.** Across the 24 polished correlated models, off-manifold RMSE
  increases with cross-edge cancellation: Spearman ρ ≥ 0.6. ρ and p are reported.
- **X6. Recoverability.** Median unit match to the reference ≥ 0.85 in all three independent cells and
  in correlated-P2; < 0.7 in correlated-P0 and in correlated-P1.

## Interpretation (fixed in advance)
- **X1–X3 hold:** the runaway needs correlated inputs plus an unpenalised affine null space. The
  published-style curvature penalty (P1) does not prevent it; closing the RBF null space (P2) does.
- **Correlated-P0 runs away but correlated-P1 does not:** penalising the SiLU weight is sufficient, and
  the RBF affine null space is harmless in this setting.
- **Any independent cell runs away:** correlated inputs are not necessary.
- **Correlated-P2 runs away:** closing the null space is not sufficient.
- **X4 and X5 hold:** the runaway drives extrapolation failure off the training correlation pattern.
  **X4 holds, X5 fails:** extrapolation degrades with polishing, but not in proportion to cancellation.
- **X6 holds:** without the runaway, a known solution's internals are recoverable. **X6 fails:**
  non-identifiability persists beyond the runaway.
- **Any other combination:** reported as found, without forcing it into a branch.

## Outputs
`paper2/synth.py`, `paper2/synth_output.txt`, `paper2/synth_results.json`, and `paper2/figs/`:
- a figure of ‖θ‖² trajectories per cell, over AdamW epochs and trust-exact iterations;
- a figure of off-manifold RMSE against cancellation.

Weights go in `gauge_weights/synth_*` (git-ignored) and are not committed. The data are generated in
memory from fixed seeds and nothing is re-read from disk.

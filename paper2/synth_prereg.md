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

---

## Verdicts (appended after the run; `paper2/synth_output.txt`, `paper2/synth_results.json`, `paper2/figs/`)

**Step 0 (calibration, no verdicts).**
- **Teacher:** accepted in round 0 (frequency scale 1.0). Its D2 term is 8.0e-4, its SiLU curvature
  term 4.0e-4, and Q 5.67. σ is 2.86e-2 (independent) and 2.47e-2 (correlated).
- **Acceptance polishes:** the RMS prediction change was 7.7e-3 – 8.9e-3, all under 0.5σ.
  - Independent P0 and P1: certified, with ‖θ‖² growth of ×7.5 and ×5.9.
  - Correlated P0 and P1: **not certified** (iteration cap), with ‖θ‖² growth of **×85 and ×1069**,
    although the function stayed within 0.5σ.
- **References, by the pre-registered rule:**
  - independent P0 and P1: the certified polished teacher;
  - correlated P0 and P1: the original teacher;
  - P2 cells: the certified regularised teacher.
- **μ calibration:**
  - μ = 1e-6: the regularised teacher passed, but the pilot's maximum growth was ×5.03, so it **failed**.
  - **μ = 1e-5: passed** (regularised teacher certified, within 7.2e-3 – 8.6e-3; pilot maximum growth
    ×1.32). **Frozen at μ = 1e-5.** The pilots are calibration only.

**Cell summary (Step 1 and Step 2; medians over 8 seeds):**

| Cell | ‖θ‖² polished/early | Certified | Cross-edge (polished) | Off-manifold RMSE polished/early | Unit match | Edge match | d_AB | Reference |
|---|---|---|---|---|---|---|---|---|
| independent P0 | 12.49 | 6/8 | 1.60 | 1.20 | 1.000 | 1.000 | 0.0035 | polished teacher (certified) |
| independent P1 | 7.89 | 6/8 | 2.72 | 0.97 | 1.000 | 1.000 | 0.0046 | polished teacher (certified) |
| independent P2 | 1.09 | 8/8 | 1.88 | 0.95 | 1.000 | 1.000 | 0.0001 | regularised teacher (certified) |
| correlated P0 | **65.71** | **0/8** | 3.29 | 1.22 | 0.774 | 0.953 | 0.0129 | original teacher |
| correlated P1 | **32.13** | **0/8** | 2.32 | 0.77 | 0.883 | 0.980 | 0.0134 | original teacher |
| correlated P2 | 1.00 | 8/8 | 1.99 | 0.78 | 1.000 | 1.000 | 0.0001 | regularised teacher (certified) |

| Prediction | Verdict | Key number |
|---|---|---|
| **X1** runaway (≥ 10) in correlated P0 and P1; ≤ 3 in the three independent cells and correlated P2 | **failed** | Correlated P0 ×65.7 and P1 ×32.1, and P2 ×1.00, as predicted. **But independent P0 ×12.5 and P1 ×7.9 exceed 3** (independent P2 ×1.09) |
| **X2** certified ≤ 2/8 in correlated P0 and P1; ≥ 6/8 elsewhere | **held** | correlated P0 0/8, P1 0/8; independent 6/8, 6/8, 8/8; correlated P2 8/8 |
| **X3** cancellation in correlated P0 and P1 each ≥ 2 × correlated P2 | **failed** | P0 3.29 and P1 2.32, against 2 × 1.99 = 3.98 |
| **X4** off-manifold ratio ≥ 3 in correlated P0 and P1, ≤ 1.5 in P2 | **failed** | P0 1.22, P1 **0.77**; P2 0.78 (the P2 part held) |
| **X5** Spearman(cancellation, off-manifold RMSE) ≥ 0.6 over the 24 correlated models | **failed** | ρ = +0.177, p = 0.407 |
| **X6** unit match ≥ 0.85 in the independent cells and correlated P2; < 0.7 in correlated P0 and P1 | **failed** | The first part held: 1.000 in all four no-runaway-predicted cells. **The second part failed: correlated P0 0.774 and P1 0.883 are not < 0.7** |

Commits: prereg `dfd281d` → results (the commit that adds this section).

**Interpretation, by the pre-registered rules:**
- **"X1–X3 hold"** does not apply.
- **"Correlated P0 runs away but correlated P1 does not"** does not apply: both run away (0/8 certified,
  ×66 and ×32).
- **"Any independent cell runs away"** applies formally. By X1's own runaway threshold (≥ 10),
  independent P0 (×12.5) qualifies, so by the rule **correlated inputs are not necessary.**
  - Caveat: 6 of 8 independent-P0 models grow to **certified** finite minima (×7–15), and only seeds
    0 and 4 grow without bound (×703 and ×170). In the correlated P0 and P1 cells, 0 of 16 are
    certified.
- **"Correlated P2 runs away"** does not apply: ×1.00, 8/8 certified. Closing the RBF null space
  prevented growth in both conditions.
- **X4 and X5 both failed,** so neither extrapolation rule applies, and this is reported as found.
  **In this setting, polishing did not degrade off-manifold extrapolation**: median ratios 0.77–1.22,
  and correlated P1 improved.
- **X6's rule maps failure to "non-identifiability persists beyond the runaway". The data contradict that
  reading.** X6 failed only because the *runaway* cells matched the original teacher *better* than
  predicted (0.774 and 0.883 against < 0.7). Every cell without a runaway recovered the reference's
  internals exactly (unit and edge match 1.000, d_AB about 1e-4). The verdict stays **failed**, as
  recorded.

Descriptive notes (not verdicts):
- **The P2 cells are fully identifiable here.** All seeds land on the same certified minimum as the
  regularised teacher (d_AB about 1e-4). In the real electrolyte data, certified seeds still differed
  (certify W2).
- **Under correlated inputs, the early-stopped models already extrapolate several times worse** than
  under independent inputs: off-manifold RMSE about 0.05–0.17, against about 0.01. Most of the
  off-manifold error is in place before any polishing.

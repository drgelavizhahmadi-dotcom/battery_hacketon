# Pre-registration: rank-constrained layer-1 coefficients (rank)

Written and committed before `rank.py` existed (2026-09-29). Verdicts are appended below the
"Verdicts" heading after the run; nothing above it is edited afterwards. Thresholds are the brief's,
unchanged.

## Discrepancies and operational choices (files take precedence)
- **X1. I5's inputs are affected in arm F.** Arm F constrains all 72 columns of M, including the
  1000/T and molality edges. The brief's "inputs outside M's affected blocks in both arms" holds
  only for arm G. I5 is computed as specified for both arms, and the arm F result is read with
  this in mind.
- **X2. Expected dimensions,** from `support_output.txt` / `rule_output.txt`: the co-solvent block has
  rank 14 of 27, and M has 36 singular values with relative value ≥ 1e-2. Any difference is flagged.
- **X3. Gradient coordinates.** In the arms, the affected coefficients are parametrised by z, so
  ∂L/∂z = Pᵀ ∂L/∂θ_affected. By construction this leaves out the gradient along the removed
  directions, which is part of the intervention, so I1 compares different coordinate systems.
  - **I1 uses max|grad| over each model's own parameters** (z plus everything else; unused salt
    embeddings have zero gradient and don't affect the max).
  - Also reported, not used for verdicts: the gradient split into z vs other, and the full θ-space
    gradient at the arm's solution.
- **X4. Weight decay.** AdamW's decoupled weight decay acts on z instead of on the affected θ
  entries. Because P is orthonormal, ‖Pz‖ = ‖z‖, so the penalty is equivalent within the retained
  subspace. The removed directions stay exactly 0.
- **X5. Initialisation.** torch.manual_seed(seed) and then `KAN(...)`, exactly as `kan.train`. Then
  z = Pᵀθ_init for each unit. So the effective initial coefficients are PPᵀθ_init: the components
  along removed directions start at 0, not at their initial values.
- **X6. Hessian parameters:** all parameters that affect the loss. The 16 embedding parameters of
  salts absent from the training rows are excluded, as in support D2. I report the 10
  algebraically smallest eigenvalues and the 10 smallest |λ|.
- **X7. The KAN-delta fold A reference (0.142)** is recomputed here with the same code (seeds 0–4,
  107 epochs, ensemble mean). Any difference is flagged.
- **X8. "Layer-1"** means the input → hidden layer (`layers.0` in `kan.py`), as before.

## Step 0: subspaces (no training)
M = `support.design_matrix` on all 2069 anchored training rows: 72 columns, input-major, [SiLU,
RBF×8] per continuous input.
- **Arm G:** the SVD of the co-solvent block (the 27 columns of eps_co, lneta_co, M_co).
  P_G = the right singular vectors with σ/σ_max ≥ 1e-2 (expected 14). The other 45 columns and
  their coefficients are unchanged.
- **Arm F:** the SVD of all of M. P_F = the right singular vectors with σ/σ_max ≥ 1e-2 (expected 36).
- Report the retained dimensions and flag any difference from 14 and 36.

## Step 1: constrained models
- **Constraint:** for every hidden unit j, the layer-1 coefficient vector on the affected columns
  (input-major [base weight, 8 RBF coefficients] per affected input) is θ_j = P z_j, with z_j
  learnable.
- **Everything else as in `final.py`:** layer-1 bias, unaffected edges, salt-embedding edges, the
  embedding, layer 2, features, loss (MSE on delta + 1e-3 · smoothness penalty, applied to the
  effective coefficients c = Pz), AdamW lr 3e-3 / wd 1e-4, 107 epochs, full batch.
- **Seeds:** 0–5 per arm.
- **Equivalence check:** each trained model is converted to an ordinary `KAN` with the same effective
  weights, and predictions must agree to ≤ 1e-5. All analysis uses that converted form.
- **Polishing:** exactly as `fiber2.certify` (float64 L-BFGS, strong Wolfe, history 100, 50-iteration
  steps, cap 20000; converged when max|grad| < 1e-7 **and** the relative change over 50 iterations is
  < 1e-12), optimising z and the other parameters.
- **Report per seed:** converged or not, max|grad| at the cap (overall, z, other), the loss change
  per 50 iterations, and the final loss.
- **Weights:** a third tarball, `../battery_hacketon_weights/gauge_weights_2026-09-29c.tar.gz`,
  appended to `weights_manifest.txt`.

## Step 2: comparison (baseline = `fiber2_ind_{0..5}.pt`), identical code for all three
- **max|grad| and final loss:** recomputed here in float64 at each capped model.
- **d_AB on delta:** for all 15 seed pairs, RMS(delta_A − delta_B) / std of the data target delta.
- **Edge correlation on 1000/T and molality:**
  - Match units Hungarian on contribution |corr| over the training rows.
  - Shared scale: a is fitted in function space, min ‖Mθ_B − a·Mθ_A − c‖² over the 72 continuous
    columns.
  - Compute the corr of the centred edges on a 100-point grid over the training range, with the
    sign set by sign(a).
  - Take the median over matched unit pairs × the 2 inputs × 15 seed pairs.
- **Hessian:** per X6.

## Step 3: accuracy
For each arm, recompute P on **fold A's training rows only** (anchored rows except PC+EA), with the
same tolerance, and report the dimensions. Then train seeds 0–4 for 107 epochs and take the
ensemble mean. Report RMSE and bias on the 635 anchored PC+EA rows, next to KAN-delta recomputed
the same way (X7).

## Predictions (per arm where stated; baseline medians are computed in this script)
- **I1.** Per arm: median max|grad| at the cap ≤ 0.1 × the baseline median.
- **I2.** Per arm: median d_AB on delta ≤ 0.5 × the baseline median.
- **I3.** SANITY, not a test: per arm, the median final loss ≤ 1.10 × the baseline median.
- **I4.** Per arm: fold A RMSE ≤ 0.150.
- **I5.** Per arm: the median edge correlation on 1000/T and molality ≥ the baseline's + 0.1
  (see X1).
- **I6.** Dose-response: arm F's improvement is at least as large as arm G's. Specifically, the ratio
  median(arm F) / median(baseline) is ≤ the same ratio for arm G, for **both** max|grad| and d_AB.

## Interpretation (fixed in advance)
- **I1 and I2 hold for arm F:** the near-null directions of the layer-1 design matrix cause a large
  part of the slow convergence and seed disagreement. That is causal support for the data-support
  mechanism.
- **They also hold for arm G:** the correlated co-solvent descriptors alone are sufficient.
- **I1 and I2 fail for both arms:** removing the data-invisible directions doesn't fix the
  landscape; the cause lies elsewhere (layer 2, saddle regions).
- **I3 fails for an arm:** the tolerance removed directions the data do see. That arm's I1 and I2
  are reported but treated as uninterpretable.
- **I6 fails while arm G passes I1 and I2:** the co-solvent block is the operative part.
- **Any other combination** is reported as found.

## Step 4: close-out
Replace the "Intervention" section of `electrolyte_case_study_summary.md` with one covering both
interventions:
- the rule (gate stopped, a no-op, and why);
- the rank constraint: the I1–I6 table with verdicts, key numbers and commits;
- one paragraph on what is and is not established.

Commit it.

Outputs: `rank.py`, `rank_output.txt`, `rank_results.json`, `figs/rank/`.

---

## Verdicts (appended after the run; `rank_output.txt`, `rank_results.json`, `figs/rank/rank_comparison.png`)

**Checks.**
- **Step 0:** arm G keeps 14 of 27 directions and arm F keeps 36 of 72, as expected. On fold A's
  training rows, arm G keeps 14 and arm F keeps **33**.
- **Reparametrisation:** RankKAN with P = I reproduces `kan.train` seed 0 exactly (difference 0.0).
- **Equivalence check: FAILED at its pre-registered tolerance, in float32.** The 1e-5 limit was
  exceeded for G2 (1.1e-5), G5 (2.0e-5) and all arm F seeds (1.0–2.7e-4).
  - Post hoc, in float64, the converted KAN and the constrained model agree **exactly** (0.0) for
    all 12 models. The float32 failure is rounding in the two float32 evaluation paths, not a
    conversion error.
  - Gradient, loss and Hessian metrics use the float64 constrained models directly. d_AB and edge
    correlations use the float32 conversions, whose error (≤ 2.7e-4 in delta) is about 1% of the
    d_AB differences (about 0.03).
- **Convergence:** no constrained model certified convergence (arm G 0/6, arm F 0/6; all at the
  20000-iteration cap).
- **KAN-delta fold A reference,** recomputed: 0.1420 (the brief says 0.142).

| Model | Median max\|grad\| (ratio) | Median d_AB (ratio) | Median final loss (ratio) | Edge corr, 1000/T & molality | Fold A RMSE / bias |
|---|---|---|---|---|---|
| Baseline (fiber2 capped) | 8.66e-5 (1) | 0.1041 (1) | 9.162e-3 (1) | 0.102 | 0.1420 / +0.0067 (KAN-delta) |
| Arm G (co-solvent, 14/27) | 3.47e-4 (**4.00**) | 0.0984 (**0.945**) | 9.195e-3 (1.004) | **0.927** | **0.1499** / −0.0439 |
| Arm F (full M, 36/72) | 2.52e-3 (**29.1**) | 0.0842 (**0.808**) | 1.007e-2 (1.099) | 0.338 | 0.1587 / −0.0582 |

| Prediction | Arm G | Arm F |
|---|---|---|
| I1: max\|grad\| ratio ≤ 0.1 | **failed** (4.00) | **failed** (29.1) |
| I2: d_AB ratio ≤ 0.5 | **failed** (0.945) | **failed** (0.808) |
| I3 (sanity): loss ratio ≤ 1.10 | passed (1.004) | passed, narrowly (1.099) |
| I4: fold A RMSE ≤ 0.150 | **held**, narrowly (0.1499) | **failed** (0.1587) |
| I5: edge corr ≥ baseline + 0.1 (0.202) | **held** (0.927) | **held** (0.338; see X1: arm F constrains these edges) |
| **I6:** arm F's ratios ≤ arm G's for both max\|grad\| and d_AB | **failed**: max\|grad\| ratio F 29.1 vs G 4.00. The d_AB ratio does favour F (0.808 vs 0.945) | |

**Interpretation, by the pre-registered rules.** I3 passed for both arms, so I1 and I2 are
interpretable. **I1 and I2 fail for both arms, so removing the data-invisible directions does not
fix the landscape, and the cause lies elsewhere (layer 2, saddle regions).** The rule "I6 fails
while arm G passes I1 and I2" does not apply, because arm G failed them. I5 is secondary: constraining
the co-solvent block made the unconstrained 1000/T and molality edges far more consistent across
seeds (0.102 → 0.927).

Notes (post hoc, not verdicts):
- **The constraint worsens the landscape's conditioning.** The largest Hessian eigenvalue rises from
  3.5e2–4.7e3 (baseline) to 5.5e2–6.3e3 (arm G) and **1.3e5–2.0e6** (arm F). All 18 models keep
  21–39 negative eigenvalues, and every model still has an exact null space (|λ| median about
  1e-14 – 1e-17). The directions P retains are defined in raw coefficient coordinates without
  rescaling, which is a likely reason the constrained models stall with larger gradients.
- **The removed directions are not quite free for accuracy.** The fit barely changes (loss +0.4% for
  G, +9.9% for F), but held-out accuracy degrades (fold A 0.142 → 0.150 for G and 0.159 for F).
  That is consistent with the loss ratio: the retained-subspace tolerance also removes directions
  the data weakly see.
- **d_AB falls only modestly** (−5.5% for G, −19% for F), far from the ≤ 0.5× predicted.
- **The baseline edge correlation (0.102)** is taken over all 6 matched units × 2 inputs, including
  units whose 1000/T or molality edge is nearly flat. It is not comparable to control 3's
  active-edge recovery (0.962 for 1000/T).

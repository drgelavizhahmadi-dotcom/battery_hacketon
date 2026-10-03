# SC-KAN-delta (shared-curve KAN-delta): pre-registration

Committed before `arch/sc_model.py` or `arch/sc_run.py` exists, and before any SC-KAN-delta model has been trained.
Verdicts will be appended below this text; the text itself and every recorded verdict will not be edited.
Files win over this document; discrepancies found while running go at the top of `arch/sc_output.txt`.
Rules: no weights in git (weights go to `gauge_weights/`, archived with `weights_manifest.txt` entries); inputs are held
in memory after SHA verification.

## Flags (recorded before any run)

1. **The β term covers the head weights w as well as the mixing weights A.** Reason: the post-hoc synthetic anatomy
   (commit ea22a56) showed that the runaway can live in the output layer, through cancellation across correlated
   hidden units; in SC-KAN-delta that layer is the head. Decided before any SC-KAN-delta model was trained.
2. **The curves are centred, not standardised (decision C).** The brief's z_i = (g_i − mean)/sd makes each curve's
   scale (α_i, c_i) → s(α_i, c_i), s > 0, leave every prediction unchanged while the penalty scales as s². Any
   minimum would then need α_i = 0 and affine RBF coefficients (near-linear curves), or the scale would shrink
   towards 0 with no minimum. With centring only, a curve's scale trades against its row of A, and the penalties
   fix it. Unit-variance standardisation becomes an analysis convention.
3. **A null-space term on the curves is added:** μ Σ_i ‖P_null c_i‖², where P_null projects onto span{1, k} (the
   constant and linear patterns in the centre index, the null space of D2). μ = 1e-5 is **carried over from the
   synthetic calibration** (paper2, dfd281d / 34cc321), not re-calibrated here. Reason: with centring only, straight
   tilts on the curves of correlated inputs can still cancel when their rows of A are proportional.
4. **Salt enters as tanh(e(salt)), as in final.py**, not as a linear e. A linear W·e + b would leave 6 unpenalised
   directions that change nothing (e → Me + t: 4 from M, 2 from t). The β term covers W and e as well. Reason for e:
   once tanh(e) saturates, e would otherwise have a flat direction to infinity (12 salts).
5. **12 salts, not 4.** The embedding has 24 parameters, so the total is about 169, not 153 (see the parameter
   count below).
6. **The LOCO folds are very uneven.** The anchored training set has 2069 rows and 10 co-solvents. Rows containing
   each co-solvent: EC 1385, EA 635, 2-MeTHF 126, Methylene chloride 96, THF 54, DME 47, Toluene 20, 3-Glyme 15,
   Sulfolane 12, 2-Glyme 5.
   - Rule: fold c's test set is every anchored training row that contains c, and its training set is every anchored
     row that does not. Rows with several co-solvents therefore sit in several test sets.
   - LOCO RMSE is reported three ways: the unweighted mean over the 10 folds (the S1 statistic, as in the brief), the
     row-weighted mean, and the unweighted mean over folds with ≥ 20 test rows. The last excludes 2-Glyme,
     Sulfolane and 3-Glyme, leaving 7 folds.
7. **Fold A coincides with the LOCO EA fold.** final.py's fold A holds out the EA+PC combination: 635 anchored rows,
   the same rows as the EA fold. It is kept as the secondary split, as the brief says.
8. **S6 is partly circular.** The Step 0 β rule uses the same ratio on the same fold A training rows. To reduce the
   overlap, the pilot uses seeds 100–102, disjoint from the Step 1 seeds 0–9. S6 is labelled "calibrated" in the
   verdict table.
9. **Reference number for S2:** the baseline KAN-delta's fold A RMSE is 0.1420 (kan.py, fixed epoch). The S2
   threshold is 0.160.
10. **The baseline is polished on its own loss** (final.py: MSE + 1e-3·ΣD2), without the K_silu, μ or β terms. Its
    model and training are otherwise unchanged.
11. **Feature names.** The brief's names map to final.py's `DELTA_IN` as: x_co→x_co, eps_co→eps_co,
    ln_eta_co→lneta_co, M_co→M_co, eps_mix→mix_eps, ln_eta_mix→mix_lneta, 1000/T→invT, molality→molal. Scaling is
    final.py's `Scaler` (min–max to [−1, 1] over the anchored train and test rows), identical for every fold.

## Model: SC-KAN-delta

- **Inputs:** the 8 continuous final.py features (flag 11), scaled exactly as in final.py. Salt enters through a
  learned 2-number embedding e(salt), passed through tanh.
- **1. Shared curves, one per input:** g_i(x_i) = α_i SiLU(x_i) + Σ_k c_ik B_k(x_i), for k = 1..8 Gaussian RBFs.
  The centres (linspace(−1, 1, 8)) and the width (2/7) are as in kan.py / final.py.
- **2. Centring (gauge fix, decision C):** z_i = g_i − mean_train(g_i).
  - The mean is taken over the current training rows and recomputed at every full-batch step; it is
    differentiable.
  - At prediction time the training-row mean is used.
- **3. Mixing into 6 hidden units:** u_j = Σ_i A_ij z_i + Σ_m W_jm tanh(e_m(salt)) + b_j, for j = 1..6. The hidden
  activation is tanh(u_j); A is the 8 × 6 mixing matrix.
- **4. Linear head:** h = Σ_j w_j tanh(u_j) + b0, where w is the 6-vector of head weights.
- **5. Physics wrapper (unchanged):** δ = x_co · h, and log k = anchor + δ.
- **Parameter count:**

  | Block | Parameters |
  |---|---|
  | Curves (8 × (1 + 8)) | 72 |
  | A | 48 |
  | Embedding e (12 × 2) | 24 |
  | W (6 × 2) | 12 |
  | b | 6 |
  | w | 6 |
  | b0 | 1 |
  | **Total** | **169** |

## Loss

Loss = data MSE
     + λ Σ_i (‖D2 c_i‖² + K_silu α_i²)
     + μ Σ_i ‖P_null c_i‖²
     + β (‖A‖² + ‖w‖² + ‖W‖² + ‖e‖²)

- λ = 1e-3 (as final.py), K_silu = 0.4430, μ = 1e-5 (flag 3), and β from the Step 0 rule.
- All terms are part of the loss, not AdamW's decoupled weight decay, so they also act during trust-exact polishing.
- AdamW's own weight decay (1e-4, final.py) applies to all parameters during AdamW training only.

## Conventions (analysis only, never applied during training)

- **Hidden-unit sign:** if w_j < 0, flip the signs of w_j, of column j of A (A_.j), of row j of W (W_j.) and of b_j.
- **Curve sign:** orient each z_i so it increases from its input's minimum to its maximum over the training rows. If
  z_i is flipped, flip row i of A (A_i.) as well.
- **Curve scale (decision C):** for comparing across seeds, report z_i / sd_train(z_i) and multiply row i of A by
  sd_train(z_i).
- **Hidden-unit matching:** hidden units are matched across seeds (Hungarian on contribution correlation).

## Step 0: checks and calibration

Runs after this prereg commit; nothing in Step 0 is a result.

- **a) Unit tests:**
  - a finite-difference check of the gradients through the centring;
  - the parameter count (169);
  - the conventions leave predictions unchanged to 1e-12.
- **b) LOCO split:** list the co-solvents and their row counts (flag 6), and build the fold masks by the flag-6 rule.
  Fold A, as in final.py, is kept as the secondary split.
- **c) β rule:**
  - Grid: {1e-5, 1e-4, 1e-3, 1e-2}.
  - Pilot on fold A **training** rows only, 3 seeds per β (100, 101, 102; flag 8).
  - Each pilot model: AdamW exactly as final.py (lr 3e-3, wd 1e-4, 107 full-batch epochs, no early stopping), with
    the SC loss above; then a trust-exact polish (float64, exact Hessian, cap 500 iterations).
  - Choose the smallest β with median (‖A‖² + ‖w‖² + ‖W‖² + ‖e‖²) polished/early ≤ 3.
  - If no β passes: stop and report.
- Log the frozen β and all Step 0 outputs, and commit them before Step 1. Pilot models are never reported as results.

## Step 1: training (both models, identical protocol)

- **Models:** SC-KAN-delta, and the baseline KAN-delta (final.py, unchanged).
- **Training:** for each of the 10 LOCO folds and for fold A, 10 seeds (0–9) of AdamW exactly as final.py (107
  epochs). The result is the "early" state.
- **Seed use:** accuracy uses the 5-seed ensemble of seeds 0–4 (the submission protocol); identifiability uses all
  10 seeds.
- **Fold A only:**
  - polish every model with trust-exact (float64, cap 500 iterations);
  - certify it with the four-part second-order criterion from certify.py (max|grad| < 1e-7; smallest eigenvalue
    ≥ −1e-8 × largest; Newton decrement on the non-null eigenspace < 1e-16; ‖g_null‖ < 1e-9). The null space is
    built from the model's exact continuous gauge directions; for SC, none is expected beyond the discrete sign
    symmetries, and any found in Step 0 is logged;
  - record the fold A RMSE of the polished ensemble.

## Step 2: measurements

- **Accuracy:**
  - LOCO RMSE for both models: unweighted mean over the folds, row-weighted mean, and the mean over folds with ≥ 20
    test rows (flag 6);
  - fold A RMSE for both models.
- **Seed agreement:** median d_AB on δ over seed pairs (fold A, early state), where d_AB = RMS(Δpred)/std over the
  fold A training rows.
- **Curve agreement** (SC only; fold A; early): per input, after the conventions, the median pairwise correlation of
  z_i over a 200-point grid on its training range.
- **Physics** (SC only; fold A; early; per seed): marginal effects on log k over the training rows, by finite
  differences:
  - d log k / d(1000/T);
  - d log k / d(ln η_mix).
- **Stability under polishing** (fold A):
  - (‖A‖² + ‖w‖² + ‖W‖² + ‖e‖²) polished/early and ‖θ‖² polished/early;
  - certified counts;
  - polished/early fold A RMSE.

## Pre-registered predictions

- **S1. Non-inferiority:** SC LOCO RMSE ≤ 1.05 × baseline LOCO RMSE (unweighted mean over the 10 folds).
- **S2.** SC fold A RMSE ≤ 0.160.
- **S3. Seed agreement:** SC median d_AB ≤ 0.5 × baseline median d_AB.
- **S4. Curve agreement:** median curve correlation ≥ 0.90 for at least 6 of the 8 inputs.
- **S5. Convergence:** ≥ 8 of 10 SC fold A models certified after polishing. The baseline is reported, with no
  threshold.
- **S6. Bounded (calibrated, flag 8):** median (‖A‖² + ‖w‖² + ‖W‖² + ‖e‖²) polished/early ≤ 3 for SC.
- **S7. Extrapolation survives polishing:** SC polished/early fold A RMSE ≤ 1.5, with the baseline reported
  alongside.
- **S8. Temperature:** d log k / d(1000/T) < 0 on ≥ 95% of training rows, in ≥ 8 of 10 seeds.
- **S9. VTF curvature:** |d log k / d(1000/T)| increases with 1000/T (Spearman ρ > 0 over the training rows), in
  ≥ 8 of 10 seeds.
- **S10. Viscosity:** d log k / d(ln η_mix) < 0 on ≥ 90% of training rows, in ≥ 8 of 10 seeds.

## Interpretation (fixed in advance)

- **S1 and S3–S6 hold:** shared curves give an identifiable, stable KAN at no material cost in accuracy.
- **S1 fails, S3–S6 hold:** identifiability is bought with accuracy; report the cost.
- **S3 or S4 fails:** sharing does not pin the curves down on correlated inputs (concurvity persists). The next
  variant is a covariance penalty on the z_i.
- **S6 fails:** a runaway route remains despite the β term. Report where the growth lives, using the same breakdown
  as paper2/anatomy_synth.py.
- **S7 holds and the baseline fails it:** removing the runaway routes makes extended training safe for
  extrapolation.
- **S8–S10 hold:** the curves are consistent with Arrhenius/VTF temperature behaviour and Walden-type viscosity
  behaviour. **If they fail:** the curves are not physically readable as they stand, however stable they are.

## End

- A verdict table for S1–S10, with the key numbers and commits.
- A figure of the 8 curves (10 seeds overlaid, after the conventions).
- A figure of the temperature marginal effect against 1000/T.
- Do not start further stages.

---
## Verdicts

(appended after Step 2; nothing above this line will be edited)

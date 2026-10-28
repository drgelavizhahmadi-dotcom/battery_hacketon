# Review 4 analysis bundle (POST-HOC, DESCRIPTIVE)

Committed before `paper2/revision/review4.py` exists. Everything here is POST-HOC: defined after every result it touches is
known. No training, no optimisation, no retuning, no new folds or models; no verdict changes; nothing from this bundle goes
into the abstract or the contribution list. Archived weights only, SHA-verified against `weights_manifest.txt` and held in
memory. Files win over this plan; discrepancies go at the top of `paper2/revision/review4_output.txt`. Results:
`paper2/revision/review4_results.json`.

## Models (all archived; integrity checks must pass or the run stops)

- **Fold A, early:** `scbase_A_s0–4` (107 AdamW epochs on fold A's training rows).
- **Fold A, polished μ = 0:** `scbase_A_s0–4_pol`, mapped with `trust.Flat(early, …).to_kan`.
- **Fold A, polished μ = 1e-6:** `close_foldA_1e-06_s0–4`; **μ = 1e-4:** `certify_foldA_0.0001_s0–4`; mapped with
  `close.FlatQ(early, …, μ).to_kan`.
- **Integrity:** the 5-seed ensembles reproduce the logged fold A RMSEs 0.1420 / 0.8147 / 1.0262 / 0.3345 (to 1e-4).
- **All rows, early-stopped:** `delta_s0–5` (gauge.py; the starts of the certified polish).
- **All rows, certified μ = 1e-4:** `close_0.0001_ind_0–5`, mapped with `close.FlatQ(delta_s{i}, all anchored rows, used, 1e-4)`,
  as in `certify.py`. Integrity: certify_point re-certifies all six; their pairwise d_AB reproduces the committed median 0.0771.
- **Rows:** fold A training rows (1434) and fold A rows (635) for items 1–2; all 2069 anchored training rows for item 3.

## 1. Attribution of the fold A damage (μ = 0 and μ = 1e-6)

- **Parameter change:** Δ = polished − early, per tensor of `kan.KAN` (the unused salt-embedding rows are copied from the early
  net, so their Δ is zero).
- **Blind part:** for every edge e, P Δc_e, with P = QQᵀ and Q the orthonormal basis of span{1, k} over the 8 centre indices (the
  `close.py` projector), applied edge-wise in both layers; plus the full change of every SiLU weight (`base.weight`, both layers).
- **Remainder:** Δ minus the blind part: the curved RBF change (I − P)Δc_e, the biases (`base.bias`), the salt embedding.
  Check: early + blind + remainder = polished exactly (max abs difference < 1e-10).
- **States evaluated:** early; early + blind; early + remainder; polished. For each: fold A RMSE and training RMSE, both in log k
  units (anchor + x_co·h against log k), per seed and for the 5-seed ensemble (mean prediction; primary).
- **Fractions** (on each RMSE, ensemble primary): f_blind = (R_blind − R_early)/(R_polished − R_early), f_rem likewise,
  interaction = 1 − f_blind − f_rem. If |interaction| > 0.25 the output says "the effect is not additive".
- **Reading of the question** (fixed wording): "the blind-direction change is nearly free on training rows and costly on fold A"
  if, for the ensemble, early + blind changes the training RMSE by less than 0.01 in absolute value AND raises fold A RMSE by
  more than half of the total damage (f_blind > 0.5 on fold A). Otherwise the numbers are reported as found.

## 2. Physics consistency (fold A ensembles: early, μ = 0, μ = 1e-6, μ = 1e-4)

- **Derivatives per row, all other inputs held at the row's observed values (float64):**
  - **d log k / d(1000/T)** = d anchor / d(1000/T) + x_co · ∂h/∂(1000/T).
    - Model part by autograd on the scaled input `invT`, times the scaler factor 2/(hi − lo) (> 0).
    - Anchor part from the committed anchor series (`anchor.build_series`, same article and salt, the row's concentration): the
      anchor is constant within ±0.3 K of a grid level, so its slope is the secant to the adjacent grid levels in 1000/T
      (mean of the two secants where both exist; one-sided at the ends of the series); rows with no usable neighbour are
      excluded and counted.
  - **d log k / d ln η_mix** = x_co · ∂h/∂(mix_lneta) · 2/(hi − lo) by autograd (the anchor does not depend on the mixture
    viscosity; `mix_lneta` is the mole-weighted log viscosity at 25 °C, so this is a partial derivative at fixed T).
  - **Signs:** the inputs are min–max scaled, a positive affine map per column, so derivative signs in scaled and physical
    units are the same. Ensemble derivative = mean of the 5 seed derivatives.
  - **Safeguard:** autograd is checked against central finite differences (step 1e-4 in scaled units) on 50 random rows per
    row set and state; the maximum relative disagreement is reported.
- **Expected sign:** negative for both. A row **violates** if the derivative is > 1e-9.
- **Reported:** per state and row set, the share of violating rows (ensemble; per-seed range beside it), the median derivative,
  and the shares by x_co tercile and by temperature tercile (terciles from np.quantile at 1/3 and 2/3 within each row set; the
  tercile row counts are reported because x_co takes few values). For d/d(1000/T), the model part alone is reported beside
  the total (no expected sign for the correction alone).
- This is a consistency check, not a test of a transport law.

## 3. Residual disagreement at certified minima

- **Seeds:** the six certified μ = 1e-4 all-rows seeds and, for contrast, the six early-stopped seeds; all 15 pairs each.
- **Unit matching (committed rule):** `gauge.Gauge` and `gauge.align` on the 2069 training rows: Hungarian assignment on
  |corr| of unit contributions ψ_j(s_j) − mean; unit sign from corr(standardised s_j, ψ_j); edges gauge-fixed as
  sign · (φ − mean) / sd(s_j). Seed A of each pair is the reference.
- **Edge correlation,** per pair, input i (the 8 continuous inputs; the 2 salt-embedding coordinates reported separately) and
  matched unit j: Pearson correlation over the training rows of the gauge-fixed φ_ij of the two seeds.
  - **Version 2:** the same after also removing each edge's blind-direction component: φ_ij minus its SiLU term and minus its
    RBF part with the null-projected coefficients P c_ij (that is, the curved part RBF·(I − P)c_ij), then mean-removed and
    sign-fixed as above.
  - Edges whose (version-specific) output has standard deviation < 1e-8 on the training rows are excluded and counted.
- **Reported:** median and range over pairs × inputs × units, and the per-input medians, for both versions and both seed sets;
  next to the prediction-level d_AB: committed for the certified seeds (median 0.0771, `certify_output.txt`); for the
  early-stopped seeds no all-rows d_AB is committed, so it is computed here with the committed definition
  (RMS of the prediction difference / std of the training targets) and labelled as computed here.

## Output

`review4_output.txt` (discrepancies first), `review4_results.json`. Commit; backup bundle. No other step.

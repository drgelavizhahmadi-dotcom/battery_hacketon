# Applicability domain (DESCRIPTIVE)

The fold errors have been seen, so this study makes no predictions and issues no verdicts. Everything in it is DESCRIPTIVE.
This plan is committed before `paper2/revision/appdomain.py` exists, and its definitions and figures are fixed before computing.
Output log: `paper2/revision/appdomain_output.txt`. There is no training: every model is an archived one, and inputs are
SHA-verified against `weights_manifest.txt` and held in memory. Files win; discrepancies go at the top of the log.

## Facts of record (the paper inherits these)

- **The polish degradation is fold A, not DEC and not all-data.** The fold A ensemble RMSE goes from 0.1420 to 1.0262 (μ = 1e-6;
  `close_output.txt`) and to 0.3345 (μ = 1e-4; `certify_output.txt`, 36f295b).
  - Fold A holds out the ethyl-acetate (EA+PC) rows: 635 anchored rows, the same rows as the LOCO EA fold (`arch/sc_output.txt`, D3).
  - The DEC test labels are hidden.
- **The committed leave-one-co-solvent-out (LOCO) structure has 10 co-solvent folds** (`arch/sc_run.setup`): 2-Glyme, 2-MeTHF,
  3-Glyme, DME, EA, EC, methylene chloride, sulfolane, THF and toluene.
  - A row with several co-solvents belongs to several folds, so the fold sizes sum to more than the 2069 anchored rows.
- **Salts:** 12 in the data, 4 in the 2069 anchored rows.
- **Dose thresholds (synthetic, [4, 3, 1], Gaussian inputs).** P0 ρ* = 0.5 and P1 ρ* = 0.9; both fail to certify from ρ = 0.95
  (`dose_prereg.md` verdicts, 541355a / 75b778b). Wherever they appear beside real-data correlations, they are stated to be synthetic
  thresholds, not real-data ones.
- **The real co-solvent correlation** corr(eps_co, ln eta_co) = 0.966 is from `support_output.txt`.
- **The committed co-solvent block rank result is 14 of 27** (`rank_prereg.md`, step 0). It is cited, not recomputed.

## Discrepancies with the brief, found while writing (files win)

- **G1. The co-solvent descriptors are constant within a single-co-solvent fold.**
  - eps_co, ln eta_co and M_co are properties of a row's co-solvent (mixture). Fold A's 635 rows, and the DEC test rows, therefore
    all share ONE point in space B.
  - Consequences in analysis 2: the off-pattern distance and the hull membership are constant across fold A rows. A per-row
    Spearman correlation with them, or a median split on them, is degenerate. It is reported as such, not computed.
  - They do vary between folds (analysis 1); the training set has 30 distinct co-solvent points.
  - **Added per-row variant (DESCRIPTIVE):** the same off-pattern residual computed on the row-level mixture descriptors (mix_eps,
    mix_lneta), against the training principal direction of that pair. These vary with x_co (7 distinct values in fold A). It is
    reported beside the brief's definition and labelled MIXTURE-LEVEL.
- **G2. The μ = 1e-4 fold A weights are not archived.** `certify_foldA_0.0001_s0–4.pt` (with their `.json`) exist in
  `gauge_weights/` but in no archive and no manifest entry.
  - They are archived now, before use, as a new archive with manifest entries recording their producing script (certify.py, 36f295b)
    and the late archiving.
  - Integrity check: the reconstructed μ = 1e-4 ensemble must reproduce the logged fold A RMSE 0.3345 (bias +0.2256). If it does
    not, the run stops.
- **G3. The early and polished fold A nets are reconstructed without training.**
  - The early-stopped fold A ensemble is `scbase_A_s0–4` (archived; 107 AdamW epochs on fold A's training rows, the KAN-delta
    protocol). It must reproduce the logged early RMSE 0.1420.
  - The polished vectors (`close_foldA_1e-06_s*`, `certify_foldA_0.0001_s*`) are mapped onto those nets with close.py's flat
    parameterisation (`FlatQ.to_kan`). The unused salt-embedding rows come from the early nets, as in certify.py.
  - The reconstruction must reproduce 1.0262 and 0.3345; otherwise stop.
- **G4. The DEC gain is reported per test salt.** The figure behind the README's DEC gain (`kan.py`, `figs/kan_delta_vs_xDEC.png`)
  has one panel per test salt (3 salts), so the hygiene output is per salt × 5 seeds × 3 temperatures.

## Question

Where do the held-out rows lie relative to the training data: fold A, the LOCO folds, and the DEC test rows? And does distance
from the training data, in particular off the training correlation pattern, explain where errors are large and where polishing
beyond early stopping hurt most?

## Features and distances (fixed now)

- **Space A:** the 8 continuous KAN-delta inputs (x_co, eps_co, ln eta_co, M_co, mix_eps, mix_lneta, invT, molal), standardised with
  the committed scaler (final.py's, via `sc_run.setup`).
- **Space B:** the 3 co-solvent descriptors (eps_co, ln eta_co, M_co), standardised with the same scaler.

Per held-out row, relative to its fold's training rows:

- **Mahalanobis distance** d_M in space A: the training mean, and the Ledoit-Wolf shrunk covariance (`sklearn.covariance.LedoitWolf`)
  of the fold's training rows. The covariance's condition number is reported.
- **Leverage** h = 1/n + d_M² / (n − 1), with n the number of training rows.
- **Euclidean distance to the nearest training row** in space A.
- **Inside or outside the training convex hull in space B,** by an exact LP feasibility check (`scipy.optimize.linprog`: λ ≥ 0,
  Σλ = 1, Σλ pᵢ = q) over the distinct training points.
- **Off-pattern distance:** the absolute residual of the row's standardised (eps_co, ln eta_co) from the training principal direction
  of that pair. That is |(z − m) · u⊥|, with m the training mean and u⊥ the second eigenvector of the pair's 2 × 2 training
  covariance (the r = 0.966 pattern).
- **Mixture-level off-pattern distance (G1):** the same construction on (mix_eps, mix_lneta).

## Analyses (fixed now)

1. **Per fold** (fold A and every LOCO fold).
   - Report the median and maximum of each distance, and the fraction of rows outside the hull.
   - Relate them to the fold's LOCO RMSE: the early-stopped baseline 5-seed ensemble, `arch/sc_results.json` accuracy/base/per
     (e00e37f).
   - The comparison is a Spearman correlation over the LOCO folds with ≥ 20 test rows (2-MeTHF, DME, EA, EC, methylene chloride,
     THF, toluene: n = 7). Fold A is excluded there as a duplicate of EA.
2. **Fold A, per row** (635 rows).
   - The quantity: the change in absolute error, |ŷ_polished − y| − |ŷ_early − y| (log k), from the early-stopped ensemble to each
     polished ensemble: μ = 1e-6 (RMSE 1.0262) and μ = 1e-4 (0.3345).
   - Against each distance (d_M, leverage, nearest-neighbour distance, mixture-level off-pattern): Spearman ρ with its p-value.
   - The median error change inside vs outside the hull, and above vs below the median off-pattern distance. Under G1 these are
     degenerate for the co-solvent definition; they are reported for the mixture-level one.
3. **DEC test rows** (no labels): the same distances relative to the full anchored training set, descriptively.
   - Where DEC sits relative to the training hull in space B.
   - Its off-pattern distance (both definitions) compared with fold A's.
4. **Correlation structure** on the 2069 anchored rows: the 8 × 8 correlation matrix of the continuous inputs.
   - Ledoit-Wolf shrunk, with a PCA-shrunk sensitivity: keep the components explaining ≥ 99% of the variance, and restore the
     discarded variance to the diagonal so the variances are preserved.
   - Reported: the eigenstructure, and the strongest pairwise and partial correlations (partial = from the inverse of the shrunk
     correlation), as an input-pair table.
   - The co-solvent block rank result (14 of 27) is cited, not recomputed.
5. **Figures:**
   - (a) per-fold LOCO RMSE against median off-pattern distance;
   - (b) fold A per-row error change against the (mixture-level) off-pattern distance, for both polished settings;
   - (c) the training data in space B (eps_co, ln eta_co), with fold A and the DEC rows overlaid;
   - (d) the input-pair correlation table as a heatmap.

## Hygiene (a separate small commit)

- **The script.** `paper2/revision/dec_gain.py` computes the learned DEC gain from the archived 5 final KAN-delta seeds and prints
  it to `paper2/revision/dec_gain_output.txt`.
  - The seeds are `delta_s0–4`: `gauge.py`, 107 epochs on all 2069 anchored rows, the protocol of `kan.py`'s `finals1`.
  - The gain is Δlog k at 233, 273 and 313 K, x_DEC = 0.7, 1 mol/kg, per test salt.
  - Printed: the per-seed values (5 seeds × 3 temperatures × 3 salts), alongside the summary statistic the paper would quote. That is
    the 5-seed mean per salt and temperature (the figure's line), and its range over salts.
- **Traceability check.** The 5 archived seeds must reproduce `submission_kan.csv` on the anchored test rows: the committed
  submission's KAN-delta part, to float32 tolerance. Otherwise the script reports the mismatch.
- **The README** points to the output, replacing the value read off the figure.

## Parked

The predictive version is not part of this step: does the correlation structure predict fold-wise polish degradation across the 10
co-solvent folds? It becomes a separate pre-registered study, with a fixed prediction and a cost estimate, only if these descriptive
results show off-pattern distance bearing on the fold A degradation. The decision point is after this report.

## Report

- Tables generated from `appdomain_results.json`, and the four figures.
- One paragraph on whether off-pattern distance, rather than general distance, explains where polishing hurt. Every claim is
  labelled DESCRIPTIVE, with G1 stated next to it.
- Commit locally and make a backup bundle. Do not start the writing stage.

---
## Results and close-out (appended after the run; nothing above this line edited)

### Note (dated 2026-10-25): the parked study

The parked predictive study is DROPPED: its pre-stated precondition (off-pattern distance bearing on the fold A polish damage) is not
met. Within fold A, the co-solvent off-pattern distance is constant, and every per-row distance is collinear with x_co, through which
the damage scales by the model's x_co * h structure. No successor study.

The artefact check added during Analysis 2 (the ensembles' own errors against distance) is POST-HOC. It is labelled as such in
`appdomain_output.txt` (dated label note) and in `appdomain_results.json` ("analysis2_posthoc_artefact_check").

### Facts for the paper (for the writing stage)

- **Where the polish degradation is.** The 0.142 → 1.03 polish degradation is fold A (ethyl acetate held out), not DEC and not
  all-data. At μ = 1e-4 it is 0.3345.
- **Where fold A sits.**
  - Fold A lies on the eps_co / ln eta_co correlation pattern (off-pattern distance 0.047, scaled units), but outside the training
    convex hull in the three-descriptor space.
  - Within fold A the damage scales with x_co.
- **Two anchored row sets, named distinctly:** the 2069 anchored training rows, and the 5313 anchored DEC test rows.
- **The DEC gain** (233 K and 313 K, x_DEC = 0.7, 1 mol/kg) is quoted only from `dec_gain_output.txt`, and always with its per-seed
  spread.
- **Units.** The distances are in the committed min–max scaled units (G5).

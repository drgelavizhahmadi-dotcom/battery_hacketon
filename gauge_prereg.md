# Pre-registration: identifiability of the CALiSol KAN models

Written and committed before any experiment in `gauge.py` was run (2026-09-23).
Verdicts are appended at the end after the run. The predictions and definitions above the
"Verdicts" heading are not edited afterwards.

## Models
- **KAN-delta:** as in `final.py` (`kan.py` layer, features `DELTA_IN` + 2-dim salt embedding,
  delta = x_co · h, [d_in, 6, 1], 107 epochs, AdamW 3e-3 / 1e-4, smoothness penalty 1e-3).
  Trained with 20 seeds (0–19) on all 2069 anchored training rows.
- **KAN-full:** as in `kanfull.py` (`FULL_IN` + salt embedding, target log10 k, [d_in, 8, 1],
  107 epochs). 20 seeds on all 6520 usable training rows.
- Weights are saved to `gauge_weights/`, and later steps load them instead of retraining.

## Notation and gauge fixing (per seed)
s_j(x) = Σ_i φ_ij(x_i) + b_j (layer 0); h = Σ_j ψ_j(s_j) + b (ψ_j includes the tanh squashing
between layers). The two salt-embedding coordinates are treated as one input,
"salt": φ_salt,j = φ_e1,j + φ_e2,j evaluated at each salt's embedding. This removes the
embedding's own rotation/scale gauge.

1. Centre: φ̄_ij = φ_ij − mean over training rows.
2. Standardise: ŝ_j = (s_j − μ_j)/σ_j, φ̂_ij = φ̄_ij/σ_j, ψ̂_j(u) = ψ_j(μ_j + σ_j u). Sign flip
   so that corr(ŝ_j, ψ_j(s_j)) > 0 over training rows.
3. Centre ψ̂_j over training rows, carrying the constant separately.
4. Contributions c_j(x) = ψ_j(s_j(x)) − mean. Hungarian matching of each seed's units to seed 0
   (for the control: to the teacher) on 1 − |corr(c_j, c_k)| over training rows.

Steps 2–3 are applied **functionally**: ψ̂ is evaluated through the original ψ, so the
re-parametrised model is exactly the original. I also report how much the prediction would
change if ψ̂ were re-fitted into the same fixed RBF basis (least squares on the training-row
ŝ values). That size is how far step 2 is from an exact symmetry of the function class.

## Measures
- **(a)** raw layer-1 edges; **(b)** centred only (unit indices as trained); **(c)** gauge-fixed
  and permutation-aligned edges; **(d)** unit contributions c_j (aligned); **(e)** predictions h,
  and delta = x_co · h for KAN-delta.
- **Grid (primary):** 100 points spanning each numeric input's training range, other inputs at
  training medians and salt at the modal training salt. Edges depend only on their own input.
  (d) and (e) are evaluated along the same one-input sweeps. For salt, the "grid" is the
  set of salts present in that model's training rows.
- **Normalised cross-seed variance (pooled):**
  NV = Σ_f mean_x Var_seeds f(x) / Σ_f Var_x mean_seeds f(x), summed over all functions f in
  the measure (edges i×j, contributions i×j, or sweeps i).
- **Regions (row-based):** every measure is also evaluated at the actual rows. Density is the
  mean distance to the k = 10 nearest training neighbours of the same salt, in the model's
  scaled numeric inputs standardised by training std (self excluded).
  - **Dense:** bottom quartile of training rows.
  - **Sparse:** top quartile of training rows.
  - **Test:** the 5313 anchored DEC test rows.
  - Region NV = Σ_f mean_{rows in region} Var_seeds / Σ_f Var_{all training rows} mean_seeds.
    The shared denominator keeps the levels comparable across regions.

## Predictions
- **P1.** For KAN-delta on the grid, gauge fixing + alignment (c) removes ≥ 50% of the
  cross-seed variance of layer-1 edges relative to centring only (b): (NV_b − NV_c)/NV_b ≥ 0.5.
- **P2.** For KAN-delta, measure (c) row-based: NV_sparse ≥ 2 × NV_dense, and NV_test is the
  highest of the three.
- **P3.** For KAN-delta on the grid: NV(e) < NV(d) < NV(c) < NV(a). The brief's "(e)  (d)"
  had a missing symbol, which I read as "<". (e) is h; delta is reported alongside.
- **P4.** With width 6, every non-reference seed (19/19) has at least one unit whose Hungarian
  match to seed 0 has |corr| < 0.8.
- **P5.** On the grid, measure (c) pooled over the shared inputs (1000/T, mix ln η,
  co-solvent ln η, mix ε, co-solvent ε, molality, salt) is higher for KAN-full than for KAN-delta.
  For salt, both models are compared on the 4 salts present in KAN-delta's training rows.
- **P6 (Part B).** Using KAN-delta, 20 seeds trained per fold, on the anchored held-out rows of
  fold A (PC+EA) and fold B (EC+PC):
  - Spearman correlation between per-row seed std and |error of the seed mean| is > 0.2 on
    both folds.
  - At 80% coverage (abstaining on the 20% highest-std rows), RMSE is below the mean RMSE of
    100 random 80% subsets.

## Control (run first on Part A; its result gates the real-data analysis)
- **Teacher:** a KAN with the KAN-delta architecture, fixed random smooth edges (RBF
  coefficients = smoothed random walks along the grid, random base weights), fixed salt
  embedding, and output rescaled so std(x_co · h_teacher) equals std(real delta).
- **Data:** targets on the real 2069 anchored training inputs, plus Gaussian noise with std equal
  to the in-sample residual std of the 20-seed real KAN-delta ensemble.
- **Students:** 20 seeds, same training.
- **Procedure:** the same gauge fixing, with students matched to the teacher's units.
- **Pass criterion:** over active teacher edges (variance ≥ 5% of the largest edge variance in
  the same unit), the median over edges × 20 seeds of corr(student φ̂_ij, teacher φ̂_ij) on
  dense-region training rows is ≥ 0.8.
- **Sanity check:** median student–teacher correlation of h on training rows ≥ 0.95. If this
  fails, it is a training failure, not a gauge failure.
- **If the control fails:** stop, report, and do not interpret Part A–B on real data.

---

## Verdicts (appended after the run; `gauge_output.txt`, `gauge_results.json`)

**Control: FAILED. Part A and Part B on real data were not run, as pre-registered.**

| Check | Criterion | Result | Verdict |
|---|---|---|---|
| Edge recovery (dense rows, active edges × 20 seeds) | median corr ≥ 0.8 | **0.301** (all rows 0.488, sparse 0.429; 33% of edge-seed pairs ≥ 0.9) | **failed** |
| Prediction recovery (sanity) | median corr(h_student, h_teacher) ≥ 0.95 | **0.913** (min 0.878) | **failed** |
| Student-to-teacher unit match (not a criterion) | – | median \|corr\| of contributions 0.469, min 0.062 | – |

Setup: noise std 0.130 (real in-sample residual), teacher delta std 0.336, variance SNR 6.7;
35 of 54 teacher edges active. Reproducibility: seeds 0–4 reproduce `submission_final.csv` to 2e-16.

Findings about the control itself:
- **The sanity check failed as well.** With 107 epochs, the students match the teacher's output
  only to corr 0.91. By the pre-registered rule, that makes this at least partly a training
  failure (the students underfit), not only a gauge failure. The edge-level result cannot
  separate "the procedure is wrong" from "the students never learned the teacher function".
- **The dense-row correlation criterion is degenerate for discrete inputs** (checked after the run).
  The dense quartile is 518 rows, **all LiBOB**, and in those rows eps_co, lneta_co and M_co take
  only **2 distinct values**. Per-edge correlations there are therefore exactly ±1, or 0 for the
  constant salt edge. The histogram in `figs/gauge/control_recovery.png` piles up at −1, 0 and +1.
  The reported eps_co / lneta_co "median 1.000" and salt "0.000" are artifacts, not evidence
  either way. This is a flaw in my pre-registered criterion: kNN density within salt puts the
  densest rows in the largest single series. Across all rows these inputs take 30 distinct
  values, so the all-rows median (0.488) is much less affected. That number and the grid plots
  also show no recovery.
- **The continuous inputs, where the criterion is meaningful, still fall short:** median dense
  corr invT 0.894, molality 0.809, mix ln η 0.681, x_co 0.522, mix ε 0.396.

| Prediction | Verdict |
|---|---|
| P1 gauge fixing removes ≥ 50% of edge variance | **inconclusive** (not run: control failed) |
| P2 sparse ≥ 2× dense, test highest | **inconclusive** (not run) |
| P3 (e) < (d) < (c) < (a) | **inconclusive** (not run) |
| P4 every seed has a unit with \|corr\| < 0.8 | **inconclusive** on real data (not run). In the control, student-to-teacher matches had median \|corr\| 0.47, which is consistent with P4 but not a test of it |
| P5 KAN-full > KAN-delta residual edge variance | **inconclusive** (not run) |
| P6 seed spread flags errors | **inconclusive** (not run: gated by the control as pre-registered, although Part B does not use the gauge procedure) |

# Pre-registration: is the additive affine gauge the whole symmetry on KAN-delta's realised fiber?

Written and committed before `fiber.py` was written or run (2026-09-29). Verdicts are appended
below the "Verdicts" heading after the run; nothing above it is edited afterwards. The thresholds
in F1–F5 are the ones in the brief, unchanged.

## Gate (checked before this file was written)
Control 3 (teacher = trained KAN-delta seed 0, 20 students, AdamW + L-BFGS) was run and committed
(`9f59d39`). On real inputs (R), student-vs-teacher delta correlation is 0.99875 overall and
**0.9918 median / 0.9907 minimum in the dense quartile**. That is at or above 0.99, so the
fiber gate passes, narrowly. The same control failed to validate gauge fixing: unit match was
0.659 and edge recovery 0.852 on (R), and the (I) cell failed its own gate. This study asks
*why* units and edges differ when the function doesn't.

## Discrepancies between the brief and the committed files (files take precedence)
1. **Salt points.** The brief says to evaluate salt edges "at the 3 salt points". KAN-delta's
   anchored training rows contain **4** salts (LiAsF6, LiBF4, LiBOB, LiPF6); the test set has 3.
   I evaluate salt edges at the 4 training salts and also report the 3-test-salt subset.
2. **On-fiber distance.** The brief defines d_AB on h. KAN-delta is fitted to
   delta = x_co · h, so h is only constrained through x_co (min x_co in the anchored rows is
   about 0.1). **The primary criterion is d_AB on h, as specified.** I also report the delta version.
3. **b_jk.** "Sum of fitted constants" leaves out the layer-0 biases and the salt edge. To make
   s_k^B ≈ a·s_j^A + b_jk exact under the fit, I define
   b_jk = Σ_i c_i + c_salt + bias_k^B − a·bias_j^A, where c_salt = the mean over training rows of
   (φ^B_salt,k − a·φ^A_salt,j).

## Models
- **Seeds:** the 20 saved KAN-delta seeds, `gauge_weights/delta_s{0..19}.pt`. They were trained
  exactly as in `final.py`; seeds 0–4 reproduce `submission_final.csv`. Nothing is retrained
  from scratch.
- **Polishing:** each seed is converted to float64 and polished with L-BFGS (strong-Wolfe line
  search, 20 iterations per step) on MSE(delta) + 1e-3 · smoothness penalty. That is the
  training objective without AdamW's decoupled weight decay, which is not part of the loss.
  - Stop when the relative change in loss between steps is < 1e-8, or at 2000 iterations.
  - The polished seeds are converted back to float32 for analysis.
  - I report the loss before and after, the iterations, and why each stopped.
- Polished weights go in `gauge_weights/fiber_*` and are not committed.

## Definitions
- **Network:** s_j(x) = Σ_i φ_ij(x_i) + bias_j; h = Σ_j ψ_j(s_j) + bias; ψ_j(u) is the layer-1 edge
  applied to tanh(u). The two salt-embedding edges are summed into one salt edge (as in
  `gauge.py`).
- **On-fiber:** d_AB = RMS over training rows of (h_A − h_B) / std over training rows of the
  20-seed mean h. A pair is on-fiber if d_AB < 0.05. Pairs are unordered, and the lower seed
  index A is the source.
- **Unit matching (Step 2):** contributions c_j = ψ_j(s_j) − mean over the 2069 training rows,
  matched by the Hungarian algorithm on 1 − |corr|. I report every match |corr|.
- **Quotients (Step 3):** for a matched pair (unit j of A → unit k of B), using the 8 continuous
  edges i. B_i is approximated from A_i:
  - (i) **centred:** B̄_i vs Ā_i, with no fitted parameters.
  - (ii) **shared affine:** B_i ≈ a·A_i + c_i, one a (sign free) for all 8 edges, 8 constants.
  - (iii) **free affine:** B_i ≈ a_i·A_i + c_i per edge.
  - All three are closed-form least squares.
  - **Pooled residual of a unit pair:** sqrt(Σ_i Σ_x (B_i − fit_i)²) / sqrt(Σ_i Σ_x B̄_i²).
    That is the RMS after the fit over the RMS of the centred target edges.
- **Evaluation domains:**
  - **Grid (primary):** 100 points per input over its training range. The fit and residual are
    both computed on the grid.
  - **Regions:** the fit is on all training rows; residuals are evaluated on the dense rows,
    sparse rows and DEC test rows. The denominator is the centred-target RMS over **all training
    rows**, shared across regions so the levels are comparable.
  - **Density:** as in `gauge.py`, the mean distance to the 10 nearest same-salt training rows in
    standardised scaled inputs. Dense = bottom quartile, sparse = top quartile. As found in the
    first control, **the dense quartile is 518 rows, all LiBOB.**
  - **Test region:** the 5313 anchored DEC test rows.
- **Salt edges:** reported separately with the same (i)/(ii)/(iii) at the 4 salt points; (ii) uses
  the continuous-edge a. The 3-test-salt subset is also reported.
- **Outgoing cross-check (Step 4):** with a and b_jk from (ii) fitted on training rows, predict
  ψ_k^B(u) ≈ ψ_j^A((u − b_jk)/a) + const at u = s_k^B(x) over training rows.
  - The residual is RMS / RMS(centred ψ_k^B).
  - **Direct fit:** the same form with (α, β, const) free, fitted by nonlinear least squares
    started from (a, b_jk).
- **Explained fraction:** 1 − SS(ii)/SS(i), where SS is the sum of squared residuals pooled over
  unit pairs and edges. It is reported on the grid and per region, for all matched unit pairs
  and for pairs with |corr| ≥ 0.9. The same is reported for (iii).

## Predictions
Unit set for F2–F5: matched unit pairs with match |corr| ≥ 0.9, from on-fiber pairs only.
If the set is empty, the prediction is inconclusive.
- **F1.** After polishing, ≥ 50% of the 190 seed pairs are on-fiber (d_AB on h < 0.05).
- **F2.** Dense rows: median over the unit set of (1 − R_ii/R_i) ≥ 0.80.
- **F3.** Dense rows: median R_ii ≤ 1.5 × median R_iii.
- **F4.** Median Step-4 cross-check residual ≤ 2 × median direct-fit residual.
- **F5.** Median R_ii (sparse) ≥ 2 × median R_ii (dense), **and** median R_ii (test) is the largest
  of dense, sparse and test.

**Exploratory (no verdict):** for target units with match |corr| < 0.8, compare the best
single-unit |corr| against (a) |corr| with the sum of the best two source units' contributions,
and (b) the multiple correlation of a two-unit linear regression. This tests whether units split.

Outputs: `fiber.py`, `fiber_output.txt`, `fiber_results.json`, figures in `figs/fiber/`
(on_fiber_pairs, residual_ladder, edge_overlay_T, outgoing_crosscheck, unit_match).

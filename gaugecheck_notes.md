# gaugecheck: POST-HOC, EXPLORATORY

Decided after seeing the certify results. There is no pre-registration, and **no pre-registered
verdict changes**: certify W1–W6 and close.py V1–V6 stand as committed. Sources: `gaugecheck.py`,
`gaugecheck_output.txt`, `gaugecheck_results.json`, `figs/gaugecheck/curvature_share.png`.
Weights `gauge_weights/gaugecheck_regteacher.*` are not committed.

## A. Curvature and slope along the gauge directions, at the 9 certified μ = 1e-4 minima

The directions are certify Step 2c's least-squares directions, normalised. None is an exact symmetry
here: the edge basis cannot represent a constant exactly, and tanh breaks exact rescaling. For scale,
the Hessian's median eigenvalue is 5.7e-3 and its largest is 6–21.

| Direction (n) | dᵀHd, full loss (median) | Without μQ | μQ alone | μQ share of the curvature (median [IQR]) | Slope of data + smoothness (median) | Total slope (max) | Δprediction for a step t = 0.01 (median / max) |
|---|---|---|---|---|---|---|---|
| Constant shift, layer 1 (432) | 3.67e-4 | 2.02e-4 | 1.65e-4 | 0.46 [0.40–0.61] | 5.5e-6 | 2.1e-12 | 1.2e-6 / 7.1e-5 |
| Constant shift, layer 2 (54) | 1.89e-4 | 7.0e-6 | 1.55e-4 | **0.96** [0.86–0.98] | 1.2e-5 | 1.1e-13 | 2.0e-7 / 3.4e-5 |
| Unit rescaling (54) | 7.22e-4 | 5.49e-4 | 1.71e-4 | 0.24 [0.05–0.78] | 3.4e-5 | 1.5e-12 | 1.6e-6 / 4.6e-5 |

The nonlinear rescaling (incoming weights × 1.01, outgoing edge refitted by least squares) changes
predictions by 1.9e-5 at the median and 8.8e-5 at most. The change in the data term is at most about 1e-7
for every direction.

## B. Students against the original and the regularised teacher (μ = 1e-4)

- **B1.** The original teacher is **not stationary** under the regularised loss: max|grad| 5.6e-3, data
  term 0 by construction, penalty 4.58e-3, μQ 6.0e-4.
- **B2.** The regularised teacher (trust-exact from the teacher) is **certified** after 121 iterations.
  Data 1.33e-4, penalty 2.81e-4, μQ 1.49e-4, ‖θ‖² 27.3 (teacher 19.8); delta correlation with the
  original teacher 0.99924.

| Student | Original: delta corr | Original: unit | Original: edge | Original: d_AB | Regularised: delta corr | Regularised: unit | Regularised: edge | Regularised: d_AB |
|---|---|---|---|---|---|---|---|---|
| 115 | 0.99922 | 0.720 | 0.841 | 0.0348 | 0.99994 | **0.902** | 0.979 | **0.0095** |
| 104 | 0.99920 | 0.680 | 0.904 | 0.0353 | 0.99988 | **0.868** | 0.970 | **0.0136** |
| 114 | 0.99926 | 0.719 | 0.907 | 0.0338 | 0.99990 | **0.876** | 0.975 | **0.0123** |

## (1) Does the penalty act as a gauge fixer?
**Partly.**
- **These directions are nearly free.** Gauge-like directions are 8–30× flatter than the Hessian's typical
  direction, and moving along them changes predictions by only about 1e-7 to 1e-6 for a step of 0.01.
- **The penalty fixes the position along them.** At every certified minimum, the slope of data plus
  smoothness along each direction (about 1e-5) is cancelled exactly by μQ's slope. The penalty is what
  sets where along these near-gauge directions the minimum sits.
- **The split depends on the direction:**
  - **Constant shifts in layer 2:** μQ supplies 96% of the curvature, so it acts as a near-pure gauge
    fixer there.
  - **Constant shifts in layer 1:** μQ supplies only about 46%. The rest comes from the edge basis'
    inability to represent a constant exactly.
  - **Unit rescaling:** μQ supplies about 24% (median), with the rest from tanh. Those two directions are
    broken by the architecture itself, not just by the penalty.
- **So the expectation ("the remainder near zero") holds for layer-2 constant shifts only.**

## (2) Does W3's failure reflect the moved minimum or genuine non-convexity?
**Mostly the moved minimum.**
- **The original teacher was the wrong reference.** It is not a minimum of the regularised loss on which
  the students were certified.
- **Against the regularised teacher, the students match much better:**
  - unit match **0.87–0.90**, against 0.68–0.72 for the original;
  - edge match 0.97–0.98, against 0.84–0.91;
  - function difference d_AB 0.010–0.014, against 0.034–0.035 (2.6–3.7× smaller).
  - The regularised teacher is itself a certified minimum close to the original teacher (delta
    correlation 0.9992).
- **A small residual mismatch remains** (unit match about 0.88, not 1), so some non-convexity may remain
  even here.
- **This does not change certify W3's pre-registered verdict (failed),** which was defined against the
  original teacher. It does not address the seeds' disagreement (W2) either, since independent seeds
  have no common reference.

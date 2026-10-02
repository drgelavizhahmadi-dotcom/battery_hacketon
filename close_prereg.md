# Pre-registration: targeted null-space penalty (close)

Written and committed before `close.py` existed (2026-10-02). Verdicts are appended below the
"Verdicts" heading after the run; nothing above it is edited afterwards. Thresholds are the brief's,
unchanged. No further stages follow.

## Discrepancies and operational choices (files take precedence)
- **K1. Student starting points.** The students have no 107-epoch weights; their earliest saved state
  is control 3's AdamW stage (`control3_R_s{k}`, 530–2680 epochs), as in stage2 S2. They start from
  there, and V3's "107-epoch value" for a student means that state's ‖θ‖².
- **K2. Q** = Σ over all 66 edges of ‖P_null c_edge‖², plus Σ of the squared SiLU base weights.
  - The 66 edges are the 6 × 10 layer-1 edges, including the 2 salt-embedding edges per unit, and the
    6 layer-2 edges.
  - P_null projects onto span{1, k}, k = 0..7, the exact null space of the second-difference penalty.
  - The SiLU base weights are `layers.0.base.weight` (60) and `layers.1.base.weight` (6).
  - Biases and the embedding are not penalised. No other term changes.
- **K3. Parameters and optimiser.**
  - The same 609-parameter vector as stage2, with the unused salt embeddings fixed.
  - trust-exact in float64 with the exact autograd Hessian, maxiter 500, gtol 1e-30.
  - A 2 h wall-clock **safety** cap per run, as in stage2 (not in the brief); any run that hits it is
    reported.
  - The callback parameter is named `intermediate_result` (stage2 crash 1).
- **K4. Convergence:** stage2's three criteria, applied to **the loss being optimised**
  (data + smoothness + μQ):
  - (a) max|grad| < 1e-7;
  - (b) relative change < 1e-12 over 10 iterations;
  - (c) the smallest eigenvalue ≥ −1e-8 × the largest.
  - All three are checked at the final point.
- **K5. Inputs.**
  - Seeds `delta_s{0..5}` (start points); students `control3_R_s{115,104,114}`; teacher `delta_s0`.
  - The fiber2 capped weights, for V4's capped d_AB and as the mapping template.
  - The `stage2_A_*` caches, for V2's reference data term.
  - All are extracted from the pinned archives, SHA-256-verified, held in memory and re-checked from
    memory, and the extraction directory is deleted at once.
- **K6. "That μ" in V2–V5:** the **selected μ**, meaning the one at which the most of the 9 models
  converge, with ties going to the smallest μ. The same μ is used for the accuracy step. V2–V5 are
  evaluated on the models that converged at that μ.
- **K7. Accuracy.**
  - Train on fold A's training rows (anchored rows except PC+EA) exactly as `final.py`: `kan.train`,
    seeds 0–4, 107 epochs, the final.py loss **without** μQ.
  - Then polish each seed with trust-exact under data + smoothness + μ_selected · Q (K3, K4).
  - Take the ensemble mean of the 5 polished models, converged or not. Report RMSE and bias on the
    635 anchored PC+EA rows, next to KAN-delta's 0.142 (recomputed earlier as 0.1420).
- **K8. Logged per run:** convergence, iterations, the stop reason and scipy's message; the data term,
  smoothness penalty and μQ; ‖θ‖² per block (stage2 S3 blocks); the median per-unit cross-edge
  cancellation ratio and the fraction of saturated hidden units (|s_j| > 3), both as in the anatomy
  post-hoc check; and the number of negative eigenvalues (< −1e-8 × the largest) at the end.
- **K9. Runtime and caching.** There are 27 runs (3 μ × 9 models) plus 5 fold A polishes, at about
  30 min per run capped at 500 iterations, so the worst case is about 16 h. Every run is cached, so
  the script can resume, and the log is appended on a resume.

## Predictions
- **V1.** At some μ, at least 4 of 6 seeds **and** at least 2 of 3 students converge.
- **V2.** At the selected μ, every converged seed's data term is ≤ 1.05 × its stage2 Part A final data
  term. Inconclusive if no seed converged.
- **V3.** At the selected μ, every converged model's ‖θ‖² is ≤ 10 × its starting value (K1).
  Inconclusive if nothing converged.
- **V4.** At the selected μ, among converged seeds: median d_AB on delta ≤ 0.5 × the median fiber2
  capped d_AB over the same pairs (stage2 S8 definition). Inconclusive if fewer than 2 seeds converged.
- **V5.** At the selected μ, among converged students: the median unit match to the teacher, over
  units × students (`trust.py` Q8 procedure), is ≥ 0.8. Inconclusive if no student converged.
- **V6.** Fold A RMSE ≤ 0.150 (K7).

## Interpretation (fixed in advance)
- **V1, V2 and V3 hold:** the smoothness penalty's null space, together with the unpenalised SiLU
  weights, causes the runaway. Closing it restores finite minima without costing fit.
  - **V4 holds:** the seed disagreement was the runaway. **V4 fails:** distinct minima remain.
  - **V5 holds:** the students' unit mismatch was the runaway.
- **V1 and V3 hold, V2 fails:** closing the null space works, but the runaway was buying real fit
  (likely through saturated, sharp features).
- **V1 fails:** the penalty's null space is not sufficient; the data-side freedom or saturation
  sustains the runaway.
- **Any other combination:** reported as found, without forcing it into a branch.

Outputs: `close.py`, `close_output.txt`, `close_results.json`, `figs/close/`. Weights go in
`gauge_weights/close_*` and are not committed.

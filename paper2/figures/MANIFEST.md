# paper2/figures: manifest

Every figure is built from **committed** results only: no training and no re-optimisation. Style is shared via `style.py`:

- vector PDF and 300-dpi PNG; 6.5 in (full) or 3.25 in (half) wide; 7–9 pt fonts;
- Okabe–Ito colour-blind-safe palette, with fixed colours per condition (runaway / not certified = vermillion, certified = blue, early state = grey; P0 orange, P1 sky, P2 green);
- no titles inside the figures; panel letters a, b, c.

Each figure has a sidecar `<name>.json` listing the data files used (path, last commit, SHA-256), any archived weights (SHA-256, archive), notes, and every plotted number.
Rebuild with `/usr/bin/python3 paper2/figures/<script>` from the repo root (pinned interpreter; or `make figures` in paper2/tex).

Paper figure numbers are in the second column. From draft revision 58 (2026-10-25), figures 2, 3, 4 and 8 of the files are
supplementary (Appendix E) and the paper renumbers in order of placement; file names are kept, so sidecars are unchanged.
Earlier, the files were numbered in order of first reference (2026-10-25 renumbering for draft revision 55: the former fig3_dose_response, fig4_where and fig5_uniqueness are now fig8_dose_response, fig3_where and fig9_uniqueness; their plotted values, notes and weights are unchanged, checked against the previous sidecars). Only the weights_manifest.txt commit in the sidecars moved (d6e9f52 -> 179ac24, archive 19 appended).

| Figure | Paper figure (draft rev 58) | Script | Data files (commit) | Archived weights | Content |
|---|---|---|---|---|---|
| fig1_mechanism | 1 | `make_fig1.py` | none (computed from committed code: kan.py, paper2/synth.py) | — | Mechanism (computed illustration): a) linear-pattern edge and best line; b) penalties vs tilt; c) pre-activation change, correlated vs independent inputs |
| fig2_synthetic | 6 (App. E) | `make_fig2.py` | `paper2/synth_results.json` (34cc321) | — | Synthetic 2x3: ||theta||^2 over trust-exact polishing iterations, per seed, certified vs not |
| fig3_where | 7 (App. E) | `make_fig3.py` | `weights_manifest.txt` (179ac24); `paper2/anatomy_synth_results.json` (ea22a56); `paper2/checks/checks_results.json` (6e1df32) | 18 files, archives gauge_weights_2026-09-29.tar.gz, gauge_weights_2026-09-29b.tar.gz, gauge_weights_2026-10-02e.tar.gz | Where the runaway lives: a) share of the parameter change by block; b) largest hidden-unit correlation |
| fig4_applicability | 8 (App. E) | `make_fig4.py` | `paper2/revision/appdomain_results.json` (b19fa2f); `weights_manifest.txt` (179ac24) | 15 files, archives gauge_weights_2026-10-03f.tar.gz, gauge_weights_2026-10-04i.tar.gz, gauge_weights_2026-10-25s.tar.gz | Applicability (descriptive): a, b) fold A per-row error change against x_co at mu = 1e-6 and 1e-4; c) training data in descriptor space with fold A and DEC rows |
| fig5_paths | 2 | `make_fig5.py` | `paper2/revision/infimum_results.json` (4003b21) | — | Certified L2 regularisation paths: ||theta*||^2 and L* against gamma, synthetic P0/P1/P2 and real (open vs closed, mu = 1e-4), one line per seed |
| fig6_correlation_dose | 3 | `make_fig6.py` | `paper2/revision/dose_results.json` (541355a) | — | Correlation dose-response: share uncertified, median D and hidden-unit correlation against rho; P2 at 0.99 |
| fig7_bspline | 4 | `make_fig7.py` | `paper2/synth_results.json` (34cc321); `paper2/revision/bspline_results.json` (ffa6e64); `paper2/revision/bspline_posthoc_results.json` (9167740) | — | B-spline replication: a) certified per cell, RBF vs B-spline; b, c) P2 paths extended to gamma = 1e-9 |
| fig8_dose_response | 9 (App. E) | `make_fig8.py` | `stage2_results.json` (b8d6488); `close_results.json` (65ed55f); `certify_results.json` (36f295b) | — | Real-data dose-response in mu: a) numerical minima / certified; b) norm growth; c) data term vs runaway; d) fold A RMSE |
| fig9_uniqueness | 5 | `make_fig9.py` | `arch/sc_results.json` (e00e37f); `certify_results.json` (36f295b); `weights_manifest.txt` (179ac24) | 40 files, archives gauge_weights_2026-10-04i.tar.gz | Convergence is not uniqueness: a) near-null Hessian eigenvalues; b) seed disagreement d_AB |
| figS1_lossdecomp | 10 (App. E) | `make_supp.py` | `paper2/lossdecomp_results.json` (4ec6fce) | — | Supplementary: loss decomposition by term (synthetic, early -> polished) |
| figS2_gaugecheck | 11 (App. E) | `make_supp.py` | `gaugecheck_results.json` (0b29351) | — | Supplementary: mu*Q share of curvature along gauge directions (mu = 1e-4 certified minima) |
| figS3_loco | 12 (App. E) | `make_supp.py` | `arch/sc_results.json` (e00e37f); `arch/ex_results.json` (579b58f) | — | Supplementary: LOCO RMSE per fold, baseline / SC-KAN-delta / EX-KAN |

## Panels that use deterministic evaluations of archived weights

These numbers exist in no committed result file. They are evaluated here from SHA-verified archived weights held in memory, with no optimisation:

- **fig3a, real data:** the split of the parameter change into affine and curved RBF parts per layer, for 107 epochs → fiber2 capped → stage2 Part A.
  - `stage2_results.json` stores only per-block norms, without an affine/curved split and without the 107 → capped change.
- **fig9b, polished fold A d_AB** for SC-KAN-δ and the baseline (marked *).
  - arch/sc committed d_AB only at the early state.
- **fig4a, b, fold A per-row error change** (early-stopped -> mu = 1e-6 and mu = 1e-4 ensembles).
  - appdomain_results.json stores only the per-row Spearman correlations and medians; the per-row values are re-evaluated with the
    reconstruction of paper2/revision/appdomain.py, and the ensemble RMSEs must reproduce the logged 0.1420 / 1.0262 / 0.3345 or the script stops.

## Not plotted (not in committed data; not recomputed)

- **fig8a students:** only the 6 real-data seeds are shown. The students' counts are in `certify_results.json`.
- **fig8d at μ = 1e-5:** no polished fold A RMSE was committed (the brief asked for 1e-6 and 1e-4 only).
- **fig9a, stage2 Part A finals (μ = 0, all rows):** no near-null eigenvalue count was stored. The baseline shown instead is the fold A KAN-δ polished in arch/sc.
  - Those baseline points are not stationary (0/10 certified, 500-iteration cap), so their count is taken at a non-minimum.
- **fig2:** only ‖θ‖² per iteration was saved, so no per-iteration loss components could be shown.
- **figS1:** no intermediate iterates exist; the decomposition covers early → polished only.
- **fig5:** path points not reached (budget-stopped real paths) are absent; no point is interpolated.
- **fig7b, c:** the B-spline P0/P1 paths are not shown (their polishes did not certify on correlated inputs; the extension was run for P2 only).

## Other inputs

- **fig3, fig4 and fig9** also read the competition CSVs (git-ignored, not redistributed) through `arch/sc_run.setup()`. They are needed to rebuild flat parameter vectors and to evaluate predictions on the training rows.
- **fig1** uses only committed code: the `kan.KANLayer` edge type and `paper2/synth.make_inputs` with fixed seeds.

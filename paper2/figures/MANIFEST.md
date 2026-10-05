# paper2/figures: manifest

Every figure is built from **committed** results only: no training and no re-optimisation. Style is shared via `style.py`:

- vector PDF and 300-dpi PNG; 6.5 in (full) or 3.25 in (half) wide; 7–9 pt fonts;
- Okabe–Ito colour-blind-safe palette, with fixed colours per condition (runaway / not certified = vermillion, certified = blue, early state = grey; P0 orange, P1 sky, P2 green);
- no titles inside the figures; panel letters a, b, c.

Each figure has a sidecar `<name>.json` listing the data files used (path, last commit, SHA-256), any archived weights (SHA-256, archive), notes, and every plotted number.
Rebuild with `/usr/bin/python3 paper2/figures/<script>` from the repo root (pinned interpreter; or `make figures` in paper2/tex).

| Figure | Script | Data files (commit) | Archived weights | Content |
|---|---|---|---|---|
| fig1_mechanism | `make_fig1.py` | none (computed from committed code: kan.py, paper2/synth.py) | — | Mechanism (computed illustration): a) linear-pattern edge and best line; b) penalties vs tilt; c) pre-activation change, correlated vs independent inputs |
| fig2_synthetic | `make_fig2.py` | `paper2/synth_results.json` (34cc321) | — | Synthetic 2x3: ||theta||^2 over trust-exact polishing iterations, per seed, certified vs not |
| fig3_dose_response | `make_fig3.py` | `stage2_results.json` (b8d6488); `close_results.json` (65ed55f); `certify_results.json` (36f295b) | — | Real-data dose-response in mu: a) numerical minima / certified; b) norm growth; c) data term vs runaway; d) fold A RMSE |
| fig4_where | `make_fig4.py` | `weights_manifest.txt` (d6e9f52); `paper2/anatomy_synth_results.json` (ea22a56); `paper2/checks/checks_results.json` (6e1df32) | 18 files, archives gauge_weights_2026-09-29.tar.gz, gauge_weights_2026-09-29b.tar.gz, gauge_weights_2026-10-02e.tar.gz | Where the runaway lives: a) share of the parameter change by block; b) largest hidden-unit correlation |
| fig5_uniqueness | `make_fig5.py` | `arch/sc_results.json` (e00e37f); `certify_results.json` (36f295b); `weights_manifest.txt` (d6e9f52) | 40 files, archives gauge_weights_2026-10-04i.tar.gz | Convergence is not uniqueness: a) near-null Hessian eigenvalues; b) seed disagreement d_AB |
| figS1_lossdecomp | `make_supp.py` | `paper2/lossdecomp_results.json` (4ec6fce) | — | Supplementary: loss decomposition by term (synthetic, early -> polished) |
| figS2_gaugecheck | `make_supp.py` | `gaugecheck_results.json` (0b29351) | — | Supplementary: mu*Q share of curvature along gauge directions (mu = 1e-4 certified minima) |
| figS3_loco | `make_supp.py` | `arch/sc_results.json` (e00e37f); `arch/ex_results.json` (579b58f) | — | Supplementary: LOCO RMSE per fold, baseline / SC-KAN-delta / EX-KAN |

## Panels that use deterministic evaluations of archived weights

These numbers exist in no committed result file. They are evaluated here from SHA-verified archived weights held in memory, with no optimisation:

- **fig4a, real data:** the split of the parameter change into affine and curved RBF parts per layer, for 107 epochs → fiber2 capped → stage2 Part A.
  - `stage2_results.json` stores only per-block norms, without an affine/curved split and without the 107 → capped change.
- **fig5b, polished fold A d_AB** for SC-KAN-δ and the baseline (marked *).
  - arch/sc committed d_AB only at the early state.

## Not plotted (not in committed data; not recomputed)

- **fig3a students:** only the 6 real-data seeds are shown. The students' counts are in `certify_results.json`.
- **fig3d at μ = 1e-5:** no polished fold A RMSE was committed (the brief asked for 1e-6 and 1e-4 only).
- **fig5a, stage2 Part A finals (μ = 0, all rows):** no near-null eigenvalue count was stored. The baseline shown instead is the fold A KAN-δ polished in arch/sc.
  - Those baseline points are not stationary (0/10 certified, 500-iteration cap), so their count is taken at a non-minimum.
- **fig2:** only ‖θ‖² per iteration was saved, so no per-iteration loss components could be shown.
- **figS1:** no intermediate iterates exist; the decomposition covers early → polished only.

## Other inputs

- **fig4 and fig5** also read the competition CSVs (git-ignored, not redistributed) through `arch/sc_run.setup()`. They are needed to rebuild flat parameter vectors and to evaluate predictions on the training rows.
- **fig1** uses only committed code: the `kan.KANLayer` edge type and `paper2/synth.make_inputs` with fixed seeds.

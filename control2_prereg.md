# Pre-registration: corrected control for KAN gauge fixing (2×2 design)

Written and committed before `control2.py` was written or run (2026-09-23). It follows the
failed first control in `gauge_prereg.md`. Verdicts are appended below the "Verdicts" heading
after the run; nothing above it is edited afterwards.

## Question
Did the first control fail because gauge fixing doesn't work, because the students were
undertrained, or because collinear training inputs make edges unidentifiable?

## Fixed across cells
- **Teacher:** the first control's teacher, unchanged (`gauge.smooth_teacher`, seed 12345,
  KAN-delta architecture [8 numeric + 2-dim salt embedding, 6, 1], output scaled on the real
  inputs so std(delta) = std(real delta)).
- **Targets:** delta = x_co · h_teacher, **noise-free**.
- **Students:** 20 seeds (0–19), same architecture, AdamW lr 3e-3, weight decay 1e-4,
  smoothness penalty 1e-3, full batch.
- **Gauge fixing:** as in Part A, steps 1–4, computed separately for the teacher and each
  student on that cell's training rows. Students are Hungarian-matched to the teacher's units.

## Factors
- **Inputs**
  - **(R):** the 2069 real anchored training rows: scaled inputs, salt, and x_co.
  - **(I):** 2069 rows where each of the 8 numeric columns and the salt column is resampled
    independently, with replacement, from its own marginal over the real rows
    (numpy default_rng(0)). The x_co multiplier in delta = x_co · h is the unscaled
    resampled x_co column.
- **Training**
  - **(S):** 107 epochs.
  - **(C):** trains until training RMSE on the target (MSE term only) is < 2% of the target's std,
    checked every epoch, or 3000 epochs, whichever comes first. I report the epoch at which each
    student stopped and whether it reached the threshold.

## Measures (per cell)
- **Prediction recovery:** corr(h_student, h_teacher) over that cell's training rows.
  Median over 20 students. I also report corr on delta.
- **Unit matching:** |corr| of the matched unit contributions c_j over the cell's rows.
  Median over 6 units × 20 students.
- **Edge recovery, on continuous inputs only:** the 8 numeric inputs x_co, eps_co, lneta_co,
  M_co, mix_eps, mix_lneta, invT and molal. Salt is excluded.
  - Per edge: |corr| between the student's and teacher's gauge-fixed layer-1 edge on a
    100-point grid spanning that input's range over the real training rows. The range is
    the same in R and I, since the marginals are identical.
  - **Active edges:** the teacher edge's variance on the cell's rows is ≥ 5% of the largest
    edge variance in the same unit.
  - Summary: median over active edges × 20 students, overall and per input.
- **Sanity gate (converged cells only):** median prediction corr ≥ 0.99. If a converged cell
  fails the gate, its edge recovery is reported but not interpreted, and predictions about it
  are **inconclusive**.

## Predictions
- **C1. (I, C):** median edge recovery |corr| ≥ 0.9 **and** median unit matching |corr| ≥ 0.9.
  In words: gauge fixing works when the data allow it.
- **C2. (R, C):**
  - invT edges have median |corr| ≥ 0.9, and molal edges have median |corr| ≥ 0.9 (each separately).
  - The co-solvent descriptor edges (eps_co, lneta_co, M_co; their active edges pooled) have
    median |corr| < 0.7.
  - In words: collinearity, not the gauge, limits those edges. If an input group has no
    active edges, that part of C2 is inconclusive.
- **C3. (I, S) and (R, S):** median prediction corr < 0.95 in both cells.
  In words: undertraining explains the first control's non-convergence.

Outputs: `control2.py`, `control2_output.txt`, `control2_results.json`,
`figs/gauge/control2_recovery.png`, weights in `gauge_weights/control2_*`.

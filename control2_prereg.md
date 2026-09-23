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

---

## Verdicts (appended after the run; `control2_output.txt`, `control2_results.json`, `figs/gauge/control2_recovery.png`)

The independent inputs did break the collinearity: the largest |corr| between numeric columns
fell from 0.966 (real) to 0.047.

| Cell | Epochs | Reached 2% tolerance | Final training RMSE / target std (median) | Prediction corr (h) | Delta corr | Unit match \|r\| | Edge \|r\| (active continuous) | invT | molal | Co-solvent descriptors | Gate ≥ 0.99 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (R,S) | 107 | 0/20 | 0.295 | 0.919 | 0.956 | 0.462 | 0.634 | 0.634 | 0.951 | 0.631 | – |
| (R,C) | 3000 (cap) | **0/20** | 0.185 | 0.969 | 0.983 | 0.416 | 0.818 | 0.796 | 0.988 | 0.860 | **FAIL** |
| (I,S) | 107 | 0/20 | 0.479 | 0.858 | 0.878 | 0.367 | 0.811 | 0.730 | 0.898 | 0.882 | – |
| (I,C) | 3000 (cap) | **0/20** | 0.286 | 0.947 | 0.958 | 0.849 | 0.978 | 0.982 | 0.995 | 0.977 | **FAIL** |

The final-RMSE column was computed after the run from the saved weights; it was not
pre-registered. M_co had no active teacher edges in any cell, so the co-solvent descriptor
median pools eps_co and lneta_co only.

| Prediction | Verdict |
|---|---|
| **C1** (I,C): edge median ≥ 0.9 and unit match ≥ 0.9 | **inconclusive**: (I,C) failed the sanity gate (prediction corr 0.947 < 0.99) |
| **C2** (R,C): invT ≥ 0.9, molal ≥ 0.9, co-solvent descriptors < 0.7 | **inconclusive**: (R,C) failed the sanity gate (0.969 < 0.99) |
| **C3** (I,S) and (R,S): prediction corr < 0.95 | **held**: 0.858 and 0.919 |

Notes (not verdicts):
- **The converged condition was never reached.** All 40 students hit the 3000-epoch cap, levelling
  off at 18–29% of target std rather than 2%. With these optimiser settings (lr 3e-3, weight decay,
  smoothness penalty), a student of the *same architecture* cannot fit the teacher closely.
  The teacher's random-walk edges may be rougher than the smoothness penalty allows. So C3
  "held" only in the narrow sense: 107 epochs is worse than 3000. Undertraining is not the whole
  explanation, because 3000 epochs does not converge either.
- **Gating on h is weaker than gating on delta.** The target is delta = x_co · h, so h is weakly
  constrained where x_co is small, and the (I) inputs resample many small x_co values. The gate
  fails on delta too (0.983 and 0.958), so the verdicts don't change.
- **Exploratory, not interpretable under the pre-registered gate:** in (I,C), edges recover at
  median 0.978 and units at 0.849, while (R,C) recovers edges at 0.818 and units at only 0.416.
  In the same direction as C1: removing collinearity improves recovery a lot at the same
  training budget. **Against C2** even if its gate had passed: in (R,C) invT reaches only 0.796,
  and the co-solvent descriptors reach 0.860, well above the predicted < 0.7.

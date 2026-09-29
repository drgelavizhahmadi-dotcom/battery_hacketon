# Pre-registration: control 3, a reachable teacher

Written and committed before `control3.py` was written or run (2026-09-23). Verdicts are
appended below the "Verdicts" heading after the run; nothing above it is edited afterwards.

**Provenance, which bears on how much weight each prediction carries:**
- **C1'** restates control 2's C1 under conditions that can reach convergence.
- **C2'** was formulated **after** seeing control 2's exploratory results (edge |r| 0.978 for
  (I,C) vs 0.818 for (R,C)). It is weaker evidence than C1'.
- **C4** was also written after control 2's exploratory per-input numbers were visible. It
  carries the same caveat.

## Why a new teacher
In control 2, no student reached the 2% tolerance in 3000 epochs: a random-walk teacher of the
same architecture was not reachable under this penalty and optimiser. Here the teacher is itself
a model trained with the same penalty, so a solution with the same smoothness is known to exist.

## Teacher and data
- **Teacher:** KAN-delta seed 0 trained on the real 2069 anchored rows exactly as in `final.py`
  (`kan.train`, 107 epochs, AdamW lr 3e-3 / wd 1e-4, smoothness 1e-3). These are the
  `gauge_weights/delta_s0.pt` weights, which reproduce `submission_final.csv`'s seed-0 member.
- **Targets:** teacher delta = x_co · h_teacher on the cell's inputs, noise-free.
- **Inputs**
  - **(R):** the real anchored training rows.
  - **(I):** each numeric column and the salt column resampled independently from its own
    marginal, exactly as in control 2 (numpy default_rng(0), with replacement, 2069 rows,
    x_co multiplier = unscaled resampled x_co).

## Students
- **Seeds:** 20 per cell, seeds 100–119 (distinct from the teacher's seed 0).
- **Architecture and objective:** same as the teacher: MSE + 1e-3 × smoothness penalty.
- **Per-student stop condition** (the per-student version of the gate): delta corr with the
  teacher ≥ 0.99 **and** training RMSE ≤ 5% of the target's std. It is checked every 10 epochs
  during AdamW and after every L-BFGS step.
- **Phase 1:** AdamW (lr 3e-3, weight decay 1e-4) for up to 3000 epochs. It stops early if the
  condition is met.
- **Phase 2**, only if the condition is still unmet:
  - L-BFGS (lr 1, strong-Wolfe line search, up to 20 iterations per step) on the same
    objective (MSE + penalty). There is no decoupled weight decay in this phase.
  - It stops when the condition is met or 500 L-BFGS iterations have been used in total.
- **Reported:** AdamW epochs and L-BFGS iterations used per student (median and range), and
  whether the student met the condition.

## Gate (per cell)
Median student-vs-teacher correlation on delta over the cell's rows ≥ 0.99, **and** median
training RMSE ≤ 5% of target std. Cells that fail the gate are reported but not interpreted:
predictions that depend on them are **inconclusive**.

## Recovery measures
- **Gauge fixing:** steps 1–4 as in Part A. Each network's gauge is computed on that cell's
  rows, and students are Hungarian-matched to the teacher's units on contribution correlation.
- **Continuous inputs:** the 8 numeric inputs x_co, eps_co, lneta_co, M_co, mix_eps, mix_lneta,
  invT and molal. Salt is excluded.
- **Active edges:** the teacher's gauge-fixed edge variance on the cell's rows is ≥ 5% of the
  largest edge variance (salt edge included) in the same unit. The active set is determined per
  cell, and I report which edges are active.
- **Per-edge recovery:** |corr| between the student's and teacher's gauge-fixed layer-1 edge on a
  100-point grid spanning the input's range over the real training rows.
- **Summaries:**
  - Median over active edges × 20 students, overall and per input.
  - Unit match: median |corr| of matched contributions over 6 units × 20 students.

## Predictions
- **C1'.** (I) passes the gate, **and** its median edge |corr| ≥ 0.9, **and** its median unit-match
  |corr| ≥ 0.9.
- **C2'.** (R) passes the gate, **and** median edge |corr|(R) ≤ median edge |corr|(I) − 0.10.
  If (I) fails the gate, C2' is **inconclusive**, because the comparison needs a valid (I).
- **C4.** If (I) passes the gate: for every continuous input with at least one active edge in
  (I), the median |corr| over that input's active edges × students is ≥ 0.9. This includes
  the co-solvent descriptors. Inputs without active edges are listed and excluded.
  If (I) fails the gate, C4 is **inconclusive**.

## What counts as "procedure validated"
The gauge-fixing procedure is called validated **if and only if C1' holds**: it recovers a known
teacher's edges and units once training converges and the inputs allow identification.
C2' and C4 describe the data's limits; they don't bear on whether the procedure is valid.

Outputs: `control3.py`, `control3_output.txt`, `control3_results.json`,
`figs/gauge/control3_recovery.png`. Weights go in `gauge_weights/control3_*` and are not committed.

---

## Verdicts (appended after the run; `control3_output.txt`, `control3_results.json`, `figs/gauge/control3_recovery.png`)

| Cell | AdamW epochs (median [range]) | L-BFGS iters | Met stop condition | Train RMSE / std | Delta corr (median [min]) | Gate | Unit match \|r\| | Edge \|r\| |
|---|---|---|---|---|---|---|---|---|
| (R) | 1355 [530–2680] | 0 | 20/20 | 0.0500 | 0.99875 [0.99875] | **PASS** | 0.659 | 0.852 |
| (I) | 3000 [cap] | 500 [cap] | **0/20** | **0.1276** | 0.99184 [0.98845] | **FAIL** (RMSE) | 0.648 | 0.960 |

Per-input edge |r| (median over active edges × 20 students):

| Input | x_co | eps_co | lneta_co | M_co | mix_eps | mix_lneta | invT | molal |
|---|---|---|---|---|---|---|---|---|
| (R) | 0.767 | 0.659 | 0.743 | 0.927 | 0.578 | 0.908 | 0.962 | 0.900 |
| (I), not interpretable | 0.970 | 0.900 | 0.801 | 0.947 | 0.958 | 0.983 | 0.986 | 0.946 |

Active continuous teacher edges (the same 38 of 48 in both cells): all 6 units for x_co, eps_co,
mix_eps and invT; units 0, 1, 3, 4, 5 for mix_lneta; units 0, 2, 3, 5 for molal; units 3, 4, 5 for
M_co; units 1 and 5 for lneta_co.

| Prediction | Verdict |
|---|---|
| **C1'** (I) passes the gate, edge ≥ 0.9, unit ≥ 0.9 | **failed**: (I) failed the gate (RMSE/std 0.128 > 0.05, although delta corr 0.992 passes); unit match 0.648 < 0.9 in any case |
| **C2'** (R) passes and edge(R) ≤ edge(I) − 0.10 | **inconclusive**: (I) failed the gate. (R) passed |
| **C4** every continuous input ≥ 0.9 in (I) | **inconclusive**: (I) failed the gate |

**Gauge-fixing procedure validated: NO** (C1' did not hold).

Notes (not verdicts):
- **(R) is the informative cell.** The students reproduce the teacher's function (delta corr
  0.99875 overall; in the dense quartile median 0.9918, min 0.9907, computed after the run), yet
  unit matching is only 0.659 and edge recovery 0.852. The mixture/co-solvent descriptor edges are
  the worst (mix_eps 0.578, eps_co 0.659). **The same function is realised by different units,**
  so steps 1–4 do not remove all of the seed-to-seed freedom on real inputs.
- **(R) students stop as soon as they meet the condition,** so "converged" here means exactly at
  the 5% RMSE threshold, not beyond it.
- **(I) failed because the teacher is hard to fit there,** not because training was cut short:
  L-BFGS hit its 500-iteration cap at 12.8% RMSE. A teacher trained on real, collinear rows is
  being evaluated on independent combinations it never saw, which is likely where its function
  is least smooth. That is a limitation of this design, not evidence about the gauge.

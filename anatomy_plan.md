# Plan: anatomy of the parameter runaway (descriptive, no optimisation)

Committed before `anatomy.py` existed (2026-10-02). This is a descriptive study: there are no
verdicts. For each hypothesis, the outcome is recorded below the "Outcomes" heading after the run,
against the "as expected if" criterion fixed here. Nothing above that heading is edited afterwards.

## Inputs (held in memory)
- **Provenance:** every file is extracted from its pinned archive, SHA-256-verified against
  `weights_manifest.txt`, read into memory, and re-checked from memory. The extraction directory is
  deleted immediately, and nothing is re-read from disk.
- **Baseline seeds 0–5** (7 points each):
  - 107-epoch: `delta_s{i}`
  - fiber: `fiber_polished_s{i}`
  - fiber2 capped: `fiber2_ind_{i}`
  - trust final: `trust_ind_{i}`
  - stage2 Part A final, Part C λ = 1e-6 final, Part C λ = 1e-5 final: `stage2_{A,C1e-06,C1e-05}_ind_{i}`
- **Students 115, 104, 114** (6 points each):
  - AdamW stage: `control3_R_s{k}`, trained 530–2680 epochs, not 107 (as in stage2 S2)
  - fiber: N/A
  - fiber2 capped: `fiber2_stu_{k}`
  - trust final: `trust_stu_{k}`
  - stage2 A, C1e-6 and C1e-5 finals
- **Teacher:** `delta_s0`.
- **Branching:** trust, A, C1e-6 and C1e-5 all start from the capped state. The **unregularised
  path** is AdamW → fiber → capped → A final, with trust final as a short side branch. The C points
  are optimised with an additional L2 term λ‖θ‖², which is reported separately.
- **Stage2 caches** hold the 609-vector of optimised parameters, which is mapped back into the network
  exactly as in `stage2.py`.

## Definitions
- **Data term:** MSE(x_co · h, y) on the 2069 anchored training rows. y is the data delta for the
  seeds and the teacher's delta for the students.
- **Penalty:** 1e-3 × Σ squared second differences of `layers.0.coef` and `layers.1.coef`.
- **Total:** data + penalty, the training loss, with no L2 term.
- **‖θ‖² per block:** as in stage2 S3 (layer-1 RBF, layer-2 RBF, SiLU base, biases, salt
  embedding), over the 609 optimised parameters.
- **Edges:**
  - Layer-1 edge (input i → unit j): its SiLU term w_ji · silu(z_i) and RBF term Σ_k c_jik φ_k(z_i),
    evaluated on the training rows. The salt edges use z = tanh(embedding).
  - Layer-2 edge (unit j → output): the same, with z = tanh(s_j).
  - The SiLU and RBF terms and their sum are each summarised by RMS over the training rows.
  - **Cancellation ratio** = (RMS SiLU + RMS RBF) / RMS sum. This is ≥ 1, and large values mean the two
    terms cancel.
  - Reported as the median over the edges in a block (layer-1 continuous, layer-1 salt, layer-2) at
    each point.
- **Null space of the penalty:** span{1, k} over the centre index k = 0..7 (constant and linear
  patterns, 2 dimensions).
  - Null-space share of an edge's RBF vector c: ‖P_null c‖² / ‖c‖² (an energy share). The random
    expectation is 2/8 = 0.25.
  - Also the null-space share of the change in RBF coefficients between consecutive points on the
    unregularised path, pooled over the edges of a block.

## Hypotheses and how each is judged
- **Q1:** along the unregularised path, as ‖θ‖² grows, the data term stays nearly constant (students
  especially) while the penalty keeps falling.
  - *As expected if:* from capped to A final, the relative change in the data term is < 10% for
    every student and for at least 4/6 seeds, **and** the penalty decreases from capped to A final
    for at least 7 of the 9 models.
  - The earlier AdamW → capped segment is reported but not judged, because that is where the data term
    is being fitted.
- **Q2:** cancellation ratios are far above 1 and grow along the trajectory, and the growth is
  concentrated in the penalty's null space.
  - *As expected if* all three hold:
    - the median layer-1 continuous cancellation ratio at A final is > 5 for at least 7 of 9 models;
    - it is higher at A final than at the AdamW stage for at least 7 of 9;
    - the pooled null-space share of the RBF-coefficient change from capped to A final is > 0.5,
      against 0.25 at random, for at least 7 of 9.
- **Q3:** the students reach a lower penalty than the teacher, at a much larger norm.
  - The teacher's data term on its own targets is 0 by construction, because the students' targets
    are the teacher's predictions. Its data term on the real data is also reported.
  - *As expected if:* at both the capped and the A-final points, every student has a lower penalty
    than the teacher **and** a ‖θ‖² at least 10× the teacher's.

Outputs: `anatomy.py`, `anatomy_output.txt`, `anatomy_results.json`, `figs/anatomy/`.
No interventions follow.

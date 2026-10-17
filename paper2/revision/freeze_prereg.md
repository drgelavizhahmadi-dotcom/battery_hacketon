# Revision step 2: is composition necessary? (plus weight decay)

This file is committed before `paper2/revision/freeze.py` exists. Verdicts will be appended below the "Verdicts" heading;
nothing above it will be edited. Files win over this text; discrepancies found while running go at the top of
`paper2/revision/freeze_output.txt`. Inputs are SHA-verified against `weights_manifest.txt` and held in memory. No weights
go into git; new weights are archived outside the repository with manifest entries.

## Decision recorded before the design

Arm A's real-data models use the original final.py loss WITHOUT a weight-decay term, because that is the loss under which
the runaway was observed.
- Revision step 1 (commit fcda1c1) found the equivalent-loss decay term material: synthetic 2.6–6.2 × the D2 penalty.
- It also found the actual training shrinkage negligible: × 0.99997 over 107 AdamW steps.
- The weight-decay question is tested only in Arms D and E.

## Theory being tested

With layer 1 fixed, the output is linear in the layer-2 parameters: a fixed basis times coefficients. The data term plus a
quadratic penalty is then a convex quadratic. It can have flat directions but no tilted ones, so it cannot run away. If the
runaway needs composition, freezing layer 1 should remove it.

## Common settings

- **Polishing:** trust-exact (scipy), float64, exact Hessian, cap 500 iterations, the stopping rule of `stage2.run_exact`.
- **Certification:** the four-part criterion of `certify.py` (`certify_point`).
- **Logged at every 25th iterate and at the final iterate:**
  - squared norm per parameter block;
  - smallest and largest Hessian eigenvalue;
  - data term and each penalty term.
- **Growth:** the squared norm of the polished block divided by its early-stopped value.

## Arms

- **A) Frozen layer 1.** Start from the early-stopped weights, freeze layer 1 (and the salt embedding where present), and
  polish layer 2 only.
  - **R:** the 6 real KAN-delta seeds (`delta_s0`–`delta_s5`, 107 AdamW epochs, all 2069 anchored rows), with the
    original final.py loss: mean squared error + 1e-3 × ΣD2 over both layers, and no weight decay.
  - **S:** synthetic correlated P0 and correlated P1, 8 seeds each, each with its own cell loss, from the committed
    early-stopped states.
- **B) EXPLORATORY, no prediction.** Freeze layer 2 and polish layer 1 only (non-convex), for the same models as A. The salt
  embedding stays frozen.
- **C) One-layer control.** A single `kan.KANLayer` [4, 1] with a linear output and no tanh.
  - Trained from scratch on the same correlated synthetic training data: seeds 0–7, P0 loss (MSE + 1e-3 × ΣD2).
  - AdamW with early stopping exactly as in the synthetic study (lr 3e-3, wd 1e-4, full batch, patience 100, at most
    10,000 epochs, validation data term).
  - Then trust-exact and certification.
- **D) Polish-time L2.** Synthetic correlated P0, 8 seeds, re-polished from the same early-stopped weights with
  loss = committed P0 loss + 1e-4 × ‖θ‖² over all parameters.
  - Decoupled AdamW decay is not exactly an L2 term; this is its closest loss-side analogue.
  - Note: the equivalent loss of AdamW decay with wd = 1e-4 is (wd/2)‖θ‖² = 5e-5‖θ‖². The brief's 1e-4‖θ‖² is twice
    that; it is used as specified.
- **E) Training-time weight decay.** Retrain from scratch with AdamW weight_decay = 0 instead of 1e-4, everything else
  identical, up to the early-stopped state.
  - **(i)** Synthetic correlated P0, 8 seeds (same data, seeds and early-stopping rule).
  - **(ii)** Real KAN-delta on the fold A split, 5 seeds (0–4), submission protocol: 107 epochs, no early stopping.
  - **Comparison at the early-stopped state, wd = 0 against wd = 1e-4:**
    - squared parameter norm;
    - data term;
    - held-out error: in-distribution and off-manifold RMSE against the noiseless teacher for (i); the fold A 5-seed
      ensemble RMSE against 0.1420 for (ii).
  - **wd = 1e-4 references:** the committed early states (`synth_correlated_P0_s*`, `scbase_A_s*`).
  - **Determinism check:** one seed per sub-arm is retrained with wd = 1e-4 and compared with its committed weights, to
    confirm the protocol is identical.

## Pre-registered predictions

- **Z1. Arm A certified:** at least 5 of 6 real seeds, and at least 7 of 8 in EACH synthetic cell (reported per cell,
  never only pooled).
- **Z2. Arm A growth:** median layer-2 squared-norm growth (polished/early) ≤ 3 in R and in each synthetic cell.
- **Z3. Arm A convexity check:** smallest Hessian eigenvalue ≥ −1e-8 × largest at every logged iterate. This is an
  implementation check; a failure means a bug and is reported.
- **Z4. Arm C:** 8 of 8 certified, and median growth ≤ 3.
- **Z5. Arm D (two-sided, no direction predicted):** the verdict is which branch applies.
- **Z6. Arm E (two-sided, no direction predicted):** the verdict is which branch applies, given separately for E(i) and
  E(ii).

## Interpretation (fixed in advance)

- **Z1, Z2 and Z4 hold:** composition is necessary for the runaway; correlated inputs alone are not sufficient; the
  one-layer problem is well posed.
- **Z1 fails while Z2 and Z3 hold:** the layer-2 problem is convex but too ill-conditioned to certify in 500 iterations.
  Report its conditioning; the mechanism is unaffected.
- **Z2 or Z4 fails:** a runaway without composition, so the mechanism needs revision. Stop and report before anything else.
- **Arm D:**
  - at least 6/8 certified and median growth ≤ 3 → an L2 term of this size at polish time bounds the runaway;
  - at most 2/8 certified → it does not;
  - otherwise → a partial effect; report.
- **Arm E** (per sub-arm):
  - median early-stopped norm with wd = 0 at least 2 × the wd = 1e-4 norm → AdamW's decay materially bounded the
    early-stopped models. The paper then attributes their stability to early stopping AND weight decay and reports the
    wd = 0 fold A RMSE.
  - within 1.25 × → early stopping alone kept them bounded; weight decay was incidental;
  - otherwise → a partial effect; report.
- **Arms D and E** answer different questions. D asks whether an L2 term at polish time stops the runaway; E asks whether
  decay during training kept the early-stopped models bounded. They are reported separately, and their verdicts are never
  combined.

## Report

- A table per arm and cell: certified count, median (range) growth, stop reasons, smallest eigenvalue (Arm A), and for Arm
  E the norm ratio and held-out errors.
- Then the verdict for Z1–Z6 and which interpretation branch applies.

---
## Verdicts

(appended after the run; nothing above this line will be edited)

### Verdicts, part 1 (recorded 2026-10-17, from `freeze_output.txt`; run stopped by the pre-registered rule, decision record 11bf8bc)

| | Prediction | Verdict | Key numbers |
|---|---|---|---|
| Z1 | Arm A certified: ≥ 5/6 real; ≥ 7/8 in each synthetic cell | **HELD** | A-R 6/6; A-S correlated P0 8/8; A-S correlated P1 8/8 |
| Z2 | Arm A median layer-2 growth ≤ 3 in R and in each synthetic cell | **FAILED** | A-R ×582.8 (×206–×11,689); P0 ×4.07 (×1.36–×245); P1 ×22.9 (×1.51–×801) |
| Z3 | Arm A smallest Hessian eigenvalue ≥ −1e-8 × largest at every logged iterate | **HELD** | worst ratio +1.99e-10 (all values positive) |
| Z4 | Arm C: 8/8 certified, median growth ≤ 3 | **HELD** (computed before the stop) | 8/8; ×2.52 (×1.68–×2.85) |
| Z5 | Arm D | pending | — |
| Z6 | Arm E | pending | — |

By the fixed interpretation, Z2 failing selects the branch "a runaway without composition: the mechanism needs revision; stop
and report before anything else". The run was stopped and reported. On 2026-10-17 the author decided, after seeing Arm A's
results, to resume Arms D, E and B (option 2).

### Interpretation note (POST-HOC, written 2026-10-17 after the Arm A results; the author's reading; changes no verdict)

- **Z2's branch premise did not occur.** That premise is "a runaway without composition". Every Arm A model reached a
  certified, finite minimum with non-negative curvature (Z3). The failure reflects the growth threshold, which did not
  separate a distant finite minimum from unbounded growth. The pre-registered rules did not anticipate this.
- **Precedent:** growth without runaway already occurred in synthetic independent P0 (×12.5, 6/8 certified; Table 1). The
  paper's runaway criterion is non-certification, not growth magnitude.
- **Conditioning:** the Arm A condition numbers at the final iterate have per-cell medians of 2.2e7 (P0), 2.0e8 (R) and
  3.6e8 (P1).
  - A near-singular quadratic has its minimum far from the start: the displacement is about 1/λ_min along near-null
    directions.
  - Early stopping on such a problem acts like ridge regularisation (Yao, Rosasco & Caponnetto 2007; Ali, Kolter &
    Tibshirani 2019).
- **Comparison with the full model:** Arm A's real median growth is ×583 (finite, certified). The full real model's growth
  from the same 107-epoch state to the end of stage-2 Part A is ×690 (never certified).
  - The two factors cover different parameter blocks. Arm A covers layer 2 only; the full polish covers layer-1 affine,
    SiLU and layer-2 affine. Their closeness is therefore weak evidence.
  - Two readings remain:
    - **(a) A receding minimum:** each frozen slice has a finite minimum, but layer 1's drift raises hidden-unit
      correlation and pushes that minimum outward, so the joint problem has none reachable.
    - **(b) A distant finite minimum** that the coupled, non-convex optimisation cannot reach within the budget.
  - The infimum test (a certified L2 regularisation path) is designed to distinguish them.
- **Arm B bears on (a):** with layer 2 frozen, layer 1 sees fixed raw inputs, so under (a) it should also certify, possibly
  with growth.
- **Arm D's rule** also uses a growth threshold (≤ 3) and shares Z2's weakness; this is noted next to Z5. Arm E uses a norm
  ratio, not this threshold.

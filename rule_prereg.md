# Pre-registration: basis-size rule for co-solvent-determined inputs (rule)

Written and committed before `rule.py` existed (2026-09-29). Verdicts are appended below the
"Verdicts" heading after the run; nothing above it is edited afterwards. Thresholds are the brief's,
unchanged.

## Anticipated gate outcome (from committed files, stated before any code)
`support_output.txt` shows that eps_co, lneta_co and M_co each take **30 distinct values** in the
2069 anchored training rows. If n_co (distinct co-solvent value triples) is also 30:
- the rule gives 3 × (K + 1) ≤ 29, so **K = 8**;
- that equals `final.py`'s 8 RBF centres, so **0 columns are removed**;
- the gate therefore **stops** the study.

Step 0 computes this on the data; the anticipation does not replace it. If the gate stops, Steps 1–4
are not run, I1–I5 are inconclusive, and the close-out section records that.

## Discrepancies and operational choices (files take precedence)
- **R1. "Layer-1"** means the input → hidden layer (`layers.0` in `kan.py`), as in `support_prereg.md`.
- **R2. The group.** Inputs whose value is constant within each distinct co-solvent combination.
  - A combination is identified by the distinct (eps_co, lneta_co, M_co) triple. These three are
    computed from the non-PC solvents only (`kan.co_props`), so a combination means a co-solvent
    set with its mole-weighted proportions.
  - Every continuous input is tested: an input joins the group if its within-combination range is
    ≤ 1e-9 for every combination.
  - n_co = the number of distinct triples.
- **R3. Numerical rank of the group block:** the columns of M (`support.design_matrix`) that belong
  to the group's inputs. Rank = the number of singular values with σ/σ_max ≥ 1e-2.
- **R4. Modified basis** (if the gate passes):
  - Each group input gets SiLU plus K RBF centres linspace(−1, 1, K), width 2/(K − 1) (final.py's
    spacing rule). For K = 1 there is one centre at 0 with width 2.
  - The smoothness penalty on a group edge uses second differences of its K coefficients, and is
    absent if K < 3.
  - All other edges are unchanged.
- **R5. Columns removed** = (9 − (K + 1)) × (number of group inputs).
  **GATE: if this is < 6, stop.**

## Step 0: feasibility gate (no training)
Report the distinct values per continuous input, the group, n_co, K, the group block's column count
and numerical rank, and the columns removed. Apply the gate.

## Step 1: train the modified model (only if the gate passes)
- **Training:** exactly as `final.py` (features, x_co · h head, [d_in, 6, 1], AdamW lr 3e-3 /
  wd 1e-4, 107 epochs, penalty 1e-3) apart from R4. Seeds 0–5.
- **Polishing:** exactly as `fiber2.py`: float64 L-BFGS, strong Wolfe, history 100, 50-iteration
  steps, cap 20000. Converged when max|grad| < 1e-7 **and** the relative change over 50 iterations
  is < 1e-12.
- **Report per seed:** converged or not, max|grad| at the cap, the last loss change per 50
  iterations, and the final loss.
- **Weights:** a third tarball, `../battery_hacketon_weights/gauge_weights_2026-09-29c.tar.gz`,
  appended to `weights_manifest.txt`.

## Step 2: compare with the baseline (`fiber2_ind_{0..5}.pt`), identical code for both
- **d_AB on delta** for all 15 seed pairs: RMS(delta_A − delta_B) / std of the data target delta.
- **Edge correlation outside the group** (1000/T, molality, mix_eps, mix_lneta, x_co):
  - Match units Hungarian on contribution |corr|.
  - Remove the shared per-unit scale: a is fitted in function space on the continuous layer-1
    pre-activation, as in `support.py`.
  - Compute the corr of the centred edges on a 100-point grid over the training range, with the
    sign set by sign(a).
  - Take the median over matched unit pairs × those 5 inputs × 15 seed pairs.
- **dim(N) of M** for each basis (reported only; it shrinks by construction).
- **Hessian:** exact, float64, over all parameters that affect the loss (unused salt embeddings
  excluded, as in support D2). Report the 10 algebraically smallest eigenvalues and the 10 smallest
  |λ| at each capped seed.

## Step 3: accuracy
Train the modified model on the fold A split (all anchored rows except PC+EA) with seeds 0–4,
107 epochs, and take the ensemble mean. Report RMSE and bias on the 635 anchored PC+EA rows.
KAN-delta is recomputed with the same code for the side-by-side comparison.

## Predictions
- **I1.** The modified seeds' median max|grad| at the cap is ≤ 0.1 × the baseline median.
- **I2.** Their median d_AB on delta is ≤ 0.5 × the baseline median.
- **I3.** Their median final training loss is ≤ 1.10 × the baseline median.
- **I4.** Fold A RMSE ≤ 0.150.
- **I5.** Their median edge correlation outside the group is ≥ the baseline's + 0.1.

If the gate stops, all five are **inconclusive (not run)**.

## Interpretation (fixed in advance)
- **I1, I2 and I3 hold:** excess basis on the co-solvent-determined inputs causes a large part of the
  flat directions and of the seed disagreement.
- **I3 fails:** the removed columns were not unconstrained, so the design rule is wrong as stated.
- **I3 holds but I1 and I2 fail:** removing the columns doesn't fix the landscape; the slow
  convergence has other causes (mixture-descriptor correlation, saddle regions).
- **I5 is secondary:** it tests whether fixing the group also stabilises the other edges.
- **Any other combination** is reported as found.

## Step 4: close-out
Append an "Intervention" section to `electrolyte_case_study_summary.md`, with the I1–I5 table
(verdict, key number, commit) and one paragraph on what the intervention does and does not
establish. Commit it.

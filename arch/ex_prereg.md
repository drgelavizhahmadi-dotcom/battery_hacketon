# EX-KAN (two-anchor excess model): pre-registration

Committed before `arch/ex_model.py` or `arch/ex_run.py` exists, and before any EX-KAN model has been trained.
Verdicts will be appended below this text; the text itself and every recorded verdict will not be edited.
Files win over this document; discrepancies found while running go at the top of `arch/ex_output.txt`.
Rules: no weights in git (weights go to `gauge_weights/`, archived with `weights_manifest.txt` entries); inputs are held
in memory after SHA verification. No polishing in this study.

## Flags (recorded before any run)

1. **g takes only x_co-independent inputs (decision B).** The inputs are eps_co, ln_eta_co, M_co, 1000/T, molality
   and the salt embedding (tanh). g no longer sees x_co, eps_mix or ln_eta_mix.
   - Reason: with all 10 final.py inputs, g sees x_co directly and through the mole-weighted mixture features. Then
     s → s + f(props, T), g → g − f/(1 − x_co) leaves predictions unchanged over the data range (x_co < 1), so s
     would be pinned only by g's inductive bias.
   - With B, δ/x_co = s + (1 − x_co)·g is linear in x_co within a group of fixed co-solvent, T, salt and molality,
     and s is that line's value at x_co = 1. Distinct x_co values therefore identify s, and the ≤ 2 distinct-x_co
     flag (Step 0a) keeps its intended meaning.
2. **g's input count is 7, not 8.** The decision message said "g = KAN [8, 6, 1]", but its input list is 5
   continuous features plus the 2-number salt embedding, which is 7. The "8" came from my own miscount in the
   question I put to you. The input list wins: g = KAN [7, 6, 1], i.e. `kan.KAN(d_num=5, widths=[6, 1])`.
3. **s has no salt or molality input.** The "1 mol/kg" in E3 has no effect on s, and the implied pure-co-solvent
   offset (log k(pure co-solvent) − log k(PC) = s at x_co = 1) is the same for every salt and concentration by
   construction.
4. **Building s:** `kan.KAN` always adds a salt embedding, so s is two `kan.KANLayer`s (4 → 3, tanh, 3 → 1) with no
   embedding, as in `paper2/synth.py`.
5. **Multi-co-solvent rows:** s and g are evaluated at the mole-weighted mix of the non-PC solvents' properties, as
   final.py's eps_co, lneta_co and M_co already are.
   - Some co-solvents occur only in blends with another co-solvent: 2-MeTHF, THF, 3-Glyme and 2-Glyme. For these,
     s at the pure co-solvent's properties may lie outside the property range of the training rows. Step 0
     reports, for each co-solvent, whether it occurs as the sole co-solvent.
   - For mol/L rows, molality is c/ρ_mix, so it shifts slightly with composition inside a nominal group. Step 0
     reports this.
6. **E3 measure:** per full-set seed, the vector of s values (at 25 °C, 1000/T = 3.354, and the pure co-solvent's
   properties) over the co-solvents not flagged in Step 0a. E3 uses the median of the pairwise Pearson correlations
   over the 45 seed pairs. It needs ≥ 3 unflagged co-solvents; otherwise E3 is inconclusive.
7. **E5 measure:** a co-solvent's spread is the SD of its s (same evaluation point) over the 10 full-set seeds. E5
   compares the median spread of the flagged co-solvents with that of the unflagged ones. It is inconclusive if
   Step 0a flags no co-solvent, or flags all of them.
8. **E4 has a small comparison group.** Only 2 co-solvents have ε ≥ 10: EC (89.8) and Sulfolane (43.3). The low-ε
   group (ε < 10) is EA, 2-MeTHF, Methylene chloride, THF, DME, Toluene, 3-Glyme and 2-Glyme.
   - Sulfolane's high viscosity (10.3 cP, against EC's 1.9) works against the prediction under Walden-type
     reasoning.
   - If E4 fails, each high-ε co-solvent's s is reported separately, with a statement of whether Sulfolane drives
     the failure.
9. **The baseline's numbers are already known.** Its early models (same folds, seeds and protocol: final.py, 107
   epochs) are the SHA-indexed `scbase_*` files from the SC study. They are reused after SHA verification, not
   retrained. Their accuracy was reported in e00e37f: LOCO unweighted mean 0.2554, fold A 0.1420.
   - E1's threshold is therefore 0.95 × 0.2554 = 0.2426, recomputed from the reloaded models.
   - E2 keeps the fixed 0.142. EX is also reported against the baseline's value in this same run.
   - The full-set baseline (all anchored rows) is not needed for any prediction and is not trained.
10. **Fold A and the LOCO EA fold are the same rows** (as in arch/sc_prereg.md, flag 7). E2 and the EA fold of E1
    are not independent. The same 10 EX models serve both.
11. **DEC never appears in training.** Its s in the test set is extrapolated in property space (its ε of 2.81 is
    near Toluene's 2.38). No prediction depends on it; Step 0c only describes the test set's x_co distribution.

## Model: EX-KAN

log k = anchor_PC + x_co · s(eps_co, ln_eta_co, M_co, 1000/T) + x_co · (1 − x_co) · g(eps_co, ln_eta_co, M_co, 1000/T, molality, salt)

- So δ = log k − anchor_PC = x_co · [s + (1 − x_co) · g].
- **s:** KAN [4, 3, 1], made of `kan.KANLayer`s with tanh between them and no embedding (flag 4). It has about 139
  parameters.
- **g:** `kan.KAN(d_num=5, widths=[6, 1], n_salt=12)`, i.e. [7, 6, 1] including tanh(salt embedding) (flag 2). It
  has about 463 parameters. Together with s, that is about 602; the baseline KAN-delta has about 625.
- **Edges:** final.py edges, SiLU + 8 Gaussian RBFs on linspace(−1, 1, 8) with width 2/7.
- **Inputs:** final.py's `Scaler`, identical to final.py and to arch/sc.
- **Loss:** MSE on δ + 1e-3 · (ΣD2 over s's edges + ΣD2 over g's edges), final.py's smoothness penalty.
- **Training:** AdamW exactly as final.py (lr 3e-3, wd 1e-4, full batch, float32, 107 epochs, no early stopping).
  `torch.manual_seed(seed)` is set before s and g are constructed. The result is the "early" state, and there is no
  polishing.

## Step 0: data checks

No training. Report the results, then stop: you decide before Step 1.

- **a)** Per co-solvent (rows containing it, anchored training rows):
  - the number of rows, and the range, distinct count and SD of x_co;
  - whether it occurs as the sole co-solvent;
  - within-group distinct x_co: groups of fixed co-solvent composition, T level, salt and molality, with how far
    mol/L molality varies inside a group.
  - **Flag** co-solvents measured at ≤ 2 distinct x_co values over all their rows: s and g are not separable there.
    This flagged set defines E3 and E5.
- **b)** Rows with pure co-solvent (x_co = 1) or x_co ≥ 0.9, per co-solvent.
- **c)** DEC test set: the x_co distribution, for all test rows and for the anchored test rows.
- **d)** LOCO folds as in arch/sc_prereg.md (flag 6 there), plus fold A. The table is checked against arch/sc's.

## Step 1: training (after your go-ahead following Step 0)

- **EX-KAN**, 10 seeds (0–9) for each of:
  - the 10 LOCO folds;
  - fold A (= the EA fold, flag 10);
  - the full set: all 2069 anchored rows, used for E3–E5 only.
- **Baseline KAN-delta:** the cached `scbase_*` early models, reused after SHA verification (flag 9).
- **Accuracy:** the 5-seed ensemble of seeds 0–4.

## Step 2: measurements

- **Accuracy:**
  - LOCO RMSE (unweighted mean over the 10 folds, plus row-weighted and folds with ≥ 20 test rows, as in arch/sc);
  - fold A RMSE, for EX and the baseline.
- **Pure-co-solvent offset (full set, per seed):** s at 25 °C and each co-solvent's pure properties.
  - Report it per co-solvent: the median and SD over seeds.
  - Also report the per-co-solvent s at the 25 °C rows' observed property values, where they exist.
- **E3 and E5:** as in flags 6 and 7.
- **E4:** per seed, mean s over the low-ε co-solvents vs mean s over the high-ε ones.

## Pre-registered predictions

- **E1. Accuracy:** EX LOCO RMSE (unweighted mean, as in arch/sc) ≤ 0.95 × KAN-delta's.
- **E2.** EX fold A RMSE ≤ 0.142 (no worse than KAN-delta).
- **E3. Pure co-solvent offset:** s at 25 °C agrees across full-set seeds: the median pairwise correlation over
  unflagged co-solvents is ≥ 0.9 (flag 6).
- **E4. Physics ordering:** per full-set seed, the mean s over low-ε co-solvents (ε < 10) is below the mean s over
  high-ε ones (EC, Sulfolane), in ≥ 8 of 10 seeds (flag 8).
- **E5. Separability:** the median seed-SD of s over co-solvents flagged in Step 0a is ≥ 2 × the median over
  unflagged co-solvents (flag 7; the data-design layer, predicted in advance).

## Interpretation (fixed in advance)

- **E1 and E2 hold:** the two-anchor structure improves extrapolation to unseen solvents.
- **E1 fails, E3 and E4 hold:** no gain in accuracy, but an interpretable, stable pure-co-solvent prediction.
- **E5 holds:** s is identifiable only where compositions vary. This is a direct demonstration of the data-design
  layer of non-identifiability.
- **E4 fails:** the log-linear ideal line is a poor baseline for low-dielectric solvents. Record this and consider
  an ideal line in another variable; report each high-ε co-solvent separately (flag 8).
- **Other combinations:** reported plainly, with no further reading.

## End

A verdict table for E1–E5, with the key numbers and commits. Do not start further stages.

---
## Verdicts

(appended after Step 2; nothing above this line will be edited)

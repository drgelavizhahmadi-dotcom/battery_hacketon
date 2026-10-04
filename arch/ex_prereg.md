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

## Amendment 1 (2026-10-04, decided after Step 0 / commit 4c268c1, before any EX-KAN model was trained)

Reason: Step 0 showed that, under decision B, s is identified only where x_co varies at a **fixed blend composition**.
The committed ≤ 2 distinct-x_co rule counts x_co values that come from different blend ratios, and each ratio also
changes eps_co, ln_eta_co and M_co. By that rule, 2-MeTHF, THF, DME and Toluene are unflagged, but they have no
fixed-composition group with two x_co values.

- **PRIMARY flag rule (used for the E3 and E5 verdicts):** a co-solvent is **SEPARABLE** if it has ≥ 1
  fixed-composition group with ≥ 2 distinct x_co values. The grouping is the concentration-free one, D5 in
  `arch/ex_output.txt`: same source, salt, temperature and blend ratio.
  - **Separable:** EA (11 of 11 groups), EC (40 of 240) and Methylene chloride (8 of 80).
  - **Flagged:** 2-Glyme, 2-MeTHF, 3-Glyme, DME, Sulfolane, THF and Toluene.
  - E3 uses the 3 separable co-solvents. E5 compares the median seed-SD of the 7 flagged with that of the 3
    separable.
- **SECONDARY (reported alongside, no separate verdict):** E3 and E5 are also computed under the committed ≤ 2
  distinct-x_co rule: 7 unflagged; 3 flagged (2-Glyme, 3-Glyme, Sulfolane).
- **E3 caveats:**
  - With 3 separable co-solvents, the pairwise correlation (over 3 values per seed) is a weak test.
  - Also reported, descriptively: each separable co-solvent's seed SD of s, and that SD relative to the spread of s
    across co-solvents. The spread is defined as the SD, over all 10 co-solvents, of their seed-median s.
  - Methylene chloride is marginal: 8 of 80 groups, at a single nominal concentration, so its separation is partly
    via the ~4% mol/L molality drift. This is noted next to E3.
- Nothing above this section has been edited.

## Verdicts (recorded 2026-10-04)

Source: `arch/ex_output.txt` and `arch/ex_results.json`. Step 0 is in 4c268c1 and Amendment 1 in fa6ae2a. The models
were early states (AdamW, 107 epochs, no polishing). The baseline was the cached `scbase_*` models, SHA-verified.

**Note:** fold A and the LOCO EA fold are the same rows (flag 10), so E2 and the EA fold of E1 are not independent.

| | Prediction | Verdict | Key numbers |
|---|---|---|---|
| E1 | EX LOCO ≤ 0.95 × KAN-delta | **FAILED** | unweighted mean EX 0.2834 vs threshold 0.2426 (base 0.2554; ratio 1.110). Row-weighted 0.2363 / 0.2297 (1.029); folds ≥ 20 rows 0.2662 / 0.2093 (1.272). EX is better on EC (0.237 vs 0.256), Sulfolane, 3-Glyme, 2-Glyme and Toluene, and worse on 2-MeTHF, THF, Methylene chloride, DME and EA |
| E2 | EX fold A ≤ 0.142 | **FAILED** | EX 0.1502 (base this run 0.1420) |
| E3 | s agrees across seeds (separable co-solvents, Amendment 1) | **HELD** | median pairwise corr +0.987 over EA, EC, Methylene chloride: a weak test with 3 values per seed, and Methylene chloride is marginal. Seed SD of s: EA 0.063 (0.39 × the spread across co-solvents, 0.159), EC 0.091 (0.57 ×), Methylene chloride 0.067 (0.42 ×). Secondary (≤ 2 rule, 7 co-solvents): +0.869 |
| E4 | mean s(low ε) < mean s(EC, Sulfolane) in ≥ 8/10 seeds | **FAILED** | 0/10: the ordering is reversed in every seed. Per high-ε co-solvent: vs EC alone 0/10 (EC median s +0.104), vs Sulfolane alone 0/10 (median −0.124). The low-ε mean is +0.19 to +0.49 per seed. Sulfolane does not drive the failure on its own: EC alone fails in every seed too |
| E5 | median seed-SD flagged ≥ 2 × separable (Amendment 1) | **FAILED** | 0.091 vs 0.067, ratio 1.37. Secondary (≤ 2 rule): 0.222 vs 0.069, ratio 3.22, no verdict; the secondary is driven by 3-Glyme (SD 0.478, M_co extrapolated) and 2-Glyme (0.222) |

Tally: 1 held (E3), 4 failed (E1, E2, E4, E5).

**Interpretation, applying the fixed rules:**
- **E1 and E2 fail:** the two-anchor structure does not improve extrapolation to unseen solvents. (The rule "E1
  fails, E3 and E4 hold" is not met, because E4 fails.)
- **E4 fails:** the log-linear ideal line is a poor baseline for low-dielectric solvents as predicted. In every seed,
  the model puts the low-ε solvents *above* both high-ε ones, i.e. it predicts their pure-solvent conductivity is
  higher relative to PC. Record this and consider an ideal line in another variable.
- **E5 fails under the primary rule:** the data-design layer is not demonstrated. Flagged co-solvents are only 1.37 ×
  as variable across seeds as separable ones.
  - The committed ≤ 2 rule gives 3.22, but that contrast is carried by the two glymes. One of them is evaluated
    outside the training range of M_co.

**POST-HOC, descriptive (decided after seeing the results; changes no verdict):** across the 10 co-solvents, the
seed-median s is ranked by viscosity more than by dielectric constant. Spearman with ln η is −0.85 (p = 0.002); with
ε, −0.50 (p = 0.14). The ordering is Sulfolane (10.3 cP) lowest, then 3-Glyme, EC and 2-Glyme, with the ~0.4–0.6 cP
ethers and esters highest. That is Walden-type behaviour, the reasoning flag 8 raised against Sulfolane, here applying
to the whole set.

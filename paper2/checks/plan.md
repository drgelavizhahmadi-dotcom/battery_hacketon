# paper2/checks: descriptive checks for paper 2's open items (plan)

Committed before `paper2/checks/checks.py` exists. These checks are **descriptive and post-hoc**: no training, no
verdict changes, and nothing here is a pre-registered prediction. Inputs are SHA-verified against
`weights_manifest.txt` and held in memory. Results go in `paper2/checks/checks_output.txt`, a results JSON and one
figure for item 1.

## Item 1: hidden-unit correlation (claim F4)

**POST-HOC EXPECTATION (not a pre-registered prediction):** models that ran away have more strongly correlated hidden
units than models that converged.

**Models**, each evaluated on its own training rows:

| Group | Models | States | Hidden units | Training rows |
|---|---|---|---|---|
| Synthetic students | all 48 (2 × 3 cells × 8 seeds; weights archive h) | early, polished | 3 | the cell's 2000 rows |
| Real-data seeds | 6 (ind 0–5) | fiber2 capped (`fiber2_ind_*`), stage2 Part A final (`stage2_A_ind_*`, flat vectors rebuilt on their fiber2 start) | 6 | all 2069 anchored rows |
| SC-KAN-delta | 10 fold A models (`sc_A_b0.0001_s*`) | early, polished | 6 | the 1434 fold A training rows |

**Per model:**
- the hidden-unit activations after tanh (the values entering the next layer);
- their correlation matrix;
- the median and maximum absolute off-diagonal correlation;
- the effective number of independent hidden units, as the participation ratio (Σλ)²/Σλ² of the activation
  covariance eigenvalues.

**Extras, labelled as such**, to give each group a converged reference:
- the real-data seeds' 107-epoch AdamW state (`delta_s*`);
- the baseline KAN-delta fold A models from arch/sc (`scbase_A_s*`, early and polished, 0/10 certified).

**Comparisons:**
- **Synthetic:** certified vs not certified, independent vs correlated cells, early vs polished.
- **Real data:** per stage (107 epochs → fiber2 capped → Part A final).
- **SC (all 10 certified) vs the baseline:** on the same fold A rows.

**Closing paragraph:** does hidden-unit correlation support the claim that the output layer can host the runaway even
with independent raw inputs? With the numbers.

## Item 2: how affine is the tilt (claim F2)

**Edge type:** final.py's, i.e. 8 Gaussian RBFs with centres linspace(−1, 1, 8) and width 2/7, on the grid range
[−1, 1]. Inputs are min–max scaled to [−1, 1], and hidden inputs are tanh values.

**Method:**
- Take the edge function produced by a unit linear coefficient pattern: c_k ∝ k − mean(k), normalised to unit norm.
- Fit the best straight line (least squares, dense grid) over the full range.
- Report the max and RMS deviation relative to the function's range (max − min).
- Repeat over the central 80% of the range ([−0.8, 0.8]).
- For reference, also report the constant pattern (c_k = 1) and its deviation from a constant.

The synth model and SC-KAN-delta use the same `kan.KANLayer` edge type (same centres, width and range); this is
checked and stated, not assumed.

## Item 3: reference commits and data citation

- **Claim A1:** the commit that produced the final.py KAN-delta ensemble, and the commit that first recorded its fold A
  RMSE of 0.142 (kan.py's fixed-epoch KAN-delta, 0.1420). Both are found in git history.
- **The CALiSol-23 dataset citation** as given in the competition material (the files in `ca-li-sol-23-challenge/`)
  or the repo docs: authors, title, venue / DOI if present. If neither carries a full citation, say so; the published
  article is only reported as found elsewhere, not presented as coming from the competition material.

Do not start further stages.

# Edge decomposition of the fold A ensembles (POST-HOC, DESCRIPTIVE)

Committed before `paper2/revision/edgedecomp.py` exists. POST-HOC: defined after every fold A result is known; it makes no
predictions, issues no verdicts and changes no verdict. No training and no optimisation: archived weights only, SHA-verified
against `weights_manifest.txt` and held in memory. Files win over this plan; discrepancies go at the top of
`paper2/revision/edgedecomp_output.txt`.

## Question

On training rows, do large opposing edge contributions cancel to a small net; on fold A rows (ethyl acetate held out), does
that cancellation fail, and does the failure account for the polished prediction change?

## Models (fold A ensembles, 5 seeds each, as in `uncertainty.py`)

- **early:** `scbase_A_s0–4` (107 AdamW epochs on fold A's training rows).
- **polished, μ = 0:** `scbase_A_s0–4_pol` (arch/sc baseline polish), mapped with `trust.Flat.to_kan`.
- **polished, μ = 1e-6:** `close_foldA_1e-06_s0–4`; **polished, μ = 1e-4:** `certify_foldA_0.0001_s0–4`; mapped with `close.FlatQ.to_kan`.
- **Integrity:** the ensembles must reproduce the logged fold A RMSEs 0.1420, 0.8147, 1.0262 and 0.3345, or the run stops.

## Decomposition (exact, from `kan.KAN`)

- Prediction: δ = x_co · h, with h = Σ_j ψ_j(u_j) + b₂, u_j = tanh(s_j), s_j = Σ_i φ_ij(z_i) + b₁ⱼ.
  - z: the 8 scaled continuous inputs and the 2 tanh salt-embedding coordinates.
  - φ_ij and ψ_j: the edge functions without bias (`KANLayer.edge`).
- **Layer 2, per row:** S₂ = Σ_j |ψ_j(u_j)| (sum of absolute contributions); N₂ = |Σ_j ψ_j(u_j)| (absolute net);
  cancellation C₂ = S₂ / N₂.
- **Layer 1, per row and hidden unit:** S₁ⱼ = Σ_i |φ_ij(z_i)|; N₁ⱼ = |Σ_i φ_ij(z_i)|; reported as the median over units
  of S₁ⱼ and N₁ⱼ, and of C₁ⱼ = S₁ⱼ / N₁ⱼ.
- The decomposition is checked to reproduce the network output to 1e-6 (relative) on every row.

## Rows and reporting

- **Training rows:** fold A's 1434 training rows. **Fold A rows:** the 635 held-out rows.
- **Per state and row set:** the medians over rows of S₂, N₂, C₂, the layer-1 medians, and the median |δ| (per seed; then
  the median over the 5 seeds).
- **Accounting for the polished change.** On each row, Δh = h_polished − h_early (per seed), split into:
  - the change in the layer-2 edge sum, Δ(Σ_j ψ_j);
  - the change in the bias, Δb₂.
  Reported: the share of Σ (x_co Δh)² over fold A rows carried by x_co Δ(Σ_j ψ_j), and the same on training rows.
  Also reported: the Spearman correlation, over fold A rows, between |x_co Δh| and the change in N₂.
- **Figure** (`figs/edgedecomp.png`):
  - (a) S₂ against N₂ per row, training vs fold A, for early and for polished μ = 1e-6 (seed 0);
  - (b) the median C₂ per state and row set, over seeds.

## Reading (descriptive; no thresholds)

"Cancellation fails on fold A" is said only if, for a polished state, the median N₂ / S₂ on fold A rows exceeds that on
training rows, and the layer-2 edge-sum change carries most of Σ (x_co Δh)² on fold A. Otherwise the numbers are reported as
found. The result is labelled POST-HOC wherever it is used.

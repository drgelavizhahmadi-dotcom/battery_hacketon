# Pre-registration: where do KAN-delta models differ, and why? (support)

Written and committed before `support.py` existed (2026-09-29), and after the inputs were pinned
(`b80a92c`: `fiber2_*` weights in `gauge_weights_2026-09-29b.tar.gz`, listed in
`weights_manifest.txt`). Verdicts are appended below the "Verdicts" heading after the run; nothing
above it is edited afterwards. Thresholds are the brief's, unchanged.

## Flags
- **F-a.** No model in this study is converged: in `fiber2`, 0/26 certified. All comparisons are
  between points on gradient-descent trajectories, including H6's two student states.
- **F-b.** Hessians are evaluated at non-stationary points. Eigenvalues may be slightly negative, and
  are reported as they are.
- **F-c.** N is defined from the inputs and the basis only. `support.py` computes and logs Step 1
  before any trained weights are loaded.

## Discrepancies with the brief, and operational choices (files take precedence)
- **D1. Naming.** The brief's "layer-1 edges" (input → hidden) are `layers.0` in `kan.py`; the output
  layer is `layers.1`. "Layer-1" below means the brief's usage.
- **D2. Hessian parameters.** The network has 625 parameters. 16 of them are the 2-dim embeddings of
  the 8 salts that never occur in the anchored training rows (only salt indices 0, 1, 3, 9 occur).
  Their gradient and Hessian rows are exactly zero, which would give 16 zero eigenvalues by
  construction. **They are excluded; the Hessian is over the other 609 parameters.**
- **D3. Prediction names in the summary.** Control 2's committed predictions are **C1–C3** (there is
  no C4). Control 3's are **C1', C2', C4** (no C3'; its fourth is named C4, not C4'). The first
  control (`gauge_prereg.md`) has a pass criterion but no letter; it appears as "Control 1".
  The summary uses the committed names.
- **D4. The "5–7×" statement.** `support.py` computes every student's training loss in both states.
  The close-out states the measured ratio. If it falls outside 5–7×, the measured value is used
  and the brief's figure is flagged.
- **D5. When H3/H4 can't be met.** If dim(N) > 24, then 3 × dim(N)/72 > 1, so the "≥ 3× random"
  part of H3/H4 is impossible. In that case H3/H4 are **inconclusive (untestable as specified)**,
  and the fraction is still reported.

## Step 1: near-null space of the layer-1 design matrix (no trained weights)
- **Design matrix M:** built on the 2069 anchored training rows, in the scaled inputs used by
  `final.py`. For each continuous input i (the 8 inputs x_co, eps_co, lneta_co, M_co, mix_eps,
  mix_lneta, invT, molal), the columns are silu(z_i) and exp(−((z_i − g_k)/h)²) for k = 1..8,
  with g = linspace(−1, 1, 8) and h = 2/7.
  - The order is input-major, which gives 72 columns; any other count is flagged.
  - The salt-embedding edges are excluded.
  - A unit j's coefficient vector uses the same order: [base.weight[j, i], coef[j, i, 0..7]]
    for each input i.
- **SVD:** raw coordinates, no column scaling. The relative singular value is σ/σ_max.
  N = the right singular vectors with σ/σ_max < 1e-2.
- **Report:** the spectrum, dim(N), each N-vector's energy per input (the squared norm of its
  9 entries), and a class for each:
  - **REDISTRIBUTION:** at least 2 inputs each carry ≥ 20% of the energy.
  - **WITHIN-EDGE:** a single input carries ≥ 90%.
  - **OTHER:** neither.
- **Also reported:** the correlation matrix of the 8 inputs.
- **Constant shifts:**
  - (a) the per-unit layer-0 bias represents a constant exactly;
  - (b) whether the constant function is in each edge's 9-column span: the relative residual of
    a least-squares fit of 1 on those columns;
  - (c) sensitivity only, not used for verdicts: dim(N) of column-centred M.

## Step 2: where do models differ?
- **Sets**
  - **(A)** the 6 independent seeds in their fiber2 capped state (`fiber2_ind_{0..5}.pt`), all
    15 pairs, with the lower seed as source.
  - **(B)** the teacher `delta_s0.pt` (unpolished) as source, and each of the 20 control-3 (R)
    students as target, in both states: (i) `control3_R_s{100..119}.pt` and
    (ii) `fiber2_stu_{100..119}.pt`.
- **Matching:** Hungarian on the |corr| of unit contributions over the training rows, as in
  `fiber.py`. Every unit's match correlation is reported, for set B in both states.
- **Gauge-removed difference:** for each matched unit pair (source θ_A, target θ_B ∈ R⁷²),
  fit a and c in **function space**: min ‖Mθ_B − a·Mθ_A − c·1‖².
  - The difference is Δ = θ_B − a·θ_A.
  - The fraction in N is ‖V_Nᵀ Δ‖² / ‖Δ‖², against a random expectation of dim(N)/72.
  - **Constant shifts** are handled by the free c, which the unit bias absorbs. How exactly this
    works follows from Step 1(b) and is flagged in the log.
  - **Sensitivity only:** the same fraction with the scale fitted in parameter space,
    a = ⟨θ_A, θ_B⟩ / ⟨θ_A, θ_A⟩.
- **Report:** medians over the matched unit pairs in each set, split by match |corr| ≥ 0.8 vs < 0.8.

## Step 3: full Hessian
- **Models:** the 6 capped independent seeds and 3 capped students. The students are chosen by
  final max |grad| in `fiber2_stu_meta.json`: the lowest, the 10th in sorted order (lower median),
  and the highest.
- **Hessian:** exact, float64, `torch.autograd.functional.hessian`, of each model's own training
  loss (MSE + 1e-3 · penalty; data targets for the seeds, teacher delta for the students), over the
  609 parameters in D2.
- **Eigenvectors:** the 10 smallest (algebraic) eigenvalues of each model are analysed. For each
  eigenvector v:
  - v_j = the layer-1 continuous block of unit j (72 entries each, 6 units).
  - **Pooled overlap** Σ_j ‖V_Nᵀ v_j‖² / Σ_j ‖v_j‖². This is the H5 metric.
  - The per-unit overlaps, and the share of ‖v‖² that lies in the layer-1 continuous block.
  - The Step-1 class, applied to the per-input energy pooled over units.

## Predictions
- **H1.** dim(N) ≥ 5.
- **H2.** ≥ 50% of the vectors in N are REDISTRIBUTION *and involve the correlated group*: at least
  2 of their ≥ 20%-energy inputs are in {eps_co, lneta_co, M_co, mix_eps, mix_lneta, x_co}.
- **H3.** Set A: the median fraction in N is ≥ 0.5 **and** ≥ 3 × dim(N)/72 (see D5).
- **H4.** Set B, capped state: the same criterion as H3.
- **H5.** Over the 10 flattest Hessian directions of all 9 models (90 eigenvectors), the median
  pooled overlap is ≥ 0.5.
- **H6.** Set B: count the students whose median fraction in N over their 6 units is higher in the
  capped state than in the control-3 state. H6 holds if that count is ≥ 15 of 20.

## Interpretation (fixed in advance)
- **H2–H5 hold:** seed disagreement and slow convergence are explained mainly by approximate
  symmetries created by correlated inputs (the data-design layer).
- **H3 and H5 hold, but N is mostly WITHIN-EDGE** (> 50% of N-vectors): SiLU/RBF basis redundancy
  is the main cause.
- **H3 and H5 both fail:** neither explanation holds; the cause is reported as open.
- **H6 holds:** fitting improves along constrained directions while differences persist along
  near-null ones, as the approximate-symmetry mechanism predicts.
- **Any other combination** is reported as found, without choosing an explanation.

## Step 4: close-out
`electrolyte_case_study_summary.md`:
- **One table** of every pre-registered prediction in the case study (Control 1, P1–P6, C1–C3,
  C1'/C2'/C4, F1–F5, G1–G5, H1–H6), with verdict, key number and commit.
- **The loss-ratio statement** (D4).
- **Three paragraphs:** what is identified, what is not and why, and which claims can go into
  the paper.

The document is committed.

Outputs: `support.py`, `support_output.txt`, `support_results.json`, `figs/support/`.

---

## Verdicts (appended after the run; `support_output.txt`, `support_results.json`, `figs/support/`)

**Run notes.**
- The first run crashed in Step 3 on an indexing bug: the end offset of each unit's RBF-coefficient
  slice was missing. After the fix, the script was re-run in full; it is deterministic. Steps 1–2
  were identical in both runs.
- Two consistency checks passed:
  - M·θ_j reproduces the summed continuous layer-1 edges to 2.5e-7.
  - Each Hessian loss function matches the model's directly computed loss to about 1e-9.

| Prediction | Verdict | Key numbers |
|---|---|---|
| **H1** dim(N) ≥ 5 | **held** | dim(N) = **36** of 72 (relative singular value < 1e-2); the random expectation is 0.500 |
| **H2** ≥ 50% of N is REDISTRIBUTION within the correlated group | **held, exactly at the threshold** | 18/36 REDISTRIBUTION (all 18 in the group), 2 WITHIN-EDGE, 16 OTHER |
| **H3** set A: median ≥ 0.5 and ≥ 3× random | **inconclusive (untestable, D5)** | 3 × 0.5 = 1.5 > 1. The median fraction in N is 0.942 |
| **H4** set B, capped: same criterion | **inconclusive (untestable, D5)** | median 0.949 |
| **H5** median overlap of the 10 flattest directions with N ≥ 0.5 | **held** (see caveat 2) | median pooled overlap **0.885** over 90 eigenvectors |
| **H6** set B: fraction in N higher when capped for ≥ 15/20 | **held** | **20/20**; median over unit pairs 0.713 → 0.949 |

**Interpretation, applying the pre-registered rules.** H3 and H4 are untestable, so neither the
"H2–H5 hold" rule nor the "WITHIN-EDGE" rule applies. By the catch-all rule this combination is
reported as found, without choosing an explanation. N is also not mostly WITHIN-EDGE (2/36).
**H6 holds, so its pre-registered reading applies:** fitting improves along constrained directions
while the differences persist along near-null ones, as the approximate-symmetry mechanism predicts.

**Caveats (facts, not verdicts):**
1. **N is half the space.** With dim(N) = 36, a random difference already puts 50% of its energy
   in N. The observed 0.94–0.95 is an enrichment of about 1.9×, not the ≥ 3× the brief
   anticipated. That design limit is why H3 and H4 are untestable.
2. **H5's metric is normalised within the layer-1 block, and most flat directions barely use that
   block.**
   - The 10 algebraically smallest eigenvalues are all **negative**: 23–39 negative eigenvalues per
     model, down to −1.5e-4 (F-b). They are negative-curvature directions of non-stationary points,
     not the flattest directions.
   - Their median layer-1 block share of ‖v‖² is only **0.08**. Post hoc: for the 27/90 with a block
     share ≥ 0.5, the overlap is still 0.925.
   - Post hoc: the eigenvalues **closest to zero** (median |λ| 7.5e-16) have a layer-1 block share of
     about 0. They are an exact null space **outside** the continuous layer-1 block, most likely in the
     salt-embedding edges, which see only 4 distinct inputs. This was not investigated further.
3. **The loss ratio is not 5–7× (D4).** The control-3 state's own training loss divided by the capped
   loss is median **9.6×**, range **7.2–13.9×**, for all 20 students.
4. **The unit mismatch grew as the fit improved.** From the control-3 state to the capped state, the
   median unit-match |corr| to the teacher fell from **0.659 to 0.594**, while the loss fell about 10×.
   The fraction of the difference in N rose for every student.
5. **Constant shifts are not the explanation.** The per-unit bias represents constants exactly.
   Within a single edge, a constant is only approximately representable: the relative residual of
   fitting a constant is 6e-6 – 1.5e-2 across edges. Centring M changes dim(N) only from 36 to 34.

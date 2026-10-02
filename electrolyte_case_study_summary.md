# Electrolyte case study: identifiability of KAN-delta, close-out

**Setting.** KAN-delta predicts Li-electrolyte conductivity for the CALiSol-23 DEC holdout. Its
prediction is log k = a measured pure-PC anchor + x_co · h(solvent descriptors, 1/T, molality, salt).
Its first layer is a [8 continuous inputs + 2 salt-embedding, 6, 1] FastKAN, with SiLU + 8 Gaussian
RBFs per edge and a smoothness penalty (`kan.py`, `final.py`).

This document summarises every pre-registered test run on it. Each prediction was committed before
its code ran; the pre-registration and results commits are listed.

## All pre-registered predictions

Commits: **prereg** = pre-registration committed before the run; **results** = verdicts committed
after it. Prediction names are the committed ones: control 2 has C1–C3, and control 3 has C1', C2'
and C4 (there is no C4 in control 2, and no C3' or C4').

| Study | Prediction | Verdict | Key number | Prereg → results |
|---|---|---|---|---|
| Gauge study (`gauge_prereg.md`) | **Control 1**: teacher edges recovered, median dense corr ≥ 0.8 | **failed** | 0.301 (sanity: h corr 0.913 < 0.95) | c11dd92 → e68e2eb |
| | P1: gauge fixing removes ≥ 50% of edge variance | inconclusive (gated by Control 1) | not run | c11dd92 → e68e2eb |
| | P2: sparse ≥ 2× dense, DEC region highest | inconclusive (gated) | not run | c11dd92 → e68e2eb |
| | P3: (e) < (d) < (c) < (a) | inconclusive (gated) | not run | c11dd92 → e68e2eb |
| | P4: every seed has a unit with \|corr\| < 0.8 | inconclusive (gated) | not run | c11dd92 → e68e2eb |
| | P5: KAN-full edge variance > KAN-delta | inconclusive (gated) | not run | c11dd92 → e68e2eb |
| | P6: seed spread flags errors | **failed** | Spearman +0.169 (A) / +0.213 (B); fold B RMSE at 80% coverage 0.153 vs random 0.137 | c11dd92 → 67fe9f2 |
| Control 2 (`control2_prereg.md`) | C1: (I,C) edges ≥ 0.9 and units ≥ 0.9 | inconclusive (gate failed) | prediction corr 0.947 < 0.99 | 45b336f → a260aef |
| | C2: (R,C) invT, molal ≥ 0.9; co-solvent < 0.7 | inconclusive (gate failed) | prediction corr 0.969 < 0.99 | 45b336f → a260aef |
| | C3: 107 epochs gives prediction corr < 0.95 | **held** | 0.858 (I,S), 0.919 (R,S) | 45b336f → a260aef |
| Control 3 (`control3_prereg.md`) | C1': (I) passes the gate, edges ≥ 0.9, units ≥ 0.9 | **failed** | (I) RMSE/std 0.128 > 0.05; unit match 0.648 | 812a0fa → 9f59d39 |
| | C2': (R) edges ≤ (I) edges − 0.10 | inconclusive ((I) failed the gate) | (R) edges 0.852, units 0.659 | 812a0fa → 9f59d39 |
| | C4: every continuous input ≥ 0.9 in (I) | inconclusive ((I) failed the gate) | – | 812a0fa → 9f59d39 |
| Fiber (`fiber_prereg.md`) | F1: ≥ 50% of seed pairs on-fiber | **failed** | 0/190 (closest d = 0.146) | e83df2e → f220a2f |
| | F2–F5 | inconclusive (no on-fiber pairs) | not run | e83df2e → f220a2f |
| Fiber 2 (`fiber2_prereg.md`) | G1: ≥ 4/6 seeds and ≥ 10/20 students certify convergence | **failed** | 0/6, 0/20 (all at the 20,000-iteration cap) | 4af5250 → 98eca9d |
| | G2–G5 | inconclusive (no converged model) | not run | 4af5250 → 98eca9d |
| Support (`support_prereg.md`) | H1: dim(N) ≥ 5 | **held** | dim(N) = 36 of 72 | b473232 → c5505e5 |
| | H2: ≥ 50% of N is REDISTRIBUTION within the correlated group | **held, exactly at threshold** | 18/36 | b473232 → c5505e5 |
| | H3: set A fraction in N ≥ 0.5 and ≥ 3× random | inconclusive (untestable: 3 × 0.5 > 1) | median 0.942 (random 0.5) | b473232 → c5505e5 |
| | H4: set B (capped), same criterion | inconclusive (untestable) | median 0.949 | b473232 → c5505e5 |
| | H5: flattest Hessian directions overlap N ≥ 0.5 | **held**, with caveat (see (b)) | median 0.885; median layer-1 block share only 0.08 | b473232 → c5505e5 |
| | H6: fraction in N rises from control-3 to capped state, ≥ 15/20 | **held** | 20/20 (0.713 → 0.949) | b473232 → c5505e5 |

**Tally (29 predictions):** 5 held (C3, H1, H2, H5, H6, with H2 at its threshold), 5 failed
(Control 1, P6, C1', F1, G1) and 19 inconclusive. Most inconclusive results come from
gates that failed, or from a criterion that could not be met by construction (H3, H4).

## Loss level of control 3's matches

**Control 3's unit and edge matches (units 0.659, edges 0.852) were measured at a training loss
7.2–13.9× (median 9.6×) above the capped students' loss** (`support_output.txt`). The brief for this
close-out stated 5–7×. That figure was an earlier rough estimate; the measured ratio replaces it.
From that state to the capped state, the median unit match to the teacher fell further, from 0.659
to 0.594.

## (a) What is identified in this model

- **The prediction function, on the support of the data.**
  - Anchor + KAN-delta predicts held-out solvent systems with log10 RMSE 0.142 (PC+EA) and 0.140
    (EC+PC). These are cross-validation results from `kan.py`, not pre-registered and not
    leaderboard scores.
  - Independent seeds agree with each other (about 0.037 in delta) more closely than with the data
    (residual about 0.097).
  - A trained teacher's function is recovered from the real inputs to correlation 0.9988.
- **The near-null space N of the first-layer design,** which is fixed by the data and basis alone.
  - N has 36 of 72 dimensions (relative singular value < 1e-2).
  - Half of it (18/36) redistributes coefficient energy among correlated co-solvent and mixture
    descriptors: the correlation between eps_co and lneta_co is 0.966, and between mix_eps and
    mix_lneta 0.907. The co-solvent descriptors also take only 30 distinct values.
- **Edges for inputs that are weakly correlated with the rest tend to be recovered better,** though
  still not reliably. In control 3 (R):
  - 1000/T scored 0.962, molality 0.900 and M_co 0.927. Their largest correlations with any other
    input are 0.13, 0.24 and 0.28.
  - x_co, eps_co, lneta_co and mix_eps scored 0.578–0.767.
  - mix_lneta (0.908) is the exception: it is strongly correlated (0.907 with mix_eps) yet was
    recovered well.

## (b) What is not identified, and why

- **Individual hidden units and first-layer edges are not identified.**
  - Seed-to-seed unit matching has a median |corr| of 0.49.
  - Student-to-teacher unit matching falls from 0.66 to 0.59 as the loss falls about 10×.
  - The gauge-fixing procedure was never validated: Control 1 and C1' failed, and control 2's gates
    failed.
  - The seeds are not one function in different gauges (F1: 0/190 on-fiber).
  - No model reached certified convergence (G1: 0/26). The independent seeds are 2–3 orders of
    magnitude from the gradient criterion, and still descending along flat directions after 20,000
    float64 L-BFGS iterations.
- **Why: the model differences lie in N.** 94–95% of the gauge-removed first-layer difference lies
  in N, against 50% expected at random. That share rose for every student (20/20) as the fit
  improved.
  - By the pre-registered reading of H6, fitting improves along constrained directions while
    differences persist along near-null ones. That is consistent with approximate symmetries created
    by the input design, not by the architecture alone (only 2/36 N-vectors are WITHIN-EDGE).
- **Limits on this explanation:**
  - The enrichment is about 1.9×, and the pre-registered ≥ 3× test was untestable because N is so
    large.
  - H2 held exactly at its threshold.
  - H5's support is weaker than it looks. The directions analysed are negative-curvature directions
    of non-stationary points, and their median first-layer block share is 0.08. Post hoc: for the 27/90
    with a block share ≥ 0.5, the overlap is 0.925.
  - The truly flat directions (|λ| about 1e-15) lie outside the continuous first-layer block,
    probably in the salt-embedding edges. They were not investigated.
  - Every comparison is between non-converged points on optimisation trajectories.

## (c) Which claims can go into the paper

**Supported:**
1. The anchor + KAN-delta approach predicts held-out solvent systems well. Report the cross-validation
   numbers with their caveats.
2. **KAN edge shapes and hidden units in this setting are not identifiable and must not be read as
   physics.** Teacher-student controls with a known, reachable teacher failed to recover them, and
   the seeds realise different functions at nearly equal loss.
3. **Seed-to-seed parameter differences lie mostly in a near-null space that the data design
   determines** (correlated, few-valued solvent descriptors). The share rises as the fit tightens.
   Report this as descriptive evidence consistent with approximate symmetries, with the caveats in (b).
4. Seed spread is not a usable error or abstention signal here (P6).

**Not supported:**
- that gauge fixing recovers edges;
- that the additive affine gauge accounts for all of the seed-to-seed variation;
- that approximate symmetries *cause* the slow convergence (H3/H4 untestable; no model converged);
- any per-edge physical interpretation;
- that H5 describes the flattest directions of the loss.

**Reproducibility.** Code and logs are in this repository. Weights are archived outside git
(`gauge_weights_2026-09-29.tar.gz` and `…29b.tar.gz`) and listed with SHA-256 checksums and producing
commits in `weights_manifest.txt`.

## Interventions

Two interventions tested whether removing layer-1 coefficient directions that the data cannot see
fixes the slow convergence and seed disagreement.

### 1. Basis-size rule (`rule_prereg.md`): gate stopped, a no-op

The rule gives each co-solvent-determined input (eps_co, lneta_co, M_co) K RBF centres, with
3 × (K + 1) ≤ n_co − 1. There are n_co = 30 distinct co-solvent combinations, so K = 8, which is
exactly final.py's basis. **0 columns were removed and the gate stopped;** I1–I5 are inconclusive
(prereg 6c93bf4 → results a53cc2a). The outcome was anticipated from committed files before any code
ran.

The redundancy it targeted is real (the co-solvent block has numerical rank 14 of 27), but counting
distinct values cannot detect it. The rank deficit comes from strong correlation among inputs that
share the same 30 points.

### 2. Rank constraint (`rank_prereg.md`)

Each unit's layer-1 coefficients on the affected columns are restricted to the retained right
singular vectors of the design matrix (θ = Pz, relative singular value ≥ 1e-2):
- **Arm G:** the co-solvent block, 14 of 27 directions kept.
- **Arm F:** all of M, 36 of 72 kept.

Training and float64 L-BFGS polishing were done exactly as for the baseline (the fiber2 capped
independent seeds). Prereg df44ead → results aeb6f87; weights in `gauge_weights_2026-09-29c.tar.gz`
(manifest 148fb3c).

| Prediction | Arm G | Arm F |
|---|---|---|
| I1: median max\|grad\| ≤ 0.1 × baseline | **failed** (4.00×) | **failed** (29.1×) |
| I2: median d_AB ≤ 0.5 × baseline | **failed** (0.945×) | **failed** (0.808×) |
| I3 (sanity): median loss ≤ 1.10 × baseline | passed (1.004×) | passed, narrowly (1.099×) |
| I4: fold A RMSE ≤ 0.150 (KAN-delta 0.1420) | **held**, narrowly (0.1499) | **failed** (0.1587) |
| I5: edge corr on 1000/T and molality ≥ baseline + 0.1 (baseline 0.102) | **held** (0.927) | **held** (0.338) |
| I6: arm F's improvement ≥ arm G's in both I1 and I2 | **failed** (max\|grad\| 29.1× vs 4.00×; d_AB does favour F, 0.808 vs 0.945) | |

No constrained model certified convergence (0/12). The pre-registered float32 equivalence check
exceeded its 1e-5 tolerance (up to 2.7e-4). In float64 the conversion is exact, so this is rounding,
not a modelling error.

**What is and is not established.**

*Established:*
- **Removing the near-null directions of the layer-1 design does not cure the optimisation problem.**
  In both arms the gradients at the iteration cap got *larger* (4× and 29×) and seed disagreement
  fell only slightly (−5% and −19%), with no model converging.
- By the pre-registered interpretation, that places the cause of the slow convergence and the seed
  disagreement **outside the data-invisible layer-1 directions**. Candidates are layer 2, the
  saddle-rich landscape (every model has 21–39 negative Hessian eigenvalues), and the conditioning,
  which the constraint made worse (largest eigenvalue up to 2e6 in arm F).
- This **weakens the causal reading of the support study.** There, the seed differences sat mostly
  in N and moved further into N as the fit improved (H6). This intervention shows that those
  differences are a symptom rather than the driver: taking N away does not make seeds agree on the
  function or converge.
- **One effect that does hold:** constraining the co-solvent block made the *other* edges (1000/T,
  molality) far more consistent across seeds (0.102 → 0.927). So identifiability of edges outside
  the correlated block can be improved without changing the fit (loss +0.4%). Accuracy moves to
  fold A 0.150, at the pre-registered limit.

*Not established:*
- that any basis or rank rule fixes convergence;
- that the improved edge consistency reflects the *true* edge shapes (there is no teacher here);
- whether a rescaled or better-conditioned version of the constraint would behave differently
  (it was not tested).

## Optimisation checks: trust-region Newton and stage 2 (`trust_prereg.md`, `stage2_prereg.md`)

| Prediction | Verdict | Key number |
|---|---|---|
| T1 / T2: trust-krylov certifies ≥ 4/6 seeds / ≥ 2/3 students | **failed** | 0/6, 0/3. All 9 stopped on scipy's "bad approximation" after 1–62 iterations; the Hessian was accurate to ≤ 2e-7 |
| T3–T5 | inconclusive | nothing converged |
| U1: trust-exact (exact Hessian, 500 iterations) certifies ≥ 4/6 seeds | **failed** | 0/6. All 9 hit the iteration cap |
| U2: with L2 λ = 1e-5, ≥ 4/6 seeds and ≥ 2/3 students | **failed** | 1/6 seeds, 0/3 students |
| U3: more converge at 1e-5 than at 1e-6 | **held** | 1 vs 0 |
| U4: converged L2 models' loss without L2 ≤ 1.05 × capped | **failed** | 1.2545 |
| U5: converged seeds' disagreement ≤ 0.5 × capped | inconclusive | only 1 converged |

Commits: trust a77e8b4 → 67586d6. Stage 2: c6a9e83 → fixes 78bcaba, 60621c6 → results
(the commit that adds this section).

**By the pre-registered rules, the landscape itself is the cause.** Even an exact second-order
optimiser on a coercive loss could not certify minima. The L2 term at λ = 1e-5 was not mild: at the
capped states it was 4–34 × the data loss, so it changed the solution materially (U4).

**Close-out note: the capped states are not minima and are not near-optimal.** Without any L2 term,
trust-exact lowered the loss of every fiber2 capped state within 500 iterations. Final / capped
loss ratios for the six seeds:

| ind 0 | ind 1 | ind 2 | ind 3 | ind 4 | ind 5 |
|---|---|---|---|---|---|
| 0.9580 | 0.9787 | 0.9849 | 0.9463 | 0.9720 | 0.9798 |

- For the three students: 0.8120, 0.8516 and 0.7135.
- Along every trajectory the parameter norm kept growing (‖θ‖² rose 1.1–2.7 × in Part A).
- The growth was mostly in the SiLU base weights and the RBF coefficients. The smoothness penalty
  does not touch the SiLU base weights, and it doesn't touch constant or linear RBF patterns.
- **All comparisons between "capped" models in this case study** (fiber2, support, rank, trust) are
  therefore between points on unfinished, norm-growing trajectories, not between minima.

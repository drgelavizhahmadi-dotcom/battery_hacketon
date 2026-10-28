# Runaway edges: affine null spaces of smoothness penalties destabilise Kolmogorov–Arnold networks on correlated inputs

Oct 4, 2026 · @gelavizh ahmadi · Draft for TMLR

## Abstract

Kolmogorov–Arnold networks (KANs) are promoted as interpretable because each edge carries a learned univariate function. We show that a widely used family of smoothness penalties leaves a blind spot that undermines this promise: second-difference and curvature penalties do not charge for affine components of an edge, and SiLU base weights are often unpenalised. When the inputs to a trained nonlinear layer are correlated, affine components on different edges can cancel, so training inflates the parameters along directions that neither the penalty nor the data can see, and independent runs do not agree. In pre-registered experiments with a known teacher, correlated inputs gave 0 of 16 converged models, including under a recently proposed B-spline curvature penalty, while a small penalty on the affine null space gave 16 of 16 certified minima and essentially exact recovery of the teacher. Along certified regularisation paths spanning five decades, the loss behaves as if its infimum lies at infinity, on synthetic data and on real electrolyte-conductivity data; closing the null space bounds it. Non-convergence rises with input correlation, and the runaway and its fix carry over from Gaussian-RBF to B-spline edges. On the real data, polishing beyond early stopping degraded predictions for one held-out co-solvent, ethyl acetate (RMSE 0.142 → 0.81 without a null-space penalty and 1.03 with a weak one; 0.33 with a stronger one). Certified minima can still be degenerate or distinct, so convergence and identifiability are separate problems. We give practical recommendations for penalised KANs.

## 1 Introduction

KANs replace a network's fixed activations with learned univariate functions on every edge. Their appeal for science rests on reading those functions: an edge that looks like an exponential decay suggests a physical law. That reading presumes the edges are determined by the data. If different training runs produce different edges with equally good predictions, the edges describe the optimiser, not the system.

This paper identifies a concrete, common mechanism that breaks that presumption. It arises from the interaction of three ingredients that are each standard in practice:

1. **A smoothness penalty with an affine null space.** Second-difference (P-spline) penalties and curvature penalties ignore straight-line components of an edge; SiLU base terms are frequently unpenalised.
2. **Correlated inputs to a layer.** Physical descriptors often move together. Hidden units can also become correlated during training, exposing the next layer.
3. **Training toward convergence.** Early stopping hides the problem; longer optimisation exposes it.

Together they create directions along which edges grow into large, mutually cancelling affine shapes. The penalty does not see them, and on the training data their contributions cancel. The loss keeps decreasing slightly along these directions, and over the range we could test it behaves as if it had no finite minimum. Seeds wander apart, no model certifies as converged, and on inputs outside the training data's support the cancellation can fail.

**Contributions.**

- We describe the mechanism: a penalty's affine blind spot plus correlated inputs to a trained nonlinear layer. A layer whose output is linear in its parameters cannot run away (Sections 3 and 6).
- In a pre-registered synthetic study, adding a SiLU-weight penalty in the style of a recent curvature penalty does not prevent the runaway on correlated inputs, nor does that penalty's exact B-spline form (Section 8); a small penalty on the affine null space prevents it and, in this well-specified setting, restores recovery of a known solution (Section 4).
- On real electrolyte-conductivity data, we trace a complete failure to converge to this mechanism and document its effect on a held-out co-solvent (Section 5).
- Along certified regularisation paths, the loss behaves as if its infimum lies at infinity over five decades of regularisation, on synthetic and real data; closing the null space bounds it (Section 6).
- Non-convergence rises with input correlation, and the runaway and its fix carry over from Gaussian-RBF to B-spline edges (Sections 7 and 8).
- Certified minima can be degenerate, distinct or not minima at all; we separate three kinds of flatness and give recommendations (Sections 9 and 10).

All confirmatory studies were pre-registered: predictions, thresholds and interpretation rules were committed before the code that tested them. Of 65 predictions in the real-data case study, 11 held, 22 failed and 32 were inconclusive; we report them as recorded. The revision studies of Sections 6–8 and the applicability analysis followed the same protocol; Appendix A lists their verdicts, and Appendix D every change made after an outcome was known.

**Terms.** We keep four phenomena apart: exact symmetries of the parameterisation, along which the loss is exactly constant and which carry no information; near-flat directions from poor conditioning; failures of the optimiser to finish, with solver errors reported separately from iteration caps; and distinct minima, which make the fitted function itself depend on the seed. "Certified" means a numerically certified local minimum under stated tolerances, not a global or unique one; Appendix G gives the tolerances and defines every metric used.

## 2 Background and related work

**KAN edges.** A KAN layer maps inputs to outputs by summing learned univariate functions, one per edge. In the original formulation (Liu et al., 2025) each edge is a SiLU base term plus a B-spline expansion. FastKAN (Li, 2024) approximates the B-splines with Gaussian radial basis functions on fixed centres, which is the edge type used in our experiments:

```latex
\varphi(x) = \alpha\,\mathrm{SiLU}(x) + \sum_{k=1}^{K} c_k \exp\!\left(-\left(\frac{x-\mu_k}{h}\right)^{2}\right)
```

**Smoothness penalties and their null spaces.** P-spline penalties charge squared second differences of the coefficients (Eilers and Marx, 1996). Constant and linear coefficient patterns cost nothing: the penalty's null space is affine. In generalised additive models this is well understood, and Marra and Wood (2011) proposed an additional penalty on the null space (the "double penalty") for variable selection; Wood (2017) treats null spaces and concurvity in additive models in depth. Our fix is the same construction; what is new is its consequence inside a KAN, where edges are summed within units and passed through further layers.

**Curvature penalties for KANs.** Bagrow (2026) proposed per-edge curvature penalties for compositional smoothness, combining the second-difference penalty with a term K\_silu·α² on the SiLU weight, and noted that the penalty has affine functions as its null space by design; for Gaussian-RBF edges he instead uses the integrated bending energy, whose curvature Gram matrix charges near-affine coefficient patterns slightly. That work evaluated on independent, uniformly sampled inputs. Second-derivative penalties on spline coefficients also appear in other KAN variants, typically on the spline part only, leaving the SiLU base term free (Cang et al., 2024; see also the free-knots KAN of 2025). We show that the affine null space becomes a runaway direction when inputs are correlated, and that penalising the SiLU weight alone does not close it.

**Concurvity.** In additive models, including neural additive models (Agarwal et al., 2021), correlated inputs let one input's shape function absorb another's, making shape functions non-identifiable; this is known as concurvity. Siems et al. (2023) proposed a regulariser penalising correlations between transformed features. The cross-edge cancellation we describe is concurvity inside a KAN unit. In an additive model with a penalised null space it shifts shape between functions; combined with an unpenalised affine null space in a KAN, it becomes unbounded growth.

**Losses without a finite minimum.** When a loss attains its infimum only at infinity, gradient descent drives the parameter norm to diverge, slowly, and optimisation can continue while held-out performance worsens (Soudry et al., 2018). Our runaway behaves like an instance of this, created by a penalty's null space. It differs from the benign flat directions of linear least squares, where gradient descent converges to the minimum-norm solution: here the direction is slightly tilted, so gradient descent keeps following it. On such ill-conditioned problems, early stopping acts much like ridge regularisation (Yao et al., 2007; Ali et al., 2019).

**Parameter-space symmetries.** Exact symmetries (unit permutations, sign flips) make parameters non-identifiable in tanh and RBF networks, and completeness results characterise them (Sussmann, 1992; Kůrková and Neruda, 1994). KANs add continuous symmetries such as affine reparametrisations of hidden units. The runaway is not an exact symmetry: it is an approximate one, a nearly flat but tilted direction, which is what prevents convergence.

## 3 Mechanism

Consider a unit that sums edges from two inputs, s = φ₁(x₁) + φ₂(x₂), each edge as in Section 2, trained with a second-difference penalty on its RBF coefficients.

**Step 1: a direction the penalty cannot see.** Add a linear pattern t·ℓ to the RBF coefficients of edge 1 and subtract it from edge 2, with ℓ\_k proportional to the centre index. Linear coefficient patterns have zero second difference, so the penalty is unchanged for every t. With Gaussian bumps on a regular grid, the pattern produces a nearly affine function: over the central 80% of the input range it deviates from a straight line by at most 2% of its range, and by up to 13% near the grid edges. Edge 1 thus gains approximately t·a·x₁ and edge 2 loses t·a·x₂.

**Step 2: a direction the data cannot see.** The unit's pre-activation changes by approximately t·a·(x₁ − x₂). If x₁ and x₂ are strongly correlated, x₁ − x₂ is small wherever the data lie, so predictions change little even for large t. The direction is nearly free in both the penalty and the data. This is concurvity (Section 2) inside a KAN unit. In an additive model, the same direction between collinear linear terms is flat but not tilted, so the fit stays bounded; what makes the KAN case worse is the trained nonlinear layer, which tilts it (Steps 3 and 4).

**Step 3: a receding minimum.** "Nearly free" is not "free". Moving along the direction lets the model shed curvature into the free affine and SiLU components, so the loss keeps decreasing slightly. Both terms contribute (post-hoc decomposition): in the synthetic runaway models the smoothness penalty fell to about a quarter of its early value and supplied a median 61% of the decrease while training error fell only 5%; on real data, from early stopping to the capped state, the penalty fell 8–88× and supplied 17–36% of the decrease, the data term the rest. The growth went into whatever the penalty left free: the linear part of the RBF coefficients grew 34–40× in the synthetic runaway cells, and the SiLU weights 85× where they were unpenalised. In 12 of 20 synthetic runaway models, training error fell while test error rose. Over the range we could test, the loss behaves as if its infimum lies at infinity: along a certified L2 regularisation path, the minimisers grow without levelling off over five decades of the regularisation weight while the loss keeps falling, and they recede along a curved valley rather than a straight ray (Section 6). Growth alone is not the signature, because a distant finite minimum also produces large growth (Step 4); what distinguishes the runaway is that the optimisation never reaches a certified minimum. Different seeds drift along different such directions.

**Step 4: the runaway needs a trained nonlinear layer.** If the parameters being trained enter the output linearly, as in the output layer when the hidden activations are held fixed, or in a one-layer KAN, the penalised loss is a convex quadratic. It may have flat directions, but no tilted ones, so it cannot run away: with the first layer frozen, every model reached a certified finite minimum, and so did a one-layer model on correlated inputs, although the minimum could lie far away (growth up to ×583 on real data). The runaway therefore needs trained parameters that act through a nonlinearity. Correlation can reach such a layer from correlated inputs, or from hidden units that become correlated during training, so the output layer can host the runaway even when the raw inputs are independent, as long as the first layer also adapts. Correlated hidden units are necessary but not sufficient: they also occur in models that converge once the null space is penalised. Whether a given nonlinear problem degenerates depends on the data: with the second layer frozen, the synthetic first-layer problems settled cleanly, while the real first-layer problem, with few-valued, strongly correlated descriptors, reached only highly degenerate minima, one of which was not a minimum at all.

**Consequences.**

- **Non-identifiability:** edges differ across seeds by large affine components that leave predictions almost unchanged; even certified minima can be degenerate or distinct from one another.
- **Non-convergence:** second-order optimisers stall and certification fails. Since growth is also large on the way to a distant finite minimum, and norms depend on how the model is parameterised, certification and function-space checks, not growth, are the criteria.
- **Extrapolation risk:** on inputs outside the training data's support, the inflated components need not cancel. In our real-data case, polishing damaged predictions for a held-out co-solvent lying outside the training descriptor hull.

**The fix.** Penalise the null space directly. For each edge, let P\_null project its RBF coefficients onto constant-plus-linear patterns in the centre index, and add

```latex
Q = \sum_{e} \lVert P_{\mathrm{null}}\, c_e \rVert^{2}
```

with a small weight μ, in addition to a penalty on the SiLU weight. This makes every affine component cost something, so the regularised minimisers stay bounded. Penalising only part of the free directions is not enough: the growth moves into whatever stays unpenalised, such as biases or embeddings, so these need a small penalty too.

## 4 Controlled study

**Design.** A teacher KAN \[4, 3, 1\] with smooth, hand-specified edges (fitted by smoothness-penalised least squares) generates targets with Gaussian noise of 5% of the output spread; n = 2000 training rows. Inputs are N(0, 1)/3, clipped to the grid, either independent or correlated in pairs (ρ = 0.95 for x₁/x₂ and x₃/x₄). Students share the teacher's architecture and are trained under three penalty levels:

- **P0:** second-difference penalty on RBF coefficients (λ = 10⁻³).
- **P1:** P0 plus K\_silu·α² on each SiLU weight (K\_silu = 0.443; Bagrow's form for B-splines, but only a surrogate for his Gaussian-RBF penalty, which uses a curvature Gram matrix; Section 8 tests his B-spline form exactly).
- **P2:** P1 plus μQ, the affine null-space penalty.

Each of the 6 cells has 8 seeds. Each student is trained with AdamW and early stopping, then polished with an exact-Hessian trust-region method and certified by a second-order criterion (gradient, curvature and Newton decrement). μ was calibrated before the experiment by a pre-registered rule: 10⁻⁶ failed the pilot (5.0× growth) and 10⁻⁵ passed (1.32×) and was frozen.

**The teacher itself runs away.** Polished under P0 and P1 on correlated inputs, the teacher's parameters grew 85× and 1069× without certifying, while its predictions stayed within half a noise standard deviation of the original. The runaway changes the parameters, not the function.

| Cell | Norm growth | Certified | Unit match to reference |
| --- | --- | --- | --- |
| Independent, P0 | 12.5× | 6/8 | 1.000 |
| Independent, P1 | 7.9× | 6/8 | 1.000 |
| Independent, P2 | 1.09× | 8/8 | 1.000 |
| Correlated, P0 | 65.7× | 0/8 | 0.774 |
| Correlated, P1 | 32.1× | 0/8 | 0.883 |
| Correlated, P2 | 1.00× | 8/8 | 1.000 |

Medians over 8 seeds; growth is the squared parameter norm, polished over early-stopped.

**Results.**

- **Penalising the SiLU weight does not prevent the runaway.** P1 halved the growth on correlated inputs but left 0 of 8 models certified.
- **Closing the null space prevents it everywhere.** P2 gave 16 of 16 certified models with no growth, in both input conditions.
- **Without the runaway, the internals are recoverable.** In this well-specified setting, where teacher and student share architecture, basis and grid, every cell without a runaway matched the reference's units and edges exactly (1.000), with seed-to-seed differences around 10⁻⁴.
- **Independent inputs are not fully safe.** Under P0 and P1, most independent seeds grew substantially (8–12×) but settled at certified minima; 4 of 16 (seeds 0 and 4 in each cell) never certified.

**Where the runaway lives.** A post-hoc anatomy shows that in the correlated cells almost all growth sits in the output layer (layer-1 share 0.110 under P0, 0.024 under P1), in the affine part of the output edges' coefficients and, under P0, their SiLU weights, with no hidden unit saturating: Step 4 of Section 3. Among independent-input models, those that ran away were exactly those whose hidden units had become correlated (largest pairwise correlation 0.70, against 0.05 for certified models), while certified P2 models kept correlations up to 0.88 without running away (post hoc).

**Verdicts.** Of six pre-registered predictions, one held (the certification pattern, X2) and five failed. Independent cells exceeded the growth threshold, mostly through finite growth. Cancellation and extrapolation effects were weaker than predicted. Recoverability failed only because the runaway cells were more recoverable than predicted, not less. In this synthetic setting, polishing did not degrade extrapolation: most of the off-pattern error in correlated cells was present at early stopping.

## 5 Real-data case study: electrolyte conductivity

**Task and model.** The CALiSol-23 challenge (de Blasio et al., 2024) asks for the log conductivity of lithium-ion electrolytes; every test sample contains diethyl carbonate (DEC), which never appears in training. KAN-delta anchors each prediction on the measured conductivity of a pure propylene-carbonate (PC) electrolyte with the same salt, from the same article (Appendix H), and learns only the correction, as in Δ-machine learning (Ramakrishnan et al., 2015), log k = log k\_PC + x\_co·h(z), where h is a FastKAN-style network of shape \[10, 6, 1\]. Its inputs are physical descriptors of the solvents (dielectric constant, log viscosity, molar mass), temperature, molality and a salt embedding. A second-difference penalty acts on the RBF coefficients, and the SiLU weights are unpenalised. Early-stopped after 107 AdamW epochs, it reached RMSE 0.142 on a held-out co-solvent (fold A: ethyl acetate held out). Its AdamW weight decay played no part: retraining without it left the early-stopped norms essentially unchanged and fold A's RMSE at 0.1420 (a two-sided pre-registered prediction whose verdict was the branch "incidental"). We use these data as an optimisation stress test with physical guidance, not as a transport model: the anchor and descriptors add structure, but the learned edges are not physical laws, and the salt embedding only interpolates among the salts seen in training.

**The inputs are correlated and few-valued.** The three co-solvent descriptors take only 30 distinct combinations in training, and dielectric constant and log viscosity correlate at r = 0.966.

**No tested run met the certification criterion.**

- Of 190 pairs of polished independent seeds, none computed the same function.
- After 20,000 float64 L-BFGS iterations (the "capped" state used below), 0 of 26 models converged; exact-Hessian trust-region methods also failed on every seed.
- Students trained on a frozen teacher matched its predictions (correlation 0.9988) but not its hidden units (0.66).

**The anatomy matches Section 3.**

- From early stopping to the capped state, the squared parameter norm grew 150–1550× while the smoothness penalty fell 8–88×.
- At least 99.9% of the coefficient change during exact-Newton polishing lay in the penalty's affine null space, in 9 of 9 models.
- Cancellation within an edge (SiLU against RBF) stayed small, 1.13–1.33. Cancellation across the edges entering a unit rose from 1.7–2.9 to 4–10, consistent with the correlated descriptors.
- The runaway spread across both layers: 30–40% of the parameter change went into layer-1 affine RBF coefficients, 37–38% into SiLU weights and 19–31% into layer-2 affine coefficients (post-hoc), and 12–39% of hidden units saturated. In the synthetic runaway the output layer dominated and no unit saturated, so saturation is not needed for the mechanism.

**Extrapolation suffered.** Polishing beyond early stopping raised fold A's RMSE from 0.142 to 0.81 with no null-space penalty and to 1.03 under a weak one (μ = 10⁻⁶); a stronger one (μ = 10⁻⁴) held it at 0.33 (Section 9). Fold A was used repeatedly during development, so it is not an untouched test set: the 107-epoch stopping point was chosen using its own held-out rows (and fold B's), which favours the early-stopped model here, and its 635 rows are one salt from one article (Appendix H). Fold A's co-solvent lies outside the training data's descriptor hull, although on the correlation pattern of dielectric constant and viscosity. Within the fold, the damage scales with the co-solvent fraction: the learned correction enters as x\_co·h, so any change polishing makes to h is magnified where x\_co is large (descriptive). Decomposing predictions into edge contributions shows the cancellation directly: on fold A a larger share of the opposing output-edge contributions survives than on training rows, already at early stopping (net-to-total ratio 0.59 against 0.31) and after polishing without the null-space penalty (0.12 against 0.06; post hoc, Appendix B). By parameter block, the blind-direction part of the polishing change is not free on its own: alone it raises the ensembles' RMSE above 2.9 on both row sets, and the rest of the change cancels most of that on training rows but less on fold A, consistent with the curved valley of Section 6. Polishing also reverses the temperature dependence on 25–33% of fold A rows, mostly at low temperature and high x\_co (0.3% early, 5% at μ = 10⁻⁴; post hoc, Appendix F.8). Across the co-solvent folds with at least 20 test rows, no distance measure distinguishably predicted where errors were large, and a planned predictive follow-up was dropped because its pre-stated precondition was not met. The degradation was much stronger than in the synthetic study, where correlation itself caused most of the off-pattern error (Section 11).

## 6 Is the minimum at infinity?

Failing to converge does not by itself show that the loss has no finite minimum. The minimum could be finite but very far away, or the problem could simply be too ill-conditioned for the optimiser to finish. Two pre-registered experiments separate these possibilities: freezing one layer, and following a certified regularisation path.

**Freezing a layer.** With the first layer frozen at its early-stopped weights, only the output layer is trained. Its parameters enter the output linearly, so the penalised loss is a convex quadratic. Every model then reached a certified minimum: 6 of 6 real KAN-δ seeds and 8 of 8 in each correlated synthetic cell, with no negative curvature at any logged iterate; a one-layer KAN trained on correlated inputs also certified, 8 of 8. The pre-registered prediction that the norm would grow at most threefold failed: the real models grew a median ×583 to reach their minima. That growth is not a runaway, because it ends at a certified point. Freezing the second layer instead, in an exploratory arm, left the synthetic models certifying but the real first layer reaching only highly degenerate endpoints (Appendix F.1). Section 3 draws the conclusion: the runaway needs trained parameters acting through a nonlinearity, and whether such a problem degenerates depends on the data.

**A certified regularisation path.** For a loss L that is bounded below and a penalty R that grows without bound with the parameters, the global minimisers of L + γR stay bounded as γ → 0 if and only if L has a finite minimiser (Appendix F.6). We use R = ‖θ‖² over all parameters, follow γ from 10⁻² down to 10⁻⁷ in half-decade steps, warm-start each value from the previous solution, and certify every point; degenerate points are accepted only after probes for remaining descent. On synthetic data, with the blind spot open, the minimisers' squared norm grew as a power of 1/γ (exponent 0.63–0.76 under P0, 0.36 under P1) while the unregularised loss kept falling; with the null space closed (P2), the path levelled off (exponent 0.014, ratio 1.004 between the two smallest γ). On the six real models, the same divergence appeared on six distinct branches: exponents 0.74–0.88, squared norms reaching 596–3052, and the unregularised loss falling from 2.05·10⁻² to 8.7–9.3·10⁻³. All pre-registered predictions held, the real-data discrimination check being taken from the synthetic controls as pre-registered. \[Figure: regularisation paths, open vs closed, synthetic and real.\]

Two details matter for how far these replicate. On synthetic data the warm-started paths start from nearly one function; under P1 and P2 they stay together along the whole path, while under P0 they separate into a few branches. The real paths, by contrast, are genuinely distinct and compute different functions: at γ ≤ 10⁻⁴ their predictions differ by a median 13.7–16.3% of the output's standard deviation.

**Robustness (post hoc).** Cold starts found no small-norm basin, and at the smallest γ often a lower objective at a larger norm, so the warm path understates the divergence; the minimisers recede along a curved valley, in the penalty's blind directions (Appendix F.7).

**Closing the null space on real data.** In a control whose two possible outcomes were fixed in advance, the same path on the real loss with the null-space penalty at μ = 10⁻⁴ levelled off: squared norms of 27.7 and 27.4 against 596 and 2200 for the same seeds with the blind spot open (ratios 1.09 and 1.10). The plateau is not a full stop: the norm still creeps up (whole-path exponents 0.105 and 0.124), in the biases, the curved RBF coefficients and the salt embedding, which this penalty leaves weakly penalised or free. The price is a higher training loss: +6% and +23% for the two seeds (1.0926·10⁻² against 1.0300·10⁻², and 1.1224·10⁻² against 9.113·10⁻³). When the penalty is instead lowered towards zero, the growth floods whichever parameters it leaves free: with the null-space penalty of Section 3 the salt embedding grows 44–218-fold, while a variant that also penalises biases and embeddings keeps the embedding bounded.

**Reading.** Over the tested range, γ = 10⁻² to 10⁻⁷, the loss with the blind spot open behaves as if its infimum lies at infinity, on synthetic and real data alike, and closing the null space bounds it. Two limits apply: the criterion concerns global minimisers while the path follows one local branch, and a finite minimum lying beyond the tested range cannot be excluded.

## 7 How much correlation does the blind spot need?

Section 4 compared two doses, independent inputs and ρ = 0.95. A pre-registered dose-response study fills in between: ρ ∈ {0, 0.5, 0.8, 0.9, 0.95, 0.99} for both input pairs, P0 and P1 at every level, 8 seeds each, with paired random draws so that only the correlation changes, plus the null-space control P2 at ρ = 0.99. A replication gate ran first: the new ρ = 0.95 cells reproduced the committed synthetic study bit for bit.

**Correlation drives non-convergence.** The share of models that fail to certify rises with ρ (prediction held; P0 with one allowed inversion). It first exceeds one half at ρ = 0.5 under P0 and at ρ = 0.9 under P1, and from ρ = 0.95 no model certifies under either penalty. These thresholds belong to this synthetic setting (\[4, 3, 1\], Gaussian inputs); they are not thresholds for real data. With the null space closed, all 8 models certify even at ρ = 0.99, and a short regularisation path levels off (median D = 1.13, where D is the ratio of the squared norms at γ = 10⁻⁷ and 10⁻⁵ along a short regularisation path). Along with the dose, the hidden units become correlated (largest pairwise correlation from about 0.05 at ρ = 0 to 0.82–0.96 at ρ ≥ 0.9), and the polished training loss falls (Spearman −0.94): more correlation lets the runaway buy more loss. \[Figure: share uncertified and median D against ρ.\]

**What a short path cannot tell.** We also predicted that a short regularisation path (γ = 10⁻³, 10⁻⁵, 10⁻⁷) would diverge more strongly as ρ grows. It did not: the path statistic D was already high at every dose, including ρ = 0. A post-hoc check with a rule fixed in advance showed why: where the polish certified, mostly at low ρ, 30 of 32 paths were heading towards that certified finite minimum, and a two-decade path cannot separate a distant finite minimum from a receding one (derivation in Appendix F.4). Receding behaviour (Section 6) is therefore shown only at ρ = 0.95 and on the real data, over the tested range; at low ρ the minima are finite but distant; at intermediate ρ these tests do not settle the question. A secondary prediction, that the affine share of the parameter change would rise steadily with ρ, also failed.

## 8 Is it the basis?

The experiments so far use Gaussian-RBF edges. To test whether the runaway belongs to the basis or to the penalty's blind spot, we replicated the synthetic 2×3 study with cubic B-spline edges as in pykan and efficient-kan: uniform knots with G = 5 intervals on \[−1, 1\], extended by three knots at each end, giving 8 basis functions per edge (matching the 8 Gaussians), plus the SiLU term. On these knots, constant and linear coefficient patterns reproduce constant and linear functions exactly, so the second-difference penalty's blind spot is exactly the affine functions. The teacher was refitted into the B-spline basis; inputs, seeds and noise were those of the RBF study.

**The runaway carries over.** With correlated inputs, none of the 16 P0 and P1 models certified, where P1 here is Bagrow's B-spline penalty up to a scale factor our edges do not have (prediction held). The runaway is not specific to the Gaussian basis: it follows the penalty's affine null space.

**The fix carries over, with a lesson about calibration.** The null-space weight was set by the same committed rule as in the RBF study (the smallest μ whose pilot models grew at most threefold), which here chose μ = 10⁻⁶, the bottom of the grid, against 10⁻⁵ for RBF edges. At that weight, the closed models certified 8 of 8 in both conditions, but the pre-registered bound on the short path failed with correlated inputs (median D = 2.23 against 1.5). Post-hoc checks, with predictions fixed in advance, traced this to the measurement window: the null-space penalty's curvature, about 1.8μ, lies between the two γ values that D compares. Extended to γ = 10⁻⁹, the paths were bounded, ending near the certified minima's norm though not always at the same point; and at the RBF study's μ = 10⁻⁵ the fix met the original standard (8 of 8 certified, D = 1.17 and 1.31). The growth-based rule chose a weight too small for a criterion carried over from another setting: a lesson about calibration, not about the fix.

**Independent inputs.** Fewer independent models certified than with RBF edges (5 and 4 of 8 under P0 and P1, against 6; one P1 failure was a solver error rather than an iteration cap), so the pre-registered prediction failed, and its fixed reading applies: the B-spline basis is less benign on independent inputs. Post hoc, two of the seven failures look like runaways, one like slow convergence, and four are mixed; with 8 models per cell, the margin is one or two models.

**A symmetry the RBFs lack.** B-spline edges carry an exact 15-dimensional symmetry, constants moved between a unit's incoming edges and its bias, which holds only approximately for RBF edges; we excluded these analytically constructed directions from certification and from the degeneracy count (Appendix F.5). This was one of five changes to the study's calibration step made after outcomes were known; none touched a prediction, a threshold or an interpretation rule, and all are listed in Appendix D.

## 9 Closing the blind spot, and what remains

**The null-space weight on real data.** Adding μQ (with the SiLU weights included in Q) and polishing from the early-stopped weights (Table 2).

| μ | Seeds at a numerical minimum | Negative curvature (seeds) | Norm growth (seeds) | Training fit vs runaway |
| --- | --- | --- | --- | --- |
| 10⁻⁶ | 0 of 6 | up to 9 eigenvalues | 21–31× | −3% to +13% |
| 10⁻⁵ | 4 of 6 | up to 4 eigenvalues | 9–43× | 12–22% worse |
| 10⁻⁴ | 6 of 6 | none | 1.6–16× | 15–39% worse |

Under a second-order certificate, all six seeds at μ = 10⁻⁴ were certified. (The originally pre-registered criterion, which required 10 stationary iterations, proved unreachable because the optimiser stops a few iterations after reaching a minimum; the certificate was defined afterwards and is reported alongside the original verdict.) Saturation fell to at most 0.5% and cross-edge cancellation to 2.2–2.9.

Against a regularised teacher, students recover its units and edges well, and the null-space penalty partly acts as gauge fixing (post hoc; Appendix F.3).

**Convergence is not uniqueness.** Independent seeds at certified minima still disagreed: their median prediction difference fell by only 26%. With the runaway removed, the loss has several distinct isolated minima. At the certified μ = 10⁻⁴ minima, matched edge functions correlate at a median 0.37, and at 0.43 with their blind-direction parts removed, whereas early-stopped edges agree mainly in their blind parts (0.49, falling to 0.01 once those are removed): the remaining disagreement is functional, not a matter of the affine null space (post hoc; Appendix F.8).

**Three kinds of flatness.** Exact symmetries are flat by construction and excluded analytically (Section 8). Near-flat directions from poor conditioning are not harmless: certified points with many of them can be degenerate, and one real-data point that passed the certificate kept descending on a longer restart while its squared norm grew 6.7-fold. We therefore report the number of near-flat directions with every certificate and probe degenerate points before counting them as minima. Certified is not unique, and not necessarily good: some certified minima lie at higher loss than points the regularisation path finds at smaller norm.

**Extrapolation.** On the held-out co-solvent, RMSE was 0.142 for the early-stopped model, 1.026 after polishing at μ = 10⁻⁶ and 0.33 at μ = 10⁻⁴. Bounding the runaway recovers most of the damage, but early stopping remains the most accurate choice here, though its stopping point was selected on this fold: on an ill-conditioned problem, stopping early keeps the poorly determined directions near their starting values, much as ridge regularisation would (Yao et al., 2007; Ali et al., 2019).

**Structured alternatives.** Two architectures designed to remove runaway routes by construction were less accurate than the early-stopped baseline (Appendix F.2).

## 10 Recommendations for practitioners

1. **Penalise the blind spot, and everything else that is free.** If a KAN uses a second-difference or curvature penalty, add a small penalty on the constant-plus-linear component of each edge's coefficients and on each SiLU weight, in every layer, and give biases and embeddings a small penalty too: penalising only part of the free directions redirects the growth into the rest. This restores bounded minima, not the accuracy of early stopping.
2. **Calibrate that penalty by certification and a plateau, not by growth.** Choose its weight so that models certify and the regularisation path levels off; a rule based on norm growth chose a weight too small in our B-spline replication.
3. **Check input correlations, including between hidden units.** Strongly correlated or few-valued inputs make a trained nonlinear layer vulnerable, and hidden units that become correlated expose the next layer.
4. **Judge convergence by certification, not by norms or the loss.** Large growth can end at a finite minimum; a runaway is growth that never reaches a certified one. Report the number of near-flat directions with every certificate, and probe degenerate points before calling them minima.
5. **Do not read edges from unconverged models.** Compare edges across seeds only after removing exact symmetries, and only for models at certified minima.
6. **Validate on genuinely new conditions.** A held-out fold from the training distribution can understate extrapolation error severalfold.

**Choosing by goal.** Early stopping and the null-space penalty serve different aims:

| Goal | What to do |
| --- | --- |
| Best held-out accuracy | Keep early stopping as the baseline (in our case selected on the same fold) |
| Reading or comparing edges | Close the null space, penalise biases and embeddings, certify, and compare edges across certified seeds |
| Diagnosing a trained model | Certify, count near-flat directions, check input and hidden-unit correlations |

## 11 Limitations and research process

**Scope.**

- One real dataset and one synthetic teacher; small networks (\[4, 3, 1\] and \[10, 6, 1\]). Deeper or wider models, multiplication nodes and grid updates were not tested, so whether the same small μ suffices at scale is open.
- Two edge types, Gaussian RBFs and cubic B-splines on fixed grids, each with a SiLU base.
- For Gaussian-RBF edges our P1 is a surrogate of Bagrow's penalty (second differences instead of his curvature Gram matrix); his B-spline form was tested exactly (Section 8). We did not vary the SiLU term's weight.
- Receding behaviour is shown over γ = 10⁻² to 10⁻⁷ only, at ρ = 0.95 and on the real data; the dose thresholds are synthetic; and the strong real-data degradation from polishing did not appear in the synthetic study, so we do not claim it as a general law.
- The null-space weight in the shared-curve model was carried over from the synthetic calibration.

**Post-hoc elements.** The analyses listed in Appendix B were defined after seeing results; we report them as exploratory, never in place of pre-registered verdicts, and say where their rules were fixed before computing. One review-stage decomposition on the development fold did not show the signature its plan expected (Section 5).

**Open questions.** A theory of the tilted valley, combining the null-space penalty with a concurvity regulariser (Siems et al., 2023), and whether the residual edge disagreement at certified minima matters scientifically all remain open.

**Process.** Each study committed its predictions, thresholds and interpretation rules before its code existed, and every verdict is linked to its commit. Appendix D describes the protocol and the AI assistance used under it, every change made after an outcome was known (five, all in the B-spline study's calibration step, none touching a prediction, threshold or interpretation rule), and a follow-up dropped because its precondition failed.

## References (to verify before submission)

- Agarwal, R. et al. (2021). Neural Additive Models: Interpretable machine learning with neural nets. NeurIPS.
- Bagrow, J. (2026). KANs need curvature: penalties for compositional smoothness. arXiv:2605.02190.
- Eilers, P. H. C. and Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. Statistical Science.
- Kůrková, V. and Neruda, R. (1994). Uniqueness of functional representations by Gaussian basis function networks. ICANN.
- Li, Z. (2024). Kolmogorov-Arnold Networks are Radial Basis Function Networks. arXiv.
- Liu, Z. et al. (2025). KAN: Kolmogorov–Arnold Networks. ICLR 2025. arXiv:2404.19756.
- Marra, G. and Wood, S. N. (2011). Practical variable selection for generalized additive models. Computational Statistics & Data Analysis.
- Siems, J. et al. (2023). Curve your enthusiasm: concurvity regularization in differentiable generalized additive models. NeurIPS.
- Soudry, D. et al. (2018). The implicit bias of gradient descent on separable data. JMLR.
- Sussmann, H. J. (1992). Uniqueness of the weights for minimal feedforward nets with a given input-output map. Neural Networks.
- de Blasio, P., Elsborg, J., Vegge, T., Flores, E. and Bhowmik, A. (2024). CALiSol-23: Experimental electrolyte conductivity data for various Li-salts and solvent combinations. Scientific Data 11, 750. doi:10.1038/s41597-024-03575-8.
- Cang, Y., Liu, Y. and Shi, L. (2024). Can KAN work? Exploring the potential of Kolmogorov-Arnold networks in computer vision. arXiv:2411.06727.

* Ali, A., Kolter, J. Z. and Tibshirani, R. J. (2019). A continuous-time view of early stopping for least squares regression. AISTATS.
* Ramakrishnan, R., Dral, P. O., Rupp, M. and von Lilienfeld, O. A. (2015). Big data meets quantum chemistry approximations: the Δ-machine learning approach. Journal of Chemical Theory and Computation.
* Wood, S. N. (2017). Generalized Additive Models: An Introduction with R, 2nd ed. CRC Press.
* Yao, Y., Rosasco, L. and Caponnetto, A. (2007). On early stopping in gradient descent learning. Constructive Approximation.

## Appendix F: Supplementary results

These passages were moved from the main text for length; they carry the same evidence and the same labels.

**F.1 Freezing the second layer (exploratory arm).** Freezing the second layer instead of the first leaves a non-convex problem. The synthetic models still certified (16 of 16), but the real first layer grew by factors of 10⁴ to 10¹⁰, mostly through cancellation across its edges, and its endpoints were highly degenerate, with 21–41 near-flat directions; one endpoint that passed the certificate turned out, on a longer restart, to be still drifting outward (exploratory arm; post-hoc follow-up).

**F.2 Structured alternatives.**

**Structured alternatives.** We tested two architectures designed to remove runaway routes by construction, against the early-stopped baseline, with leave-one-co-solvent-out (LOCO) validation:

| Model | Idea | LOCO RMSE | Converged | Seed agreement |
| --- | --- | --- | --- | --- |
| Baseline KAN-delta | As above | 0.255 | 0/10 | 0.112 |
| Shared-curve KAN | One curve per input, mixed by penalised weights | 0.292 | 10/10 | 0.187 early; 0.085 polished |
| Two-anchor excess model | Pure-solvent offset plus x(1 − x) mixing term | 0.283 | not tested | not tested |

Seed agreement is the median prediction difference between seeds (lower is better). The shared-curve model converged every time, ending with no near-flat directions where the baseline had a median of 106 (range 86–116, at non-stationary points), and it was less accurate. At the early-stopped state its seeds agreed less than the baseline's (the pre-registered comparison); after polishing to certified minima their median disagreement fell to 0.085, with some seed pairs reaching the same minimum (post-hoc). The excess model's "pure co-solvent offset" tracked viscosity rather than dielectric constant (rank correlation −0.85, post hoc), reflecting that no training row contains a pure co-solvent. The shared-curve model's learned viscosity dependence also had the wrong sign on more than 10% of rows in 3 of 10 seeds (a pre-registered prediction that failed). Chemically, conductivity should fall with viscosity at fixed composition, but in these data viscosity and dielectric constant rise together (r = 0.966), and higher permittivity favours salt dissociation; with the two nearly collinear, the data cannot separate their partial effects, which is the kind of non-identifiability this paper is about.

**F.3 The regularised teacher and gauge fixing.**

**A known solution becomes recoverable.** Under the regularised loss, the original teacher is no longer a minimum. Optimising it under the same loss gives a "regularised teacher" computing nearly the same function (correlation 0.9992). Students match its units at 0.87–0.90 and its edges at 0.97–0.98, against 0.68–0.72 and 0.84–0.91 for the original teacher. This comparison was defined after the results and is exploratory.

**The penalty acts partly as gauge fixing.** At the certified minima, the layer-2 constant shift, an essentially exact symmetry, owes 96% of its curvature to μQ: the regulariser selects one representative. The layer-1 constant shift (46%) and unit rescaling (24%) are only approximate symmetries in this architecture, because Gaussian bumps cannot represent constants exactly and tanh breaks rescaling.

**F.4 What a short path cannot tell.**

**What a short path cannot tell.** We also predicted that a short regularisation path (γ = 10⁻³, 10⁻⁵, 10⁻⁷), summarised by D = ‖θ\*(10⁻⁷)‖² / ‖θ\*(10⁻⁵)‖², would diverge more strongly as ρ grows. It did not (Spearman 0.71 under P0 and 0.14 under P1, against the required 0.8): D was already high at every dose, including ρ = 0. A post-hoc check, with its classification rule fixed before computing, explains why. Where the unregularised polish certified, mostly at low ρ, 30 of 32 paths were heading towards that certified finite minimum or its valley, none moved outward past it, and 2 found a better minimum at smaller norm. Along one direction with curvature λ, D = ((λ + 2γ₁)/(λ + 2γ₂))² with γ₁ = 10⁻⁵ and γ₂ = 10⁻⁷, so a finite minimum whose curvature lies within or near the window 2·10⁻⁷ to 2·10⁻⁵ keeps D large while the path is still approaching it; the observed low-ρ medians (4.54 to 15.05) correspond to λ = 6.7·10⁻⁶ to 1.7·10⁻⁵. A two-decade D therefore cannot separate a distant finite minimum from a receding one. Receding behaviour (Section 6) is shown only at ρ = 0.95 and on the real data, over the tested range; at low ρ the minima are finite but distant; at intermediate ρ these tests do not settle the question. A secondary prediction, that the affine share of the parameter change would rise steadily with ρ, failed.

**F.5 The exact symmetry of B-spline edges.** Because B-splines sum exactly to one, constants can be moved between a unit's incoming edges and its bias without changing the loss: an exact 15-dimensional symmetry in the \[4, 3, 1\] model, which holds only approximately for RBF edges. It made the solver hang at converged points and would have made every certified point look degenerate. We excluded these analytically constructed directions, never fitted ones, from certification and from the degeneracy count; exact symmetries are redundancy in the parameterisation, while the runaway directions, which are approximate and data-dependent, remain subject to every check.

**F.6 The boundedness criterion.** Let L be continuous and bounded below, and R ≥ 0 continuous and coercive (R(θ) → ∞ as ‖θ‖ → ∞). Then L + γR is coercive for every γ > 0 and has a global minimiser θ\_γ. If L has a finite minimiser θ₀, then L(θ\_γ) + γR(θ\_γ) ≤ L(θ₀) + γR(θ₀) and L(θ₀) ≤ L(θ\_γ), so R(θ\_γ) ≤ R(θ₀) for every γ, and coercivity keeps the θ\_γ bounded. Conversely, if the θ\_γ stay bounded as γ → 0, they have a limit point θ̄. For any θ, L(θ\_γ) ≤ L(θ) + γR(θ) − γR(θ\_γ) ≤ L(θ) + γR(θ), so by continuity L(θ̄) ≤ L(θ): θ̄ is a finite minimiser of L. The path in Section 6 follows certified local minima rather than global ones, so it provides evidence for the criterion's conclusion, not a proof of it.

**F.7 Robustness of the regularisation path (post hoc).** Polishing from independent cold starts found no small-norm basin at any γ; at γ = 10⁻⁷ under P1, 8 of 9 cold starts reached a *lower* regularised objective than the warm path, at a larger norm. The warm path therefore understates the divergence. A straight line along the final drift direction rises almost at once: the minimisers recede along a curved valley, not along a ray. The growth sits in the penalty's blind directions, the affine RBF coefficients and SiLU weights of both layers, while the curved coefficients and the salt embedding shrink.

**F.8 Attribution, sign checks and edge agreement on fold A (post hoc).** Three decompositions on the archived weights, with their plan committed before computing. (i) Splitting each polishing change into its blind-direction part (the edge-wise null-space projection of the RBF-coefficient change plus the SiLU-weight change) and the remainder, and applying each alone to the early-stopped model: the plan expected the blind part to be nearly free on training rows and costly on fold A. It was not: alone it raises both the ensembles' training and fold A RMSE above 2.9 (single seeds vary), and the remainder offsets most of that on training rows and less on fold A. On fold A the interaction is large and negative (−3.2 at μ = 0, −2.9 at μ = 10⁻⁶); on training rows it cannot be read as a share of the damage, because polishing lowers the training error. (ii) The sign of d log k/d(1000/T), row by row with the other inputs at their observed values: the anchor supplies the expected sign on almost every row, while polishing turns it on 25–33% of fold A rows at the weaker penalties (0.3% at early stopping, 5% at μ = 10⁻⁴), concentrated at low temperature and high x\_co; the viscosity sign shows no clean ordering across states, although at μ = 10⁻⁴ its fold A violations sit mostly in the highest-x\_co third (185 of 217 rows). (iii) Correlations of matched edge functions across seeds, with and without each edge's blind-direction part. The tables are generated from the results file.

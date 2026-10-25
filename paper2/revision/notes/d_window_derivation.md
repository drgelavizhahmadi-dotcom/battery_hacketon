# D and the γ window: a derivation note (dated 2026-10-25)

Algebra only. There are no new results and no verdicts. This note supports claims-table item J5.

## Convention

- **The penalty.** The committed L2 path minimises L(θ) + γ‖θ‖², with no factor 1/2: the convention of the infimum test, the
  dose-response study and step 2's Arm D.
- **D (dose study, `dose_prereg.md`).** D = ‖θ*(γ₂)‖² / ‖θ*(γ₁)‖², with γ₁ = 1e-5 and γ₂ = 1e-7: the smaller-γ point over the
  larger-γ point.
- **Labelling.** The brief that requested this note writes the ratio with γ₁ in the numerator. Its expected formula below is the
  committed D, and this note uses the committed definition.

## Model

- **The loss along one direction.** Along one direction, L is quadratic with curvature λ > 0 and minimiser θ_min:

  ```latex
  L(\theta) = L_0 + \tfrac{\lambda}{2}\,(\theta - \theta_{\min})^2
  ```

- **The regularised minimiser.** Stationarity of L + γθ² gives λ(θ − θ_min) + 2γθ = 0, so

  ```latex
  \theta^*(\gamma) = \frac{\lambda}{\lambda + 2\gamma}\,\theta_{\min}
  ```

- **D.**

  ```latex
  D = \frac{\theta^*(\gamma_2)^2}{\theta^*(\gamma_1)^2} = \left(\frac{\lambda + 2\gamma_1}{\lambda + 2\gamma_2}\right)^{2}
  ```

- **Properties of D.**
  - D does not depend on θ_min, so it carries no information about how far away the minimum is.
  - D → 1 when λ ≫ 2γ₁ (the minimum is already reached at both γ).
  - D → (γ₁/γ₂)² = 1e4 when λ ≪ 2γ₂.
  - In several directions, with an isotropic penalty, each eigen-direction contributes its own factor λᵢ/(λᵢ + 2γ). D is then a
    θ_min-weighted mixture of these single-direction values.

## The curvature window observed at low ρ

- **The observed values.** The low-ρ cells are ρ = 0 and 0.5, P0 and P1. Their median D values are (`paper2/revision/dose_results.json`, `cells.*.D_med`, commit 541355a):

| Cell | Median D | Certified polishes (of 8) |
|---|---|---|
| P0, ρ = 0 | 8.014 | 7 |
| P0, ρ = 0.5 | 15.049 | 3 |
| P1, ρ = 0 | 4.542 | 7 |
| P1, ρ = 0.5 | 4.882 | 4 |

- **Inverting the formula.** With √D = (λ + 2γ₁)/(λ + 2γ₂),

  ```latex
  \lambda = \frac{2\gamma_1 - 2\sqrt{D}\,\gamma_2}{\sqrt{D} - 1}
  ```

  - D = 4.542 (the smallest) gives λ = 1.730e-5.
  - D = 15.049 (the largest) gives λ = 6.677e-6.
- **So a single-direction finite minimum with curvature between 6.7e-6 and 1.7e-5 reproduces the observed range.** That is, curvature
  within the window 2γ₂ = 2e-7 … 2γ₁ = 2e-5.
- **An order of magnitude outside the range:**
  - λ = 1.73e-4 (10 × the upper end) gives D = 1.24;
  - λ = 6.68e-7 (the lower end / 10) gives D = 567.

## Claim (as used in the claims table)

A finite minimum whose curvature lies within or near the γ window keeps D large while the path is still approaching it. So a
two-decade D cannot separate a distant finite minimum from a receding one.

## Superseded figure

The earlier figure "curvature ~5e-6 gives D ~ 8" used a different convention: L + (γ/2)‖θ‖², for which
D = ((λ + γ₁)/(λ + γ₂))² and λ = 5e-6 gives 8.65. It is superseded. Under the committed convention, λ = 5e-6 gives D = 23.1.

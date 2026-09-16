# notes/derivations.md

Step numbering matches docs/math_fixed.md §B (the ten-step table). This file is append-only
across phases: Phase 1 fills Steps 1-4 and 10; Phase 2 fills Steps 5-9 (see notes/decisions.md
for the checkpoint history behind each choice).

## Symbols (English, restated from docs/math_fixed.md §A)

- `Y ⊂ R`: the outcome space (Borel-measurable subset of the reals).
- `P_t`: the joint law, at year t, of (outcome Y, survey-response indicator R ∈ {0,1}) over the
  general population.
- `P_{t,Y}`: the Y-marginal of P_t, i.e. `P_{t,Y}(A) = P_t(A × {0,1})` for Borel A.
- `S_t`: the law of Y given R=1 at year t, i.e. `S_t(A) = P_t(A × {1}) / ρ̄_t`.
- `π_t(y) := P_{P_t}[R=1 | Y=y]`: the response probability given outcome y (a regular
  conditional probability, so `P_t(A×{1}) = ∫_A π_t dP_{t,Y}`).
- `ρ̄_t := P_{P_t}[R=1]`: the overall response rate at year t (a scalar in (0,1]).
- `λ`: the dominating measure; `p_t`, `s_t`: densities of `P_{t,Y}`, `S_t` w.r.t. λ.
- `ρ_t := dP_{t,Y}/dS_t`: the density ratio (population over survey).
- `w(y,y';t) := π_t(y)/π_t(y')`: relative selection probability (Assumption 1: constant in t).
- `g`, `c_t`: the factor form of π_t under Assumption 1, `π_t(y) = c_t · g(y)`.
- `θ*, α*`: the truth in the separable class `r(y,t) = θ(y) + α(t)`.
- `m < M`: last year with P_t data, last year with S_t data.

## Step 1 — Bayes

`s_t(y) = π_t(y) p_t(y) / ρ̄_t`. Derivation: `S_t(A) = P_t(A×{1})/ρ̄_t = (∫_A π_t dP_{t,Y})/ρ̄_t`
by the regular-conditional-probability definition of π_t, and `P_{t,Y}(dy) = p_t(y) λ(dy)` by
definition of the density, so `S_t(dy) = π_t(y) p_t(y) / ρ̄_t · λ(dy)`, i.e. the stated identity.
Code: `fusion/dgp.density_grid` computes `s_t = π_t · p_t / ρ̄_t` directly on a y-grid (Phase 1
figure (a)); `ρ̄_t` is also read off there for figure (b).

## Step 2 — ratio

`ρ_t(y) = p_t(y)/s_t(y) = ρ̄_t / π_t(y)`, immediate from Step 1 (divide both sides by `p_t/s_t`'s
reciprocal). Not implemented as a standalone function since Step 3 replaces it with the factored,
c_t-free form actually used everywhere downstream.

## Step 3 — factor form (Assumption 1)

Assumption 1 (`π_t(y)/π_t(y')` independent of t) holds **iff** `π_t(y) = c_t · g(y)` for some
year-independent `g` and year-dependent scalar `c_t` (this DGP: `g(y) = Φ(βy)`,
`c_t = c_t_intercept + c_t_slope · t`; test T3 checks the ratio is t-free to 1e-12). Substituting
into Step 2: `ρ_t(y) = ρ̄_t/(c_t g(y))`, and since `ρ̄_t = c_t · E_{P_t}[g(Y)]` (definition of
overall response rate under the factor form), the `c_t` cancels:
`ρ_t(y) = E_{P_t}[g(Y)] / g(y)`. Taking logs and anchoring at t=m (defining `θ* := log ρ_m`, so
`α*(m) = 0` by construction):
`log ρ_t(y) = −log g(y) + log E_{P_t}[g(Y)] =: θ*(y) + α*(t)`.
For this DGP, `g(y)=Φ(βy)` and `E_{P_t}[g(Y)] = Φ(a_t)` (shown in docs/math_fixed.md §D via
`W − βY ~ N(−βm_t, 1+β²σ²)`, so `P(W<0) = Φ(a_t)` with `a_t = βm_t/√(1+β²σ²)`), giving the
closed forms `θ*(y) = log Φ(a_m) − log Φ(βy)`, `α*(t) = log Φ(a_t) − log Φ(a_m)`.
Code: `fusion/dgp.theta_star`, `fusion/dgp.alpha_star`, `fusion/dgp.a_t`.

## Step 4 — estimator

For `t > m` we only observe `S_t`, not `P_t`, so `μ(t) = E_{P_t}[Y]` must be recovered from `S_t`
via the ratio: `μ(t) = E_{S_t}[ρ_t(Y) · Y] = E_{S_t}[e^{θ*(Y)} Y] / E_{S_t}[e^{θ*(Y)}]`, using
`ρ_t(y) = e^{θ*(y)+α*(t)}` and cancelling `e^{α*(t)}` between numerator and denominator (this is
why the estimator needs no α: it is invariant to `θ → θ+c` for any constant c). This is the
Hájek self-normalised form; code: `fusion/estimate.mu_tilde` (Phase 3), using the fitted `θ̃` in
place of `θ*` here.

## Step 10 — the DGP itself (docs/math_fixed.md §D, CP1 = `default`)

Chosen (notes/decisions.md, CP1): `P_{t,Y} = N(m_t, σ²)`, `σ=1`; `π_t(y) = c_t Φ(βy)`, `β=1.5`;
`m_t = −0.5+0.25t`; `c_t = 0.9−0.06t`; `m=6, M=9, n=2000`. Survey sampling = draw `Y~P_{t,Y}`,
keep iff `R=1 ~ Bern(π_t(Y))` (this literally implements "condition on R=1", i.e. draws from
`s_t` by Step 1). Closed forms proved via Stein's lemma (`E[(Y−m)h(Y)] = σ²E[h'(Y)]` for
`h=Φ(β·)`, giving `E_{S_t}[Y]`) are in docs/math_fixed.md §D and verified empirically by test T1
(simulation vs. closed form) and T2 (kept-sample law vs. numeric integration of `s_t`).
Code: `fusion/dgp.py` (all functions); figures: `figures/phase1a_densities_*.png` (p_t, s_t, π_t
per year) and `figures/phase1b_mu_vs_survey_vs_rho_*.png` (μ(t), E_{S_t}[Y], ρ̄_t vs t) — see
LOG.md for the run that produced them and the 1-line description of each in the CP1 report.

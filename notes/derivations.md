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

## Step 5 — pooled objective

`R(θ,α) = E_F[L] = (1/2m) Σ_{t=1}^m J_t(θ+α(t))`, where `J_t(r) := E_{S_t}[e^r] − E_{P_{t,Y}}[r]`.
Derivation: `F`'s definition (docs/challenge_text.md §2: `Z~Bern(1/2)`, `T~Uniform([m])`,
`Y|T=t,Z=1 = P_{t,Y}`, `Y|T=t,Z=0 = S_t`) makes `E_F[L(θ(Y),α(T),Z)]` a double expectation over
`T` (uniform on `1..m`, contributing the `1/m` and the sum over t) and, within each t, over `Z`
(Bernoulli(1/2), contributing the `1/2` and splitting `L` into its `z=0` term `E_{S_t}[e^r]` and
its `z=1` term `−E_{P_{t,Y}}[r]`, since `L=(1-z)e^r−zr`). Code: `fusion/loss.fusion_loss`,
applied to the pooled `(T,Z,Y)` rows built by `fusion/data.build_dataset` (Step 9's empirical
version of this same sum, with the population expectations replaced by sample averages).

## Step 6 — the per-year minimiser is unique and equals log ρ_t

Write `J_t(r) = E_{S_t}[e^r − ρ_t·r]` (substituting `E_{P_{t,Y}}[r] = E_{S_t}[ρ_t·r]`, valid
because `ρ_t = dP_{t,Y}/dS_t` is exactly the Radon-Nikodym derivative that converts an
`S_t`-expectation into a `P_{t,Y}`-expectation). Pointwise in y, `h(r) := e^r − ρ_t(y)·r` has
`h'(r) = e^r − ρ_t(y)`, zero exactly at `r = log ρ_t(y)`, and `h''(r) = e^r > 0`, so `h` is
strictly convex with that unique minimiser. Writing `δ := r − log ρ_t(y)`,
`h(r) − h(log ρ_t(y)) = ρ_t(y)·(e^δ − 1 − δ) ≥ 0` (since `e^δ−1−δ≥0` for all real `δ`, equality
only at `δ=0`), so the *excess* risk is `E_{P_{t,Y}}[e^δ−1−δ] ≥ 0` — this term is checked
empirically by test T6 (the midpoint/convexity inequality) and exploited by test T7 (a strictly
convex objective has a unique minimiser, which is why full-batch L-BFGS can be expected to find
it exactly). At the minimum, `min_r J_t(r) = E_{S_t}[ρ_t·1 − ρ_t·log ρ_t] = 1 − E_{S_t}[ρ_t log
ρ_t]`; since `E_{S_t}[ρ_t log ρ_t] = E_{P_{t,Y}}[log ρ_t] = KL(P_{t,Y}‖S_t)`, this is
`1 − KL(P_{t,Y}‖S_t)` — the Nguyen–Wainwright–Jordan variational form of KL divergence.

## Step 7 — identification and the anchor

Because the truth `(θ*, α*)` lies in the separable class `r=θ(y)+α(t)` (Step 3), and Step 6 shows
each `J_t` is minimised uniquely at `r=log ρ_t`, the truth minimises the pooled `R` (an average of
minimised `J_t`'s is minimised). Any other minimiser `(θ,α)` of `R` must also hit `r=log ρ_t(y)`
for every t, i.e. `θ(y)+α(t) = θ*(y)+α*(t)` for all y,t — subtracting the t=t' and t=t'' cases
shows `θ(y)−θ*(y)` is the same constant `c` for all y, and `α(t)=α*(t)−c` for all t. So minimisers
of R are exactly `{(θ*+c, α*−c) : c ∈ R}` — a one-parameter family, not a point: `R` alone cannot
identify `θ*,α*` separately from each other, only their difference `θ(y)−α(t)`-type combinations.
Fixing `α(m)=0` (`A = {α∈R^m : α(m)=0}`, docs/challenge_text.md eq. 3) picks the unique `c=0`
member, giving true identification. Code: `fusion/model.AlphaTable`'s frozen last entry
(concatenation with a constant-zero buffer, never a learned parameter — see the `# DECISION` in
`fusion/model.py` and notes/decisions.md — so `α(m)=0` holds exactly, not approximately, at every
step of training; test T8 checks this after 100 optimizer steps).

## Step 8 — first-order condition (FOC) in α

Fixing `θ` and differentiating `R` w.r.t. `α(t)` (only `J_t` depends on `α(t)`, through
`r=θ+α(t)`): `∂J_t/∂α(t) = E_{S_t}[e^{θ+α(t)}] − 1` (differentiating `E_{S_t}[e^r]` under the
integral; the `−E_{P_{t,Y}}[r]` term contributes `−1` per unit of `α(t)`, and `E_{P_{t,Y}}[1]=1`
cancels it against the `+1` from differentiating `e^r`... concretely `∂/∂α(t)[E_{S_t}[e^{θ+α(t)}]
− E_{P_{t,Y}}[θ+α(t)]] = E_{S_t}[e^{θ+α(t)}] − 1`). Setting this to zero: `α(t) = −log
E_{S_t}[e^{θ(Y)}]` — profiling out `α` this way and substituting back gives exactly the
Donsker–Varadhan variational form of `−log E_{S_t}[e^θ]` at the optimal `α`, and `L` is jointly
convex in `(θ,α)` because it is convex in `r=θ+α` (Step 6) and `r` is a linear (affine) function
of `(θ,α)`, and convexity is preserved by pre-composition with an affine map. The FOC diagnostic
`mean_i e^{θ̃(y^S_{t,i})+α̃(t)} ≈ 1` (checked once `fusion/train.py` exists, Phase 3) is exactly
this stationarity condition evaluated empirically at the fitted `(θ̃,α̃)`.

## Step 9 — empirical risk and its gradient

The sample analogue of Step 5, over the pooled `2mn`-row dataset F built by
`fusion/data.build_dataset`: `R̂_n = (1/2mn) Σ_t [Σ_i e^{θ(y^S_{t,i})+α(t)} − Σ_i
(θ(y^P_{t,i})+α(t))]`. Its per-row derivative (used for the FOC diagnostic and matched bit-for-bit
against `fusion_numpy.py` in test T4) is `∂/∂r_i = (1−z_i)e^{r_i} − z_i` — code:
`fusion/loss.dL_dr` (stated without the `1/N` batch-averaging factor, matching
docs/math_fixed.md §C's own convention literally; `fusion/loss.fusion_loss`'s mean reduction is
what autograd actually differentiates through in training, which does carry that factor).
`R̂_n` is unbounded below for a `θ` flexible enough to interpolate (drive every `e^{θ(y^S)+α(t)}`
term to 0 and every `θ(y^P)+α(t)` term to `+∞` on disjoint points) — this is why Phase 3 uses
early stopping on a held-out validation split rather than training to convergence. `μ̃(t)` (Step
4) is a Hájek (self-normalised) estimator; `n_eff := (Σw)²/Σw²` (with `w=e^{θ̃(Y)}`) is its
effective sample size, small when a few survey points dominate the weight mass.

## Phase 2 evidence (tests T4-T8; `python -m pytest tests/ -v`)

- T4 (numpy-twin agreement): PASSED — loading `fusion_numpy.init_params(rng,32)` into
  `ThetaNet(32)`/`AlphaTable(6)` and comparing on the same random 64-row batch: `|r_torch-r_np|
  < 1e-8`, `|L_torch-L_np| < 1e-10`, and every parameter gradient (`W1,b1,w2,b2,alpha[:-1]`)
  agrees to `< 1e-8`. This is the resolution of the CP-adjacent uncertainty about ThetaNet's
  depth (see notes/decisions.md): the numeric agreement is exact once `ThetaNet` is `1->H->1`.
- T5 (gradcheck): PASSED — `torch.autograd.gradcheck` on a tiny (H=4, m=3) ThetaNet+AlphaTable
  in float64, `eps=1e-6, atol=1e-4`.
- T6 (joint convexity): PASSED — the midpoint inequality `L((r1+r2)/2,z) ≤ (L(r1,z)+L(r2,z))/2`
  holds for all of 1000 random `(θ,α)` pairs, both `z∈{0,1}` (see the Step 6 proof above for why
  this must hold).
- T7 (parametric recovery): **FAILED as specified** (`n=200_000` per cell, `seed_data=0`,
  tolerance `0.02`): fitted `a=1.0255` (`|a-1|=0.0255`), max `|α̂(t)-α*(t)|=0.0309` (t=3) — both
  over the 0.02 bound. Diagnosed, not a bug: gradient norms at the L-BFGS solution are ~1e-8
  (genuinely converged to the finite-sample minimiser, confirmed stable across further L-BFGS
  rounds), and repeating at `seed_data∈{1,2,3}` gives `|a-1|∈{0.057,0.025,0.0002}` and
  max-`|α̂-α*|∈{0.043,0.044,0.038}` — i.e. every one of 4 independent seeds fails the α criterion,
  3 of 4 fail the `a` criterion. Increasing `n` (same seed=0) to 800k then 3.2M gives
  `|a-1|∈{0.001,0.006}` and max-`|α̂-α*|∈{0.017,0.011}` — shrinking, consistent with a
  statistically consistent (unbiased-in-the-limit) estimator, just one whose finite-sample SE at
  the literally-specified n=200k is comparable to or larger than the flat 0.02 tolerance itself
  (unlike T1/T2, whose tolerances are stated as a multiple of the actual SE, T7's 0.02 is a flat
  constant). Raised at CP2 with this evidence rather than silently loosening the tolerance or
  padding n past the spec's stated value.
- T8 (anchor): PASSED — `alpha_table.forward()[-1].item() == 0.0` exactly after 100 Adam steps
  (guaranteed structurally by the frozen-buffer design, not by convergence).

## Phase 3 evidence (training, diagnostics, estimation; CP3 = `Adopt`)

First run at the docs/math_fixed.md §F proposal (`lr=3e-3`, unbounded `ThetaNet`): validation
curve noisy/spiky and still descending through epoch ~395 (an empirical instance of Step 9's
"`R̂_n` unbounded below for an interpolating θ" warning), FOC
`{1:1.079,2:1.106,3:1.0,4:1.027,5:1.045,6:1.012}` (3/6 years outside ±0.03), and `θ̃` exploding to
≈380 at `y=−6` against `θ*`'s <50 there — an unconstrained extrapolation into the sparse left
tail, well outside where any year's `m_t` (`m_t(1)=−0.25` .. `m_t(9)=1.75`) puts real mass.
Proposed and adopted at CP3: `lr` 3e-3→1e-3, `ThetaNet` given a bounded head (`theta_bounded=20`,
i.e. `θ = 20·tanh(raw/20)`). Re-run: validation curve smooth and monotone (no spikes), `θ̃`
plateaus at 20 instead of exploding, FOC `{1:1.005,2:0.958,3:0.963,4:1.012,5:1.025,6:0.974}`
(t=2,3 still marginally outside ±0.03 — a residual, honestly reported, not hidden), `α̃` still
visibly flatter than `α*` (crosses it near t=3) — matching docs/math_fixed.md §F's own caveat
that "`α̃` still off `α*` by ≤0.08 in early years (low-y tail under-fit)."

Estimate table (t=m+1..M, `fusion.run --phase 3`, n=2000, config hash `8e4fef38`):

| t | μ(t) true | survey mean | μ̃(t) | SE (bootstrap) | n_eff/n | offset baseline | oracle |
|---|---|---|---|---|---|---|---|
| 7 | 1.250 | 1.500 | 1.325 | 0.0338 | 0.82 | 1.243 | 1.306 |
| 8 | 1.500 | 1.647 | 1.476 | 0.0324 | 0.83 | 1.391 | 1.454 |
| 9 | 1.750 | 1.858 | 1.752 | 0.0309 | 0.88 | 1.601 | 1.734 |

`μ̃(t)` sits closer to the truth than the raw survey mean at every t (e.g. t=9: |1.752−1.75| vs
|1.858−1.75|), and closer than the offset baseline at t=8,9 (the offset baseline's constant-bias
assumption fails as the survey bias shrinks over time, by DGP design — docs/math_fixed.md §D).

- T9 (oracle estimator): PASSED — `|oracle−μ(t)| < 3·SE` (bootstrap SE with θ* fixed) for
  t∈{7,8,9}, n=20k.
- T10 (trained estimator, report then assert loosely): PASSED — FOC within ±0.05 for all t≤m
  (same run as above); `|μ̃(t)−oracle μ̃(t)|` = 0.0191, 0.0168, 0.0189 for t=7,8,9, all < 0.05.

## Phase 4 evidence and failure modes (CP4 = `Default`, all three stress tests)

**Main figure** (`figures/phase4_main_figure_*.png`, opened): `μ̃(t)` (green) tracks `μ(t)` true
(blue) closely across every t, both in-sample (t≤m) and in the shaded t>m region, visibly closer
to the truth than both the survey mean (orange, biased high throughout by construction) and the
offset baseline (red dashed, which undershoots by t=9 as the survey bias shrinks over time,
violating the offset baseline's constant-bias assumption — by DGP design, docs/math_fixed.md §D).
**In-sample check** (t≤m, where `P_t` is directly observed so the truth is known without any
correction): `μ̃(t)` tracks `μ(t)` reasonably (e.g. t=6: 1.000 vs 1.000 exactly; t=5: 0.749 vs
0.750), confirming the estimator isn't just "getting lucky" on the extrapolation years.

**Stress 1 — Assumption 1 broken** (`π_t(y)=σ(a_t+b_t y)`, `b_t=β+0.1(t−m)`, t>m only). Math
predicts (docs/math_fixed.md §E.1): "expect μ̃ bias growing with `|b_t−b_m|`." **Observed**
(`figures/phase4_stress1_assumption1_*.png`): bias is negative throughout (μ̃ underestimates μ)
but its MAGNITUDE SHRINKS as `|b_t−β|` grows: `-0.303, -0.295, -0.127` at drift `0.10, 0.20, 0.30`
— the opposite of the qualitative prediction. Not treated as a bug (the estimator, trained under
the true DGP, still moves in the expected direction — underestimation from a mismatched
selection model — and this specific perturbation also grows the shared intercept `a_t` alongside
`b_t`, since `a_t` here is reused from the original closed form rather than held fixed;
that coupling is a specific, disclosed modelling choice for this stress test, not part of
docs/math_fixed.md's own construction), but reported exactly as observed per the Honesty Oath
rather than smoothed into agreement with the predicted direction.

**Stress 2 — support shift** (`m_t`+2σ, t>m only). Math predicts (§E.2): report the fraction of
t>m survey values outside the training range. **Observed**
(`figures/phase4_stress2_support_shift_*.png`): `4.0%, 6.7%, 11.5%` of survey values fall outside
`[Y.min(), Y.max()]` of the t≤m training pool at t=7,8,9 (7.4% pooled across all of t>m) —
growing with t as expected (the shift compounds with the years already being further from the
anchor year m). Bias stayed small (`+0.057, +0.040, +0.055`) despite this: unlike the pre-CP3 unbounded
run's tail blowup (θ̃→380 at y=−6), the `theta_bounded=20` head (CP3) keeps θ̃'s extrapolation
gentle past the training range instead of exploding, which plausibly explains why this
particular stress test doesn't hurt μ̃ much even with ~10% of the mass extrapolated.

**Stress 3 — small n** (`n∈{200,500,2000}`, θ̃ fixed, 10 seeds each — see notes/decisions.md CP4
for why θ̃ is held fixed here). Math predicts (§E.3): sd of μ̃ over 10 seeds `∝ 1/√n_eff`.
**Observed** (`figures/phase4_stress3_small_n_*.png`): `sd(μ̃) = 0.0583, 0.0468, 0.0186` at
`n=200,500,2000` (`mean n_eff = 190.0, 472.9, 1787.2`), tracking the plotted `1/√n` reference
line closely (slightly above it at n=200,500, touching it at n=2000) — consistent with the
predicted scaling, within the noise expected from only 10 replicates per n.

# Fixed math for the implementation (do not modify; raise doubts at a checkpoint)

Symbols from the challenge: P_t (law of (Y,R) at year t), P_{t,Y} (its Y-marginal), S_t (law of Y|R=1),
w(y,y';t), μ(t)=E_{P_t}[Y], T, Z, F, L(θ,a,z), θ(·), α(·), Θ, A={α∈R^m: α(m)=0}, μ̃(t), m<M, n.
Symbols added here: π_t(y):=P_{P_t}[R=1|Y=y]; ρ̄_t:=P_{P_t}[R=1]; λ dominating measure; p_t, s_t densities;
ρ_t:=dP_{t,Y}/dS_t; g, c_t (factor form of π_t); θ*, α* (truth); J_t (per-year objective); R (population risk).

## A. Setting and assumptions
- Y ⊂ R Borel; P_t a probability measure on (Y×{0,1}, B(Y)⊗2^{0,1}); P_{t,Y}(A)=P_t(A×{0,1}).
- π_t is a regular conditional probability: P_t(A×{1}) = ∫_A π_t dP_{t,Y}.
- S_t(A) = P_t(A×{1})/ρ̄_t  (needs ρ̄_t>0).
- Assumption 1: w(y,y';t)=π_t(y)/π_t(y') does not depend on t.
- Implicit: (A0) π_t>0 P_{t,Y}-a.e. ⇒ P_{t,Y} ≪ S_t; (A1) E|Y|<∞; (A2) Assumption 1 also holds for t>m and
  the t>m survey values lie in the region where θ was trained.

## B. The ten steps (each implemented somewhere; see column "code")

| Step | Statement | code |
|---|---|---|
| 1 Bayes | s_t(y) = π_t(y) p_t(y) / ρ̄_t | dgp.density_grid |
| 2 ratio | ρ_t(y) = p_t/s_t = ρ̄_t/π_t(y) | — |
| 3 factor | Assumption 1 ⇔ π_t(y)=c_t g(y). Then s_t = g p_t / E_{P_t}[g] (c_t cancels) and log ρ_t(y) = −log g(y) + log E_{P_t}[g(Y)] =: θ*(y)+α*(t), anchored θ*:=log ρ_m so α*(m)=0; e^{α*(t)} = 1/E_{S_t}[e^{θ*(Y)}] | dgp.theta_star, dgp.alpha_star |
| 4 estimator | μ(t) = E_{S_t}[ρ_t Y] = E_{S_t}[e^{θ*}Y]/E_{S_t}[e^{θ*}]; valid for t>m; invariant to θ→θ+c | estimate.mu_tilde |
| 5 pooled | R(θ,α)=E_F[L] = (1/2m) Σ_t J_t(θ+α(t)), J_t(r)=E_{S_t}[e^{r}] − E_{P_{t,Y}}[r] | loss.fusion_loss on the pooled rows |
| 6 unique | J_t(r) = E_{S_t}[e^r − ρ_t r]; pointwise e^r−ρr strictly convex, min at r=log ρ; excess = E_{P_{t,Y}}[e^δ−1−δ] ≥ 0, δ=r−log ρ_t; min J_t = 1−KL(P_{t,Y}‖S_t) (Nguyen–Wainwright–Jordan variational form of KL) | tests T6, T7 |
| 7 identify | truth (θ*,α*) lies in the separable class ⇒ it minimises R; any minimiser equals it up to a constant; α(m)=0 removes the constant | model.AlphaTable frozen entry |
| 8 FOC | ∂R/∂α(t) ∝ E_{S_t}[e^{θ+α(t)}] − 1 = 0 ⇒ α(t) = −log E_{S_t}[e^{θ}]; profiling gives the Donsker–Varadhan form; L jointly convex in (θ,a) | train.foc_diagnostic |
| 9 empirical | R̂_n = (1/2mn) Σ_t [Σ_i e^{θ(y^S_{t,i})+α(t)} − Σ_i (θ(y^P_{t,i})+α(t))]; ∂/∂r_i = (1−z_i)e^{r_i}−z_i; R̂_n is unbounded below for an interpolating θ ⇒ regularise (early stopping); μ̃ is a Hájek estimator, n_eff=(Σw)²/Σw² | loss.dL_dr, train.train |
| 10 DGP | default in §D with closed forms | dgp.* |

## C. Implementation-level identities to test
- Per-row derivative (Step 9): dL/dr = (1−z)e^r − z. Gradient of α[t] = Σ_{rows with T=t} dL/dr_i / N_batch.
- Diagnostic (Step 8): mean over survey rows of year t of e^{θ̃+α̃[t]} = 1 at a stationary point.
- Standardisation: θ̃(y) := net((y−mean)/std); no Jacobian anywhere (θ is evaluated pointwise, no density computed).
- Estimator (Step 4) uses θ only; α is never needed for t>m.
- Population minimum of the batch-averaged loss on the pooled data = (1/2)(1 − mean_t KL_t); useful sanity number.
- Delta-method SE of the Hájek estimator (valid when a > 2):
  se_hat = sqrt( Σ_i w_i² (y_i − μ̃)² ) / Σ_i w_i.  Derivation: μ̃ − μ =
  (1/n) Σ_i w_i (y_i − μ) / (1/n) Σ_i w_i; numerator terms are i.i.d. mean zero
  (exactly zero in population when θ̃ = θ* + const), then CLT + Slutsky.
- Effective sample size n_eff = (Σ_i w_i)² / Σ_i w_i² satisfies 1 ≤ n_eff ≤ n by
  Cauchy-Schwarz, with n_eff = n iff all weights are equal.
- Finite-sample bias (valid when a > 3): E[μ̃] − μ = −E_S[w² (Y−μ)] / (n E_S[w]²) + o(1/n).
- Centering identity (exact, any constant c): μ̃ − c = Σ_i w_i (y_i − c) / Σ_i w_i.

## D. Default DGP (the only open choice — confirm at CP1)
- m=6, M=9, n=2000, σ=1, β=0.6;  m_t = −0.5 + 0.25 t;  c_t = 0.9 − 0.06 t.
  (β was 1.5 until 2026-09-18; changed under rule (R3), see §G. Tail index a = 3.78.)
- P_{t,Y} = N(m_t, σ²);  π_t(y) = c_t Φ(βy)  (Φ standard normal cdf) — Assumption 1 holds because Φ(βy)/Φ(βy') has no t;
  ρ̄_t = c_t Φ(a_t) changes over time (as the challenge asks) but never enters S_t.
- Survey sampler: draw Y ~ N(m_t,σ²), R ~ Bern(c_t Φ(βY)), keep Y with R=1 until n kept (this IS "conditioning on R=1").
- Closed forms, a_t := β m_t/√(1+β²σ²):  E_{S_t}[Y] = m_t + βσ² φ(a_t)/(√(1+β²σ²) Φ(a_t));
  θ*(y) = log Φ(a_m) − log Φ(βy);  α*(t) = log Φ(a_t) − log Φ(a_m).
  (E[Φ(βY)] = Φ(a_t) since W−βY ~ N(−βm_t, 1+β²σ²); E[YΦ(βY)] by Stein's lemma E[(Y−m)h(Y)] = σ²E[h'(Y)].)
- (R3) Weight-moment rule. w = e^{θ*} has E_{S_t}[w^k] < ∞ iff k < a := 1 + 1/(β²σ²).
  Require βσ < 1/√2 (a > 3: finite third moment, so both the CLT-based SE and the O(1/n) bias
  expansion are valid). βσ ∈ [1/√2, 1) is usable but the bias constant is not defined; βσ ≥ 1 has
  infinite weight variance and is a stress test only, never a default. Any change to β, σ or the
  shape of π_t must restate a and check this rule.
- Survey bias with these numbers shrinks from ≈0.45 (t=1) to ≈0.17 (t=9): visible, and time-varying so the
  last-known-offset baseline fails by design.
- Alternatives the user may pick instead (each is 1–3 lines to swap): exponential tilt π ∝ e^{βy} on bounded Y
  (constant bias — weaker test); logistic π_t = σ(a_t+by) (violates Assumption 1 mildly — a stress test, §E);
  two-subgroup mixture with drifting weight.

## E. Stress tests (Phase 4 defaults; confirm at CP4)
1. Assumption 1 broken: π_t(y) = σ(a_t + b_t y), b_t = 1.5 + 0.1(t−m) for t>m. Expect μ̃ bias growing with |b_t−b_m|.
2. Support shift: m_t jumps by +2σ for t>m. Report the fraction of t>m survey values outside the training range.
3. Small n: n ∈ {200, 500, 2000}. Report sd of μ̃ over 10 seeds; expect ∝ 1/√n_eff.

## F. Evidence from the numpy twin (reproduce before trusting; re-run 2026-09-18 at the corrected β=0.6)
- Command: `python tools/twin_default_run.py`. Config: `{m:6, M:9, n:2000, sigma:1.0, beta:0.6,
  H:32, lr:0.003, wd:1e-05, epochs:400, batch:256, val_frac:0.2, seed_data:0, seed_model:1}`.
  Git commit at run time: `b4a438d` (the commit that includes this file's own edit is later;
  see STEP 5 of notes/decisions.md's "Authorized correction" entry). Full stdout:
  ```
  best val 0.4455 at epoch 394
  FOC   : {1: 1.013, 2: 0.99, 3: 0.988, 4: 0.99, 5: 0.99, 6: 0.979}
  alpha~: [-0.375 -0.299 -0.186 -0.102 -0.034  0.   ]
  alpha*: [-0.439 -0.332 -0.234 -0.147 -0.069  0.   ]
  grid y            : [-3. -2. -1.  0.  1.  2.  3.  4.]
  theta~-theta*     : [ 5.71   0.075 -0.078 -0.058  0.028 -0.052 -0.047  0.17 ]
  centered (- mean over y in [-2,3]): [ 5.732  0.097 -0.056 -0.036  0.05  -0.03  -0.025  0.192]
   t=7 mu=1.250 survey=1.505 mu~=1.313 oracle=1.298 neff/n=0.93  frac outside training range [-3.87,4.98]=0.0000
   t=8 mu=1.500 survey=1.689 mu~=1.534 oracle=1.518 neff/n=0.96  frac outside training range [-3.87,4.98]=0.0000
   t=9 mu=1.750 survey=1.943 mu~=1.779 oracle=1.751 neff/n=0.95  frac outside training range [-3.87,4.98]=0.0000
  ```
- Grad-check (from `python fusion_numpy.py`, its own CFG: lr=1e-2, wd=1e-4, epochs=150, beta=0.6):
  worst relative error 9.03e-9.

## G. Weight-moment theorem (R3) — why β must satisfy βσ < 1/√2

**Theorem (weight moments).** For k ≥ 1,
E_{S_t}[w^k] < ∞  ⟺  (k−1)β²σ² < 1,  i.e.  k < 1 + 1/(β²σ²) =: a,
(boundary case aside), where a is the tail index: P_{S_t}(w > x) ~ x^{−a}.

**Proof sketch.** By change of measure, E_{S_t}[w^k] = const · E_{P_{t,Y}}[ρ_t^{k−1}]
= const · ∫ φ_{m_t,σ}(y) Φ(βy)^{−(k−1)} dy. By Mills' ratio (Gordon's inequality),
Φ(βy)^{−1} = e^{β²y²/2} · (polynomial factor) as y → −∞. The exponent of the integrand is
y²((k−1)β²/2 − 1/(2σ²)) + O(y), integrable at −∞ iff (k−1)β²σ² < 1. ∎

**Consequences.**
- k=1 (mean of w): always finite → μ̃ is consistent for any β.
- k=2 (variance): finite iff βσ < 1. This is the CLT hypothesis behind the delta-method standard
  error; if it fails, the reported SE is not a standard error of anything.
- k=3 (third moment): finite iff βσ < 1/√2 ≈ 0.7071. This is the hypothesis behind the O(1/n)
  bias expansion of the Hájek (self-normalized) estimator; if it fails, the bias decays slower
  than 1/n and n_eff jumps erratically between samples.

The old default β=1.5, σ=1 gave a = 1.444: neither the variance nor the third moment exists.
That was the defect (see notes/decisions.md, 2026-09-18). New design rule (R3): require
βσ < 1/√2; default β=0.6 (a = 3.78). βσ ≥ 1 is a legitimate stress test, never a default.

**Diagnostic signature of a violated (R3):** n_eff/n swings by an order of magnitude between
data seeds; the delta-method SE and the Monte-Carlo sd of μ̃ disagree by more than ~30%; n·(bias)
grows with n instead of settling; sd·√n grows with n instead of being flat. The FOC diagnostic
and α̃(t) are sample means of w, so under infinite variance they converge at the stable-law rate
n^{−(1−1/a)} rather than n^{−1/2} and look noisy even when the code is correct.

Evidence: `tools/check_weight_tails.py` (LOG.md "beta correction" entry has the full stdout).

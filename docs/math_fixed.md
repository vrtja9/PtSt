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
1. Assumption 1 broken: π_t(y) = σ(a_t + b_t y), b_t = 1.0 + 0.1(t−m) for t>m. Expect μ̃ bias growing with |b_t−b_m|.
   (2026-09-18, authorized correction: the baseline slope is chosen so the LOGISTIC selection at
   t=m matches the STRENGTH of the default probit DGP at t=m, not its old β=1.5 literal value —
   Φ(x) ≈ σ(1.702x), so β=0.6 (the current default) corresponds to a logistic slope of
   1.702×0.6≈1.02≈1.0. Using the old literal β=1.5 as the logistic baseline would have perturbed
   both the functional FORM (probit→logistic) and the STRENGTH at once, confounding the reading
   of the stress test's own bias trend.)

   **Caveat — this stress test does NOT exercise (R3).** Under logistic selection, the weight
   behaves like e^{−a−by} as y→−∞ (LINEAR in the exponent), and against a Gaussian density
   e^{−y²/2σ²} every moment of w is finite for ANY b (the Gaussian's quadratic decay always wins
   against the logistic's linear one). Probit selection is different precisely because
   −logΦ(βy) ~ β²y²/2 is QUADRATIC and competes with the Gaussian on equal terms — that
   competition is what produces §G's power-law weight tail. So stress test 1 tests a violation
   of Assumption 1 ONLY; it must never be cited as covering the (R3) heavy-tail failure mode
   (see stress test 4 for that).
2. Support shift: m_t jumps by +2σ for t>m. Report the fraction of t>m survey values outside the training range.
3. Small n: n ∈ {200, 500, 2000}. Report sd of μ̃ over 10 seeds; expect ∝ 1/√n_eff.
4. (2026-09-18, authorized addition) Violate (R3) on purpose: PROBIT selection (the default
   family, not stress test 1's logistic) with βσ ≥ 1 — `Config(beta=1.5, allow_heavy_tails=True)`
   (the `allow_heavy_tails` opt-in exists exactly for this). Report n_eff/n CV over 10 seeds, the
   delta-SE-to-Monte-Carlo-sd ratio, and the fitted decay exponent of the oracle bias; check all
   three against §G's diagnostic signature (CV≥0.20-ish, ratio far from 1, exponent below the
   finite-moment n^{-1} rate). This is the ONLY stress test that exercises (R3); stress test 1
   does not (see its caveat above).

## F. Evidence from the numpy twin (reproduce before trusting; re-run 2026-09-18 at the corrected β=0.6)
- Command: `python tools/twin_default_run.py`. Config: `{m:6, M:9, n:2000, sigma:1.0, beta:0.6,
  H:32, lr:0.003, wd:1e-05, epochs:400, batch:256, val_frac:0.2, seed_data:0, seed_model:1}`.
  Git commit: `a350bbd` (verified: `git show a350bbd --stat | grep twin_default` lists
  `tools/twin_default_run.py`, so checking out this commit reproduces the config above). Full stdout:
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

## H. Residual error after (R3): θ̃ approximation error, not sampling error

With (R3) satisfied (β=0.6), n_eff/n = 0.93–0.96 (Phase 3 table, LOG.md) — the weight-VARIANCE
term of the estimator's error is healthy. The residual gap between μ̃(t) and the oracle for t>m
is instead the APPROXIMATION term of the error decomposition — θ̃ vs θ* itself — concentrated
where S_t has mass but the t≤m training data thins out.

**Verified in-repo (2026-09-18).** Interpolating the θ̃−θ* grid (`fusion.train.theta_grid_data`)
onto S_9 draws and using it in place of θ̃ reproduces the actual μ̃(9)−oracle(9) gap essentially
exactly: implied shift = `-0.0286` (paired at the same seeds against the actual trained-model gap:
also `-0.0286`, sd `0.0055` over 500 reps of n=200); at n=2000 (Phase 3's own table) the actual
single-draw gap is `-0.0273`. So the θ̃-vs-θ* drift alone fully accounts for the μ̃(9) undershoot —
there is no separate, unexplained error term. S_9's mass above y=3,4,5 is `0.1273, 0.0147, 0.0008`
— the right tail (y>3, ≈12.7% of S_9's mass) is exactly where the drift is largest (LOG.md's
θ̃−θ* grid: diff=−0.114 at y=3, growing to −0.652 at y=7).

**Three structural facts the architecture (a generic ReLU MLP) currently ignores** about
θ*(y) = log Φ(a_m) − log Φ(βy):
- Strictly DECREASING in y (verified: monotone decreasing over a 2000-point grid on [-10,10]).
- CONVEX (−log Φ is convex, Φ being log-concave).
- BOUNDED BELOW by log Φ(a_m) = `-0.3616` (verified) — θ* never goes below this as y→+∞.

None of these are imposed on θ̃'s functional form; an unconstrained ReLU MLP can (and does)
extrapolate with the wrong slope and curvature past where training data is dense.

**`cfg.theta_bounded=20` never binds.** θ* ∈ `[-0.362, 4.232]` over the training range
`[-3.868, 4.982]` (verified: θ*(-3.868)=`4.229`, θ*(0)=`0.332`, θ*(4.982)=`-0.360`), while
`B·tanh(·/B)` at B=20 only starts to compress once |θ| approaches 20 — it would need B≈5 to
meaningfully constrain anything in this range. Recorded here, not changed in this pass, so any
slide claiming the CP3 bounded-head fix addresses the right-tail drift would be dishonest — it
doesn't; it only ever addressed the left-tail blowup at the old β=1.5 default (docs/math_fixed.md
§F's history, notes/decisions.md CP3).

**Consequence: why T7 (parametric recovery) fails under a violated (R3).** T7 fits
θ(y)=a·(−logΦ(βy))+b by full-batch L-BFGS and asserts |â−1|<0.02. Its survey-side term is
(1/n)Σᵢ e^{a·θ*(yᵢ)+...}, which at the truth a=1 is the sample mean of w=e^{θ*(Y)}; the
SAMPLING VARIANCE of that sample mean is Var_S[w] = E_S[w²]−E_S[w]², finite iff βσ<1 (tail
index a>2 — the k=2 case above). At β=1.5, a=1.444≤2: this variance is infinite (§A's k=2
column diverges as the integration range grows for β=1.5, converges for β=0.6/0.8). A sample
mean of a heavy-tailed, right-skewed positive variable is typically BELOW its true expectation
in any one finite sample (the rare huge draws that would pull it up are usually absent), so the
survey term's exponential penalty on a moving away from 1 is under-felt, and L-BFGS settles at
an â slightly ABOVE 1 for a typical sample. This is exactly what CP2 observed (â=1.0255 at
β=1.5, notes/decisions.md) — a SYMPTOM of this same (R3) violation, not an independent
training/optimisation bug — and it is why T7 passes at β=0.6 (`python -m pytest`: 0 failed,
LOG.md "beta correction") without any change to T7's own code: the mechanism, not the fit,
changed.
Reference numbers from a separate sandbox on this same fusion_numpy.py (n=30000/cell, 3 seeds,
BFGS; **NOT evidence for this repo** — reproduce with a tools/ script if wanted in-repo):
β=0.6: â=0.9927,0.9824,1.0006 (mean 0.9919, sd 0.0074), max|α̂−α*| 0.006–0.009, 3/3 pass;
β=1.5: â=1.0310,1.0106,1.0299 (mean 1.0238, sd 0.0094), max|α̂−α*| 0.017–0.082, 2/3 FAIL —
consistently biased upward at β=1.5, matching the mechanism above.

**Measured decay rate (this repo's own evidence).** From `check_weight_tails.py` §B at β=1.5
(a=1.444, so the k=2 column above already shows infinite weight variance): bias(n) := (n·bias)/n
= 13.98/80=0.1748, 30.86/320=0.0964, 83.27/1280=0.0651 at n=80,320,1280. Each ×4 increase in n
shrinks the bias by a factor 0.1748/0.0964=1.812 then 0.0964/0.0651=1.482, giving empirical
exponents log₄(1.812)=0.429 and log₄(1.482)=0.284 (bias ∝ n^{-0.429} then n^{-0.284}). §G's
stable-law prediction is n^{-(1-1/a)} = n^{-(1-1/1.444)} = n^{-0.308} — the two empirical
exponents bracket this prediction, consistent with it given the run's own Monte-Carlo noise
(MC SE on n·bias was 0.83, 2.87, 8.50 at n=80,320,1280 respectively — non-trivial next to the
signal, especially at n=1280). Finite third moment (a>3, e.g. β=0.6) would instead give the
ordinary n^{-1} rate.

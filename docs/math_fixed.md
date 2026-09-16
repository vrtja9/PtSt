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

## D. Default DGP (the only open choice — confirm at CP1)
- m=6, M=9, n=2000, σ=1, β=1.5;  m_t = −0.5 + 0.25 t;  c_t = 0.9 − 0.06 t.
- P_{t,Y} = N(m_t, σ²);  π_t(y) = c_t Φ(βy)  (Φ standard normal cdf) — Assumption 1 holds because Φ(βy)/Φ(βy') has no t;
  ρ̄_t = c_t Φ(a_t) changes over time (as the challenge asks) but never enters S_t.
- Survey sampler: draw Y ~ N(m_t,σ²), R ~ Bern(c_t Φ(βY)), keep Y with R=1 until n kept (this IS "conditioning on R=1").
- Closed forms, a_t := β m_t/√(1+β²σ²):  E_{S_t}[Y] = m_t + βσ² φ(a_t)/(√(1+β²σ²) Φ(a_t));
  θ*(y) = log Φ(a_m) − log Φ(βy);  α*(t) = log Φ(a_t) − log Φ(a_m).
  (E[Φ(βY)] = Φ(a_t) since W−βY ~ N(−βm_t, 1+β²σ²); E[YΦ(βY)] by Stein's lemma E[(Y−m)h(Y)] = σ²E[h'(Y)].)
- Survey bias with these numbers shrinks from ≈0.78 (t=1) to ≈0.12 (t=9): visible, and time-varying so the
  last-known-offset baseline fails by design.
- Alternatives the user may pick instead (each is 1–3 lines to swap): exponential tilt π ∝ e^{βy} on bounded Y
  (constant bias — weaker test); logistic π_t = σ(a_t+by) (violates Assumption 1 mildly — a stress test, §E);
  two-subgroup mixture with drifting weight.

## E. Stress tests (Phase 4 defaults; confirm at CP4)
1. Assumption 1 broken: π_t(y) = σ(a_t + b_t y), b_t = 1.5 + 0.1(t−m) for t>m. Expect μ̃ bias growing with |b_t−b_m|.
2. Support shift: m_t jumps by +2σ for t>m. Report the fraction of t>m survey values outside the training range.
3. Small n: n ∈ {200, 500, 2000}. Report sd of μ̃ over 10 seeds; expect ∝ 1/√n_eff.

## F. Evidence from the numpy twin (already run; reproduce before trusting)
- grad-check worst relative error 1.2e-8 (float64, central differences).
- lr 3e-3, wd 1e-5, 400 epochs: FOC 0.98–1.03; μ̃(7,8,9) = 1.307, 1.507, 1.777 vs μ = 1.25, 1.5, 1.75;
  survey means 1.50, 1.66, 1.90; α̃ still off α* by ≤0.08 in early years (low-y tail under-fit) — FOC≈1 is
  necessary, not sufficient; always plot θ̃ vs θ*.

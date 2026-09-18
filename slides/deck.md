# Data Fusion Interview Challenge -- slides (Keynote-paste fallback)

## The data-generating process

- Sample space: (Y x {0,1}, B(Y) (x) 2^{0,1}); pi_t(y) := P[R=1|Y=y] = c_t * Phi(beta*y); P_t,Y = N(m_t, sigma^2)
- S_t = Law(Y|R=1), density s_t = pi_t*p_t / rho_bar_t
- m=6, M=9, n=2000, sigma=1.0, beta=0.6, m_t=-0.5+0.25t, c_t=0.9-0.06t
- Assumption 1 holds exactly: Phi(beta*y)/Phi(beta*y') has no t. rho_bar_t = c_t*Phi(a_t) varies with t but cancels out of S_t
- Sampler: Y~N(m_t,sigma^2), R~Bern(c_t*Phi(beta*Y)), keep Y with R=1 -- rejection sampling IS conditioning on R=1
- Definition: a_t := beta*m_t / sqrt(1+beta^2 sigma^2); at t=m, a_m=0.5145 (live)
- theta*(y) = log Phi(a_m) - log Phi(beta*y); alpha*(t) = log Phi(a_t) - log Phi(a_m)
- Closed form: E_St[Y] = m_t + beta*sigma^2*phi(a_t) / (sqrt(1+beta^2 sigma^2)*Phi(a_t)); measured survey bias E_St[Y]-m_t = +0.4535 at t=1, +0.1677 at t=9
- Design rule (R3): beta*sigma < 1/sqrt(2) (=0.7071); tail index kappa = 1+1/(beta^2 sigma^2) = 3.78 (R3 holds: True)

_Survey bias E_St[Y]-mu(t) falls from +0.454 at t=1 to +0.168 at t=9 -- a constant-bias baseline must fail_

![](../figures/phase1a_densities_96169d0c.png)

## Solving the optimization problem


### Objective
- F: T~Unif[m], Z~Bern(½), Y|T=t,Z=1 ~ P_t,Y, Y|T=t,Z=0 ~ S_t. Rows (t,z,y): 2mn = 24 000
- L = mean[(1−z)·e^r − z·r], r = θ(y)+α[t], α ∈ 𝒜 = {α ∈ ℝ^m : α(m)=0}
- E_F[L] = (1/2m)·Σ_t J_t(θ+α(t)), J_t(r) = E_St[e^r] − E_Pt,Y[r] = E_St[e^r − ρ_t·r]
- φ(u) = e^u − ρu, φ″ = e^u > 0 ⇒ unique min at r = log ρ_t = θ*(y)+α*(t)
- min J_t = 1 − KL(P_t,Y‖S_t) (NWJ); excess = E_Pt,Y[e^δ −1− δ] ≥ 0, δ = r − log ρ_t
- ∂/∂α(t) = 0 ⇒ E_St[e^{θ+α(t)}] = 1 ⇒ α(t) = −log Z_t, Z_t = E_St[e^θ] (log-partition)
- (θ+c, α−c) leaves r fixed ⇒ α(m)=0 pins the flat direction

### Model and training
- θ: MLP 1→32→1, ReLU, 3H+1 = 97 params; α: free ℝ^5 ⊕ {0}; float64
- Adam lr 1e-3, wd 1e-5 (net only), batch 256, stratified 80/20 per (t,z), early stop on val risk
- Regularisation mandatory: θ_k = +k on z=1, −k on z=0 (continuous PL, representable) ⇒ R̂_n = ½(e^{−k}−k) → −∞. Measured k=0,1,2,4,8: **0.5000, -0.3161, -0.9323, -1.9908, -3.9998**
- Best val **0.4481 @ epoch 140/400** (θ≡0 ⇒ 0.5)

### Tests 14/14, 66.10 s
- DGP — T1 closed forms vs MC; T2 kept-sample law vs quadrature; T3 w(y,y') constant in t
- Loss/gradients — T4 torch vs hand-derived numpy backward: loss 5.6e-17, grad 6.9e-17; T5 autograd gradcheck (float64); T6 joint convexity in (theta,a); T7 parametric recovery a_hat = 1.0063; T8 alpha[m] = 0
- Estimator — T9 oracle mu~ vs mu within 3 SE; T10 trained mu~ vs oracle
- Weights — T11a kappa = 3.78 > 3; T11b n_eff CV = 0.048 < 0.20; T11c SE/sd = 0.99 in [0.75,1.33]; T11d n*bias -> 1.084

### Diagnostics before any μ̃
- FOC mean_i e^{θ̃+α̃(t)} = 1.037, 1.012, 1.003, 0.998, 0.988, 0.969 (necessary, not sufficient)
- θ̃−θ* grid, mean-centred; n_eff = (Σw)²/Σw²

![](../figures/phase3_val_curve_96169d0c.png)

## Results and learnings

- μ̃(t) = Σᵢwᵢyᵢ / Σᵢwᵢ, w = e^{θ̃(y)}, yᵢ ~ S_t, t > m
-  t    μ(t)  survey       μ̃(t)±SE  offset  oracle  n_eff/n bias cut
 7   1.250   1.469   1.265±0.022   1.181   1.275     0.94      93%
 8   1.500   1.693   1.484±0.024   1.405   1.502     0.94      91%
 9   1.750   1.881   1.684±0.026   1.593   1.709     0.95      49%
- **L1 — α cancels.** μ(t) = e^{α*(t)}E_St[e^{θ*}Y] and 1 = e^{α*(t)}E_St[e^{θ*}] ⇒ μ(t) = E_St[e^{θ*}Y]/E_St[e^{θ*}]. Paired years give θ's *shape*; the current survey's own Z_t gives the level. t > m needs no P_t.
- **L2 — κ decides validity.** w = Φ(a_m)/Φ(βy); 1/Φ(βy) ~ e^{β²y²/2} (Mills) ⇒ E_St[w^k] < ∞ ⇔ k < κ = 1+1/(β²σ²). k=2 ⇔ βσ<1 (CLT/SE); k=3 ⇔ βσ<1/√2 (O(1/n) bias) ⇒ (R3).
-                                 β=1.5, κ=1.44          β=0.6, κ=3.78
  n_eff/n CV, 10 seeds                  0.790                  0.048
         delta-SE ÷ sd                   2.49                   0.99
            bias decay n^-0.36 (pred n^-0.31)         n·bias → 1.084
                  T7 â               1.0255 ✗               1.0063 ✓
- One cause, four symptoms.
- **L3 — residual is approximation, not sampling.** μ̃−μ = (oracle−μ) + (μ̃−oracle):
-  t      μ̃−μ  sampling     model     SE
 7   +0.0153   +0.0250   -0.0097  +0.7
 8   -0.0165   +0.0019   -0.0184  -0.7
 9   -0.0662   -0.0409   -0.0253  -2.6
Model term monotone; sampling term swings sign.
- Ablation δ(y) = θ̃−θ*, held flat past c, paired on S_9 draws:
     c  mean shift    contributes
  full     -0.0286              —
     3     -0.0178       y>3: 38%
     2     -0.0023       y>2: 92%
⇒ band 2<y≤3 = 54%
- Survey mass share, pooled t≤m vs S_9:
  y>  pooled t≤m      S_9   ratio
   2      10.0%    46.4%    4.7x
   3       1.2%    12.7%   10.3x
   4       0.1%     1.5%   22.8x
**Temporal covariate shift**, not extrapolation; excess risk is weighted by the *training* measure.
- θ* strictly decreasing, convex, ≥ log Φ(a_m) = -0.362; B=20 never binds (θ* ∈ [-0.36, 4.23]). Fix: reweight training years toward the forecast y-region; constrain θ̃ monotone + convex.

![](../figures/phase4_main_figure_96169d0c.png)

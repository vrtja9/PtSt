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

- Pooled F over rows (t,z,y), 2mn=24000 rows; L=mean((1-z)*exp(r)-z*r), r=theta(y)+alpha[t]
- Why this loss: its minimizer over separable r is log dP_t,Y/dS_t (the NWJ variational form of KL); alpha(t)=-log E_St[e^theta] is a per-year log-partition constant; the anchor alpha(m)=0 removes the flat direction (theta+c, alpha-c)
- Implementation: theta=MLP(1->H->1,ReLU,H=32); alpha=free vector in R^(m-1) concat with a constant 0; float64; Adam lr=0.001, wd=1e-05 (network only), batch=256; stratified 80/20 split; early stopping on validation risk
- Early stopping is REQUIRED, not cosmetic: the piecewise-linear theta_k (+k at population rows, -k at survey rows) gives R_hat_n(k)=(1/2)(e^-k-k) -> -inf as k->inf. Measured [k=0:+0.5000, k=1:-0.3161, k=2:-0.9323, k=4:-1.9908, k=8:-3.9998] vs the trained best val loss 0.4481 at epoch 140 of 400 run -- the best checkpoint is NOT the last epoch
- Verification: 14 passed in 41.68s. T4 torch-vs-numpy: loss diff 5.6e-17, worst grad diff 6.9e-17; T7 parametric recovery a_hat=1.0063; T11 weight tails kappa=3.78, n_eff CV=0.048, SE ratio=0.99
- Diagnostics before any mu~ is reported: FOC mean_i e^(theta~+alpha~(t)) per year [t=1:1.037, t=2:1.012, t=3:1.003, t=4:0.998, t=5:0.988, t=6:0.969]; theta~ vs theta* grid; n_eff

![](../figures/phase3_val_curve_96169d0c.png)

## Results and learnings

-  t   mu(t)  survey     mu~(t)+-SE  offset  oracle  n_eff/n bias cut
 7   1.250   1.469   1.265+-0.022   1.181   1.275     0.94      93%
 8   1.500   1.693   1.484+-0.024   1.405   1.502     0.94      91%
 9   1.750   1.881   1.684+-0.026   1.593   1.709     0.95      49%
- Learning 1: alpha(t) is the unknown per-year normalising constant and it CANCELS in mu~=E_St[e^theta Y]/E_St[e^theta] -- lagged population data teaches theta's SHAPE only, never the current-year level, which is exactly what makes t>m possible
- Learning 2: the weight tail index decides whether ANY of the inference is valid. E_St[w^k]<inf iff k<kappa=1+1/(beta^2 sigma^2). Old beta=1.5: kappa=1.44, n_eff/n CV=0.790 across seeds, delta-SE 2.49x the true sampling sd, bias decay ~n^-0.36 (predicted n^-0.31) instead of n^-1, T7 a_hat=1.0255 (fails |a-1|<0.02). beta=0.6: kappa=3.78, CV=0.048, SE ratio=0.99, T7 a_hat=1.0063 (passes). One defect, not four
- Learning 3 (headline): mu~ cuts the survey's bias vs mu(t) by t=7:93%, t=8:91%, t=9:49% -- decomposition mu~-mu=(oracle-mu)+(mu~-oracle): t=7: +0.0153=+0.0250+-0.0097 (+0.7 SE); t=8: -0.0165=+0.0019+-0.0184 (-0.7 SE); t=9: -0.0662=-0.0409+-0.0253 (-2.6 SE); the model-drift term is present at every forecast year and grows monotonically (-0.0097, -0.0184, -0.0253); the sampling term is what swings in sign and size (+0.0250, +0.0019, -0.0409). Only at t=9 do the two align and push mu~-mu past 2 SE
- Learning 3 (mechanism, not circular): hold theta~'s drift delta(y):=theta~(y)-theta*(y) FLAT beyond a cutoff c (delta(min(y,c))), paired on the same S_9 draws -- an ablation of the trained net's own output, not a re-interpolation of it. Full drift shift -0.0286 (sd 0.0018); flat beyond y=3 -> -0.0178 (y>3 contributes 38%); flat beyond y=2 -> -0.0023 (y>2 contributes 92%) -- at t=9, -0.0253 of the -0.0662 total gap (38%) is this model-drift term, the rest is sampling (-0.0286 is the 200-rep ablation mean; -0.0253 is this sample's realisation, 1.8 sd apart)
- Learning 3 (diagnosis): the far tail y>3 is 12.7% of S_9's mass but contributes only 38% of the shift; the 2<y<=3 band (33.7% of mass) dominates -- that band is data-RICH at t=9 but data-POOR in training: survey mass share above y=2/3/4, pooled t<=m vs S_9: 10.0% vs 46.4% (4.7x), 1.2% vs 12.7% (10.3x), 0.1% vs 1.5% (22.8x) -- TEMPORAL COVARIATE SHIFT, not extrapolation. mu~-oracle grows monotonically with t (-0.0097, -0.0184, -0.0253) -- a structural property of this data-fusion setting, not an implementation bug
- theta* is strictly decreasing, convex, bounded below by log Phi(a_m)=-0.362; the bounded head B=20 never binds (theta* in [-0.36, 4.23] over the training range) -- identified, not yet applied. Sharper fix from the covariate-shift diagnosis: weight/augment training years toward the forecast years' y-region, or constrain theta~ to be monotone+convex (matching theta*'s known shape) instead of an unconstrained MLP

![](../figures/phase4_main_figure_96169d0c.png)

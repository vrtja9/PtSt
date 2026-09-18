# notes/talk_through.md -- pipeline walkthrough: math step -> code -> test

Each bullet maps one step of the pipeline to (i) the math step in docs/math_fixed.md, (ii) the
code file, (iii) the test that verifies it.

1. **Bayes gives s_t.** Step 1 (§B): `s_t(y) = π_t(y) p_t(y) / ρ̄_t`. Code:
   `fusion/dgp.density_grid`. Test: T2 (kept-sample law matches numeric integration of
   `s_t ∝ Φ(βy)·N(m_t,σ²)`).
2. **The density ratio.** Step 2 (§B): `ρ_t = p_t/s_t = ρ̄_t/π_t(y)`. Code: implicit in the
   derivation behind `fusion/dgp.theta_star`/`alpha_star` (no standalone function — §C never
   tests it in isolation). Test: T1 (closed forms — the same DGP whose ratio this is).
3. **Assumption 1 as a factorization.** Step 3 (§B): Assumption 1 ⇔ `π_t(y)=c_t·g(y)`. Code:
   `fusion/dgp.pi_t`. Test: T3 (`π_t(y,t)/π_t(y',t)` independent of t to 1e-12).
4. **Why log ρ_t separates as θ*(y)+α*(t).** Step 3's anchored decomposition (`θ*:=log ρ_m`, so
   `α*(m)=0`). Code: `fusion/dgp.theta_star`, `alpha_star`. Test: T7 (parametric recovery of
   `θ(y)=a·(−logΦ(βy))+b` — recovering `a≈1` is recovering this exact functional form).
5. **The change of measure giving μ̃.** Step 4 (§B): `μ(t)=E_St[ρ_t Y]=E_St[e^θ* Y]/E_St[e^θ*]`,
   invariant to `θ→θ+c`. Code: `fusion/estimate.mu_tilde`, `oracle_estimate`. Test: T9 (oracle
   estimator matches the truth within `3·SE`).
6. **The pooled objective F.** Step 5 (§B): `R(θ,α)=E_F[L]=(1/2m)Σ_t J_t(θ+α(t))`. Code:
   `fusion/loss.fusion_loss` applied to `fusion/data.build_dataset`'s pooled rows. Test: T6
   (indirectly — the convexity it checks is a property this pooled objective relies on).
7. **Why the loss's minimizer is the log ratio.** Step 6 (§B): `e^r−ρr` is strictly convex in r,
   minimized at `r=log ρ`; this is what makes the whole scheme identifiable. Code: no standalone
   function (a mathematical fact underlying `fusion/loss.py`'s form). Test: T6 (the midpoint
   inequality directly checks strict convexity for 1000 random pairs).
8. **The FOC that defines α.** Step 8 (§B): `∂R/∂α(t) ∝ E_St[e^{θ+α(t)}]−1=0`. Code:
   `fusion/train.foc_diagnostic`. Test: T10 (FOC within ±0.05 for t≤m) and T8 (the anchor
   `α(m)=0` holds exactly, structurally, after 100 optimizer steps).
9. **The empirical risk is unbounded below.** Step 9 (§B): an interpolating θ drives
   `R̂_n → −∞`, which is why early stopping is required, not cosmetic. Code:
   `fusion/train.train` (best-checkpoint early stopping), `fusion/loss.dL_dr` (the per-row
   derivative T4 checks bit-for-bit). Test: T4 (numpy-twin agreement on the gradient this
   unbounded loss produces); no test asserts unboundedness directly — slide 2 instead quotes the
   measured evidence (best val loss at epoch 140 of a 400-epoch run, i.e. the best checkpoint is
   *not* the last epoch).
10. **The Hájek estimator's SE, bias and n_eff.** docs/math_fixed.md §C's three added identities
    (delta-method SE, finite-sample bias, `1≤n_eff≤n`). Code: `fusion/estimate.mu_tilde` (n_eff),
    `bootstrap_se`. Test: T11(b) (n_eff/n stability), T11(c) (SE calibration), T11(d)
    (finite-sample bias vs the numerically-integrated prediction).
11. **Rule (R3).** §G's theorem: `E_St[w^k]<∞ ⟺ k<κ:=1+1/(β²σ²)`. Code:
    `fusion/config.Config.__post_init__` (raises unless `βσ<1/√2` or `allow_heavy_tails=True`).
    Test: T11(a) (asserts `βσ<1/√2`, quoting `κ`).
12. **The residual error after (R3).** §H: with (R3) satisfied, n_eff/n is healthy (0.93–0.96 —
    slide 3's table), so part of the remaining `μ̃(t>m)` gap is sampling noise and part is θ̃'s
    approximation error. Decomposing `μ̃(t)−μ(t) = (oracle(t)−μ(t)) + (μ̃(t)−oracle(t))` shows only
    t=9 clears 2 SE, and even there the model term is 38% of the gap, not all of it. That model
    term concentrates in the 2<y≤3 band (33.7% of `S_9`'s mass), NOT the far tail y>3 (12.7% of
    mass, only 38% of the shift) — verified by a counterfactual ablation that holds θ̃'s drift
    flat beyond a cutoff c, not by re-interpolating θ̃'s own output onto itself (that check was
    circular and has been removed, §H). That band is data-rich at t=9 but data-poor during
    training (survey mass share above y=2, pooled t≤m vs `S_9`: 10.0% vs 46.4%, a 4.7× gap) — a
    TEMPORAL COVARIATE SHIFT, not tail extrapolation. Code: `fusion/build_deck.py`'s
    counterfactual-repair and covariate-shift blocks. Test: none dedicated — an observational
    finding from the θ̃ vs θ* figure and the live measurements above, not a pass/fail assertion.

## Three questions to expect, and the one-sentence answer

- **"Why does the estimator need lagged population data at all, if α cancels in μ̃?"** Because
  θ's SHAPE (not the level) can't be identified from survey data `S_t` alone — only the paired
  `(P_t, S_t)` data for `t≤m` pins θ down, via the pooled loss's unique minimizer (Steps 6–7).
- **"Why β=0.6 and not some other value?"** Because (R3) requires `βσ<1/√2` for the weight's
  third moment — hence the Hájek estimator's `O(1/n)` bias expansion and the delta-method SE —
  to be finite; β=0.6 gives tail index `κ=3.78`, safely above 3, while the original β=1.5 gave
  `κ=1.44` (and, as a byproduct, explained T7's earlier CP2 failure).
- **"Is this finished, or is there a known issue?"** The sampling-variance side is healthy
  (`n_eff/n≈0.93–0.96`), but there is one known, quantified residual: θ̃'s approximation error vs
  θ* in the 2<y≤3 band, which explains part — not all, see the next question — of the small
  `μ̃(9)` undershoot; a targeted fix (a monotone/convex constraint on θ̃, or reweighting training
  years toward the forecast years' y-region) is identified and quantified, not yet applied.
- **"Why does μ̃ undershoot at t=9?"** Two parts, not one. (i) Sampling: `μ̃(9)−μ(9)=-0.0662` is
  `2.6` SE from zero — at t=7,8 the same kind of gap is only `0.7` SE, indistinguishable from
  sampling noise, so t=9's gap is the one that is actually real. (ii) Model: of that `-0.0662`,
  only `-0.0253` (38%) is θ̃'s approximation error vs θ*, and a counterfactual ablation (holding
  the drift flat beyond a cutoff, not re-interpolating θ̃ onto itself) shows that error
  concentrates in the 2<y≤3 band — a region `S_9` visits far more (33.7% of its mass) than the
  t≤m training years do (their combined mass above y=2 is 10.0% vs `S_9`'s 46.4%, a 4.7× gap):
  temporal covariate shift, not tail extrapolation. The other 62% (`-0.0409`) is plain sampling
  variability in a single survey draw.

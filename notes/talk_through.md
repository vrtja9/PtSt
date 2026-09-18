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
11. **Rule (R3).** §G's theorem: `E_St[w^k]<∞ ⟺ k<a:=1+1/(β²σ²)`. Code:
    `fusion/config.Config.__post_init__` (raises unless `βσ<1/√2` or `allow_heavy_tails=True`).
    Test: T11(a) (asserts `βσ<1/√2`, quoting `a`).
12. **The residual right-tail error.** §H: with (R3) satisfied, n_eff/n is healthy (0.93–0.96 —
    slide 3's table), so the remaining `μ̃(t>m)` gap is θ̃'s approximation error, concentrated
    where `S_t` has mass (`y>3`, ~12.6% of `S_9`) but t≤m training data thins out. Code:
    `fusion/train.theta_grid_data`. Test: none dedicated — this is an observational finding from
    opening the θ̃ vs θ* figure and re-deriving it numerically (`fusion/build_deck.py`'s
    interpolated-drift check, C4(a)), not a pass/fail assertion.

## Three questions to expect, and the one-sentence answer

- **"Why does the estimator need lagged population data at all, if α cancels in μ̃?"** Because
  θ's SHAPE (not the level) can't be identified from survey data `S_t` alone — only the paired
  `(P_t, S_t)` data for `t≤m` pins θ down, via the pooled loss's unique minimizer (Steps 6–7).
- **"Why β=0.6 and not some other value?"** Because (R3) requires `βσ<1/√2` for the weight's
  third moment — hence the Hájek estimator's `O(1/n)` bias expansion and the delta-method SE —
  to be finite; β=0.6 gives tail index `a=3.78`, safely above 3, while the original β=1.5 gave
  `a=1.44` (and, as a byproduct, explained T7's earlier CP2 failure).
- **"Is this finished, or is there a known issue?"** The sampling-variance side is healthy
  (`n_eff/n≈0.93–0.96`), but there is one known, quantified residual: θ̃ drifts from θ* in the
  right tail (`y>3`, ~12.6% of `S_9`'s mass), which fully explains the small `μ̃(9)` undershoot —
  a targeted fix (a monotone or genuinely-binding constraint on θ) is identified and quantified,
  not yet applied.

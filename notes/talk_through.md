# notes/talk_through.md -- 10 bullets mapping slides to code and docs/math_fixed.md steps

1. Slide 1's DGP (P_{t,Y}=N(m_t,sigma^2), pi_t(y)=c_t*Phi(beta*y)) is `fusion/dgp.py`
   (`m_t`, `c_t`, `pi_t`, `draw_pop`, `draw_survey`) and Step 10 of docs/math_fixed.md §B; the
   closed forms (`a_t`, `ES_closed`, `theta_star`, `alpha_star`) implement §D's formulas exactly.
2. Slide 1's figure is `figures/phase1a_densities_4b6ab92b.png`, built by
   `fusion/run.phase1_figures` from `fusion/dgp.density_grid` (Step 1, Bayes:
   `s_t = pi_t*p_t/rho_bar_t`).
3. Slide 2's loss `L=(1-z)e^r-z*r` is `fusion/loss.fusion_loss`, Step 9's empirical risk over the
   pooled rows `fusion/data.build_dataset` produces; `dL_dr` matches Step 9's per-row identity
   and is checked bit-for-bit against `fusion_numpy.py` in test T4.
4. Slide 2's model (`theta` = MLP 1->32->1 with a bounded head, `alpha` anchored at `alpha(6)=0`)
   is `fusion/model.ThetaNet`/`AlphaTable`; the anchor implements Step 7 (identification) and is
   checked exactly (not approximately) by test T8.
5. Slide 2's training loop and diagnostics are `fusion/train.train`/`foc_diagnostic`/
   `theta_grid_data`; the FOC check is Step 8's stationarity condition
   (`E_St[e^{theta+alpha(t)}]=1`) evaluated empirically.
6. Slide 2's CP3 story (spiky/unbounded first run -> lr=1e-3 + bounded head) is logged in
   `notes/decisions.md` "CP3" and `LOG.md` "Phase 3", with before/after figures
   `figures/phase3_val_curve_{4b6ab92b,8e4fef38}.png`.
7. Slide 3's estimator `mu~(t) = sum(e^theta*y)/sum(e^theta)` is `fusion/estimate.mu_tilde`,
   Step 4's Hajek estimator; the baselines (survey mean, offset, oracle) are
   `survey_mean_baseline`/`offset_baseline`/`oracle_estimate` in the same file.
8. Slide 3's main figure is `figures/phase4_main_figure_8e4fef38.png`, from
   `fusion/evaluate.main_figure_data` + `fusion/run.phase4_run` -- directly answers
   docs/challenge_text.md §4.3's ask (compare `mu(t)`, `E_St[Y]`, `mu~(t)`).
9. Slide 3's Learning 2 (tail blowup) and Learning 3 (Assumption-1-broken bias direction) are
   evidenced in `notes/derivations.md` "Phase 3 evidence" and "Phase 4 evidence and failure
   modes", from `fusion/evaluate.stress_test_assumption1_broken`/`draw_survey_assumption1_broken`
   (a t>m-only perturbation of Step 3's factor-form Assumption 1).
10. Every number on every slide came from a command run in this session (LOG.md has the full
    trail); test T7's known, documented failure (notes/decisions.md CP2) is a reminder that not
    every literally-specified tolerance survives contact with finite-sample Monte Carlo noise --
    reported rather than hidden, per CLAUDE.md §2.1.

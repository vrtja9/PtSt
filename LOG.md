# LOG

One block per run: config hash, command, key printed lines.

## Phase 0 (commit 6e7531a)
- `python -c "import torch,numpy,scipy,matplotlib; print(torch.__version__)"` -> `2.5.1+cu124`
- `python fusion_numpy.py` -> grad-check worst rel. error 1.173e-08; best val loss 0.1756 @ epoch
  88; FOC {1:1.105,2:1.022,3:1.053,4:1.059,5:1.04,6:1.033} (script's own CFG: lr=1e-2, wd=1e-4,
  epochs=150 -- NOT the lr=3e-3/400-epoch config docs/math_fixed.md §F cites); t=7,8,9 mu_tilde
  1.313/1.512/1.772 vs mu_true 1.25/1.5/1.75, survey_mean 1.5/1.657/1.904.

## Phase 1 (config hash 4b6ab92b, branch phase-1-dgp)
- `python -m pytest tests/test_dgp.py -v` -> T1, T2, T3 all PASSED (3 passed in 0.82s).
- `python -m fusion.run --phase 1` -> saved figures/phase1a_densities_4b6ab92b.png,
  figures/phase1b_mu_vs_survey_vs_rho_4b6ab92b.png (each with a sibling .json of the Config).
  Opened both: (a) s_t visibly right-shifted from p_t at t=1, nearly coincides with p_t by t=9,
  π_t's sigmoid shifts right and its plateau shrinks as t grows; (b) mu(t) rises linearly while
  E_St[Y] sits above it with a shrinking gap, rho_bar_t is a low non-monotone hump (~0.35-0.45).

## Phase 2 (branch phase-2-data-model-loss)
- Uncertainty raised and resolved before coding: ThetaNet architecture conflict between
  CLAUDE.md §3 ("1->H->H->1") and fusion_numpy.py's actual init_params/theta_of ("1->H->1",
  which T4 requires bit-exact). User answered "Yes" to option A (match the twin). Logged in
  notes/decisions.md.
- `python -m pytest tests/ -v` -> **1 failed, 7 passed** (5.23s):
  - test_T1/T2/T3 (Phase 1): PASSED.
  - test_T4_numpy_twin_agreement: PASSED (|r diff|<1e-8, |L diff|<1e-10, all grads <1e-8).
  - test_T5_gradcheck_tiny_net: PASSED.
  - test_T6_joint_convexity_midpoint: PASSED.
  - test_T7_parametric_recovery_lbfgs: **FAILED** -- `assert abs(a.item()-1.0) < 0.02` ->
    `0.025452701852180626 < 0.02` is False (a=1.0254527018521806). See notes/derivations.md
    "Phase 2 evidence" for the full diagnostic (converged L-BFGS solution, gradient norm ~1e-8;
    4-seed and 3-n sweep showing the error shrinks with n, consistent with MC noise not a bug).
  - test_T8_anchor_exactly_zero_after_training: PASSED.
- Diagnostic sweep (not part of the committed test suite, run interactively):
  `seed_data=0,1,2,3` at n=200k -> `|a-1| = 0.0255, 0.0567, 0.0247, 0.0002`; max`|alpha_hat-
  alpha_star| = 0.0309, 0.0430, 0.0444, 0.0380` (all 4 seeds fail the alpha criterion, 3 of 4
  fail the a criterion). `n=200k,800k,3.2M` at seed=0 -> `|a-1| = 0.0255, 0.0010, 0.0062`;
  max`|alpha_hat-alpha_star| = 0.0309, 0.0169, 0.0111` (shrinking with n).

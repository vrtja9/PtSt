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

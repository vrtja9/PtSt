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

## Phase 3 (branch phase-3-training-estimation)
- First run (lr=3e-3, unbounded, config hash 4b6ab92b): `python -m fusion.run --phase 3` ->
  best_epoch 395, FOC {1:1.079,2:1.106,3:1.0,4:1.027,5:1.045,6:1.012}. Opened figures: val curve
  spiky/still-descending at ep.395; alpha~ flatter than alpha*, crosses near t=3; theta~ blows up
  to ~380 at y=-6 vs theta*'s <50. Raised at CP3; user replied `Adopt` (lr->1e-3, bounded head B=20).
- Re-run (lr=1e-3, bounded=20, config hash 8e4fef38): best_epoch 399, best_val_loss 0.2034,
  FOC {1:1.005,2:0.958,3:0.963,4:1.012,5:1.025,6:0.974}. Val curve smooth/monotone (no spikes);
  theta~ plateaus at 20 instead of exploding.
  Estimate table (t=7,8,9): mu_true=1.25/1.5/1.75; survey_mean=1.500/1.647/1.858;
  mu_tilde=1.325/1.476/1.752 (SE 0.0338/0.0324/0.0309, n_eff/n 0.82/0.83/0.88);
  offset=1.243/1.391/1.601; oracle=1.306/1.454/1.734.
- `python -m pytest tests/ -v` -> 1 failed (T7, unchanged/documented), 9 passed: T1-T6,T8 (Phase
  1-2), T9 PASSED (oracle vs truth < 3*SE, n=20k), T10 PASSED (FOC within +-0.05 for t<=m;
  |mu_tilde(t)-oracle(t)| = 0.0191/0.0168/0.0189 for t=7,8,9, all < 0.05).

## Phase 4 (branch phase-4-evaluation-stress-tests, config hash 8e4fef38)
- `python -m fusion.run --phase 4` (30.99s):
  - in_sample_check (t<=m): mu_tilde tracks mu_true closely, e.g. t=6: 1.000 vs 1.000, t=5: 0.749
    vs 0.750.
  - stress 1 (Assumption 1 broken): bias -0.303/-0.295/-0.127 at |b_drift|=0.10/0.20/0.30 --
    magnitude SHRINKS as drift grows, opposite of docs/math_fixed.md §E.1's stated expectation
    (see notes/derivations.md "Phase 4 evidence and failure modes" for the disclosed reason:
    a_t is coupled to t in this perturbation, not held fixed).
  - stress 2 (support shift, +2sigma): frac outside training range 4.0%/6.7%/11.5% (t=7,8,9,
    7.4% pooled), bias only +0.057/+0.040/+0.055 -- small despite ~10% extrapolated mass,
    plausibly because CP3's bounded head (theta_bounded=20) keeps theta~'s extrapolation gentle.
  - stress 3 (small n, theta~ fixed, 10 seeds/n): sd(mu_tilde) = 0.0583/0.0468/0.0186 at
    n=200/500/2000 (mean n_eff 190.0/472.9/1787.2), close to the 1/sqrt(n) reference line.
  - saved figures/phase4_{main_figure,stress1_assumption1,stress2_support_shift,stress3_small_n}_8e4fef38.png,
    all opened.
- `python -m pytest tests/ -v` -> unchanged: 1 failed (T7, documented), 9 passed.

## Phase 5 (branch phase-5-deliverable)
- CP5 slide text proposed and approved (`Go`; notes/decisions.md).
- `python -m fusion.render_slides` -> saved slides/deck.pptx, slides/deck.md.
- Rendering environment note: `soffice --convert-to pdf/png` initially failed on EVERY input,
  even a plain .txt file, with "Error: source file could not be loaded" -- `dpkg -l` showed only
  `libreoffice-core`/`libreoffice-common` installed, not `libreoffice-impress` (no Impress import
  filter registered). Fixed with `apt-get install -y libreoffice-impress poppler-utils` (both
  installs succeeded after `apt-get update`, which resolved an initial 404 on a stale index).
  After the fix: `soffice --convert-to pdf` -> deck.pdf; `pdftoppm -png` -> slide-{1,2,3}.png.
- Opened all 3 rendered slide PNGs. First pass showed multi-line bullet strings (ones I'd
  manually pre-wrapped with a leading "  ") rendering as spurious extra dashed bullets instead of
  wrapped continuation text; fixed by joining each bullet into one string and letting
  `word_wrap=True` handle wrapping, then rebuilt and re-rendered -- confirmed clean on re-open:
  - Slide 1 "The data-generating process": 4 bullets (DGP, selection, S_t/m/M/n, shrinking bias)
    + figures/phase1a_densities_4b6ab92b.png (3 panels, t=1/6/9), all legible.
  - Slide 2 "Optimization pipeline": 5 bullets (loss, model, Adam settings, CP3 fix, diagnostics
    gate) + figures/phase3_val_curve_8e4fef38.png (smooth post-CP3 curve), all legible.
  - Slide 3 "Comparison and learnings": 4 bullets (headline + 3 learnings) +
    figures/phase4_main_figure_8e4fef38.png, all legible.
- notes/talk_through.md: 10 bullets written, mapping each slide to its code files and
  docs/math_fixed.md step numbers.
- Final `python -m pytest tests/ -v` -> 1 failed (T7, documented at CP2, unchanged throughout),
  9 passed (T1,T2,T3,T4,T5,T6,T8,T9,T10).

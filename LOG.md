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

## Beta correction (2026-09-18): 1.5 -> 0.6 under rule (R3), authorized
See notes/decisions.md "Authorized correction" for the full narrative and the theorem.
STOP CONDITION encountered and resolved before any edit: `python tools/check_weight_tails.py`
raised `ModuleNotFoundError: No module named 'fusion_numpy'` (script's own dir, not cwd, is
sys.path[0] for a direct script invocation) -- reported and held for the user's answer; fixed by
an in-file `sys.path` shim (their decision), same shim added to tools/twin_default_run.py.
`conftest.py` (empty) added at repo root; verified it does NOT fix bare `pytest` in this
container (that command uses a completely separate `uv tool`-installed interpreter with no
project dependencies -- `pytest --version` 9.0.2 there vs 8.3.3 under `python -m pytest`,
confirmed via the shebang `/root/.local/share/uv/tools/pytest/bin/python`); `python -m pytest`
(or bare `pytest` in an environment where the `pytest` binary and `python3` share site-packages)
remains the working invocation.

`python tools/check_weight_tails.py` full stdout:
```
== A. E_S[w^k] integrals vs integration range (divergence = infinite moment) ==
  beta=0.6  tail index a= 3.78  (a>2 finite var, a>3 finite 3rd moment)  L=10,20,40  k=2: ['3.21', '3.21', '3.21']   k=3: ['28.9', '28.9', '28.9']
  beta=0.8  tail index a= 2.56  (a>2 finite var, a>3 finite 3rd moment)  L=10,20,40  k=2: ['5.86', '5.86', '5.86']   k=3: ['7.43e+08', '3.3e+28', 'nan']
  beta=1.0  tail index a= 2.00  (a>2 finite var, a>3 finite 3rd moment)  L=10,20,40  k=2: ['307', '9.26e+03', '1.36e+06']   k=3: ['1.51e+24', '5.16e+90', 'nan']
  beta=1.5  tail index a= 1.44  (a>2 finite var, a>3 finite 3rd moment)  L=10,20,40  k=2: ['1.94e+28', '6.41e+110', '4.71e+168']   k=3: ['1.91e+78', 'nan', 'nan']
== B. predicted vs observed O(1/n) bias of mu~ at t=1 (oracle theta*) ==
  beta=0.6  a=3.78  predicted n*bias -> 1.084  [check E_S[w] = e^{-alpha*(1)}: 1.5519 vs 1.5519]
     n=   80  n*bias=   +0.48 (MC SE 0.44)   sd*sqrt(n)=1.541
     n=  320  n*bias=   +2.56 (MC SE 0.87)   sd*sqrt(n)=1.532
     n= 1280  n*bias=   -0.56 (MC SE 1.84)   sd*sqrt(n)=1.628
  beta=1.5  a=1.44  predicted n*bias -> not defined (a<=3, third moment infinite)
     n=   80  n*bias=  +13.98 (MC SE 0.83)   sd*sqrt(n)=2.945
     n=  320  n*bias=  +30.86 (MC SE 2.87)   sd*sqrt(n)=5.066
     n= 1280  n*bias=  +83.27 (MC SE 8.50)   sd*sqrt(n)=7.510
== C. delta-method SE vs Monte-Carlo sd of mu~ (t=1, n=2000, 200 reps) ==
  beta=0.6  delta SE 0.0342   MC sd 0.0347   ratio 0.99   MC mean -0.2509 (mu=-0.250)
  beta=1.5  delta SE 0.5189   MC sd 0.2086   ratio 2.49   MC mean -0.2021 (mu=-0.250)
== D. n_eff/n at t=1, n=2000, across 10 data seeds ==
  beta=0.6  mean 0.713  sd 0.034  CV 0.048  min 0.670  max 0.760
  beta=1.5  mean 0.122  sd 0.096  CV 0.790  min 0.007  max 0.321
== E. survey bias E_S[Y]-mu over t, at the proposed default beta=0.6 ==
  t=1:+0.454  t=2:+0.411  t=3:+0.369  t=4:+0.330  t=5:+0.293  t=6:+0.258  t=7:+0.226  t=8:+0.195  t=9:+0.168
```
STOP CONDITIONS checked against this output: none fired (A converges at 0.6/diverges at 1.5 as
the theorem predicts; C ratio 0.99 at beta=0.6, within [0.75,1.33]; D CV=0.048 at beta=0.6,
under 0.20).

`python tools/twin_default_run.py` full stdout (config beta=0.6,H=32,lr=3e-3,wd=1e-5,epochs=400,
batch=256,seed_data=0,seed_model=1; commit a350bbd -- corrected from an earlier draft that cited
b4a438d, which does not contain tools/twin_default_run.py; verified via
`git show a350bbd --stat | grep twin_default`):
```
config: {'m': 6, 'M': 9, 'n': 2000, 'sigma': 1.0, 'beta': 0.6, 'H': 32, 'lr': 0.003, 'wd': 1e-05, 'epochs': 400, 'batch': 256, 'val_frac': 0.2, 'seed_data': 0, 'seed_model': 1}
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
This matches EVERY reference number quoted in the task prompt exactly (best val, FOC range,
alpha~, alpha*, theta~-theta* grid, mu~/oracle/n_eff at t=7,8,9) -- strong confirmation the
environment (numpy/scipy versions) matches the prior session's.

`python fusion_numpy.py` (its own CFG: lr=1e-2,wd=1e-4,epochs=150, now beta=0.6) full stdout:
```
4.3 grad-check worst rel. error: 9.030793659470643e-09
5.2 best val loss 0.4469 at epoch 34
5.3 FOC diagnostic (should be ~1): {1: np.float64(1.046), 2: np.float64(1.034), 3: np.float64(1.0), 4: np.float64(1.005), 5: np.float64(0.985), 6: np.float64(0.983)}
    learned alpha: [-0.371 -0.269 -0.202 -0.099 -0.051  0.   ]
    true    alpha: [-0.439 -0.332 -0.234 -0.147 -0.069  0.   ]

 t   mu_true  survey_mean  mu_tilde  n_eff/n  offset_baseline  oracle_theta*
 7   1.250     1.505       1.286    0.92      1.247            1.298
 8   1.500     1.689       1.500    0.95      1.431            1.518
 9   1.750     1.943       1.708    0.92      1.685            1.751
```

`python -m pytest -v tests/` -> **14 passed, 0 failed** (T1..T11, all exist, none skipped):
T1,T2,T3 (test_dgp.py); T4,T5,T6,T7,T8 (test_model_loss.py); T9,T10 (test_estimate.py);
T11a,T11b,T11c,T11d (test_t11_weight_tails.py). Notably T7 -- the CP2-documented failure at the
old beta=1.5 (a=1.0255 vs tol 0.02) -- now PASSES: at beta=0.6 the parametric-recovery fit
converges with much lower variance (finite weight moments). No test hardcodes a beta=1.5-derived
expected value; all expected values come from `dgp.*` closed-form functions parameterized by
`cfg.beta`, so none needed hand re-deriving.

Figures regenerated (all 9 kinds, previously at beta=1.5 hashes 4b6ab92b/8e4fef38, now at
beta=0.6 hash 96169d0c): `python -m fusion.run --phase 1`, `--phase 3`
(best_epoch 140, best_val_loss 0.4481, FOC {1:1.037,2:1.012,3:1.003,4:0.998,5:0.988,6:0.969}),
`--phase 4`. Opened all 9 regenerated PNGs:
- phase1a_densities: s_t sits only slightly right of p_t at every t (weaker separation than the
  old beta=1.5 tails).
- phase1b: mu(t)/E_St[Y] gap and rho_bar_t's hump are qualitatively the same shape as before.
- phase3_val_curve: far smoother/flatter than the old beta=1.5 runs, converges by epoch ~140/400.
- phase3_alpha_hat_vs_star: alpha~ sits uniformly above alpha* across t=1..5, meeting at the t=6
  anchor -- a small roughly-constant offset (CLAUDE.md SS3's new reading rule: not a failure by
  itself).
- phase3_theta_hat_vs_star: **CORRECTED 2026-09-18 (M2)** -- the original "track closely for
  y<-2" line above was wrong; a linear-scale plot dominated by theta values near 8 (at y~-6)
  visually hides a real secondary divergence. Precise grid from a re-run of the SAME torch model
  behind this figure (`python -m fusion.run --phase 3`, deterministic, same seeds; NOT the numpy
  twin -- `tools/twin_default_run.py`'s printed grid, e.g. +5.71 at y=-3, is a DIFFERENT,
  unbounded model with its own hyperparameters and is not comparable to this figure), x-range
  [-6.25, 7.75], y-axis auto-scaled to the data (no fixed limits set in fusion/run.py):
  y=-6:diff=-0.078  y=-5:diff=+0.605  y=-4:diff=+0.850  y=-3:diff=+0.699  y=-2.5:diff=+0.489
  y=-2:diff=+0.195  y=-1:diff=-0.049  y=0:diff=-0.049  y=1:diff=-0.015  y=2:diff=-0.054
  y=3:diff=-0.114  y=4:diff=-0.227  y=5:diff=-0.367  y=6:diff=-0.507  y=7:diff=-0.652
  (Y.min/max = -3.87/4.98). Per-region description consistent with these numbers: (1) left tail
  y in [-5,-2.5]: a real bump, theta~ OVERSHOOTS theta* by up to +0.85 (at y~-4), shrinking back
  toward 0 by y~-6; (2) middle -2<y<2 (where most S_t mass for t<=m lives): good agreement,
  |diff|<=0.2; (3) right tail y>3: theta~ UNDERSHOOTS theta* by a growing amount, from -0.11 at
  y=3 to -0.65 at y=7, while theta*(y) itself is nearly flat (plateaus near -0.36) -- this is the
  region relevant to t=7,8,9 (Y ranges m_t(7..9) +/- a few sigma), so it is the more consequential
  tail for mu~(t>m).
- phase4_main_figure: mu~(t) tracks mu(t) closely in- and out-of-sample, better than survey mean,
  comparable to the offset baseline.
- phase4_stress1: bias -0.092/-0.160/-0.155 at drift 0.10/0.20/0.30 -- still non-monotonic.
- phase4_stress2: 7.5% of t>m survey values outside training range; bias stays small
  (-0.09 to -0.13).
- phase4_stress3: sd(mu~) vs n tracks the 1/sqrt(n) reference closely.

Discrepancies against the task prompt's reference numbers (Section B/C individual MC numbers use
fresh randomness not claimed bit-reproducible, unlike twin_default_run.py's): section B's
n*bias trajectory at beta=0.6 (+0.48,+2.56,-0.56 here vs quoted 0.68,0.80,0.93) and section C's
beta=0.6 delta SE/MC sd (0.0342/0.0347 here vs quoted 0.0397/0.0360, quoted at 400 reps vs this
script's 200) differ in the individual numbers while the qualitative phenomenon (flat sd*sqrt(n)
at beta=0.6, ratio near 1) matches; section A, D, E numbers match the reference closely or
exactly where directly comparable.

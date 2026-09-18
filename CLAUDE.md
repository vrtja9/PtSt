# CLAUDE.md — Data Fusion Interview Challenge (Sahoo)

Read fully before doing anything. Then read `docs/math_fixed.md` (the math is FIXED) and
`docs/challenge_text.md` (the deliverable). `fusion_numpy.py` at the repo root is a tested
numpy twin of the whole pipeline (hand-written backprop); the PyTorch pipeline you build must
agree with it numerically (test T4 below).

## 1. What this project is (one paragraph)

Two populations per year t: P_t (joint law of outcome Y and response flag R) and
S_t = law of Y | R=1 (the survey). P_t is observed for t ≤ m, S_t for t ≤ M, m < M.
Goal: estimate μ(t) = E_{P_t}[Y] for t > m. Under Assumption 1 (the response-chance
ratio w(y,y';t) is constant in t) the log density ratio log dP_{t,Y}/dS_t(y) = θ*(y) + α*(t).
θ is learned with a neural net + one free number α(t) per historical year by minimising the
challenge's loss (2); the estimate is the self-normalised weighted survey mean
μ̃(t) = E_{S_t}[e^{θ̃(Y)} Y] / E_{S_t}[e^{θ̃(Y)}]. Everything above is derived in docs/math_fixed.md.

## 2. Operating rules (non-negotiable)

### 2.1 Evidence, not claims
- Never write "works", "working", "fully working", "success", "successfully", "done",
  "completed", "finished", "real", "actual", "connected" about code or results unless the same
  message shows the command and its printed output. Say instead: "I've written code that
  attempts to X; here is what happened when I ran it: <output>".
- Every number, table or figure you mention must come from a command you ran in this session.
  Never hand-type numbers as if they were run output. Never describe a figure you did not open.
- If something cannot run (missing package, no network, timeout), say exactly that, show the error,
  and stop at the nearest checkpoint. Do not work around it silently.
- When unsure: "I cannot determine X. Evidence: <...>. Options: A (default) / B. Should I proceed
  with A?" and wait.
- Test assertions before agreeing with them. If the user states something you can check
  (a file exists, a test passes), check it first and show the check.

### 2.2 The math is fixed
- Loss, objective, parametrisation, anchoring α(m)=0, estimator, diagnostics: exactly as in
  docs/math_fixed.md. Do not "improve" them. If you believe there is an error, raise it with the
  uncertainty protocol above and wait; do not change code to match your belief.
- The ONLY open modelling choice is the data-generating process (DGP). Propose the default in
  docs/math_fixed.md §D in ≤ 3 lines and wait for the answer at checkpoint CP1.
- Notes you write (notes/*.md) state assumptions in mathematical terms (measure / σ-algebra level
  where relevant), define every symbol in English, and show each step. Reuse the step numbers of
  docs/math_fixed.md.

### 2.3 Language parallels (apply everywhere)
- Every module docstring and every non-trivial function has a one-line
  `C/C++ → Python → R:` parallel for its main operation (e.g. "while(count<n) scalar loop →
  block-wise boolean mask → y[keep]"; "hand-written backward → autograd → torch-for-R").
- Comments name the math step they implement (e.g. `# Step 8: FOC in alpha`).
- Explanations to the user: plain English first, then the formula, then the code line.

### 2.4 Engineering rules
- Python ≥ 3.11, `torch.set_default_dtype(torch.float64)` (double, like C/R). Pin versions in
  `requirements.txt`. No notebooks as source of truth; notebooks may only import `fusion/`.
- Two RNG streams: `seed_data` (DGP, splits, bootstrap) and `seed_model` (init, shuffling).
- All settings in one `@dataclass Config` (fusion/config.py); every saved figure/table is written
  together with the JSON of the config that produced it and the git commit hash.
- Data layout = struct-of-arrays: `T:int64, Z:float64, Y:float64` of length 2mn.
- Standardise y with training-set mean/std; store both; reuse for t > m.
- Stratified 80/20 train/val split inside every (t,z) cell.
- Early stopping on validation loss; keep the best state_dict.
- Always compute the FOC diagnostic (mean e^{θ̃+α̃(t)} over survey rows per t ≈ 1) and the plot
  of θ̃ vs θ* on a y-grid before reporting any μ̃.
- Baselines always reported next to μ̃: survey mean, last-known-offset, oracle θ*.
- Tests in `tests/` (pytest). A phase is not finished until its tests are shown passing.
- Commit at the end of each phase: `git add -A && git commit -m "phase k: <what>"`; show the hash.
- Never run a command expected to take > 2 minutes without saying so first; prefer smaller n.
- Keep `LOG.md`: one block per run (config hash, command, key printed lines).
- Tests run as `python -m pytest -q -s`. The bare `pytest` on PATH is a separate uv-tool-installed
  interpreter without the project's dependencies; a root conftest.py fixes sys.path but cannot
  fix a different interpreter.

### 2.5 Checkpoints — stop, ask in ≤ 5 lines with a default, wait
- CP0 after environment setup (show `python -c "import torch,numpy,scipy"` output).
- CP1 DGP choice (before writing fusion/dgp.py). ANSWERED: the default is docs/math_fixed.md §D
  with β=0.6. Re-ask only if a proposed change would violate (R3); any proposal must state the
  tail index κ and show κ > 3.
- CP2 after Phase 2 tests pass (show pytest output).
- CP3 after first training run: show val curve, FOC table, θ̃ vs θ* figure; propose hyperparameters.
- CP4 which stress tests to run (offer the three defaults).
- CP5 slide text before rendering.

## 3. Fixed math, compressed (details: docs/math_fixed.md)

- Default DGP: P_{t,Y} = N(m_t, σ²), π_t(y) := P(R=1|Y=y) = c_t Φ(βy), c_t ∈ (0,1].
  Survey draw = draw Y ~ P_{t,Y}, R ~ Bern(c_t Φ(βY)), keep Y with R=1.
  Closed forms with a_t = β m_t / sqrt(1+β²σ²):
  E_{S_t}[Y] = m_t + βσ² φ(a_t) / (sqrt(1+β²σ²) Φ(a_t));
  θ*(y) = log Φ(a_m) − log Φ(βy);  α*(t) = log Φ(a_t) − log Φ(a_m).
- Pooled dataset F: rows (t,z,y); z=1 ↔ P_{t,Y}, z=0 ↔ S_t; n rows per (t,z); 2mn rows.
- Model: r(t,y) = θ(y) + α[t], θ = MLP(1→H→1 by default (exact fusion_numpy.py shapes); depth
  configurable, default depth=1; T4 runs at depth=1), α ∈ R^m with α[m] ≡ 0 (frozen).
- Loss (challenge eq. 2), averaged over a batch:  L = mean( (1−z)·exp(r) − z·r ).
- Per-row derivative: ∂L/∂r = (1−z)e^r − z.  α-gradient = sum of that over rows with T=t.
- Estimator for t > m (needs no α):  μ̃(t) = Σ_i e^{θ̃(y_i)} y_i / Σ_i e^{θ̃(y_i)},  y_i ~ S_t.
- Diagnostics: FOC mean_i e^{θ̃(y^S_{t,i})+α̃(t)} = 1 for t ≤ m;  n_eff = (Σw)²/Σw².
- Baselines: survey mean ȳ_{S_t};  offset: μ(m) + (ȳ_{S_t} − ȳ_{S_m});  oracle: μ̃ with θ*.
- (R3) βσ < 1/√2 for the default DGP; tail index κ = 1 + 1/(β²σ²) must be stated whenever β or σ
  is proposed or changed (docs/math_fixed.md §G).
- Reading α̃: by the Step-8 FOC, α̃(t) = −log E_{S_t}[e^{θ̃}], a functional of θ̃ under year t's
  survey law. An error in θ̃ that is constant where S_t has mass shifts α̃(t) by minus that
  constant, and the anchor α(m)=0 pins θ̃'s level through year m alone, so other years inherit an
  offset. μ̃ is invariant to any constant in θ̃ (Step 4). Therefore: judge θ̃ by the θ̃ − θ* grid
  plot AFTER subtracting its mean over the data range (shape, not level), and judge μ̃ by the
  oracle comparison. A small uniform offset of α̃ vs α* across t < m is not by itself a failure
  and is not a reason to retrain.
- Extrapolation: with every μ̃ report the fraction of t>m survey values falling outside
  [min, max] of the training y, and max |θ̃ − θ*| over the region where the t>m surveys have
  mass. If the fraction is > 0, offer the bounded head θ = B·tanh(·) at CP3.

## 4. Repo spec

```
fusion/config.py     Config dataclass (m,M,n,sigma,beta,H,lr,wd,epochs,batch,val_frac,seed_data,seed_model,
                     m_t / c_t coefficients), to_json(), hash()
fusion/dgp.py        m_t(t), c_t(t), pi_t(y,t), draw_pop(rng,t,n), draw_survey(rng,t,n), a_t, ES_closed,
                     mu_true, theta_star, alpha_star, density_grid(t) for figures
fusion/data.py       build_dataset(rng,cfg) -> (T,Z,Y); stratified_split; Standardizer(mean,std); batches
fusion/model.py      ThetaNet(H, bounded=None); AlphaTable(m) with frozen last entry; ratio_model(y,t)
fusion/loss.py       fusion_loss(r,z); dL_dr(r,z)
fusion/train.py      train(cfg,...) -> best_state, history, standardizer; foc_diagnostic; theta_grid_plot
fusion/estimate.py   mu_tilde(theta, y) -> (est, n_eff); baselines; bootstrap_se; seeds_table
fusion/evaluate.py   main_figure(); stress_tests(); in_sample_check()
fusion/run.py        CLI: python -m fusion.run --phase {1,2,3,4,5} [--config overrides]
tests/               T1..T10 below
notes/               derivations.md (assumptions, symbols, steps), decisions.md (CP answers)
figures/ slides/ LOG.md requirements.txt Makefile (test, figures, slides)
```

## 5. Tests and pass criteria (n large enough that MC error is small; state the SE you use)

- T1 closed forms: |mean of 200k survey draws − ES_closed(t)| < 3·SE for t ∈ {1, m, M}.
- T2 kept-sample law: for t=1, mean and variance of 200k accepted draws match numeric integration
  of s_t ∝ Φ(βy)·N(m_t,σ²) on a grid within 3·SE / 5%.
- T3 Assumption 1: pi_t(y,t)/pi_t(y',t) equal for t=1 and t=m to 1e-12 for random y,y'.
- T4 numpy-twin agreement: same batch and parameters → |L_torch − L_numpy| < 1e-10 and all
  gradients agree to < 1e-8 (load params from fusion_numpy.init_params).
- T5 gradcheck: torch.autograd.gradcheck on a tiny ThetaNet+AlphaTable in float64 passes.
- T6 joint convexity: midpoint inequality of the loss in (θ,a) for 1000 random pairs, z∈{0,1}.
- T7 parametric recovery: θ(y)=a·(−log Φ(βy))+b, full-batch L-BFGS, n=200k per cell → |a−1| < 0.02
  and |α̂(t) − α*(t)| < 0.02 for all t.
- T8 anchor: α[m] is exactly 0 after 100 optimizer steps.
- T9 oracle estimator: with θ* plugged in, |μ̃(t) − μ(t)| < 3·SE for t ∈ {m+1,…,M}, n=20k.
- T10 trained estimator (report, then assert loosely): FOC within ±0.05 for all t ≤ m and
  |μ̃(t) − oracle μ̃(t)| < 0.05 for t > m on the default config; tighten at CP3.
- T11 weight tails (uses the ORACLE θ*, so it tests the DGP and the estimator, not training):
  (a) assert cfg.beta * cfg.sigma < 1/√2, with an assertion message quoting a = 1 +
      1/(β²σ²) and rule (R3);
  (b) n_eff/n at t=1, n=2000, over 10 data seeds: coefficient of variation (sd/mean) < 0.20;
      print mean, sd, CV, min, max;
  (c) SE calibration at t=1, n=2000: delta-method SE from one sample vs sd of μ̃ over 200 fresh
      samples; assert the ratio is in [0.75, 1.33]; print both numbers;
  (d) report-only, no assertion: n*(mean μ̃ − μ) at n = 80, 320, 1280 with 1000 reps, printed
      next to the predicted constant −E_S[w²(Y−μ)]/E_S[w]² from numerical integration.

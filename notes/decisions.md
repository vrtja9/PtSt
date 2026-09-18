# Decisions log (checkpoint answers)

Format: one entry per checkpoint, dated, with the exact reply received and what changed as a
result. Every code-level decision also gets a `# DECISION:` comment at its point of use, stating
what was chosen, why, and which downstream artifacts (tests/figures/later phases) depend on it —
so a later "change X" request can be traced to exactly what must be re-derived or re-run.

## CP0 — environment (2026-09-16)
Reply: `ok`. Evidence: torch 2.5.1+cu124 import check passed; `python fusion_numpy.py` ran end to
end (grad-check 1.17e-8; FOC 1.02-1.11 under the script's own CFG, not the lr=3e-3/400-epoch
config docs/math_fixed.md §F cites). No changes requested.

## CP1 — DGP choice (2026-09-16)
Reply: `default`. DGP = docs/math_fixed.md §D exactly:
P_{t,Y} = N(m_t, σ²), σ=1; π_t(y) = c_t·Φ(βy), β=1.5; m_t=−0.5+0.25t; c_t=0.9−0.06t; m=6, M=9,
n=2000. Implemented in `fusion/config.py` (the numeric constants) and `fusion/dgp.py` (the
functions). If this changes: `fusion/config.py` defaults are the single edit point; everything
downstream that must be re-run is listed in the `# DECISION` comment next to those fields.

## Uncertainty raised mid-Phase-2 — ThetaNet architecture (2026-09-16)
Raised per CLAUDE.md §2.1/§2.2 before writing `fusion/model.py`: CLAUDE.md §3 states
`θ = MLP(1→H→H→1, ReLU)` (two hidden layers), but `fusion_numpy.py`'s `init_params`/`theta_of`
(the numpy twin T4 must agree with bit-for-bit) implement a single hidden layer only
(`W1:(H,1), b1:(H,), w2:(H,), b2:scalar`), and T4 requires loading those exact shapes. The two
cannot both be literally true. Options offered: (A) implement `1→H→1` matching the twin, treating
the `1→H→H→1` line in §3 as a compression error [recommended, since T4 is mechanically
unsatisfiable otherwise]; (B) implement the literal two-hidden-layer net and let T4 fail/be
reinterpreted. **Reply: A.** `fusion/model.py`'s `ThetaNet` is therefore `1→H→1`, matching
`fusion_numpy.py` exactly. If this is ever revisited (e.g. a real second hidden layer is wanted):
`ThetaNet.__init__`/`forward` in `fusion/model.py` are the only edit points, but T4 would then
need a different agreement check (the twin does not have a second-layer analogue to compare against).

## CP2 — Phase 2 tests (2026-09-16)
Reply: `Default` (accept T7 as documented). `python -m pytest tests/` result kept as-is: 7 passed
(T1-T6, T8), T7 FAILED at the literally-specified n=200k/seed_data=0/tol=0.02 (`a=1.0255`). No
code changed to force a pass — `tests/test_model_loss.py::test_T7_parametric_recovery_lbfgs`
keeps `n=200_000`, `Config()` default `seed_data=0`, and the `0.02` thresholds exactly as
CLAUDE.md §5 states them. The diagnostic evidence (4-seed sweep, 3-n sweep showing the error
shrinks with n) stays in notes/derivations.md "Phase 2 evidence" and LOG.md as the permanent
record of why this is treated as known finite-sample MC variance, not a defect, and why Phase 3
proceeded without re-touching T7. If revisited later: the three literal values above
(`n`, `seed_data`, `0.02`) in that one test function are the only edit points.

## CP3 — first training run, hyperparameters (2026-09-16)
Evidence at `lr=3e-3` (unbounded ThetaNet): noisy/spiky val curve still descending at epoch
~395, FOC `{1:1.079,2:1.106,3:1.0,4:1.027,5:1.045,6:1.012}` (3/6 outside ±0.03), θ̃ blowing up to
~380 at y=-6 vs θ*'s <50. Proposed default: `lr=1e-3` + bounded head `B=20`. **Reply: `Adopt`.**
Changed in `fusion/config.py`: `lr` 3e-3→1e-3, added `theta_bounded=20.0` field (see the
`# DECISION (CP3...)` comment there); `fusion/train.py` now constructs `ThetaNet(cfg.H,
bounded=cfg.theta_bounded)`. Re-run evidence: val curve smooth/monotone (no spikes), θ̃ plateaus
at 20 instead of exploding, FOC `{1:1.005,2:0.958,3:0.963,4:1.012,5:1.025,6:0.974}` (2/6 still
marginally outside ±0.03: t=2,3). Judged good enough to proceed (large, qualitative improvement;
CP3's own reply templates don't include "iterate again" and none was requested) — the residual
t=2,3 FOC gap and the still-flattish α̃ vs α* shape are recorded honestly in
notes/derivations.md rather than hidden. If revisited: `fusion/config.py`'s `lr`/`theta_bounded`
fields are the edit points; re-run `python -m fusion.run --phase 3` and re-open the three figures.

## CP4 — stress test selection (2026-09-16)
Reply: `Default` (all three of docs/math_fixed.md §E). Implemented in `fusion/evaluate.py`:
(1) Assumption 1 broken (`π_t(y)=σ(a_t+b_t y)`, `b_t=1.5+0.1(t−m)`, t>m only);
(2) support shift (`m_t`+2σ, t>m only); (3) small n (`n∈{200,500,2000}`, sd of μ̃ over 10 seeds).
Design choice for (3), not fully pinned by the kickoff prompt: the trained θ̃ from the Phase 3
CP3-adopted run is held FIXED, and only the S_t draw seed varies across the 10 replicates per n
(isolating estimator/sampling variance from training variance, and keeping the test's cost to a
few hundred forward passes instead of 30 full retrains). If a training-variance version is
wanted instead: `fusion/evaluate.stress_test_small_n` is the only function to change.

## Authorized correction — beta 1.5 -> 0.6 under rule (R3) (2026-09-18)
User-authorized change to CLAUDE.md §2.2's frozen DGP default (explicit override of that section
for this one item; loss/parametrization/anchor/estimator/pooled-dataset untouched). What changed:
`beta` 1.5 -> 0.6 everywhere it was a literal default (docs/math_fixed.md §D, fusion_numpy.py
CFG, fusion/config.py Config.beta); new design rule (R3) requiring βσ < 1/√2; new
docs/math_fixed.md §G stating and proving the weight-moment theorem behind (R3); three new
identities in §C (delta-method SE, n_eff bounds, finite-sample bias, centering identity); new
test T11 (CLAUDE.md §5, tests/test_t11_weight_tails.py); two new CLAUDE.md §3 reading rules
(judging α̃ by shape not level; extrapolation reporting); Config now raises ValueError if
beta*sigma >= 1/√2 unless `allow_heavy_tails=True` is passed.

The theorem: E_{S_t}[w^k] < ∞ iff k < a := 1+1/(β²σ²) (Mills'-ratio tail argument). At β=1.5,
σ=1: a=1.444 — neither the weight's variance (k=2) nor third moment (k=3) exists, invalidating
both the delta-method SE and the Hájek estimator's O(1/n) bias expansion. At β=0.6: a=3.78, both
exist.

Evidence from `tools/check_weight_tails.py` (full stdout in LOG.md "beta correction" entry; run
2026-09-18):
- §A: E_S[w^2] converges as the integration range grows at β=0.6,0.8 (3.21/3.21/3.21;
  5.86/5.86/5.86) and diverges at β=1.0,1.5 (307→9.26e3→1.36e6; 1.94e28→6.41e110→4.71e168) —
  matches the theorem's k=2 threshold (a>2) exactly; none of the STOP CONDITIONS on this section
  fired.
- §B: at β=0.6 the predicted n·bias (1.084) and the observed sd·√n (1.53–1.63, flat across
  n=80,320,1280) show the O(1/n) bias regime holding; at β=1.5 sd·√n grows with n (2.95→5.07→7.51)
  and n·bias grows without settling (14.0→30.9→83.3) — the predicted constant is undefined there
  (a≤3).
- §C: delta SE vs 200-rep Monte-Carlo sd at t=1,n=2000 — β=0.6 ratio 0.99 (within the [0.75,1.33]
  stop-condition bound); β=1.5 ratio 2.49 (delta-method SE badly miscalibrated under infinite
  variance).
- §D: n_eff/n over 10 seeds at t=1,n=2000 — β=0.6 CV=0.048 (within the <0.20 stop-condition
  bound), range [0.670,0.760]; β=1.5 CV=0.790, range [0.007,0.321] (47x swing — the
  infinite-variance fingerprint).
- §E: survey bias at β=0.6 shrinks from +0.454 (t=1) to +0.168 (t=9) — replaces the stale
  ≈0.78/≈0.12 numbers (computed at the old β=1.5) in docs/math_fixed.md §D.

CP1 is thereby re-answered: default is now docs/math_fixed.md §D with β=0.6 (CLAUDE.md §2.5's
CP1 line updated to say so; re-ask only if a future proposal would violate (R3)).

Side effect worth recording: `python -m pytest tests/ -v` now shows **14 passed, 0 failed**
(T1-T11), including `test_T7_parametric_recovery_lbfgs`, which was the CP2-documented failure at
the old β=1.5 (n=200k, seed_data=0, `a=1.0255` vs tol 0.02). At β=0.6 the weight has finite
variance and third moment (a=3.78), so the L-BFGS parametric-recovery fit converges with far
lower variance; the CP2 entry above is left as the historical record of what was observed and
decided at the time, not rewritten.

**T7's CP2 failure is now explained, not just fixed** (2026-09-18, docs/math_fixed.md §G
"Consequence: why T7 (parametric recovery) fails under a violated (R3)"): T7's survey term at
the truth a=1 is the sample mean of w=e^{θ*(Y)}; its sampling variance is infinite whenever
βσ≥1 (a≤2 — confirmed for β=1.5 by `check_weight_tails.py` §A's diverging k=2 column), and a
sample mean of a heavy-tailed positive variable is typically below its expectation in any one
finite sample, biasing the L-BFGS fit's â upward — exactly the direction CP2 observed
(â=1.0255). T7's old failure was therefore a SYMPTOM of the same (R3) violation being corrected
in this pass, not an independent training-hyperparameter or optimizer bug, and must not be
re-litigated as one; see docs/math_fixed.md §G for the full mechanism and the measured decay
rate matching the theorem's stable-law exponent.

Known, deliberately out-of-scope staleness (not fixed, since the authorization did not cover it):
docs/math_fixed.md §E stress-test 1's `b_t = 1.5 + 0.1(t−m)` literal still reads the old β value
in its prose (the actual code, `fusion/evaluate.stress_test_assumption1_broken`, already uses
`cfg.beta + 0.1*(t-m)` and so is automatically consistent with the new default); `slides/deck.md`
and `fusion/render_slides.py`'s slide-1 bullet text still say "beta=1.5" (Phase 5 deliverable,
not touched under this authorization's scope).

## C1-C4 follow-up corrections (2026-09-18)
C1: `git_commit_hash()`'s docstring now states the code-commit invariant (the commit whose
CURRENT code reproduces the artifact, not the commit that added the file); re-audited all 10
current figures against it (LOG.md "C1-C4 follow-up corrections" has the per-figure diff).
C3: `check_weight_tails.py` §B and T11(d) now use a three-outcome label (DISAGREES / CONSISTENT /
TOO WIDE TO DISTINGUISH) keyed on interval half-width, not containment alone, at n=40,80,160,320
with 10000 reps (n=1280 dropped as unresolvable). C4: new docs/math_fixed.md §H "Residual error
after (R3)" records that the remaining μ̃(t>m) error is θ̃'s approximation error (not sampling
error, since n_eff/n=0.93-0.96), verified to fully account for μ̃(9)'s gap via the interpolated
θ̃−θ* drift (LOG.md has the numbers), and that `cfg.theta_bounded=20` never binds over the
training range — recorded, not changed, so slide 3 doesn't overclaim what the CP3 fix did.

## Deck rebuild under the HARD RULE (2026-09-18)
User's HARD RULE: every number on a slide is read programmatically from the repo at build time;
no number is hand-typed into the build script; the build fails loudly if a value is missing.
Implemented as two scripts: `fusion/build_deck.py` (`collect()`) re-runs the live pipeline --
trains a fresh model, re-derives T4's numpy-twin agreement and T7's L-BFGS fit (at both the
current default and, via `allow_heavy_tails=True`, the old β=1.5 for the slide-3 contrast), runs
the full `pytest` suite as a subprocess to get a real "N passed" string, and computes the
C4(a)-style implied-shift/mass-share numbers -- and writes everything to `slides/slide_data.json`
(the "results table" the HARD RULE allows). `fusion/render_slides.py` then only interpolates
`slide_data.json`'s fields into static bullet prose via f-strings; the only hand-written numeric
literal anywhere is `math.sqrt(2)` inside the (R3) rule's prose, itself computed, not typed as a
decimal. `fusion/run.py --phase 5` (hence `make slides`) was updated to call both in sequence,
since the old phase-5 code called `render_slides`'s old zero-argument functions, which no longer
exist -- verified by an end-to-end re-run (75-79s, dominated by the fresh training run + T7 x2 +
the full pytest subprocess). All three slides re-rendered to PNG and opened; content confirmed
legible and numbers cross-checked against LOG.md's independently-computed values (exact matches
on every deterministic-seed quantity: FOC, T7 a_hat at both β's, T11 numbers, θ* bounds).

## Known stale: Phase 5 slides reference beta=1.5-era content (2026-09-18) -- RESOLVED below
At the time this was written, not edited (explicitly out of scope; user said "decide and report,
do not act"). Superseded by the same day's deck rebuild (see "Deck rebuild" entry below): all 3
slides were rebuilt from `slides/slide_data.json`, sourced live from the repo, so every number
named in the checklist below is now current. Left as the historical record of what was stale and
why, not deleted.
**3 of 3 slides were affected.** Checklist that guided the rebuild
(`fusion/render_slides.py`'s `SLIDES` list, `slides/deck.pptx`, `slides/deck.md`):
- **Slide 1** ("The data-generating process"): text bullet "beta=1.5" is wrong (now 0.6); text
  bullet "shrinks from ~0.78 (t=1) to ~0.12 (t=9)" is wrong (now ≈0.45/≈0.17, docs/math_fixed.md
  §D); embedded figure `phase1a_densities_4b6ab92b.png` was generated at β=1.5 (visually more
  separated p_t/s_t curves than the current β=0.6 version) — replace with
  `phase1a_densities_96169d0c.png`.
- **Slide 2** ("Optimization pipeline"): text bullets don't cite a beta value so are not wrong,
  but the embedded figure `phase3_val_curve_8e4fef38.png` was generated at the old β=1.5 default
  (post-CP3) — replace with `phase3_val_curve_96169d0c.png` (both are smooth/monotone post-CP3,
  but the underlying numbers differ).
- **Slide 3** ("Comparison and learnings"): text bullet "Learning 1: ... (t=9: 1.752 vs true
  1.75, survey mean 1.858, offset 1.601)" is wrong — current β=0.6 values (`python -m fusion.run
  --phase 3`, config hash `96169d0c`) are t=9: mu_true=1.750, survey_mean=1.888, mu_tilde=1.696,
  offset=1.677; embedded figure `phase4_main_figure_8e4fef38.png` was generated at β=1.5 —
  replace with `phase4_main_figure_96169d0c.png`. Learnings 2 and 3's prose describes historical
  events (the CP3 tail-blowup fix, the stress-1 non-monotonic finding) that remain true as
  historical claims, though stress 1's own numbers have since changed twice (beta correction,
  then A3(a)'s baseline fix) — worth re-checking against notes/derivations.md's current stress-1
  numbers when the deck is rebuilt.
Note: `slides/deck.pptx`'s embedded images are baked into the file at build time (python-pptx
copies image bytes in, not a live reference), so deleting the stale source PNGs from `figures/`
(this session's M3 fix) does not corrupt the existing pptx — it just means `fusion/render_slides.py`
can no longer rebuild slide 1/2/3's images from the now-deleted `_4b6ab92b`/`_8e4fef38` filenames
without first re-pointing the `SLIDES` list at the current `_96169d0c` figures, which is exactly
the rebuild this checklist is for.

## CP5 — slide text (2026-09-16)
Proposed 3-slide text (docs/challenge_text.md §4) sent to the user; **reply: `Go`**. Slide 1:
DGP + `figures/phase1a_densities_*.png`. Slide 2: optimization pipeline (loss, Adam, bounded
ThetaNet, FOC/θ̃-vs-θ* diagnostics) + `figures/phase3_val_curve_*.png`. Slide 3: comparison +
3 learnings (μ̃ beats both baselines; unbounded nets can blow up in sparse tails; Assumption-1
violation biases μ̃ in a non-obvious direction) + `figures/phase4_main_figure_*.png`. Rendered in
`fusion/render_slides.py` (python-pptx) to `slides/deck.pptx`.

## Deck review round 2 — slide 3 corrections and tail-index rename (2026-09-18)
User reviewed the rebuilt (HARD-RULE, live-numbers) deck and found 4 wrong/unsupported claims and
2 gaps, all on slide 3, plus a notation collision on slide 1 and a missing R̂_n measurement on
slide 2. Four fixes, applied in this pass:

1. **Notation (slide 1).** `a_t`/`a_m` (§D's DGP scalars) and the weight-moment tail index
   (§G/(R3)) were both called `a` on the same slide. The tail index is renamed **κ** everywhere:
   `docs/math_fixed.md` §C/§D/§G/§H, `CLAUDE.md` §2.5/§3, `fusion/config.py`, `fusion/evaluate.py`,
   `fusion/run.py`, `fusion/build_deck.py`, `fusion_numpy.py`, `tests/test_t11_weight_tails.py`,
   `tools/check_weight_tails.py`. `a_t`, `a_m`, and T7's fitted slope `â` are untouched — they are
   different quantities that happened to share the letter. Slide 1 also gained the missing
   definition `a_t := βm_t/√(1+β²σ²)` and the closed form `E_St[Y] = m_t + βσ²φ(a_t)/(√(1+β²σ²)Φ(a_t))`,
   and the sample space is now written `(Y×{0,1}, B(Y)⊗2^{0,1})`.
2. **R̂_n unbounded-below (slide 2).** Quoted the live values of the piecewise-linear
   `θ_k=±k`-construction's `R̂_n(k)=(1/2)(e^{-k}-k)` at k=0,1,2,4,8 next to the trained best
   validation loss, so "unbounded below" is a measurement, not an assertion.
3. **Slide 3, three claims corrected, one added** — see `docs/math_fixed.md` §H (the full
   derivation and live numbers) and `notes/talk_through.md` items 11-12 and its Q&A section for
   the corrected narrative:
   - The **circular check** (interpolating θ̃−θ* back onto itself and calling the near-exact match
     a verification) is deleted. Replaced with a genuine ablation: hold θ̃'s drift flat beyond a
     cutoff c and recompute μ̃(9) on paired draws.
   - **Attribution**: μ̃−μ = (oracle−μ)+(μ̃−oracle) shows the t=9 gap is only 38% model drift, the
     rest sampling; t=7,8's gaps (0.7 SE) are not distinguishable from sampling noise at all.
   - **Region diagnosis**: the far tail (y>3, 12.7% of `S_9`'s mass) explains only ~38% of the
     drift-shift; the 2<y≤3 band (33.7% of mass) dominates it, and that band is data-rich at t=9
     but data-poor in training (mass share above y=2, pooled t≤m vs `S_9`: 10.0% vs 46.4%, 4.7×)
     — a temporal covariate shift, not a tail-extrapolation problem. Reframed as a structural
     property of the data-fusion setup (μ̃−oracle grows monotonically with t), and added the
     sharper fix this diagnosis implies: reweight/augment training years toward the forecast
     years' y-region, or constrain θ̃ to be monotone+convex.
   - **Added headline** (previously missing): μ̃ cuts the survey's bias vs μ(t) by 93%/91%/49% at
     t=7/8/9 (`1 − |μ̃−μ|/|survey−μ|`), now the table's last column and Learning 3's opening line.
4. `notes/talk_through.md` was re-examined for the same three errors and corrected to match,
   including giving "why does μ̃ undershoot at t=9" a two-part answer (sampling is the larger
   share at 62%; covariate shift explains the smaller, model-attributable 38%) instead of the
   previous one-part ("θ̃ drift alone explains it") answer.

**Reproducibility note — CLOSED (2026-09-18, see the round-3 entry below for the closing
reasoning).** The user's own reference numbers for the counterfactual repair (full drift
`-0.0375`; y>3 contributes 30%; y>2 contributes 76%) did not reproduce exactly on this pass's
fresh training run (measured: full drift `-0.0286`; y>3 contributes 38%; y>2 contributes 92% —
same qualitative finding, the near-tail band dominates over the far tail, just a different
split). The decomposition, bias-reduction percentages, and covariate-shift mass-share ratios all
matched the user's reference numbers closely (within Monte Carlo noise). Per the HARD RULE, the
live numbers from this run — not the stale reference — are what the rebuilt slide/docs quote.

## Deck review round 3 — headline-sentence fix, counterfactual item closed (2026-09-18)
User confirmed the deck; one wording error, one open item, one optional clause:

1. **Fixed**: Learning 3's headline bullet closed with "sampling variability, not model error,
   explains most of the other gaps," which conflates "within 0.7 SE of zero" (a statement about
   *significance*, true at t=7,8) with "mostly sampling" (an *attribution* claim, false at t=8:
   the table's own row gives sampling `+0.0019` vs model `-0.0184` — the model term is ~10x the
   sampling term there). Replaced with the decomposition read the right way round: the
   model-drift term (`mu_tilde_minus_oracle`) is present at every forecast year and grows
   monotonically (`-0.0097, -0.0184, -0.0253`); the sampling term (`oracle_minus_mu`) is what
   swings in sign and size (`+0.0250, +0.0019, -0.0409`); only at t=9 do the two align in sign
   and push `mu~-mu` past 2 SE. Generated in `fusion/render_slides.py` from `slide_data.json`'s
   table rows, not typed. Checked `notes/talk_through.md` for the same conflation — its Q&A
   section (added in round 2) already frames this correctly (significance vs. attribution kept
   separate: "the same kind of gap is only 0.7 SE" is about the total gap, not a claim that
   sampling explains the per-row model term) — no edit needed there.
2. **Closed, not open**: the counterfactual "could not reproduce" item from round 2 is resolved.
   The user's reference (`-0.0375`, y>3 30%, y>2 76%) was computed from the θ̃−θ* grid captured
   during the M2 re-derivation — an earlier model checkpoint. The model has been retrained since
   (current: best val loss `0.4481` at epoch 140), so a different drift function δ(y) is the
   expected cause of the numeric difference, and the current numbers supersede the stale
   reference rather than needing to chase it. The internal cross-check that validates the
   *current* numbers: the 200-rep ablation gives full drift shift `-0.0286` (per-rep sd
   `0.0018`), and the main table's single-sample `mu_tilde_minus_oracle` at t=9 gives `-0.0253`
   — two independent routes to the same quantity (a 200-rep Monte Carlo average vs. one
   realized survey draw), `1.77` sd apart, i.e. consistent under normal sampling variability.
   No re-run was performed to chase the old reference number, per instruction.
3. **Added (optional clause taken)**: slide 3's mechanism bullet now parenthetically
   distinguishes the two numbers above where they appear two sentences apart — "(`-0.0286` is
   the 200-rep ablation mean; `-0.0253` is this sample's realisation, `1.8` sd apart)" — so the
   cross-check in point 2 is visible on the slide itself, not just in this log. Whether it still
   fits without overflow was checked by re-rendering the deck to PNG after this edit (see LOG.md,
   this date's entry, for the outcome).

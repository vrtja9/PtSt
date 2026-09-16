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

## CP5 — slide text (2026-09-16)
Proposed 3-slide text (docs/challenge_text.md §4) sent to the user; **reply: `Go`**. Slide 1:
DGP + `figures/phase1a_densities_*.png`. Slide 2: optimization pipeline (loss, Adam, bounded
ThetaNet, FOC/θ̃-vs-θ* diagnostics) + `figures/phase3_val_curve_*.png`. Slide 3: comparison +
3 learnings (μ̃ beats both baselines; unbounded nets can blow up in sparse tails; Assumption-1
violation biases μ̃ in a non-obvious direction) + `figures/phase4_main_figure_*.png`. Rendered in
`fusion/render_slides.py` (python-pptx) to `slides/deck.pptx`.

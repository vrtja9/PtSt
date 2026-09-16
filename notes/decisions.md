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

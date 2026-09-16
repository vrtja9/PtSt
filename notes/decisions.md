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

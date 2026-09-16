# Kickoff prompt for Claude Code (paste as the first message, in the repo root, after `claude` starts)

---

Read CLAUDE.md, docs/math_fixed.md, docs/challenge_text.md and fusion_numpy.py completely before acting.
Confirm in ≤ 5 lines what the fixed math is and what the single open choice is, then start Phase 0.

Time box: 4–8 hours of wall clock in total, in six phases. Stop at every checkpoint (CP0–CP5), ask in
≤ 5 lines with a stated default, and wait for my answer. Between checkpoints, work autonomously.
Report evidence, never claims (CLAUDE.md §2.1). Every explanation to me: plain English → formula → code line,
with the C/C++ → Python → R parallel (CLAUDE.md §2.3).

## Phase 0 — environment (≤ 30 min)
1. Create `requirements.txt` (numpy, scipy, torch CPU, matplotlib, pytest, python-pptx; pin versions),
   `Makefile` (targets: test, figures, slides), `.gitignore`, `LOG.md`, `notes/decisions.md`.
2. Install; run `python -c "import torch,numpy,scipy,matplotlib; print(torch.__version__)"` and
   `python fusion_numpy.py` (the numpy twin; ≈1–2 min). Paste both outputs.
3. `git init` if needed; first commit. → CP0.

## Phase 1 — DGP (≤ 60 min)
1. → CP1 first: propose the default DGP of docs/math_fixed.md §D in ≤ 3 lines and wait.
2. Implement fusion/config.py and fusion/dgp.py exactly to the agreed DGP, with the closed forms.
3. Tests T1, T2, T3. Show pytest output.
4. Figures: (a) p_t, s_t, π_t overlaid for t ∈ {1, m, M}; (b) μ(t), E_{S_t}[Y], ρ̄_t vs t.
   Save to figures/ with config JSON + commit hash. Open each PNG and describe what is visible in 1 line each.
5. notes/derivations.md: Steps 1–4 and 10 written out with assumptions and symbol definitions
   (copy the step numbering of docs/math_fixed.md). Commit.

## Phase 2 — data, model, loss, tests (≤ 90 min)
1. fusion/data.py (SoA rows, stratified split, Standardizer), fusion/model.py (ThetaNet, AlphaTable with the
   frozen last entry — implement as concatenation with a constant zero, not a mask), fusion/loss.py.
2. Tests T4 (agreement with fusion_numpy.py on one batch and one parameter set), T5, T6, T7, T8. Show output.
   T4 is the key one: load parameters from fusion_numpy.init_params(rng,H), copy them into ThetaNet/AlphaTable,
   compare loss and every gradient.
3. notes/derivations.md: Steps 5–9. Commit. → CP2.

## Phase 3 — training, diagnostics, estimation (≤ 90 min)
1. fusion/train.py: Adam, stratified batches, early stopping on validation loss, best state_dict,
   history; foc_diagnostic; theta_grid_plot (θ̃ vs θ* on a y-grid spanning all S_t supports).
2. First run with lr 3e-3, wd 1e-5, batch 256, epochs 400, H 32 (the settings that gave FOC 0.98–1.03 in the twin).
   Show: validation curve (figure), FOC table, α̃ vs α*, θ̃ vs θ* figure (open it, describe it). → CP3
   (propose hyperparameter changes if FOC is outside ±0.03 or θ̃ visibly deviates where t>m surveys live).
3. fusion/estimate.py: mu_tilde with n_eff; baselines (survey mean, last-known-offset, oracle θ*);
   bootstrap SE with θ̃ fixed; seeds_table over 5 seeds (seed_data and seed_model varied separately).
4. Tests T9, T10. Table for t = m+1..M: μ, survey mean, μ̃ ± SE, n_eff/n, offset baseline, oracle. Commit.

## Phase 4 — evaluation and stress tests (≤ 60 min)
1. → CP4: offer the three stress tests of docs/math_fixed.md §E; wait.
2. fusion/evaluate.py: main_figure (μ(t), E_{S_t}[Y], μ̃(t) with error bars, offset baseline, t>m shaded),
   in_sample_check (t ≤ m), the chosen stress tests with one figure each.
3. notes/derivations.md: a "failure modes" section stating for each stress test what the math predicts and
   what was observed (numbers from the runs). Commit.

## Phase 5 — deliverable (≤ 60 min)
1. → CP5: propose the text of the three slides (DGP + figure; optimisation pipeline + loss + diagnostics;
   comparison figure + three learnings) in ≤ 15 lines; wait.
2. Render slides/deck.pptx with python-pptx (fallback: slides/deck.md for Keynote paste); embed the figures.
   Open the rendered slide images (convert to PNG) and describe each in 1 line.
3. Write notes/talk_through.md: 10 bullet points mapping each slide to code files and to docs/math_fixed.md steps.
4. Final LOG.md entry; commit; print `git log --oneline`.

## Definition of finished (all must be shown in one final message)
- `pytest -q` output with T1–T10 passing (or the exact failing ones and why).
- The table of Phase 3 step 4 and the main figure opened and described.
- `git log --oneline`. No success adjectives anywhere; evidence only.

---

# Resume prompt (any later session)

Read CLAUDE.md, LOG.md and notes/decisions.md. Tell me in ≤ 5 lines which phase and checkpoint we are at,
what the last evidence was, and what the next action is. Then continue; stop at the next checkpoint.

---

# My reply templates (what I will type at checkpoints)

- CP0: `ok` | `fix: <error>`
- CP1: `default` | `default but beta=<x>, m_t=<a>+<b>t` | `use exponential tilt on Y in [<lo>,<hi>]`
- CP2: `ok` | `show me T4 details` | `explain <test> plain English → formula → code`
- CP3: `adopt` | `try lr=<x> epochs=<k>` | `add bounded head B=<x>` | `add weight decay <x>`
- CP4: `all three` | `1 and 3` | `add: <one-line description>`
- CP5: `go` | `slide 2: <edit>`

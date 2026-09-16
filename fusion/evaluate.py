"""fusion/evaluate.py -- Phase 4: main_figure, in_sample_check, stress_tests (docs/math_fixed.md §E).

CP4 = `Default` (all three stress tests; notes/decisions.md). Every stress-test draw function
below perturbs the DGP ONLY for t>m (the years the pipeline never sees during training), so
`fusion/dgp.py`'s default DGP -- and everything trained on it in Phase 3 -- is untouched; these
functions live here, not in dgp.py, because they are evaluation-time "what if" perturbations,
not part of the fixed default DGP (docs/math_fixed.md §D vs §E).

C/C++ -> Python -> R: rejection-sampling `while` loops, same pattern as fusion/dgp.py throughout.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from fusion import dgp
from fusion.config import Config
from fusion.data import Standardizer
from fusion.estimate import mu_tilde, offset_baseline, survey_mean_baseline, bootstrap_se
from fusion.model import ThetaNet


def main_figure_data(cfg: Config, theta_net: ThetaNet, standardizer: Standardizer,
                      rng: np.random.Generator, n_boot: int = 200) -> list[dict]:
    """mu(t), E_St[Y], mu~(t)+-SE, offset baseline for t=1..M (docs/challenge_text.md §Conclusions)."""
    survey_mean_m = survey_mean_baseline(dgp.draw_survey(cfg, rng, cfg.m, cfg.n))
    mu_m_true = float(dgp.mu_true(cfg, cfg.m))

    rows = []
    for t in range(1, cfg.M + 1):
        y_t = dgp.draw_survey(cfg, rng, t, cfg.n)
        survey_mean_t = survey_mean_baseline(y_t)
        est, n_eff_frac = mu_tilde(theta_net, standardizer, y_t)
        se = bootstrap_se(theta_net, standardizer, y_t, n_boot, rng)
        rows.append({
            "t": t,
            "mu_true": float(dgp.mu_true(cfg, t)),
            "survey_mean": survey_mean_t,
            "mu_tilde": est,
            "se": se,
            "n_eff_over_n": n_eff_frac,
            "offset": offset_baseline(mu_m_true, survey_mean_m, survey_mean_t),
        })
    return rows


def in_sample_check(rows: list[dict], cfg: Config) -> list[dict]:
    """t<=m rows of main_figure_data: mu~(t) should track mu(t) closely, since P_t (hence the
    truth) IS observed in-sample there -- a sanity check the pipeline should pass trivially."""
    return [r for r in rows if r["t"] <= cfg.m]


# ---------------- Stress test 1: Assumption 1 broken (t>m only) ----------------

def draw_survey_assumption1_broken(cfg: Config, rng: np.random.Generator, t: int, n: int) -> np.ndarray:
    """pi_t(y) = sigmoid(a_t + b_t*y), b_t = beta + 0.1*(t-m), for t>m only.

    w(y,y';t) = pi_t(y)/pi_t(y') now depends on t through b_t -- Assumption 1 is violated by
    construction, growing worse as |b_t-b_m| grows with t.
    """
    a = dgp.a_t(cfg, t)
    b = cfg.beta + 0.1 * (t - cfg.m)
    out = np.empty(0)
    while out.size < n:
        y = dgp.draw_pop(cfg, rng, t, 4 * n)
        keep = rng.random(4 * n) < expit(a + b * y)
        out = np.concatenate([out, y[keep]])
    return out[:n]


def stress_test_assumption1_broken(cfg: Config, theta_net: ThetaNet, standardizer: Standardizer,
                                    rng: np.random.Generator) -> list[dict]:
    rows = []
    for t in range(cfg.m + 1, cfg.M + 1):
        y_t = draw_survey_assumption1_broken(cfg, rng, t, cfg.n)
        est, _ = mu_tilde(theta_net, standardizer, y_t)
        b_t = cfg.beta + 0.1 * (t - cfg.m)
        rows.append({"t": t, "mu_true": float(dgp.mu_true(cfg, t)), "mu_tilde": est,
                      "bias": est - float(dgp.mu_true(cfg, t)), "abs_b_drift": abs(b_t - cfg.beta)})
    return rows


# ---------------- Stress test 2: support shift (t>m only) ----------------

def draw_survey_support_shift(cfg: Config, rng: np.random.Generator, t: int, n: int, shift: float) -> np.ndarray:
    """Same selection pi_t(y), but P_{t,Y} mean shifted by `shift` (a support shift), t>m only."""
    out = np.empty(0)
    while out.size < n:
        y = rng.normal(dgp.m_t(cfg, t) + shift, cfg.sigma, 4 * n)
        keep = rng.random(4 * n) < dgp.pi_t(cfg, y, t)
        out = np.concatenate([out, y[keep]])
    return out[:n]


def stress_test_support_shift(cfg: Config, theta_net: ThetaNet, standardizer: Standardizer,
                               rng: np.random.Generator, train_y_range: tuple[float, float],
                               shift: float | None = None) -> list[dict]:
    shift = 2 * cfg.sigma if shift is None else shift
    lo, hi = train_y_range
    rows = []
    for t in range(cfg.m + 1, cfg.M + 1):
        y_t = draw_survey_support_shift(cfg, rng, t, cfg.n, shift)
        est, _ = mu_tilde(theta_net, standardizer, y_t)
        mu_true_shifted = float(dgp.mu_true(cfg, t)) + shift
        frac_outside = float(np.mean((y_t < lo) | (y_t > hi)))
        rows.append({"t": t, "mu_true_shifted": mu_true_shifted, "mu_tilde": est,
                      "bias": est - mu_true_shifted, "frac_outside_train_range": frac_outside,
                      "y_sample": y_t})
    return rows


# ---------------- Stress test 3: small n ----------------

def stress_test_small_n(cfg: Config, theta_net: ThetaNet, standardizer: Standardizer, t: int,
                         ns=(200, 500, 2000), n_seeds: int = 10) -> list[dict]:
    """theta~ held FIXED (see notes/decisions.md CP4); only the S_t draw seed varies, isolating
    estimator/sampling variance. Expect sd(mu~) ~ 1/sqrt(n_eff)."""
    rows = []
    for n in ns:
        ests, n_effs = [], []
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 * n + seed)
            y = dgp.draw_survey(cfg, rng, t, n)
            est, n_eff_frac = mu_tilde(theta_net, standardizer, y)
            ests.append(est)
            n_effs.append(n_eff_frac * n)
        ests_arr = np.array(ests)
        rows.append({"n": n, "sd_mu_tilde": float(ests_arr.std(ddof=1)),
                     "mean_n_eff": float(np.mean(n_effs))})
    return rows

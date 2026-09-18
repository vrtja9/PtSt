"""fusion/evaluate.py -- Phase 4: main_figure, in_sample_check, stress_tests (docs/math_fixed.md §E).

CP4 = `Default` (all three stress tests; notes/decisions.md). Every stress-test draw function
below perturbs the DGP ONLY for t>m (the years the pipeline never sees during training), so
`fusion/dgp.py`'s default DGP -- and everything trained on it in Phase 3 -- is untouched; these
functions live here, not in dgp.py, because they are evaluation-time "what if" perturbations,
not part of the fixed default DGP (docs/math_fixed.md §D vs §E).

C/C++ -> Python -> R: rejection-sampling `while` loops, same pattern as fusion/dgp.py throughout.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.special import expit

from fusion import dgp
from fusion.config import Config
from fusion.data import Standardizer
from fusion.estimate import mu_tilde, offset_baseline, survey_mean_baseline, bootstrap_se, oracle_estimate
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

# DECISION (2026-09-18, A3(a)): the logistic baseline slope at t=m is set to match the STRENGTH
# of the default probit DGP (Phi(x) ~= sigmoid(1.702x), so beta=0.6 -> logistic slope ~1.02~=1.0),
# not the old literal beta=1.5 -- using beta itself here would confound "change the functional
# form" with "also change the strength" (docs/math_fixed.md SS E stress test 1's caveat).
LOGISTIC_BASELINE_B = 1.0


def draw_survey_assumption1_broken(cfg: Config, rng: np.random.Generator, t: int, n: int) -> np.ndarray:
    """pi_t(y) = sigmoid(a_t + b_t*y), b_t = LOGISTIC_BASELINE_B + 0.1*(t-m), for t>m only.

    w(y,y';t) = pi_t(y)/pi_t(y') now depends on t through b_t -- Assumption 1 is violated by
    construction, growing worse as |b_t-b_m| grows with t. This tests Assumption 1 ONLY: against
    a Gaussian p_t, a logistic weight e^{-a-by} has every moment finite for any b (linear exponent
    loses to the Gaussian's quadratic one) -- it never exercises (R3)'s heavy-tail failure mode,
    which needs the quadratic probit log-weight (see stress_test_violate_r3 below).
    """
    a = dgp.a_t(cfg, t)
    b = LOGISTIC_BASELINE_B + 0.1 * (t - cfg.m)
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
        b_t = LOGISTIC_BASELINE_B + 0.1 * (t - cfg.m)
        rows.append({"t": t, "mu_true": float(dgp.mu_true(cfg, t)), "mu_tilde": est,
                      "bias": est - float(dgp.mu_true(cfg, t)), "abs_b_drift": abs(b_t - LOGISTIC_BASELINE_B)})
    return rows


# ---------------- Stress test 4: violate (R3) on purpose (default probit family) ----------------

def stress_test_violate_r3(cfg: Config, n_seeds: int = 10, n_bootstrap: int = 200,
                            ns=(80, 320, 1280), n_reps: int = 1000) -> dict:
    """Stress test 4 (docs/math_fixed.md SS E.4): violate (R3) on purpose with PROBIT selection
    (the default family) at beta*sigma >= 1, via Config(beta=1.5, allow_heavy_tails=True) -- the
    opt-in flag added in fusion/config.py exists exactly for this. Uses the ORACLE theta* (like
    T11), so this probes the DGP/estimator, not training. Reports n_eff/n CV over 10 seeds, the
    delta-SE-to-MC-sd ratio, and the fitted bias decay exponent, to check against SS G's signature.
    """
    heavy = replace(cfg, beta=1.5, allow_heavy_tails=True)
    t1 = 1

    # n_eff/n CV over 10 seeds (mirrors T11(b) / check_weight_tails.py SS D)
    n_eff_fracs = []
    for s in range(n_seeds):
        y = dgp.draw_survey(heavy, np.random.default_rng(100 + s), t1, heavy.n)
        _, n_eff_frac = mu_tilde_oracle(heavy, y)
        n_eff_fracs.append(n_eff_frac)
    n_eff_fracs = np.array(n_eff_fracs)
    n_eff_cv = float(n_eff_fracs.std() / n_eff_fracs.mean())

    # delta-SE vs MC sd (mirrors T11(c) / check_weight_tails.py SS C)
    rng = np.random.default_rng(7)
    y0 = dgp.draw_survey(heavy, rng, t1, heavy.n)
    w0 = np.exp(dgp.theta_star(heavy, y0))
    mh0 = float((w0 * y0).sum() / w0.sum())
    delta_se = float(np.sqrt((w0 ** 2 * (y0 - mh0) ** 2).sum()) / w0.sum())
    boot = np.array([oracle_estimate(heavy, dgp.draw_survey(heavy, rng, t1, heavy.n)) for _ in range(n_bootstrap)])
    se_ratio = delta_se / float(boot.std())

    # fitted bias decay exponent (mirrors docs/math_fixed.md SS G "Measured decay rate")
    mu = float(dgp.m_t(heavy, t1))
    rng2 = np.random.default_rng(21)
    n_biases = []
    for n in ns:
        est = np.array([oracle_estimate(heavy, dgp.draw_survey(heavy, rng2, t1, n)) for _ in range(n_reps)])
        n_biases.append(float(n * (est.mean() - mu)))
    biases = [nb / n for nb, n in zip(n_biases, ns)]
    exponents = [float(np.log(biases[i] / biases[i + 1]) / np.log(ns[i + 1] / ns[i]))
                 for i in range(len(ns) - 1)]

    a = 1 + 1 / (heavy.beta ** 2 * heavy.sigma ** 2)
    predicted_exponent = 1 - 1 / a
    return {
        "beta": heavy.beta, "sigma": heavy.sigma, "a": a,
        "n_eff_cv": n_eff_cv, "n_eff_mean": float(n_eff_fracs.mean()), "n_eff_sd": float(n_eff_fracs.std()),
        "delta_se": delta_se, "mc_sd": float(boot.std()), "se_ratio": se_ratio,
        "ns": ns, "n_biases": n_biases, "biases": biases, "exponents": exponents,
        "predicted_exponent": predicted_exponent,
    }


def mu_tilde_oracle(cfg: Config, y_raw: np.ndarray):
    """Oracle-theta* analogue of estimate.mu_tilde (Step 4), for stress tests that need a
    trained-net-free n_eff (theta_net not yet fit, or deliberately bypassed, e.g. stress test 4)."""
    w = np.exp(dgp.theta_star(cfg, y_raw))
    est = float((w * y_raw).sum() / w.sum())
    n_eff_over_n = float((w.sum() ** 2 / (w ** 2).sum()) / len(y_raw))
    return est, n_eff_over_n


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

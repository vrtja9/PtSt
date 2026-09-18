"""tests/test_t11_weight_tails.py -- T11 (a)-(d): weight-moment rule (R3), docs/math_fixed.md §G.

Uses the ORACLE theta* (fusion.dgp.theta_star), so this tests the DGP and the estimator, not
training. C/C++ -> Python -> R: quad() is adaptive Gauss-Kronrod numerical integration (GSL's
gsl_integration_qag in C; integrate() in R); the Monte-Carlo loops below obtain the same
quantities by sampling instead of quadrature.
"""
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

from fusion import dgp
from fusion.config import Config


def _oracle_w(cfg: Config, y: np.ndarray) -> np.ndarray:
    return np.exp(dgp.theta_star(cfg, y))


def _hajek_oracle(cfg: Config, y: np.ndarray) -> float:
    w = _oracle_w(cfg, y)
    return float((w * y).sum() / w.sum())


def test_T11a_weight_moment_rule_assertion():
    """(a) beta*sigma < 1/sqrt(2), assertion message quotes a = 1+1/(beta^2 sigma^2) and (R3)."""
    cfg = Config()
    a = 1 + 1 / (cfg.beta ** 2 * cfg.sigma ** 2)
    assert cfg.beta * cfg.sigma < 1 / np.sqrt(2), (
        f"(R3) violated: beta*sigma={cfg.beta * cfg.sigma:.4f} >= 1/sqrt(2); "
        f"tail index a={a:.3f} <= 3 (docs/math_fixed.md SS G)"
    )
    print(f"T11(a): beta={cfg.beta} sigma={cfg.sigma} beta*sigma={cfg.beta*cfg.sigma:.4f} a={a:.3f}")


def test_T11b_n_eff_stability_over_seeds():
    """(b) n_eff/n at t=1, n=2000, over 10 data seeds: CV(sd/mean) < 0.20."""
    cfg = Config()
    vals = []
    for s in range(10):
        y = dgp.draw_survey(cfg, np.random.default_rng(100 + s), 1, cfg.n)
        w = _oracle_w(cfg, y)
        vals.append((w.sum() ** 2 / (w ** 2).sum()) / cfg.n)
    vals = np.array(vals)
    cv = vals.std() / vals.mean()
    print(f"T11(b): mean={vals.mean():.3f} sd={vals.std():.3f} CV={cv:.3f} "
          f"min={vals.min():.3f} max={vals.max():.3f}")
    assert cv < 0.20, f"n_eff/n CV={cv:.3f} >= 0.20 across seeds"


def test_T11c_se_calibration():
    """(c) delta-method SE (one sample) vs MC sd of mu~ (200 fresh samples), ratio in [0.75,1.33]."""
    cfg = Config()
    rng = np.random.default_rng(7)
    y = dgp.draw_survey(cfg, rng, 1, cfg.n)
    w = _oracle_w(cfg, y)
    mh = (w * y).sum() / w.sum()
    se = np.sqrt((w ** 2 * (y - mh) ** 2).sum()) / w.sum()
    reps = np.array([_hajek_oracle(cfg, dgp.draw_survey(cfg, rng, 1, cfg.n)) for _ in range(200)])
    ratio = se / reps.std()
    print(f"T11(c): delta_SE={se:.4f} MC_sd={reps.std():.4f} ratio={ratio:.2f}")
    assert 0.75 <= ratio <= 1.33, f"SE/MC-sd ratio={ratio:.2f} outside [0.75, 1.33]"


def test_T11d_finite_sample_bias_report_only():
    """(d) report-only: n*(mean mu~ - mu) at n=80,320,1280 (1000 reps) vs the predicted constant
    -E_S[w^2(Y-mu)]/E_S[w]^2 from numerical integration. No assertion (per Edit H)."""
    cfg = Config()
    mu = float(dgp.m_t(cfg, 1))
    am, a1 = dgp.a_t(cfg, cfg.m), dgp.a_t(cfg, 1)

    def wf(y):
        return norm.cdf(am) / max(norm.cdf(cfg.beta * y), 1e-300)

    def sf(y):
        return norm.cdf(cfg.beta * y) * norm.pdf(y, mu, cfg.sigma) / norm.cdf(a1)

    Ew = quad(lambda y: wf(y) * sf(y), -30, 30, limit=400)[0]
    Ew2 = quad(lambda y: wf(y) ** 2 * (y - mu) * sf(y), -30, 30, limit=400)[0]
    predicted = -Ew2 / Ew ** 2
    print(f"T11(d): predicted n*bias = {predicted:.3f}")

    rng = np.random.default_rng(21)
    for n in (80, 320, 1280):
        est = np.array([_hajek_oracle(cfg, dgp.draw_survey(cfg, rng, 1, n)) for _ in range(1000)])
        print(f"T11(d): n={n:5d}  n*bias={n * (est.mean() - mu):+.2f} "
              f"(MC SE {n * est.std() / np.sqrt(1000):.2f})  sd*sqrt(n)={est.std() * np.sqrt(n):.3f}")

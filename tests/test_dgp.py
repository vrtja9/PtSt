"""tests/test_dgp.py -- T1, T2, T3 (docs/CLAUDE.md §5; Phase 1 of PROMPT_kickoff.md).

C/C++ -> Python -> R: `assert(fabs(x-y)<tol)` -> `assert abs(x-y) < tol` -> `stopifnot(abs(x-y)<tol)`.
SE throughout = sample_std / sqrt(n) (CLT), stated per-call as the kickoff prompt requires.
"""
import numpy as np
from scipy.stats import norm

from fusion.config import Config
from fusion import dgp


def test_T1_closed_forms_match_simulation():
    """|mean of 200k survey draws - ES_closed(t)| < 3*SE for t in {1, m, M}."""
    cfg = Config()
    rng = np.random.default_rng(cfg.seed_data)
    for t in (1, cfg.m, cfg.M):
        y = dgp.draw_survey(cfg, rng, t, 200_000)
        se = y.std(ddof=1) / np.sqrt(len(y))
        diff = abs(y.mean() - dgp.ES_closed(cfg, t))
        assert diff < 3 * se, f"t={t}: |diff|={diff:.5f} !< 3*SE={3*se:.5f}"


def test_T2_kept_sample_matches_numeric_integration():
    """For t=1: mean/variance of 200k accepted draws vs numeric integration of s_t on a grid."""
    cfg = Config()
    rng = np.random.default_rng(cfg.seed_data)
    t = 1
    y = dgp.draw_survey(cfg, rng, t, 200_000)

    grid = np.linspace(dgp.m_t(cfg, t) - 8 * cfg.sigma, dgp.m_t(cfg, t) + 8 * cfg.sigma, 20_001)
    s = norm.cdf(cfg.beta * grid) * norm.pdf(grid, dgp.m_t(cfg, t), cfg.sigma)
    s = s / np.trapezoid(s, grid)  # s_t ∝ pi_t * p_t (Step 1), normalise numerically
    mean_num = np.trapezoid(grid * s, grid)
    var_num = np.trapezoid((grid - mean_num) ** 2 * s, grid)

    se = y.std(ddof=1) / np.sqrt(len(y))
    assert abs(y.mean() - mean_num) < 3 * se
    assert abs(y.var(ddof=1) - var_num) / var_num < 0.05


def test_T3_assumption1_ratio_independent_of_t():
    """pi_t(y,t)/pi_t(y',t) equal for t=1 and t=m to 1e-12, for random y,y'."""
    cfg = Config()
    rng = np.random.default_rng(12345)
    y1 = rng.normal(size=50)
    y2 = rng.normal(size=50)
    ratio_t1 = dgp.pi_t(cfg, y1, 1) / dgp.pi_t(cfg, y2, 1)
    ratio_tm = dgp.pi_t(cfg, y1, cfg.m) / dgp.pi_t(cfg, y2, cfg.m)
    assert np.max(np.abs(ratio_t1 - ratio_tm)) < 1e-12

"""fusion/dgp.py -- the data-generating process (docs/math_fixed.md §D, Step 10).

This module implements the ONLY open modelling choice (CLAUDE.md §2.2); every function here is a
pure function of `cfg: Config` plus its own arguments, so a DGP change is entirely a
`fusion/config.py` edit (see the `# DECISION (CP1...)` comment there) -- no code in this file
should need to change for a parameter change, only for a genuinely different DGP family
(docs/math_fixed.md §D lists three named alternatives, each "1-3 lines to swap").

C/C++ -> Python -> R: a `for` loop filling a fixed-size `double y[n]` one draw at a time ->
vectorised numpy draws + a boolean mask -> `rnorm`/`ifelse` on whole vectors. Kept throughout.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from fusion.config import Config


def m_t(cfg: Config, t) -> np.ndarray:
    """P_{t,Y} mean. Step 10 (DGP): m_t(t) = m_t_intercept + m_t_slope * t."""
    return cfg.m_t_intercept + cfg.m_t_slope * np.asarray(t, dtype=float)


def c_t(cfg: Config, t) -> np.ndarray:
    """Overall response level c_t in (0, 1]; never enters S_t (Step 3: c_t cancels)."""
    return cfg.c_t_intercept + cfg.c_t_slope * np.asarray(t, dtype=float)


def pi_t(cfg: Config, y, t):
    """Step 3 (factor form): pi_t(y) = c_t * g(y), g(y) = Phi(beta*y).

    Assumption 1 holds because pi_t(y)/pi_t(y') = Phi(beta*y)/Phi(beta*y') has no t (test T3).
    """
    return c_t(cfg, t) * norm.cdf(cfg.beta * np.asarray(y, dtype=float))


def draw_pop(cfg: Config, rng: np.random.Generator, t: int, n: int) -> np.ndarray:
    """Step 10: draw n i.i.d. Y ~ P_{t,Y} = N(m_t(t), sigma^2).

    C/C++ -> Python -> R: `for(i<n) y[i]=m+sigma*gauss()` -> `rng.normal(m,sigma,n)` -> `rnorm(n,m,sigma)`.
    """
    return rng.normal(m_t(cfg, t), cfg.sigma, n)


def draw_survey(cfg: Config, rng: np.random.Generator, t: int, n: int) -> np.ndarray:
    """Step 1 (Bayes)/Step 10: draw Y~P_{t,Y}, keep R=1 ~ Bern(pi_t(Y)) until n kept.

    Rejection sampling over-draws by 4x per round and re-draws the shortfall; this IS
    "conditioning on R=1" (docs/challenge_text.md), i.e. sampling s_t(y) = pi_t(y) p_t(y) / rho_bar_t.
    C/C++ -> Python -> R: `while(count<n){...}` scalar loop -> block-wise boolean mask,
    re-loop on shortfall -> `while(length(out)<n) out <- c(out, y[keep])`.
    """
    out = np.empty(0)
    while out.size < n:
        y = draw_pop(cfg, rng, t, 4 * n)
        keep = rng.random(4 * n) < pi_t(cfg, y, t)
        out = np.concatenate([out, y[keep]])
    return out[:n]


def a_t(cfg: Config, t) -> np.ndarray:
    """Step 10 closed form: a_t := beta*m_t(t) / sqrt(1 + beta^2*sigma^2)."""
    b, s = cfg.beta, cfg.sigma
    return b * m_t(cfg, t) / np.sqrt(1 + b * b * s * s)


def ES_closed(cfg: Config, t) -> np.ndarray:
    """Step 10 closed form: E_{S_t}[Y] via Stein's lemma (E[(Y-m)h(Y)] = sigma^2 E[h'(Y)])."""
    b, s = cfg.beta, cfg.sigma
    at = a_t(cfg, t)
    return m_t(cfg, t) + b * s * s * norm.pdf(at) / (np.sqrt(1 + b * b * s * s) * norm.cdf(at))


def mu_true(cfg: Config, t) -> np.ndarray:
    """mu(t) := E_{P_t}[Y] = E_{P_{t,Y}}[Y] = m_t(t) for this Gaussian DGP."""
    return m_t(cfg, t)


def theta_star(cfg: Config, y) -> np.ndarray:
    """Step 3 (factor form), anchored so alpha*(m)=0: theta*(y) = log Phi(a_m) - log Phi(beta*y)."""
    am = a_t(cfg, cfg.m)
    return np.log(norm.cdf(am)) - np.log(norm.cdf(cfg.beta * np.asarray(y, dtype=float)))


def alpha_star(cfg: Config, t) -> np.ndarray:
    """Step 3: alpha*(t) = log Phi(a_t) - log Phi(a_m); alpha*(m) == 0 by construction."""
    return np.log(norm.cdf(a_t(cfg, t))) - np.log(norm.cdf(a_t(cfg, cfg.m)))


def density_grid(cfg: Config, t: int, npts: int = 400) -> dict:
    """Step 1 (Bayes): s_t(y) = pi_t(y) p_t(y) / rho_bar_t, rho_bar_t = c_t*Phi(a_t) (Step 3).

    Returns a dict of arrays on a fixed y-grid spanning +/- 6 sigma around m_t(t), for the
    Phase 1 (a) figure (p_t, s_t, pi_t overlaid) -- rho_bar_t itself is a scalar, used in (b).
    """
    y = np.linspace(m_t(cfg, t) - 6 * cfg.sigma, m_t(cfg, t) + 6 * cfg.sigma, npts)
    p = norm.pdf(y, m_t(cfg, t), cfg.sigma)
    pi = pi_t(cfg, y, t)
    rho_bar_t = c_t(cfg, t) * norm.cdf(a_t(cfg, t))
    s = pi * p / rho_bar_t
    return {"y": y, "p_t": p, "s_t": s, "pi_t": pi, "rho_bar_t": float(rho_bar_t)}

"""fusion/estimate.py -- Step 4 estimator, baselines, bootstrap SE, seeds_table.

C/C++ -> Python -> R: a manual weighted-sum accumulator loop -> numpy vectorised sum/dot ->
`weighted.mean()`-style vector ops.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from fusion import dgp
from fusion.config import Config
from fusion.data import Standardizer, build_dataset, stratified_split
from fusion.model import ThetaNet
from fusion.train import train


def mu_tilde(theta_net: ThetaNet, standardizer: Standardizer, y_raw: np.ndarray):
    """Step 4 (Hajek/self-normalised estimator): mu~(t) = sum(e^theta*y)/sum(e^theta); alpha not
    needed (cancels between numerator and denominator). Returns (estimate, n_eff/n)."""
    ys = standardizer.transform(y_raw)
    with torch.no_grad():
        theta = theta_net(torch.as_tensor(ys)).numpy()
    w = np.exp(theta)
    est = float((w * y_raw).sum() / w.sum())
    n_eff_over_n = float((w.sum() ** 2 / (w ** 2).sum()) / len(w))
    return est, n_eff_over_n


def survey_mean_baseline(y_raw: np.ndarray) -> float:
    """Baseline 1: the naive (unweighted) survey mean E_St[Y]-hat."""
    return float(np.mean(y_raw))


def offset_baseline(mu_m_true: float, survey_mean_m: float, survey_mean_t: float) -> float:
    """Baseline 2: last-known-offset = mu(m) + (ybar_{S_t} - ybar_{S_m})."""
    return mu_m_true + (survey_mean_t - survey_mean_m)


def oracle_estimate(cfg: Config, y_raw: np.ndarray) -> float:
    """Baseline 3: mu~ with the TRUE theta* plugged in instead of the fitted theta~."""
    theta = dgp.theta_star(cfg, y_raw)
    w = np.exp(theta)
    return float((w * y_raw).sum() / w.sum())


def bootstrap_se(theta_net: ThetaNet, standardizer: Standardizer, y_raw: np.ndarray,
                  n_boot: int, rng: np.random.Generator) -> float:
    """Bootstrap SE of mu~(t) with theta~ FIXED: resample the S_t draw only, never refit theta."""
    n = len(y_raw)
    ests = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ests[b], _ = mu_tilde(theta_net, standardizer, y_raw[idx])
    return float(ests.std(ddof=1))


def seeds_table(cfg: Config, seeds=(0, 1, 2, 3, 4)) -> list[dict]:
    """mu~(t) variability from seed_data vs seed_model, varied SEPARATELY (the other held at
    cfg's own value) -- PROMPT_kickoff.md Phase 3 step 3."""
    rows = []
    for kind in ("seed_data", "seed_model"):
        for s in seeds:
            cfg_s = replace(cfg, **{kind: s})
            data_rng = np.random.default_rng(cfg_s.seed_data)
            T, Z, Y = build_dataset(data_rng, cfg_s)
            tr_mask, va_mask = stratified_split(data_rng, T, Z, cfg_s.val_frac)
            result = train(cfg_s, T, Z, Y, tr_mask, va_mask, seed_model=cfg_s.seed_model)
            row = {"varied": kind, "seed": s}
            for t in range(cfg_s.m + 1, cfg_s.M + 1):
                y_t = dgp.draw_survey(cfg_s, data_rng, t, cfg_s.n)
                est, _ = mu_tilde(result.theta_net, result.standardizer, y_t)
                row[f"mu_tilde_t{t}"] = est
            rows.append(row)
    return rows

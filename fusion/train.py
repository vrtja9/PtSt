"""fusion/train.py -- Adam training, early stopping (best-checkpoint), FOC diagnostic,
theta_grid_plot data (docs/math_fixed.md §B Steps 8-9; PROMPT_kickoff.md Phase 3).

C/C++ -> Python -> R: hand-written Adam moment buffers updated in a `for` loop
(fusion_numpy.py's adam_step) -> `torch.optim.Adam` -> R's own `optim`-family wrapper, same math.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from fusion.config import Config
from fusion.data import Standardizer, iterate_batches
from fusion.loss import fusion_loss
from fusion.model import AlphaTable, ThetaNet, ratio_model


@dataclass
class TrainResult:
    theta_net: ThetaNet
    alpha_table: AlphaTable
    standardizer: Standardizer
    history: dict
    best_epoch: int
    best_val_loss: float


def train(cfg: Config, T: np.ndarray, Z: np.ndarray, Y: np.ndarray,
          train_mask: np.ndarray, val_mask: np.ndarray, seed_model: int | None = None) -> TrainResult:
    """Adam over stratified train/val masks (fusion.data.stratified_split), weight decay on
    ThetaNet only (CLAUDE.md §2.4), early stopping = keep the state_dict with the lowest val loss.
    """
    seed_model = cfg.seed_model if seed_model is None else seed_model
    torch.manual_seed(seed_model)          # seeds ThetaNet's own nn.init calls
    shuffle_rng = np.random.default_rng(seed_model)   # seed_model also drives batch shuffling

    standardizer = Standardizer.fit(Y[train_mask])
    Ys_t = torch.as_tensor(standardizer.transform(Y))
    T_t = torch.as_tensor(T, dtype=torch.int64)
    Z_t = torch.as_tensor(Z)

    theta_net = ThetaNet(cfg.H)
    alpha_table = AlphaTable(cfg.m)
    opt = torch.optim.Adam([
        {"params": theta_net.parameters(), "weight_decay": cfg.wd},
        {"params": alpha_table.parameters(), "weight_decay": 0.0},
    ], lr=cfg.lr)

    idx_tr = np.where(train_mask)[0]
    idx_va = np.where(val_mask)[0]

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, best_epoch = float("inf"), None, -1

    for epoch in range(cfg.epochs):
        for b in iterate_batches(shuffle_rng, idx_tr, cfg.batch):
            opt.zero_grad()
            r = ratio_model(theta_net, alpha_table, Ys_t[b], T_t[b])
            fusion_loss(r, Z_t[b]).backward()
            opt.step()
        with torch.no_grad():
            train_loss = fusion_loss(
                ratio_model(theta_net, alpha_table, Ys_t[idx_tr], T_t[idx_tr]), Z_t[idx_tr]
            ).item()
            val_loss = fusion_loss(
                ratio_model(theta_net, alpha_table, Ys_t[idx_va], T_t[idx_va]), Z_t[idx_va]
            ).item()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = (copy.deepcopy(theta_net.state_dict()), copy.deepcopy(alpha_table.state_dict()))

    theta_net.load_state_dict(best_state[0])
    alpha_table.load_state_dict(best_state[1])
    return TrainResult(theta_net, alpha_table, standardizer, history, best_epoch, best_val)


def foc_diagnostic(cfg: Config, theta_net: ThetaNet, alpha_table: AlphaTable,
                    T: np.ndarray, Z: np.ndarray, Y: np.ndarray, standardizer: Standardizer) -> dict:
    """Step 8: mean_i e^{theta+alpha(t)} over survey (z=0) rows of year t; ~=1 at a stationary point."""
    Ys = standardizer.transform(Y)
    out = {}
    with torch.no_grad():
        for t in range(1, cfg.m + 1):
            sel = (T == t) & (Z == 0.0)
            y_sel = torch.as_tensor(Ys[sel])
            t_sel = torch.as_tensor(T[sel], dtype=torch.int64)
            r = ratio_model(theta_net, alpha_table, y_sel, t_sel)
            out[t] = torch.exp(r).mean().item()
    return out


def theta_grid_data(cfg: Config, theta_net: ThetaNet, standardizer: Standardizer, npts: int = 400) -> dict:
    """theta_tilde vs theta_star on a y-grid spanning ALL S_t supports, t=1..M (not just t<=m)."""
    from fusion import dgp

    lo = min(float(dgp.m_t(cfg, t)) - 6 * cfg.sigma for t in range(1, cfg.M + 1))
    hi = max(float(dgp.m_t(cfg, t)) + 6 * cfg.sigma for t in range(1, cfg.M + 1))
    y_grid = np.linspace(lo, hi, npts)
    with torch.no_grad():
        theta_hat = theta_net(torch.as_tensor(standardizer.transform(y_grid))).numpy()
    return {"y": y_grid, "theta_hat": theta_hat, "theta_star": dgp.theta_star(cfg, y_grid)}

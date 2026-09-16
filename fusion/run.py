"""fusion/run.py -- CLI entry point: python -m fusion.run --phase {1,2,3,4,5}.

C/C++ -> Python -> R: a `switch(phase)` dispatching to `void phaseN()` -> a dict of callables
keyed by phase number -> a named `list` of closures indexed by `[[phase]]`.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fusion.config import Config, git_commit_hash
from fusion import dgp
from fusion.data import build_dataset, stratified_split
from fusion.train import train, foc_diagnostic, theta_grid_data
from fusion.estimate import mu_tilde, survey_mean_baseline, offset_baseline, oracle_estimate, bootstrap_se

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")


def _save_with_config(fig, name: str, cfg: Config) -> str:
    """CLAUDE.md §2.4: every saved figure ships with the config JSON + git commit hash."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    stem = f"{name}_{cfg.hash()}"
    png_path = os.path.join(FIGURES_DIR, f"{stem}.png")
    json_path = os.path.join(FIGURES_DIR, f"{stem}.json")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    meta = {"config": json.loads(cfg.to_json()), "commit": git_commit_hash(), "figure": stem}
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return png_path


def phase1_figures(cfg: Config) -> list[str]:
    """Phase 1 step 4: (a) p_t,s_t,pi_t overlaid for t in {1,m,M}; (b) mu(t), E_St[Y], rho_bar_t vs t."""
    saved = []

    # (a) densities + selection probability for three representative years
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, t in zip(axes, (1, cfg.m, cfg.M)):
        g = dgp.density_grid(cfg, t)
        ax.plot(g["y"], g["p_t"], label="p_t (P_t,Y density)")
        ax.plot(g["y"], g["s_t"], label="s_t (survey density)")
        ax2 = ax.twinx()
        ax2.plot(g["y"], g["pi_t"], color="green", linestyle="--", label="pi_t (P[R=1|Y=y])")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="lower right", fontsize=7)
        ax.set_title(f"t={t}")
        ax.set_xlabel("y")
    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Phase 1(a): p_t, s_t (left axis) and pi_t (right axis, dashed)")
    saved.append(_save_with_config(fig, "phase1a_densities", cfg))
    plt.close(fig)

    # (b) mu(t), E_{S_t}[Y], rho_bar_t vs t, t = 1..M
    ts = np.arange(1, cfg.M + 1)
    mu = dgp.mu_true(cfg, ts)
    es = dgp.ES_closed(cfg, ts)
    rho = np.array([dgp.density_grid(cfg, int(t))["rho_bar_t"] for t in ts])

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(ts, mu, marker="o", label="mu(t) = E_Pt[Y]")
    ax1.plot(ts, es, marker="s", label="E_St[Y] (survey mean, closed form)")
    ax1.axvline(cfg.m, color="gray", linestyle=":", label="t=m (P_t stops)")
    ax1.set_xlabel("t")
    ax1.set_ylabel("Y units")
    ax1.legend(loc="upper left", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(ts, rho, marker="^", color="red", label="rho_bar_t = P[R=1]")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("rho_bar_t")
    ax2.legend(loc="upper right", fontsize=8)
    fig.suptitle("Phase 1(b): mu(t) vs E_St[Y] vs response rate rho_bar_t")
    saved.append(_save_with_config(fig, "phase1b_mu_vs_survey_vs_rho", cfg))
    plt.close(fig)

    return saved


def phase3_run(cfg: Config):
    """Phase 3 step 2: first training run + validation curve, FOC table, alpha~ vs alpha*,
    theta~ vs theta* figure. Returns (train_result, foc, saved_figure_paths)."""
    data_rng = np.random.default_rng(cfg.seed_data)
    T, Z, Y = build_dataset(data_rng, cfg)
    tr_mask, va_mask = stratified_split(data_rng, T, Z, cfg.val_frac)
    result = train(cfg, T, Z, Y, tr_mask, va_mask)
    foc = foc_diagnostic(cfg, result.theta_net, result.alpha_table, T, Z, Y, result.standardizer)

    saved = []

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result.history["train_loss"], label="train")
    ax.plot(result.history["val_loss"], label="val")
    ax.axvline(result.best_epoch, color="gray", linestyle=":", label=f"best (epoch {result.best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("fusion_loss")
    ax.legend(fontsize=8)
    fig.suptitle("Phase 3: validation curve")
    saved.append(_save_with_config(fig, "phase3_val_curve", cfg))
    plt.close(fig)

    ts = np.arange(1, cfg.m + 1)
    alpha_hat = result.alpha_table.forward().detach().numpy()
    alpha_star = dgp.alpha_star(cfg, ts)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ts, alpha_hat, marker="o", label="alpha~(t)")
    ax.plot(ts, alpha_star, marker="s", label="alpha*(t)")
    ax.set_xlabel("t")
    ax.legend(fontsize=8)
    fig.suptitle("Phase 3: alpha~ vs alpha*")
    saved.append(_save_with_config(fig, "phase3_alpha_hat_vs_star", cfg))
    plt.close(fig)

    grid = theta_grid_data(cfg, result.theta_net, result.standardizer)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid["y"], grid["theta_hat"], label="theta~(y)")
    ax.plot(grid["y"], grid["theta_star"], label="theta*(y)")
    ax.set_xlabel("y")
    ax.legend(fontsize=8)
    fig.suptitle("Phase 3: theta~ vs theta* (y-grid spans all S_t supports, t=1..M)")
    saved.append(_save_with_config(fig, "phase3_theta_hat_vs_star", cfg))
    plt.close(fig)

    return result, foc, saved, data_rng, T, Z, Y


def phase3_table(cfg: Config, result, T, Z, Y, data_rng) -> list[dict]:
    """Phase 3 step 3-4: table for t=m+1..M (mu, survey mean, mu~+-SE, n_eff/n, offset, oracle)."""
    survey_mean_m = float(Y[(T == cfg.m) & (Z == 0.0)].mean())
    mu_m_true = float(dgp.mu_true(cfg, cfg.m))

    rows = []
    for t in range(cfg.m + 1, cfg.M + 1):
        y_t = dgp.draw_survey(cfg, data_rng, t, cfg.n)
        survey_mean_t = survey_mean_baseline(y_t)
        est, n_eff_frac = mu_tilde(result.theta_net, result.standardizer, y_t)
        se = bootstrap_se(result.theta_net, result.standardizer, y_t, n_boot=200, rng=data_rng)
        rows.append({
            "t": t,
            "mu_true": float(dgp.mu_true(cfg, t)),
            "survey_mean": survey_mean_t,
            "mu_tilde": est,
            "se": se,
            "n_eff_over_n": n_eff_frac,
            "offset": offset_baseline(mu_m_true, survey_mean_m, survey_mean_t),
            "oracle": oracle_estimate(cfg, y_t),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    cfg = Config()
    if args.phase == 1:
        for path in phase1_figures(cfg):
            print("saved:", path)
    elif args.phase == 3:
        result, foc, saved, data_rng, T, Z, Y = phase3_run(cfg)
        print("best_epoch:", result.best_epoch, "best_val_loss:", result.best_val_loss)
        print("FOC:", {k: round(v, 3) for k, v in foc.items()})
        for path in saved:
            print("saved:", path)
        rows = phase3_table(cfg, result, T, Z, Y, data_rng)
        print(f"\n{'t':>2} {'mu_true':>8} {'survey_mean':>11} {'mu_tilde':>9} {'SE':>7} "
              f"{'n_eff/n':>7} {'offset':>8} {'oracle':>8}")
        for r in rows:
            print(f"{r['t']:>2} {r['mu_true']:>8.3f} {r['survey_mean']:>11.3f} {r['mu_tilde']:>9.3f} "
                  f"{r['se']:>7.4f} {r['n_eff_over_n']:>7.2f} {r['offset']:>8.3f} {r['oracle']:>8.3f}")
    else:
        raise NotImplementedError(f"phase {args.phase} not implemented yet")


if __name__ == "__main__":
    main()

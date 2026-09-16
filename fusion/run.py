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
from fusion import evaluate

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


def phase4_run(cfg: Config):
    """Phase 4: main_figure, in_sample_check, the three CP4 stress tests, one figure each."""
    data_rng = np.random.default_rng(cfg.seed_data)
    T, Z, Y = build_dataset(data_rng, cfg)
    tr_mask, va_mask = stratified_split(data_rng, T, Z, cfg.val_frac)
    result = train(cfg, T, Z, Y, tr_mask, va_mask)
    train_y_range = (float(Y.min()), float(Y.max()))

    saved = []

    # main_figure: mu(t), E_St[Y], mu~(t)+-SE, offset baseline, t>m shaded
    rows = evaluate.main_figure_data(cfg, result.theta_net, result.standardizer, data_rng)
    ts = [r["t"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axvspan(cfg.m + 0.5, cfg.M + 0.5, color="gray", alpha=0.15, label="t>m (extrapolation)")
    ax.plot(ts, [r["mu_true"] for r in rows], marker="o", label="mu(t) true")
    ax.plot(ts, [r["survey_mean"] for r in rows], marker="s", label="E_St[Y] survey mean")
    ax.errorbar(ts, [r["mu_tilde"] for r in rows], yerr=[r["se"] for r in rows],
                marker="^", label="mu~(t) +- bootstrap SE", capsize=3)
    ax.plot(ts, [r["offset"] for r in rows], marker="v", linestyle="--", label="offset baseline")
    ax.set_xlabel("t")
    ax.legend(fontsize=7)
    fig.suptitle("Phase 4: main figure -- mu(t) vs E_St[Y] vs mu~(t) vs offset baseline")
    saved.append(_save_with_config(fig, "phase4_main_figure", cfg))
    plt.close(fig)
    in_sample = evaluate.in_sample_check(rows, cfg)

    # Stress test 1: Assumption 1 broken
    st1 = evaluate.stress_test_assumption1_broken(cfg, result.theta_net, result.standardizer, data_rng)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([r["abs_b_drift"] for r in st1], [r["bias"] for r in st1], marker="o")
    ax.set_xlabel("|b_t - beta| (selection-slope drift)")
    ax.set_ylabel("mu~(t) - mu(t) (bias)")
    fig.suptitle("Stress 1: Assumption 1 broken -- bias vs selection drift")
    saved.append(_save_with_config(fig, "phase4_stress1_assumption1", cfg))
    plt.close(fig)

    # Stress test 2: support shift
    st2 = evaluate.stress_test_support_shift(cfg, result.theta_net, result.standardizer, data_rng, train_y_range)
    fig, ax = plt.subplots(figsize=(6, 4))
    all_y = np.concatenate([r["y_sample"] for r in st2])
    ax.hist(all_y, bins=60, alpha=0.7)
    ax.axvline(train_y_range[0], color="red", linestyle="--", label="training y-range")
    ax.axvline(train_y_range[1], color="red", linestyle="--")
    overall_frac = float(np.mean((all_y < train_y_range[0]) | (all_y > train_y_range[1])))
    ax.set_xlabel("y")
    ax.legend(fontsize=8)
    fig.suptitle(f"Stress 2: support shift (+2sigma), {overall_frac:.1%} of t>m survey values outside training range")
    saved.append(_save_with_config(fig, "phase4_stress2_support_shift", cfg))
    plt.close(fig)
    for r in st2:
        del r["y_sample"]  # not JSON/print friendly; frac_outside_train_range already captures it

    # Stress test 3: small n
    st3 = evaluate.stress_test_small_n(cfg, result.theta_net, result.standardizer, t=cfg.M)
    fig, ax = plt.subplots(figsize=(5, 4))
    ns = [r["n"] for r in st3]
    sds = [r["sd_mu_tilde"] for r in st3]
    ax.loglog(ns, sds, marker="o", label="observed sd(mu~)")
    ref = sds[0] * np.sqrt(ns[0] / np.array(ns))
    ax.loglog(ns, ref, linestyle="--", label="1/sqrt(n) reference")
    ax.set_xlabel("n")
    ax.set_ylabel(f"sd(mu~(t={cfg.M})) over {10} seeds")
    ax.legend(fontsize=8)
    fig.suptitle("Stress 3: small n -- sd(mu~) vs n")
    saved.append(_save_with_config(fig, "phase4_stress3_small_n", cfg))
    plt.close(fig)

    return {"in_sample": in_sample, "st1": st1, "st2": st2, "st3": st3, "saved": saved}


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
    elif args.phase == 4:
        out = phase4_run(cfg)
        print("in_sample_check (t<=m):")
        for r in out["in_sample"]:
            print(f"  t={r['t']}: mu_true={r['mu_true']:.3f} mu_tilde={r['mu_tilde']:.3f} "
                  f"survey_mean={r['survey_mean']:.3f}")
        print("stress 1 (Assumption 1 broken):")
        for r in out["st1"]:
            print(f"  t={r['t']}: mu_true={r['mu_true']:.3f} mu_tilde={r['mu_tilde']:.3f} "
                  f"bias={r['bias']:+.3f} |b_drift|={r['abs_b_drift']:.2f}")
        print("stress 2 (support shift):")
        for r in out["st2"]:
            print(f"  t={r['t']}: mu_true_shifted={r['mu_true_shifted']:.3f} mu_tilde={r['mu_tilde']:.3f} "
                  f"bias={r['bias']:+.3f} frac_outside_train_range={r['frac_outside_train_range']:.2%}")
        print("stress 3 (small n):")
        for r in out["st3"]:
            print(f"  n={r['n']}: sd(mu_tilde)={r['sd_mu_tilde']:.4f} mean_n_eff={r['mean_n_eff']:.1f}")
        for path in out["saved"]:
            print("saved:", path)
    else:
        raise NotImplementedError(f"phase {args.phase} not implemented yet")


if __name__ == "__main__":
    main()

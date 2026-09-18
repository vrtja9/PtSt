"""fusion/build_deck.py -- collect every slide number programmatically, then render the deck.

HARD RULE (2026-09-18): every number on a slide is read from the repo at build time (Config
fields, a live re-run of the pipeline, or a live pytest subprocess) and written to
slides/slide_data.json; render_deck() then only interpolates those values into static bullet
prose -- no numeric literal is hand-typed into a bullet string. The only hand-written numbers
anywhere in this file are pure mathematical constants (sqrt(2), and the logit/probit slope
constant 1.702), never a result.

Run:  python -m fusion.build_deck   (from any cwd via `python -m`, or `python fusion/build_deck.py`
via the shim below)
C/C++ -> Python -> R: a driver/main() that fixes the run configuration and asserts every value it
needs exists before using it -- "fail loudly" -- rather than defaulting a missing field to 0/NA.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import subprocess
import time

import numpy as np
import torch
from scipy.stats import norm

import fusion_numpy as npref
from fusion import dgp
from fusion.config import Config
from fusion.data import build_dataset, stratified_split
from fusion.estimate import mu_tilde, oracle_estimate, survey_mean_baseline, offset_baseline, bootstrap_se
from fusion.evaluate import stress_test_violate_r3
from fusion.loss import fusion_loss
from fusion.model import AlphaTable, ThetaNet, ratio_model
from fusion.train import foc_diagnostic, theta_grid_data, train

REPO_ROOT = Path(__file__).resolve().parent.parent
SLIDES_DIR = REPO_ROOT / "slides"
SLIDE_DATA_PATH = SLIDES_DIR / "slide_data.json"


def _require(value, name: str):
    """Fail loudly (per the HARD RULE) rather than silently substituting a placeholder."""
    if value is None:
        raise RuntimeError(f"build_deck: required value '{name}' is missing -- refusing to build")
    return value


def _t7_a_hat(cfg: Config) -> float:
    """Live re-run of T7's L-BFGS fit (tests/test_model_loss.py::test_T7_parametric_recovery_lbfgs's
    procedure) at the given cfg, returning the achieved a_hat (not asserted here, just measured)."""
    rng = np.random.default_rng(cfg.seed_data)
    T, Z, Y = build_dataset(rng, cfg)
    a_param = torch.zeros((), dtype=torch.float64, requires_grad=True)
    b_param = torch.zeros((), dtype=torch.float64, requires_grad=True)
    alpha_table = AlphaTable(cfg.m)
    Y_t = torch.as_tensor(Y)
    T_t = torch.as_tensor(T, dtype=torch.int64)
    Z_t = torch.as_tensor(Z)
    phi = -torch.log(torch.as_tensor(norm.cdf(cfg.beta * Y)))
    opt = torch.optim.LBFGS([a_param, b_param, alpha_table.free], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        theta = a_param * phi + b_param
        r = theta + alpha_table.at(T_t)
        loss = fusion_loss(r, Z_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(a_param.item())


def _t4_agreement() -> dict:
    """Live re-run of T4's numpy-twin agreement check, returning the achieved differences."""
    H, m = 32, 6
    rng = np.random.default_rng(7)
    p = npref.init_params(rng, H)
    batch = 64
    t_np = rng.integers(1, m + 1, size=batch)
    z_np = rng.integers(0, 2, size=batch).astype(np.float64)
    y_np = rng.normal(size=batch)
    r_np, _ = npref.forward(p, y_np, t_np)
    L_np, g_np = npref.loss_and_grad(p, y_np, t_np, z_np)

    theta_net = ThetaNet(H)
    theta_net.load_from_numpy(p)
    alpha_table = AlphaTable(m)
    with torch.no_grad():
        alpha_table.free.copy_(torch.as_tensor(p["alpha"][:-1]))
    y_t = torch.as_tensor(y_np)
    t_t = torch.as_tensor(t_np, dtype=torch.int64)
    z_t = torch.as_tensor(z_np)
    r_t = ratio_model(theta_net, alpha_table, y_t, t_t)
    r_diff = float(np.max(np.abs(r_t.detach().numpy() - r_np)))
    L_t = fusion_loss(r_t, z_t)
    L_diff = float(abs(L_t.item() - L_np))
    L_t.backward()
    grad_diffs = [
        float(np.max(np.abs(theta_net.W1.grad.numpy() - g_np["W1"]))),
        float(np.max(np.abs(theta_net.b1.grad.numpy() - g_np["b1"]))),
        float(np.max(np.abs(theta_net.w2.grad.numpy() - g_np["w2"]))),
        float(abs(theta_net.b2.grad.item() - g_np["b2"])),
        float(np.max(np.abs(alpha_table.free.grad.numpy() - g_np["alpha"][:-1]))),
    ]
    return {"r_diff": r_diff, "L_diff": L_diff, "worst_grad_diff": max(grad_diffs)}


def _t11_at(cfg: Config, n_boot: int = 200) -> dict:
    """T11(a)-(c)'s numbers (oracle theta*, no training) at an arbitrary cfg -- used for both the
    current default and, via a heavy-tailed cfg, the old beta=1.5 contrast (slide 3 learning 2)."""
    a_tail = 1 + 1 / (cfg.beta ** 2 * cfg.sigma ** 2)
    fracs = []
    for s in range(10):
        y = dgp.draw_survey(cfg, np.random.default_rng(100 + s), 1, cfg.n)
        w = np.exp(dgp.theta_star(cfg, y))
        fracs.append((w.sum() ** 2 / (w ** 2).sum()) / cfg.n)
    fracs = np.array(fracs)
    cv = float(fracs.std() / fracs.mean())

    rng = np.random.default_rng(7)
    y0 = dgp.draw_survey(cfg, rng, 1, cfg.n)
    w0 = np.exp(dgp.theta_star(cfg, y0))
    mh0 = float((w0 * y0).sum() / w0.sum())
    delta_se = float(np.sqrt((w0 ** 2 * (y0 - mh0) ** 2).sum()) / w0.sum())
    reps = np.array([oracle_estimate(cfg, dgp.draw_survey(cfg, rng, 1, cfg.n)) for _ in range(n_boot)])
    se_ratio = delta_se / float(reps.std())
    return {"a": a_tail, "n_eff_cv": cv, "delta_se": delta_se, "mc_sd": float(reps.std()), "se_ratio": se_ratio}


def collect() -> dict:
    cfg = Config()
    data: dict = {"config": json.loads(cfg.to_json())}

    # ---------------- Slide 1: DGP ----------------
    a_tail = 1 + 1 / (cfg.beta ** 2 * cfg.sigma ** 2)
    r3_holds = bool(cfg.beta * cfg.sigma < 1 / np.sqrt(2))
    bias_t1 = float(dgp.ES_closed(cfg, 1) - dgp.m_t(cfg, 1))
    bias_tM = float(dgp.ES_closed(cfg, cfg.M) - dgp.m_t(cfg, cfg.M))
    data["slide1"] = {
        "m": cfg.m, "M": cfg.M, "n": cfg.n, "sigma": cfg.sigma, "beta": cfg.beta,
        "m_t_intercept": cfg.m_t_intercept, "m_t_slope": cfg.m_t_slope,
        "c_t_intercept": cfg.c_t_intercept, "c_t_slope": cfg.c_t_slope,
        "tail_index_a": a_tail, "r3_holds": r3_holds,
        "survey_bias_t1": bias_t1, "survey_bias_tM": bias_tM,
        "figure": "phase1a_densities_96169d0c.png",
    }

    # ---------------- shared training run (Slides 2 & 3) ----------------
    rng = np.random.default_rng(cfg.seed_data)
    T, Z, Y = build_dataset(rng, cfg)
    tr, va = stratified_split(rng, T, Z, cfg.val_frac)
    result = train(cfg, T, Z, Y, tr, va)
    foc = foc_diagnostic(cfg, result.theta_net, result.alpha_table, T, Z, Y, result.standardizer)
    alpha_hat = result.alpha_table.forward().detach().numpy().tolist()
    ts_m = np.arange(1, cfg.m + 1)
    alpha_star = dgp.alpha_star(cfg, ts_m).tolist()

    # ---------------- Slide 2: optimization ----------------
    t4 = _t4_agreement()
    t7_a_hat = _t7_a_hat(Config(n=200_000))
    t11_default = _t11_at(cfg)

    t0 = time.time()
    proc = subprocess.run(["python3", "-m", "pytest", "-q", "tests/"], cwd=str(REPO_ROOT),
                           capture_output=True, text=True)
    pytest_runtime = time.time() - t0
    pytest_summary = [l for l in proc.stdout.strip().splitlines() if l][-1] if proc.stdout.strip() else ""
    if proc.returncode != 0:
        raise RuntimeError(f"build_deck: pytest failed, refusing to claim tests pass:\n{proc.stdout[-2000:]}")

    data["slide2"] = {
        "rows_total": 2 * cfg.m * cfg.n,
        "H": cfg.H, "lr": cfg.lr, "wd": cfg.wd, "batch": cfg.batch, "epochs": cfg.epochs,
        "best_val_loss": result.best_val_loss, "best_epoch": result.best_epoch,
        "foc": {str(k): v for k, v in foc.items()},
        "t4": t4, "t7_a_hat": t7_a_hat, "t11_a": t11_default["a"],
        "t11_n_eff_cv": t11_default["n_eff_cv"], "t11_se_ratio": t11_default["se_ratio"],
        "pytest_summary": pytest_summary, "pytest_runtime_s": pytest_runtime,
        "figure_val_curve": "phase3_val_curve_96169d0c.png",
        "figure_alpha": "phase3_alpha_hat_vs_star_96169d0c.png",
    }

    # ---------------- Slide 3: results and learnings ----------------
    survey_mean_m = survey_mean_baseline(dgp.draw_survey(cfg, rng, cfg.m, cfg.n))
    mu_m_true = float(dgp.mu_true(cfg, cfg.m))
    table = []
    for t in range(cfg.m + 1, cfg.M + 1):
        y_t = dgp.draw_survey(cfg, rng, t, cfg.n)
        survey_mean_t = survey_mean_baseline(y_t)
        est, n_eff_frac = mu_tilde(result.theta_net, result.standardizer, y_t)
        se = bootstrap_se(result.theta_net, result.standardizer, y_t, n_boot=200, rng=rng)
        table.append({
            "t": t, "mu_true": float(dgp.mu_true(cfg, t)), "survey_mean": survey_mean_t,
            "mu_tilde": est, "se": se, "n_eff_over_n": n_eff_frac,
            "offset": offset_baseline(mu_m_true, survey_mean_m, survey_mean_t),
            "oracle": oracle_estimate(cfg, y_t),
        })

    # Learning 2: beta=1.5 contrast (live, via the heavy-tail opt-in)
    heavy_cfg = Config(beta=1.5, allow_heavy_tails=True)
    t11_heavy = _t11_at(heavy_cfg)
    r3_stress = stress_test_violate_r3(cfg)  # same numbers as _t11_at(heavy) plus decay exponents
    t7_a_hat_heavy = _t7_a_hat(Config(n=200_000, beta=1.5, allow_heavy_tails=True))

    # Learning 3: residual right-tail error (C4(a) verification, live)
    grid = theta_grid_data(cfg, result.theta_net, result.standardizer, npts=400)
    grid_y, drift = grid["y"], (grid["theta_hat"] - grid["theta_star"])
    y_big = dgp.draw_survey(cfg, rng, cfg.M, 100_000)
    mass_shares = {str(k): float(np.mean(y_big > k)) for k in (3, 4, 5)}

    n_rep_draw, n_reps_implied = 200, 300
    rng_imp = np.random.default_rng(999)
    implied = np.empty(n_reps_implied)
    actual = np.empty(n_reps_implied)
    for i in range(n_reps_implied):
        y9 = dgp.draw_survey(cfg, rng_imp, cfg.M, n_rep_draw)
        theta_star_y = dgp.theta_star(cfg, y9)
        d_interp = np.interp(y9, grid_y, drift)
        w_oracle = np.exp(theta_star_y)
        w_interp = np.exp(theta_star_y + d_interp)
        implied[i] = (w_interp * y9).sum() / w_interp.sum() - (w_oracle * y9).sum() / w_oracle.sum()
        est_i, _ = mu_tilde(result.theta_net, result.standardizer, y9)
        actual[i] = est_i - oracle_estimate(cfg, y9)

    am = dgp.a_t(cfg, cfg.m)
    train_lo, train_hi = float(Y.min()), float(Y.max())
    theta_star_bounds = {
        "log_phi_am": float(np.log(norm.cdf(am))),
        "at_train_lo": float(dgp.theta_star(cfg, train_lo)),
        "at_0": float(dgp.theta_star(cfg, 0.0)),
        "at_train_hi": float(dgp.theta_star(cfg, train_hi)),
        "train_lo": train_lo, "train_hi": train_hi,
    }

    data["slide3"] = {
        "table": table,
        "learning2": {
            "beta06": {"a": t11_default["a"], "cv": t11_default["n_eff_cv"], "se_ratio": t11_default["se_ratio"],
                       "t7_a_hat": t7_a_hat},
            "beta15": {"a": t11_heavy["a"], "cv": t11_heavy["n_eff_cv"], "se_ratio": t11_heavy["se_ratio"],
                       "t7_a_hat": t7_a_hat_heavy, "decay_exponents": r3_stress["exponents"],
                       "predicted_exponent": r3_stress["predicted_exponent"]},
        },
        "learning3": {
            "mass_shares_above": mass_shares,
            "implied_shift_mean": float(implied.mean()), "implied_shift_sd": float(implied.std(ddof=1)),
            "actual_shift_mean": float(actual.mean()), "actual_shift_sd": float(actual.std(ddof=1)),
            "n_reps": n_reps_implied, "n_per_rep": n_rep_draw,
            "theta_star_bounds": theta_star_bounds, "theta_bounded": cfg.theta_bounded,
            "grid_y_sample": [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
            "drift_at_y_sample": [float(np.interp(yy, grid_y, drift)) for yy in (-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7)],
        },
        "figure": "phase4_main_figure_96169d0c.png",
    }

    for k in ("slide1", "slide2", "slide3"):
        _require(data.get(k), k)
    return data


if __name__ == "__main__":
    SLIDES_DIR.mkdir(exist_ok=True)
    print("Collecting slide data (trains a model, runs T7 x2, runs the full pytest suite -- this takes a while)...")
    t_start = time.time()
    data = collect()
    with open(SLIDE_DATA_PATH, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print(f"saved: {SLIDE_DATA_PATH}  (collection took {time.time() - t_start:.1f}s)")

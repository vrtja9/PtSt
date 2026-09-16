"""tests/test_estimate.py -- T9, T10 (CLAUDE.md §5; Phase 3 of PROMPT_kickoff.md).

C/C++ -> Python -> R: `assert(fabs(x-y)<tol)` -> `assert abs(x-y) < tol` -> `stopifnot(abs(x-y)<tol)`.
"""
import numpy as np

from fusion import dgp
from fusion.config import Config
from fusion.data import build_dataset, stratified_split
from fusion.estimate import mu_tilde, oracle_estimate
from fusion.train import foc_diagnostic, train


def _bootstrap_se_oracle(cfg: Config, y_raw: np.ndarray, n_boot: int, rng: np.random.Generator) -> float:
    """Bootstrap SE of the ORACLE estimator (theta* fixed, known exactly) -- same resampling
    idea as fusion.estimate.bootstrap_se, but for oracle_estimate rather than a fitted theta~."""
    n = len(y_raw)
    ests = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ests[b] = oracle_estimate(cfg, y_raw[idx])
    return float(ests.std(ddof=1))


def test_T9_oracle_estimator_matches_truth():
    """With theta* plugged in, |mu~(t)-mu(t)| < 3*SE for t in {m+1,...,M}, n=20k."""
    cfg = Config(n=20_000)
    rng = np.random.default_rng(cfg.seed_data)
    for t in range(cfg.m + 1, cfg.M + 1):
        y = dgp.draw_survey(cfg, rng, t, cfg.n)
        oracle = oracle_estimate(cfg, y)
        mu_true = float(dgp.mu_true(cfg, t))
        se = _bootstrap_se_oracle(cfg, y, n_boot=200, rng=rng)
        assert abs(oracle - mu_true) < 3 * se, f"t={t}: |diff|={abs(oracle-mu_true):.4f} !< 3*SE={3*se:.4f}"


def test_T10_trained_estimator_report_then_loose_assert():
    """Report FOC and mu~ vs oracle, then assert loosely: FOC within +-0.05 for t<=m, and
    |mu~(t)-oracle_mu~(t)| < 0.05 for t>m, on the default (CP3-adopted) config."""
    cfg = Config()
    rng = np.random.default_rng(cfg.seed_data)
    T, Z, Y = build_dataset(rng, cfg)
    tr_mask, va_mask = stratified_split(rng, T, Z, cfg.val_frac)
    result = train(cfg, T, Z, Y, tr_mask, va_mask)

    foc = foc_diagnostic(cfg, result.theta_net, result.alpha_table, T, Z, Y, result.standardizer)
    print("T10 FOC:", {k: round(v, 3) for k, v in foc.items()})
    for t, v in foc.items():
        assert abs(v - 1.0) < 0.05, f"FOC[{t}]={v:.4f} outside +-0.05"

    for t in range(cfg.m + 1, cfg.M + 1):
        y = dgp.draw_survey(cfg, rng, t, cfg.n)
        est, _ = mu_tilde(result.theta_net, result.standardizer, y)
        oracle = oracle_estimate(cfg, y)
        print(f"T10 t={t}: mu~={est:.4f} oracle={oracle:.4f} |diff|={abs(est-oracle):.4f}")
        assert abs(est - oracle) < 0.05, f"t={t}: |mu~-oracle|={abs(est-oracle):.4f} >= 0.05"

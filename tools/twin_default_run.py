"""Numpy-twin reference run at the corrected default (docs/math_fixed.md §D, §F).
Run:  python tools/twin_default_run.py   (from any cwd; the shim below puts the repo root on sys.path)
C/C++ -> Python -> R: this is a driver/main() that fixes the run configuration explicitly rather
than relying on the library's own defaults, so the printed numbers are reproducible."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: fusion_numpy.py lives there
# C/C++ -> Python -> R: the runtime equivalent of -I<repo> on the compiler's include path
# (R: the .libPaths()/here::here() idiom).

import numpy as np
from scipy.stats import norm
import fusion_numpy as F

F.CFG.update(beta=0.6, H=32, lr=3e-3, wd=1e-5, epochs=400)
rng_d = np.random.default_rng(F.CFG['seed_data']); rng_m = np.random.default_rng(F.CFG['seed_model'])
T, Z, Y = F.build_dataset(rng_d)
tr, va = F.stratified_split(rng_d, T, Z, F.CFG['val_frac'])
p = F.init_params(rng_m, F.CFG['H'])
p, nc, best = F.train(p, T, Z, Y, tr, va, rng_m)
print("config:", {k: F.CFG[k] for k in ('m','M','n','sigma','beta','H','lr','wd','epochs','batch','val_frac','seed_data','seed_model')})
print("best val %.4f at epoch %d" % (best[0], best[2]))
print("FOC   :", {t: round(float(v), 3) for t, v in F.foc_diagnostic(p, T, Z, Y, nc).items()})
atrue = [np.log(norm.cdf(F.a_t(t))) - np.log(norm.cdf(F.a_t(F.CFG['m']))) for t in range(1, F.CFG['m'] + 1)]
print("alpha~:", np.round(p['alpha'], 3)); print("alpha*:", np.round(atrue, 3))
grid = np.linspace(-3, 4, 8)
th, _ = F.theta_of(p, (grid - nc[0]) / nc[1]); d = th - F.theta_star(grid)
print("grid y            :", grid)
print("theta~-theta*     :", np.round(d, 3))
print("centered (- mean over y in [-2,3]):", np.round(d - d[1:7].mean(), 3))
lo, hi = Y.min(), Y.max()
for t in range(F.CFG['m'] + 1, F.CFG['M'] + 1):
    ys = F.draw_survey(rng_d, t, F.CFG['n'])
    est, neff = F.mu_tilde(p, ys, nc); wo = np.exp(F.theta_star(ys))
    print(" t=%d mu=%.3f survey=%.3f mu~=%.3f oracle=%.3f neff/n=%.2f  frac outside training range [%.2f,%.2f]=%.4f"
          % (t, F.m_t(t), ys.mean(), est, (wo * ys).sum() / wo.sum(), neff, lo, hi,
             float(np.mean((ys < lo) | (ys > hi)))))

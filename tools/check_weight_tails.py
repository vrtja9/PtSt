"""Evidence for docs/math_fixed.md §G: moments of the importance weight w = e^{theta*}.
Run:  python tools/check_weight_tails.py   (from any cwd; the shim below puts the repo root on sys.path)
C/C++ -> Python -> R: quad() is adaptive Gauss-Kronrod numerical integration (GSL's
gsl_integration_qag in C; integrate() in R); the loops below are the same quantities
obtained by Monte-Carlo sampling instead of quadrature."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: fusion_numpy.py lives there
# C/C++ -> Python -> R: the runtime equivalent of -I<repo> on the compiler's include path
# (R: the .libPaths()/here::here() idiom).

import warnings
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import fusion_numpy as F

warnings.filterwarnings("ignore")
SIGMA = F.CFG['sigma']; T1 = 1

def w_of(y):                       # oracle weight w = exp(theta*)  (Step 3 / Theorem 10.7)
    return np.exp(F.theta_star(y))

def hajek(y):                      # self-normalized estimator (Step 9, Definition 9.2)
    w = w_of(y); return (w * y).sum() / w.sum()

print("== A. E_S[w^k] integrals vs integration range (divergence = infinite moment) ==")
for beta in (0.6, 0.8, 1.0, 1.5):
    F.CFG['beta'] = beta
    a = 1 + 1 / (beta ** 2 * SIGMA ** 2)
    out = []
    for k in (2, 3):
        vals = []
        for L in (10, 20, 40):
            f = lambda y: norm.pdf(y, F.m_t(T1), SIGMA) / max(norm.cdf(beta * y), 1e-300) ** (k - 1)
            vals.append(quad(f, -L, 10, limit=400)[0])
        out.append("k=%d: %s" % (k, ["%.3g" % v for v in vals]))
    print("  beta=%.1f  tail index a=%5.2f  (a>2 finite var, a>3 finite 3rd moment)  L=10,20,40  %s"
          % (beta, a, "   ".join(out)))

print("== B. predicted vs observed O(1/n) bias of mu~ at t=1 (oracle theta*) ==")
for beta in (0.6, 1.5):
    F.CFG['beta'] = beta
    a = 1 + 1 / (beta ** 2 * SIGMA ** 2); mu = F.m_t(T1)
    if a > 3:
        am, at = F.a_t(F.CFG['m']), F.a_t(T1)
        wf = lambda y: norm.cdf(am) / max(norm.cdf(beta * y), 1e-300)
        sf = lambda y: norm.cdf(beta * y) * norm.pdf(y, mu, SIGMA) / norm.cdf(at)
        Ew = quad(lambda y: wf(y) * sf(y), -30, 30, limit=400)[0]
        Ew2 = quad(lambda y: wf(y) ** 2 * (y - mu) * sf(y), -30, 30, limit=400)[0]
        pred = "%.3f" % (-Ew2 / Ew ** 2)
        chk = "  [check E_S[w] = e^{-alpha*(1)}: %.4f vs %.4f]" % (
            Ew, np.exp(-(np.log(norm.cdf(at)) - np.log(norm.cdf(am)))))
    else:
        pred, chk = "not defined (a<=3, third moment infinite)", ""
    print("  beta=%.1f  a=%.2f  predicted n*bias -> %s%s" % (beta, a, pred, chk))
    rng = np.random.default_rng(21)
    for n in (80, 320, 1280):
        est = np.array([hajek(F.draw_survey(rng, T1, n)) for _ in range(1000)])
        print("     n=%5d  n*bias=%+8.2f (MC SE %.2f)   sd*sqrt(n)=%.3f"
              % (n, n * (est.mean() - mu), n * est.std() / np.sqrt(1000), est.std() * np.sqrt(n)))

print("== C. delta-method SE vs Monte-Carlo sd of mu~ (t=1, n=2000, 200 reps) ==")
for beta in (0.6, 1.5):
    F.CFG['beta'] = beta; rng = np.random.default_rng(7)
    y = F.draw_survey(rng, T1, 2000); w = w_of(y); mh = (w * y).sum() / w.sum()
    se = np.sqrt((w ** 2 * (y - mh) ** 2).sum()) / w.sum()
    reps = np.array([hajek(F.draw_survey(rng, T1, 2000)) for _ in range(200)])
    print("  beta=%.1f  delta SE %.4f   MC sd %.4f   ratio %.2f   MC mean %.4f (mu=%.3f)"
          % (beta, se, reps.std(), se / reps.std(), reps.mean(), F.m_t(T1)))

print("== D. n_eff/n at t=1, n=2000, across 10 data seeds ==")
for beta in (0.6, 1.5):
    F.CFG['beta'] = beta
    v = []
    for s in range(10):
        y = F.draw_survey(np.random.default_rng(100 + s), T1, 2000); w = w_of(y)
        v.append((w.sum() ** 2 / (w ** 2).sum()) / 2000)
    v = np.array(v)
    print("  beta=%.1f  mean %.3f  sd %.3f  CV %.3f  min %.3f  max %.3f"
          % (beta, v.mean(), v.std(), v.std() / v.mean(), v.min(), v.max()))

print("== E. survey bias E_S[Y]-mu over t, at the proposed default beta=0.6 ==")
F.CFG['beta'] = 0.6
print("  " + "  ".join("t=%d:%+.3f" % (t, F.ES_closed(t) - F.m_t(t))
                       for t in range(1, F.CFG['M'] + 1)))

"""Evidence for docs/math_fixed.md §G: moments of the importance weight w = e^{theta*}.
Run:  python tools/check_weight_tails.py   (from any cwd; the shim below puts the repo root on sys.path)
C/C++ -> Python -> R: quad() is adaptive Gauss-Kronrod numerical integration (GSL's
gsl_integration_qag in C; integrate() in R); the loops below are the same quantities
obtained by Monte-Carlo sampling instead of quadrature.
Section B note (C3, 2026-09-18, supersedes the A5 note this replaced -- that one wrongly labeled
an interval containing the prediction as "UNINFORMATIVE"; containment there means the test
AGREES with the prediction, not that it says nothing): at n=40..320 with 10000 reps, the interval
half-width (2*MC_SE) verifies the SS C bias identity (E[mu~]-mu ~= -E_S[w^2(Y-mu)]/(n E_S[w]^2))
directly -- CONSISTENT means the prediction is inside the interval AND the interval is tight
enough (half-width <= 50% of the prediction) to mean something; DISAGREES means the prediction
falls outside; TOO WIDE TO DISTINGUISH means the interval contains the prediction but is too
wide (>50% of it) for that containment to be informative. n=1280 is dropped here: even at 10000
reps its MC SE cannot resolve the identity (verified below) -- the informative statistic at
large n is instead the FLATNESS of sd*sqrt(n) across n (flat = O(1/n) bias regime holding), not
any one n*bias value."""
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
print("   (legend: kappa>2 -> finite variance (k=2); kappa>3 -> finite 3rd moment (k=3); L=10,20,40 is the integration range)")
for beta in (0.6, 0.8, 1.0, 1.5):
    F.CFG['beta'] = beta
    kappa = 1 + 1 / (beta ** 2 * SIGMA ** 2)
    out = []
    for k in (2, 3):
        vals = []
        for L in (10, 20, 40):
            f = lambda y: norm.pdf(y, F.m_t(T1), SIGMA) / max(norm.cdf(beta * y), 1e-300) ** (k - 1)
            vals.append(quad(f, -L, 10, limit=400)[0])
        out.append("k=%d: %s" % (k, ["%.3g" % v for v in vals]))
    print("  beta=%.1f  tail index kappa=%5.2f  %s" % (beta, kappa, "   ".join(out)))

print("== B. predicted vs observed O(1/n) bias of mu~ at t=1 (oracle theta*) ==")
for beta in (0.6, 1.5):
    F.CFG['beta'] = beta
    kappa = 1 + 1 / (beta ** 2 * SIGMA ** 2); mu = F.m_t(T1)
    pred_val = None
    if kappa > 3:
        am, at = F.a_t(F.CFG['m']), F.a_t(T1)
        wf = lambda y: norm.cdf(am) / max(norm.cdf(beta * y), 1e-300)
        sf = lambda y: norm.cdf(beta * y) * norm.pdf(y, mu, SIGMA) / norm.cdf(at)
        Ew = quad(lambda y: wf(y) * sf(y), -30, 30, limit=400)[0]
        Ew2 = quad(lambda y: wf(y) ** 2 * (y - mu) * sf(y), -30, 30, limit=400)[0]
        pred_val = -Ew2 / Ew ** 2
        pred = "%.3f" % pred_val
        chk = "  [check E_S[w] = e^{-alpha*(1)}: %.4f vs %.4f]" % (
            Ew, np.exp(-(np.log(norm.cdf(at)) - np.log(norm.cdf(am)))))
    else:
        pred, chk = "not defined (kappa<=3, third moment infinite)", ""
    print("  beta=%.1f  kappa=%.2f  predicted n*bias -> %s%s" % (beta, kappa, pred, chk))
    rng = np.random.default_rng(21)
    n_reps = 10000
    for n in (40, 80, 160, 320):
        est = np.array([hajek(F.draw_survey(rng, T1, n)) for _ in range(n_reps)])
        nbias = n * (est.mean() - mu)
        mc_se = n * est.std() / np.sqrt(n_reps)
        half_width = 2 * mc_se
        lo, hi = nbias - half_width, nbias + half_width
        flag = ""
        if pred_val is not None:
            if not (lo <= pred_val <= hi):
                flag = "  DISAGREES"
            elif half_width <= 0.5 * abs(pred_val):
                flag = "  CONSISTENT (half-width %.0f%% of prediction)" % (100 * half_width / abs(pred_val))
            else:
                flag = "  TOO WIDE TO DISTINGUISH"
        print("     n=%5d  n*bias=%+8.3f  observed +-2*MC_SE=[%+.3f, %+.3f]   sd*sqrt(n)=%.3f%s"
              % (n, nbias, lo, hi, est.std() * np.sqrt(n), flag))

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

"""fusion_numpy.py  --  runs on iPad (Pyto / Carnets: numpy + scipy only, no torch)

Node numbers match the implementation tree in the chat.
Same math as the challenge eq. (2)-(3); the PyTorch version replaces the hand-written
backward pass (node 4.2) with autograd and the hand-written Adam (node 5.1) with torch.optim.Adam.

C/C++ readers: a numpy array = contiguous double[] (SoA layout); "@" = matrix multiply;
boolean masks = filtered copies; rng objects = explicit RNG state passed by reference.
"""
import numpy as np
from scipy.stats import norm

# ---------------- 0.4 config (a C struct with defaults, serialised with every output) ----------------
CFG = dict(m=6, M=9, n=2000, sigma=1.0, beta=0.6, H=32, lr=1e-2, wd=1e-4,  # (R3): beta*sigma < 1/sqrt(2); tail index kappa = 3.78
           epochs=150, batch=256, seed_data=0, seed_model=1, val_frac=0.2)
m_t = lambda t: -0.5 + 0.25 * t          # population mean drifts (P_{t,Y} = N(m_t, sigma^2))
c_t = lambda t: 0.9 - 0.06 * t           # overall response level drifts (never enters S_t)


# ---------------- 1. DGP ----------------
def draw_pop(rng, t, n):                                    # 1.2  Y ~ P_{t,Y}
    return rng.normal(m_t(t), CFG['sigma'], n)

def draw_survey(rng, t, n):                                 # 1.3-1.4  keep R=1  => density ∝ pi_t * p_t = s_t
    out = np.empty(0)
    while out.size < n:                                     # C: while(count<n){...}  vectorised block-wise
        y = draw_pop(rng, t, 4 * n)
        keep = rng.random(4 * n) < c_t(t) * norm.cdf(CFG['beta'] * y)
        out = np.concatenate([out, y[keep]])
    return out[:n]

def a_t(t):                                                 # 1.5 closed forms (Step 10 of the derivation)
    b, s = CFG['beta'], CFG['sigma']
    return b * m_t(t) / np.sqrt(1 + b * b * s * s)

def ES_closed(t):
    b, s = CFG['beta'], CFG['sigma']
    return m_t(t) + b * s * s * norm.pdf(a_t(t)) / (np.sqrt(1 + b * b * s * s) * norm.cdf(a_t(t)))

def theta_star(y):                                          # oracle log-weight, anchored so alpha*(m)=0
    return np.log(norm.cdf(a_t(CFG['m']))) - np.log(norm.cdf(CFG['beta'] * y))


# ---------------- 2. dataset: pooled F as struct-of-arrays (T:int, Z:float, Y:float) ----------------
def build_dataset(rng):                                     # 2.1  2mn rows
    T, Z, Y = [], [], []
    for t in range(1, CFG['m'] + 1):
        ys, yp = draw_survey(rng, t, CFG['n']), draw_pop(rng, t, CFG['n'])
        T += [t] * (2 * CFG['n']); Z += [0.0] * CFG['n'] + [1.0] * CFG['n']; Y += list(ys) + list(yp)
    return np.array(T), np.array(Z), np.array(Y)

def stratified_split(rng, T, Z, frac):                      # 2.2  hold out frac inside every (t,z) cell
    val = np.zeros(len(T), bool)
    for t in np.unique(T):
        for z in (0.0, 1.0):
            idx = np.where((T == t) & (Z == z))[0]
            val[rng.choice(idx, int(frac * len(idx)), replace=False)] = True
    return ~val, val


# ---------------- 3. model: theta = MLP(1 -> H -> 1) ; alpha in R^m with alpha[m] frozen at 0 ----------------
def init_params(rng, H):
    return dict(W1=rng.normal(0, 1, (H, 1)), b1=np.zeros(H),
                w2=rng.normal(0, 1 / np.sqrt(H), H), b2=0.0, alpha=np.zeros(CFG['m']))

def theta_of(p, ys):                                        # 3.1  ys already standardised
    pre = p['W1'] @ ys[None, :] + p['b1'][:, None]          # (H,B)
    h = np.maximum(pre, 0)                                  # ReLU
    return p['w2'] @ h + p['b2'], (pre, h)

def forward(p, ys, t):                                      # 3.3  r = theta(y) + alpha[t]   (t is 1-based -> index t-1)
    theta, cache = theta_of(p, ys)
    return theta + p['alpha'][t - 1], cache


# ---------------- 4. loss (2) and hand-written backward (chain rule, reverse order) ----------------
def loss_and_grad(p, ys, t, z):
    r, (pre, h) = forward(p, ys, t)
    L = np.mean((1 - z) * np.exp(r) - z * r)                # 4.1  eq. (2) averaged over the batch
    dr = ((1 - z) * np.exp(r) - z) / len(ys)                # dL/dr_i  (each row's own derivative)
    g = {}
    g['b2'] = dr.sum();               g['w2'] = h @ dr      # theta = w2.h + b2
    dh = np.outer(p['w2'], dr);       dpre = dh * (pre > 0) # ReLU' = 1[pre>0]
    g['W1'] = dpre @ ys[:, None];     g['b1'] = dpre.sum(1)
    g['alpha'] = np.bincount(t - 1, weights=dr, minlength=CFG['m'])   # C: grad_alpha[t-1] += dr_i
    g['alpha'][-1] = 0.0                                    # frozen anchor alpha[m]=0
    return L, g

def grad_check(p, ys, t, z, eps=1e-6):                      # 4.3  central differences, float64
    _, g = loss_and_grad(p, ys, t, z); worst = 0.0
    for key in ('W1', 'b1', 'w2', 'alpha'):
        flat = p[key].reshape(-1)
        for i in [0, len(flat) // 2, len(flat) - 1]:
            if key == 'alpha' and i == len(flat) - 1: continue   # frozen entry
            old = flat[i]
            flat[i] = old + eps; Lp, _ = loss_and_grad(p, ys, t, z)
            flat[i] = old - eps; Lm, _ = loss_and_grad(p, ys, t, z)
            flat[i] = old
            fd = (Lp - Lm) / (2 * eps); an = g[key].reshape(-1)[i]
            worst = max(worst, abs(fd - an) / (abs(an) + 1e-12))
    return worst


# ---------------- 5. training: Adam by hand + early stopping ----------------
def adam_init(p): return dict(m={k: np.zeros_like(v) for k, v in p.items()}, v={k: np.zeros_like(v) for k, v in p.items()}, k=0)

def adam_step(p, g, st, lr, wd, b1=0.9, b2=0.999, eps=1e-8):      # 5.1
    st['k'] += 1; k = st['k']
    for key in p:
        gk = g[key] + (wd * p[key] if key != 'alpha' else 0.0)     # L2 weight decay on the network only
        st['m'][key] = b1 * st['m'][key] + (1 - b1) * gk
        st['v'][key] = b2 * st['v'][key] + (1 - b2) * gk ** 2
        mhat = st['m'][key] / (1 - b1 ** k); vhat = st['v'][key] / (1 - b2 ** k)
        p[key] = p[key] - lr * mhat / (np.sqrt(vhat) + eps)

def train(p, T, Z, Y, tr, va, rng):
    ymean, ystd = Y[tr].mean(), Y[tr].std()                 # 2.3 standardisation constants (stored!)
    Ys = (Y - ymean) / ystd
    st = adam_init(p); best = (np.inf, None); idx_tr = np.where(tr)[0]
    for ep in range(CFG['epochs']):
        rng.shuffle(idx_tr)                                 # Fisher-Yates on an index array
        for s in range(0, len(idx_tr), CFG['batch']):
            b = idx_tr[s:s + CFG['batch']]
            _, g = loss_and_grad(p, Ys[b], T[b], Z[b])
            adam_step(p, g, st, CFG['lr'], CFG['wd'])
        Lva, _ = loss_and_grad(p, Ys[va], T[va], Z[va])     # 5.2 validation risk = unbiased estimate of R
        if Lva < best[0]: best = (Lva, {k: np.copy(v) for k, v in p.items()}, ep)
    return best[1], (ymean, ystd), best


# ---------------- 6. estimation, diagnostics, baselines ----------------
def mu_tilde(p, ys_raw, norm_c):                             # 6.1 self-normalised (Hajek) estimator, alpha not needed
    th, _ = theta_of(p, (ys_raw - norm_c[0]) / norm_c[1]); w = np.exp(th)
    return (w * ys_raw).sum() / w.sum(), (w.sum() ** 2 / (w ** 2).sum()) / len(w)   # (estimate, n_eff/n)

def foc_diagnostic(p, T, Z, Y, norm_c):                     # 5.3 mean exp(theta+alpha[t]) over survey rows per t ~ 1
    Ys = (Y - norm_c[0]) / norm_c[1]; out = {}
    for t in range(1, CFG['m'] + 1):
        sel = (T == t) & (Z == 0.0); r, _ = forward(p, Ys[sel], T[sel]); out[t] = np.exp(r).mean()
    return out


if __name__ == "__main__":
    rng_d = np.random.default_rng(CFG['seed_data']); rng_m = np.random.default_rng(CFG['seed_model'])
    T, Z, Y = build_dataset(rng_d); tr, va = stratified_split(rng_d, T, Z, CFG['val_frac'])
    p = init_params(rng_m, CFG['H'])
    b = np.arange(64); print("4.3 grad-check worst rel. error:", grad_check(p, (Y[b] - Y.mean()) / Y.std(), T[b], Z[b]))
    p, norm_c, best = train(p, T, Z, Y, tr, va, rng_m)
    print("5.2 best val loss %.4f at epoch %d" % (best[0], best[2]))
    print("5.3 FOC diagnostic (should be ~1):", {t: round(v, 3) for t, v in foc_diagnostic(p, T, Z, Y, norm_c).items()})
    print("    learned alpha:", np.round(p['alpha'], 3))
    print("    true    alpha:", np.round([np.log(norm.cdf(a_t(t))) - np.log(norm.cdf(a_t(CFG['m']))) for t in range(1, CFG['m'] + 1)], 3))
    print("\n t   mu_true  survey_mean  mu_tilde  n_eff/n  offset_baseline  oracle_theta*")
    ES_m = ES_closed(CFG['m'])
    for t in range(CFG['m'] + 1, CFG['M'] + 1):
        ys = draw_survey(rng_d, t, CFG['n'])
        est, neff = mu_tilde(p, ys, norm_c)
        w_or = np.exp(theta_star(ys)); oracle = (w_or * ys).sum() / w_or.sum()
        offset = m_t(CFG['m']) + (ys.mean() - ES_m)
        print(" %d   %.3f     %.3f       %.3f    %.2f      %.3f            %.3f" % (t, m_t(t), ys.mean(), est, neff, offset, oracle))

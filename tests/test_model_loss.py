"""tests/test_model_loss.py -- T4, T5, T6, T7, T8 (CLAUDE.md §5; Phase 2 of PROMPT_kickoff.md).

C/C++ -> Python -> R: `assert(fabs(x-y)<tol)` -> `assert abs(x-y) < tol` -> `stopifnot(abs(x-y)<tol)`.
"""
import numpy as np
import torch

import fusion_numpy as npref
from fusion.config import Config
from fusion.data import build_dataset, stratified_split, Standardizer, iterate_batches
from fusion.loss import fusion_loss
from fusion.model import AlphaTable, ThetaNet, ratio_model


def test_T4_numpy_twin_agreement():
    """Same batch and parameters -> |L_torch-L_numpy|<1e-10, all grads agree to <1e-8.

    Parameters loaded from fusion_numpy.init_params, per CLAUDE.md §5 T4 and the ThetaNet
    architecture DECISION in fusion/model.py / notes/decisions.md (1->H->1, matching the twin).
    """
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
    assert np.max(np.abs(r_t.detach().numpy() - r_np)) < 1e-8

    L_t = fusion_loss(r_t, z_t)
    assert abs(L_t.item() - L_np) < 1e-10

    L_t.backward()
    assert np.max(np.abs(theta_net.W1.grad.numpy() - g_np["W1"])) < 1e-8
    assert np.max(np.abs(theta_net.b1.grad.numpy() - g_np["b1"])) < 1e-8
    assert np.max(np.abs(theta_net.w2.grad.numpy() - g_np["w2"])) < 1e-8
    assert abs(theta_net.b2.grad.item() - g_np["b2"]) < 1e-8
    assert np.max(np.abs(alpha_table.free.grad.numpy() - g_np["alpha"][:-1])) < 1e-8


def test_T5_gradcheck_tiny_net():
    """torch.autograd.gradcheck on a tiny ThetaNet+AlphaTable in float64 passes."""
    torch.manual_seed(0)
    H, m = 4, 3
    theta_net = ThetaNet(H)
    alpha_table = AlphaTable(m)
    y = torch.randn(6, dtype=torch.float64, requires_grad=True)
    t = torch.tensor([1, 2, 3, 1, 2, 3], dtype=torch.int64)
    z = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    params = (theta_net.W1, theta_net.b1, theta_net.w2, theta_net.b2, alpha_table.free, y)

    def f(W1, b1, w2, b2, free, y_):
        pre = W1 @ y_[None, :] + b1[:, None]
        h = torch.relu(pre)
        theta = w2 @ h + b2
        alpha = torch.cat([free, torch.zeros(1, dtype=free.dtype)])
        r = theta + alpha[t - 1]
        return fusion_loss(r, z)

    assert torch.autograd.gradcheck(f, params, eps=1e-6, atol=1e-4)


def test_T6_joint_convexity_midpoint():
    """Midpoint inequality of the loss in (theta,alpha) for 1000 random pairs, z in {0,1}.

    Step 6: L(r,z) is convex in r=theta+alpha (composition of convex exp/linear with a linear
    map is convex), checked directly against the loss formula (challenge eq. 2), not via the
    model classes (this is a property of the loss alone).
    """
    rng = np.random.default_rng(1)
    theta1, a1 = rng.normal(size=1000), rng.normal(size=1000)
    theta2, a2 = rng.normal(size=1000), rng.normal(size=1000)
    r1, r2 = theta1 + a1, theta2 + a2
    rm = 0.5 * (r1 + r2)

    def L(r, z):
        return (1 - z) * np.exp(r) - z * r

    for z in (0.0, 1.0):
        assert np.all(L(rm, z) <= 0.5 * (L(r1, z) + L(r2, z)) + 1e-9)


def test_T7_parametric_recovery_lbfgs():
    """theta(y)=a*(-log Phi(beta*y))+b, full-batch L-BFGS, n=200k/cell -> |a-1|<0.02,
    |alpha_hat(t)-alpha*(t)|<0.02 for all t."""
    from scipy.stats import norm as spnorm

    cfg = Config(n=200_000)
    rng = np.random.default_rng(cfg.seed_data)
    T, Z, Y = build_dataset(rng, cfg)

    a = torch.zeros((), dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    alpha_table = AlphaTable(cfg.m)

    Y_t = torch.as_tensor(Y)
    T_t = torch.as_tensor(T, dtype=torch.int64)
    Z_t = torch.as_tensor(Z)
    phi = -torch.log(torch.as_tensor(spnorm.cdf(cfg.beta * Y)))

    opt = torch.optim.LBFGS([a, b, alpha_table.free], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        theta = a * phi + b
        r = theta + alpha_table.at(T_t)
        loss = fusion_loss(r, Z_t)
        loss.backward()
        return loss

    opt.step(closure)

    assert abs(a.item() - 1.0) < 0.02
    ts = np.arange(1, cfg.m + 1)
    from fusion import dgp
    alpha_star = dgp.alpha_star(cfg, ts)
    alpha_hat = alpha_table.forward().detach().numpy()
    assert np.max(np.abs(alpha_hat - alpha_star)) < 0.02


def test_T8_anchor_exactly_zero_after_training():
    """alpha[m] is exactly 0 after 100 optimizer steps."""
    cfg = Config()
    H, m = cfg.H, cfg.m
    theta_net = ThetaNet(H)
    alpha_table = AlphaTable(m)
    opt = torch.optim.Adam(list(theta_net.parameters()) + list(alpha_table.parameters()), lr=1e-2)

    rng = np.random.default_rng(3)
    y = torch.as_tensor(rng.normal(size=128))
    t = torch.as_tensor(rng.integers(1, m + 1, size=128), dtype=torch.int64)
    z = torch.as_tensor(rng.integers(0, 2, size=128).astype(np.float64))

    for _ in range(100):
        opt.zero_grad()
        r = ratio_model(theta_net, alpha_table, y, t)
        loss = fusion_loss(r, z)
        loss.backward()
        opt.step()

    assert alpha_table.forward()[-1].item() == 0.0

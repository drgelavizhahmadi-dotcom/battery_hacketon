"""arch/sc_model: SC-KAN-delta (shared-curve KAN-delta), see arch/sc_prereg.md (0053d01).

g_i(x_i) = alpha_i SiLU(x_i) + sum_k c_ik B_k(x_i)        one shared curve per input (8)
z_i = g_i - mean_train(g_i)                               centring only (decision C)
u_j = sum_i A_ij z_i + sum_m W_jm tanh(e_m(salt)) + b_j   6 hidden units, tanh
h = sum_j w_j tanh(u_j) + b0;  delta = x_co * h
Loss = MSE + lam sum_i(||D2 c_i||^2 + K_silu alpha_i^2) + mu sum_i ||P_null c_i||^2 + beta(||A||^2 + ||w||^2 + ||W||^2 + ||e||^2)
"""
import copy
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from kan import LR, N_RBF, SMOOTH, WD

LAM, K_SILU, MU = SMOOTH, (30 + math.pi ** 2) / 90, 1e-5
H_UNITS, D_IN = 6, 8
NB = torch.tensor(np.linalg.qr(np.column_stack([np.ones(N_RBF), np.arange(N_RBF, dtype=float)]))[0])


class SCKAN(nn.Module):
    def __init__(self, n_salt, d=D_IN, H=H_UNITS):
        super().__init__()
        self.register_buffer("grid", torch.linspace(-1, 1, N_RBF))
        self.h = 2 / (N_RBF - 1)
        self.alpha = nn.Parameter(torch.empty(d).uniform_(-1, 1))      # as nn.Linear(1, 1) init
        self.coef = nn.Parameter(torch.randn(d, N_RBF) * 0.1)            # as kan.KANLayer
        self.mix = nn.Linear(d, H)                                       # weight = A^T (H x d), bias = b
        self.emb = nn.Embedding(n_salt, 2)
        self.salt = nn.Linear(2, H, bias=False)                          # W (H x 2)
        self.head = nn.Linear(H, 1)                                      # w, b0
        self.register_buffer("center", torch.zeros(d))

    def curves(self, x):
        rbf = torch.exp(-(((x[..., None] - self.grid) / self.h) ** 2))  # (n, d, K)
        return self.alpha * F.silu(x) + (rbf * self.coef).sum(-1)

    def hidden(self, x, s, fit=False):
        g = self.curves(x)
        z = g - (g.mean(0) if fit else self.center)
        return self.mix(z) + self.salt(torch.tanh(self.emb(s)))

    def forward(self, x, s, fit=False):
        return self.head(torch.tanh(self.hidden(x, s, fit)))[:, 0]

    @torch.no_grad()
    def set_center(self, x):
        self.center.copy_(self.curves(x).mean(0))


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def beta_norm(prm):
    return sum((prm[k] ** 2).sum() for k in ["mix.weight", "head.weight", "salt.weight", "emb.weight"])


def penalty(prm, beta):
    c = prm["coef"]
    d2 = c[:, 2:] - 2 * c[:, 1:-1] + c[:, :-2]
    return (LAM * ((d2 ** 2).sum() + K_SILU * (prm["alpha"] ** 2).sum())
            + MU * ((c @ NB.to(c.dtype)) ** 2).sum() + beta * beta_norm(prm))


def sc_loss(net, prm, b, beta):
    pred = b["x_co"] * functional_call(net, prm, (b["x"], b["s"]), {"fit": True})
    return F.mse_loss(pred, b["y"]) + penalty(prm, beta)


def train_sc(n_salt, b, beta, epochs, seed):
    """AdamW exactly as final.py / kan.train (lr 3e-3, wd 1e-4, full batch, fixed epochs, float32), SC loss."""
    torch.manual_seed(seed)
    net = SCKAN(n_salt)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    for _ in range(epochs):
        opt.zero_grad()
        loss = sc_loss(net, dict(net.named_parameters()), b, beta)
        loss.backward()
        opt.step()
    net.set_center(b["x"])
    return net.eval()


def predict_sc(net, b):
    with torch.no_grad():
        dt = next(net.parameters()).dtype
        return (b["x_co"].to(dt) * net(b["x"].to(dt), b["s"])).double().numpy()


class FlatSC:
    """All 169 parameters as one float64 vector; loss = full SC loss on the training batch (centring recomputed)."""

    def __init__(self, net, b, beta):
        self.net = copy.deepcopy(net).double()
        self.b = {k: (v.double() if v.is_floating_point() else v) for k, v in b.items()}
        self.beta = beta
        self.meta = [(n, p.shape, p.numel()) for n, p in self.net.named_parameters()]
        self.slices, o = {}, 0
        for n, shp, k in self.meta:
            self.slices[n] = (o, o + k, shp); o += k
        self.n = o
        self.x0 = torch.cat([p.detach().reshape(-1) for p in self.net.parameters()]).numpy().copy()
        self._cache = None

    def params(self, v):
        return {n: v[a:b_].reshape(shp) for n, (a, b_, shp) in self.slices.items()}

    def loss_t(self, v):
        return sc_loss(self.net, self.params(v), self.b, self.beta)

    def base_loss(self, x):
        with torch.no_grad():
            prm = self.params(torch.tensor(x))
            pred = self.b["x_co"] * functional_call(self.net, prm, (self.b["x"], self.b["s"]), {"fit": True})
            return float(F.mse_loss(pred, self.b["y"]))

    def fun(self, x):
        if self._cache is not None and np.array_equal(self._cache[0], x):
            return self._cache[1], self._cache[2]
        v = torch.tensor(x, requires_grad=True)
        L = self.loss_t(v)
        (g,) = torch.autograd.grad(L, v)
        self._cache = (x.copy(), float(L), g.numpy().copy())
        return self._cache[1], self._cache[2]

    def hessian(self, x):
        H = torch.autograd.functional.hessian(self.loss_t, torch.tensor(x)).numpy()
        return (H + H.T) / 2

    def jac_pred(self, x):
        """Jacobian of the training-row predictions (n x 169), centring recomputed."""
        def f(v):
            return self.b["x_co"] * functional_call(self.net, self.params(v), (self.b["x"], self.b["s"]), {"fit": True})
        return torch.autograd.functional.jacobian(f, torch.tensor(x)).numpy()

    def to_net(self, x):
        net = copy.deepcopy(self.net)
        with torch.no_grad():
            for n, t in self.params(torch.tensor(x)).items():
                dict(net.named_parameters())[n].copy_(t)
        net.set_center(self.b["x"])
        return net.eval()


@torch.no_grad()
def conventions(net, x_train):
    """Analysis-only conventions (prereg): curve sign, curve unit variance, hidden-unit sign. Returns a float64 copy."""
    n = copy.deepcopy(net).double()
    x = x_train.double()
    lo, hi = x.min(0).values, x.max(0).values
    glo, ghi = n.curves(lo[None])[0], n.curves(hi[None])[0]
    flip = torch.where(ghi < glo, -1.0, 1.0).double()
    n.alpha.mul_(flip); n.coef.mul_(flip[:, None]); n.center.mul_(flip); n.mix.weight.mul_(flip[None, :])
    sd = (n.curves(x) - n.center).std(0, unbiased=False)
    n.alpha.div_(sd); n.coef.div_(sd[:, None]); n.center.div_(sd); n.mix.weight.mul_(sd[None, :])
    sj = torch.where(n.head.weight[0] < 0, -1.0, 1.0).double()
    n.head.weight.mul_(sj[None, :]); n.mix.weight.mul_(sj[:, None]); n.mix.bias.mul_(sj); n.salt.weight.mul_(sj[:, None])
    return n.eval()


def curve_scale_dirs(flat, x):
    """Analytic tangent of the curve-scale symmetry of the data term: (alpha_i, c_i) * (1+t), A_.i / (1+t)."""
    prm = flat.params(torch.tensor(x))
    out = []
    for i in range(D_IN):
        v = np.zeros(flat.n)
        a, _, _ = flat.slices["alpha"]; v[a + i] = float(prm["alpha"][i])
        a, _, _ = flat.slices["coef"]; v[a + i * N_RBF:a + (i + 1) * N_RBF] = prm["coef"][i].numpy()
        a, _, shp = flat.slices["mix.weight"]
        for j in range(shp[0]):
            v[a + j * shp[1] + i] = -float(prm["mix.weight"][j, i])
        out.append(v)
    return out

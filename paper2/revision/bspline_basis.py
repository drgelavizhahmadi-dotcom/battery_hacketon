"""paper2/revision/bspline_basis.py: cubic B-spline KAN edges (implementation only; see bspline_prereg.md, 5491481).

Mirrors efficient-kan (Blealtan, KANLinear.b_splines): a uniform grid of G intervals on [-1, 1] extended by k knots beyond each
end (not clamped), Cox-de Boor recursion; G = 5, k = 3 -> 12 knots, 8 basis functions per edge (matching the 8 RBFs of
kan.KANLayer). Edge = w_base * SiLU(x) + sum_j c_j B_j(x). Differences from efficient-kan: per-unit bias in the base Linear (as
kan.KANLayer), no spline_scaler, no grid update, initialisation as kan.KANLayer, regulariser = the committed D2 (+ SiLU, + mu Q).
Knots are computed in the dtype of x (never stored as a float32 buffer), so the float64 polish uses exactly uniform knots."""
import torch
import torch.nn as nn
import torch.nn.functional as F

G, K = 5, 3
N_B = G + K                                    # 8


def knots(dtype=torch.float64):
    h = 2.0 / G
    return torch.arange(-K, G + K + 1, dtype=dtype) * h - 1.0          # t_0 .. t_11 = -2.2, -1.8, ..., 2.2


def greville(dtype=torch.float64):
    t = knots(dtype)
    return torch.stack([t[j + 1:j + K + 1].mean() for j in range(N_B)])   # -1.4, -1.0, ..., 1.4


def basis(x):
    """B_0..B_7 at x (any shape) -> x.shape + (8,), Cox-de Boor as efficient-kan's b_splines"""
    t = knots(x.dtype)
    x = x.unsqueeze(-1)
    b = ((x >= t[:-1]) & (x < t[1:])).to(x.dtype)
    for p in range(1, K + 1):
        b = (x - t[:-(p + 1)]) / (t[p:-1] - t[:-(p + 1)]) * b[..., :-1] + (t[p + 1:] - x) / (t[p + 1:] - t[1:(-p)]) * b[..., 1:]
    return b


class BSplineLayer(nn.Module):
    """Drop-in for kan.KANLayer: the same parameters (base Linear with bias, coef (d_out, d_in, 8)) and methods."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.base = nn.Linear(d_in, d_out)
        self.coef = nn.Parameter(torch.randn(d_out, d_in, N_B) * 0.1)

    def rbf(self, x):                          # name kept for code that reads layer.rbf (degeneracy.hidden); returns the B-spline basis
        return basis(x)

    def forward(self, x):
        return self.base(F.silu(x)) + torch.einsum("bik,oik->bo", basis(x), self.coef)

    def smoothness(self):
        d2 = self.coef[..., 2:] - 2 * self.coef[..., 1:-1] + self.coef[..., :-2]
        return (d2 ** 2).sum()

    def edge(self, i, x):
        return F.silu(x)[:, None] * self.base.weight[:, i] + basis(x) @ self.coef[:, i, :].T


class BSynthKAN(nn.Module):
    """KAN [4, 3, 1] with B-spline edges and tanh between the layers, as synth.SynthKAN."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([BSplineLayer(4, 3), BSplineLayer(3, 1)])

    def forward(self, x):
        return self.layers[1](torch.tanh(self.layers[0](x)))


def d2_matrix(dtype=torch.float64):
    D = torch.zeros(N_B - 2, N_B, dtype=dtype)
    for r in range(N_B - 2):
        D[r, r], D[r, r + 1], D[r, r + 2] = 1, -2, 1
    return D


def null_basis(dtype=torch.float64):
    """QR basis of span{1, j} (as synth.NB / close.null_basis)"""
    Q, _ = torch.linalg.qr(torch.stack([torch.ones(N_B, dtype=dtype), torch.arange(N_B, dtype=dtype)], 1))
    return Q

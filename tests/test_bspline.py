"""Property tests for paper2/revision/bspline_basis.py (gates (a)-(d) of bspline_prereg.md; (e) is descriptive)."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "paper2" / "revision"), str(ROOT / "paper2"), str(ROOT)]
import bspline_basis as B

X = torch.linspace(-1, 1, 10001, dtype=torch.float64)
XK = torch.cat([X, B.knots()[(B.knots() >= -1) & (B.knots() <= 1)]])     # dense grid plus the interior knots exactly


def test_a_partition_of_unity():
    assert float((B.basis(XK).sum(-1) - 1).abs().max()) <= 1e-12


def test_b_exact_affine_reproduction():
    xi = B.greville()
    assert torch.allclose(xi, torch.linspace(-1.4, 1.4, B.N_B, dtype=torch.float64), atol=1e-14, rtol=0)   # equally spaced
    Bm = B.basis(XK)
    for a, b in [(0.0, 1.0), (1.0, 0.0), (-0.7, 2.3), (3.1, -1.9)]:
        assert float((Bm @ (a + b * xi) - (a + b * XK)).abs().max()) <= 1e-12


def test_c_penalty_null_space():
    D = B.d2_matrix()
    for c in (torch.ones(B.N_B, dtype=torch.float64), torch.arange(B.N_B, dtype=torch.float64), 0.3 - 1.7 * torch.arange(B.N_B, dtype=torch.float64)):
        assert float(((D @ c) ** 2).sum()) <= 1e-12
    s = torch.linalg.svdvals(D)
    assert int((s > 1e-10 * s.max()).sum()) == B.N_B - 2               # null space of dimension exactly 2


def test_d_null_basis_orthonormal():
    Q = B.null_basis()
    assert float((Q.T @ Q - torch.eye(2, dtype=torch.float64)).abs().max()) <= 1e-12
    # the same matrix as synth.NB / close.null_basis
    ref, _ = np.linalg.qr(np.column_stack([np.ones(B.N_B), np.arange(B.N_B, dtype=float)]))
    assert np.allclose(Q.numpy(), ref, atol=1e-14)


def affinity_deviation():
    """(e) descriptive: max deviation of each null-space basis function from its least-squares line on [-1, 1]"""
    f = B.basis(X) @ B.null_basis()                                    # (n, 2)
    A = torch.stack([torch.ones_like(X), X], 1)
    out = []
    for m in range(2):
        coef = torch.linalg.lstsq(A, f[:, m:m + 1]).solution
        out.append(float((A @ coef - f[:, m:m + 1]).abs().max()))
    return out


def test_e_affinity_deviation_reported():
    dev = affinity_deviation()
    print(f"(e) null-space basis functions' max deviation from affinity on [-1, 1]: {dev[0]:.2e}, {dev[1]:.2e}")
    assert all(np.isfinite(dev))

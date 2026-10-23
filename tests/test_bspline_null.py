"""Tests for paper2/revision/bspline_null.py (amendment 03638f8): the analytic constant-shift null set and null-set-aware
certification."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "paper2" / "revision"), str(ROOT / "paper2"), str(ROOT / "arch"), str(ROOT), str(ROOT / "tests")]
import bspline as BP
import bspline_basis as BS
import bspline_null as BN
import infimum as I
import synth as SY
from certify import certify_point


def flats():
    d = BP.data("independent")
    T = BP.teacher()
    fs = SY.FlatS(T, d["X"], d["y"], "P1", 0.0)
    net = T.double(); full = dict(net.named_parameters())
    fi = I.Flat(net, full, list(full), I.syn_parts(net, d["X"], d["y"], "P1", 0.0), None, {})
    return fs, fi


def test_basis_dimension_and_orthogonality():
    fs, fi = flats()
    N, C = BN.null_basis_for(fs)
    assert N.shape == (139, 15) and C.shape == (139, 124)
    assert np.abs(N.T @ N - np.eye(15)).max() < 1e-12 and np.abs(C.T @ C - np.eye(124)).max() < 1e-12 and np.abs(N.T @ C).max() < 1e-12
    N2, _ = BN.null_basis_for(fi)
    assert np.abs(N2 @ N2.T - N @ N.T).max() < 1e-12            # same space in both layouts


def test_plain_mode_equals_certify_point():
    fs, _ = flats()
    a, _, ev_a = certify_point(fs, fs.x0)
    b, _, ev_b = BN.certify_na(fs, fs.x0, False)
    for k in ("grad", "mineig", "maxeig", "n_null", "newton", "gnull", "certified"):
        assert a[k] == b[k], k
    assert np.array_equal(ev_a, ev_b)


def test_exact_invariance_and_zero_gradient_along_null_set():
    fs, _ = flats()
    N, _ = BN.null_basis_for(fs)
    L0, g = fs.fun(fs.x0)
    assert np.abs(N.T @ g).max() < 1e-12
    for j in range(15):
        assert abs(float(fs.loss_t(torch.tensor(fs.x0 + 0.1 * N[:, j]))) - L0) / abs(L0) < 1e-12

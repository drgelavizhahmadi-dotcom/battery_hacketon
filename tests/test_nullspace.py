"""Property tests for the smoothness penalty's null space and the null-space penalty Q (descriptive; no training).
Run from the repo root with the pinned interpreter: /usr/bin/python3 -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "paper2", ROOT / "arch"):
    sys.path.insert(0, str(p))

import synth as SY                      # noqa: E402  synthetic study: NB, penalty_terms (D2, SiLU, Q)
import close as CL                      # noqa: E402  real-data null-space penalty (FlatQ)
import sc_model as SC                   # noqa: E402  SC-KAN-delta penalty
from kan import N_RBF                   # noqa: E402

K = N_RBF
D2 = np.zeros((K - 2, K))
for i in range(K - 2):
    D2[i, i:i + 3] = [1, -2, 1]
AFFINE = np.column_stack([np.ones(K), np.arange(K, dtype=float)])


def bases():
    return {"synth.NB": SY.NB.numpy(), "close.NB": np.asarray(CL.NB, dtype=float), "sc_model.NB": SC.NB.numpy()}


@pytest.mark.parametrize("name", ["synth.NB", "close.NB", "sc_model.NB"])
def test_nullspace_basis_orthonormal_and_in_D2_kernel(name):
    nb = bases()[name]
    assert nb.shape == (K, 2)
    assert np.allclose(nb.T @ nb, np.eye(2), atol=1e-12)
    assert np.linalg.norm(D2 @ nb) < 1e-12
    # spans exactly the affine patterns {1, k}
    P = nb @ nb.T
    assert np.allclose(P @ AFFINE, AFFINE, atol=1e-12)


def random_prm(seed=0):
    torch.manual_seed(seed)
    net = SY.SynthKAN().double()
    with torch.no_grad():
        for p in net.parameters():
            p.normal_()
    return {k: v.detach().clone() for k, v in net.named_parameters()}


def test_D2_penalty_unchanged_by_affine_pattern_on_an_edge():
    prm = random_prm(1)
    d2_0, _, _ = SY.penalty_terms(prm)
    rng = np.random.default_rng(0)
    for layer in ("layers.0.coef", "layers.1.coef"):
        p2 = {k: v.clone() for k, v in prm.items()}
        o, i = rng.integers(p2[layer].shape[0]), rng.integers(p2[layer].shape[1])
        a, b = rng.normal(size=2) * 5
        p2[layer][o, i] += torch.tensor(a + b * np.arange(K))
        d2_1, _, _ = SY.penalty_terms(p2)
        assert abs(float(d2_1) - float(d2_0)) <= 1e-10 * max(1.0, float(d2_0))


def test_Q_unchanged_by_curved_null_orthogonal_change():
    prm = random_prm(2)
    _, _, q0 = SY.penalty_terms(prm)
    nb = SY.NB.numpy()
    rng = np.random.default_rng(1)
    for layer in ("layers.0.coef", "layers.1.coef"):
        p2 = {k: v.clone() for k, v in prm.items()}
        v = rng.normal(size=K) * 3
        v -= nb @ (nb.T @ v)                     # remove the affine part: v is curved (null-orthogonal)
        p2[layer][0, 0] += torch.tensor(v)
        _, _, q1 = SY.penalty_terms(p2)
        assert abs(float(q1) - float(q0)) <= 1e-10 * max(1.0, float(q0))


def test_P2_Q_matches_direct_numpy_projection():
    prm = random_prm(3)
    _, _, q = SY.penalty_terms(prm)
    # direct projection onto span{1, k} by least squares, independent of NB
    Pa = AFFINE @ np.linalg.solve(AFFINE.T @ AFFINE, AFFINE.T)
    direct = sum(float(((c.numpy().reshape(-1, K) @ Pa.T) ** 2).sum()) for c in (prm["layers.0.coef"], prm["layers.1.coef"]))
    assert abs(float(q) - direct) <= 1e-10 * max(1.0, direct)

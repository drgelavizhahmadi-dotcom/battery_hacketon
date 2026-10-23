"""paper2/revision/bspline_null.py: null-set-aware certification, degeneracy count and probes for the B-spline study
(POST-HOC amendment 03638f8 to bspline_prereg.md).

The null set is the 15-dimensional constant-shift space of BSynthKAN, constructed analytically from the parameterisation (never
fitted from observed flat directions): per unit o with in-degree m, the m directions (+1 on all 8 spline coefficients of incoming
edge i, -1 on the bias of o), orthonormalised. Exact invariance (partition of unity) implies exactly zero gradient along it.
- certification of the gamma = 0 objectives (P0, P1): certify_point's four criteria evaluated in the complement;
- the degeneracy count (everywhere) and all probes: in the complement;
- P2 and the regularised path: no certification null set (null_aware=False)."""
import time

import numpy as np
import torch

import infimum as I
from certify import EIGREL, GNULL, GTOL, NEWTON

_BASIS = {}


def null_basis_for(flat):
    """(N, C): orthonormal null-set basis (n x 15) and its orthonormal complement (n x (n - 15)), in flat's coordinates"""
    if isinstance(getattr(flat, "meta", None), list) and flat.meta and len(flat.meta[0]) == 3:      # synth.FlatS: (name, shape, k)
        lay, o = [], 0
        for n, shp, k in flat.meta:
            lay.append((n, tuple(shp), o)); o += k
        n_tot = o
    else:                                                                                            # infimum.Flat: (name, shape, a, b)
        lay = [(n, tuple(shp), a) for n, shp, a, b in flat.meta]; n_tot = flat.n
    key = (n_tot, tuple((n, s, a) for n, s, a in lay))
    if key in _BASIS:
        return _BASIS[key]
    pos = {n: (s, a) for n, s, a in lay}
    vecs = []
    for L in (0, 1):
        (d_out, d_in, nb), a_c = pos[f"layers.{L}.coef"]
        _, a_b = pos[f"layers.{L}.base.bias"]
        for u in range(d_out):
            for i in range(d_in):
                v = np.zeros(n_tot)
                v[a_c + (u * d_in + i) * nb + np.arange(nb)] = 1.0
                v[a_b + u] = -1.0
                vecs.append(v)
    A = np.array(vecs).T
    U, s, _ = np.linalg.svd(A, full_matrices=True)
    r = int((s > 1e-10 * s.max()).sum())
    N, C = U[:, :r], U[:, r:]
    _BASIS[key] = (N, C)
    return N, C


def certify_from(g, H, N=None, C=None, gtol=GTOL):
    """certify_point's criteria from a gradient and Hessian; with (N, C) given, evaluated in the complement C (null-set-aware)"""
    gmax = float(np.abs(g).max())
    if C is not None:
        Hc, gc = C.T @ H @ C, C.T @ g
    else:
        Hc, gc = H, g
    ev, Vc = np.linalg.eigh((Hc + Hc.T) / 2)
    lmax = ev[-1]
    null = np.abs(ev) < EIGREL * lmax
    proj = Vc.T @ gc
    newton = float(np.sum(proj[~null] ** 2 / ev[~null])) if (~null).any() else 0.0
    gnull = float(np.linalg.norm(proj[null]))
    crit = dict(a=gmax < gtol, b=bool(ev[0] >= -EIGREL * lmax), c=newton < NEWTON, d=gnull < GNULL)
    out = dict(grad=gmax, mineig=float(ev[0]), maxeig=float(lmax), n_null=int(null.sum()), newton=newton, gnull=gnull, crit=crit,
               certified=all(crit.values()), null_aware=C is not None)
    if N is not None:
        out["grad_along_null"] = float(np.linalg.norm(N.T @ g))
    V = C @ Vc if C is not None else Vc
    return out, V[:, null], ev, V


def certify_na(flat, x, null_aware, gtol=GTOL):
    """certify_point (null_aware=False) or its null-set-aware version (True); returns (cert, V_null, ev) like certify_point"""
    _, g = flat.fun(x)
    H = flat.hessian(x)
    N, C = null_basis_for(flat) if null_aware else (None, None)
    cert, Vn, ev, _ = certify_from(g, H, N, C, gtol)
    return cert, Vn, ev


class CertifiedStop(Exception):
    def __init__(self, x):
        super().__init__("certified")
        self.x = np.asarray(x).copy()


def terminating_hessian(flat, null_aware, gtol=GTOL, on_accept=None):
    """wrap flat.hessian: at every accepted iterate, certify (null-set-aware if requested) from the Hessian already computed;
    raise CertifiedStop at the first iterate that certifies (amendment 03638f8, item 5). on_accept(x) is called first."""
    orig = flat.hessian
    st = dict(last=None)

    def hess(x):
        x = np.asarray(x)
        new = st["last"] is None or not np.array_equal(st["last"], x)
        if new and on_accept is not None:
            on_accept(x)
        H = orig(x)
        if new:
            st["last"] = x.copy()
            _, g = flat.fun(x)
            N, C = null_basis_for(flat) if null_aware else (None, None)
            cert, _, _, _ = certify_from(g, H, N, C, gtol)
            if cert["certified"]:
                raise CertifiedStop(x)
        return H
    return hess


# ------------------------------------------------------------------ degeneracy count and probes in the complement
def complement_eig(flat, x):
    H = flat.hessian(x)
    _, C = null_basis_for(flat)
    ev, Vc = np.linalg.eigh(C.T @ H @ C)
    return ev, C @ Vc, C


def probes_c(flat, x, ev, V, C, restart_fn):
    """infimum.probes with the flat-direction search on complement eigenvectors and the scaling probes' displacement projected
    onto the complement; classification and stalled-restart flag unchanged"""
    tp = time.monotonic()
    L0, th0 = flat.loss(x), float((x ** 2).sum())
    rel = lambda L: (L - L0) / abs(L0)
    Pc = lambda y: x + C @ (C.T @ (y - x))
    found = []

    def test(tag, y):
        th = float((y ** 2).sum()); r = rel(flat.loss(y))
        if th > th0 and r < -I.REL:
            found.append(dict(probe=tag, rel_loss=r, rel_theta=(th - th0) / th0))
        return r
    sc = {}
    for kind, fn in (("affine", I.affine_scaled), ("all_L1", I.l1_scaled)):
        sc[kind] = [test(f"scale {kind} c={c}", Pc(fn(flat, x, c))) for c in I.CS]
    order = np.argsort(np.abs(ev))[:I.NFLAT]
    nx = float(np.sqrt(th0))
    fs = []
    for j in order:
        for sg in (1, -1):
            fs.append(dict(eig=float(ev[j]), sign=sg, rel=[test(f"flat eig {ev[j]:+.1e} sign {sg:+d} s={s:.3g}", x + sg * s * nx * V[:, j]) for s in I.STEPS]))
    tr = time.monotonic()
    xs, info = restart_fn(x)
    tr = time.monotonic() - tr
    L1, th1 = flat.loss(xs), float((xs ** 2).sum())
    rs = dict(stop=info["stop"], iters=info["iters"], wall_s=info["wall_s"], rel_loss=rel(L1), rel_theta=(th1 - th0) / th0,
              certified=bool(info["cert"]["certified"]))
    rs["lowers_and_grows"] = bool(rs["rel_loss"] < -I.REL and rs["rel_theta"] > I.REL)
    capped = "cap" in rs["stop"] or "hang" in rs["stop"]
    rs["stalled"] = bool(capped and abs(rs["rel_loss"]) < I.REL and abs(rs["rel_theta"]) < I.REL)
    rs["cap_type"] = ("iteration" if "iteration" in rs["stop"] else "wall-clock") if capped else None
    cls = "PLATEAU" if (rs["lowers_and_grows"] or found) else "FLAT MINIMUM"
    best = min(found, key=lambda d: d["rel_loss"]) if found else None
    return dict(cls=cls, restart=rs, scaling=sc, flat_search=fs, n_descents=len(found), best_descent=best,
                min_rel_scaling=float(min(min(v) for v in sc.values())), min_rel_flat=float(min(min(d["rel"]) for d in fs)),
                compute_s=time.monotonic() - tp - tr, in_complement=True)


def analyse_c(flat, net, zin, xk, info, spec, stem, wlog, run_point):
    """infimum.analyse with the degeneracy count and probes in the complement of the constant-shift space; info["cert"] is the
    certification decided by the caller (null-set-aware for gamma = 0 objectives)"""
    rec = dict(status="reached", stop=info["stop"], iters=info["iters"], wall_s=info["wall_s"], certified=bool(info["cert"]["certified"]),
               crit=info["cert"]["crit"], grad=info["cert"]["grad"], resumed=info.get("resumed", False),
               cert_null_aware=bool(info["cert"].get("null_aware", False)),
               theta2=float((xk ** 2).sum()), blocks=I.block_norms(flat, xk), Lstar=flat.Lstar(xk), terms=flat.terms(xk))
    if rec["certified"]:
        ev, V, C = complement_eig(flat, xk)
        near = np.abs(ev) < I.NEAR * ev[-1]
        rec.update(n_near_flat=int(near.sum()), eig_deciles=[float(q) for q in np.quantile(ev, np.linspace(0, 1, 11))],
                   min_max=float(ev[0] / ev[-1]), sat=I.saturation(flat, net, zin, xk), degeneracy_in_complement=True)
        rec["degenerate"] = bool(rec["n_near_flat"] > I.DEG_NEAR or rec["sat"] > I.DEG_SAT)
        if rec["degenerate"]:
            def restart_fn(xc):
                rstart = I.W / f"{stem}_rs_start.npy"
                if not rstart.exists():
                    np.save(rstart, xc)
                sp = dict(spec, start=str(rstart), name=f"{stem}_rs.pt", ck=str(I.W / f"{stem}_rs_ck.npz"), gtol=I.GRAD_TOL_RS)
                return run_point(sp, flat, wlog)
            pr, _ = I.cached(f"{stem}_probes.pt", lambda: probes_c(flat, xk, ev, V, C, restart_fn))
            rec["probes"] = pr
            rec["class"] = pr["cls"]
            rec["included"] = pr["cls"] == "FLAT MINIMUM"
            if pr["cls"] == "PLATEAU":
                rec["reason"] = "PLATEAU by probe"
            rec["wall_s"] += pr["restart"]["wall_s"] + pr["compute_s"]
        else:
            rec["class"] = None; rec["included"] = True
    else:
        rec.update(degenerate=None, included=False, reason=I.uncert_reason(info)); rec["class"] = None
    return rec

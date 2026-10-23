"""paper2/revision/bspline_nullspace_diag.py: DESCRIPTIVE diagnostic after the Step 0 (ii) stop (no training; changes no record).
Is there an exact constant-shift symmetry of the B-spline objective (partition of unity: a constant added to all 8 coefficients of
an edge shifts the edge function by that constant, absorbed by the unit's bias), and does it explain the near-null Hessian
eigenspace and the hangs? Run from the repo root with the pinned interpreter:
  /usr/bin/python3 paper2/revision/bspline_nullspace_diag.py >> paper2/revision/bspline_output.txt"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "paper2", REPO / "arch", HERE, REPO / "tests"):
    sys.path.insert(0, str(p))

import numpy as np
import torch

import bspline as BP
import bspline_basis as BS
import infimum as I
import synth as SY

NEAR = 1e-8                      # near-null threshold, as certify.py (|lambda| < 1e-8 x largest)
EXACT_TOL = 1e-12                # (i) float-level: relative loss change <= 1e-12 at both steps (fixed before computing)
OVERLAP_TOL = 0.999              # (ii) cosine of every principal angle >= 0.999 (fixed before computing)
STEPS = (1e-3, 1e-1)


def log(*a):
    print(*a, flush=True)


def shift_basis(fl):
    """orthonormal basis of the exact constant-shift space in FlatS coordinates: per unit o with in-degree m, the m directions
    (+1 on all 8 coefficients of incoming edge i, -1 on bias o); dimension = sum of in-degrees"""
    sl, o = {}, 0
    for n, shp, k in fl.meta:
        sl[n] = (o, shp); o += k
    vecs, labels = [], []
    for L in (0, 1):
        a_c, shp_c = sl[f"layers.{L}.coef"]; a_b, _ = sl[f"layers.{L}.base.bias"]
        d_out, d_in, nb = shp_c
        for u in range(d_out):
            for i in range(d_in):
                v = np.zeros(fl.n)
                idx = a_c + (u * d_in + i) * nb + np.arange(nb)
                v[idx] = 1.0; v[a_b + u] = -1.0
                vecs.append(v); labels.append(f"L{L + 1} unit {u} edge {i}")
    Q, _ = np.linalg.qr(np.array(vecs).T)
    return Q, labels, sl


def exactness(fl, x, Q):
    L0 = fl.fun(x)[0]
    out = []
    for s in STEPS:
        rel = [abs(float(fl.loss_t(torch.tensor(x + s * Q[:, j]))) - L0) / abs(L0) for j in range(Q.shape[1])]
        out.append(dict(step=s, max=float(max(rel)), median=float(np.median(rel))))
    return out


def near_null(fl, x):
    ev, V = np.linalg.eigh(fl.hessian(x))
    m = np.abs(ev) < NEAR * ev[-1]
    return ev, V, m


def main():
    log("\n" + "#" * 100)
    log(f"paper2/revision/bspline_nullspace_diag.py {time.strftime('%Y-%m-%d %H:%M:%S')}: DESCRIPTIVE diagnostic after the Step 0 (ii) stop")
    log("  (no training; no mu calibration, Step 1 or optimisation; changes no record)")
    log("#" * 100)
    log("  stated before results: the exact constant-shift space has dimension sum of in-degrees = 3 x 4 + 1 x 3 = 15; confirmation needs")
    log(f"  (i) relative loss change <= {EXACT_TOL:g} along every basis direction at steps {STEPS} at all tested points, (ii) every principal")
    log(f"  angle cosine between the constructed space and the near-null eigenspace >= {OVERLAP_TOL} with near-null count = 15, (iii) the RBF")
    log("  shift only approximate, (iv) both hangs at gradients below the certification threshold (max|g| < 1e-7)")
    log("  inputs: the four Step 0 (ii) end points from the SHA-indexed cache (gauge_weights/bsp_teacher_*.pt; verified against")
    log("  sc_index.json on load; not yet archived)")
    pts = {}
    for c, _ in BP.CONDS:
        for lev in ("P0", "P1"):
            sp = dict(mode="teacher", cond=c, lev=lev, mu=0.0, name=f"bsp_teacher_{c}_{lev}.pt")
            obj, _ = I.cached(sp["name"], lambda: (_ for _ in ()).throw(RuntimeError(sp["name"])))
            pts[f"{c}_{lev}"] = (BP.teacher_flat(sp), I.npx(obj["x"]), obj)
    torch.manual_seed(0)
    R = BS.BSynthKAN().double()
    d = BP.data("correlated")
    for lev in ("P0", "P1"):
        fl = SY.FlatS(R, d["X"], d["y"], lev, 0.0)
        pts[f"random_init_{lev}"] = (fl, fl.x0.copy(), None)
    Q, labels, sl = shift_basis(next(iter(pts.values()))[0])
    log(f"\n  constructed space: {Q.shape[1]} orthonormal directions (||Q'Q - I||_max {np.abs(Q.T @ Q - np.eye(Q.shape[1])).max():.1e})")
    res, ok_i, ok_ii = {}, True, True
    log("\n==================== (1)+(2) near-null Hessian eigenvalues, exactness and completeness ====================")
    for name, (fl, x, obj) in pts.items():
        ev, V, m = near_null(fl, x)
        ex = exactness(fl, x, Q)
        Vn = V[:, m]
        cos_in = np.linalg.svd(Q.T @ Vn, compute_uv=False) if m.any() else np.zeros(0)       # cosines of principal angles
        k = int(m.sum())
        r = dict(n_near=k, near_vals=ev[m].tolist(), maxeig=float(ev[-1]), exact=ex, cos=cos_in.tolist(),
                 cos_min=(float(cos_in.min()) if len(cos_in) else None))
        ok_i &= all(e["max"] <= EXACT_TOL for e in ex)
        ok_ii &= (k == Q.shape[1] and len(cos_in) == Q.shape[1] and cos_in.min() >= OVERLAP_TOL)
        if k > Q.shape[1]:
            P = Vn - Q @ (Q.T @ Vn)                                   # parts of the near-null space outside the constructed space
            U, s, _ = np.linalg.svd(P, full_matrices=False)
            extra = U[:, s > 1e-6]
            loads = {}
            for n, (a, shp) in sl.items():
                b = a + int(np.prod(shp))
                loads[n] = float((extra[a:b] ** 2).sum() / max(extra.shape[1], 1))
            coef_pos = {}
            for L in (0, 1):
                a, shp = sl[f"layers.{L}.coef"]
                blk = (extra[a:a + int(np.prod(shp))] ** 2).reshape(*shp, -1).sum(-1).sum(axis=(0, 1))
                coef_pos[f"L{L + 1}"] = (blk / max(blk.sum(), 1e-300)).tolist()
            r["extra"] = dict(n=int(extra.shape[1]), block_loadings=loads, basis_position_share=coef_pos)
        res[name] = r
        log(f"  {name:18s}: near-null {k:2d} (|lambda| < 1e-8 x {ev[-1]:.3e}); values " + (", ".join(f"{v:+.1e}" for v in ev[m][:20]) if k else "-")
            + (" ..." if k > 20 else ""))
        log(f"  {'':18s}  exactness: step 1e-3 max rel. loss change {ex[0]['max']:.1e} (median {ex[0]['median']:.1e}); step 1e-1 max {ex[1]['max']:.1e} "
            f"(median {ex[1]['median']:.1e}) | principal-angle cosines (constructed vs near-null): "
            + (f"min {cos_in.min():.6f}, max {cos_in.max():.6f} over {len(cos_in)}" if len(cos_in) else "none (no near-null directions)"))
        if "extra" in r:
            e = r["extra"]
            log(f"  {'':18s}  {e['n']} EXTRA near-null directions outside the constructed space (NOT in the null set): block loadings "
                + ", ".join(f"{n} {v:.2f}" for n, v in e["block_loadings"].items() if v > 0.01)
                + "; share by basis-function index L1 " + " ".join(f"{v:.2f}" for v in e["basis_position_share"]["L1"])
                + " | L2 " + " ".join(f"{v:.2f}" for v in e["basis_position_share"]["L2"]))
    log("\n==================== (3) RBF contrast: the same shifts at the committed synthetic teacher (34cc321) ====================")
    Tr = SY.build_teacher(float(json_res()["teacher_scale"]))
    y0r = SY.predict(Tr, d["X"]); sig_r = 0.05 * y0r.std()
    yr = y0r + sig_r * np.random.default_rng(5001).standard_normal(SY.N)          # the committed correlated labels (34cc321 rule)
    flr = SY.FlatS(Tr, d["X"], yr, "P0", 0.0)
    exr = exactness(flr, flr.x0.copy(), Q)
    g = torch.linspace(-1, 1, 10001, dtype=torch.float64)
    pu = Tr.double().layers[0].rbf(g).sum(-1)
    res["rbf"] = dict(exact=exr, rbf_sum_range=[float(pu.min()), float(pu.max())])
    log(f"  RBF teacher, P0 objective on the correlated inputs (the committed correlated labels: RBF teacher + noise, seed 5001): step 1e-3 max rel. loss change "
        f"{exr[0]['max']:.1e} (median {exr[0]['median']:.1e}); step 1e-1 max {exr[1]['max']:.1e}; the 8 RBFs sum to {float(pu.min()):.3f}-{float(pu.max()):.3f} "
        f"on [-1, 1] (no partition of unity)")
    bsp_max = max(e["max"] for r in res.values() if "exact" in r and isinstance(r, dict) and "n_near" in r for e in r["exact"])
    ok_iii = all(e["max"] > 1e3 * max(bsp_max, 1e-300) for e in exr)
    log(f"  contrast: RBF / B-spline worst relative change >= 1e3 at both steps -> {ok_iii} (B-spline worst {bsp_max:.1e})")
    log("\n==================== (4) gradient at the hung polishes' last accepted iterates ====================")
    ok_iv = True
    for name in ("independent_P0", "independent_P1"):
        fl, x, obj = pts[name]
        _, gr = fl.fun(x)
        gm = float(np.abs(gr).max())
        res[name]["grad_max"] = gm; res[name]["grad_norm"] = float(np.linalg.norm(gr))
        ok_iv &= gm < 1e-7
        log(f"  {name}: max|g| {gm:.2e}, ||g|| {np.linalg.norm(gr):.2e} (certification threshold max|g| < 1e-7) -> {'below' if gm < 1e-7 else 'NOT below'}")
    log("\n==================== (5) P2 objective along the constructed directions (descriptive) ====================")
    fl0, x0, _ = pts["independent_P0"]
    for mu in (1e-5, 1e-4):
        sp = dict(cond="independent", lev="P2", mu=mu)
        f2 = BP.teacher_flat(sp)
        H2 = f2.hessian(x0); H1 = fl0.hessian(x0)
        qf2 = np.einsum("ij,jk,ki->i", Q.T, H2, Q); qf1 = np.einsum("ij,jk,ki->i", Q.T, H1, Q)
        ev2 = np.linalg.eigvalsh(H2)
        n2 = int((np.abs(ev2) < NEAR * ev2[-1]).sum())
        res.setdefault("p2", {})[f"{mu:g}"] = dict(qf=qf2.tolist(), qf_p0=qf1.tolist(), n_near=n2)
        log(f"  mu {mu:g}: v'Hv along the 15 directions {qf2.min():.2e}-{qf2.max():.2e} (median {np.median(qf2):.2e}; / mu {np.median(qf2) / mu:.2f}); "
            f"under P0 {np.abs(qf1).max():.1e}; P2 near-null count {n2}")
    conf = dict(i=bool(ok_i), ii=bool(ok_ii), iii=bool(ok_iii), iv=bool(ok_iv))
    log(f"\n  CONFIRMATION CRITERIA (fixed before results): (i) exactness {conf['i']}, (ii) completeness {conf['ii']}, (iii) RBF approximate only "
        f"{conf['iii']}, (iv) hangs below the gradient threshold {conf['iv']} -> {'SYMMETRY CONFIRMED' if all(conf.values()) else 'NOT ALL HOLD: report and stop'}")
    res["confirmation"] = conf
    import json
    (HERE / "bspline_nullspace_diag.json").write_text(json.dumps(res, indent=1, default=float))
    log("  wrote bspline_nullspace_diag.json")


def json_res():
    import json
    return json.loads((REPO / "paper2/synth_results.json").read_text())


if __name__ == "__main__":
    main()

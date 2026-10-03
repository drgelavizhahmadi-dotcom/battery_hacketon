"""certify: second-order certification of close.py's final points, analysis at certified minima, and
fold A accuracy at mu = 1e-4 (see certify_prereg.md). Run: python3 certify.py >> certify_output.txt
Weights: gauge_weights/certify_* (not committed). Inputs held in memory after SHA verification."""
import io
import itertools
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from anchor import rmse
from close import FlatQ, block_norms, cross_edge
from gauge import Gauge, align, corr, grid_inputs, parts, setup
from kan import KAN, N_RBF, delta_head, predict, subset, to_batch, train
from stage2 import extract_verified, in_memory, run_exact
from support import design_matrix

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
ROOT = Path(__file__).parent
FIG = ROOT / "figs" / "certify"
W_DIR = ROOT / "gauge_weights"
MUS = ["1e-06", "1e-05", "0.0001"]
MUV = {"1e-06": 1e-6, "1e-05": 1e-5, "0.0001": 1e-4}
IND, STU, FOLD = list(range(6)), [115, 104, 114], list(range(5))
GTOL, EIGREL, NEWTON, GNULL = 1e-7, 1e-8, 1e-16, 1e-9
RESULTS = {}


def log(*a):
    print(*a, flush=True)


def certify_point(flat, x):
    L, g = flat.fun(x)
    ev, V = np.linalg.eigh(flat.hessian(x))
    lmax = ev[-1]
    null = np.abs(ev) < EIGREL * lmax
    proj = V.T @ g
    newton = float(np.sum(proj[~null] ** 2 / ev[~null])) if (~null).any() else 0.0
    gnull = float(np.linalg.norm(proj[null]))
    gmax = float(np.abs(g).max())
    crit = dict(a=gmax < GTOL, b=bool(ev[0] >= -EIGREL * lmax), c=newton < NEWTON, d=gnull < GNULL)
    return dict(grad=gmax, mineig=float(ev[0]), maxeig=float(lmax), n_null=int(null.sum()), newton=newton, gnull=gnull,
                crit=crit, certified=all(crit.values())), V[:, null], ev


def gauge_directions(flat, x, Z, sidx, d):
    """Least-squares constant-shift and unit-rescaling directions in the flat 609-vector (E4), with residuals."""
    sl = flat.slices
    net = flat.to_kan(x, d, len(flat.full["emb.weight"]))
    n = flat.n
    g = torch.linspace(-1, 1, N_RBF, dtype=torch.float64); h = 2 / (N_RBF - 1)

    def basis(u):
        return np.column_stack([u / (1 + np.exp(-u))] + [np.exp(-(((u - gk) / h) ** 2)) for gk in g.numpy()])

    def idx(name, *ix):
        a, b_, shp = sl[name]
        return a + int(np.ravel_multi_index(ix, shp))

    with torch.no_grad():
        zin = torch.cat([torch.tensor(Z, dtype=torch.float64), torch.tanh(net.emb(torch.tensor(sidx)))], 1)
        s = net.layers[0](zin)
    out = {"const_L1": [], "const_L2": [], "rescale": []}
    for j in range(6):
        for i in range(d):  # continuous layer-1 edges: represent +1 with edge (i, j), compensate with bias_j
            B = basis(Z[:, i]); cf, *_ = np.linalg.lstsq(B, np.ones(len(B)), rcond=None)
            v = np.zeros(n)
            v[idx("layers.0.base.weight", j, i)] = cf[0]
            for k in range(N_RBF):
                v[idx("layers.0.coef", j, i, k)] = cf[1 + k]
            v[idx("layers.0.base.bias", j)] = -1.0
            out["const_L1"].append((v, float(np.sqrt(np.mean((B @ cf - 1) ** 2)))))
        u = torch.tanh(s[:, j]).numpy()
        B = basis(u); cf, *_ = np.linalg.lstsq(B, np.ones(len(B)), rcond=None)
        v = np.zeros(n)
        v[idx("layers.1.base.weight", 0, j)] = cf[0]
        for k in range(N_RBF):
            v[idx("layers.1.coef", 0, j, k)] = cf[1 + k]
        v[idx("layers.1.base.bias", 0)] = -1.0
        out["const_L2"].append((v, float(np.sqrt(np.mean((B @ cf - 1) ** 2)))))
        # rescaling of unit j: incoming params * a, compensate the outgoing edge for -psi'(s) * s
        v = np.zeros(n)
        for i in range(d + 2):
            v[idx("layers.0.base.weight", j, i)] = float(net.layers[0].base.weight[j, i])
            for k in range(N_RBF):
                v[idx("layers.0.coef", j, i, k)] = float(net.layers[0].coef[j, i, k])
        v[idx("layers.0.base.bias", j)] = float(net.layers[0].base.bias[j])
        sj = s[:, j].clone().requires_grad_(True)
        psi = net.layers[1].edge(j, torch.tanh(sj))[:, 0]
        (dpsi,) = torch.autograd.grad(psi.sum(), sj)
        target = -(dpsi * s[:, j]).detach().numpy()
        cf, *_ = np.linalg.lstsq(B, target, rcond=None)
        v[idx("layers.1.base.weight", 0, j)] = cf[0]
        for k in range(N_RBF):
            v[idx("layers.1.coef", 0, j, k)] = cf[1 + k]
        res = float(np.sqrt(np.mean((B @ cf - target) ** 2)) / max(np.sqrt(np.mean(target ** 2)), 1e-300))
        out["rescale"].append((v, res))
    return out


def l1_blocks(flat, v, d):
    """Unit-wise layer-1 continuous blocks (72 entries: [base_w, rbf x8] per continuous input)."""
    a_c, _, shp_c = flat.slices["layers.0.coef"]; a_w, _, shp_w = flat.slices["layers.0.base.weight"]
    I = shp_w[1]
    return [np.concatenate([np.r_[v[a_w + j * I + i], v[a_c + (j * I + i) * N_RBF:a_c + (j * I + i + 1) * N_RBF]] for i in range(d)]) for j in range(6)]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    tr, te, te_anc, D, _ = setup()
    d, m = D["d_num"], D["train"]
    Ztr, str_ = D["Z"][m], D["s"][m]
    b = subset(D["batch"], m)
    bd = {k: (v.double() if v.is_floating_point() else v) for k, v in b.items()}
    used = np.unique(str_).tolist()
    log("DISCREPANCIES / choices (files win; see certify_prereg.md):")
    log("  F-a certification criterion defined AFTER the stage2/close.py criterion proved unreachable; reported alongside close.py V1 (failed)")
    log("  F-b W4/W5 set after close.py's mu=1e-6 fold A result; F-c W6 has 10 points (weak test)")
    log("  E3 'fiber.py procedure' = unit match + control-3 edge match, as in close.py/trust.py")
    log("  E4 gauge directions are least-squares (not exact symmetries here); residuals reported")

    names = ([f"close_{mu}_{t}.pt" for mu in MUS for t in [f"ind_{i}" for i in IND] + [f"stu_{s}" for s in STU]] +
             [f"close_foldA_1e-06_s{s}.pt" for s in FOLD] + [f"delta_s{i}.pt" for i in IND] +
             [f"control3_R_s{s}.pt" for s in STU] + [f"fiber2_ind_{i}.pt" for i in IND])
    log("\n==================== inputs (SHA-256 verified, held in memory) ====================")
    blobs = in_memory(extract_verified(names))

    def kan_from(blob):
        k = KAN(d, [6, 1], D["n_salt"])
        miss, unexp = k.load_state_dict({n: t.float() for n, t in torch.load(io.BytesIO(blob)).items()}, strict=False)
        assert not unexp and all(x.endswith(".grid") for x in miss), (miss, unexp)
        return k.eval()

    teacher = kan_from(blobs["delta_s0.pt"])
    yT = torch.tensor(predict(teacher, delta_head, to_batch(Ztr, str_, x_co=b["x_co"].numpy())), dtype=torch.float64)
    models = [(f"ind_{i}", f"delta_s{i}.pt", bd["y"]) for i in IND] + [(f"stu_{s}", f"control3_R_s{s}.pt", yT) for s in STU]

    # ---------------- Step 1: candidates first
    pts = {}
    for mu in MUS:
        for tag, start, y in models:
            bb = dict(bd); bb["y"] = y
            flat = FlatQ(kan_from(blobs[start]), bb, used, MUV[mu])
            x = torch.load(io.BytesIO(blobs[f"close_{mu}_{tag}.pt"])).numpy()
            pts[(mu, tag)] = (flat, x, float(np.abs(flat.fun(x)[1]).max()))
    cands = [k for k, v in pts.items() if v[2] < GTOL]
    log(f"\n==================== STEP 1: candidates (recomputed max|grad| < 1e-7): {len(cands)} ====================")
    for mu, tag in cands:
        log(f"  mu {mu:7s} {tag:8s} max|grad| {pts[(mu, tag)][2]:.2e}")
    log("  non-candidates: " + ", ".join(f"{mu}/{t} ({pts[(mu, t)][2]:.1e})" for (mu, t) in pts if (mu, t) not in cands))

    log("\n  certification (a) max|grad| < 1e-7, (b) min eig >= -1e-8 x max, (c) Newton decrement on non-null < 1e-16, (d) ||g_null|| < 1e-9:")
    cert, nulls = {}, {}
    for key in cands:
        flat, x, _ = pts[key]
        r, Vn, ev = certify_point(flat, x)
        cert[key], nulls[key] = r, Vn
        log(f"  mu {key[0]:7s} {key[1]:8s} certified {str(r['certified']):5s} | grad {r['grad']:.1e} min eig {r['mineig']:.2e} max {r['maxeig']:.2e} "
            f"null set {r['n_null']:3d} | Newton {r['newton']:.1e} g_null {r['gnull']:.1e} | " +
            " ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in r["crit"].items()))
    counts = {mu: (sum(cert[k]["certified"] for k in cert if k[0] == mu and k[1].startswith("ind")),
                   sum(cert[k]["certified"] for k in cert if k[0] == mu and k[1].startswith("stu"))) for mu in MUS}
    log("  certified per mu: " + "; ".join(f"mu {mu}: {a}/6 seeds, {c}/3 students" for mu, (a, c) in counts.items()))

    # ---------------- Step 2a / 2b
    log("\n==================== STEP 2: analysis at certified minima ====================")
    sd_y = bd["y"].std().item()
    capped = {i: predict(kan_from(blobs[f"fiber2_ind_{i}.pt"]), delta_head, b) for i in IND}
    after = {}
    for mu in MUS:
        cs = [int(t.split("_")[1]) for (mm, t) in cert if mm == mu and t.startswith("ind") and cert[(mm, t)]["certified"]]
        if len(cs) >= 2:
            dl = {i: predict(pts[(mu, f"ind_{i}")][0].to_kan(pts[(mu, f"ind_{i}")][1], d, D["n_salt"]).float(), delta_head, b) for i in cs}
            pairs = list(itertools.combinations(cs, 2))
            dab = [float(np.sqrt(np.mean((dl[A] - dl[B]) ** 2)) / sd_y) for A, B in pairs]
            dab0 = [float(np.sqrt(np.mean((capped[A] - capped[B]) ** 2)) / sd_y) for A, B in pairs]
            after[mu] = dict(seeds=cs, dAB=dab, dAB_capped=dab0)
            log(f"  (a) mu {mu}: certified seeds {cs}; d_AB median {np.median(dab):.4f} (capped, same pairs {np.median(dab0):.4f})")
        else:
            log(f"  (a) mu {mu}: {len(cs)} certified seed(s); d_AB not computed")
    sweeps = grid_inputs(D)
    gT = Gauge(teacher, Ztr, str_, d)
    varT = gT.fixed(parts(teacher, Ztr, str_, d)[0]).var(1)
    active = varT >= 0.05 * varT.max(0, keepdims=True)
    edgeT = [gT.fixed(parts(teacher, *sweeps[i], d)[0])[i] for i in range(d)]
    stu_rows = []
    for (mu, t), r in cert.items():
        if not (t.startswith("stu") and r["certified"]):
            continue
        net = pts[(mu, t)][0].to_kan(pts[(mu, t)][1], d, D["n_salt"]).float()
        g = Gauge(net, Ztr, str_, d)
        um = np.abs(align(gT, g))
        pe = [abs(corr(g.fixed(parts(net, *sweeps[i], d)[0])[i][:, j], edgeT[i][:, j])) for i in range(d) for j in range(6) if active[i, j]]
        pc = corr(predict(net, delta_head, b), yT.numpy())
        stu_rows.append(dict(mu=mu, student=t, pred_corr=pc, units=um.tolist(), unit=float(np.median(um)), edge=float(np.median(pe))))
        log(f"  (b) mu {mu} {t}: delta corr {pc:.6f}; unit match median {np.median(um):.3f} {np.round(um, 3).tolist()}; edge |corr| median {np.median(pe):.3f}")

    # ---------------- Step 2c
    log("\n  (c) null-set overlaps (exploratory): squared overlap of each normalised gauge direction with the null set; "
        "N-overlap of null eigenvectors' layer-1 blocks")
    M = design_matrix(Ztr)
    _, S, Vt = np.linalg.svd(M, full_matrices=False)
    VN = Vt[S / S[0] < 1e-2].T
    nullrows = []
    for key in cert:
        if not cert[key]["certified"]:
            continue
        flat, x, _ = pts[key]
        Vn = nulls[key]
        gd = gauge_directions(flat, x, Ztr, str_, d)
        row = dict(mu=key[0], model=key[1], n_null=Vn.shape[1])
        for fam, items in gd.items():
            ov = [float(np.sum((Vn.T @ (v / np.linalg.norm(v))) ** 2)) for v, _ in items]
            row[fam] = (float(np.median(ov)), float(min(ov)), float(max(ov)), float(np.median([r_ for _, r_ in items])))
        shares, ovs = [], []
        for q in range(Vn.shape[1]):
            blks = l1_blocks(flat, Vn[:, q], d)
            tot = sum(np.sum(bl ** 2) for bl in blks)
            shares.append(tot); ovs.append(sum(np.sum((VN.T @ bl) ** 2) for bl in blks) / tot if tot > 1e-30 else np.nan)
        shares, ovs = np.array(shares), np.array(ovs)
        ok = shares > 1e-30
        row["N"] = (float(np.median(shares)), float(np.nansum(shares[ok] * ovs[ok]) / shares[ok].sum()) if ok.any() else float("nan"))
        nullrows.append(row)
        log(f"    mu {key[0]:7s} {key[1]:8s} null {row['n_null']:3d} | const L1 {row['const_L1'][0]:.3f} [{row['const_L1'][1]:.3f}-{row['const_L1'][2]:.3f}] (res {row['const_L1'][3]:.1e}) "
            f"| const L2 {row['const_L2'][0]:.3f} [{row['const_L2'][1]:.3f}-{row['const_L2'][2]:.3f}] (res {row['const_L2'][3]:.1e}) "
            f"| rescale {row['rescale'][0]:.3f} [{row['rescale'][1]:.3f}-{row['rescale'][2]:.3f}] (rel res {row['rescale'][3]:.1e}) "
            f"| layer-1 block share {row['N'][0]:.3f}, share-weighted N-overlap {row['N'][1]:.3f} (random {VN.shape[1] / 72:.2f})")

    # ---------------- Step 3
    log("\n==================== STEP 3: fold A accuracy at mu = 1e-4 ====================")
    ho = (tr.combo == "EA+PC").values & m
    dt = m & ~ho
    bt, bh = subset(D["batch"], dt), subset(D["batch"], ho)
    btd = {k: (v.double() if v.is_floating_point() else v) for k, v in bt.items()}
    used_f = np.unique(D["s"][dt]).tolist()
    yh, anc = tr.log_k.values[ho], tr.anchor.values[ho]
    rows3, P = [], {"1e-06": [], "0.0001": []}
    for mu in ["1e-06", "0.0001"]:
        for s in FOLD:
            net0 = train([6, 1], d, D["n_salt"], delta_head, bt, epochs=107, seed=s)[0]
            flat = FlatQ(net0, btd, used_f, MUV[mu])
            if mu == "1e-06":
                x = torch.load(io.BytesIO(blobs[f"close_foldA_1e-06_s{s}.pt"])).numpy(); stop = "close.py cache"; cached = True
            else:
                cache, meta = W_DIR / f"certify_foldA_0.0001_s{s}.pt", W_DIR / f"certify_foldA_0.0001_s{s}.json"
                cached = cache.exists() and meta.exists()
                if cached:
                    x = torch.load(cache).numpy(); info = json.loads(meta.read_text())
                else:
                    x, info, _ = run_exact(f"foldA s{s}", flat)
                    torch.save(torch.tensor(x), cache); meta.write_text(json.dumps(info, default=float))
                stop = f"{info['stop']} after {info['iters']} iters"
            net = flat.to_kan(x, d, D["n_salt"])
            p = predict(net.float(), delta_head, bh); P[mu].append(p)
            ce, sat = cross_edge(net.double(), btd)
            th = sum(block_norms(net.double(), used_f).values())
            row = dict(mu=mu, seed=s, rmse=rmse(anc + p, yh), bias=float(np.mean(anc + p - yh)), cross_edge=ce, theta2=th, saturated=sat, stop=stop)
            if mu == "0.0001":
                rc, _, _ = certify_point(flat, x)
                row["certified"] = rc["certified"]; row["crit"] = rc["crit"]
            rows3.append(row)
            log(f"  mu {mu:7s} seed {s}: {stop}{'  [cached]' if cached and mu == '0.0001' else ''} | held-out RMSE {row['rmse']:.4f} (bias {row['bias']:+.3f}) | "
                f"cross-edge {ce:.2f} | ||theta||^2 {th:.1f} | saturated {sat:.3f}" +
                (f" | Step-1 certified {row['certified']} " + " ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in row['crit'].items()) if mu == "0.0001" else ""))
    ens = {mu: (rmse(anc + np.mean(P[mu], 0), yh), float(np.mean(anc + np.mean(P[mu], 0) - yh))) for mu in P}
    log(f"  ensemble fold A RMSE: mu 1e-4 {ens['0.0001'][0]:.4f} (bias {ens['0.0001'][1]:+.4f}); mu 1e-6 {ens['1e-06'][0]:.4f} "
        f"(bias {ens['1e-06'][1]:+.4f}; close.py reported 1.0262); KAN-delta 0.1420")
    rho, pval = spearmanr([r["cross_edge"] for r in rows3], [r["rmse"] for r in rows3])
    log(f"  W6: Spearman(cross-edge ratio, held-out RMSE) over {len(rows3)} models: rho {rho:+.3f}, p {pval:.3f}")

    # ---------------- verdicts
    c4 = [k for k in cert if k[0] == "0.0001" and k[1].startswith("ind") and cert[k]["certified"]]
    v = {"W1": ("held" if len(c4) >= 5 else "failed") + f" ({len(c4)}/6 seeds certified at mu 1e-4)"}
    a4 = after.get("0.0001")
    v["W2"] = (("held" if np.median(a4["dAB"]) <= 0.5 * np.median(a4["dAB_capped"]) else "failed") +
               f" (median {np.median(a4['dAB']):.4f} vs 0.5 x {np.median(a4['dAB_capped']):.4f})") if a4 else \
        f"inconclusive ({len(c4)} certified seed(s) at mu 1e-4)"
    um = [u for r in stu_rows for u in r["units"]]
    v["W3"] = (("held" if np.median(um) >= 0.8 else "failed") + f" (median {np.median(um):.3f} over {len(stu_rows)} certified student instances)") \
        if um else "inconclusive (no certified student)"
    v["W4"] = ("held" if ens["0.0001"][0] <= 0.150 else "failed") + f" (RMSE {ens['0.0001'][0]:.4f})"
    v["W5"] = ("held" if ens["0.0001"][0] <= 0.5131 else "failed") + f" (RMSE {ens['0.0001'][0]:.4f} vs 0.5131)"
    v["W6"] = ("held" if rho >= 0.6 else "failed") + f" (rho {rho:+.3f}, p {pval:.3f}; weak test, F-c)"
    log("\nVERDICTS: " + "; ".join(f"{k}: {x}" for k, x in v.items()))
    RESULTS.update(candidates=[list(k) for k in cands], cert={f"{k[0]}|{k[1]}": {kk: vv for kk, vv in r.items()} for k, r in cert.items()},
                   counts=counts, after=after, students=stu_rows, nullsets=nullrows, step3=rows3, ensemble=ens,
                   w6=dict(rho=float(rho), p=float(pval)), verdicts=v)
    (ROOT / "certify_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))

    fig, ax = plt.subplots(figsize=(7, 5))
    for mu, mk in [("1e-06", "o"), ("0.0001", "s")]:
        rr = [r for r in rows3 if r["mu"] == mu]
        ax.scatter([r["cross_edge"] for r in rr], [r["rmse"] for r in rr], marker=mk, s=50, label=f"mu {mu}")
    ax.axhline(0.142, color="k", ls=":", label="KAN-delta 0.142")
    ax.set_xlabel("cross-edge cancellation ratio (fold A training rows)"); ax.set_ylabel("per-seed held-out RMSE (PC+EA)")
    ax.set_yscale("log"); ax.legend(); ax.set_title(f"W6: Spearman rho {rho:+.2f} (p {pval:.2f}, n = 10)")
    fig.tight_layout(); fig.savefig(FIG / "w6_crossedge_vs_rmse.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()

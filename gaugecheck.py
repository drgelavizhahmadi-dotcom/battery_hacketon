"""gaugecheck: POST-HOC, EXPLORATORY check decided after seeing the certify results. Changes no pre-registered verdict.
A) does the mu*Q penalty act as a gauge fixer at the certified mu = 1e-4 minima?
B) is the regularised teacher the right reference for the students (certify W3)?
Run: python3 gaugecheck.py >> gaugecheck_output.txt   Weights: gauge_weights/gaugecheck_* (not committed)"""
import io
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from certify import certify_point, gauge_directions
from close import FlatQ, block_norms
from gauge import Gauge, align, corr, grid_inputs, parts, setup
from kan import KAN, N_RBF, delta_head, predict, subset, to_batch
from stage2 import extract_verified, in_memory, run_exact

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
ROOT = Path(__file__).parent
FIG = ROOT / "figs" / "gaugecheck"
W_DIR = ROOT / "gauge_weights"
MU = 1e-4
STEPS = [1e-3, 1e-2]
RESULTS = {}


def log(*a):
    print(*a, flush=True)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    tr, te, te_anc, D, _ = setup()
    d, m = D["d_num"], D["train"]
    Ztr, str_ = D["Z"][m], D["s"][m]
    b = subset(D["batch"], m)
    bd = {k: (v.double() if v.is_floating_point() else v) for k, v in b.items()}
    used = np.unique(str_).tolist()
    log("#" * 100)
    log("POST-HOC, EXPLORATORY (decided after seeing the certify results). No pre-registration; no pre-registered verdict is changed.")
    log("#" * 100)
    log("NOTES / discrepancies (files win):")
    log("  gauge directions = certify.py Step 2c least-squares directions (const edge->bias in layer 1 and 2; unit rescaling with")
    log("  outgoing compensation). No EXACT nonlinear version exists here: a constant is not exactly representable by one edge's")
    log("  basis (residual ~3.5e-3), and tanh between layers breaks exact rescaling. Constant shifts are linear in the parameters,")
    log("  so finite steps ARE their nonlinear version; for rescaling the nonlinear version is: scale unit j's incoming parameters")
    log("  by a = 1 + t, then refit outgoing edge j by least squares to reproduce psi_j(tanh(s_j)) from tanh(a s_j).")
    log("  B.3 d_AB = RMS(delta_student - delta_reference) / std(data target delta), the fiber2 normalisation.")

    cert = json.loads((ROOT / "certify_results.json").read_text())["cert"]
    tags = [f"ind_{i}" for i in range(6)] + [f"stu_{s}" for s in (115, 104, 114)]
    certified = [t for t in tags if cert.get(f"0.0001|{t}", {}).get("certified")]
    log(f"  certified mu = 1e-4 minima (from certify_results.json): {certified}")
    names = [f"close_0.0001_{t}.pt" for t in certified] + [f"delta_s{i}.pt" for i in range(6)] + [f"control3_R_s{s}.pt" for s in (115, 104, 114)]
    log("\n==================== inputs (SHA-256 verified, held in memory) ====================")
    blobs = in_memory(extract_verified(sorted(set(names))))

    def kan_from(blob):
        k = KAN(d, [6, 1], D["n_salt"])
        miss, unexp = k.load_state_dict({n: t.float() for n, t in torch.load(io.BytesIO(blob)).items()}, strict=False)
        assert not unexp and all(x.endswith(".grid") for x in miss), (miss, unexp)
        return k.eval()

    teacher = kan_from(blobs["delta_s0.pt"])
    yT = torch.tensor(predict(teacher, delta_head, to_batch(Ztr, str_, x_co=b["x_co"].numpy())), dtype=torch.float64)
    sd_y = bd["y"].std().item()

    def flat_pair(tag):
        start = blobs[f"delta_s{tag[4:]}.pt"] if tag.startswith("ind") else blobs[f"control3_R_s{tag[4:]}.pt"]
        bb = dict(bd); bb["y"] = bd["y"] if tag.startswith("ind") else yT
        return FlatQ(kan_from(start), bb, used, MU), FlatQ(kan_from(start), bb, used, 0.0), bb

    # ==================== A ====================
    log("\n==================== A: is the penalty what breaks the gauge? (curvature and slope split along gauge directions) ====================")
    A = {}
    for tag in certified:
        fq, f0, bb = flat_pair(tag)
        x = torch.load(io.BytesIO(blobs[f"close_0.0001_{tag}.pt"])).numpy()
        Hf, H0 = fq.hessian(x), f0.hessian(x)
        HQ = Hf - H0
        gf, g0 = fq.fun(x)[1], f0.fun(x)[1]
        gQ = gf - g0
        ev = np.linalg.eigvalsh(Hf)
        dirs = gauge_directions(fq, x, Ztr, str_, d)
        net0 = fq.to_kan(x, d, D["n_salt"])
        with torch.no_grad():
            base_pred = delta_head(net0, bb).numpy()
        base = fq.report(x)
        fam_out = {}
        for fam, items in dirs.items():
            rows = []
            for v, res in items:
                u = v / np.linalg.norm(v)
                cf, c0, cq = float(u @ Hf @ u), float(u @ H0 @ u), float(u @ HQ @ u)
                sf, s0, sq = float(u @ gf), float(u @ g0), float(u @ gQ)
                steps = {}
                for t in STEPS:
                    xt = x + t * u
                    rep = fq.report(xt)
                    with torch.no_grad():
                        pr = delta_head(fq.to_kan(xt, d, D["n_salt"]), bb).numpy()
                    steps[t] = dict(dpred=float(np.sqrt(np.mean((pr - base_pred) ** 2))), ddata=rep["data"] - base["data"],
                                    dpen=rep["pen"] - base["pen"], dmuQ=rep["muQ"] - base["muQ"])
                rows.append(dict(curv=(cf, c0, cq), slope=(sf, s0, sq), steps=steps, res=res))
            fam_out[fam] = rows
        # nonlinear rescaling with least-squares refit of the outgoing edge (the closest nonlinear version)
        nl = []
        with torch.no_grad():
            zin = torch.cat([bb["x"], torch.tanh(net0.emb(bb["s"]))], 1)
            s = net0.layers[0](zin)
        gk = np.linspace(-1, 1, N_RBF); hw = 2 / (N_RBF - 1)
        basis = lambda uu: np.column_stack([uu / (1 + np.exp(-uu))] + [np.exp(-(((uu - g_) / hw) ** 2)) for g_ in gk])
        for j in range(6):
            for t in STEPS:
                a = 1 + t
                net = fq.to_kan(x, d, D["n_salt"])
                with torch.no_grad():
                    L0, L1 = net.layers
                    psi_old = L1.edge(j, torch.tanh(s[:, j]))[:, 0].numpy()
                    L0.coef[j] *= a; L0.base.weight[j] *= a; L0.base.bias[j] *= a
                    u_new = torch.tanh(a * s[:, j]).numpy()
                    B = basis(u_new)
                    cf_, *_ = np.linalg.lstsq(B, psi_old, rcond=None)
                    L1.base.weight[0, j] = float(cf_[0]); L1.coef[0, j, :] = torch.tensor(cf_[1:])
                    pr = delta_head(net, bb).numpy()
                    data = float(F.mse_loss(torch.tensor(pr), bb["y"]))
                    pen = float(1e-3 * net.smoothness())
                nl.append(dict(unit=j, t=t, dpred=float(np.sqrt(np.mean((pr - base_pred) ** 2))), ddata=data - base["data"], dpen=pen - base["pen"]))
        A[tag] = dict(fams=fam_out, nonlinear=nl, eig_median=float(np.median(ev)), eig_max=float(ev[-1]))
        log(f"\n  {tag}: Hessian eigenvalues median {np.median(ev):.2e}, max {ev[-1]:.2e}")
        for fam, rows in fam_out.items():
            cf = np.array([r["curv"] for r in rows]); sl = np.array([r["slope"] for r in rows])
            share = cf[:, 2] / cf[:, 0]
            st = {t: np.median([r["steps"][t]["dpred"] for r in rows]) for t in STEPS}
            dd = {t: np.median([abs(r["steps"][t]["ddata"]) for r in rows]) for t in STEPS}
            log(f"    {fam:9s} (n={len(rows)}): d'Hd full {np.median(cf[:, 0]):.2e} = no-Q {np.median(cf[:, 1]):.2e} + muQ {np.median(cf[:, 2]):.2e}; "
                f"muQ share median {np.median(share):.3f} [{share.min():.3f}-{share.max():.3f}] | d'g full {np.median(np.abs(sl[:, 0])):.1e}, "
                f"no-Q {np.median(np.abs(sl[:, 1])):.1e}, muQ {np.median(np.abs(sl[:, 2])):.1e} | step t=1e-3: dpred {st[1e-3]:.1e}, |ddata| {dd[1e-3]:.1e}; "
                f"t=1e-2: dpred {st[1e-2]:.1e}, |ddata| {dd[1e-2]:.1e}")
        nlm = {t: (np.median([r["dpred"] for r in nl if r["t"] == t]), np.median([abs(r["ddata"]) for r in nl if r["t"] == t])) for t in STEPS}
        log(f"    rescale (nonlinear, LS-refit outgoing): t=1e-3 dpred {nlm[1e-3][0]:.1e}, |ddata| {nlm[1e-3][1]:.1e}; t=1e-2 dpred {nlm[1e-2][0]:.1e}, |ddata| {nlm[1e-2][1]:.1e}")

    # ==================== B ====================
    log("\n==================== B: the right reference for the students ====================")
    bbT = dict(bd); bbT["y"] = yT
    fT = FlatQ(kan_from(blobs["delta_s0.pt"]), bbT, used, MU)
    rT, _, evT = certify_point(fT, fT.x0)
    repT = fT.report(fT.x0)
    log(f"  B1 teacher under the certify loss (mu 1e-4, students' targets): max|grad| {rT['grad']:.2e}; smallest eig {rT['mineig']:.2e} "
        f"(max {rT['maxeig']:.2e}); data {repT['data']:.1e} (0 by construction), penalty {repT['pen']:.3e}, mu*Q {repT['muQ']:.3e} -> "
        f"{'stationary' if rT['grad'] < 1e-7 else 'NOT stationary'}")
    cache, meta = W_DIR / "gaugecheck_regteacher.pt", W_DIR / "gaugecheck_regteacher.json"
    if cache.exists() and meta.exists():
        xR = torch.load(cache).numpy(); infoR = json.loads(meta.read_text()); cached = True
    else:
        xR, infoR, _ = run_exact("reg teacher", fT); cached = False
        torch.save(torch.tensor(xR), cache); meta.write_text(json.dumps(infoR, default=float))
    rR, _, _ = certify_point(fT, xR)
    repR = fT.report(xR)
    netR = fT.to_kan(xR, d, D["n_salt"])
    predR = predict(netR.float(), delta_head, b)
    thR = sum(block_norms(netR.double(), used).values())
    thT = sum(block_norms(teacher.double(), used).values())
    log(f"  B2 regularised teacher: {infoR['stop']} after {infoR['iters']} iters{' [cached]' if cached else ''}; Step-1 certified {rR['certified']} "
        + " ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in rR["crit"].items()) +
        f" | data {repR['data']:.3e}, penalty {repR['pen']:.3e}, mu*Q {repR['muQ']:.3e}, ||theta||^2 {thR:.1f} (teacher {thT:.1f}); "
        f"delta corr with original teacher {corr(predR, yT.numpy()):.6f}")
    teacher_f = teacher.float()
    netR_f = fT.to_kan(xR, d, D["n_salt"]).float()
    sweeps = grid_inputs(D)
    refs = {}
    for name, net in [("original", teacher_f), ("regularised", netR_f)]:
        g = Gauge(net, Ztr, str_, d)
        var = g.fixed(parts(net, Ztr, str_, d)[0]).var(1)
        act = var >= 0.05 * var.max(0, keepdims=True)
        refs[name] = (g, act, [g.fixed(parts(net, *sweeps[i], d)[0])[i] for i in range(d)], predict(net, delta_head, b))
    B = []
    for tag in [t for t in certified if t.startswith("stu")]:
        fq, _, _ = flat_pair(tag)
        x = torch.load(io.BytesIO(blobs[f"close_0.0001_{tag}.pt"])).numpy()
        net = fq.to_kan(x, d, D["n_salt"]).float()
        ps = predict(net, delta_head, b)
        gS = Gauge(net, Ztr, str_, d)
        row = dict(student=tag)
        for name, (gR, act, edgeR, pR) in refs.items():
            gS_ = Gauge(net, Ztr, str_, d)
            um = np.abs(align(gR, gS_))
            pe = [abs(corr(gS_.fixed(parts(net, *sweeps[i], d)[0])[i][:, j], edgeR[i][:, j])) for i in range(d) for j in range(6) if act[i, j]]
            row[name] = dict(pred_corr=corr(ps, pR), unit=float(np.median(um)), units=um.tolist(), edge=float(np.median(pe)),
                             dAB=float(np.sqrt(np.mean((ps - pR) ** 2)) / sd_y))
        B.append(row)
        log(f"  B3 {tag}: vs ORIGINAL teacher: delta corr {row['original']['pred_corr']:.6f}, unit {row['original']['unit']:.3f}, edge {row['original']['edge']:.3f}, d_AB {row['original']['dAB']:.4f} | "
            f"vs REGULARISED teacher: delta corr {row['regularised']['pred_corr']:.6f}, unit {row['regularised']['unit']:.3f}, edge {row['regularised']['edge']:.3f}, d_AB {row['regularised']['dAB']:.4f}")
    RESULTS.update(post_hoc=True, A=A, B1=dict(rT, **repT), B2=dict(info={k: v for k, v in infoR.items() if k != "eig_checks"}, cert=rR, report=repR, theta2=thR),
                   B3=B)
    (ROOT / "gaugecheck_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, fam in enumerate(["const_L1", "const_L2", "rescale"]):
        sh = [r["curv"][2] / r["curv"][0] for t in A for r in A[t]["fams"][fam]]
        ax.boxplot(sh, positions=[k], widths=0.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(["const shift L1", "const shift L2", "unit rescale"])
    ax.set_ylabel("share of d'Hd from mu*Q"); ax.set_title("POST-HOC: curvature along gauge directions coming from mu*Q (mu = 1e-4 certified minima)")
    fig.tight_layout(); fig.savefig(FIG / "curvature_share.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()

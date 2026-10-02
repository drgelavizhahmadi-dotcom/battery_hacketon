"""anatomy: descriptive analysis of the parameter runaway (no optimisation). See anatomy_plan.md.
Run: python3 anatomy.py    Figures: figs/anatomy/    Inputs held in memory after SHA verification."""
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

from gauge import setup
from kan import KAN, N_RBF, SMOOTH, delta_head, subset
from stage2 import extract_verified, in_memory
from trust import Flat

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
ROOT = Path(__file__).parent
FIG = ROOT / "figs" / "anatomy"
IND, STU = list(range(6)), [115, 104, 114]
BLOCKS = {"layer-1 RBF": ["layers.0.coef"], "layer-2 RBF": ["layers.1.coef"],
          "SiLU base": ["layers.0.base.weight", "layers.1.base.weight"],
          "biases": ["layers.0.base.bias", "layers.1.base.bias"], "salt embedding": ["emb.weight"]}
PATH_SEED = ["107-epoch", "fiber", "capped", "trust", "A", "C1e-06", "C1e-05"]
PATH_STU = ["AdamW-stage", "capped", "trust", "A", "C1e-06", "C1e-05"]
UNREG = {"seed": ["107-epoch", "fiber", "capped", "A"], "stu": ["AdamW-stage", "capped", "A"]}
RESULTS = {}


def log(*a):
    print(*a, flush=True)


def null_basis():
    k = np.arange(N_RBF, dtype=float)
    Q, _ = np.linalg.qr(np.column_stack([np.ones(N_RBF), k]))
    return Q  # (8, 2) orthonormal basis of the second-difference null space


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    tr, te, te_anc, D, _ = setup()
    d, m = D["d_num"], D["train"]
    Ztr, str_ = D["Z"][m], D["s"][m]
    b = subset(D["batch"], m)
    bd = {k: (v.double() if v.is_floating_point() else v) for k, v in b.items()}
    used = np.unique(str_).tolist()
    NB = null_basis()

    log("NOTES (files win; see anatomy_plan.md):")
    log("  students' first point is the control-3 AdamW stage (530-2680 epochs, not 107); students have no fiber point")
    log("  trust, A, C1e-6, C1e-5 all branch from the capped state; unregularised path = AdamW -> (fiber) -> capped -> A")
    log("  C points were optimised with an extra L2 term lambda*||theta||^2, reported separately; 'total' = data + penalty only")

    log("\n==================== inputs (SHA-256 verified, held in memory) ====================")
    names = (["delta_s0.pt"] + [f"delta_s{i}.pt" for i in IND[1:]] + [f"fiber_polished_s{i}.pt" for i in IND] +
             [f"fiber2_ind_{i}.pt" for i in IND] + [f"trust_ind_{i}.pt" for i in IND] +
             [f"control3_R_s{s}.pt" for s in STU] + [f"fiber2_stu_{s}.pt" for s in STU] + [f"trust_stu_{s}.pt" for s in STU] +
             [f"stage2_{p}_{t}.pt" for p in ["A", "C1e-06", "C1e-05"] for t in [f"ind_{i}" for i in IND] + [f"stu_{s}" for s in STU]])
    blobs = in_memory(extract_verified(names))

    def kan_from(blob):
        k = KAN(d, [6, 1], D["n_salt"])
        miss, unexp = k.load_state_dict({n: t.float() for n, t in torch.load(io.BytesIO(blob)).items()}, strict=False)
        assert not unexp and all(x.endswith(".grid") for x in miss), (miss, unexp)
        return k.eval().double()

    teacher = kan_from(blobs["delta_s0.pt"])
    with torch.no_grad():
        yT = delta_head(teacher, bd)

    def points(kind, idx):
        if kind == "seed":
            capped = kan_from(blobs[f"fiber2_ind_{idx}.pt"])
            pts = {"107-epoch": kan_from(blobs[f"delta_s{idx}.pt"]), "fiber": kan_from(blobs[f"fiber_polished_s{idx}.pt"]),
                   "capped": capped, "trust": kan_from(blobs[f"trust_ind_{idx}.pt"])}
            tag = f"ind_{idx}"
        else:
            capped = kan_from(blobs[f"fiber2_stu_{idx}.pt"])
            pts = {"AdamW-stage": kan_from(blobs[f"control3_R_s{idx}.pt"]), "capped": capped,
                   "trust": kan_from(blobs[f"trust_stu_{idx}.pt"])}
            tag = f"stu_{idx}"
        fl = Flat(capped.float(), bd, used)  # same 609-vector mapping as stage2.py
        for p in ["A", "C1e-06", "C1e-05"]:
            x = torch.load(io.BytesIO(blobs[f"stage2_{p}_{tag}.pt"])).numpy()
            pts[p] = fl.to_kan(x, d, D["n_salt"])
        return pts

    def analyse(net, y):
        with torch.no_grad():
            pred = delta_head(net, {**bd, "y": y})
            data = float(F.mse_loss(pred, y)); pen = float(SMOOTH * net.smoothness())
            sd = dict(net.named_parameters())
            norms = {blk: float(sum(((sd[k][used] if k == "emb.weight" else sd[k]) ** 2).sum() for k in keys)) for blk, keys in BLOCKS.items()}
            L0, L1 = net.layers
            z = torch.cat([bd["x"], torch.tanh(net.emb(bd["s"]))], 1)
            s_pre = L0(z)
            u = torch.tanh(s_pre)
            edges = {"layer-1 continuous": [], "layer-1 salt": [], "layer-2": []}
            coefs = {"layer-1 continuous": [], "layer-1 salt": [], "layer-2": []}

            def edge(Lr, inp, i, j, blk):
                silu_t = Lr.base.weight[j, i] * F.silu(inp)
                rbf_t = Lr.rbf(inp) @ Lr.coef[j, i, :]
                rs, rr, rt = (float(t.pow(2).mean().sqrt()) for t in (silu_t, rbf_t, silu_t + rbf_t))
                edges[blk].append(((rs + rr) / rt if rt > 0 else np.inf, rs, rr, rt))
                coefs[blk].append(Lr.coef[j, i, :].numpy().copy())

            for j in range(6):
                for i in range(z.shape[1]):
                    edge(L0, z[:, i], i, j, "layer-1 continuous" if i < d else "layer-1 salt")
                edge(L1, u[:, j], j, 0, "layer-2")
        ns = {blk: [float(np.sum((NB.T @ c) ** 2) / max(np.sum(c ** 2), 1e-300)) for c in cs] for blk, cs in coefs.items()}
        return dict(data=data, pen=pen, total=data + pen, theta2=sum(norms.values()), norms=norms,
                    canc={blk: float(np.median([e[0] for e in es])) for blk, es in edges.items()},
                    rms={blk: [float(np.median([e[k] for e in es])) for k in (1, 2, 3)] for blk, es in edges.items()},
                    ns_median={blk: float(np.median(v)) for blk, v in ns.items()}, coefs=coefs)

    models = [("seed", i, f"ind {i}", bd["y"]) for i in IND] + [("stu", s, f"stu {s}", yT) for s in STU]
    A = {}
    for kind, idx, tag, y in models:
        pts = points(kind, idx)
        A[tag] = {p: analyse(net, y) for p, net in pts.items()}
        A[tag]["_kind"] = kind
    with torch.no_grad():
        t_own = analyse(teacher, yT); t_real = float(F.mse_loss(delta_head(teacher, bd), bd["y"]))

    # ---------------- Q1
    log("\n==================== Q1: loss decomposition ====================")
    log(f"  {'model':8s} {'point':12s} {'data':>11s} {'penalty':>11s} {'total':>11s} {'||th||^2':>10s}   " + "  ".join(f"{b[:10]:>10s}" for b in BLOCKS))
    for kind, idx, tag, _ in models:
        for p in (PATH_SEED if kind == "seed" else PATH_STU):
            r = A[tag][p]
            log(f"  {tag:8s} {p:12s} {r['data']:11.4e} {r['pen']:11.4e} {r['total']:11.4e} {r['theta2']:10.1f}   " +
                "  ".join(f"{r['norms'][b_]:10.1f}" for b_ in BLOCKS))
    q1 = []
    for kind, idx, tag, _ in models:
        c, a = A[tag]["capped"], A[tag]["A"]
        q1.append(dict(tag=tag, kind=kind, data_rel=(a["data"] - c["data"]) / c["data"], pen_ratio=a["pen"] / c["pen"],
                       theta_ratio=a["theta2"] / c["theta2"]))
    log("\n  capped -> A final (unregularised continuation):")
    for r in q1:
        log(f"    {r['tag']:8s} ||th||^2 x{r['theta_ratio']:.2f}; data term change {r['data_rel']:+.2%}; penalty x{r['pen_ratio']:.3f}")
    n_stu_ok = sum(abs(r["data_rel"]) < 0.10 for r in q1 if r["kind"] == "stu")
    n_seed_ok = sum(abs(r["data_rel"]) < 0.10 for r in q1 if r["kind"] == "seed")
    n_pen = sum(r["pen_ratio"] < 1 for r in q1)
    q1_ok = n_stu_ok == 3 and n_seed_ok >= 4 and n_pen >= 7
    log(f"  criterion: data change < 10% for students {n_stu_ok}/3, seeds {n_seed_ok}/6; penalty falls for {n_pen}/9 -> "
        f"{'AS EXPECTED' if q1_ok else 'NOT AS EXPECTED'}")

    # ---------------- Q2
    log("\n==================== Q2: cancellation and the penalty's null space ====================")
    log("  median cancellation ratio (RMS SiLU + RMS RBF) / RMS sum, and median null-space share of RBF vectors (random 0.25)")
    log(f"  {'model':8s} {'point':12s} {'L1-cont canc':>12s} {'L1-salt canc':>12s} {'L2 canc':>9s} | {'L1-cont ns':>10s} {'L1-salt ns':>10s} {'L2 ns':>7s} | L1-cont median RMS (SiLU, RBF, sum)")
    for kind, idx, tag, _ in models:
        for p in (PATH_SEED if kind == "seed" else PATH_STU):
            r = A[tag][p]
            log(f"  {tag:8s} {p:12s} {r['canc']['layer-1 continuous']:12.2f} {r['canc']['layer-1 salt']:12.2f} {r['canc']['layer-2']:9.2f} | "
                f"{r['ns_median']['layer-1 continuous']:10.3f} {r['ns_median']['layer-1 salt']:10.3f} {r['ns_median']['layer-2']:7.3f} | "
                + ", ".join(f"{v:.3g}" for v in r["rms"]["layer-1 continuous"]))
    log("\n  pooled null-space share of the RBF-coefficient CHANGE between consecutive unregularised-path points (random 0.25):")
    q2 = []
    for kind, idx, tag, _ in models:
        path = UNREG[kind]
        segs = []
        for a_, b_ in zip(path[:-1], path[1:]):
            row = {}
            for blk in ["layer-1 continuous", "layer-1 salt", "layer-2"]:
                dc = [cb - ca for ca, cb in zip(A[tag][a_]["coefs"][blk], A[tag][b_]["coefs"][blk])]
                row[blk] = float(sum(np.sum((NB.T @ x) ** 2) for x in dc) / max(sum(np.sum(x ** 2) for x in dc), 1e-300))
            segs.append((f"{a_}->{b_}", row))
        log(f"    {tag:8s} " + "; ".join(f"{s}: L1c {r['layer-1 continuous']:.3f}, L1s {r['layer-1 salt']:.3f}, L2 {r['layer-2']:.3f}" for s, r in segs))
        first = path[0]
        q2.append(dict(tag=tag, canc_A=A[tag]["A"]["canc"]["layer-1 continuous"], canc_first=A[tag][first]["canc"]["layer-1 continuous"],
                       ns_cappedA=dict(segs)[f"capped->A"]["layer-1 continuous"], segs=segs))
    c1 = sum(r["canc_A"] > 5 for r in q2); c2 = sum(r["canc_A"] > r["canc_first"] for r in q2); c3 = sum(r["ns_cappedA"] > 0.5 for r in q2)
    q2_ok = c1 >= 7 and c2 >= 7 and c3 >= 7
    log(f"  criterion (layer-1 continuous): cancellation > 5 at A final {c1}/9; higher at A than at the AdamW stage {c2}/9; "
        f"null-space share of the capped->A change > 0.5 {c3}/9 -> {'AS EXPECTED' if q2_ok else 'NOT AS EXPECTED'}")

    # ---------------- Q3
    log("\n==================== Q3: teacher vs students ====================")
    log(f"  teacher: data on its own targets {t_own['data']:.3e} (0 by construction); data on the real data {t_real:.4e}; "
        f"penalty {t_own['pen']:.4e}; ||theta||^2 {t_own['theta2']:.1f}; L1-cont cancellation {t_own['canc']['layer-1 continuous']:.2f}; "
        f"L1-cont null-space share {t_own['ns_median']['layer-1 continuous']:.3f}")
    q3 = []
    for kind, idx, tag, _ in models:
        if kind != "stu":
            continue
        for p in PATH_STU:
            r = A[tag][p]
            log(f"  {tag:8s} {p:12s} data {r['data']:.3e}; penalty {r['pen']:.4e} ({r['pen'] / t_own['pen']:.2f} x teacher); "
                f"||theta||^2 {r['theta2']:.1f} ({r['theta2'] / t_own['theta2']:.1f} x teacher)")
        for p in ["capped", "A"]:
            r = A[tag][p]
            q3.append(r["pen"] < t_own["pen"] and r["theta2"] >= 10 * t_own["theta2"])
    q3_ok = all(q3)
    log(f"  criterion: lower penalty AND >= 10x teacher ||theta||^2 at capped and A for every student: {sum(q3)}/6 -> "
        f"{'AS EXPECTED' if q3_ok else 'NOT AS EXPECTED'}")

    outcomes = {"Q1": q1_ok, "Q2": q2_ok, "Q3": q3_ok}
    log("\nOUTCOMES: " + "; ".join(f"{k}: {'as expected' if v else 'not as expected'}" for k, v in outcomes.items()))
    RESULTS.update(points={t: {p: {k: v for k, v in r.items() if k != "coefs"} for p, r in pts.items() if not p.startswith("_")} for t, pts in A.items()},
                   teacher=dict(own={k: v for k, v in t_own.items() if k != "coefs"}, data_real=t_real),
                   q1=q1, q2=[{k: v for k, v in r.items()} for r in q2], q3=q3, outcomes=outcomes)
    (ROOT / "anatomy_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))

    # ---------------- figures
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for kind, idx, tag, _ in models:
        path = UNREG[kind]
        col = "tab:blue" if kind == "seed" else "tab:orange"
        for ax, key in zip(axs, ["data", "pen"]):
            ax.loglog([A[tag][p]["theta2"] for p in path], [A[tag][p][key] for p in path], "o-", color=col, alpha=0.7, ms=3)
            for p, mk in [("C1e-06", "s"), ("C1e-05", "^"), ("trust", "x")]:
                ax.loglog(A[tag][p]["theta2"], A[tag][p][key], mk, color=col, ms=5, alpha=0.7)
    for ax, key, lab in zip(axs, ["data", "pen"], ["data term", "smoothness penalty"]):
        ax.loglog(t_own["theta2"], t_own[key] if key == "pen" else max(t_own[key], 1e-12), "*", color="k", ms=14, label="teacher")
        ax.set_xlabel("||theta||^2"); ax.set_ylabel(lab); ax.set_title(f"{lab} vs ||theta||^2 (lines: unregularised path; s: C1e-6, ^: C1e-5, x: trust)")
    axs[0].plot([], [], "o-", color="tab:blue", label="seeds"); axs[0].plot([], [], "o-", color="tab:orange", label="students"); axs[0].legend()
    fig.tight_layout(); fig.savefig(FIG / "q1_loss_vs_norm.png", dpi=110); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for kind, idx, tag, _ in models:
        col = "tab:blue" if kind == "seed" else "tab:orange"
        P = PATH_SEED if kind == "seed" else PATH_STU
        axs[0].semilogy([A[tag][p]["theta2"] for p in P], [A[tag][p]["canc"]["layer-1 continuous"] for p in P], "o", color=col, alpha=0.7)
        axs[1].semilogx([A[tag][p]["theta2"] for p in P], [A[tag][p]["ns_median"]["layer-1 continuous"] for p in P], "o", color=col, alpha=0.7)
    axs[0].set_xscale("log"); axs[0].set_title("median layer-1 cancellation ratio vs ||theta||^2"); axs[0].axhline(1, color="k", lw=0.5)
    axs[1].axhline(0.25, color="k", ls=":", label="random 0.25"); axs[1].set_title("median null-space share of layer-1 RBF vectors"); axs[1].legend()
    for ax in axs:
        ax.set_xlabel("||theta||^2")
    fig.tight_layout(); fig.savefig(FIG / "q2_cancellation.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()

"""Corrected 2x2 control for KAN gauge fixing (see control2_prereg.md). Run: python3 control2.py"""
import json
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from gauge import FIG, W_DIR, Gauge, align, corr, grid_inputs, parts, setup, smooth_teacher
from kan import KAN, LR, SMOOTH, WD, delta_head, predict, to_batch

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
ROOT = Path(__file__).parent
SEEDS = list(range(20))
EPOCHS_S, EPOCHS_MAX, TOL_FRAC = 107, 3000, 0.02
ACTIVE_FRAC, GATE = 0.05, 0.99
CO_DESC = ["eps_co", "lneta_co", "M_co"]


def log(*a):
    print(*a, flush=True)


def train_until(M, b, seed, max_ep, tol):
    """kan.train's loop (same init, optimizer, penalty); optionally stop once training RMSE < tol."""
    torch.manual_seed(seed)
    net = KAN(M["d_num"], M["widths"], M["n_salt"])
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    for ep in range(1, max_ep + 1):
        opt.zero_grad()
        loss = F.mse_loss(delta_head(net, b), b["y"]) + SMOOTH * net.smoothness()
        loss.backward()
        opt.step()
        if tol is not None:
            with torch.no_grad():
                if math.sqrt(float(F.mse_loss(delta_head(net, b), b["y"]))) < tol:
                    return net.eval(), ep, True
    return net.eval(), max_ep, False


def students(cell, M, b, max_ep, tol):
    meta_p = W_DIR / f"control2_{cell}_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    nets = []
    for s in SEEDS:
        p = W_DIR / f"control2_{cell}_s{s}.pt"
        if p.exists() and str(s) in meta:
            net = KAN(M["d_num"], M["widths"], M["n_salt"])
            net.load_state_dict(torch.load(p))
            nets.append(net.eval())
        else:
            net, ep, reached = train_until(M, b, s, max_ep, tol)
            torch.save(net.state_dict(), p)
            meta[str(s)] = [ep, reached]
            nets.append(net)
    meta_p.write_text(json.dumps(meta))
    return nets, np.array([meta[str(s)][0] for s in SEEDS]), np.array([meta[str(s)][1] for s in SEEDS])


def main():
    W_DIR.mkdir(exist_ok=True)
    tr, te, te_anc, D, _ = setup()
    m = D["train"]
    ZR, sR = D["Z"][m], D["s"][m]
    xR = D["batch"]["x_co"][torch.tensor(m)].numpy()
    rng = np.random.default_rng(0)
    n = len(ZR)
    ZI = np.column_stack([ZR[rng.integers(0, n, n), j] for j in range(ZR.shape[1])])
    sI = sR[rng.integers(0, n, n)]
    xI = D["sc"].inv(D["cols"].index("x_co"), ZI[:, D["cols"].index("x_co")])
    inputs = {"R": (ZR, sR, xR), "I": (ZI, sI, xI)}
    r_max = max(abs(np.corrcoef(ZI[:, i], ZI[:, j])[0, 1]) for i in range(8) for j in range(i + 1, 8))
    log(f"(I) inputs: max |corr| between numeric columns {r_max:.3f} (real: "
        f"{max(abs(np.corrcoef(ZR[:, i], ZR[:, j])[0, 1]) for i in range(8) for j in range(i + 1, 8)):.3f})")

    T = smooth_teacher(D, tr.delta.values[m])
    names, d = D["names"], D["d_num"]
    sweeps = grid_inputs(D)
    results, box = {}, {}
    for inp in ["R", "I"]:
        Z, s, xc = inputs[inp]
        bT = to_batch(Z, s, x_co=xc)
        yT = predict(T, delta_head, bT)
        b = to_batch(Z, s, x_co=xc, y=yT)
        gT = Gauge(T, Z, s, d)
        phiT_rows = gT.fixed(parts(T, Z, s, d)[0])
        varT = phiT_rows.var(1)
        active = varT >= ACTIVE_FRAC * varT.max(0, keepdims=True)
        hT = parts(T, Z, s, d)[3]
        edgeT = [gT.fixed(parts(T, *sweeps[i], d)[0])[i] for i in range(d)]
        for trn in ["S", "C"]:
            cell = inp + trn
            tol = TOL_FRAC * float(np.std(yT)) if trn == "C" else None
            nets, eps, reached = students(cell, D, b, EPOCHS_S if trn == "S" else EPOCHS_MAX, tol)
            pc, dc, um = [], [], []
            per = {nm: [] for nm in names[:d]}
            for net in nets:
                g = Gauge(net, Z, s, d)
                um.extend(np.abs(align(gT, g)))
                h = parts(net, Z, s, d)[3]
                pc.append(corr(h, hT)); dc.append(corr(h * xc, hT * xc))
                for i in range(d):
                    e = g.fixed(parts(net, *sweeps[i], d)[0])[i]
                    per[names[i]].extend(abs(corr(e[:, j], edgeT[i][:, j])) for j in range(e.shape[1]) if active[i, j])
            allv = [v for vs in per.values() for v in vs]
            co = [v for nm in CO_DESC for v in per[nm]]
            gate = float(np.median(pc)) >= GATE
            r = dict(epochs_median=int(np.median(eps)), epochs_min=int(eps.min()), epochs_max=int(eps.max()),
                     n_reached=int(reached.sum()), tol=tol, pred_corr=float(np.median(pc)), pred_corr_min=float(np.min(pc)),
                     delta_corr=float(np.median(dc)), unit_match=float(np.median(um)),
                     edge_all=float(np.median(allv)) if allv else None,
                     per_input={nm: (float(np.median(v)) if v else None) for nm, v in per.items()},
                     n_active={nm: int(active[names.index(nm)].sum()) for nm in per},
                     co_desc=float(np.median(co)) if co else None, gate=bool(gate) if trn == "C" else None)
            results[cell] = r
            box[cell] = (per, um)
            log(f"\ncell ({inp},{trn}): epochs median {r['epochs_median']} [{r['epochs_min']}-{r['epochs_max']}], "
                f"reached tol {r['n_reached']}/20" + (f" (tol {tol:.4f})" if tol else ""))
            log(f"  prediction corr median {r['pred_corr']:.4f} (min {r['pred_corr_min']:.4f}), delta corr {r['delta_corr']:.4f}"
                + (f"  gate >= {GATE}: {'PASS' if gate else 'FAIL'}" if trn == "C" else ""))
            log(f"  unit match |corr| median {r['unit_match']:.3f}; edge recovery |corr| median (active continuous) {r['edge_all']:.3f}")
            log("  per input: " + ", ".join(f"{k} {v:.3f} (n_act {r['n_active'][k]})" if v is not None else f"{k} n/a (n_act 0)"
                                          for k, v in r["per_input"].items()))

    # 2x2 table
    log("\n2x2 summary (medians):")
    log(f"{'cell':6s} {'epochs':>7s} {'reached':>8s} {'pred r':>7s} {'unit |r|':>9s} {'edge |r|':>9s} {'invT':>6s} {'molal':>6s} {'co-desc':>8s} gate")
    for c in ["RS", "RC", "IS", "IC"]:
        r = results[c]
        f = lambda v: f"{v:.3f}" if v is not None else "  n/a"
        log(f"({c[0]},{c[1]})  {r['epochs_median']:7d} {r['n_reached']:5d}/20 {r['pred_corr']:7.4f} {r['unit_match']:9.3f} {f(r['edge_all']):>9s} "
            f"{f(r['per_input']['invT']):>6s} {f(r['per_input']['molal']):>6s} {f(r['co_desc']):>8s} "
            f"{'' if r['gate'] is None else ('PASS' if r['gate'] else 'FAIL')}")

    # verdicts as pre-registered
    v = {}
    ic, rc = results["IC"], results["RC"]
    v["C1"] = "inconclusive (gate failed)" if not ic["gate"] else (
        "held" if ic["edge_all"] >= 0.9 and ic["unit_match"] >= 0.9 else "failed")
    if not rc["gate"]:
        v["C2"] = "inconclusive (gate failed)"
    else:
        parts_ = []
        for nm in ["invT", "molal"]:
            x = rc["per_input"][nm]
            parts_.append(None if x is None else x >= 0.9)
        parts_.append(None if rc["co_desc"] is None else rc["co_desc"] < 0.7)
        v["C2"] = "inconclusive (no active edges in a group)" if None in parts_ else ("held" if all(parts_) else "failed")
    v["C3"] = "held" if results["IS"]["pred_corr"] < 0.95 and results["RS"]["pred_corr"] < 0.95 else "failed"
    log("\nverdicts: " + "; ".join(f"{k}: {x}" for k, x in v.items()))
    (ROOT / "control2_results.json").write_text(json.dumps(dict(cells=results, verdicts=v), indent=1))

    # figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 8.5), sharey=True)
    for ax, c in zip(axs.flat, ["RS", "RC", "IS", "IC"]):
        per, um = box[c]
        labels = [k for k in per if per[k]] + ["units"]
        data = [per[k] for k in per if per[k]] + [um]
        ax.boxplot(data, labels=labels, showfliers=True)
        ax.axhline(0.9, color="r", ls="--", lw=0.8)
        r = results[c]
        ax.set_title(f"({c[0]},{c[1]}): pred r {r['pred_corr']:.3f}, epochs {r['epochs_median']}, "
                     f"edge |r| {r['edge_all']:.2f}, unit |r| {r['unit_match']:.2f}", fontsize=10)
        ax.tick_params(axis="x", labelsize=8, rotation=30)
    axs[0, 0].set_ylabel("|corr| student vs teacher (gauge-fixed)")
    axs[1, 0].set_ylabel("|corr| student vs teacher (gauge-fixed)")
    fig.suptitle("Control 2: edge recovery (active continuous edges x 20 students) and unit matching. R = real inputs, I = independent; S = 107 ep, C = converged")
    fig.tight_layout(); fig.savefig(FIG / "control2_recovery.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()

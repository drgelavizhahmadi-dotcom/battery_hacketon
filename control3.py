"""Control 3: reachable teacher (KAN-delta seed 0), AdamW then L-BFGS polishing (see control3_prereg.md).
Run: python3 control3.py"""
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

from gauge import FIG, W_DIR, Gauge, align, corr, grid_inputs, parts, setup
from kan import KAN, LR, SMOOTH, WD, delta_head, predict, to_batch

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
ROOT = Path(__file__).parent
SEEDS = list(range(100, 120))
ADAM_MAX, CHECK_EVERY, LBFGS_MAX, LBFGS_STEP = 3000, 10, 500, 20
CORR_MIN, RMSE_FRAC = 0.99, 0.05
ACTIVE_FRAC = 0.05


def log(*a):
    print(*a, flush=True)


def fit_student(M, b, seed, tol):
    torch.manual_seed(seed)
    net = KAN(M["d_num"], M["widths"], M["n_salt"])
    y = b["y"]

    def status():
        with torch.no_grad():
            p = delta_head(net, b)
            rm = math.sqrt(float(F.mse_loss(p, y)))
            c = corr(p.numpy(), y.numpy())
        return rm <= tol and c >= CORR_MIN, rm, c

    def objective():
        return F.mse_loss(delta_head(net, b), y) + SMOOTH * net.smoothness()

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    epochs = ADAM_MAX
    for ep in range(1, ADAM_MAX + 1):
        opt.zero_grad()
        objective().backward()
        opt.step()
        if ep % CHECK_EVERY == 0 and status()[0]:
            epochs = ep
            break
    iters = 0
    if not status()[0]:
        lb = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=LBFGS_STEP, line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad()
            loss = objective()
            loss.backward()
            return loss

        while iters < LBFGS_MAX:
            lb.param_groups[0]["max_iter"] = min(LBFGS_STEP, LBFGS_MAX - iters)
            lb.step(closure)
            new = lb.state[lb._params[0]]["n_iter"]
            stalled = new == iters
            iters = new
            if status()[0] or stalled:
                break
    ok, rm, c = status()
    return net.eval(), epochs, iters, ok, rm, c


def main():
    W_DIR.mkdir(exist_ok=True)
    tr, te, te_anc, D, _ = setup()
    m = D["train"]
    ZR, sR = D["Z"][m], D["s"][m]
    xR = D["batch"]["x_co"][torch.tensor(m)].numpy()
    rng = np.random.default_rng(0)  # identical resampling to control 2
    n = len(ZR)
    ZI = np.column_stack([ZR[rng.integers(0, n, n), j] for j in range(ZR.shape[1])])
    sI = sR[rng.integers(0, n, n)]
    xI = D["sc"].inv(D["cols"].index("x_co"), ZI[:, D["cols"].index("x_co")])

    T = KAN(D["d_num"], D["widths"], D["n_salt"])
    T.load_state_dict(torch.load(W_DIR / "delta_s0.pt"))
    T.eval()
    names, d = D["names"], D["d_num"]
    sweeps = grid_inputs(D)
    results, box = {}, {}
    for cell, (Z, s, xc) in {"R": (ZR, sR, xR), "I": (ZI, sI, xI)}.items():
        yT = predict(T, delta_head, to_batch(Z, s, x_co=xc))
        b = to_batch(Z, s, x_co=xc, y=yT)
        tol = RMSE_FRAC * float(np.std(yT))
        gT = Gauge(T, Z, s, d)
        varT = gT.fixed(parts(T, Z, s, d)[0]).var(1)
        active = varT >= ACTIVE_FRAC * varT.max(0, keepdims=True)
        edgeT = [gT.fixed(parts(T, *sweeps[i], d)[0])[i] for i in range(d)]
        meta_p = W_DIR / f"control3_{cell}_meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        stats, pc, um = [], [], []
        per = {nm: [] for nm in names[:d]}
        for sd in SEEDS:
            p = W_DIR / f"control3_{cell}_s{sd}.pt"
            if p.exists() and str(sd) in meta:
                net = KAN(d, D["widths"], D["n_salt"]); net.load_state_dict(torch.load(p)); net.eval()
            else:
                net, ep, it, ok, rm, c = fit_student(D, b, sd, tol)
                torch.save(net.state_dict(), p)
                meta[str(sd)] = dict(epochs=ep, lbfgs=it, ok=bool(ok))
            with torch.no_grad():
                pdelta = delta_head(net, b).numpy()
            stats.append(dict(**meta[str(sd)], rmse_frac=float(np.sqrt(np.mean((pdelta - yT) ** 2)) / yT.std()),
                              delta_corr=corr(pdelta, yT)))
            g = Gauge(net, Z, s, d)
            um.extend(np.abs(align(gT, g)))
            for i in range(d):
                e = g.fixed(parts(net, *sweeps[i], d)[0])[i]
                per[names[i]].extend(abs(corr(e[:, j], edgeT[i][:, j])) for j in range(e.shape[1]) if active[i, j])
        meta_p.write_text(json.dumps(meta))
        dcor = np.array([x["delta_corr"] for x in stats]); rmf = np.array([x["rmse_frac"] for x in stats])
        gate = bool(np.median(dcor) >= CORR_MIN and np.median(rmf) <= RMSE_FRAC)
        allv = [v for vs in per.values() for v in vs]
        r = dict(epochs=[x["epochs"] for x in stats], lbfgs=[x["lbfgs"] for x in stats], ok=[x["ok"] for x in stats],
                 rmse_frac_median=float(np.median(rmf)), delta_corr_median=float(np.median(dcor)), delta_corr_min=float(dcor.min()),
                 gate=gate, unit_match=float(np.median(um)), edge_all=float(np.median(allv)),
                 per_input={k: (float(np.median(v)) if v else None) for k, v in per.items()},
                 active_edges=[[names[i], int(j)] for i, j in zip(*np.nonzero(active)) if i < d])
        results[cell], box[cell] = r, (per, um)
        ep, it = np.array(r["epochs"]), np.array(r["lbfgs"])
        log(f"\ncell ({cell}): AdamW epochs median {int(np.median(ep))} [{ep.min()}-{ep.max()}], L-BFGS iters median {int(np.median(it))} "
            f"[{it.min()}-{it.max()}], students meeting stop condition {sum(r['ok'])}/20")
        log(f"  train RMSE/std median {r['rmse_frac_median']:.4f}; delta corr median {r['delta_corr_median']:.5f} (min {r['delta_corr_min']:.5f}); "
            f"gate {'PASS' if gate else 'FAIL'}")
        log(f"  active continuous teacher edges ({len(r['active_edges'])}): " + ", ".join(f"{a}->u{b_}" for a, b_ in r["active_edges"]))
        log(f"  unit match |r| median {r['unit_match']:.3f}; edge |r| median {r['edge_all']:.3f}")

    log("\ncontrol 3 table (medians):")
    log(f"{'cell':5s} {'epochs':>7s} {'L-BFGS':>7s} {'RMSE/std':>9s} {'delta r':>8s} {'gate':>5s} {'unit|r|':>8s} {'edge|r|':>8s}  per-input edge |r|")
    for c in ["R", "I"]:
        r = results[c]
        pi = ", ".join(f"{k} {v:.3f}" if v is not None else f"{k} n/a" for k, v in r["per_input"].items())
        log(f"({c})   {int(np.median(r['epochs'])):7d} {int(np.median(r['lbfgs'])):7d} {r['rmse_frac_median']:9.4f} {r['delta_corr_median']:8.5f} "
            f"{'PASS' if r['gate'] else 'FAIL':>5s} {r['unit_match']:8.3f} {r['edge_all']:8.3f}  {pi}")

    R, I = results["R"], results["I"]
    v = {"C1'": "held" if I["gate"] and I["edge_all"] >= 0.9 and I["unit_match"] >= 0.9 else "failed"}
    v["C2'"] = ("inconclusive ((I) failed gate)" if not I["gate"] else "inconclusive ((R) failed gate)" if not R["gate"]
                else "held" if R["edge_all"] <= I["edge_all"] - 0.10 else "failed")
    if not I["gate"]:
        v["C4"] = "inconclusive ((I) failed gate)"
    else:
        vals = {k: x for k, x in I["per_input"].items() if x is not None}
        v["C4"] = "held" if all(x >= 0.9 for x in vals.values()) else "failed"
    log("\nverdicts: " + "; ".join(f"{k}: {x}" for k, x in v.items()))
    log(f"procedure validated (iff C1' holds): {'YES' if v[chr(67) + '1' + chr(39)] == 'held' else 'NO'}")
    (ROOT / "control3_results.json").write_text(json.dumps(dict(cells=results, verdicts=v), indent=1))

    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    for ax, c in zip(axs, ["R", "I"]):
        per, um = box[c]
        labels = [k for k in per if per[k]] + ["units"]
        ax.boxplot([per[k] for k in per if per[k]] + [um], labels=labels)
        ax.axhline(0.9, color="r", ls="--", lw=0.8)
        r = results[c]
        ax.set_title(f"({c}): delta r {r['delta_corr_median']:.4f}, RMSE/std {r['rmse_frac_median']:.3f}, gate {'PASS' if r['gate'] else 'FAIL'}; "
                     f"edge |r| {r['edge_all']:.2f}, unit |r| {r['unit_match']:.2f}", fontsize=9)
        ax.tick_params(axis="x", labelsize=8, rotation=30)
    axs[0].set_ylabel("|corr| student vs teacher (gauge-fixed)")
    fig.suptitle("Control 3: teacher = trained KAN-delta seed 0; 20 students, AdamW + L-BFGS")
    fig.tight_layout(); fig.savefig(FIG / "control3_recovery.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()

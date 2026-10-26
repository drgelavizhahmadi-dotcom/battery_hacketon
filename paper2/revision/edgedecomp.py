"""paper2/revision/edgedecomp.py: edge decomposition of the fold A ensembles (POST-HOC, DESCRIPTIVE; plan edgedecomp_plan.md).
No training, no optimisation; archived weights SHA-verified and held in memory.
Output: edgedecomp_output.txt, edgedecomp_results.json, figs/edgedecomp.png. Run from the repo root."""
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "arch", REPO / "paper2"):
    sys.path.insert(0, str(p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

import sc_run as SR
from close import FlatQ
from kan import DELTA_IN, KAN, subset
from stage2 import extract_verified, in_memory
from trust import Flat

LOG = []
LOGGED = {"early": 0.1420, "mu = 0": 0.8147, "mu = 1e-6": 1.0262, "mu = 1e-4": 0.3345}
STATES = list(LOGGED)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.append(s)


def decompose(net, b):
    """exact edge decomposition of kan.KAN: phi (n, d_in, d_hid), psi (n, d_hid), b2, h; checked against the forward pass"""
    net = net.double().eval()
    with torch.no_grad():
        x, s = b["x"].double(), b["s"]
        z = torch.cat([x, torch.tanh(net.emb(s))], 1)
        L1, L2 = net.layers
        phi = torch.stack([L1.edge(i, z[:, i]) for i in range(z.shape[1])], 1)          # (n, 10, 6)
        pre = phi.sum(1) + L1.base.bias
        assert torch.allclose(pre, L1(z), rtol=1e-9, atol=1e-9)
        u = torch.tanh(pre)
        psi = torch.stack([L2.edge(j, u[:, j])[:, 0] for j in range(u.shape[1])], 1)  # (n, 6)
        h = psi.sum(1) + L2.base.bias[0]
        ref = net(x, s)[:, 0]
        rel = float(((h - ref).abs() / ref.abs().clamp_min(1e-12)).max())
        assert rel < 1e-6, rel
    return phi.numpy(), psi.numpy(), float(L2.base.bias[0]), h.numpy(), rel


def main():
    log("#" * 100)
    log("paper2/revision/edgedecomp.py: edge decomposition of the fold A ensembles (POST-HOC, DESCRIPTIVE; plan edgedecomp_plan.md)")
    log("#" * 100)
    S = SR.setup()
    tr, fa = S["tr"], S["folds"]["A"]
    names = [f"scbase_A_s{s}.pt" for s in range(5)] + [f"scbase_A_s{s}_pol.pt" for s in range(5)] + \
            [f"close_foldA_1e-06_s{s}.pt" for s in range(5)] + [f"certify_foldA_0.0001_s{s}.pt" for s in range(5)]
    blobs = in_memory(extract_verified(names))
    bt, bh = subset(S["B"], fa["train"]), subset(S["B"], fa["test"])
    btd = SR.dbl(bt)
    used = np.unique(S["B"]["s"].numpy()[fa["train"]]).tolist()
    d, ns = len(DELTA_IN), len(S["salts"])
    rows = {"train": bt, "foldA": bh}
    xco = {k: v["x_co"].double().numpy() for k, v in rows.items()}
    y = tr.log_k.values[fa["test"]]; anc = tr.anchor.values[fa["test"]]
    D = {st: [] for st in STATES}    # per seed: {rowset: decomposition}
    maxrel = 0.0
    for s in range(5):
        sd = torch.load(io.BytesIO(blobs[f"scbase_A_s{s}.pt"]), weights_only=False)
        sd = sd["early"] if isinstance(sd, dict) and "early" in sd else sd
        def fresh():
            n = KAN(d, [6, 1], ns); n.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False); return n.eval()
        nets = {"early": fresh(),
                "mu = 0": Flat(fresh(), btd, used).to_kan(torch.load(io.BytesIO(blobs[f"scbase_A_s{s}_pol.pt"]), weights_only=False)["x"].numpy(), d, ns)}
        for mu, nm, st in (("1e-06", f"close_foldA_1e-06_s{s}.pt", "mu = 1e-6"), ("0.0001", f"certify_foldA_0.0001_s{s}.pt", "mu = 1e-4")):
            xv = torch.load(io.BytesIO(blobs[nm]), weights_only=False)
            xv = xv.numpy() if torch.is_tensor(xv) else np.asarray(xv)
            nets[st] = FlatQ(fresh(), btd, used, float(mu)).to_kan(xv, d, ns)
        for st, net in nets.items():
            rec = {}
            for rs, b in rows.items():
                phi, psi, b2, h, rel = decompose(net, b)
                maxrel = max(maxrel, rel)
                rec[rs] = dict(phi=phi, psi=psi, b2=b2, h=h)
            D[st].append(rec)
    log(f"  decomposition reproduces the network output: max relative error {maxrel:.2e} (required < 1e-6)")
    rm = {st: float(np.sqrt(np.mean((anc + np.mean([xco['foldA'] * r['foldA']['h'] for r in D[st]], 0) - y) ** 2))) for st in STATES}
    log("  fold A ensemble RMSE: " + ", ".join(f"{st} {rm[st]:.4f} (logged {LOGGED[st]:.4f})" for st in STATES))
    if not all(abs(rm[st] - LOGGED[st]) < 1e-4 for st in STATES):
        raise SystemExit("STOP: the ensembles do not reproduce the logged fold A RMSEs")

    def stats(r):
        S2 = np.abs(r["psi"]).sum(1); N2 = np.abs(r["psi"].sum(1))
        S1 = np.abs(r["phi"]).sum(1); N1 = np.abs(r["phi"].sum(1))       # (n, 6)
        return dict(S2=float(np.median(S2)), N2=float(np.median(N2)), C2=float(np.median(S2 / N2)), N2_over_S2=float(np.median(N2 / S2)),
                    S1=float(np.median(np.median(S1, 1))), N1=float(np.median(np.median(N1, 1))), C1=float(np.median(np.median(S1 / N1, 1))))
    res = {"rmse": rm, "max_rel_error": maxrel, "states": {}, "change": {}}
    log("\n==================== medians over rows (per seed), then median over the 5 seeds ====================")
    log(f"  {'state':10s} {'rows':6s} {'S2':>9s} {'N2':>9s} {'C2=S2/N2':>9s} {'N2/S2':>7s} {'S1':>8s} {'N1':>8s} {'C1':>7s}")
    for st in STATES:
        for rs in rows:
            per = [stats(r[rs]) for r in D[st]]
            m = {k: float(np.median([p[k] for p in per])) for k in per[0]}
            res["states"][f"{st}|{rs}"] = dict(median=m, per_seed=per)
            log(f"  {st:10s} {rs:6s} {m['S2']:9.3f} {m['N2']:9.3f} {m['C2']:9.2f} {m['N2_over_S2']:7.3f} {m['S1']:8.3f} {m['N1']:8.3f} {m['C1']:7.2f}")
    log("\n==================== accounting for the polished change (per seed; median over seeds) ====================")
    log("  NOTE (definition fixed here, the plan named the quantity only): share = sum(x_co d(sum_j psi_j) * x_co dh) / sum((x_co dh)^2), the")
    log("  projection of the change onto its edge-sum part; edge-sum and bias shares add to 1, so a share above 1 means the bias change")
    log("  partly offsets the edge-sum change")
    for st in STATES[1:]:
        out = {}
        for rs in rows:
            share, rho = [], []
            for e, p in zip(D["early"], D[st]):
                dsum = (p[rs]["psi"].sum(1) - e[rs]["psi"].sum(1)) * xco[rs]
                dh = (p[rs]["h"] - e[rs]["h"]) * xco[rs]
                share.append(float(np.sum(dsum * dh) / np.sum(dh ** 2)))         # projection share of sum (x_co dh)^2 carried by the edge sum
                dN2 = np.abs(p[rs]["psi"].sum(1)) - np.abs(e[rs]["psi"].sum(1))
                rho.append(float(spearmanr(np.abs(dh), dN2)[0]))
            ddel = [float(np.median(np.abs((p[rs]["h"] - e[rs]["h"]) * xco[rs]))) for e, p in zip(D["early"], D[st])]
            out[rs] = dict(edge_sum_share=float(np.median(share)), spearman_absdelta_dN2=float(np.median(rho)), median_abs_delta_change=float(np.median(ddel)),
                           per_seed_share=share, per_seed_rho=rho)
            log(f"  {st:10s} {rs:6s}: median |x_co dh| {out[rs]['median_abs_delta_change']:.4f}; share of sum (x_co dh)^2 carried by the layer-2 edge sum "
                f"{out[rs]['edge_sum_share']:.3f} (bias {1 - out[rs]['edge_sum_share']:.3f}); Spearman(|x_co dh|, change in N2) {out[rs]['spearman_absdelta_dN2']:+.3f}")
        res["change"][st] = out
    # reading, by the plan's fixed wording
    log("\n==================== reading (plan: descriptive, no thresholds) ====================")
    for st in STATES[1:]:
        a = res["states"][f"{st}|foldA"]["median"]["N2_over_S2"]; t = res["states"][f"{st}|train"]["median"]["N2_over_S2"]
        sh = res["change"][st]["foldA"]["edge_sum_share"]
        fails = a > t and sh > 0.5
        log(f"  {st}: median N2/S2 fold A {a:.3f} vs training {t:.3f}; edge-sum share of the fold A change {sh:.3f} -> "
            + ("'cancellation fails on fold A' (plan wording)" if fails else "the plan's condition for 'cancellation fails' is not met; reported as found"))
        res.setdefault("reading", {})[st] = dict(foldA_N2_over_S2=a, train_N2_over_S2=t, edge_sum_share=sh, cancellation_fails=bool(fails))
    e_a, e_t = res["states"]["early|foldA"]["median"]["N2_over_S2"], res["states"]["early|train"]["median"]["N2_over_S2"]
    log(f"  note: the fold A / training contrast is present already at early stopping (median N2/S2 {e_a:.3f} vs {e_t:.3f}); polishing")
    log("  raises the absolute contributions S2 far more on both row sets, and the net on fold A rows grows with them")
    # figure
    (HERE / "figs").mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, st in zip(axs[:2], ("early", "mu = 1e-6")):
        r = D[st][0]
        for rs, col in (("train", "#999999"), ("foldA", "#D55E00")):
            ax.scatter(np.abs(r[rs]["psi"]).sum(1), np.abs(r[rs]["psi"].sum(1)), s=4, alpha=0.5, color=col, lw=0, label={"train": "training rows", "foldA": "fold A rows"}[rs])
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1]); ax.plot([0, lim], [0, lim], "k:", lw=0.6)
        ax.set_xlabel(r"$S_2=\sum_j|\psi_j|$"); ax.set_ylabel(r"$N_2=|\sum_j\psi_j|$"); ax.set_title(f"{st}, seed 0 (POST-HOC)", fontsize=8)
        ax.legend(fontsize=6, frameon=False)
    ax = axs[2]
    w = 0.38
    for k, (rs, col) in enumerate((("train", "#999999"), ("foldA", "#D55E00"))):
        ax.bar(np.arange(len(STATES)) + (k - 0.5) * w, [res["states"][f"{st}|{rs}"]["median"]["C2"] for st in STATES], w, color=col, label=rs)
    ax.set_xticks(range(len(STATES))); ax.set_xticklabels(STATES, fontsize=7); ax.set_yscale("log"); ax.set_ylabel(r"median $C_2=S_2/N_2$"); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(HERE / "figs" / "edgedecomp.png", dpi=150); plt.close(fig)
    (HERE / "edgedecomp_results.json").write_text(json.dumps(res, indent=1, default=float))
    (HERE / "edgedecomp_output.txt").write_text("\n".join(LOG) + "\n")
    log("wrote edgedecomp_results.json, edgedecomp_output.txt, figs/edgedecomp.png")


if __name__ == "__main__":
    main()

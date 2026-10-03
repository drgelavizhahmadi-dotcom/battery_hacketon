"""paper2/anatomy_synth: POST-HOC, DESCRIPTIVE anatomy of the synthetic runaway (no verdicts change).
Run from the repo root: python3 paper2/anatomy_synth.py >> paper2/anatomy_synth_output.txt
Inputs: the archived synth weights, SHA-verified and held in memory."""
import io
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import synth as S
from stage2 import extract_verified, in_memory

warnings.filterwarnings("ignore")
FIG = ROOT / "figs" / "anatomy"
CELLS = [("correlated", "P0"), ("correlated", "P1"), ("independent", "P0")]
PAIRS = {"x1/x2": [0, 1], "x3/x4": [2, 3]}
PARTS = ["affine RBF", "curved RBF", "SiLU", "bias"]


def log(*a):
    print(*a, flush=True)


def rms(a, axis=0):
    return np.sqrt((a ** 2).mean(axis))


def load(sd):
    net = S.SynthKAN().double()
    net.load_state_dict({k: v.double() for k, v in sd.items()})
    return net.eval()


def blocks(net):
    """Per-layer parameter energy split into affine RBF / curved RBF / SiLU / bias, plus raw tensors."""
    sd = {k: v.detach() for k, v in net.named_parameters()}
    out = {}
    for L in (0, 1):
        c = sd[f"layers.{L}.coef"]
        aff = float(((c @ S.NB) ** 2).sum()); tot = float((c ** 2).sum())
        out[L] = {"affine RBF": aff, "curved RBF": tot - aff, "SiLU": float((sd[f"layers.{L}.base.weight"] ** 2).sum()),
                  "bias": float((sd[f"layers.{L}.base.bias"] ** 2).sum())}
    return out, sd


def delta_shares(e, p):
    _, a = blocks(e); _, b = blocks(p)
    d = {k: b[k] - a[k] for k in a}
    parts = {}
    for L in (0, 1):
        c = d[f"layers.{L}.coef"]
        aff = float(((c @ S.NB) ** 2).sum()); tot = float((c ** 2).sum())
        parts[(L, "affine RBF")] = aff; parts[(L, "curved RBF")] = tot - aff
        parts[(L, "SiLU")] = float((d[f"layers.{L}.base.weight"] ** 2).sum()); parts[(L, "bias")] = float((d[f"layers.{L}.base.bias"] ** 2).sum())
    T = sum(parts.values())
    # input-pair and hidden-unit concentration of the change
    c0, w0 = d["layers.0.coef"], d["layers.0.base.weight"]          # (3, 4, 8), (3, 4)
    edge_e = (c0 ** 2).sum(-1) + w0 ** 2                             # (3 units, 4 inputs)
    pair = {k: float(edge_e[:, v].sum() / edge_e.sum()) for k, v in PAIRS.items()}
    unit_e = edge_e.sum(1) + d["layers.0.base.bias"] ** 2 + (d["layers.1.coef"][0] ** 2).sum(-1) + d["layers.1.base.weight"][0] ** 2
    us = (unit_e / unit_e.sum()).numpy()
    l1 = sum(v for (L, _), v in parts.items() if L == 0) / T
    return dict(shares={f"L{L + 1} {k}": v / T for (L, k), v in parts.items()}, layer1=l1, pair=pair,
                unit_max=float(us.max()), unit_eff=float(1 / (us ** 2).sum()), total=T)


def cancellation(net, X):
    phi, s, ps = S.parts(net, X)                                     # phi (4, n, 3), s (n, 3), ps (n, 3)
    within, between, cross = [], [], []
    for j in range(3):
        e = phi[:, :, j]
        for v in PAIRS.values():
            within.append((rms(e[v[0]]) + rms(e[v[1]])) / rms(e[v[0]] + e[v[1]]))
        SA, SB = e[0] + e[1], e[2] + e[3]
        between.append((rms(SA) + rms(SB)) / rms(SA + SB))
        cross.append(rms(e, 1).sum() / rms(e.sum(0)))
    l2 = rms(ps).sum() / rms(ps.sum(1))
    u = np.tanh(s)
    return dict(within=float(np.median(within)), between=float(np.median(between)), cross=float(np.median(cross)), layer2=float(l2),
                sat95=float((np.abs(u) > 0.95).mean()), sat_s3=float((np.abs(s) > 3).mean()))


def summarise(rows, key):
    return float(np.median([r[key] for r in rows]))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    log("#" * 100)
    log("POST-HOC, DESCRIPTIVE (decided after the synth results). No pre-registration; no verdict changes.")
    log("#" * 100)
    log("NOTES / discrepancies (files win):")
    log("  every input belongs to a correlated pair (x1/x2, x3/x4), so no layer-1 edge comes from an uncorrelated input;")
    log("  (b) is answered as: share of layer-1 edge growth per pair, layer-1 vs layer-2, unit concentration, and cancellation")
    log("  WITHIN a correlated pair vs BETWEEN the two pairs (both at each hidden unit)")
    log("  affine RBF = projection of each edge's 8 RBF coefficients onto span{1, k} (the D2 null space); curved = the rest")
    res = json.loads((ROOT / "synth_results.json").read_text())["models"]
    names = [f"synth_{c}_{l}_s{s}.pt" for c, l in CELLS for s in S.SEEDS]
    log("\n==================== inputs (SHA-256 verified, held in memory) ====================")
    blobs = in_memory(extract_verified(names))
    data = {c: S.make_inputs(r, 1000 + i) for i, (c, r) in enumerate(S.CONDS.items())}

    rows = {}
    for cond, lev in CELLS:
        X = data[cond]
        for s in S.SEEDS:
            key = f"{cond}_{lev}_s{s}"
            st = torch.load(io.BytesIO(blobs[f"synth_{key}.pt"]))
            e, p = load(st["early"]), load(st["polished"])
            be, _ = blocks(e); bp, _ = blocks(p)
            r = dict(key=key, cell=f"{cond}-{lev}", seed=s, certified=res[key]["certified"],
                     theta_e=sum(sum(v.values()) for v in be.values()), theta_p=sum(sum(v.values()) for v in bp.values()),
                     aff_share_p=bp[0]["affine RBF"] / (bp[0]["affine RBF"] + bp[0]["curved RBF"]),
                     aff_share_e=be[0]["affine RBF"] / (be[0]["affine RBF"] + be[0]["curved RBF"]),
                     d=delta_shares(e, p), ce=cancellation(e, X), cp=cancellation(p, X))
            rows[key] = r

    def report(label, rr):
        log(f"\n  --- {label} (n = {len(rr)}) ---")
        log(f"  ||theta||^2 early -> polished (median): {summarise(rr, 'theta_e'):.1f} -> {summarise(rr, 'theta_p'):.1f} "
            f"(x{np.median([r['theta_p'] / r['theta_e'] for r in rr]):.1f})")
        sh = {k: float(np.median([r["d"]["shares"][k] for r in rr])) for k in rr[0]["d"]["shares"]}
        log("  (a) share of ||delta theta||^2 (median): " + ", ".join(f"{k} {v:.3f}" for k, v in sh.items()))
        log(f"      layer 1 total {np.median([r['d']['layer1'] for r in rr]):.3f}; affine share of layer-1 RBF energy early {summarise(rr, 'aff_share_e'):.3f} "
            f"-> polished {summarise(rr, 'aff_share_p'):.3f}")
        log(f"  (b) layer-1 edge growth by input pair (median share): " + ", ".join(f"{k} {np.median([r['d']['pair'][k] for r in rr]):.3f}" for k in PAIRS) +
            f"; largest single unit's share {np.median([r['d']['unit_max'] for r in rr]):.3f} "
            f"(effective number of units {np.median([r['d']['unit_eff'] for r in rr]):.2f} of 3)")
        for nm, k in [("within a correlated pair", "within"), ("between the two pairs", "between"), ("all 4 layer-1 edges", "cross"),
                      ("(d) layer 2, across the 3 edges into the output", "layer2")]:
            log(f"      cancellation {nm}: early {np.median([r['ce'][k] for r in rr]):.2f} -> polished {np.median([r['cp'][k] for r in rr]):.2f}")
        log(f"  (c) saturation |tanh(s)| > 0.95: early {np.median([r['ce']['sat95'] for r in rr]):.3f} -> polished {np.median([r['cp']['sat95'] for r in rr]):.3f}; "
            f"|s| > 3: early {np.median([r['ce']['sat_s3'] for r in rr]):.3f} -> polished {np.median([r['cp']['sat_s3'] for r in rr]):.3f}")

    log("\n==================== per cell ====================")
    for cond, lev in CELLS:
        report(f"{cond}-{lev}", [rows[f"{cond}_{lev}_s{s}"] for s in S.SEEDS])
    log("\n==================== (e) independent-P0: never certified vs certified ====================")
    ip = [rows[f"independent_P0_s{s}"] for s in S.SEEDS]
    report("independent-P0 NOT certified (unbounded growth): seeds " + ", ".join(str(r["seed"]) for r in ip if not r["certified"]), [r for r in ip if not r["certified"]])
    report("independent-P0 certified (finite growth): seeds " + ", ".join(str(r["seed"]) for r in ip if r["certified"]), [r for r in ip if r["certified"]])
    log("\n  per model (growth | top two parts of the change | within-pair, between-pair, layer-2 cancellation at polished | sat95):")
    for r in rows.values():
        top = sorted(r["d"]["shares"].items(), key=lambda kv: -kv[1])[:2]
        log(f"    {r['key']:24s} cert {str(r['certified']):5s} x{r['theta_p'] / r['theta_e']:8.1f} | {top[0][0]} {top[0][1]:.2f}, {top[1][0]} {top[1][1]:.2f} | "
            f"{r['cp']['within']:.2f}, {r['cp']['between']:.2f}, {r['cp']['layer2']:.2f} | {r['cp']['sat95']:.3f}")
    (ROOT / "anatomy_synth_results.json").write_text(json.dumps(dict(post_hoc=True, rows=rows), indent=1, default=float))

    # figures
    groups = [("corr-P0", [rows[f"correlated_P0_s{s}"] for s in S.SEEDS]), ("corr-P1", [rows[f"correlated_P1_s{s}"] for s in S.SEEDS]),
              ("indep-P0 cert", [r for r in ip if r["certified"]]), ("indep-P0 uncert", [r for r in ip if not r["certified"]])]
    keys = list(rows[next(iter(rows))]["d"]["shares"])
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    bottom = np.zeros(len(groups))
    for k in keys:
        v = np.array([np.median([r["d"]["shares"][k] for r in g]) for _, g in groups])
        axs[0].bar([g for g, _ in groups], v, bottom=bottom, label=k); bottom += v
    axs[0].set_ylabel("median share of ||delta theta||^2 (early -> polished)"); axs[0].legend(fontsize=7); axs[0].set_title("POST-HOC: where the parameter change goes")
    x = np.arange(len(groups)); w = 0.2
    for o, (nm, k) in enumerate([("within pair", "within"), ("between pairs", "between"), ("layer 2", "layer2")]):
        axs[1].bar(x + (o - 1) * w, [np.median([r["cp"][k] for r in g]) for _, g in groups], w, label=f"{nm} (polished)")
        axs[1].scatter(x + (o - 1) * w, [np.median([r["ce"][k] for r in g]) for _, g in groups], color="k", s=12, zorder=3)
    axs[1].set_xticks(x); axs[1].set_xticklabels([g for g, _ in groups]); axs[1].set_ylabel("cancellation ratio (dots: early)")
    axs[1].legend(fontsize=8); axs[1].set_title("POST-HOC: cancellation within/between correlated pairs and in layer 2")
    fig.tight_layout(); fig.savefig(FIG / "anatomy_synth.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()

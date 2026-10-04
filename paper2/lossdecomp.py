"""paper2/lossdecomp: POST-HOC, DESCRIPTIVE loss decomposition along the synthetic runaway (no verdicts change).
Run from the repo root: python3 paper2/lossdecomp.py > paper2/lossdecomp_output.txt
Inputs: the archived synth weights (archive h), SHA-verified against weights_manifest.txt and held in memory;
data and teacher regenerated in memory from synth.py's fixed seeds."""
import io
import json
import re
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
TERMS = ["data", "D2", "SiLU", "muQ"]


def log(*a):
    print(*a, flush=True)


def load(sd):
    net = S.SynthKAN().double()
    net.load_state_dict({k: v.double() for k, v in sd.items()})
    return net.eval()


def terms(net, X, y, level, mu):
    with torch.no_grad():
        prm = dict(net.named_parameters())
        d2, silu, q = S.penalty_terms(prm)
        data = float(((S.predict(net, X) - y) ** 2).mean())
    t = dict(data=data, D2=float(d2), SiLU_raw=float(silu), Q_raw=float(q))
    t["SiLU"] = t["SiLU_raw"] if level in ("P1", "P2") else 0.0
    t["muQ"] = mu * t["Q_raw"] if level == "P2" else 0.0
    t["total"] = t["data"] + t["D2"] + t["SiLU"] + t["muQ"]
    return t


def real_reference():
    """Seeds' 107-epoch -> capped and capped -> A final rows from the committed anatomy_output.txt (data, penalty, total)."""
    txt = (REPO / "anatomy_output.txt").read_text().splitlines()
    rows = {}
    for l in txt:
        m = re.match(r"\s+ind (\d)\s+(107-epoch|capped|A)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)", l)
        if m:
            rows.setdefault(int(m.group(1)), {}).setdefault(m.group(2), tuple(float(m.group(k)) for k in (3, 4, 5)))  # first table only
    return rows


def main():
    (ROOT / "figs").mkdir(exist_ok=True)
    log("#" * 100)
    log("POST-HOC, DESCRIPTIVE (decided after the synth results). No pre-registration; no verdict changes.")
    log("#" * 100)
    log("NOTES / discrepancies (files win):")
    log("  history: synth_results.json stores only ||theta||^2 per AdamW 10 epochs ('traj') and per trust-exact iteration ('trust_theta2');")
    log("    no intermediate iterates and no per-iteration loss components were saved -> EARLY vs POLISHED ONLY")
    log("  'early' = AdamW early-stopped on the validation data term (synth.py), not 107 epochs; 'polished' = trust-exact on the cell loss from early")
    log("  loss = MSE + lambda*sum||D2 c||^2 (+ lambda*K_silu*sum alpha^2 in P1/P2) (+ mu*Q in P2), synth.cell_penalty; mu = 1e-5 (frozen)")
    log("  SiLU raw and Q raw are reported for every cell; they enter the loss only where the cell uses them")
    log("  shares: per model, share_k = (term_k change) / (total change); shares sum to 1 per model and for the pooled (sum over seeds)")
    log("    cell figures; per-cell MEDIANS of shares are reported too and need not sum to 1")
    log("  runaway set = correlated P0, correlated P1 (all 8 each) + independent P0/P1 seeds 0 and 4 (the 4 never-certified independent models)")
    log("  reference numbers in the brief combine two different real-data comparisons (anatomy_output.txt, seeds):")
    log("    'penalty fell 8-88x' is 107 epochs -> capped (where the data term ALSO fell about 50%);")
    log("    'data term changed 1-5%' is capped -> A final (where the penalty changed only x0.71-1.20, up or down); both are recomputed below")

    res = json.loads((ROOT / "synth_results.json").read_text())
    mu = res["mu"]
    names = [f"synth_{c}_{l}_s{s}.pt" for c in S.CONDS for l in S.LEVELS for s in S.SEEDS]
    log("\n==================== inputs (SHA-256 verified against weights_manifest.txt, held in memory) ====================")
    blobs = in_memory(extract_verified(names))

    teacher = S.build_teacher(res["teacher_scale"])
    data = {}
    for ci, (cond, rho) in enumerate(S.CONDS.items()):
        X = S.make_inputs(rho, 1000 + ci)
        y0 = S.predict(teacher, X); sig = 0.05 * y0.std()
        y = y0 + sig * np.random.default_rng(5000 + ci).standard_normal(S.N)
        Xt, Xo = S.make_inputs(rho, 3000 + ci), S.make_inputs(rho, 4000 + ci, independent=True)
        data[cond] = dict(X=X, y=y, Xt=Xt, yt=S.predict(teacher, Xt), Xo=Xo, yo=S.predict(teacher, Xo))

    rows, chk = {}, []
    for cond in S.CONDS:
        d = data[cond]
        for lev in S.LEVELS:
            for s in S.SEEDS:
                key = f"{cond}_{lev}_s{s}"
                st = torch.load(io.BytesIO(blobs[f"synth_{key}.pt"]))
                r = dict(cond=cond, lev=lev, seed=s, certified=res["models"][key]["certified"],
                         runaway=(cond == "correlated" and lev in ("P0", "P1")) or (cond == "independent" and lev in ("P0", "P1") and s in (0, 4)))
                for state in ("early", "polished"):
                    net = load(st[state])
                    t = terms(net, d["X"], d["y"], lev, mu)
                    t["test"] = float(((S.predict(net, d["Xt"]) - d["yt"]) ** 2).mean())
                    t["off"] = float(((S.predict(net, d["Xo"]) - d["yo"]) ** 2).mean())
                    r[state] = t
                stored = res["models"][key]["early_terms"]["data"]
                chk.append(abs(r["early"]["data"] - stored) / stored)
                dT = r["polished"]["total"] - r["early"]["total"]
                r["dtotal"] = dT
                r["share"] = {k: (r["polished"][k] - r["early"][k]) / dT for k in TERMS}
                rows[key] = r
    log(f"  regeneration check: recomputed early data term vs synth_results.json early_terms: max rel diff {max(chk):.1e}")

    def summary(label, rr):
        log(f"\n  --- {label} (n = {len(rr)}) ---")
        dt = [r["dtotal"] for r in rr]
        rel = [r["dtotal"] / r["early"]["total"] for r in rr]
        log(f"  total loss early -> polished: median {np.median([r['early']['total'] for r in rr]):.4e} -> {np.median([r['polished']['total'] for r in rr]):.4e}; "
            f"change median {np.median(dt):+.3e} ({np.median(rel):+.2%})")
        pooled = {k: sum(r["polished"][k] - r["early"][k] for r in rr) / sum(dt) for k in TERMS}
        log("  share of the change, pooled (sums to 1): " + ", ".join(f"{k} {v:+.3f}" for k, v in pooled.items()))
        log("  share of the change, median over models: " + ", ".join(f"{k} {np.median([r['share'][k] for r in rr]):+.3f}" for k in TERMS))
        for k, nm in [("data", "data MSE"), ("D2", "D2 penalty"), ("SiLU_raw", "SiLU raw"), ("Q_raw", "Q raw"), ("test", "test MSE (teacher)"), ("off", "off-manifold MSE")]:
            ratio = [r["polished"][k] / r["early"][k] if r["early"][k] > 0 else np.nan for r in rr]
            log(f"    {nm:20s} median {np.median([r['early'][k] for r in rr]):.4e} -> {np.median([r['polished'][k] for r in rr]):.4e}  "
                f"(per-model ratio median x{np.nanmedian(ratio):.3f}, range x{np.nanmin(ratio):.3f}-x{np.nanmax(ratio):.3f})")
        return pooled

    log("\n==================== 1. per cell ====================")
    pooled = {}
    for cond in S.CONDS:
        for lev in S.LEVELS:
            pooled[(cond, lev)] = summary(f"{cond} {lev}", [rows[f"{cond}_{lev}_s{s}"] for s in S.SEEDS])

    log("\n==================== 2. runaway vs certified ====================")
    run = [r for r in rows.values() if r["runaway"]]
    cert = [r for r in rows.values() if r["certified"]]
    log(f"  runaway models: {len(run)} (certified among them: {sum(r['certified'] for r in run)}); certified models: {len(cert)}")
    summary("RUNAWAY (correlated P0/P1 + independent P0/P1 s0, s4)", run)
    summary("CERTIFIED (all cells)", cert)
    summary("CERTIFIED, P0/P1 only (independent seeds 1-3, 5-7)", [r for r in cert if r["lev"] != "P2"])

    log("\n==================== 3. overfitting check in the runaway models ====================")
    n_of = 0
    for r in run:
        tr_ = r["polished"]["data"] < r["early"]["data"]; te_ = r["polished"]["test"] > r["early"]["test"]
        n_of += tr_ and te_
        log(f"    {r['cond']:11s} {r['lev']} s{r['seed']}: train MSE {r['early']['data']:.4e} -> {r['polished']['data']:.4e} ({r['polished']['data'] / r['early']['data'] - 1:+.2%}); "
            f"test {r['early']['test']:.4e} -> {r['polished']['test']:.4e} ({r['polished']['test'] / r['early']['test'] - 1:+.2%}); "
            f"off-manifold {r['polished']['off'] / r['early']['off'] - 1:+.2%}; total {r['dtotal'] / r['early']['total']:+.2%}"
            f"{'  <- train down, test up' if tr_ and te_ else ''}")
    log(f"  train MSE down AND test MSE up: {n_of} of {len(run)} runaway models")

    log("\n==================== real-data reference (recomputed from the committed anatomy_output.txt, seeds ind 0-5) ====================")
    ref = real_reference()
    for a, b in [("107-epoch", "capped"), ("capped", "A")]:
        sh, dd, pp = [], [], []
        for i, v in sorted(ref.items()):
            if a in v and b in v:
                (d0, p0, t0), (d1, p1, t1) = v[a], v[b]
                sh.append((d1 - d0) / (t1 - t0)); dd.append(d1 / d0 - 1); pp.append(p0 / p1)
        log(f"  {a} -> {b}: data term change {min(dd):+.1%} to {max(dd):+.1%}; penalty before/after ratio x{min(pp):.2f}-x{max(pp):.2f} (>1 = fell); "
            f"data share of the total-loss change {min(sh):.3f}-{max(sh):.3f} (median {np.median(sh):.3f})")

    # figure
    cells = [(c, l) for c in S.CONDS for l in S.LEVELS]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(cells))
    for sign in (1, -1):
        bottom = np.zeros(len(cells))
        for k, col in zip(TERMS, ["C0", "C1", "C2", "C3"]):
            v = np.array([np.median([rows[f"{c}_{l}_s{s}"]["polished"][k] - rows[f"{c}_{l}_s{s}"]["early"][k] for s in S.SEEDS])
                          / abs(np.median([rows[f"{c}_{l}_s{s}"]["early"]["total"] for s in S.SEEDS])) for c, l in cells])
            v = np.where(sign * v > 0, v, 0)
            ax.bar(x, v, bottom=bottom, color=col, label=k if sign == 1 else None); bottom += v
    ax.scatter(x, [np.median([rows[f"{c}_{l}_s{s}"]["dtotal"] / rows[f"{c}_{l}_s{s}"]["early"]["total"] for s in S.SEEDS]) for c, l in cells],
               color="k", zorder=3, label="total (median)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n{l}" for c, l in cells])
    ax.set_ylabel("median term change / median early total loss")
    ax.set_title("POST-HOC: loss change early -> polished by term (synthetic 2 x 3, median over 8 seeds)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(ROOT / "figs" / "lossdecomp.png", dpi=110); plt.close(fig)
    (ROOT / "lossdecomp_results.json").write_text(json.dumps(dict(post_hoc=True, rows=rows), indent=1, default=float))


if __name__ == "__main__":
    main()

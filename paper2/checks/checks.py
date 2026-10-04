"""paper2/checks: DESCRIPTIVE, POST-HOC checks for paper 2's open items (see paper2/checks/plan.md, 5745103).
No training, no verdict changes. Run from the repo root:
  python3 paper2/checks/checks.py > paper2/checks/checks_output.txt
Inputs: archived weights SHA-verified against weights_manifest.txt and held in memory."""
import io
import json
import subprocess
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "paper2")); sys.path.insert(0, str(REPO / "arch"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(4)


def log(*a):
    print(*a, flush=True)


def stats(H):
    """H: (n, units) activations after tanh."""
    sd = H.std(0)
    live = sd > 1e-12
    C = np.corrcoef(H[:, live].T) if live.sum() > 1 else np.ones((1, 1))
    off = np.abs(C[~np.eye(len(C), dtype=bool)]) if len(C) > 1 else np.array([np.nan])
    ev = np.clip(np.linalg.eigvalsh(np.cov(H.T)), 0, None)
    return dict(corr=C.round(4).tolist(), med=float(np.median(off)), max=float(np.max(off)), pr=float(ev.sum() ** 2 / (ev ** 2).sum()),
                units=int(H.shape[1]), dead=int((~live).sum()))


def main():
    from stage2 import extract_verified, in_memory
    import synth as SY
    import sc_run as SR
    from kan import KAN, DELTA_IN, N_RBF, subset
    from trust import Flat
    from sc_model import SCKAN, FlatSC

    log("#" * 100)
    log("paper2/checks: DESCRIPTIVE, POST-HOC (plan 5745103). No training; no verdict changes; nothing here is pre-registered.")
    log("#" * 100)
    log("NOTES / discrepancies (files win):")
    log("  real-data training rows = all 2069 anchored rows (the seeds' training set); SC/baseline fold A rows = the 1434 fold A training rows")
    log("  stage2 Part A finals are saved as flat parameter vectors; rebuilt with trust.Flat on their fiber2-capped start (unused salt rows fixed)")
    log("  SC polished models are flat vectors too, rebuilt with sc_model.FlatSC.to_net (centring mean recomputed on the training rows)")
    log("  synth 'runaway' = not certified (synth_results.json), as in the brief; real-data stages have no certified models (stage2 U1 failed)")

    names = ([f"synth_{c}_{l}_s{s}.pt" for c in SY.CONDS for l in SY.LEVELS for s in SY.SEEDS] +
             [f"delta_s{i}.pt" for i in range(6)] + [f"fiber2_ind_{i}.pt" for i in range(6)] + [f"stage2_A_ind_{i}.pt" for i in range(6)] +
             [f"sc_A_b0.0001_s{s}.pt" for s in range(10)] + [f"sc_A_b0.0001_s{s}_pol.pt" for s in range(10)] +
             [f"scbase_A_s{s}.pt" for s in range(10)] + [f"scbase_A_s{s}_pol.pt" for s in range(10)])
    log("\n==================== inputs (SHA-256 verified against weights_manifest.txt, held in memory) ====================")
    blobs = in_memory(extract_verified(names))
    ld = lambda n: torch.load(io.BytesIO(blobs[n]), weights_only=False)
    R = {"item1": {}}

    # ---------------- ITEM 1
    log("\n==================== ITEM 1: hidden-unit correlation (claim F4; POST-HOC expectation: runaway > converged) ====================")
    res = json.loads((REPO / "paper2" / "synth_results.json").read_text())
    rows = []
    for ci, (cond, rho) in enumerate(SY.CONDS.items()):
        X = torch.tensor(SY.make_inputs(rho, 1000 + ci), dtype=torch.float64)
        for lev in SY.LEVELS:
            for s in SY.SEEDS:
                st = ld(f"synth_{cond}_{lev}_s{s}.pt")
                cert = res["models"][f"{cond}_{lev}_s{s}"]["certified"]
                for state in ("early", "polished"):
                    net = SY.SynthKAN().double(); net.load_state_dict({k: v.double() for k, v in st[state].items()})
                    with torch.no_grad():
                        H = torch.tanh(net.layers[0](X)).numpy()
                    rows.append(dict(group="synth", cond=cond, lev=lev, seed=s, state=state, certified=cert, **stats(H)))

    S = SR.setup()
    B, n_salt, d = S["B"], len(S["salts"]), len(DELTA_IN)
    def kan_from(sd):
        k = KAN(d, [6, 1], n_salt)
        miss, unexp = k.load_state_dict({n: t.float() for n, t in sd.items()}, strict=False)
        assert not unexp and all(x.endswith(".grid") for x in miss)
        return k.eval()
    def kan_hidden(net, b):
        net = net.double()
        with torch.no_grad():
            z = torch.cat([b["x"].double(), torch.tanh(net.emb(b["s"]))], 1)
            return torch.tanh(net.layers[0](z)).numpy()
    ball = subset(B, S["use"])
    used_all = np.unique(ball["s"].numpy()).tolist()
    bd_all = SR.dbl(ball)
    for i in range(6):
        for stage, src in [("107-epoch (extra)", f"delta_s{i}.pt"), ("fiber2 capped", f"fiber2_ind_{i}.pt"), ("stage2 Part A final", f"stage2_A_ind_{i}.pt")]:
            if stage.startswith("stage2"):
                fl = Flat(kan_from(ld(f"fiber2_ind_{i}.pt")), bd_all, used_all)
                net = fl.to_kan(ld(src).numpy(), d, n_salt)
            else:
                net = kan_from(ld(src))
            rows.append(dict(group="real", seed=i, state=stage, certified=False, **stats(kan_hidden(net, ball))))

    bA = subset(B, S["folds"]["A"]["train"]); bAd = SR.dbl(bA)
    usedA = np.unique(bA["s"].numpy()).tolist()
    for s in range(10):
        e = SCKAN(n_salt); e.load_state_dict(ld(f"sc_A_b0.0001_s{s}.pt")["early"]); e = e.double().eval()
        po = ld(f"sc_A_b0.0001_s{s}_pol.pt")
        p = FlatSC(e, bA, 1e-4).to_net(po["x"].numpy())
        for state, net in [("early", e), ("polished", p)]:
            with torch.no_grad():
                H = torch.tanh(net.hidden(bAd["x"], bAd["s"])).numpy()
            rows.append(dict(group="sc", seed=s, state=state, certified=bool(po["cert"]["certified"]) if state == "polished" else None, **stats(H)))
        be = kan_from(ld(f"scbase_A_s{s}.pt")["early"])
        bp_ = ld(f"scbase_A_s{s}_pol.pt")
        bp = Flat(kan_from(ld(f"scbase_A_s{s}.pt")["early"]), bAd, usedA).to_kan(bp_["x"].numpy(), d, n_salt)
        for state, net in [("early", be), ("polished", bp)]:
            rows.append(dict(group="scbase (extra)", seed=s, state=state, certified=bool(bp_["cert"]["certified"]) if state == "polished" else None,
                             **stats(kan_hidden(net, bA))))
    R["item1"]["rows"] = rows

    def summ(label, rr):
        if not rr:
            return
        log(f"  {label:58s} n {len(rr):3d}  median|offdiag| {np.median([r['med'] for r in rr]):.3f}  max|offdiag| median {np.median([r['max'] for r in rr]):.3f}"
            f" (range {min(r['max'] for r in rr):.3f}-{max(r['max'] for r in rr):.3f})  PR median {np.median([r['pr'] for r in rr]):.2f} of {rr[0]['units']}"
            + (f"  dead units {sum(r['dead'] for r in rr)}" if any(r['dead'] for r in rr) else ""))
    syn = [r for r in rows if r["group"] == "synth"]
    log("\n  synthetic students (3 hidden units), per cell and state:")
    for cond in SY.CONDS:
        for lev in SY.LEVELS:
            for st in ("early", "polished"):
                rr = [r for r in syn if r["cond"] == cond and r["lev"] == lev and r["state"] == st]
                summ(f"{cond} {lev} {st} (certified {sum(r['certified'] for r in rr)}/8)", rr)
    log("\n  synthetic: certified vs not (polished state; early state in brackets of the same models):")
    for c in (False, True):
        summ(f"{'NOT certified (runaway)' if not c else 'certified'} - polished", [r for r in syn if r["certified"] == c and r["state"] == "polished"])
        summ(f"{'NOT certified (runaway)' if not c else 'certified'} - early", [r for r in syn if r["certified"] == c and r["state"] == "early"])
    log("\n  synthetic: independent vs correlated cells, P0/P1 only (the runaway levels):")
    for cond in SY.CONDS:
        for st in ("early", "polished"):
            summ(f"{cond} P0+P1 {st}", [r for r in syn if r["cond"] == cond and r["lev"] in ("P0", "P1") and r["state"] == st])
    log("\n  synthetic independent P0/P1: the 4 never-certified (s0, s4) vs the 12 certified, polished:")
    summ("independent P0/P1 NOT certified (s0, s4) - polished", [r for r in syn if r["cond"] == "independent" and r["lev"] != "P2" and not r["certified"] and r["state"] == "polished"])
    summ("independent P0/P1 certified - polished", [r for r in syn if r["cond"] == "independent" and r["lev"] != "P2" and r["certified"] and r["state"] == "polished"])
    for r in [r for r in syn if r["cond"] == "independent" and r["lev"] != "P2" and r["state"] == "polished"]:
        pass
    log("\n  real-data KAN-delta seeds (6 hidden units, 2069 anchored rows), per stage:")
    for st in ("107-epoch (extra)", "fiber2 capped", "stage2 Part A final"):
        summ(st, [r for r in rows if r["group"] == "real" and r["state"] == st])
    log("\n  fold A, 1434 rows: SC-KAN-delta (10/10 certified at polish) vs baseline KAN-delta (extra; 0/10 certified):")
    for g in ("sc", "scbase (extra)"):
        for st in ("early", "polished"):
            summ(f"{g} {st}", [r for r in rows if r["group"] == g and r["state"] == st])
    log("\n  per-model detail (real seeds): " + "; ".join(f"ind{r['seed']} {r['state'].split()[0]} max {r['max']:.2f} PR {r['pr']:.2f}" for r in rows if r["group"] == "real"))

    # figure
    fig, ax = plt.subplots(figsize=(13, 5))
    groups = []
    for cond in SY.CONDS:
        for lev in SY.LEVELS:
            for st in ("early", "polished"):
                groups.append((f"{cond[:5]} {lev}\n{st[:4]}", [r for r in syn if r["cond"] == cond and r["lev"] == lev and r["state"] == st]))
    for st in ("107-epoch (extra)", "fiber2 capped", "stage2 Part A final"):
        groups.append((f"real\n{st.split()[0]}", [r for r in rows if r["group"] == "real" and r["state"] == st]))
    for g in ("sc", "scbase (extra)"):
        for st in ("early", "polished"):
            groups.append((f"{g.split()[0]}\n{st[:4]}", [r for r in rows if r["group"] == g and r["state"] == st]))
    for k, (lab, rr) in enumerate(groups):
        y = [r["max"] for r in rr]
        col = ["C3" if (r["certified"] is False) else "C0" if r["certified"] else "grey" for r in rr]
        ax.scatter(np.full(len(y), k) + np.random.default_rng(k).uniform(-0.2, 0.2, len(y)), y, c=col, s=14, alpha=0.8)
        ax.plot([k - 0.3, k + 0.3], [np.median(y)] * 2, color="k")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=6)
    ax.set_ylabel("max |off-diagonal correlation| of hidden units"); ax.set_ylim(0, 1.02)
    ax.set_title("POST-HOC: hidden-unit correlation (red: not certified / runaway, blue: certified, grey: early state with no certificate)")
    fig.tight_layout(); fig.savefig(HERE / "hidden_corr.png", dpi=110); plt.close(fig)

    # ---------------- ITEM 2
    log("\n==================== ITEM 2: how affine is the tilt (claim F2) ====================")
    from kan import KANLayer
    L = KANLayer(1, 1)
    g, h = L.grid.numpy(), L.h
    log(f"  final.py edge (kan.KANLayer): {N_RBF} Gaussian RBFs, centres linspace({g[0]:.0f}, {g[-1]:.0f}, {N_RBF}), width h = {h:.6f} (= 2/7), grid range [-1, 1]")
    from sc_model import SCKAN as _S
    sct = _S(12)
    same = bool(np.allclose(sct.grid.numpy(), g) and abs(sct.h - h) < 1e-15)
    log(f"  synth model: SynthKAN = two kan.KANLayers -> identical edge type; SC-KAN-delta curves: same centres/width: {same}")
    x = np.linspace(-1, 1, 20001)
    Bm = np.exp(-(((x[:, None] - g) / h) ** 2))
    k = np.arange(N_RBF, dtype=float)
    out = {}
    for nm, c in [("linear pattern c_k ~ k - mean(k)", (k - k.mean()) / np.linalg.norm(k - k.mean())), ("constant pattern c_k = 1", np.ones(N_RBF) / np.sqrt(N_RBF))]:
        f = Bm @ c
        res_ = {}
        for rng_nm, m in [("full [-1, 1]", np.ones_like(x, bool)), ("central 80% [-0.8, 0.8]", np.abs(x) <= 0.8)]:
            fm, xm = f[m], x[m]
            if nm.startswith("linear"):
                A = np.column_stack([np.ones_like(xm), xm]); coef, *_ = np.linalg.lstsq(A, fm, rcond=None); fit = A @ coef; rngv = fm.max() - fm.min()
            else:
                fit = np.full_like(fm, fm.mean()); rngv = fm.mean()   # relative to the level for the constant pattern
            dev = fm - fit
            res_[rng_nm] = dict(max_rel=float(np.abs(dev).max() / rngv), rms_rel=float(np.sqrt((dev ** 2).mean()) / rngv),
                                ref=float(rngv), slope=float(coef[1]) if nm.startswith("linear") else None)
            log(f"  {nm:34s} {rng_nm:24s}: max deviation {res_[rng_nm]['max_rel']:.3%}, RMS {res_[rng_nm]['rms_rel']:.3%} of the "
                f"{'function range' if nm.startswith('linear') else 'mean level'} ({rngv:.4f})")
        out[nm] = res_
    R["item2"] = out

    # ---------------- ITEM 3
    log("\n==================== ITEM 3: reference commits and data citation ====================")
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    fin = git("log", "--format=%h %ad %s", "--date=short", "--diff-filter=A", "--", "final.py")
    rec = git("log", "--format=%h %ad %s", "--date=short", "-S", "KAN 0.1420", "--", "kan_output.txt")
    line = [l for l in (REPO / "kan_output.txt").read_text().splitlines() if "PRE-REG 1" in l]
    log(f"  final.py (KAN-delta ensemble, seeds 0-4, 107 epochs) first committed in: {fin}")
    log(f"  fold A RMSE 0.1420 first recorded (kan_output.txt) in: {rec}")
    log(f"    kan_output.txt: {line[0].strip() if line else 'n/a'}")
    log("    (the same 0.1420 is reproduced by the cached baseline ensemble in arch/sc, e00e37f, and arch/ex, 579b58f)")
    hits = []
    for p in list((REPO / "ca-li-sol-23-challenge").glob("*.csv")) + [REPO / "README.md"]:
        if p.name in ("train.csv", "test.csv", "sample_submission.csv"):
            continue
        for l in p.read_text(errors="ignore").splitlines():
            if any(w in l.lower() for w in ["cite", "citation", "et al", "scientific data", "s41597", "de blasio", "licen"]):
                hits.append(f"{p.name}: {l[:120]}")
    log(f"  citation search in the competition material (metaData.csv, solvent_properties.csv) and README.md: {len(hits)} hits")
    for hh in hits:
        log(f"    {hh}")
    log("  -> neither the competition files nor the repo docs carry a citation for CALiSol-23 (the metadata only calls it 'the CALiSol-23 benchmark').")
    log("  Published dataset article (found online via PubMed Central PMC11237020, NOT from the competition material):")
    log("    de Blasio, P., Elsborg, J., Vegge, T., Flores, E. & Bhowmik, A. CALiSol-23: Experimental electrolyte conductivity data for")
    log("    various Li-salts and solvent combinations. Scientific Data 11, 750 (2024). https://doi.org/10.1038/s41597-024-03575-8")
    log("    (article licence CC BY 4.0, quoted from the PMC page; dataset at DTU Data under CC BY 4.0 per a tool summary of its data availability statement, not checked verbatim)")
    R["item3"] = dict(final_py_commit=fin, foldA_0142_commit=rec, citation_hits=hits,
                      citation="de Blasio P, Elsborg J, Vegge T, Flores E, Bhowmik A. Sci Data 11, 750 (2024). doi:10.1038/s41597-024-03575-8",
                      citation_source="PubMed Central PMC11237020 (not competition material)")
    (HERE / "checks_results.json").write_text(json.dumps(R, indent=1, default=float))


if __name__ == "__main__":
    main()

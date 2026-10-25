"""paper2/revision/appdomain.py: applicability domain (DESCRIPTIVE; plan appdomain_plan.md, ec769ad). No training.
Run from the repo root with the pinned interpreter:
  /usr/bin/python3 paper2/revision/appdomain.py >> paper2/revision/appdomain_output.txt"""
import io
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "paper2", REPO / "arch", HERE):
    sys.path.insert(0, str(p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linprog
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf

import anchor as AN
import sc_run as SR
from close import FlatQ
from kan import DELTA_IN, KAN, co_props, delta_head, predict, subset
from stage2 import extract_verified, in_memory

FIG = HERE / "figs"
OUT = HERE / "appdomain_results.json"
IA = list(range(8))
IB = [DELTA_IN.index(c) for c in ("eps_co", "lneta_co", "M_co")]
IP = [DELTA_IN.index(c) for c in ("eps_co", "lneta_co")]
IM = [DELTA_IN.index(c) for c in ("mix_eps", "mix_lneta")]
LOGGED = {"early": 0.1420, "1e-06": 1.0262, "0.0001": 0.3345}


def log(*a):
    print(*a, flush=True)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ------------------------------------------------------------------ distances (plan, fixed)
def in_hull(P, q):
    P = np.unique(np.round(P, 12), axis=0)
    n = len(P)
    A_eq = np.vstack([P.T, np.ones(n)]); b_eq = np.concatenate([q, [1.0]])
    r = linprog(np.zeros(n), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n, method="highs")
    return bool(r.status == 0)


def off_pattern(Ztr, Zte):
    m = Ztr.mean(0); C = np.cov((Ztr - m).T)
    ev, V = np.linalg.eigh(C)
    u_perp = V[:, 0]                          # second principal direction (smallest eigenvalue)
    return np.abs((Zte - m) @ u_perp), float(np.corrcoef(Ztr.T)[0, 1])


def distances(Xtr, Xte):
    lw = LedoitWolf().fit(Xtr[:, IA])
    S = lw.covariance_; Si = np.linalg.inv(S); m = Xtr[:, IA].mean(0)
    dM = np.sqrt(np.einsum("ij,jk,ik->i", Xte[:, IA] - m, Si, Xte[:, IA] - m))
    n = len(Xtr)
    lev = 1.0 / n + dM ** 2 / (n - 1)
    nn = np.array([np.min(np.linalg.norm(Xtr[:, IA] - q, axis=1)) for q in Xte[:, IA]])
    PB = Xtr[:, IB]
    uq = {tuple(np.round(q, 12)): None for q in Xte[:, IB]}
    for q in uq:
        uq[q] = in_hull(PB, np.array(q))
    hull = np.array([uq[tuple(np.round(q, 12))] for q in Xte[:, IB]])
    op, r = off_pattern(Xtr[:, IP], Xte[:, IP])
    opm, rm = off_pattern(Xtr[:, IM], Xte[:, IM])
    return dict(dM=dM, lev=lev, nn=nn, inside=hull, off=op, off_mix=opm, cond=float(np.linalg.cond(S)), shrinkage=float(lw.shrinkage_),
                r_pair=r, r_mix=rm)


def summary(d):
    return {k: dict(median=float(np.median(d[k])), max=float(np.max(d[k]))) for k in ("dM", "lev", "nn", "off", "off_mix")} | \
        dict(frac_outside=float(1 - d["inside"].mean()), cond=d["cond"], shrinkage=d["shrinkage"])


# ------------------------------------------------------------------ main
def main():
    log("#" * 100)
    log(f"paper2/revision/appdomain.py {time.strftime('%Y-%m-%d %H:%M:%S')}: applicability domain, DESCRIPTIVE (plan ec769ad); no training")
    log("#" * 100)
    log("DISCREPANCIES / choices (files win; G1-G4 as in the plan):")
    log("  G1 co-solvent descriptors are constant within a single-co-solvent fold (fold A, DEC): co-solvent off-pattern distance and hull")
    log("     membership are reported per fold; per-row analyses use the MIXTURE-LEVEL off-pattern distance (mix_eps, mix_lneta)")
    log("  G2 certify_foldA_0.0001_s0-4 archived late (archive 19, 179ac24); integrity = reproduce the logged ensemble RMSE 0.3345")
    log("  G3 early fold A nets = archived scbase_A_s0-4; polished vectors mapped with close.FlatQ.to_kan; logged RMSEs must reproduce")
    log("  G5 the committed scaler is a min-max map to [-1, 1] over the anchored train + anchored test rows (kan.Scaler), not a z-score:")
    log("     'standardised' means that map here; Mahalanobis distance and leverage are scale-invariant, the nearest-neighbour and off-")
    log("     pattern distances are in these units")
    S = SR.setup()
    tr = S["tr"]; X = S["B"]["x"].numpy().astype(float); use = S["use"]
    names = [f"scbase_A_s{s}.pt" for s in range(5)] + [f"close_foldA_1e-06_s{s}.pt" for s in range(5)] + [f"certify_foldA_0.0001_s{s}.pt" for s in range(5)]
    log("\n==================== inputs (SHA-256 verified against weights_manifest.txt, held in memory) ====================")
    blobs = in_memory(extract_verified(names))
    res = {}

    # ---------------- fold A ensembles (G3, G2)
    fa = S["folds"]["A"]
    bt, bh = subset(S["B"], fa["train"]), subset(S["B"], fa["test"])
    btd = SR.dbl(bt)
    used_f = np.unique(S["B"]["s"].numpy()[fa["train"]]).tolist()
    y = tr.log_k.values[fa["test"]]; anc = tr.anchor.values[fa["test"]]
    P = {"early": [], "1e-06": [], "0.0001": []}
    for s in range(5):
        net0 = KAN(len(DELTA_IN), [6, 1], len(S["salts"]))
        sd = torch.load(io.BytesIO(blobs[f"scbase_A_s{s}.pt"]), weights_only=False)
        sd = sd["early"] if isinstance(sd, dict) and "early" in sd else sd
        net0.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False); net0.eval()
        P["early"].append(predict(net0, delta_head, bh))
        for mu, nm in (("1e-06", f"close_foldA_1e-06_s{s}.pt"), ("0.0001", f"certify_foldA_0.0001_s{s}.pt")):
            flat = FlatQ(net0, btd, used_f, float(mu))
            x = torch.load(io.BytesIO(blobs[nm]), weights_only=False)
            x = x.numpy() if torch.is_tensor(x) else np.asarray(x)
            net = flat.to_kan(x, len(DELTA_IN), len(S["salts"]))
            P[mu].append(predict(net.float(), delta_head, bh))
    ens = {k: np.mean(v, 0) for k, v in P.items()}
    rm = {k: rmse(anc + ens[k], y) for k in ens}
    ok = all(abs(rm[k] - LOGGED[k]) < 1e-4 for k in rm)
    log("  fold A ensemble RMSE reconstructed: " + ", ".join(f"{k} {rm[k]:.4f} (logged {LOGGED[k]:.4f})" for k in rm) + f" -> {'REPRODUCED' if ok else 'MISMATCH'}")
    if not ok:
        log("STOP: the archived fold A ensembles do not reproduce the logged RMSEs (plan G2/G3)"); return
    err = {k: np.abs(anc + ens[k] - y) for k in ens}

    # ---------------- analysis 1: per fold
    log("\n==================== 1. per fold (held-out rows vs the fold's training rows) ====================")
    base = json.loads((REPO / "arch/sc_results.json").read_text())["accuracy"]["base"]["per"]
    folds = {}
    for f, m in S["folds"].items():
        d = distances(X[m["train"]], X[m["test"]])
        folds[f] = summary(d) | dict(n=int(m["test"].sum()), loco_rmse=base[f]["rmse"], n_distinct_B=int(len(np.unique(np.round(X[m["test"]][:, IB], 9), axis=0))))
        if f == "A":
            dA = d
        v = folds[f]
        log(f"  {f:20s} n {v['n']:4d} | LOCO RMSE {v['loco_rmse']:.4f} | d_M median {v['dM']['median']:.2f} max {v['dM']['max']:.2f} | leverage median "
            f"{v['lev']['median']:.4f} | NN median {v['nn']['median']:.3f} max {v['nn']['max']:.3f} | outside hull {v['frac_outside']:.2f} | off-pattern "
            f"median {v['off']['median']:.3f} max {v['off']['max']:.3f} | mixture off-pattern median {v['off_mix']['median']:.3f} | cov cond {v['cond']:.1f} "
            f"(LW shrinkage {v['shrinkage']:.3f}) | distinct space-B points {v['n_distinct_B']}")
    ge20 = [f for f in S["cos"] if folds[f]["n"] >= 20]
    sp = {}
    for q in ("dM", "lev", "nn", "off", "off_mix"):
        a = [folds[f][q]["median"] for f in ge20]; b = [folds[f]["loco_rmse"] for f in ge20]
        r_, p_ = spearmanr(a, b); sp[q] = dict(rho=float(r_), p=float(p_))
    a = [folds[f]["frac_outside"] for f in ge20]; r_, p_ = spearmanr(a, [folds[f]["loco_rmse"] for f in ge20]); sp["frac_outside"] = dict(rho=float(r_), p=float(p_))
    log(f"  Spearman(fold median distance, LOCO RMSE) over the {len(ge20)} LOCO folds with >= 20 test rows ({', '.join(ge20)}; fold A excluded as EA's duplicate):")
    for q, v in sp.items():
        log(f"    {q:12s} rho {v['rho']:+.3f}, p {v['p']:.3f} (n = {len(ge20)})")
    res["analysis1"] = dict(folds=folds, ge20=ge20, spearman=sp)

    # ---------------- analysis 2: fold A per row
    log("\n==================== 2. fold A per row (635 rows): absolute-error change early -> polished ====================")
    a2 = {}
    for mu in ("1e-06", "0.0001"):
        de = err[mu] - err["early"]
        row = dict(median_change=float(np.median(de)), mean_change=float(np.mean(de)))
        for q in ("dM", "lev", "nn", "off_mix"):
            r_, p_ = spearmanr(dA[q], de); row[q] = dict(rho=float(r_), p=float(p_))
        row["off_constant"] = bool(np.ptp(dA["off"]) < 1e-12); row["inside_constant"] = bool(dA["inside"].all() or (~dA["inside"]).all())
        med = np.median(dA["off_mix"])
        hi, lo = de[dA["off_mix"] > med], de[dA["off_mix"] <= med]
        row["mix_split"] = dict(above=float(np.median(hi)), below=float(np.median(lo)), n_above=int(len(hi)), n_below=int(len(lo)))
        a2[mu] = row
        log(f"  mu {mu}: median |error| change {row['median_change']:+.4f} (mean {row['mean_change']:+.4f}) | Spearman vs d_M {row['dM']['rho']:+.3f} "
            f"(p {row['dM']['p']:.1e}), leverage {row['lev']['rho']:+.3f} (p {row['lev']['p']:.1e}), NN {row['nn']['rho']:+.3f} (p {row['nn']['p']:.1e}), "
            f"mixture off-pattern {row['off_mix']['rho']:+.3f} (p {row['off_mix']['p']:.1e})")
        log(f"           co-solvent off-pattern constant over fold A: {row['off_constant']} (value {dA['off'][0]:.3f}); hull membership constant: "
            f"{row['inside_constant']} (inside: {bool(dA['inside'][0])}) -> per-row co-solvent analyses degenerate (G1); mixture off-pattern median split: "
            f"above {row['mix_split']['above']:+.4f} (n {row['mix_split']['n_above']}), below {row['mix_split']['below']:+.4f} (n {row['mix_split']['n_below']})")
    res["analysis2"] = a2

    # ---------------- analysis 3: DEC test rows
    log("\n==================== 3. DEC test rows (no labels) vs the full anchored training set ====================")
    _, te, _, _, Xte, _, _, _, has = AN.prepare(verbose=False)
    Xte[["eps_co", "lneta_co", "M_co"]] = co_props(te)
    Zte = S["sc"](Xte)[has].astype(float)
    dD = distances(X[use], Zte)
    dec = summary(dD) | dict(n=int(has.sum()), n_distinct_B=int(len(np.unique(np.round(Zte[:, IB], 9), axis=0))))
    log(f"  DEC anchored test rows n {dec['n']} | d_M median {dec['dM']['median']:.2f} max {dec['dM']['max']:.2f} | NN median {dec['nn']['median']:.3f} | "
        f"outside the training hull (space B) {dec['frac_outside']:.2f} | co-solvent off-pattern median {dec['off']['median']:.3f} (fold A {folds['A']['off']['median']:.3f}) | "
        f"mixture off-pattern median {dec['off_mix']['median']:.3f} (fold A {folds['A']['off_mix']['median']:.3f}) | distinct space-B points {dec['n_distinct_B']}")
    res["analysis3"] = dec

    # ---------------- analysis 4: correlation structure
    log("\n==================== 4. correlation structure of the 8 continuous inputs (2069 anchored rows) ====================")
    Z = X[use][:, IA]
    lw = LedoitWolf().fit(Z); C = lw.covariance_; sd_ = np.sqrt(np.diag(C)); R_lw = C / np.outer(sd_, sd_)
    Zc = Z - Z.mean(0); Craw = np.cov(Zc.T); ev, V = np.linalg.eigh(Craw); ev, V = ev[::-1], V[:, ::-1]
    k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.99) + 1)
    Cp = V[:, :k] @ np.diag(ev[:k]) @ V[:, :k].T; Cp += np.diag(np.diag(Craw) - np.diag(Cp)); sp_ = np.sqrt(np.diag(Cp)); R_pca = Cp / np.outer(sp_, sp_)
    def partial(R):
        Pm = np.linalg.inv(R); d = np.sqrt(np.diag(Pm)); return -Pm / np.outer(d, d)
    evl = np.linalg.eigvalsh(R_lw)[::-1]
    pairs = []
    for i in range(8):
        for j in range(i + 1, 8):
            pairs.append(dict(a=DELTA_IN[i], b=DELTA_IN[j], r_lw=float(R_lw[i, j]), r_pca=float(R_pca[i, j]),
                              partial_lw=float(partial(R_lw)[i, j]), partial_pca=float(partial(R_pca)[i, j])))
    pairs.sort(key=lambda d: -abs(d["r_lw"]))
    log(f"  Ledoit-Wolf shrinkage {lw.shrinkage_:.4f}; correlation eigenvalues {', '.join(f'{v:.3f}' for v in evl)} (condition {evl[0] / evl[-1]:.1f})")
    log(f"  PCA-shrunk sensitivity: {k} of 8 components explain >= 99% of the variance (cumulative {np.cumsum(ev)[k - 1] / ev.sum():.4f})")
    log("  strongest input pairs (sorted by |r|, Ledoit-Wolf):   pair                     r (LW)   r (PCA)  partial (LW)  partial (PCA)")
    for d in pairs[:12]:
        log(f"    {d['a'] + ' / ' + d['b']:26s} {d['r_lw']:+.3f}   {d['r_pca']:+.3f}   {d['partial_lw']:+.3f}        {d['partial_pca']:+.3f}")
    log("  co-solvent block numerical rank: 14 of 27 (committed, rank_prereg.md step 0; cited, not recomputed)")
    log("  dose thresholds beside these correlations are SYNTHETIC ([4, 3, 1], Gaussian inputs): P0 rho* 0.5, P1 rho* 0.9, both fail to")
    log("  certify from rho 0.95 (dose study); they are not real-data thresholds")
    res["analysis4"] = dict(R_lw=R_lw.tolist(), R_pca=R_pca.tolist(), eig_lw=evl.tolist(), k99=k, pairs=pairs, shrinkage=float(lw.shrinkage_))

    OUT.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    figures(res, folds, ge20, dA, err, X, use, Zte, fa)
    log("\nwrote appdomain_results.json and figs/appdomain_*.png")


def figures(res, folds, ge20, dA, err, X, use, Zte, fa):
    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for f in folds:
        if f == "A":
            continue
        v = folds[f]; c = "#0072B2" if f in ge20 else "#999999"
        ax.scatter(v["off"]["median"], v["loco_rmse"], s=12 + v["n"] / 10, color=c)
        ax.annotate(f, (v["off"]["median"], v["loco_rmse"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("median off-pattern distance (eps_co, ln eta_co), committed scaling"); ax.set_ylabel("LOCO RMSE (early-stopped baseline ensemble)")
    ax.set_title("(a) DESCRIPTIVE: per-fold LOCO RMSE vs off-pattern distance (blue: >= 20 test rows)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "appdomain_a_fold_rmse_vs_offpattern.png", dpi=120); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, mu in zip(axs, ("1e-06", "0.0001")):
        ax.scatter(dA["off_mix"], err[mu] - err["early"], s=6, alpha=0.5)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("mixture-level off-pattern distance (mix_eps, mix_lneta)"); ax.set_ylabel("|error| change, early -> polished")
        ax.set_title(f"(b) DESCRIPTIVE: fold A, mu = {mu}", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "appdomain_b_foldA_change_vs_offpattern.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    T = X[use]
    ax.scatter(T[:, IP[0]], T[:, IP[1]], s=8, color="#999999", label="training (anchored)")
    A = X[fa["test"]]
    ax.scatter(A[:, IP[0]], A[:, IP[1]], s=40, marker="s", facecolors="none", edgecolors="#D55E00", label="fold A (EA)")
    ax.scatter(Zte[:, IP[0]], Zte[:, IP[1]], s=60, marker="*", color="#0072B2", label="DEC test")
    ax.set_xlabel("eps_co (committed scaling)"); ax.set_ylabel("ln eta_co (committed scaling)"); ax.legend(fontsize=7)
    ax.set_title("(c) DESCRIPTIVE: space B (eps_co, ln eta_co)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "appdomain_c_spaceB.png", dpi=120); plt.close(fig)
    R = np.array(res["analysis4"]["R_lw"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(R, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(8)); ax.set_xticklabels(DELTA_IN, rotation=45, ha="right", fontsize=7); ax.set_yticks(range(8)); ax.set_yticklabels(DELTA_IN, fontsize=7)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im); ax.set_title("(d) DESCRIPTIVE: input correlations (Ledoit-Wolf), 2069 anchored rows", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "appdomain_d_corr_heatmap.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()

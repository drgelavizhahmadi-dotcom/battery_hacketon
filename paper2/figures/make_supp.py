"""Supplementary figures: loss decomposition (lossdecomp), gauge-direction curvature shares (gaugecheck), LOCO RMSE per fold.
Run from the repo root: python3 paper2/figures/make_supp.py"""
import numpy as np
import matplotlib.pyplot as plt
from style import C, FULL, HALF, OI, Sidecar, letter

# ---------------- S1: loss decomposition by term
sc = Sidecar("figS1_lossdecomp")
ld = sc.json("paper2/lossdecomp_results.json")["rows"]
sc.note("POST-HOC (lossdecomp, 4ec6fce): change in each loss term from early to polished, summed over the 8 seeds of a cell and divided by the summed "
        "change in total loss (pooled shares; sum to 1 per cell). Only early vs polished states were saved.")
TERMS = [("data", "data (MSE)", OI["blue"]), ("D2", "D2 smoothness", OI["orange"]), ("SiLU", "SiLU term", OI["green"]), ("muQ", r"$\mu Q$", OI["purple"])]
cells = [(c, l) for c in ("independent", "correlated") for l in ("P0", "P1", "P2")]
pooled = {}
for c, l in cells:
    rr = [ld[f"{c}_{l}_s{s}"] for s in range(8)]
    dt = sum(r["polished"]["total"] - r["early"]["total"] for r in rr)
    pooled[f"{c} {l}"] = {k: sum(r["polished"][k] - r["early"][k] for r in rr) / dt for k, _, _ in TERMS}
    pooled[f"{c} {l}"]["median_total_change_rel"] = float(np.median([(r["polished"]["total"] - r["early"]["total"]) / r["early"]["total"] for r in rr]))
fig, ax = plt.subplots(figsize=(FULL, 2.2))
x = np.arange(len(cells))
pos, neg = np.zeros(len(cells)), np.zeros(len(cells))
for k, lab, col in TERMS:
    v = np.array([pooled[f"{c} {l}"][k] for c, l in cells])
    ax.bar(x, np.where(v > 0, v, 0), 0.6, bottom=pos, color=col, label=lab); pos += np.where(v > 0, v, 0)
    ax.bar(x, np.where(v < 0, v, 0), 0.6, bottom=neg, color=col); neg += np.where(v < 0, v, 0)
ax.axhline(0, color=OI["black"], lw=0.6)
ax.set_xticks(x); ax.set_xticklabels([f"{c}\n{l}" for c, l in cells], fontsize=7)
ax.set_ylabel("share of the total-loss decrease")
for j, (c, l) in enumerate(cells):
    ax.text(j, pos[j] + 0.03, f"{pooled[f'{c} {l}']['median_total_change_rel']:+.1%}", ha="center", fontsize=6)
ax.set_ylim(min(neg.min() - 0.05, -0.15), 1.18)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=7)
sc.put("pooled_shares", pooled)
sc.note("numbers above the bars: median relative change of the total loss, early -> polished")
fig.tight_layout(); sc.save(fig)

# ---------------- S2: gauge-direction curvature shares
sc = Sidecar("figS2_gaugecheck")
gc = sc.json("gaugecheck_results.json")["A"]
sc.note("POST-HOC (gaugecheck, 0b29351): at the mu = 1e-4 certified minima, share of the curvature d'Hd along each least-squares gauge direction that comes "
        "from the mu*Q term (curv = [full, without Q, mu*Q]); families: const_L1 (48 per model), const_L2 (6), rescale (6)")
fams = ["const_L1", "const_L2", "rescale"]
fig, ax = plt.subplots(figsize=(HALF, 2.2))
vals = {}
for j, f in enumerate(fams):
    for kind, off, col in [("ind", -0.17, C["baseline"]), ("stu", 0.17, OI["sky"])]:
        sh = [d["curv"][2] / d["curv"][0] for m, v in gc.items() if m.startswith(kind) for d in v["fams"][f]]
        vals[f"{f} {kind}"] = dict(median=float(np.median(sh)), q1=float(np.percentile(sh, 25)), q3=float(np.percentile(sh, 75)), n=len(sh))
        ax.scatter(j + off + np.random.default_rng(j * 7 + len(kind)).uniform(-0.08, 0.08, len(sh)), sh, s=3, color=col, lw=0, alpha=0.6,
                   label=("seeds" if kind == "ind" else "students") if j == 0 else None)
        ax.plot([j + off - 0.1, j + off + 0.1], [np.median(sh)] * 2, color=OI["black"], lw=1)
ax.set_xticks(range(3)); ax.set_xticklabels(["constant\nlayer 1", "constant\nlayer 2", "unit\nrescale"], fontsize=7)
ax.set_ylabel(r"$\mu Q$ share of curvature"); ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=6, loc="upper left", markerscale=2)
sc.put("muQ_share", vals)
fig.tight_layout(); sc.save(fig)

# ---------------- S3: LOCO RMSE per fold
sc = Sidecar("figS3_loco")
sr = sc.json("arch/sc_results.json")["accuracy"]
er = sc.json("arch/ex_results.json")["accuracy"]
sc.note("5-seed ensembles (seeds 0-4), early state, RMSE on delta (= log k); baseline = KAN-delta (final.py protocol), identical in arch/sc and arch/ex "
        "(same cached models). The EA fold = fold A rows.")
folds = sorted([f for f in sr["sc"]["per"] if f != "A"], key=lambda f: -sr["sc"]["per"][f]["n"])
chk = max(abs(sr["base"]["per"][f]["rmse"] - er["base"]["per"][f]["rmse"]) for f in folds)
sc.put("baseline_max_abs_diff_sc_vs_ex_runs", chk)
fig, ax = plt.subplots(figsize=(FULL, 2.3))
x = np.arange(len(folds)); w = 0.26
series = [("KAN-δ baseline", sr["base"]["per"], C["baseline"]), ("SC-KAN-δ", sr["sc"]["per"], C["sc"]), ("EX-KAN", er["ex"]["per"], C["ex"])]
for k, (lab, per, col) in enumerate(series):
    ax.bar(x + (k - 1) * w, [per[f]["rmse"] for f in folds], w, color=col, label=lab)
ax.set_xticks(x); ax.set_xticklabels([f.replace(" ", "\n") + f"\n(n = {sr['sc']['per'][f]['n']})" for f in folds], fontsize=6)
ax.set_ylabel("held-out RMSE (log k)"); ax.legend(frameon=False, ncol=3, fontsize=7, loc="upper left")
sc.put("per_fold", {f: {lab: per[f]["rmse"] for lab, per, _ in series} | {"n": sr["sc"]["per"][f]["n"]} for f in folds})
sc.put("means", {lab: dict(unweighted=src["loco_unweighted"], row_weighted=src["loco_rowweighted"], ge20=src["loco_ge20"], foldA=src["foldA"])
                 for lab, src in [("KAN-δ baseline", sr["base"]), ("SC-KAN-δ", sr["sc"]), ("EX-KAN", er["ex"])]})
fig.tight_layout(); sc.save(fig)

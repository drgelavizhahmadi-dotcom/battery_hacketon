"""Figure 8: real-data dose-response in mu (null-space penalty). Run from the repo root."""
import numpy as np
import matplotlib.pyplot as plt
from style import C, FULL, OI, Sidecar, letter

sc = Sidecar("fig8_dose_response")
st = sc.json("stage2_results.json")
cl = sc.json("close_results.json")
ce = sc.json("certify_results.json")
SEEDS = range(6)
MUS = ["0", "1e-06", "1e-05", "0.0001"]
LAB = ["0\n(Part A)", r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$"]
sc.note("seeds = the 6 real-data KAN-delta seeds (ind 0-5); students are not plotted")
sc.note("mu = 0 is stage2 Part A (trust-exact from fiber2 capped, no extra term): the runaway reference")
sc.note("numerical minimum = recomputed max|grad| < 1e-7 (certify.py Step 1 candidates); certified = certify.py four-part criterion")
sc.note("norm growth = ||theta||^2 at the end of polishing / ||theta||^2 at the 107-epoch AdamW state; mu = 0 from stage2 partB block norms (sum of squares)")
sc.note("data term relative to the runaway = training MSE at the end of polishing / close_results ref_A (data term at the stage2 Part A final)")

# a) numerical minima and certified
cand = {(m, t) for m, t in ce["candidates"]}
nmin = {"0": sum(st["infos"][f"A|ind {i}"]["grad"] < 1e-7 for i in SEEDS)}
ncert = {"0": 0}
for m in MUS[1:]:
    nmin[m] = sum((m, f"ind_{i}") in cand for i in SEEDS)
    ncert[m] = sum(ce["cert"].get(f"{m}|ind_{i}", {}).get("certified", False) for i in SEEDS)
# b) growth
sq = lambda d: sum(v ** 2 for v in d.values())   # partB stores block NORMS; ||theta||^2 = sum of squares (107-epoch: 19.76 = close theta2_start)
growth = {"0": [sq(st["partB"][f"ind {i}"]["Part A final"]) / sq(st["partB"][f"ind {i}"]["AdamW-107"]) for i in SEEDS]}
for m in MUS[1:]:
    growth[m] = [cl["runs"][f"{m}|ind {i}"]["theta2"] / cl["runs"][f"{m}|ind {i}"]["theta2_start"] for i in SEEDS]
# c) data term relative to runaway
rel = {"0": [1.0] * 6}
for m in MUS[1:]:
    rel[m] = [cl["runs"][f"{m}|ind {i}"]["data"] / cl["ref_A"][str(i)] for i in SEEDS]
# d) fold A RMSE
early = 0.142
ens = {"1e-06": ce["ensemble"]["1e-06"][0], "0.0001": ce["ensemble"]["0.0001"][0]}
per = {m: [s["rmse"] for s in ce["step3"] if s["mu"] == m] for m in ("1e-06", "0.0001")}

x = np.arange(4)
fig, axs = plt.subplots(2, 2, figsize=(FULL, 3.7)); axs = axs.ravel()
ax = axs[0]
ax.bar(x - 0.18, [nmin[m] for m in MUS], 0.36, color=OI["sky"], label="numerical minimum")
ax.bar(x + 0.18, [ncert[m] for m in MUS], 0.36, color=C["certified"], label="certified")
ax.set_ylim(0, 7.5); ax.set_ylabel("seeds (of 6)"); ax.legend(frameon=False, loc="upper left", fontsize=6.5, ncol=2)
letter(ax, "a", dx=-0.13)
ax = axs[1]
for j, m in enumerate(MUS):
    v = np.asarray(growth[m])
    ax.errorbar(j, np.median(v), yerr=[[np.median(v) - v.min()], [v.max() - np.median(v)]], fmt="o", color=C["runaway"] if j == 0 else C["certify"], ms=3.5, capsize=2, lw=0.8)
ax.set_yscale("log"); ax.set_ylabel(r"$\|\theta\|^2$ / 107-epoch")
letter(ax, "b", dx=-0.13)
ax = axs[2]
for j, m in enumerate(MUS):
    v = np.asarray(rel[m])
    ax.errorbar(j, np.median(v), yerr=[[np.median(v) - v.min()], [v.max() - np.median(v)]], fmt="o", color=C["runaway"] if j == 0 else C["certify"], ms=3.5, capsize=2, lw=0.8)
ax.axhline(1, color=OI["grey"], lw=0.6, ls=":")
ax.set_ylabel("data term / runaway")
letter(ax, "c", dx=-0.13)
ax = axs[3]
ax.axhline(early, color=OI["black"], lw=0.8, ls="--", label="early-stopped (0.142)")
for m, j in [("1e-06", 1), ("0.0001", 3)]:
    ax.scatter(np.full(len(per[m]), j) + np.linspace(-0.12, 0.12, len(per[m])), per[m], s=8, color=OI["grey"], label="single seed" if j == 1 else None, zorder=2)
    ax.scatter(j, ens[m], s=22, color=C["certify"], marker="D", label="5-seed ensemble" if j == 1 else None, zorder=3)
ax.set_yscale("log"); ax.set_ylim(0.1, 30); ax.set_ylabel("fold A RMSE (log k)"); ax.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=1)
letter(ax, "d", dx=-0.13)
for ax in axs:
    ax.set_xticks(x); ax.set_xticklabels(LAB, fontsize=7); ax.set_xlabel(r"$\mu$")
fig.tight_layout(h_pad=1.0, w_pad=1.5)
sc.put("a_numerical_minimum", nmin); sc.put("a_certified", ncert)
sc.put("b_growth", {m: dict(values=growth[m], median=float(np.median(growth[m])), min=float(min(growth[m])), max=float(max(growth[m]))) for m in MUS})
sc.put("c_data_rel_runaway", {m: dict(values=rel[m], median=float(np.median(rel[m])), min=float(min(rel[m])), max=float(max(rel[m]))) for m in MUS})
sc.put("d_foldA", dict(early_stopped=early, ensemble=ens, per_seed=per))
sc.note("d: polished fold A RMSE at mu 1e-6 (close.py fold A run) and 1e-4 (certify.py Step 3), 5 seeds and their ensemble; the early-stopped value is final.py's 0.142 (kan_output.txt, 3081cc0)")
sc.save(fig)

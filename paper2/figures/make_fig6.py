"""Figure 6: correlation dose-response (synthetic [4, 3, 1], Gaussian inputs). a) share of models not certified, b) median short-path D,
c) largest hidden-unit correlation, against rho on a log 1/(1 - rho) axis; P0 and P1 at every level, P2 at rho = 0.99.
Run from the repo root: /usr/bin/python3 paper2/figures/make_fig6.py"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from style import C, FULL, Sidecar, letter

sc = Sidecar("fig6_correlation_dose")
res = sc.json("paper2/revision/dose_results.json")
sc.note("cell medians over 8 seeds as committed (dose_results.json cells: uncert_frac, D_med, hidden_corr_med; 1a18569 -> 541355a); "
        "D = ||theta*(1e-7)||^2 / ||theta*(1e-5)||^2; thresholds are synthetic, not real-data thresholds")
RHO = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
xs = lambda r: 1.0 / (1.0 - np.asarray(r))
fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.2))
q = [("uncert_frac", "share not certified"), ("D_med", "median $D$"), ("hidden_corr_med", "largest hidden-unit corr.")]
for lev in ("P0", "P1", "P2"):
    keys = [k for k in res["cells"] if k.startswith(lev + "_")]
    rr = [res["cells"][k]["rho"] for k in keys]
    vals = {f: [res["cells"][k][f] for k in keys] for f, _ in q}
    for ax, (f, lab) in zip(axs, q):
        if lev == "P2":
            ax.scatter(xs(rr), vals[f], s=28, marker="D", color=C["P2"], zorder=4)
        else:
            ax.plot(xs(rr), vals[f], "-o", ms=3, color=C[lev])
    sc.put(lev, dict(rho=rr, **vals, certified={k: res["cells"][k]["certified"] for k in keys}))
for ax, (f, lab), s in zip(axs, q, "abc"):
    ax.set_xscale("log")
    ax.set_xticks(xs(RHO)); ax.set_xticklabels([f"{r:g}" for r in RHO]); ax.minorticks_off()
    ax.set_xlabel(r"input correlation $\rho$")
    ax.set_ylabel(lab)
    letter(ax, s, dx=-0.25)
axs[0].axhline(0.5, color=C["early"], lw=0.5, ls="--")
axs[0].set_ylim(-0.04, 1.04)
axs[1].set_yscale("log"); axs[1].axhline(1.5, color=C["early"], lw=0.5, ls="--")
axs[2].set_ylim(0, 1.0)
axs[0].legend(handles=[Line2D([], [], color=C["P0"], marker="o", ms=3, label="P0"), Line2D([], [], color=C["P1"], marker="o", ms=3, label="P1"),
                       Line2D([], [], color=C["P2"], marker="D", ms=4, lw=0, label=r"P2 ($\rho = 0.99$)")], frameon=False, loc="lower right")
fig.tight_layout(w_pad=1.0)
sc.save(fig)

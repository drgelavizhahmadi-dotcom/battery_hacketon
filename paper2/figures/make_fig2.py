"""Figure 2: synthetic 2 x 3, ||theta||^2 over trust-exact polishing iterations. Run from the repo root."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from style import C, FULL, Sidecar, letter

sc = Sidecar("fig2_synthetic")
res = sc.json("paper2/synth_results.json")
sc.note("trust_theta2 = ||theta||^2 at every trust-exact iteration (iteration 0 = early-stopped AdamW state), as saved by synth.py")
conds, levs = ["independent", "correlated"], ["P0", "P1", "P2"]
fig, axs = plt.subplots(2, 3, figsize=(FULL, 3.6), sharex=True, sharey=True)
for r, cond in enumerate(conds):
    for c, lev in enumerate(levs):
        ax = axs[r, c]
        growth, final, iters, cert = [], [], [], []
        for s in range(8):
            m = res["models"][f"{cond}_{lev}_s{s}"]
            th = np.asarray(m["trust_theta2"])
            ax.plot(np.arange(len(th)), th, color=C["certified"] if m["certified"] else C["runaway"], lw=0.8, alpha=0.9)
            growth.append(th[-1] / th[0]); final.append(th[-1]); iters.append(len(th) - 1); cert.append(bool(m["certified"]))
        ax.set_yscale("log")
        if r == 1:
            ax.set_xlabel("polishing iteration")
        if c == 0:
            ax.set_ylabel(f"{cond}\n" + r"$\|\theta\|^2$")
        if r == 0:
            ax.text(0.5, 1.02, lev, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
        ax.text(0.97, 0.05, f"{sum(cert)}/8 cert.\nmedian ×{np.median(growth):.1f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)
        letter(ax, "abcdef"[3 * r + c], dx=-0.12)
        sc.put(f"{cond}_{lev}", dict(median_growth=float(np.median(growth)), growth=growth, median_final_theta2=float(np.median(final)),
                                     median_initial_theta2=float(np.median([res["models"][f"{cond}_{lev}_s{s}"]["trust_theta2"][0] for s in range(8)])),
                                     median_iters=float(np.median(iters)), certified=sum(cert)))
axs[0, 2].legend(handles=[Line2D([], [], color=C["certified"], label="certified"), Line2D([], [], color=C["runaway"], label="not certified")],
                 frameon=False, loc="upper left")
fig.tight_layout(h_pad=0.8, w_pad=0.8)
sc.save(fig)

"""Figure 5: certified L2 regularisation paths (infimum test). ||theta*||^2 and L* against gamma (log-log), one line per seed:
synthetic correlated P0/P1/P2 (Phase 1) and real data, blind spot open (Phase 2) vs closed (control arm, mu = 1e-4).
Run from the repo root: /usr/bin/python3 paper2/figures/make_fig5.py"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from style import C, FULL, Sidecar, letter

sc = Sidecar("fig5_paths")
res = sc.json("paper2/revision/infimum_results.json")
sc.note("path points with status 'reached'; filled markers = included in the verdict fits (certified, or degenerate and accepted after the probes), "
        "open markers = reached but not included; points not reached (path stopped by the budget) are absent")
sc.note("synthetic: Phase 1, correlated inputs, 8 seeds per penalty level (ef3b3c2); real: Phase 2, the 6 KAN-delta seeds with the blind spot open "
        "(349022b / 4003b21), and the POST-HOC control arm with the null-space penalty at mu = 1e-4 fixed, seeds ind0 and ind1 (4003b21)")
sc.note("L* = the loss without the gamma term (for the closed real paths it includes mu*Q, the loss that path minimises)")


def pts(path):
    r = [p for p in path if p.get("status") == "reached"]
    return (np.array([p["coef"] for p in r]), np.array([p["theta2"] for p in r]), np.array([p["Lstar"] for p in r]),
            np.array([bool(p.get("included")) for p in r]))


def draw(ax_t, ax_l, path, col, lw=0.8):
    g, t, l, inc = pts(path)
    for ax, v in ((ax_t, t), (ax_l, l)):
        ax.plot(g, v, color=col, lw=lw, alpha=0.85)
        ax.scatter(g[inc], v[inc], s=6, color=col, zorder=3, lw=0)
        ax.scatter(g[~inc], v[~inc], s=10, facecolors="white", edgecolors=col, zorder=3, lw=0.7)
    return g, t, l, inc


fig, axs = plt.subplots(2, 2, figsize=(FULL, 4.0), sharex=True)
syn = res["phase1"]["models"]
for lev in ("P0", "P1", "P2"):
    rec = []
    for s in range(8):
        g, t, l, inc = draw(axs[0, 0], axs[1, 0], syn[f"{lev}-s{s}"]["path"], C[lev])
        rec.append(dict(seed=s, gamma=g, theta2=t, Lstar=l, included=inc))
    sc.put(f"synthetic_{lev}", dict(alpha=[res["phase1"]["rows"][f"{lev}-s{s}"]["fit"] for s in range(8)],
                                    ratio=[res["phase1"]["rows"][f"{lev}-s{s}"]["ratio"] for s in range(8)], paths=rec))
real, ctrl = res["phase2"]["models"], res["control_posthoc"]["models"]
rec = []
for k in sorted(real):
    g, t, l, inc = draw(axs[0, 1], axs[1, 1], real[k]["path"], C["runaway"])
    rec.append(dict(model=k, gamma=g, theta2=t, Lstar=l, included=inc))
sc.put("real_open", dict(alpha={k: res["phase2"]["rows"][k]["fit"] for k in sorted(real)}, paths=rec))
rec = []
for k in sorted(ctrl):
    g, t, l, inc = draw(axs[0, 1], axs[1, 1], ctrl[k]["path"], C["P2"], lw=1.2)
    rec.append(dict(model=k, gamma=g, theta2=t, Lstar=l, included=inc))
sc.put("real_closed_mu1e-4", dict(rows={k: {q: res["control_posthoc"]["rows"][k][q] for q in ("ratio", "fit") if q in res["control_posthoc"]["rows"][k]}
                                        for k in sorted(ctrl)}, paths=rec))

for ax in axs.flat:
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(2e-2, 5e-8)                        # gamma decreases left to right, the direction of the path
for ax in axs[1]:
    ax.set_xlabel(r"regularisation weight $\gamma$")
axs[0, 0].set_ylabel(r"$\|\theta^*(\gamma)\|^2$"); axs[1, 0].set_ylabel(r"$L^*(\gamma)$ (loss without $\gamma$ term)")
axs[0, 0].text(0.5, 1.03, "synthetic, correlated inputs", transform=axs[0, 0].transAxes, ha="center", fontsize=8)
axs[0, 1].text(0.5, 1.03, "real data (KAN-δ)", transform=axs[0, 1].transAxes, ha="center", fontsize=8)
axs[0, 0].legend(handles=[Line2D([], [], color=C[l], label=l) for l in ("P0", "P1", "P2")], frameon=False, loc="upper left")
axs[0, 1].legend(handles=[Line2D([], [], color=C["runaway"], label="blind spot open (6 seeds)"),
                          Line2D([], [], color=C["P2"], lw=1.2, label=r"closed, $\mu = 10^{-4}$ (ind0, ind1)"),
                          Line2D([], [], color="k", marker="o", ms=3, lw=0, markerfacecolor="white", label="reached, not included")],
                 frameon=False, loc="upper left")
for ax, s in zip(axs.flat, "abcd"):
    letter(ax, s, dx=-0.17)
fig.tight_layout(h_pad=0.6, w_pad=1.2)
sc.save(fig)

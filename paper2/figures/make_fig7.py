"""Figure 7: B-spline replication. a) certified models per cell, Gaussian-RBF study vs B-spline replication; b, c) P2 regularisation
paths extended to gamma = 1e-9 (post-hoc check 1), norm relative to each seed's certified polish minimum.
Run from the repo root: /usr/bin/python3 paper2/figures/make_fig7.py"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from style import C, FULL, OI, Sidecar, letter

sc = Sidecar("fig7_bspline")
rbf = sc.json("paper2/synth_results.json")["cells"]
bs = sc.json("paper2/revision/bspline_results.json")["step1"]
ph = sc.json("paper2/revision/bspline_posthoc_results.json")["check1"]
sc.note("a: certified of 8 per cell; RBF from the synthetic study (dfd281d -> 34cc321), B-spline from the replication Step 1 (5491481 -> ffa6e64 / f0accc0), "
        "frozen mu: RBF 1e-5, B-spline 1e-6")
sc.note("b, c: P2 at the frozen mu = 1e-6; gamma = 1e-3, 1e-5, 1e-7 from Step 1 (bspline_results.json), 1e-8 and 1e-9 from POST-HOC check 1 "
        "(bspline_posthoc_results.json; plan ff5bd20 -> results 9167740); y = ||theta*(gamma)|| / ||theta_polish||, the norm ratio of check 1 "
        "(prediction: correlated median within [0.9, 1.1] at 1e-9)")
cells = [("independent", "P0"), ("independent", "P1"), ("independent", "P2"), ("correlated", "P0"), ("correlated", "P1"), ("correlated", "P2")]
fig = plt.figure(figsize=(FULL, 2.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1])
ax = fig.add_subplot(gs[0])
x = np.arange(len(cells)); w = 0.38
vr = [rbf[f"{c}|{l}"]["certified"] for c, l in cells]
vb = [bs["cells"][f"{c}_{l}"]["certified"] for c, l in cells]
ax.bar(x - w / 2, vr, w, color=OI["grey"], label="Gaussian RBF")
ax.bar(x + w / 2, vb, w, color=OI["blue"], label="cubic B-spline")
for i, (a, b) in enumerate(zip(vr, vb)):
    ax.text(i - w / 2, a + 0.15, str(a), ha="center", fontsize=6); ax.text(i + w / 2, b + 0.15, str(b), ha="center", fontsize=6)
ax.set_xticks(x); ax.set_xticklabels([f"{l}\n{'ind.' if c == 'independent' else 'corr.'}" for c, l in cells], fontsize=6.5)
ax.set_ylim(0, 9.5); ax.set_yticks([0, 2, 4, 6, 8]); ax.set_ylabel("certified (of 8)")
ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=6, bbox_to_anchor=(0.5, 1.12))
letter(ax, "a", dx=-0.2, dy=1.06)
sc.put("a_certified", dict(rbf={f"{c}_{l}": v for (c, l), v in zip(cells, vr)}, bspline={f"{c}_{l}": v for (c, l), v in zip(cells, vb)}))
G = [1e-3, 1e-5, 1e-7, 1e-8, 1e-9]
for j, cond in enumerate(("independent", "correlated")):
    a = fig.add_subplot(gs[j + 1], sharey=None)
    rows = {}
    for s in range(8):
        k = f"{cond}_P2_s{s}"
        m = bs["models"][k]; r = ph["rows"][k]
        t2 = [m["path"][0]["theta2"], m["path"][1]["theta2"], m["path"][2]["theta2"], r["theta2"][2], r["theta2"][3]]
        assert abs(t2[1] - r["theta2"][0]) < 1e-9 * t2[1] and abs(t2[2] - r["theta2"][1]) < 1e-9 * t2[2]
        y = np.sqrt(np.array(t2) / m["polish"]["theta2"])
        assert abs(y[-1] - r["norm_ratio"]) < 1e-6, (k, y[-1], r["norm_ratio"])
        a.plot(G, y, "-o", ms=2.5, lw=0.8, color=C["P2"], alpha=0.85)
        rows[k] = dict(gamma=G, norm_ratio=y.tolist(), theta2_polish=m["polish"]["theta2"])
    a.axhline(1.0, color="k", lw=0.5, ls="--")
    a.axhspan(0.9, 1.1, color=OI["grey"], alpha=0.15, lw=0)
    a.set_xscale("log"); a.set_xlim(2e-3, 5e-10)
    a.set_xlabel(r"$\gamma$"); a.set_ylim(0, 1.25)
    a.text(0.5, 1.02, f"P2, {cond} inputs", transform=a.transAxes, ha="center", fontsize=7.5)
    if j == 0:
        a.set_ylabel(r"$\|\theta^*(\gamma)\| \,/\, \|\theta_{\mathrm{polish}}\|$")
    letter(a, "bc"[j], dx=-0.22, dy=1.06)
    sc.put(f"{'bc'[j]}_{cond}", dict(rows=rows, median_norm_ratio_1e9=ph["cells"][cond]["norm_ratio_med"], D_prime_med=ph["cells"][cond]["D_prime_med"]))
fig.tight_layout(w_pad=1.0)
sc.save(fig)

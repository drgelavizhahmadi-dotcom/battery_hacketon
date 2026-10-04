"""Figure 1: mechanism (computed illustration). Run from the repo root: python3 paper2/figures/make_fig1.py"""
import numpy as np
import matplotlib.pyplot as plt
from style import C, FULL, OI, Sidecar, letter

from kan import N_RBF, SMOOTH, KANLayer
import synth as SY

sc = Sidecar("fig1_mechanism")
sc.note("computed illustration: final.py edge type (kan.KANLayer: 8 Gaussian RBFs, centres linspace(-1,1,8), width 2/7, range [-1,1])")
L = KANLayer(1, 1)
g, h = L.grid.numpy(), L.h
k = np.arange(N_RBF, dtype=float)
p = (k - k.mean()) / np.linalg.norm(k - k.mean())               # unit linear coefficient pattern
basis = lambda x: np.exp(-(((np.asarray(x)[..., None] - g) / h) ** 2))
phi = lambda x: basis(x) @ p
NB = np.linalg.qr(np.column_stack([np.ones(N_RBF), k]))[0]
D2 = np.zeros((N_RBF - 2, N_RBF))
for i in range(N_RBF - 2):
    D2[i, i:i + 3] = [1, -2, 1]
lam, mu, ksilu = SMOOTH, 1e-5, SY.K_SILU

fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.0))
# a) edge of the linear pattern + best straight line, central 80% shaded
x = np.linspace(-1, 1, 2001); f = phi(x)
A = np.column_stack([np.ones_like(x), x]); cf, *_ = np.linalg.lstsq(A, f, rcond=None); fit = A @ cf
m80 = np.abs(x) <= 0.8
A8 = np.column_stack([np.ones(m80.sum()), x[m80]]); cf8, *_ = np.linalg.lstsq(A8, f[m80], rcond=None)
rng_ = f.max() - f.min(); rng8 = f[m80].max() - f[m80].min()
dev = dict(full_max=float(np.abs(f - fit).max() / rng_), full_rms=float(np.sqrt(np.mean((f - fit) ** 2)) / rng_),
           c80_max=float(np.abs(f[m80] - A8 @ cf8).max() / rng8), c80_rms=float(np.sqrt(np.mean((f[m80] - A8 @ cf8) ** 2)) / rng8))
ax = axs[0]
ax.axvspan(-0.8, 0.8, color=OI["grey"], alpha=0.15, lw=0)
ax.plot(x, f, color=OI["black"], label="edge, linear pattern")
ax.plot(x, fit, color=OI["vermillion"], ls="--", label="best line")
ax.set_xlabel("input x"); ax.set_ylabel(r"$\sum_k c_k B_k(x)$"); ax.legend(frameon=False, loc="lower right", handlelength=1.5)
letter(ax, "a")
sc.put("a_deviation_relative_to_range", dev); sc.put("a_line_fit_full", cf.tolist()); sc.put("a_central80_shaded", [-0.8, 0.8])

# b) tilt +t on edge 1, -t on edge 2 (two inputs of one unit); penalties vs t
t = np.linspace(-3, 3, 121)
pen = {"D2 (P0)": [], "P1 = D2 + SiLU": [], "P2 = P1 + mu Q": []}
for ti in t:
    c = np.vstack([ti * p, -ti * p])                            # tilt only (zero base coefficients, SiLU weights unchanged)
    d2 = lam * float(((c @ D2.T) ** 2).sum()); silu = 0.0
    q = float(((c @ NB) ** 2).sum())
    pen["D2 (P0)"].append(d2); pen["P1 = D2 + SiLU"].append(d2 + lam * ksilu * silu); pen["P2 = P1 + mu Q"].append(d2 + mu * q)
ax = axs[1]
ax.plot(t, pen["P2 = P1 + mu Q"], color=C["P2"], label=r"P2: $+\mu Q$")
ax.plot(t, pen["P1 = D2 + SiLU"], color=C["P1"], lw=2.2, label="P1: D2 + SiLU")
ax.plot(t, pen["D2 (P0)"], color=C["P0"], ls="--", label="P0: D2")
ax.set_xlabel("tilt t"); ax.set_ylabel("penalty"); ax.legend(frameon=False, loc="upper center")
ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
letter(ax, "b")
sc.put("b_lambda_mu_Ksilu", [lam, mu, ksilu]); sc.put("b_max_abs_D2_over_t", float(np.abs(pen["D2 (P0)"]).max()))
sc.put("b_P2_at_t", {f"{v:g}": float(pen["P2 = P1 + mu Q"][i]) for i, v in enumerate(t) if abs(v) in (1.0, 2.0, 3.0)})
sc.note("b: the tilt is a pure linear RBF pattern, so D2 is exactly 0 and the SiLU term is unchanged (P0 = P1 = 0); only P2's mu*Q sees it (Q = 2 t^2)")

# c) pre-activation change for the tilt: correlated (rho 0.95) vs independent inputs
Xc = SY.make_inputs(0.95, 1001)[:, :2]; Xi = SY.make_inputs(0.0, 1000)[:, :2]
d_c = phi(Xc[:, 0]) - phi(Xc[:, 1]); d_i = phi(Xi[:, 0]) - phi(Xi[:, 1])
tt = np.linspace(0, 3, 61)
rc, ri = float(np.sqrt(np.mean(d_c ** 2))), float(np.sqrt(np.mean(d_i ** 2)))
ax = axs[2]
ax.plot(tt, ri * tt, color=C["independent"], label=r"independent ($\rho$ = 0)")
ax.plot(tt, rc * tt, color=C["correlated"], label=r"correlated ($\rho$ = 0.95)")
ax.set_xlabel("tilt t"); ax.set_ylabel(r"RMS $\Delta$ pre-activation"); ax.legend(frameon=False, loc="upper left", handlelength=1.5)
letter(ax, "c")
sc.put("c_rms_per_unit_t", dict(correlated=rc, independent=ri, ratio_independent_over_correlated=ri / rc))
sc.put("c_inputs", "synth.make_inputs(0.95, seed 1001) and (0.0, seed 1000), first two columns (x1, x2), 2000 points each")
fig.tight_layout(w_pad=1.5)
sc.save(fig)

"""Figure 4: applicability (descriptive). a) fold A per-row |error| change against x_co, mu = 1e-6 and 1e-4; b) training data in
descriptor space with fold A and DEC rows. Run from the repo root: /usr/bin/python3 paper2/figures/make_fig4.py"""
import io
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from style import C, FULL, OI, Sidecar, letter

import anchor as AN
import sc_run as SR
from close import FlatQ
from kan import DELTA_IN, KAN, co_props, delta_head, predict, subset

sc = Sidecar("fig4_applicability")
ad = sc.json("paper2/revision/appdomain_results.json")
LOGGED = {"early": 0.1420, "1e-06": 1.0262, "0.0001": 0.3345}   # close_output.txt, certify_output.txt, appdomain_output.txt
sc.note("a: per-row |error| change from the early-stopped fold A ensemble (archived scbase_A_s0-4) to each polished ensemble "
        "(close_foldA_1e-06_s0-4, certify_foldA_0.0001_s0-4; mapped with close.FlatQ.to_kan as in paper2/revision/appdomain.py, ec769ad / 656b06c). "
        "Per-row values are in no committed result; they are evaluated here deterministically from SHA-verified archived weights held in memory, "
        "with no optimisation, and the ensemble RMSEs must reproduce the logged 0.1420 / 1.0262 / 0.3345")
sc.note("b: space B co-solvent descriptors (eps_co, ln eta_co) in the committed min-max scaling (kan.Scaler, G5 of appdomain_plan.md); "
        "fold A's co-solvent and the DEC test rows each occupy one point (G1)")
sc.note("DESCRIPTIVE (plan committed first, ec769ad); x_co on the horizontal axis because the correction enters as x_co * h (L2)")

S = SR.setup()
tr, X, use = S["tr"], S["B"]["x"].numpy().astype(float), S["use"]
names = [f"scbase_A_s{s}.pt" for s in range(5)] + [f"close_foldA_1e-06_s{s}.pt" for s in range(5)] + [f"certify_foldA_0.0001_s{s}.pt" for s in range(5)]
blobs = sc.load_weights(names)
fa = S["folds"]["A"]
bt, bh = subset(S["B"], fa["train"]), subset(S["B"], fa["test"])
btd = SR.dbl(bt)
used_f = np.unique(S["B"]["s"].numpy()[fa["train"]]).tolist()
y, anc = tr.log_k.values[fa["test"]], tr.anchor.values[fa["test"]]
P = {k: [] for k in LOGGED}
for s in range(5):
    net0 = KAN(len(DELTA_IN), [6, 1], len(S["salts"]))
    sd = torch.load(io.BytesIO(blobs[f"scbase_A_s{s}.pt"]), weights_only=False)
    sd = sd["early"] if isinstance(sd, dict) and "early" in sd else sd
    net0.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False); net0.eval()
    P["early"].append(predict(net0, delta_head, bh))
    for mu, nm in (("1e-06", f"close_foldA_1e-06_s{s}.pt"), ("0.0001", f"certify_foldA_0.0001_s{s}.pt")):
        x = torch.load(io.BytesIO(blobs[nm]), weights_only=False)
        x = x.numpy() if torch.is_tensor(x) else np.asarray(x)
        net = FlatQ(net0, btd, used_f, float(mu)).to_kan(x, len(DELTA_IN), len(S["salts"]))
        P[mu].append(predict(net.float(), delta_head, bh))
ens = {k: np.mean(v, 0) for k, v in P.items()}
rm = {k: float(np.sqrt(np.mean((anc + ens[k] - y) ** 2))) for k in ens}
if not all(abs(rm[k] - LOGGED[k]) < 1e-4 for k in rm):
    raise SystemExit(f"STOP: fold A ensembles do not reproduce the logged RMSEs: {rm}")
sc.put("ensemble_rmse_reproduced", rm)
err = {k: np.abs(anc + ens[k] - y) for k in ens}
xco = S["B"]["x_co"].numpy().astype(float)[fa["test"]]

fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.3), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
for ax, mu, lab, col in zip(axs[:2], ("1e-06", "0.0001"), (r"$\mu = 10^{-6}$", r"$\mu = 10^{-4}$"), (C["runaway"], C["certified"])):
    de = err[mu] - err["early"]
    ax.scatter(xco, de, s=4, color=col, alpha=0.45, lw=0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"co-solvent fraction $x_{\mathrm{co}}$")
    ax.text(0.03, 0.97, f"{lab}\nfold A RMSE {rm['early']:.4f}" + r"$\to$" + f"{rm[mu]:.4f}", transform=ax.transAxes, va="top", fontsize=6.5)
    r_, p_ = spearmanr(xco, de)
    sc.put(f"a_{mu}", dict(median_change=float(np.median(de)), mean_change=float(np.mean(de)), spearman_xco=dict(rho=float(r_), p=float(p_)),
                           n=int(len(de)), n_distinct_xco=int(len(np.unique(np.round(xco, 9)))),
                           committed_spearman=dict(dM=ad["analysis2"][mu]["dM"]["rho"], off_mix=ad["analysis2"][mu]["off_mix"]["rho"])))
axs[0].set_ylabel(r"$|$error$|$ change, early$\to$polished")
axs[1].sharey(axs[0])
letter(axs[0], "a", dx=-0.22); letter(axs[1], "b", dx=-0.12)

ax = axs[2]
IP = [DELTA_IN.index("eps_co"), DELTA_IN.index("lneta_co")]
_, te, _, _, Xte, _, _, _, has = AN.prepare(verbose=False)
Xte[["eps_co", "lneta_co", "M_co"]] = co_props(te)
Zte = S["sc"](Xte)[has].astype(float)
T, A = X[use], X[fa["test"]]
ax.scatter(T[:, IP[0]], T[:, IP[1]], s=10, color=C["early"], lw=0, label="training (2069 anchored rows)")
ax.scatter(A[:, IP[0]], A[:, IP[1]], s=30, marker="s", facecolors="none", edgecolors=C["runaway"], lw=0.9, label="fold A (ethyl acetate)")
ax.scatter(Zte[:, IP[0]], Zte[:, IP[1]], s=45, marker="*", color=C["certified"], lw=0, label=f"DEC test ({int(has.sum())} anchored rows)")
ax.set_xlabel(r"$\varepsilon_{\mathrm{co}}$ (scaled)"); ax.set_ylabel(r"$\ln\eta_{\mathrm{co}}$ (scaled)")
ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(0.0, 1.0), handletextpad=0.3, borderaxespad=0.1)
ax.set_ylim(-1.15, 1.9)
letter(ax, "c", dx=-0.2)
sc.put("c_points", dict(n_train_rows=int(use.sum()), n_train_distinct=int(len(np.unique(np.round(T[:, IP], 9), axis=0))),
                        foldA_point=np.unique(np.round(A[:, IP], 6), axis=0).tolist(), dec_point=np.unique(np.round(Zte[:, IP], 6), axis=0).tolist(),
                        n_dec_rows=int(has.sum()), foldA_offpattern=ad["analysis1"]["folds"]["A"]["off"]["median"], dec_offpattern=ad["analysis3"]["off"]["median"]))
fig.tight_layout(w_pad=1.0)
sc.save(fig)

"""Figure 9: convergence is not uniqueness. Run from the repo root: python3 paper2/figures/make_fig9.py"""
import io
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt
from style import C, FULL, OI, Sidecar, letter

sc = Sidecar("fig9_uniqueness")
scr = sc.json("arch/sc_results.json")
ce = sc.json("certify_results.json")

# ---- a) near-null Hessian eigenvalues at the end of polishing (|eig| < 1e-8 x max, certify_point)
nn = {"KAN-δ\nfold A\npolished": ([p["cert"]["n_null"] for p in scr["polish"]["base"]], [p["cert"]["certified"] for p in scr["polish"]["base"]]),
      "μ = 1e-4\nall rows\nfinal": ([ce["cert"][f"0.0001|ind_{i}"]["n_null"] for i in range(6)], [ce["cert"][f"0.0001|ind_{i}"]["certified"] for i in range(6)]),
      "SC-KAN-δ\nfold A\npolished": ([p["cert"]["n_null"] for p in scr["polish"]["sc"]], [p["cert"]["certified"] for p in scr["polish"]["sc"]])}
sc.note("a: n_null = eigenvalues with |eig| < 1e-8 x max at the end of polishing (certify.py's certify_point); KAN-δ baseline and SC from arch/sc (e00e37f), "
        "mu = 1e-4 from certify.py (36f295b), the 6 real-data seeds. The baseline points are NOT stationary (0/10 certified, 500-iteration cap), so its count is "
        "at a non-minimum; stage2 Part A finals have no stored near-null count and are not shown")

# ---- b) seed disagreement d_AB
sc.note("b: committed d_AB: baseline and SC at the fold A early state (arch/sc, 45 pairs); capped baseline and certified mu = 1e-4 on all rows (certify.py, 15 pairs)")
sc.note("b: POLISHED fold A d_AB (baseline and SC) is not in any committed result; evaluated here deterministically from the archived polished weights "
        "(SHA-verified, in memory; predictions on the 1434 fold A training rows / SD of their delta targets, as arch/sc)")
from kan import KAN, DELTA_IN, subset
from trust import Flat
from sc_model import SCKAN, FlatSC, predict_sc
import sc_run as SR
names = [f"sc_A_b0.0001_s{s}.pt" for s in range(10)] + [f"sc_A_b0.0001_s{s}_pol.pt" for s in range(10)] + \
        [f"scbase_A_s{s}.pt" for s in range(10)] + [f"scbase_A_s{s}_pol.pt" for s in range(10)]
blobs = sc.load_weights(names)
ld = lambda n: torch.load(io.BytesIO(blobs[n]), weights_only=False)
S = SR.setup()
n_salt, d = len(S["salts"]), len(DELTA_IN)
bA = subset(S["B"], S["folds"]["A"]["train"]); bAd = SR.dbl(bA); usedA = np.unique(bA["s"].numpy()).tolist()
sd_y = float(bA["y"].std())


def kan_from(sd):
    k = KAN(d, [6, 1], n_salt)
    miss, unexp = k.load_state_dict({n: t.float() for n, t in sd.items()}, strict=False)
    assert not unexp and all(x.endswith(".grid") for x in miss)
    return k.eval()


pol = {"sc": [], "base": []}
for s in range(10):
    e = SCKAN(n_salt); e.load_state_dict(ld(f"sc_A_b0.0001_s{s}.pt")["early"]); e = e.double().eval()
    pol["sc"].append(predict_sc(FlatSC(e, bA, 1e-4).to_net(ld(f"sc_A_b0.0001_s{s}_pol.pt")["x"].numpy()), bAd))
    b = Flat(kan_from(ld(f"scbase_A_s{s}.pt")["early"]), bAd, usedA).to_kan(ld(f"scbase_A_s{s}_pol.pt")["x"].numpy(), d, n_salt)
    pol["base"].append(SR.base_pred(b, bAd))
dab_pol = {k: [float(np.sqrt(np.mean((v[a] - v[b_]) ** 2)) / sd_y) for a, b_ in itertools.combinations(range(10), 2)] for k, v in pol.items()}
db = {"KAN-δ\nfold A\nearly": (scr["dab"]["base"]["all"], "runaway"), "KAN-δ\nfold A\npolished*": (dab_pol["base"], "runaway"),
      "KAN-δ\nall rows\ncapped": (ce["after"]["0.0001"]["dAB_capped"], "runaway"), "μ = 1e-4\nall rows\ncertified": (ce["after"]["0.0001"]["dAB"], "certified"),
      "SC-KAN-δ\nfold A\nearly": (scr["dab"]["sc"]["all"], "early"), "SC-KAN-δ\nfold A\npolished*": (dab_pol["sc"], "certified")}

fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.4), gridspec_kw=dict(width_ratios=[0.8, 1.2]))
ax = axs[0]
for j, (lab, (v, cert)) in enumerate(nn.items()):
    v = np.asarray(v)
    ax.scatter(j + np.random.default_rng(j).uniform(-0.15, 0.15, len(v)), v, s=9, color=[C["certified"] if c else C["runaway"] for c in cert], lw=0, alpha=0.85)
    ax.plot([j - 0.25, j + 0.25], [np.median(v)] * 2, color=OI["black"], lw=1)
ax.set_yscale("symlog", linthresh=1); ax.set_ylim(-0.3, 300)
ax.set_xticks(range(len(nn))); ax.set_xticklabels(list(nn), fontsize=6); ax.set_ylabel("near-null Hessian eigenvalues")
letter(ax, "a", dx=-0.22)
ax = axs[1]
for j, (lab, (v, kind)) in enumerate(db.items()):
    v = np.asarray(v)
    ax.scatter(j + np.random.default_rng(10 + j).uniform(-0.15, 0.15, len(v)), v, s=6, color=C[kind], lw=0, alpha=0.7)
    q1, q3 = np.percentile(v, [25, 75])
    ax.plot([j, j], [q1, q3], color=OI["black"], lw=2.2, solid_capstyle="butt")
    ax.plot([j - 0.25, j + 0.25], [np.median(v)] * 2, color=OI["black"], lw=1)
ax.set_xticks(range(len(db))); ax.set_xticklabels(list(db), fontsize=6); ax.set_ylabel(r"seed disagreement $d_{AB}$"); ax.set_ylim(0, None)
letter(ax, "b", dx=-0.13)
fig.tight_layout(w_pad=1.2)
sc.put("a_n_null", {k.replace("\n", " "): dict(values=v, certified=c, median=float(np.median(v))) for k, (v, c) in nn.items()})
sc.put("b_dAB", {k.replace("\n", " "): dict(n_pairs=len(v), median=float(np.median(v)), q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)),
                                            min=float(min(v)), max=float(max(v)), values=list(map(float, v))) for k, (v, _) in db.items()})
sc.note("b: '*' = evaluated from archived weights in this script (see above); bar = median, thick line = interquartile range")
sc.save(fig)

"""Figure 4: where the runaway lives. Run from the repo root: python3 paper2/figures/make_fig4.py"""
import io
import numpy as np
import torch
import matplotlib.pyplot as plt
from style import BLOCK_C, C, FULL, OI, Sidecar, letter

sc = Sidecar("fig4_where")
BLOCKS = ["layer-1 affine RBF", "layer-1 curved RBF", "layer-2 affine RBF", "layer-2 curved RBF", "SiLU weights", "biases", "salt embedding"]

# ---- a) real data: deterministic evaluation of archived weights (no committed affine/curved split exists)
from kan import KAN, DELTA_IN, N_RBF, subset
from trust import Flat
import sc_run as SR
sc.note("a, real data: shares of ||delta theta||^2 by block computed here from the archived weights (SHA-verified, in memory): "
        "delta_s* (107 epochs), fiber2_ind_* (capped), stage2_A_ind_* (Part A final, flat vectors rebuilt on their fiber2 start); "
        "no committed result splits the real-data change into affine/curved RBF parts")
sc.note("affine RBF = projection of each edge's 8 RBF coefficients onto span{1, k}; curved = the rest; layer-1 RBF includes the salt-embedding edges; "
        "'salt embedding' (the embedding table itself) exists only in the real model")
names = [f"{p}{i}.pt" for p in ("delta_s", "fiber2_ind_", "stage2_A_ind_") for i in range(6)]
blobs = sc.load_weights(names)
ld = lambda n: torch.load(io.BytesIO(blobs[n]), weights_only=False)
S = SR.setup()
n_salt, d = len(S["salts"]), len(DELTA_IN)
ball = SR.dbl(subset(S["B"], S["use"])); used = np.unique(ball["s"].numpy()).tolist()
NB = torch.tensor(np.linalg.qr(np.column_stack([np.ones(N_RBF), np.arange(N_RBF, dtype=float)]))[0])


def kan_from(sd):
    k = KAN(d, [6, 1], n_salt)
    miss, unexp = k.load_state_dict({n: t.float() for n, t in sd.items()}, strict=False)
    assert not unexp and all(x.endswith(".grid") for x in miss)
    return k.double().eval()


def params(net):
    return {k: v.detach().double() for k, v in net.named_parameters()}


def shares(a, b):
    dl = {k: b[k] - a[k] for k in a}
    e = {}
    for L in (0, 1):
        c = dl[f"layers.{L}.coef"]; aff = float(((c @ NB) ** 2).sum()); tot = float((c ** 2).sum())
        e[f"layer-{L + 1} affine RBF"], e[f"layer-{L + 1} curved RBF"] = aff, tot - aff
    e["SiLU weights"] = float((dl["layers.0.base.weight"] ** 2).sum() + (dl["layers.1.base.weight"] ** 2).sum())
    e["biases"] = float((dl["layers.0.base.bias"] ** 2).sum() + (dl["layers.1.base.bias"] ** 2).sum())
    e["salt embedding"] = float((dl["emb.weight"] ** 2).sum())
    T = sum(e.values())
    return {k: v / T for k, v in e.items()}


real = {"107 epochs → capped": [], "capped → Part A": []}
for i in range(6):
    p107 = params(kan_from(ld(f"delta_s{i}.pt")))
    fcap = kan_from(ld(f"fiber2_ind_{i}.pt")); pcap = params(fcap)
    pA = params(Flat(kan_from(ld(f"fiber2_ind_{i}.pt")), ball, used).to_kan(ld(f"stage2_A_ind_{i}.pt").numpy(), d, n_salt))
    real["107 epochs → capped"].append(shares(p107, pcap)); real["capped → Part A"].append(shares(pcap, pA))

# ---- synthetic: committed anatomy_synth results (per-model shares; SiLU and biases summed over layers)
an = sc.json("paper2/anatomy_synth_results.json")
syn = {}
for lev in ("P0", "P1"):
    rr = []
    for s in range(8):
        sh = an["rows"][f"correlated_{lev}_s{s}"]["d"]["shares"]
        rr.append({"layer-1 affine RBF": sh["L1 affine RBF"], "layer-1 curved RBF": sh["L1 curved RBF"], "layer-2 affine RBF": sh["L2 affine RBF"],
                   "layer-2 curved RBF": sh["L2 curved RBF"], "SiLU weights": sh["L1 SiLU"] + sh["L2 SiLU"], "biases": sh["L1 bias"] + sh["L2 bias"],
                   "salt embedding": 0.0})
    syn[f"synthetic correlated {lev}\n(early → polished)"] = rr
sc.note("a, synthetic: anatomy_synth_results.json (ea22a56) per-model shares, correlated P0/P1, 8 seeds each; bars = mean over models (sums to 1)")

groups = {"real\n107 ep→cap.": real["107 epochs → capped"], "real\ncap.→Part A": real["capped → Part A"],
          "synth.\ncorr. P0": syn["synthetic correlated P0\n(early → polished)"], "synth.\ncorr. P1": syn["synthetic correlated P1\n(early → polished)"]}
fig, axs = plt.subplots(1, 2, figsize=(FULL, 2.9), gridspec_kw=dict(width_ratios=[1, 1.05]))
ax = axs[0]
x = np.arange(len(groups)); bottom = np.zeros(len(groups))
mean = {g: {b: float(np.mean([r[b] for r in rr])) for b in BLOCKS} for g, rr in groups.items()}
for b in BLOCKS:
    v = np.array([mean[g][b] for g in groups])
    ax.bar(x, v, 0.65, bottom=bottom, color=BLOCK_C[b], label=b, edgecolor="white", lw=0.3); bottom += v
ax.set_xticks(x); ax.set_xticklabels(list(groups), fontsize=6.5); ax.set_ylabel(r"share of $\|\Delta\theta\|^2$"); ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, columnspacing=0.8, handlelength=1.2)
letter(ax, "a", dx=-0.12)
sc.put("a_mean_shares", mean)
sc.put("a_per_model", {g: rr for g, rr in groups.items()})
sc.note("a: real = mean over 6 seeds; real 107 ep -> capped and capped -> Part A are the two unregularised segments")

# ---- b) largest hidden-unit correlation (paper2/checks)
ck = sc.json("paper2/checks/checks_results.json")["item1"]["rows"]
cats = [("synth.\ncertified", [r for r in ck if r["group"] == "synth" and r["state"] == "polished" and r["certified"]], "certified"),
        ("synth.\nnot cert.", [r for r in ck if r["group"] == "synth" and r["state"] == "polished" and not r["certified"]], "runaway"),
        ("real\ncapped", [r for r in ck if r["group"] == "real" and r["state"] == "fiber2 capped"], "runaway"),
        ("real\nPart A", [r for r in ck if r["group"] == "real" and r["state"] == "stage2 Part A final"], "runaway"),
        ("SC-KAN-δ\npolished", [r for r in ck if r["group"] == "sc" and r["state"] == "polished"], "certified"),
        ("KAN-δ\npolished", [r for r in ck if r["group"] == "scbase (extra)" and r["state"] == "polished"], "runaway")]
ax = axs[1]
vals = {}
for j, (lab, rr, kind) in enumerate(cats):
    y = np.array([r["max"] for r in rr]); vals[lab.replace("\n", " ")] = dict(values=y.tolist(), median=float(np.median(y)), n=len(y), kind=kind)
    ax.scatter(j + np.random.default_rng(j).uniform(-0.18, 0.18, len(y)), y, s=7, color=C[kind], alpha=0.8, lw=0)
    ax.plot([j - 0.28, j + 0.28], [np.median(y)] * 2, color=OI["black"], lw=1)
ax.set_xticks(range(len(cats))); ax.set_xticklabels([c[0] for c in cats], fontsize=6)
ax.set_ylabel("largest |hidden-unit corr.|"); ax.set_ylim(0, 1.02)
letter(ax, "b", dx=-0.2)
sc.put("b_max_hidden_corr", vals)
sc.note("b: paper2/checks (6e1df32), training rows; SC-KAN-δ and KAN-δ are the fold A models of arch/sc; colour = certified (blue) vs not certified / runaway (vermillion); bar = median")
fig.tight_layout(w_pad=1.0)
sc.save(fig)

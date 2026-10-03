"""close: targeted null-space penalty intervention (see close_prereg.md).
Loss = data + smoothness + mu * Q,  Q = sum_edges ||P_null c||^2 + sum of squared SiLU base weights.
Run: python3 close.py >> close_output.txt    Weights: gauge_weights/close_* (not committed)   Figures: figs/close/"""
import io
import itertools
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from anchor import rmse
from gauge import Gauge, align, setup
from kan import KAN, N_RBF, SMOOTH, delta_head, predict, subset, to_batch, train
from stage2 import extract_verified, in_memory, run_exact
from trust import EIG_TOL, Flat

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
ROOT = Path(__file__).parent
FIG = ROOT / "figs" / "close"
W_DIR = ROOT / "gauge_weights"
MUS = [1e-6, 1e-5, 1e-4]
IND, STU, FOLD_SEEDS = list(range(6)), [115, 104, 114], list(range(5))
BLOCKS = {"layer-1 RBF": ["layers.0.coef"], "layer-2 RBF": ["layers.1.coef"],
          "SiLU base": ["layers.0.base.weight", "layers.1.base.weight"],
          "biases": ["layers.0.base.bias", "layers.1.base.bias"], "salt embedding": ["emb.weight"]}
RESULTS = {}


def log(*a):
    print(*a, flush=True)


def null_basis():
    Q, _ = np.linalg.qr(np.column_stack([np.ones(N_RBF), np.arange(N_RBF, dtype=float)]))
    return torch.tensor(Q)


NB = null_basis()


class FlatQ(Flat):
    """Flat 609-vector with loss = data + smoothness + mu * Q."""

    def __init__(self, net, bd, used, mu):
        super().__init__(net, bd, used)
        self.mu = mu

    def terms(self, v):
        prm = self.params(v)
        pred = functional_call(self.net, prm, (self.bd["x"], self.bd["s"]))[:, 0]
        data = F.mse_loss(self.bd["x_co"] * pred, self.bd["y"])
        c0, c1 = prm["layers.0.coef"], prm["layers.1.coef"]
        pen = SMOOTH * sum(((c[..., 2:] - 2 * c[..., 1:-1] + c[..., :-2]) ** 2).sum() for c in (c0, c1))
        q = sum(((c @ NB) ** 2).sum() for c in (c0, c1)) + (prm["layers.0.base.weight"] ** 2).sum() + (prm["layers.1.base.weight"] ** 2).sum()
        return data, pen, q

    def loss_t(self, v):
        data, pen, q = self.terms(v)
        return data + pen + self.mu * q

    def base_loss(self, x):
        with torch.no_grad():
            data, pen, _ = self.terms(torch.tensor(x))
            return float(data + pen)

    def report(self, x):
        with torch.no_grad():
            data, pen, q = self.terms(torch.tensor(x))
        return dict(data=float(data), pen=float(pen), muQ=float(self.mu * q), Q=float(q))


def block_norms(net, used):
    sd = dict(net.named_parameters())
    return {blk: float(sum(((sd[k][used] if k == "emb.weight" else sd[k]) ** 2).sum() for k in keys)) for blk, keys in BLOCKS.items()}


def cross_edge(net, bd):
    with torch.no_grad():
        L0 = net.layers[0]
        z = torch.cat([bd["x"], torch.tanh(net.emb(bd["s"]))], 1)
        E = torch.stack([L0.edge(i, z[:, i]) for i in range(z.shape[1])])
        s = E.sum(0) + L0.base.bias
        ratio = (E.pow(2).mean(1).sqrt().sum(0) / E.sum(0).pow(2).mean(0).sqrt()).numpy()
        return float(np.median(ratio)), float((s.abs() > 3).double().mean())


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    tr, te, te_anc, D, _ = setup()
    d, m = D["d_num"], D["train"]
    Ztr, str_ = D["Z"][m], D["s"][m]
    b = subset(D["batch"], m)
    bd = {k: (v.double() if v.is_floating_point() else v) for k, v in b.items()}
    used = np.unique(str_).tolist()
    log("DISCREPANCIES / choices (files win; see close_prereg.md):")
    log("  K1 students start from the control-3 AdamW stage (530-2680 epochs); no 107-epoch weights exist for them")
    log("  K2 Q covers all 66 edges (incl. the 2 salt edges per unit) + layers.0/1 SiLU base weights; biases, embedding unpenalised")
    log("  K3 2 h wall-clock safety cap per run (as stage2), not in the brief")
    log("  K6 'that mu' = mu converging the most models (ties: smallest); V2-V5 on models converged at that mu")

    log("\n==================== inputs (SHA-256 verified, held in memory) ====================")
    names = ([f"delta_s{i}.pt" for i in IND] + [f"control3_R_s{s}.pt" for s in STU] + [f"fiber2_ind_{i}.pt" for i in IND] +
             [f"stage2_A_ind_{i}.pt" for i in IND])
    blobs = in_memory(extract_verified(names))

    def kan_from(blob):
        k = KAN(d, [6, 1], D["n_salt"])
        miss, unexp = k.load_state_dict({n: t.float() for n, t in torch.load(io.BytesIO(blob)).items()}, strict=False)
        assert not unexp and all(x.endswith(".grid") for x in miss), (miss, unexp)
        return k.eval()

    teacher = kan_from(blobs["delta_s0.pt"])
    yT = torch.tensor(predict(teacher, delta_head, to_batch(Ztr, str_, x_co=b["x_co"].numpy())), dtype=torch.float64)
    models = [(f"ind {i}", f"delta_s{i}.pt", bd["y"]) for i in IND] + [(f"stu {s}", f"control3_R_s{s}.pt", yT) for s in STU]

    # references: stage2 Part A final data term (V2), fiber2 capped predictions (V4)
    ref_A, capped_delta = {}, {}
    for i in IND:
        cap = kan_from(blobs[f"fiber2_ind_{i}.pt"])
        capped_delta[i] = predict(cap, delta_head, b)
        xa = torch.load(io.BytesIO(blobs[f"stage2_A_ind_{i}.pt"])).numpy()
        ka = Flat(cap, bd, used).to_kan(xa, d, D["n_salt"])
        with torch.no_grad():
            ref_A[i] = float(F.mse_loss(delta_head(ka, bd), bd["y"]))
    log("  stage2 Part A final data terms (V2 reference): " + ", ".join(f"ind {i} {v:.4e}" for i, v in ref_A.items()))

    # ---------------- runs
    runs, finals = {}, {}
    for mu in MUS:
        log(f"\n==================== mu = {mu:g}: trust-exact from the pre-runaway start ====================")
        for tag, fname, y in models:
            bb = dict(bd); bb["y"] = y
            start = kan_from(blobs[fname])
            flat = FlatQ(start, bb, used, mu)
            key = f"{mu:g}|{tag}"
            cache = W_DIR / f"close_{mu:g}_{tag.replace(' ', '_')}.pt"
            meta = W_DIR / f"close_{mu:g}_{tag.replace(' ', '_')}.json"
            cached = cache.exists() and meta.exists()
            if cached:
                x = torch.load(cache).numpy(); info = json.loads(meta.read_text())
            else:
                x, info, hist = run_exact(tag, flat)
                net = flat.to_kan(x, d, D["n_salt"])
                n0 = sum(block_norms(start.double(), used).values())
                info.update(flat.report(x), start=flat.report(flat.x0), norms=block_norms(net, used), theta2=sum(block_norms(net, used).values()),
                            theta2_start=n0, cross_edge=cross_edge(net, bb)[0], saturated=cross_edge(net, bb)[1], hist=hist)
                torch.save(torch.tensor(x), cache)
                meta.write_text(json.dumps(info, default=float))
            runs[key], finals[key] = info, (flat, x)
            log(f"  {tag}: {info['stop']} after {info['iters']} iters ({info['wall_s']} s) | scipy: {info['message']}" + ("  [cached]" if cached else ""))
            log(f"      converged {info['converged']}; data {info['data']:.4e} (start {info['start']['data']:.4e}); penalty {info['pen']:.3e}; "
                f"mu*Q {info['muQ']:.3e}; ||theta||^2 {info['theta2']:.1f} ({info['theta2'] / info['theta2_start']:.2f} x start); "
                f"max|grad| {info['grad']:.1e}; neg eigs {info['neg']}; cross-edge {info['cross_edge']:.2f}; saturated {info['saturated']:.3f}")
            log("      ||theta||^2 per block: " + ", ".join(f"{k} {v:.1f}" for k, v in info["norms"].items()))

    conv = {mu: [t for t, _, _ in models if runs[f"{mu:g}|{t}"]["converged"]] for mu in MUS}
    log("\n  converged: " + "; ".join(f"mu {mu:g}: {len(c)}/9 {c}" for mu, c in conv.items()))
    sel = max(MUS, key=lambda mu: (len(conv[mu]), -mu))
    log(f"  selected mu (most converged; ties smallest): {sel:g}")

    # ---------------- after convergence at the selected mu
    ci = [int(t.split()[1]) for t in conv[sel] if t.startswith("ind")]
    cs = [int(t.split()[1]) for t in conv[sel] if t.startswith("stu")]
    after = dict(selected=sel, conv_ind=ci, conv_stu=cs)
    sd_y = bd["y"].std().item()
    if len(ci) >= 2:
        pairs = list(itertools.combinations(ci, 2))
        dl = {i: predict(finals[f"{sel:g}|ind {i}"][0].to_kan(finals[f"{sel:g}|ind {i}"][1], d, D["n_salt"]).float(), delta_head, b) for i in ci}
        after["dAB"] = [float(np.sqrt(np.mean((dl[A] - dl[B]) ** 2)) / sd_y) for A, B in pairs]
        after["dAB_capped"] = [float(np.sqrt(np.mean((capped_delta[A] - capped_delta[B]) ** 2)) / sd_y) for A, B in pairs]
        log(f"  d_AB over {len(pairs)} converged seed pairs: median {np.median(after['dAB']):.4f} (fiber2 capped, same pairs {np.median(after['dAB_capped']):.4f})")
    if cs:
        gT = Gauge(teacher, Ztr, str_, d)
        um_all = []
        for s in cs:
            net = finals[f"{sel:g}|stu {s}"][0].to_kan(finals[f"{sel:g}|stu {s}"][1], d, D["n_salt"]).float()
            um = np.abs(align(gT, Gauge(net, Ztr, str_, d)))
            um_all += um.tolist()
            log(f"  student {s}: unit match to teacher median {np.median(um):.3f} {np.round(um, 3).tolist()}")
        after["unit_match"] = um_all

    # ---------------- accuracy (fold A)
    log(f"\n==================== accuracy: fold A, mu = {sel:g} ====================")
    ho = (tr.combo == "EA+PC").values & m
    dt = m & ~ho
    bt = subset(D["batch"], dt)
    btd = {k: (v.double() if v.is_floating_point() else v) for k, v in bt.items()}
    bh = subset(D["batch"], ho)
    used_f = np.unique(D["s"][dt]).tolist()
    preds, fold_info = [], []
    for s in FOLD_SEEDS:
        cache = W_DIR / f"close_foldA_{sel:g}_s{s}.pt"
        meta = W_DIR / f"close_foldA_{sel:g}_s{s}.json"
        net0 = train([6, 1], d, D["n_salt"], delta_head, bt, epochs=107, seed=s)[0]
        flat = FlatQ(net0, btd, used_f, sel)
        cached = cache.exists() and meta.exists()
        if cached:
            x = torch.load(cache).numpy(); info = json.loads(meta.read_text())
        else:
            x, info, _ = run_exact(f"foldA s{s}", flat)
            torch.save(torch.tensor(x), cache); meta.write_text(json.dumps(info, default=float))
        net = flat.to_kan(x, d, D["n_salt"]).float()
        preds.append(predict(net, delta_head, bh))
        fold_info.append(info)
        log(f"  fold A seed {s}: {info['stop']} after {info['iters']} iters; converged {info['converged']}" + ("  [cached]" if cached else ""))
    yh = tr.log_k.values[ho]; anc = tr.anchor.values[ho]
    pa = anc + np.mean(preds, 0)
    fa = (rmse(pa, yh), float(np.mean(pa - yh)))
    log(f"  fold A RMSE {fa[0]:.4f}, bias {fa[1]:+.4f}  (KAN-delta 0.1420)")

    # ---------------- verdicts
    v = {}
    best = {mu: (sum(t.startswith("ind") for t in conv[mu]), sum(t.startswith("stu") for t in conv[mu])) for mu in MUS}
    v1 = [mu for mu in MUS if best[mu][0] >= 4 and best[mu][1] >= 2]
    v["V1"] = ("held" if v1 else "failed") + " (" + ", ".join(f"mu {mu:g}: {a}/6 seeds, {c}/3 students" for mu, (a, c) in best.items()) + ")"
    if ci:
        r2 = {i: runs[f"{sel:g}|ind {i}"]["data"] / ref_A[i] for i in ci}
        v["V2"] = ("held" if all(x <= 1.05 for x in r2.values()) else "failed") + " (" + ", ".join(f"ind {i}: {x:.3f}" for i, x in r2.items()) + ")"
    else:
        v["V2"] = f"inconclusive (no seed converged at mu {sel:g})"
    if conv[sel]:
        r3 = {t: runs[f"{sel:g}|{t}"]["theta2"] / runs[f"{sel:g}|{t}"]["theta2_start"] for t in conv[sel]}
        v["V3"] = ("held" if all(x <= 10 for x in r3.values()) else "failed") + " (" + ", ".join(f"{t}: {x:.2f}" for t, x in r3.items()) + ")"
    else:
        v["V3"] = f"inconclusive (nothing converged at mu {sel:g})"
    v["V4"] = (("held" if np.median(after["dAB"]) <= 0.5 * np.median(after["dAB_capped"]) else "failed") +
               f" (median {np.median(after['dAB']):.4f} vs 0.5 x {np.median(after['dAB_capped']):.4f})") if "dAB" in after else \
        f"inconclusive ({len(ci)} seed(s) converged)"
    v["V5"] = (("held" if np.median(after["unit_match"]) >= 0.8 else "failed") + f" (median {np.median(after['unit_match']):.3f})") \
        if "unit_match" in after else "inconclusive (no student converged)"
    v["V6"] = ("held" if fa[0] <= 0.150 else "failed") + f" (RMSE {fa[0]:.4f}, bias {fa[1]:+.4f})"
    log("\nVERDICTS: " + "; ".join(f"{k}: {x}" for k, x in v.items()))
    RESULTS.update(runs={k: {kk: vv for kk, vv in r.items() if kk != "hist"} for k, r in runs.items()}, conv=conv, selected=sel,
                   after=after, foldA=dict(rmse=fa[0], bias=fa[1], runs=[{k: vv for k, vv in i.items() if k != "eig_checks"} for i in fold_info]),
                   ref_A=ref_A, verdicts=v)
    (ROOT / "close_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))

    fig, axs = plt.subplots(1, len(MUS), figsize=(18, 4.8), sharey=True)
    for ax, mu in zip(axs, MUS):
        for tag, _, _ in models:
            h = runs[f"{mu:g}|{tag}"].get("hist")
            if h:
                ax.semilogy(h["grad"], lw=0.9, label=tag)
        ax.axhline(1e-7, color="r", ls="--"); ax.set_title(f"mu = {mu:g}: max|grad|"); ax.set_xlabel("trust-exact iteration")
    axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "close_grad.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()

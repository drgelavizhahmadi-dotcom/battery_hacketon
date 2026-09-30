"""rank: rank-constrained layer-1 coefficients (arm G: co-solvent block; arm F: full M). See rank_prereg.md.
Run: python3 rank.py      Weights: gauge_weights/rank_* (not committed)   Figures: figs/rank/"""
import copy
import itertools
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.func import functional_call

from anchor import rmse
from fiber2 import certify
from gauge import W_DIR, corr_matrix, grid_inputs, parts, setup
from kan import KAN, LR, N_RBF, SMOOTH, WD, delta_head, predict, subset, to_batch, train
from support import design_matrix, theta

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
ROOT = Path(__file__).parent
FIG = ROOT / "figs" / "rank"
REL = 1e-2
SEEDS, FOLD_SEEDS, EPOCHS = list(range(6)), list(range(5)), 107
CO = ["eps_co", "lneta_co", "M_co"]
EDGE_INPUTS = ["invT", "molal"]
RESULTS = {}


def log(*a):
    print(*a, flush=True)


def retained(A):
    _, S, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt[S / S[0] >= REL].T


class RankKAN(nn.Module):
    """KAN-delta with layer-1 coefficients on the affected inputs constrained to theta_j = P z_j."""

    def __init__(self, d, n_salt, P, aff, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        tpl = KAN(d, [6, 1], n_salt)  # identical initialisation to kan.train
        L0 = tpl.layers[0]
        self.d, self.aff, self.I = d, list(aff), d + 2
        self.free = [i for i in range(self.I) if i not in self.aff]
        self.emb, self.l1 = tpl.emb, tpl.layers[1]
        self.register_buffer("grid", L0.grid.clone())
        self.h = L0.h
        self.register_buffer("P", torch.tensor(P, dtype=torch.float32))
        with torch.no_grad():
            th0 = torch.cat([torch.cat([L0.base.weight[:, i:i + 1], L0.coef[:, i, :]], 1) for i in self.aff], 1)
            self.z = nn.Parameter(th0 @ self.P)
            self.bias0 = nn.Parameter(L0.base.bias.clone())
            self.bw_free = nn.Parameter(L0.base.weight[:, self.free].clone())
            self.coef_free = nn.Parameter(L0.coef[:, self.free, :].clone())

    def eff(self):
        th = (self.z @ self.P.T).reshape(self.z.shape[0], len(self.aff), 1 + N_RBF)
        w, c = [None] * self.I, [None] * self.I
        for a, i in enumerate(self.aff):
            w[i], c[i] = th[:, a, 0], th[:, a, 1:]
        for f, i in enumerate(self.free):
            w[i], c[i] = self.bw_free[:, f], self.coef_free[:, f, :]
        return torch.stack(w, 1), torch.stack(c, 1)

    def forward(self, x, salt):
        W, C = self.eff()
        zin = torch.cat([x, torch.tanh(self.emb(salt))], 1)
        rbf = torch.exp(-(((zin[..., None] - self.grid) / self.h) ** 2))
        s = F.silu(zin) @ W.T + self.bias0 + torch.einsum("bik,oik->bo", rbf, C)
        return self.l1(torch.tanh(s))

    def smoothness(self):
        _, C = self.eff()
        d2 = C[..., 2:] - 2 * C[..., 1:-1] + C[..., :-2]
        return (d2 ** 2).sum() + self.l1.smoothness()

    def to_kan(self, n_salt):
        K = KAN(self.d, [6, 1], n_salt)
        W, C = self.eff()
        with torch.no_grad():
            K.layers[0].base.weight.copy_(W); K.layers[0].coef.copy_(C); K.layers[0].base.bias.copy_(self.bias0)
            K.emb.weight.copy_(self.emb.weight)
            K.layers[1].load_state_dict({k: v.float() for k, v in self.l1.state_dict().items()})
        return K.eval()


def train_rank(D, P, aff, seed, b):
    net = RankKAN(D["d_num"], D["n_salt"], P, aff, seed)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    for _ in range(EPOCHS):
        opt.zero_grad()
        (F.mse_loss(delta_head(net, b), b["y"]) + SMOOTH * net.smoothness()).backward()
        opt.step()
    return net.eval()


def dbl(b):
    return {k: (v.double() if v.is_floating_point() else v) for k, v in b.items()}


def grad_stats(net, bd, groups=None):
    """Own-parameter loss and max|grad| (float64); optional split by parameter-name prefix."""
    net.zero_grad()
    L = F.mse_loss(delta_head(net, bd), bd["y"]) + SMOOTH * net.smoothness()
    L.backward()
    out = dict(loss=float(L), grad=max(float(p.grad.abs().max()) for p in net.parameters()))
    for g, pref in (groups or {}).items():
        v = [float(p.grad.abs().max()) for n, p in net.named_parameters() if any(n.startswith(x) for x in pref)]
        out[g] = max(v) if v else 0.0
    net.zero_grad()
    return out


class Wrap(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, s):
        return self.net(x, s), self.net.smoothness()


def hessian_eigs(net, bd, used):
    w = Wrap(net)
    named = [(n, p.detach().clone()) for n, p in w.named_parameters()]
    pieces, meta = [], []
    for n, p in named:
        v = p[used] if n.endswith("emb.weight") else p
        pieces.append(v.reshape(-1)); meta.append((n, v.shape, v.numel()))
    flat0 = torch.cat(pieces)
    full = dict(named)
    xd, sd, xc, y = bd["x"], bd["s"], bd["x_co"], bd["y"]

    def loss(v):
        prm, o = {}, 0
        for n, shp, k in meta:
            t = v[o:o + k].reshape(shp); o += k
            prm[n] = full[n].index_put((torch.tensor(used),), t) if n.endswith("emb.weight") else t
        pred, pen = functional_call(w, prm, (xd, sd))
        return F.mse_loss(xc * pred[:, 0], y) + SMOOTH * pen

    H = torch.autograd.functional.hessian(loss, flat0).numpy()
    ev = np.linalg.eigvalsh((H + H.T) / 2)
    return ev, flat0.numel()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    tr, te, te_anc, D, _ = setup()
    names, d = D["cols"], D["d_num"]
    m = D["train"]
    Ztr, str_ = D["Z"][m], D["s"][m]
    b = subset(D["batch"], m)
    bd = dbl(b)
    y = b["y"].numpy()
    used = np.unique(str_).tolist()
    log("DISCREPANCIES / choices (files win; see rank_prereg.md):")
    log("  X1 arm F constrains ALL 72 columns incl. invT and molal -> I5's 'outside the affected blocks' holds for arm G only")
    log("  X3 arms' max|grad| is w.r.t. their own parameters (z + others); full theta-space gradient also reported")
    log("  X5 init = projection of final.py's init onto P (removed components start at 0)")
    log("  X6 Hessian excludes the 16 embedding params of salts absent from training rows")

    # ---------------- Step 0
    log("\n==================== STEP 0: subspaces (no training) ====================")
    M = design_matrix(Ztr)
    co_idx = [names.index(n) for n in CO]
    co_cols = np.concatenate([np.arange(i * 9, (i + 1) * 9) for i in co_idx])
    PG, PF = retained(M[:, co_cols]), retained(M)
    log(f"  arm G: co-solvent block {len(co_cols)} columns -> retained {PG.shape[1]} (expected 14){'' if PG.shape[1] == 14 else '  <-- FLAG'}")
    log(f"  arm F: full M {M.shape[1]} columns -> retained {PF.shape[1]} (expected 36){'' if PF.shape[1] == 36 else '  <-- FLAG'}")
    arms = {"G": (PG, co_idx), "F": (PF, list(range(d)))}
    RESULTS["step0"] = dict(dimG=PG.shape[1], dimF=PF.shape[1])

    # sanity: identity P reproduces kan.train (seed 0)
    ident = train_rank(D, np.eye(72), list(range(d)), 0, b)
    ref = train([6, 1], d, D["n_salt"], delta_head, b, epochs=EPOCHS, seed=0)[0]
    diff = np.abs(predict(ident, delta_head, b) - predict(ref, delta_head, b)).max()
    log(f"  check: RankKAN with P = I reproduces kan.train seed 0: max |delta diff| = {diff:.2e}")

    # ---------------- Step 1
    log("\n==================== STEP 1: constrained models (train 107 epochs, then certify) ====================")
    W_DIR.mkdir(exist_ok=True)
    models = {}
    for arm, (P, aff) in arms.items():
        meta_p = W_DIR / f"rank_{arm}_meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        models[arm] = {}
        for s in SEEDS:
            p = W_DIR / f"rank_{arm}_s{s}.pt"
            if p.exists() and str(s) in meta:
                net = RankKAN(d, D["n_salt"], P, aff, s).double()
                net.load_state_dict(torch.load(p))
            else:
                net0 = train_rank(D, P, aff, s, b)
                net, info = certify(net0, b)
                info.pop("trace", None)
                torch.save(net.state_dict(), p)
                meta[str(s)] = info
                meta_p.write_text(json.dumps(meta))
                log(f"    arm {arm} seed {s}: {info['status']} after {info['iters']} iters, loss {info['loss']:.8e}, "
                    f"max|grad| {info['grad']:.2e}, rel {info['rel']:.1e}")
            net.eval()
            kf = net.to_kan(D["n_salt"])
            eq = np.abs(predict(kf, delta_head, b) - predict(copy.deepcopy(net).float(), delta_head, b)).max()
            models[arm][s] = dict(net=net, kan=kf, meta=meta[str(s)], eq=float(eq))
    base = {}
    for s in SEEDS:
        k = KAN(d, [6, 1], D["n_salt"]); k.load_state_dict(torch.load(W_DIR / f"fiber2_ind_{s}.pt")); k.eval()
        base[s] = dict(net=copy.deepcopy(k).double(), kan=k)
    log("  equivalence (constrained vs converted KAN), max |delta diff|: " +
        ", ".join(f"{a}{s} {models[a][s]['eq']:.1e}" for a in arms for s in SEEDS))

    # ---------------- Step 2
    log("\n==================== STEP 2: comparison (baseline = fiber2 capped independent seeds) ====================")
    sd_y = y.std()
    groups = {"z": ["z"], "other": ["bias0", "bw_free", "coef_free", "emb", "l1"]}
    stats = {"base": {s: grad_stats(base[s]["net"], bd) for s in SEEDS}}
    for arm in arms:
        stats[arm] = {}
        for s in SEEDS:
            g = grad_stats(models[arm][s]["net"], bd, groups)
            kt = copy.deepcopy(models[arm][s]["kan"]).double()
            g["theta_grad"] = grad_stats(kt, bd)["grad"]
            stats[arm][s] = g
    log(f"  {'model':6s} {'seed':>4s} {'status':>12s} {'iters':>6s} {'final loss':>14s} {'max|grad|':>10s} {'z':>9s} {'other':>9s} {'theta-space':>11s} {'rel(50 it)':>10s}")
    for s in SEEDS:
        x = stats["base"][s]
        log(f"  {'base':6s} {s:4d} {'(fiber2)':>12s} {'':>6s} {x['loss']:14.8e} {x['grad']:10.2e} {'':>9s} {'':>9s} {x['grad']:11.2e}")
    for arm in arms:
        for s in SEEDS:
            x, mt = stats[arm][s], models[arm][s]["meta"]
            log(f"  {'arm ' + arm:6s} {s:4d} {mt['status']:>12s} {mt['iters']:6d} {x['loss']:14.8e} {x['grad']:10.2e} {x['z']:9.2e} "
                f"{x['other']:9.2e} {x['theta_grad']:11.2e} {mt['rel']:10.1e}")
    conv = {arm: sum(models[arm][s]["meta"]["status"] == "converged" for s in SEEDS) for arm in arms}
    log(f"  converged: " + ", ".join(f"arm {a} {c}/6" for a, c in conv.items()))

    dAB, ecorr = {}, {}
    sweeps = grid_inputs(D)
    for tag, group in [("base", base)] + [(a, models[a]) for a in arms]:
        dl = {s: predict(group[s]["kan"], delta_head, b) for s in SEEDS}
        dAB[tag] = [float(np.sqrt(np.mean((dl[A] - dl[B]) ** 2)) / sd_y) for A, B in itertools.combinations(SEEDS, 2)]
        vals = []
        for A, B in itertools.combinations(SEEDS, 2):
            kA, kB = group[A]["kan"], group[B]["kan"]
            cA, cB = parts(kA, Ztr, str_, d)[2], parts(kB, Ztr, str_, d)[2]
            Cm = corr_matrix(cA - cA.mean(0), cB - cB.mean(0))
            r, k = linear_sum_assignment(1 - np.abs(Cm))
            for j, kk in zip(r, k):
                tA, tB = theta(kA, j, d), theta(kB, kk, d)
                (a, _), *_ = np.linalg.lstsq(np.column_stack([M @ tA, np.ones(len(M))]), M @ tB, rcond=None)
                for nm in EDGE_INPUTS:
                    i = names.index(nm)
                    eA = parts(kA, *sweeps[i], d)[0][i][:, j]; eB = parts(kB, *sweeps[i], d)[0][i][:, kk]
                    c = np.corrcoef(eA, eB)[0, 1] if eA.std() > 0 and eB.std() > 0 else 0.0
                    vals.append(float(np.sign(a) * c))
        ecorr[tag] = vals
    med = lambda v: float(np.median(v))
    base_grad = med([stats["base"][s]["grad"] for s in SEEDS]); base_loss = med([stats["base"][s]["loss"] for s in SEEDS])
    log(f"\n  {'':6s} {'median max|grad|':>17s} {'ratio':>7s} {'median d_AB':>12s} {'ratio':>7s} {'median loss':>13s} {'ratio':>7s} {'edge corr (invT, molal)':>24s}")
    summ = {}
    for tag in ["base", "G", "F"]:
        g = med([stats[tag][s]["grad"] for s in SEEDS]); L = med([stats[tag][s]["loss"] for s in SEEDS])
        summ[tag] = dict(grad=g, grad_ratio=g / base_grad, dAB=med(dAB[tag]), dAB_ratio=med(dAB[tag]) / med(dAB["base"]),
                         loss=L, loss_ratio=L / base_loss, ecorr=med(ecorr[tag]))
        x = summ[tag]
        log(f"  {tag:6s} {x['grad']:17.2e} {x['grad_ratio']:7.3f} {x['dAB']:12.4f} {x['dAB_ratio']:7.3f} {x['loss']:13.6e} {x['loss_ratio']:7.3f} {x['ecorr']:24.3f}")

    log("\n  Hessian (float64, own parameters; unused salt embeddings excluded):")
    hess = {}
    for tag, group in [("base", base)] + [(a, models[a]) for a in arms]:
        hess[tag] = {}
        for s in SEEDS:
            ev, npar = hessian_eigs(group[s]["net"], bd, used)
            small = ev[:10]; near0 = ev[np.argsort(np.abs(ev))[:10]]
            hess[tag][s] = dict(n=npar, smallest=small.tolist(), near_zero=near0.tolist(), n_neg=int((ev < 0).sum()), max=float(ev[-1]))
            log(f"    {tag:4s} seed {s}: {npar} params, {int((ev < 0).sum())} negative, max {ev[-1]:.2e}; "
                f"10 smallest: " + " ".join(f"{v:.1e}" for v in small) + " | 10 nearest 0 (|eig| median " +
                f"{np.median(np.abs(near0)):.1e})")

    # ---------------- Step 3
    log("\n==================== STEP 3: fold A accuracy ====================")
    ho = (tr.combo == "EA+PC").values & m
    dt = m & ~ho
    Mf = design_matrix(D["Z"][dt])
    PGf, PFf = retained(Mf[:, co_cols]), retained(Mf)
    log(f"  fold A training rows {dt.sum()}; P recomputed: arm G {PGf.shape[1]} dims, arm F {PFf.shape[1]} dims")
    bt, bh = subset(D["batch"], dt), subset(D["batch"], ho)
    yh = tr.log_k.values[ho]; anc = tr.anchor.values[ho]
    acc = {}
    pk = np.mean([predict(train([6, 1], d, D["n_salt"], delta_head, bt, epochs=EPOCHS, seed=s)[0], delta_head, bh) for s in FOLD_SEEDS], 0)
    acc["KAN-delta"] = (rmse(anc + pk, yh), float(np.mean(anc + pk - yh)))
    for arm, (P, aff) in [("G", (PGf, co_idx)), ("F", (PFf, list(range(d))))]:
        pa = np.mean([predict(train_rank(D, P, aff, s, bt), delta_head, bh) for s in FOLD_SEEDS], 0)
        acc[arm] = (rmse(anc + pa, yh), float(np.mean(anc + pa - yh)))
    for k, (r_, b_) in acc.items():
        log(f"  {k:10s} fold A RMSE {r_:.4f}, bias {b_:+.4f}")
    log(f"  (KAN-delta reference in the brief: 0.142; recomputed here: {acc['KAN-delta'][0]:.4f})")

    # ---------------- verdicts
    v = {}
    for arm in arms:
        x = summ[arm]
        v[f"I1-{arm}"] = ("held" if x["grad_ratio"] <= 0.1 else "failed") + f" (ratio {x['grad_ratio']:.3f})"
        v[f"I2-{arm}"] = ("held" if x["dAB_ratio"] <= 0.5 else "failed") + f" (ratio {x['dAB_ratio']:.3f})"
        v[f"I3-{arm}"] = ("passed" if x["loss_ratio"] <= 1.10 else "FAILED sanity") + f" (ratio {x['loss_ratio']:.3f})"
        v[f"I4-{arm}"] = ("held" if acc[arm][0] <= 0.150 else "failed") + f" (RMSE {acc[arm][0]:.4f})"
        v[f"I5-{arm}"] = ("held" if x["ecorr"] >= summ["base"]["ecorr"] + 0.1 else "failed") + \
            f" ({x['ecorr']:.3f} vs baseline {summ['base']['ecorr']:.3f} + 0.1)"
    i6 = summ["F"]["grad_ratio"] <= summ["G"]["grad_ratio"] and summ["F"]["dAB_ratio"] <= summ["G"]["dAB_ratio"]
    v["I6"] = ("held" if i6 else "failed") + f" (grad ratio F {summ['F']['grad_ratio']:.3f} vs G {summ['G']['grad_ratio']:.3f}; " \
        f"d_AB ratio F {summ['F']['dAB_ratio']:.3f} vs G {summ['G']['dAB_ratio']:.3f})"
    log("\nVERDICTS: " + "; ".join(f"{k}: {x}" for k, x in v.items()))
    RESULTS.update(summary=summ, stats={t: {str(s): x for s, x in st.items()} for t, st in stats.items()},
                   dAB=dAB, ecorr=ecorr, hessian={t: {str(s): x for s, x in h.items()} for t, h in hess.items()},
                   accuracy=acc, converged=conv, verdicts=v, foldA_dims=dict(G=PGf.shape[1], F=PFf.shape[1]))
    (ROOT / "rank_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))

    # ---------------- figures
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.5))
    tags = ["base", "G", "F"]
    axs[0].bar(tags, [summ[t]["grad"] for t in tags]); axs[0].set_yscale("log"); axs[0].set_title("median max|grad| at cap")
    axs[1].boxplot([dAB[t] for t in tags], labels=tags); axs[1].set_title("d_AB on delta (15 pairs)")
    axs[2].boxplot([ecorr[t] for t in tags], labels=tags); axs[2].set_title("edge corr, 1000/T & molality")
    for t in tags:
        axs[3].semilogy(sorted(np.abs(np.concatenate([hess[t][s]["near_zero"] for s in SEEDS]))), label=t)
    axs[3].set_title("|eig| of the 10 nearest-zero Hessian eigenvalues (6 seeds)"); axs[3].legend()
    fig.tight_layout(); fig.savefig(FIG / "rank_comparison.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()

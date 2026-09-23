"""Phase 3: FastKAN-style models in pure PyTorch (CPU). Run: python3 kan.py

Model 1 (KAN-delta): log_k = pure-PC anchor + x_co * h(features)   [anchored rows]
Model 2 (VTF-KAN):   log_k = logA + log10 c + alpha c/ln10 - B/((T-T0) ln10), params from a KAN  [fallback rows]
"""
import copy
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from anchor import DELTA_COLS, LOGK_MIN, fit_predict_delta, prepare, rmse
from pipeline import PROPS, TEST_SALTS, fit_predict

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
ROOT = Path(__file__).parent
FIGS = ROOT / "figs"
SEEDS = [0, 1, 2, 3, 4]
N_RBF, SMOOTH, LR, WD, MAX_EP = 8, 1e-3, 3e-3, 1e-4, 400
LN10 = math.log(10)
DELTA_IN = ["x_co", "eps_co", "lneta_co", "M_co", "mix_eps", "mix_lneta", "invT", "molal"]
VTF_IN = ["mix_eps", "mix_lneta", "x_linear", "mix_M", "molal"]


# ------------------------------------------------------------------ KAN
class KANLayer(nn.Module):
    """Each edge: w_base * SiLU(x) + sum_k c_k exp(-((x - g_k)/h)^2), grid on [-1, 1]."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.register_buffer("grid", torch.linspace(-1, 1, N_RBF))
        self.h = 2 / (N_RBF - 1)
        self.base = nn.Linear(d_in, d_out)
        self.coef = nn.Parameter(torch.randn(d_out, d_in, N_RBF) * 0.1)

    def rbf(self, x):
        return torch.exp(-(((x[..., None] - self.grid) / self.h) ** 2))

    def forward(self, x):
        return self.base(F.silu(x)) + torch.einsum("bik,oik->bo", self.rbf(x), self.coef)

    def smoothness(self):
        d2 = self.coef[..., 2:] - 2 * self.coef[..., 1:-1] + self.coef[..., :-2]
        return (d2 ** 2).sum()

    def edge(self, i, x):
        """Edge functions from input i to every output, bias excluded. x: (n,) -> (n, d_out)."""
        return F.silu(x)[:, None] * self.base.weight[:, i] + self.rbf(x) @ self.coef[:, i, :].T


class KAN(nn.Module):
    def __init__(self, d_num, widths, n_salt):
        super().__init__()
        self.emb = nn.Embedding(n_salt, 2)
        dims = [d_num + 2] + widths
        self.layers = nn.ModuleList(KANLayer(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x, salt):
        z = torch.cat([x, torch.tanh(self.emb(salt))], 1)  # tanh keeps the embedding on the RBF grid
        for i, layer in enumerate(self.layers):
            z = layer(torch.tanh(z) if i else z)  # hidden activations squashed onto [-1, 1] before the next grid
        return z

    def smoothness(self):
        return sum(l.smoothness() for l in self.layers)


def delta_head(net, b):
    return b["x_co"] * net(b["x"], b["s"])[:, 0]


def vtf_params(net, b, min_T):
    r = net(b["x"], b["s"])
    logA = r[:, 0] + b["logA0"]  # fixed data-driven offset so 400 AdamW steps can reach the right level
    alpha = -F.softplus(r[:, 1])
    B = 1000 * F.softplus(r[:, 2])
    T0 = 60 + (min_T - 70) * torch.sigmoid(r[:, 3])
    return logA, alpha, B, T0


def make_vtf_head(min_T):
    def head(net, b):
        logA, alpha, B, T0 = vtf_params(net, b, min_T)
        c = b["c"]
        return logA + torch.log10(c) + alpha * c / LN10 - B / ((b["T"] - T0) * LN10)
    return head


def train(widths, d_num, n_salt, head, btr, bva=None, epochs=MAX_EP, seed=0):
    torch.manual_seed(seed)
    net = KAN(d_num, widths, n_salt)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    best = (float("inf"), epochs, None)
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = F.mse_loss(head(net, btr), btr["y"]) + SMOOTH * net.smoothness()
        loss.backward()
        opt.step()
        if bva is not None:
            with torch.no_grad():
                r = float(torch.sqrt(F.mse_loss(head(net, bva), bva["y"])))
            if r < best[0]:
                best = (r, ep, copy.deepcopy(net.state_dict()))
    if bva is not None:
        net.load_state_dict(best[2])
    return net, best[1]


def predict(net, head, b):
    with torch.no_grad():
        return head(net, b).numpy()


# ------------------------------------------------------------------ features
def co_props(df):
    """Mole-weighted properties of the non-PC solvents (co-solvent block)."""
    out = {"eps_co": [], "lneta_co": [], "M_co": []}
    for names, fr in zip(df.solvents.str.split(";"), df.solvent_fracs_mol.str.split(";")):
        pairs = [(n, float(x)) for n, x in zip(names, fr) if n != "PC" and n in PROPS]
        tot = sum(x for _, x in pairs)
        for key, i, f in [("eps_co", 2, lambda v: v), ("lneta_co", 3, np.log), ("M_co", 0, lambda v: v)]:
            out[key].append(sum(x * f(PROPS[n][i]) for n, x in pairs) / tot if tot > 0 else np.nan)
    return pd.DataFrame(out, index=df.index)


class Scaler:
    def __init__(self, frames, cols):
        allv = pd.concat([f[cols] for f in frames])
        self.cols, self.lo, self.hi = cols, allv.min().values, allv.max().values

    def __call__(self, f):
        return 2 * (f[self.cols].values - self.lo) / (self.hi - self.lo) - 1

    def inv(self, i, z):
        return (z + 1) / 2 * (self.hi[i] - self.lo[i]) + self.lo[i]


def to_batch(Z, salt_idx, **extra):
    b = {"x": torch.tensor(Z, dtype=torch.float32), "s": torch.tensor(salt_idx, dtype=torch.long)}
    for k, v in extra.items():
        b[k] = torch.tensor(np.asarray(v, float), dtype=torch.float32)
    return b


def subset(b, m):
    m = torch.tensor(np.asarray(m))
    return {k: v[m] for k, v in b.items()}


def table(y, preds, groups, stds=None):
    rows = []
    for gname, m in groups:
        r = {"subset": gname, "n": int(m.sum())}
        for name, p in preds.items():
            r[f"{name}_rmse"] = rmse(p[m], y[m])
            r[f"{name}_bias"] = float(np.mean(p[m] - y[m]))
        for name, s in (stds or {}).items():
            r[f"{name}_seedstd"] = float(np.mean(s[m]))
        rows.append(r)
    return rows


def main():
    tr, te, ss, X, Xte, ok, use, te_anc, has = prepare(verbose=False)
    gbm_cols = list(X.columns)  # Phase 2 GBM features (+ redundant x_co, as in anchor.py)
    salts = sorted(set(tr.salt_name) | set(te.salt_name))
    sidx = {s: i for i, s in enumerate(salts)}
    for f, df in [(X, tr), (Xte, te)]:
        f[["eps_co", "lneta_co", "M_co"]] = co_props(df)
        f["x_linear"] = 1 - f.frac_high_eps
    s_tr, s_te = tr.salt_name.map(sidx).values, te.salt_name.map(sidx).values
    min_T = float(min(tr.temperature_K.min(), te.temperature_K.min()))

    # ---------------- Model 1: KAN-delta on anchored rows
    sc1 = Scaler([X[use], Xte[has]], DELTA_IN)
    B1 = to_batch(sc1(X), s_tr, x_co=X.x_co, y=tr.delta.fillna(0))
    B1te = to_batch(sc1(Xte), s_te, x_co=Xte.x_co)

    # ---------------- Model 2: VTF-KAN on all usable train rows
    sc2 = Scaler([X[ok], Xte], VTF_IN)
    a0, B0, T00 = -math.log(2), 1000 * math.log(2), 60 + (min_T - 70) * 0.5  # values at r = 0
    c_tr, c_te = X.molal.values, Xte.molal.values
    logA0 = float(np.mean((tr.log_k - (np.log10(c_tr) + a0 * c_tr / LN10 - B0 / ((tr.temperature_K - T00) * LN10)))[ok]))
    B2 = to_batch(sc2(X), s_tr, c=c_tr, T=tr.temperature_K, y=tr.log_k, logA0=np.full(len(tr), logA0))
    B2te = to_batch(sc2(Xte), s_te, c=c_te, T=te.temperature_K, logA0=np.full(len(te), logA0))
    vtf_head = make_vtf_head(min_T)

    print("pre-registered: KAN-delta fold A RMSE <= 0.175 (HGB-delta); |bias| on fold B T<250 < 0.204 (HGB-delta)\n")
    folds, best_ep1, best_ep2 = [], [], []
    Xk, yk = X.loc[ok, gbm_cols], tr.log_k[ok]
    for fold, combo in [("A: PC+EA", "EA+PC"), ("B: EC+PC", "EC+PC")]:
        ho_all = ((tr.combo == combo) & ok).values
        ho = ho_all & use.values
        dtrain = use.values & ~ho_all
        train_m = ok.values & ~ho_all
        gbm = np.full(len(tr), np.nan)
        gbm[ho_all] = fit_predict(Xk[~ho_all[ok.values]], yk[~ho_all[ok.values]], np.ones((~ho_all[ok.values]).sum()), X.loc[ho_all, gbm_cols])
        hgb_d = tr.anchor.values[ho] + fit_predict_delta(X.loc[dtrain, DELTA_COLS], tr.delta[dtrain], X.loc[ho, DELTA_COLS])
        kp, vp = [], []
        for s in SEEDS:  # early stop on this fold (as specified; optimistic)
            net, ep = train([6, 1], len(DELTA_IN), len(salts), delta_head, subset(B1, dtrain), subset(B1, ho), seed=s)
            kp.append(tr.anchor.values[ho] + predict(net, delta_head, subset(B1, ho)))
            best_ep1.append(ep)
            net, ep = train([8, 4], len(VTF_IN), len(salts), vtf_head, subset(B2, train_m), subset(B2, ho_all), seed=s)
            vp.append(predict(net, vtf_head, subset(B2, ho_all)))
            best_ep2.append(ep)
        folds.append(dict(fold=fold, ho_all=ho_all, ho=ho, dtrain=dtrain, train_m=train_m, gbm=gbm, hgb_d=hgb_d,
                          kp=np.array(kp), vp=np.array(vp)))
        print(f"{fold} done (KAN-delta best epochs {best_ep1[-5:]}, VTF-KAN best epochs {best_ep2[-5:]})", flush=True)

    # fixed-epoch refits (median best epoch, no peeking at the fold): the honest number, matches the final model
    ep1, ep2 = int(np.median(best_ep1)), int(np.median(best_ep2))
    print(f"median best epochs: KAN-delta {ep1}, VTF-KAN {ep2}")
    y = tr.log_k.values
    anchored_rows, fallback_rows = [], []
    for f in folds:
        ho, ho_all = f["ho"], f["ho_all"]
        kf = np.array([tr.anchor.values[ho] + predict(train([6, 1], len(DELTA_IN), len(salts), delta_head, subset(B1, f["dtrain"]), epochs=ep1, seed=s)[0], delta_head, subset(B1, ho)) for s in SEEDS])
        vf = np.array([predict(train([8, 4], len(VTF_IN), len(salts), vtf_head, subset(B2, f["train_m"]), epochs=ep2, seed=s)[0], vtf_head, subset(B2, ho_all)) for s in SEEDS])
        for m_rows, preds, stds, out in [
            (ho, {"GBM": f["gbm"][ho], "HGBd": f["hgb_d"], "KANd_es": f["kp"].mean(0), "KANd": kf.mean(0),
                  "blendHK": 0.5 * (f["hgb_d"] + kf.mean(0))}, {"KANd": kf.std(0)}, anchored_rows),
            (ho_all, {"GBM": f["gbm"][ho_all], "VTF_es": f["vp"].mean(0), "VTFKAN": vf.mean(0)}, {"VTFKAN": vf.std(0)}, fallback_rows)]:
            g = [("all", np.ones(m_rows.sum(), bool)), ("T<250", tr.temperature_K.values[m_rows] < 250)]
            g += [(s, tr.salt_name.values[m_rows] == s) for s in TEST_SALTS if (tr.salt_name.values[m_rows] == s).any()]
            out += [{"fold": f["fold"], **r} for r in table(y[m_rows], preds, g, stds)]

    at = pd.DataFrame(anchored_rows)
    ft = pd.DataFrame(fallback_rows)
    print("\nAnchored held-out rows (log10 RMSE / bias). KANd = fixed-epoch (honest); KANd_es = early-stopped on the fold (optimistic);")
    print("KANd_seedstd = mean per-row std over 5 seeds; blendHK uses fixed-epoch KANd")
    print(at.round(4).to_string(index=False))
    print("\nAll held-out rows, fallback models (GBM vs VTF-KAN; VTFKAN = fixed-epoch, VTF_es = early-stopped):")
    print(ft.round(4).to_string(index=False))

    a_all = at[(at.fold == "A: PC+EA") & (at.subset == "all")].iloc[0]
    b_cold = at[(at.fold == "B: EC+PC") & (at.subset == "T<250")].iloc[0]
    held1 = a_all.KANd_rmse <= a_all.HGBd_rmse
    held2 = abs(b_cold.KANd_bias) < abs(b_cold.HGBd_bias)
    print(f"\nPRE-REG 1 (KAN-delta fold A <= HGB-delta {a_all.HGBd_rmse:.4f}): KAN {a_all.KANd_rmse:.4f} -> {'HELD' if held1 else 'FAILED'}")
    print(f"PRE-REG 2 (|bias| fold B T<250 < HGB-delta {abs(b_cold.HGBd_bias):.4f}): KAN {abs(b_cold.KANd_bias):.4f} -> {'HELD' if held2 else 'FAILED'}")

    # fallback choice: mean of fold A and fold B RMSE over all held-out rows
    fa = ft[ft.subset == "all"]
    use_vtf = fa.VTFKAN_rmse.mean() < fa.GBM_rmse.mean()
    print(f"fallback choice (mean fold A/B RMSE): GBM {fa.GBM_rmse.mean():.4f} vs VTF-KAN {fa.VTFKAN_rmse.mean():.4f} -> {'VTF-KAN' if use_vtf else 'GBM'}")

    # ---------------- final models (median best epoch from the folds)
    finals1 = [train([6, 1], len(DELTA_IN), len(salts), delta_head, subset(B1, use.values), epochs=ep1, seed=s)[0] for s in SEEDS]
    finals2 = [train([8, 4], len(VTF_IN), len(salts), vtf_head, subset(B2, ok.values), epochs=ep2, seed=s)[0] for s in SEEDS]

    kan_te = np.array([predict(n, delta_head, subset(B1te, has)) for n in finals1])
    vtf_te = np.array([predict(n, vtf_head, subset(B2te, ~has)) for n in finals2])
    gbm_te = pd.read_csv(ROOT / "submission_logk.csv").log_k.values
    hgb_anchor = pd.read_csv(ROOT / "submission_anchor.csv").log_k.values
    fallback = vtf_te.mean(0) if use_vtf else gbm_te[~has]

    kan_log, kb_log = np.empty(len(te)), np.empty(len(te))
    kan_log[has] = te_anc[has] + kan_te.mean(0)
    kb_log[has] = 0.5 * (hgb_anchor[has] + kan_log[has])
    kan_log[~has] = kb_log[~has] = fallback
    flag = "" if held1 else "  [FLAG: KAN-delta did NOT beat HGB-delta on fold A]"
    for name, v in [("submission_kan.csv", kan_log), ("submission_kanblend.csv", kb_log)]:
        out = pd.DataFrame({"id": te.id, "log_k": np.clip(v, LOGK_MIN, None)})
        assert out.id.equals(ss.id) and list(out.columns) == list(ss.columns) and out.log_k.notna().all()
        out.to_csv(ROOT / name, index=False)
        print(f"wrote {name}{flag}")
    print(f"test anchored rows: {has.sum()}, fallback rows: {(~has).sum()} ({'VTF-KAN' if use_vtf else 'GBM'})")
    print("mean(KAN-delta - HGB-delta) on anchored test rows by salt:",
          pd.Series(kan_log[has] - hgb_anchor[has]).groupby(te.salt_name.values[has]).mean().round(3).to_dict())
    print(f"fallback rows: mean(VTF-KAN - GBM) = {np.mean(vtf_te.mean(0) - gbm_te[~has]):.3f}, "
          f"rmse between them = {rmse(vtf_te.mean(0), gbm_te[~has]):.3f}")

    # ---------------- figures
    # 1. delta vs x_DEC for binary PC+DEC, ~1 mol/kg
    xg = np.linspace(0.0, 0.7, 71)
    pc, dec = PROPS["PC"], PROPS["DEC"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, salt in zip(axs, TEST_SALTS):
        for T, col in [(233, "tab:blue"), (273, "tab:green"), (313, "tab:red")]:
            f = pd.DataFrame({"x_co": xg, "eps_co": dec[2], "lneta_co": np.log(dec[3]), "M_co": dec[0],
                              "mix_eps": (1 - xg) * pc[2] + xg * dec[2],
                              "mix_lneta": (1 - xg) * np.log(pc[3]) + xg * np.log(dec[3]),
                              "invT": 1000 / T, "molal": 1.0})
            b = to_batch(sc1(f), np.full(len(xg), sidx[salt]), x_co=xg)
            P = np.array([predict(n, delta_head, b) for n in finals1])
            ax.plot(xg, P.mean(0), color=col, label=f"{T} K")
            ax.fill_between(xg, P.min(0), P.max(0), color=col, alpha=0.2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{salt}, 1 mol/kg"); ax.set_xlabel("DEC mole fraction")
    axs[0].set_ylabel("predicted Δlog10 k vs pure PC"); axs[0].legend()
    fig.suptitle("KAN-delta: conductivity gain from DEC (band = min/max over 5 seeds)")
    fig.tight_layout(); fig.savefig(FIGS / "kan_delta_vs_xDEC.png", dpi=120); plt.close(fig)

    # 2/3. first-layer edges of h, centred to zero mean over the training rows
    Ztr = torch.tensor(sc1(X[use]), dtype=torch.float32)
    for feat, fname, xlabel in [("invT", "kan_edge_T.png", "1000/T (1/K)"), ("lneta_co", "kan_edge_eta.png", "ln η co-solvent (cP)")]:
        i = DELTA_IN.index(feat)
        zg = torch.linspace(-1, 1, 200)
        fig, axs = plt.subplots(1, len(SEEDS), figsize=(4 * len(SEEDS), 3.4), sharey=True)
        for ax, net, s in zip(axs, finals1, SEEDS):
            with torch.no_grad():
                e = net.layers[0].edge(i, zg).numpy() - net.layers[0].edge(i, Ztr[:, i]).numpy().mean(0)
            ax.plot(sc1.inv(i, zg.numpy()), e)
            ax.set_title(f"seed {s}"); ax.set_xlabel(xlabel)
            if feat == "lneta_co":
                ax.axvline(np.log(dec[3]), color="k", ls="--", lw=0.8)
                ax.text(np.log(dec[3]), ax.get_ylim()[1] * 0.9, " DEC", fontsize=8)
        axs[0].set_ylabel("edge φ(x), centred")
        fig.suptitle(f"KAN-delta layer-1 edges for {feat} (6 hidden units per seed; hidden units are not aligned across seeds)")
        fig.tight_layout(); fig.savefig(FIGS / fname, dpi=120); plt.close(fig)

    # 4. VTF T0 per salt; flag narrow-T series
    with torch.no_grad():
        bt = subset(B2, ok.values)
        T0s = np.array([vtf_params(n, bt, min_T)[3].numpy() for n in finals2])
        Bs = np.array([vtf_params(n, bt, min_T)[2].numpy() for n in finals2])
    d = tr[ok].assign(T0=T0s.mean(0), T0_sd=T0s.std(0), B=Bs.mean(0), B_sd=Bs.std(0))
    key = ["source_doi", "salt_name", "solvents", "solvent_fracs"]
    span = d.groupby(key).temperature_K.transform(lambda t: t.max() - t.min())
    nT = d.groupby(key).temperature_K.transform(lambda t: t.round(0).nunique())
    d["narrow"] = (span < 40) | (nT < 4)
    order = d.groupby("salt_name").size().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for j, s in enumerate(order):
        g = d[d.salt_name == s]
        ax.boxplot(g.T0[~g.narrow], positions=[j], widths=0.5, showfliers=False)
        ax.scatter(np.full(g.narrow.sum(), j) + np.random.uniform(-0.15, 0.15, g.narrow.sum()), g.T0[g.narrow], s=6, c="tab:red", alpha=0.6)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=30)
    ax.set_ylabel("learned T0 (K)"); ax.set_title("VTF-KAN: T0 per salt (box = series spanning ≥40 K; red = narrow-T series, poorly constrained)")
    fig.tight_layout(); fig.savefig(FIGS / "vtf_T0_by_salt.png", dpi=120); plt.close(fig)
    print("\nVTF identifiability: median seed-to-seed std, narrow vs wide-T series:")
    print(d.groupby("narrow")[["T0_sd", "B_sd"]].median().round(1).rename(index={True: "narrow", False: "wide"}).to_string())
    print("narrow-T rows:", int(d.narrow.sum()), "of", len(d))
    print("T0 by salt (mean over rows):", d.groupby("salt_name").T0.mean().round(1).to_dict())


if __name__ == "__main__":
    main()

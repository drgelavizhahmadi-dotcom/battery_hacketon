"""Phase 4: GP and MLP on the KAN-delta setup (same rows, folds, delta = x_co * h). Run: python3 compare.py"""
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from anchor import LOGK_MIN, prepare, rmse
from kan import DELTA_IN, LR, MAX_EP, SEEDS, WD, Scaler, co_props, delta_head, predict, subset, table, to_batch, train
from pipeline import TEST_SALTS

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
ROOT = Path(__file__).parent
KAN_EPOCHS = 107  # median best epoch from kan.py
X_CO_MIN = 0.05


class MLP(nn.Module):
    def __init__(self, d_num, n_salt):
        super().__init__()
        self.emb = nn.Embedding(n_salt, 2)
        self.net = nn.Sequential(nn.Linear(d_num + 2, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x, salt):
        return self.net(torch.cat([x, torch.tanh(self.emb(salt))], 1))

    def smoothness(self):
        return torch.zeros(())


def train_mlp(d_num, n_salt, btr, bva=None, epochs=MAX_EP, seed=0):
    torch.manual_seed(seed)
    net = MLP(d_num, n_salt)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    best = (float("inf"), epochs, None)
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        F.mse_loss(delta_head(net, btr), btr["y"]).backward()
        opt.step()
        if bva is not None:
            with torch.no_grad():
                r = float(torch.sqrt(F.mse_loss(delta_head(net, bva), bva["y"])))
            if r < best[0]:
                best = (r, ep, copy.deepcopy(net.state_dict()))
    if bva is not None:
        net.load_state_dict(best[2])
    return net, best[1]


def gp_design(X, salt, salts):
    """Numeric DELTA_IN + salt one-hot (GP has no embedding)."""
    oh = pd.get_dummies(pd.Categorical(salt, categories=salts)).values.astype(float)
    return np.hstack([X[DELTA_IN].values, oh])


def fit_gp(Z, y):
    d = Z.shape[1]
    k = ConstantKernel(1.0, (1e-3, 1e2)) * RBF(np.ones(d), (1e-2, 1e3)) + WhiteKernel(0.1, (1e-5, 1e1))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1, random_state=0).fit(Z, y)


def main():
    tr, te, ss, X, Xte, ok, use, te_anc, has = prepare(verbose=False)
    salts = sorted(set(tr.salt_name) | set(te.salt_name))
    sidx = {s: i for i, s in enumerate(salts)}
    for f, df in [(X, tr), (Xte, te)]:
        f[["eps_co", "lneta_co", "M_co"]] = co_props(df)
    s_tr, s_te = tr.salt_name.map(sidx).values, te.salt_name.map(sidx).values
    sc = Scaler([X[use], Xte[has]], DELTA_IN)
    B = to_batch(sc(X), s_tr, x_co=X.x_co, y=tr.delta.fillna(0))
    gp_names = DELTA_IN + [f"salt={s}" for s in salts]
    G = gp_design(X, tr.salt_name, salts)
    Gte = gp_design(Xte, te.salt_name, salts)
    x_co = X.x_co.values
    y = tr.log_k.values
    anchor = tr.anchor.values
    print("pre-registered: GP within 0.01 of KAN-delta on fold A; MLP within 0.02\n")

    # pass 1: MLP best epochs (early stop on the fold) -> median, as done for KAN
    folds = []
    for fold, combo in [("A: PC+EA", "EA+PC"), ("B: EC+PC", "EC+PC")]:
        ho_all = ((tr.combo == combo) & ok).values
        folds.append((fold, ho_all & use.values, use.values & ~ho_all))
    mlp_eps = [train_mlp(len(DELTA_IN), len(salts), subset(B, dt), subset(B, ho), seed=s)[1] for _, ho, dt in folds for s in SEEDS]
    mlp_ep = int(np.median(mlp_eps))
    print(f"MLP best epochs {mlp_eps} -> fixed {mlp_ep}; KAN fixed {KAN_EPOCHS}")

    rows = []
    for fold, ho, dt in folds:
        kan = np.array([anchor[ho] + predict(train([6, 1], len(DELTA_IN), len(salts), delta_head, subset(B, dt), epochs=KAN_EPOCHS, seed=s)[0],
                                             delta_head, subset(B, ho)) for s in SEEDS])
        mlp = np.array([anchor[ho] + predict(train_mlp(len(DELTA_IN), len(salts), subset(B, dt), epochs=mlp_ep, seed=s)[0],
                                             delta_head, subset(B, ho)) for s in SEEDS])
        # GP on h = delta / x_co, standardized inputs from the training fold
        fit_m = dt & (x_co > X_CO_MIN)
        mu, sd = G[fit_m].mean(0), G[fit_m].std(0) + 1e-9
        gp = fit_gp((G[fit_m] - mu) / sd, tr.delta.values[fit_m] / x_co[fit_m])
        h_mean, h_std = gp.predict((G[ho] - mu) / sd, return_std=True)
        gp_pred, gp_std = anchor[ho] + x_co[ho] * h_mean, x_co[ho] * h_std
        g = [("all", np.ones(ho.sum(), bool)), ("T<250", tr.temperature_K.values[ho] < 250)]
        g += [(s, tr.salt_name.values[ho] == s) for s in TEST_SALTS if (tr.salt_name.values[ho] == s).any()]
        rows += [{"fold": fold, **r} for r in table(y[ho], {"KANd": kan.mean(0), "GP": gp_pred, "MLP": mlp.mean(0)}, g,
                                                    {"GP_pred": gp_std, "MLP": mlp.std(0), "KANd": kan.std(0)})]
        print(f"{fold} done; GP kernel: {gp.kernel_}", flush=True)

    t = pd.DataFrame(rows)
    cols = ["fold", "subset", "n"] + [f"{m}_{k}" for m in ["KANd", "GP", "MLP"] for k in ["rmse", "bias"]]
    print("\nAnchored held-out rows (log10 RMSE / bias):\n", t[cols].round(4).to_string(index=False))
    print("\nUncertainty (mean per row): GP predictive std, MLP/KAN seed std:\n",
          t[["fold", "subset", "GP_pred_seedstd", "MLP_seedstd", "KANd_seedstd"]].rename(columns={"GP_pred_seedstd": "GP_predstd"})
          .round(4).to_string(index=False))

    a = t[(t.fold == "A: PC+EA") & (t.subset == "all")].iloc[0]
    print(f"\nPRE-REG GP within 0.01 of KAN on fold A: GP {a.GP_rmse:.4f} vs KAN {a.KANd_rmse:.4f} (diff {a.GP_rmse - a.KANd_rmse:+.4f}) -> "
          f"{'HELD' if abs(a.GP_rmse - a.KANd_rmse) <= 0.01 or a.GP_rmse < a.KANd_rmse else 'FAILED'}")
    print(f"PRE-REG MLP within 0.02 of KAN on fold A: MLP {a.MLP_rmse:.4f} (diff {a.MLP_rmse - a.KANd_rmse:+.4f}) -> "
          f"{'HELD' if a.MLP_rmse - a.KANd_rmse <= 0.02 else 'FAILED'}")

    # full GP on all anchored rows: length scales (+ test prediction if it beat KAN on fold A)
    fit_m = use.values & (x_co > X_CO_MIN)
    mu, sd = G[fit_m].mean(0), G[fit_m].std(0) + 1e-9
    gp = fit_gp((G[fit_m] - mu) / sd, tr.delta.values[fit_m] / x_co[fit_m])
    ls = gp.kernel_.k1.k2.length_scale
    print(f"\nGP fit on all anchored rows ({fit_m.sum()} with x_co > {X_CO_MIN}): {gp.kernel_}")
    print("length scales (standardized inputs; short = relevant, >~1e2 = ignored):")
    for n, l in sorted(zip(gp_names, ls), key=lambda p: p[1]):
        print(f"  {n:14s} {l:9.3f}")

    if a.GP_rmse < a.KANd_rmse:
        h = gp.predict((Gte[has] - mu) / sd)
        final = pd.read_csv(ROOT / "submission_final.csv")
        v = final.log_k.values.copy()
        v[has] = te_anc[has] + Xte.x_co.values[has] * h
        out = pd.DataFrame({"id": ss.id, "log_k": np.clip(v, LOGK_MIN, None)})
        assert out.id.equals(te.id) and out.log_k.notna().all()
        out.to_csv(ROOT / "submission_gp.csv", index=False)
        print("GP beat KAN-delta on fold A -> wrote submission_gp.csv (GP on anchored rows, GBM fallback)")
    else:
        print("GP did not beat KAN-delta on fold A -> submission_gp.csv not written")


if __name__ == "__main__":
    main()

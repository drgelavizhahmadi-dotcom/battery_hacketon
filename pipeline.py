"""Phase 2: CALiSol-23 DEC holdout. HGB on log10(k) with physics features. Run: python3 pipeline.py"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
D = ROOT / "ca-li-sol-23-challenge"
pd.set_option("display.width", 200)

# M g/mol, density g/cm3, dielectric eps, viscosity cP (~25 C; EC at 40 C). User-supplied rows first;
# the rest are approximate literature values (TFP / MOEMC / FEC eps & eta are the least certain).
PROPS = {
    "PC": (102.09, 1.20, 64.9, 2.53), "DEC": (118.13, 0.97, 2.81, 0.75), "EA": (88.11, 0.90, 6.0, 0.43),
    "EC": (88.06, 1.32, 89.8, 1.90), "DMC": (90.08, 1.06, 3.11, 0.59), "EMC": (104.10, 1.01, 2.96, 0.65),
    "DME": (90.12, 0.87, 7.2, 0.46),
    "TFP": (344.07, 1.59, 11.0, 2.5), "MA": (74.08, 0.93, 6.68, 0.36), "FEC": (106.05, 1.45, 78.4, 4.1),
    "2-Glyme": (134.17, 0.94, 7.4, 0.99), "3-Glyme": (178.23, 0.99, 7.5, 1.96), "4-Glyme": (222.28, 1.01, 7.8, 3.39),
    "AN": (41.05, 0.78, 35.9, 0.34), "MOEMC": (134.13, 1.10, 9.0, 1.5), "2-MeTHF": (86.13, 0.85, 6.97, 0.46),
    "THF": (72.11, 0.89, 7.4, 0.46), "DMSO": (78.13, 1.10, 46.5, 1.99), "Sulfolane": (120.17, 1.26, 43.3, 10.3),
    "g-Butyrolactone": (86.09, 1.13, 39.0, 1.73), "DOL": (74.08, 1.06, 7.1, 0.59),
    "Methylene chloride": (84.93, 1.33, 8.93, 0.41), "Toluene": (92.14, 0.87, 2.38, 0.56), "DMF": (73.09, 0.94, 36.7, 0.80),
}
SAME_SOURCE = {"10.1149/1.1566019", "10.1149/1.1630593", "10.1149/1.1833611", "10.1149/1.1914757",
               "10.1021/JE020219O", "10.1021/je0498863"}
TEST_SALTS = ["LiBF4", "LiPF6", "LiBOB"]
T0_GRID = [130, 150, 170]
SEEDS = [0, 1, 2]
SANITY_T = [round(t, 1) for t in (213.95, 233.75, 253.35, 292.65, 332.15)]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def featurize(df, T0):
    names = df.solvents.str.split(";")
    xs = df.solvent_fracs_mol.str.split(";").apply(lambda l: np.array(l, float))
    ok = names.apply(lambda l: all(n in PROPS for n in l))
    f = pd.DataFrame(index=df.index)
    for i, key in enumerate(["M", "rho", "eps", "lneta"]):
        def mix(row_names, x, i=i):
            v = np.array([PROPS[n][i] for n in row_names]) if all(n in PROPS for n in row_names) else np.full(len(x), np.nan)
            return float(np.sum(x * (np.log(v) if i == 3 else v)))
        f[f"mix_{key}"] = [mix(n, x) for n, x in zip(names, xs)]
    f["frac_high_eps"] = [float(sum(x_ for n, x_ in zip(nn, x) if n in PROPS and PROPS[n][2] > 20)) for nn, x in zip(names, xs)]
    f["frac_PC"] = [float(sum(x_ for n, x_ in zip(nn, x) if n == "PC")) for nn, x in zip(names, xs)]
    f["n_solv"] = names.str.len()
    # molality: native for mol/kg; for mol/L approx c / rho_mix (neglects salt mass)
    m = np.where(df.conc_unit == "mol/kg", df.conc_native, df.conc_native / df.mixture_density_g_cm3)
    f["molal"] = m
    f["sqrt_m"] = np.sqrt(m)
    f["m2"] = m ** 2
    f["c_molL"] = df.conc_molL_est
    f["c_over_eps"] = m / f.mix_eps
    f["unit_molL"] = (df.conc_unit == "mol/L").astype(int)
    T = df.temperature_K
    f["T"] = T
    f["invT"] = 1000.0 / T
    f["vtf"] = 1.0 / (T - T0)
    f["lneta_over_T"] = f.mix_lneta / T * 300  # viscosity grows at low T; joint term
    f["salt"] = pd.Categorical(df.salt_name, categories=sorted(set(df.salt_name) | set(TEST_SALTS)))
    return f, ok


MONO = {"T": 1, "invT": -1, "vtf": -1}


def make_model(seed):
    cols_mono = MONO
    return HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=700, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
        categorical_features="from_dtype", monotonic_cst=cols_mono, random_state=seed)


def fit_predict(Xtr, ytr, wtr, Xte):
    preds = []
    for s in SEEDS:
        m = make_model(s).fit(Xtr, ytr, sample_weight=wtr)
        preds.append(m.predict(Xte))
    return np.mean(preds, axis=0)


def main():
    tr = pd.read_csv(D / "train.csv")
    te = pd.read_csv(D / "test.csv")
    ss = pd.read_csv(D / "sample_submission.csv")
    n0 = len(tr)
    tr = tr[tr.k >= 0.01].reset_index(drop=True)
    print(f"dropped k<0.01: {n0 - len(tr)} rows")
    combo = tr.solvents.str.split(";").apply(lambda l: "+".join(sorted(l)))

    results = {}
    for T0 in T0_GRID:
        X, ok = featurize(tr, T0)
        keep = ok.values
        Xk, yk, ck, sk = X[keep], tr.log_k[keep], combo[keep], tr.salt_name[keep]
        Tk = tr.temperature_K[keep]
        same = tr.source_doi[keep].isin(SAME_SOURCE).values
        if T0 == T0_GRID[0]:
            print(f"rows with all solvent props: {keep.sum()} / {len(tr)} (dropped {(~keep).sum()})")
        for wname, w in [("unweighted", np.ones(len(Xk))), ("same-src x3", np.where(same, 3.0, 1.0))]:
            r = {}
            # Fold A: hold out PC+EA
            ho = (ck == "EA+PC").values
            p = fit_predict(Xk[~ho], yk[~ho], w[~ho], Xk[ho])
            r["A"] = rmse(p, yk[ho])
            cold = (Tk[ho] < 250).values
            r["A_T<250"] = rmse(p[cold], yk[ho].values[cold])
            r["A_bias"] = float(np.mean(p - yk[ho]))
            # Fold B: hold out EC+PC binaries
            ho = (ck == "EC+PC").values
            p = fit_predict(Xk[~ho], yk[~ho], w[~ho], Xk[ho])
            r["B"] = rmse(p, yk[ho])
            for s in TEST_SALTS:
                mm = (sk[ho] == s).values
                if mm.any():
                    r[f"B_{s}"] = rmse(p[mm], yk[ho].values[mm])
            results[(T0, wname)] = r
            print(f"T0={T0} {wname:12s} " + " ".join(f"{k}={v:.4f}" for k, v in r.items()), flush=True)

    res = pd.DataFrame(results).T
    print("\nValidation (RMSE, log10 space):\n", res.round(4).to_string())
    best = res["A"].idxmin()
    T0, wname = best
    print(f"\nselected by fold A: T0={T0}, {wname}")

    # final fit on everything usable
    X, ok = featurize(tr, T0)
    keep = ok.values
    w = np.where(tr.source_doi[keep].isin(SAME_SOURCE), 3.0, 1.0) if wname != "unweighted" else np.ones(keep.sum())
    Xte, ok_te = featurize(te, T0)
    assert ok_te.all(), "test solvent missing props"
    pred_log = fit_predict(X[keep], tr.log_k[keep], w, Xte)
    k_pred = np.clip(10 ** pred_log, 0.01, None)

    sub = pd.DataFrame({"id": te.id, "k": k_pred})
    assert sub.id.equals(ss.id)
    sub.to_csv(ROOT / "submission.csv", index=False)
    pd.DataFrame({"id": te.id, "log_k": np.log10(k_pred)}).to_csv(ROOT / "submission_logk.csv", index=False)
    print("wrote submission.csv (id,k) and submission_logk.csv (id,log_k)")

    te = te.assign(pred_k=k_pred, pred_log=pred_log)
    print("\npredicted k by salt:\n", te.groupby("salt_name").pred_k.describe().round(3).to_string())
    pcdec = te[te.solvents == "PC;DEC"]
    print("\npredicted k by PC:DEC (w) ratio:\n", pcdec.groupby("solvent_fracs").pred_k.describe().round(3).to_string())

    # sanity: at low T, how does k move with DEC fraction? (fixed salt, nearest-molality ~1 mol/kg)
    print("\nlow-T sanity: median predicted k at molality 0.8-1.2 (LiBOB 0.6-0.9), by DEC w-fraction")
    for s in TEST_SALTS:
        g = pcdec[pcdec.salt_name == s]
        lo, hi = (0.6, 0.9) if s == "LiBOB" else (0.8, 1.2)
        g = g[g.conc_native.between(lo, hi)]
        g = g.assign(dec_w=g.solvent_fracs.str.split(";").str[1].astype(float), Tr=g.temperature_K.round(1))
        tab = g[g.Tr.isin(SANITY_T)].pivot_table(index="Tr", columns="dec_w", values="pred_k", aggfunc="median")
        print(f"\n{s}:\n", tab.round(3).to_string())
        # compare train analogue: EA+PC LiBOB
    ea = tr[(combo == "EA+PC") & tr.conc_native.between(0.6, 0.9)]
    ea = ea.assign(ea_w=ea.solvent_fracs.str.split(";").str[1].astype(float), Tr=ea.temperature_K.round(1))
    print("\ntrain reference, LiBOB in EA+PC (measured k, same c window), by EA w-fraction:\n",
          ea[ea.Tr.isin(SANITY_T)].pivot_table(index="Tr", columns="ea_w", values="k", aggfunc="median").round(3).to_string())


if __name__ == "__main__":
    main()

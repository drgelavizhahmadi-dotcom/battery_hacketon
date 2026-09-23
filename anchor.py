"""Phase 2b: anchor-delta model. log10k = pure-PC anchor (same source, same salt/c/T) + HGB(delta). Run: python3 anchor.py"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pipeline import SAME_SOURCE, SEEDS, TEST_SALTS, featurize, fit_predict

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
D = ROOT / "ca-li-sol-23-challenge"
T0 = 150
T_TOL = 0.3  # K, same grid level
DELTA_COLS = ["x_co", "mix_eps", "mix_lneta", "mix_M", "T", "molal", "salt"]
LOGK_MIN = -2.0


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def build_series(pure):
    """{T_level: (c_sorted, logk)} for one pure-PC (doi, salt) series; duplicates in c averaged."""
    out = {}
    for T, g in pure.groupby(pure.temperature_K.round(1)):
        g = g.groupby("conc_native").log_k.mean()
        out[float(T)] = (g.index.values, g.values)
    return out


def interp_c(curve, c):
    cs, ys = curve
    if c < cs[0] * (1 - 1e-3) or c > cs[-1] * (1 + 1e-3):
        return np.nan
    return float(np.interp(c, cs, ys))


def anchor_value(series, c, T):
    levels = np.array(sorted(series))
    near = np.abs(levels - T) < T_TOL
    if near.any():
        return interp_c(series[levels[near][0]], c)
    if T < levels[0] or T > levels[-1]:
        return np.nan
    hi = levels[np.searchsorted(levels, T)]
    lo = levels[np.searchsorted(levels, T) - 1]
    ylo, yhi = interp_c(series[lo], c), interp_c(series[hi], c)
    if np.isnan(ylo) or np.isnan(yhi):
        return np.nan
    w = (1 / T - 1 / lo) / (1 / hi - 1 / lo)  # linear in 1/T
    return ylo + w * (yhi - ylo)


def delta_model(seed):
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=400, max_leaf_nodes=15, min_samples_leaf=20,
                                         l2_regularization=1.0, categorical_features="from_dtype", random_state=seed)


def fit_predict_delta(Xtr, ytr, Xte):
    return np.mean([delta_model(s).fit(Xtr, ytr).predict(Xte) for s in SEEDS], axis=0)


def prepare(verbose=True):
    """Shared data prep: filtered train, features, same-source anchors for train mixtures and test rows."""
    tr = pd.read_csv(D / "train.csv")
    te = pd.read_csv(D / "test.csv")
    ss = pd.read_csv(D / "sample_submission.csv")
    tr = tr[tr.k >= 0.01].reset_index(drop=True)
    for df in (tr, te):
        df["combo"] = df.solvents.str.split(";").apply(lambda l: "+".join(sorted(l)))
        df["has_pc"] = df.solvents.str.split(";").apply(lambda l: "PC" in l)

    X, ok = featurize(tr, T0)
    Xte, _ = featurize(te, T0)
    for f in (X, Xte):
        f["x_co"] = 1 - f.frac_PC

    series = {k: build_series(g) for k, g in tr[tr.combo == "PC"].groupby(["source_doi", "salt_name"])}
    if verbose:
        print(f"pure-PC series in train: {len(series)}")

    # ---- train deltas: PC mixtures whose own (doi, salt) has a pure-PC series
    cand = ok & tr.has_pc & (tr.combo != "PC") & [(d, s) in series for d, s in zip(tr.source_doi, tr.salt_name)]
    anc = pd.Series(np.nan, index=tr.index)
    for i in tr.index[cand]:
        r = tr.loc[i]
        anc[i] = anchor_value(series[(r.source_doi, r.salt_name)], r.conc_native, r.temperature_K)
    use = cand & anc.notna()
    tr["anchor"], tr["delta"] = anc, tr.log_k - anc
    if verbose:
        print(f"mixture rows with a same-source pure-PC series: {cand.sum()}, anchored (no extrapolation): {use.sum()}")
        print("anchored rows by (doi, salt, combo):\n", tr[use].groupby(["source_doi", "salt_name", "combo"]).size().to_string())
        print("\ndelta by combo (mean, sd):\n", tr[use].groupby("combo").delta.agg(["size", "mean", "std"]).round(3).to_string())

    # ---- test: anchors from same-source mol/kg pure-PC series of the same salt (average where several cover)
    te_series = {s: [v for (d, salt), v in series.items() if salt == s and d in SAME_SOURCE] for s in TEST_SALTS}
    te_anc = np.full(len(te), np.nan)
    for j, r in enumerate(te.itertuples()):
        if not r.has_pc or r.conc_unit != "mol/kg":
            continue
        vals = [anchor_value(sr, r.conc_native, r.temperature_K) for sr in te_series[r.salt_name]]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            te_anc[j] = np.mean(vals)
    has = ~np.isnan(te_anc)
    return tr, te, ss, X, Xte, ok, use, te_anc, has


def main():
    tr, te, ss, X, Xte, ok, use, te_anc, has = prepare()

    # ---- validation, side by side with the Phase 2 GBM on the same anchored rows
    rows = []
    Xk, yk = X[ok], tr.log_k[ok]
    for fold, combo in [("A: PC+EA", "EA+PC"), ("B: EC+PC", "EC+PC")]:
        ho_all = (tr.combo == combo) & ok
        ho = ho_all & use
        gbm = pd.Series(fit_predict(Xk[~ho_all[ok]], yk[~ho_all[ok]], np.ones((~ho_all[ok]).sum()), X[ho_all]), index=tr.index[ho_all])
        dtrain = use & ~ho_all
        pd_ = fit_predict_delta(X.loc[dtrain, DELTA_COLS], tr.delta[dtrain], X.loc[ho, DELTA_COLS])
        pa = tr.anchor[ho].values + pd_
        pg = gbm[ho].values
        y = tr.log_k[ho].values
        pb = 0.5 * (pa + pg)
        groups = [("all", np.ones(len(y), bool)), ("T<250", (tr.temperature_K[ho] < 250).values)]
        groups += [(s, (tr.salt_name[ho] == s).values) for s in TEST_SALTS if (tr.salt_name[ho] == s).any()]
        for gname, m in groups:
            rows.append({"fold": fold, "subset": gname, "n": int(m.sum()),
                         "GBM_rmse": rmse(pg[m], y[m]), "GBM_bias": float(np.mean(pg[m] - y[m])),
                         "anchor_rmse": rmse(pa[m], y[m]), "anchor_bias": float(np.mean(pa[m] - y[m])),
                         "blend_rmse": rmse(pb[m], y[m]), "blend_bias": float(np.mean(pb[m] - y[m]))})
        print(f"{fold}: {ho_all.sum()} held-out rows, {ho.sum()} anchored (GBM on all held-out: {rmse(gbm, tr.log_k[ho_all]):.4f})", flush=True)
    print("\nValidation on anchored held-out rows (log10):\n", pd.DataFrame(rows).round(4).to_string(index=False))

    final_delta = fit_predict_delta(X.loc[use, DELTA_COLS], tr.delta[use], Xte.loc[has, DELTA_COLS])
    gbm_te = pd.read_csv(ROOT / "submission_logk.csv")
    assert gbm_te.id.equals(te.id)
    gbm_log = gbm_te.log_k.values
    anchor_log = gbm_log.copy()
    anchor_log[has] = te_anc[has] + final_delta
    blend_log = 0.5 * (anchor_log + gbm_log)

    for name, v in [("submission_anchor.csv", anchor_log), ("submission_blend.csv", blend_log)]:
        out = pd.DataFrame({"id": te.id, "log_k": np.clip(v, LOGK_MIN, None)})
        assert out.id.equals(ss.id) and list(out.columns) == list(ss.columns) and out.log_k.notna().all()
        out.to_csv(ROOT / name, index=False)
    print(f"\nwrote submission_anchor.csv, submission_blend.csv (id,log_k, clipped at {LOGK_MIN})")

    miss = te[~has]
    print(f"\ntest rows without usable anchor: {(~has).sum()} / {len(te)} -> fall back to Phase 2 GBM")
    reason = np.where(~miss.has_pc, "no PC in mixture",
                      np.where(miss.temperature_K < 230, "T below anchor grid", "c outside anchor range"))
    print(pd.crosstab([miss.salt_name, miss.combo], reason).to_string())
    te = te.assign(gbm=gbm_log, anchor=anchor_log, blend=blend_log)
    pc = te[has]
    print("\nanchored rows: mean(anchor - GBM) by salt:", pc.assign(d=pc.anchor - pc.gbm).groupby("salt_name").d.mean().round(3).to_dict())
    print("by PC:DEC w-ratio:", pc.assign(d=pc.anchor - pc.gbm).groupby("solvent_fracs").d.mean().round(3).to_dict())
    lo = pc[pc.temperature_K.between(233, 234) & pc.conc_native.between(0.8, 1.2) & (pc.salt_name != "LiBOB")]
    print("\nlow-T check (233.75 K, ~1 mol/kg), median anchor-model k by DEC w:\n",
          lo.assign(dec_w=lo.solvent_fracs.str.split(";").str[1], kk=10 ** lo.anchor)
          .pivot_table(index="salt_name", columns="dec_w", values="kk", aggfunc="median").round(3).to_string())


if __name__ == "__main__":
    main()

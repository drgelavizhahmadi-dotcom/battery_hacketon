"""Phase 5: extend pure-PC anchors in c (quadratic in ln c) and T (VTF per c) to anchor more fallback rows.
Run: python3 extend.py"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from anchor import LOGK_MIN, T_TOL, build_series, interp_c, prepare, rmse
from kan import DELTA_IN, Scaler, co_props, delta_head, predict, subset, table, to_batch, train
from pipeline import SAME_SOURCE, TEST_SALTS, fit_predict

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SEEDS = list(range(15))
KAN_EPOCHS = 107  # median best epoch from kan.py
C_EXT, T_EXT = 0.30, 20.0  # fraction of measured ln-c width; K below coldest measured T
MIN_PTS = 4
T0_GRID = np.arange(100, 181, 2.0)
MAX_EXT_RMSE = 0.15


# ------------------------------------------------------------------ fits
def fit_quad(c, y):
    lc = np.log(c)
    w = lc.max() - lc.min()
    return np.polyfit(lc, y, 2), lc.min() - C_EXT * w, lc.max() + C_EXT * w


def fit_vtf(T, y):
    best = None
    for T0 in T0_GRID:
        A = np.column_stack([np.ones_like(T), -1 / (T - T0)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        if coef[1] <= 0:
            continue
        sse = float(np.sum((A @ coef - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, coef[0], coef[1], T0)
    return None if best is None else best[1:]


def vtf_eval(p, T):
    A, B, T0 = p
    return A - B / (T - T0)


def build_ext(pure):
    """Per (doi, salt) pure-PC series: level curves, per-T quadratic c fits, per-c VTF fits."""
    levels = build_series(pure)
    cfits = {T: fit_quad(cs, ys) for T, (cs, ys) in levels.items() if len(cs) >= MIN_PTS}
    vfits = []
    for c, g in pure.groupby(pure.conc_native.round(3)):
        g = g.groupby(g.temperature_K.round(1)).log_k.mean()
        if len(g) >= MIN_PTS:
            p = fit_vtf(g.index.values, g.values)
            if p is not None:
                vfits.append((float(c), p, float(g.index.min()), float(g.index.max())))
    return {"levels": levels, "cfits": cfits, "vfits": sorted(vfits, key=lambda v: v[0])}


def value_at_level(ext, L, c, allow):
    v = interp_c(ext["levels"][L], c)
    if not np.isnan(v):
        return v, "interp"
    if L in ext["cfits"]:
        coef, lo, hi = ext["cfits"][L]
        lc = np.log(c)
        cs = ext["levels"][L][0]
        side = "c_lo" if c < cs[0] else "c_hi"
        if allow[side] and lo <= lc <= hi:
            return float(np.polyval(coef, lc)), "c_ext"
    return np.nan, None


def anchor_ext(ext, c, T, allow):
    """(value, kind) with kind in {interp, c_ext, T_ext}; never extrapolates in both c and T."""
    levels = np.array(sorted(ext["levels"]))
    near = np.abs(levels - T) < T_TOL
    v, kind = np.nan, None
    if near.any():
        v, kind = value_at_level(ext, levels[near][0], c, allow)
    elif levels[0] < T < levels[-1]:
        i = np.searchsorted(levels, T)
        lo, hi = levels[i - 1], levels[i]
        (ylo, klo), (yhi, khi) = value_at_level(ext, lo, c, allow), value_at_level(ext, hi, c, allow)
        if not (np.isnan(ylo) or np.isnan(yhi)):
            w = (1 / T - 1 / lo) / (1 / hi - 1 / lo)
            v, kind = ylo + w * (yhi - ylo), ("interp" if klo == khi == "interp" else "c_ext")
    if not np.isnan(v):
        return v, kind
    if allow["T_cold"]:
        pts = [(cg, vtf_eval(p, T)) for cg, p, tmin, tmax in ext["vfits"] if tmin - T_EXT <= T <= tmax]
        if len(pts) >= 2 and pts[0][0] * (1 - 1e-3) <= c <= pts[-1][0] * (1 + 1e-3):
            cg, yg = zip(*pts)
            return float(np.interp(c, cg, yg)), "T_ext"
    return np.nan, None


# ------------------------------------------------------------------ extension validation
def validate_extension(pure_all):
    err = {"c_lo": [], "c_hi": [], "T_cold": []}
    for _, pure in pure_all.groupby(["source_doi", "salt_name"]):
        for _, (cs, ys) in build_series(pure).items():
            if len(cs) < MIN_PTS + 1:
                continue
            for side, drop in [("c_lo", 0), ("c_hi", len(cs) - 1)]:
                keep = np.arange(len(cs)) != drop
                coef, lo, hi = fit_quad(cs[keep], ys[keep])
                if lo <= np.log(cs[drop]) <= hi:
                    err[side].append(float(np.polyval(coef, np.log(cs[drop]))) - ys[drop])
        for _, g in pure.groupby(pure.conc_native.round(3)):
            g = g.groupby(g.temperature_K.round(1)).log_k.mean()
            if len(g) < MIN_PTS + 1:
                continue
            T, y = g.index.values, g.values
            p = fit_vtf(T[1:], y[1:])
            if p is not None and T[0] >= T[1] - T_EXT:
                err["T_cold"].append(vtf_eval(p, T[0]) - y[0])
    return {k: (len(v), float(np.sqrt(np.mean(np.square(v)))) if v else np.nan) for k, v in err.items()}


def main():
    tr, te, ss, X, Xte, ok, use_old, te_anc_old, has_old = prepare(verbose=False)
    pure_all = tr[tr.combo == "PC"]

    # 2. validate each extension direction on held-out edge points of the train pure-PC series
    val = validate_extension(pure_all)
    print("extension validation (hold out outermost point, extrapolate; log10 RMSE):")
    allow = {}
    for k, (n, r) in val.items():
        allow[k] = bool(n > 0 and r <= MAX_EXT_RMSE)
        print(f"  {k:7s} n={n:4d} rmse={r:.4f} -> {'USE' if allow[k] else 'DO NOT USE'}")

    # 1. extended anchors for train mixtures (own series) and test rows (same-source series, same salt)
    exts = {k: build_ext(g) for k, g in pure_all.groupby(["source_doi", "salt_name"])}
    print(f"series: {len(exts)}; per-T c-fits: {sum(len(e['cfits']) for e in exts.values())}; "
          f"per-c VTF fits: {sum(len(e['vfits']) for e in exts.values())}")
    cand = ok & tr.has_pc & (tr.combo != "PC") & [(d, s) in exts for d, s in zip(tr.source_doi, tr.salt_name)]
    anc, kind = pd.Series(np.nan, index=tr.index), pd.Series(None, index=tr.index, dtype=object)
    for i in tr.index[cand]:
        r = tr.loc[i]
        anc[i], kind[i] = anchor_ext(exts[(r.source_doi, r.salt_name)], r.conc_native, r.temperature_K, allow)
    use = cand & anc.notna()
    tr["anchor"], tr["delta"], tr["akind"] = anc, tr.log_k - anc, kind
    print(f"train mixture anchors: {use_old.sum()} -> {use.sum()}  by kind: {tr.akind[use].value_counts().to_dict()}")
    same_old = np.allclose(tr.anchor[use_old], tr.anchor[use_old])  # interp anchors are unchanged by construction
    assert (tr.akind[use_old] == "interp").all() and same_old

    te_ext = {s: [e for (d, salt), e in exts.items() if salt == s and d in SAME_SOURCE] for s in TEST_SALTS}
    te_anc, te_kind = np.full(len(te), np.nan), np.full(len(te), None, dtype=object)
    for j, r in enumerate(te.itertuples()):
        if not r.has_pc or r.conc_unit != "mol/kg":
            continue
        res = [anchor_ext(e, r.conc_native, r.temperature_K, allow) for e in te_ext[r.salt_name]]
        for k in ["interp", "c_ext", "T_ext"]:  # prefer the least extrapolated kind available
            vals = [v for v, kk in res if kk == k]
            if vals:
                te_anc[j], te_kind[j] = np.mean(vals), k
                break
    has = ~np.isnan(te_anc)
    assert np.allclose(te_anc[has_old], te_anc_old[has_old])
    gained = has & ~has_old
    print(f"\ntest anchored: {has_old.sum()} -> {has.sum()}; fallback rows gaining an anchor: {gained.sum()} / {(~has_old).sum()}")
    print(f"PRE-REG (>= 400 of 646 gain an anchor): {'HELD' if gained.sum() >= 400 else 'FAILED'}")
    print("gained by salt x kind:\n", pd.crosstab(te.salt_name[gained], te_kind[gained]).to_string())
    left = ~has
    print("still unanchored by salt x combo:\n", pd.crosstab(te.salt_name[left], te.combo[left]).to_string())

    # 3. KAN-delta (15 seeds, fixed epochs) with the new anchors; folds A / B
    salts = sorted(set(tr.salt_name) | set(te.salt_name))
    sidx = {s: i for i, s in enumerate(salts)}
    gbm_cols = list(X.columns)
    for f, df in [(X, tr), (Xte, te)]:
        f[["eps_co", "lneta_co", "M_co"]] = co_props(df)
    sc = Scaler([X[use], Xte[has]], DELTA_IN)
    B = to_batch(sc(X), tr.salt_name.map(sidx).values, x_co=X.x_co, y=tr.delta.fillna(0))
    Bte = to_batch(sc(Xte), te.salt_name.map(sidx).values, x_co=Xte.x_co)
    y = tr.log_k.values
    Xk, yk = X.loc[ok, gbm_cols], tr.log_k[ok]
    rows = []
    for fold, combo in [("A: PC+EA", "EA+PC"), ("B: EC+PC", "EC+PC")]:
        ho_all = ((tr.combo == combo) & ok).values
        ho, dt = ho_all & use.values, use.values & ~ho_all
        gbm = np.full(len(tr), np.nan)
        gbm[ho_all] = fit_predict(Xk[~ho_all[ok.values]], yk[~ho_all[ok.values]], np.ones((~ho_all[ok.values]).sum()), X.loc[ho_all, gbm_cols])
        kp = np.array([predict(train([6, 1], len(DELTA_IN), len(salts), delta_head, subset(B, dt), epochs=KAN_EPOCHS, seed=s)[0],
                               delta_head, subset(B, ho)) for s in SEEDS])
        kan = tr.anchor.values[ho] + kp.mean(0)
        kd = tr.akind.values[ho]
        g = [("all", np.ones(ho.sum(), bool)), ("old anchors", kd == "interp"), ("new: c_ext", kd == "c_ext"),
             ("new: T_ext", kd == "T_ext"), ("T<250", tr.temperature_K.values[ho] < 250)]
        g += [(s, tr.salt_name.values[ho] == s) for s in TEST_SALTS if (tr.salt_name.values[ho] == s).any()]
        g = [(n, m) for n, m in g if m.any()]
        rows += [{"fold": fold, **r} for r in table(y[ho], {"GBM": gbm[ho], "KANd": kan}, g, {"KANd": kp.std(0)})]
        print(f"{fold}: held-out {ho_all.sum()}, anchored {ho.sum()} (was {(ho_all & use_old.values).sum()})", flush=True)
    print("\nAnchored held-out rows, new anchors (log10 RMSE / bias; KAN-delta 15 seeds, fixed epochs):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # 4. final: KAN-delta on anchored test rows, Phase 2 GBM elsewhere
    kte = np.array([predict(train([6, 1], len(DELTA_IN), len(salts), delta_head, subset(B, use.values), epochs=KAN_EPOCHS, seed=s)[0],
                            delta_head, subset(Bte, has)) for s in SEEDS])
    gbm_te = pd.read_csv(ROOT / "submission_logk.csv").log_k.values
    old = pd.read_csv(ROOT / "submission_final.csv")
    v = gbm_te.copy()
    v[has] = te_anc[has] + kte.mean(0)
    out = pd.DataFrame({"id": ss.id, "log_k": np.clip(v, LOGK_MIN, None)})
    assert out.id.equals(te.id) and out.id.equals(old.id) and list(out.columns) == list(ss.columns) and out.log_k.notna().all()
    out.to_csv(ROOT / "submission_final_v2.csv", index=False)
    d = out.log_k - old.log_k
    print(f"\nwrote submission_final_v2.csv: KAN-delta {has.sum()}, GBM {(~has).sum()}; log_k range [{out.log_k.min():.4f}, {out.log_k.max():.4f}]")
    print(f"rows moved GBM -> KAN-delta: {gained.sum()}; mean change {d[gained].mean():+.4f}, mean|change| {d[gained].abs().mean():.4f}, max|change| {d[gained].abs().max():.4f}")
    print(f"previously anchored rows (5→15 seeds, retrained on larger delta set): mean|change| {d[has_old].abs().mean():.4f}, max {d[has_old].abs().max():.4f}")
    print("mean change on moved rows by salt x kind:\n",
          pd.DataFrame({"salt": te.salt_name[gained], "kind": te_kind[gained], "d": d[gained]}).pivot_table(index="salt", columns="kind", values="d", aggfunc=["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    main()

"""Phase 6: KAN on the full target log10 k (no anchor, no delta). Run: python3 kanfull.py"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from anchor import LOGK_MIN, prepare
from kan import DELTA_IN, Scaler, co_props, delta_head, predict, subset, table, to_batch, train
from pipeline import TEST_SALTS, fit_predict

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SEEDS = list(range(15))
EPOCHS = 107
# x_cyclic := fraction of high-eps (>20) solvent, as in pipeline.frac_high_eps; x_linear = 1 - x_cyclic
FULL_IN = ["mix_eps", "mix_lneta", "mix_M", "mix_rho", "x_cyclic", "x_linear", "frac_PC",
           "eps_co", "lneta_co", "M_co", "invT", "molal"]


def full_head(net, b):
    return net(b["x"], b["s"])[:, 0]


def main():
    tr, te, ss, X, Xte, ok, use, te_anc, has = prepare(verbose=False)
    gbm_cols = list(X.columns)
    salts = sorted(set(tr.salt_name) | set(te.salt_name))
    sidx = {s: i for i, s in enumerate(salts)}
    for f, df in [(X, tr), (Xte, te)]:
        f[["eps_co", "lneta_co", "M_co"]] = co_props(df)
        f["x_cyclic"] = f.frac_high_eps
        f["x_linear"] = 1 - f.frac_high_eps
    s_tr, s_te = tr.salt_name.map(sidx).values, te.salt_name.map(sidx).values
    # KAN-full: co-solvent props zero where there is no co-solvent (pure PC)
    Xf, Xfte = X.copy(), Xte.copy()
    for f in (Xf, Xfte):
        f[["eps_co", "lneta_co", "M_co"]] = f[["eps_co", "lneta_co", "M_co"]].fillna(0.0)
    scf = Scaler([Xf[ok], Xfte], FULL_IN)
    BF = to_batch(scf(Xf), s_tr, y=tr.log_k)  # rows outside ok keep NaN mix props but are never used
    BFte = to_batch(scf(Xfte), s_te)
    # KAN-delta exactly as kan.py (old anchors), for the comparison column
    scd = Scaler([X[use], Xte[has]], DELTA_IN)
    BD = to_batch(scd(X), s_tr, x_co=X.x_co, y=tr.delta.fillna(0))
    print("pre-registered: KAN-full fold A RMSE in [0.19, 0.25] with negative bias (clearly worse than KAN-delta 0.142)\n")

    y = tr.log_k.values
    Xk, yk = X.loc[ok, gbm_cols], tr.log_k[ok]
    rows, fb_rows = [], []
    for fold, combo in [("A: PC+EA", "EA+PC"), ("B: EC+PC", "EC+PC")]:
        ho_all = ((tr.combo == combo) & ok).values
        ho, train_m, dt = ho_all & use.values, ok.values & ~ho_all, use.values & ~ho_all
        gbm = np.full(len(tr), np.nan)
        gbm[ho_all] = fit_predict(Xk[~ho_all[ok.values]], yk[~ho_all[ok.values]], np.ones((~ho_all[ok.values]).sum()), X.loc[ho_all, gbm_cols])
        kf = np.full((len(SEEDS), len(tr)), np.nan)
        kf[:, ho_all] = [predict(train([8, 1], len(FULL_IN), len(salts), full_head, subset(BF, train_m), epochs=EPOCHS, seed=s)[0],
                                 full_head, subset(BF, ho_all)) for s in SEEDS]
        kd = np.array([tr.anchor.values[ho] + predict(train([6, 1], len(DELTA_IN), len(salts), delta_head, subset(BD, dt), epochs=EPOCHS, seed=s)[0],
                                                      delta_head, subset(BD, ho)) for s in SEEDS])
        g = [("all", np.ones(ho.sum(), bool)), ("T<250", tr.temperature_K.values[ho] < 250)]
        g += [(s, tr.salt_name.values[ho] == s) for s in TEST_SALTS if (tr.salt_name.values[ho] == s).any()]
        rows += [{"fold": fold, **r} for r in table(y[ho], {"GBM": gbm[ho], "KANfull": kf[:, ho].mean(0), "KANd": kd.mean(0)}, g,
                                                    {"KANfull": kf[:, ho].std(0)})]
        # fallback-type held-out rows: in the fold but without an anchor (the analogue of the 646 test fallback rows)
        fb = ho_all & ~use.values
        g = [("no-anchor all", np.ones(fb.sum(), bool)), ("no-anchor T<250", tr.temperature_K.values[fb] < 250)]
        g += [(f"no-anchor {s}", tr.salt_name.values[fb] == s) for s in TEST_SALTS if (tr.salt_name.values[fb] == s).any()]
        g = [(n, m) for n, m in g if m.any()]
        fb_rows += [{"fold": fold, **r} for r in table(y[fb], {"GBM": gbm[fb], "KANfull": kf[:, fb].mean(0)}, g, {"KANfull": kf[:, fb].std(0)})]
        print(f"{fold} done: {ho.sum()} anchored held-out rows, {fb.sum()} without anchor", flush=True)

    t = pd.DataFrame(rows)
    print("\nAnchored held-out rows (log10 RMSE / bias; 15 seeds, 107 epochs each):\n", t.round(4).to_string(index=False))
    print("\nFallback-type held-out rows (in fold, no anchor):\n", pd.DataFrame(fb_rows).round(4).to_string(index=False))
    a = t[(t.fold == "A: PC+EA") & (t.subset == "all")].iloc[0]
    held = 0.19 <= a.KANfull_rmse <= 0.25 and a.KANfull_bias < 0
    print(f"\nPRE-REG: KAN-full fold A rmse {a.KANfull_rmse:.4f}, bias {a.KANfull_bias:+.4f} -> {'HELD' if held else 'FAILED'}")

    # reference submission: KAN-full everywhere
    kte = np.array([predict(train([8, 1], len(FULL_IN), len(salts), full_head, subset(BF, ok.values), epochs=EPOCHS, seed=s)[0],
                            full_head, BFte) for s in SEEDS]).mean(0)
    out = pd.DataFrame({"id": ss.id, "log_k": np.clip(kte, LOGK_MIN, None)})
    assert out.id.equals(te.id) and list(out.columns) == list(ss.columns) and out.log_k.notna().all()
    out.to_csv(ROOT / "submission_kanfull.csv", index=False)
    final = pd.read_csv(ROOT / "submission_final.csv").log_k.values
    gbm_te = pd.read_csv(ROOT / "submission_logk.csv").log_k.values
    print(f"wrote submission_kanfull.csv (reference only). vs submission_final: anchored rows mean {np.mean(out.log_k[has] - final[has]):+.4f}, "
          f"rmse {np.sqrt(np.mean((out.log_k[has] - final[has]) ** 2)):.4f}; fallback rows vs GBM mean {np.mean(out.log_k[~has] - gbm_te[~has]):+.4f}, "
          f"rmse {np.sqrt(np.mean((out.log_k[~has] - gbm_te[~has]) ** 2)):.4f}")


if __name__ == "__main__":
    main()

"""Reproducible rebuild of the recommended submission. Run: python3 final.py

KAN-delta (kan.py setup, 107 epochs, seeds 0-4) on anchored test rows; Phase 2 GBM (pipeline.py, T0=130,
unweighted, seeds 0-2) on the rest. Refuses to overwrite submission_final.csv if the rebuild drifts.
"""
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from anchor import LOGK_MIN, prepare
from kan import DELTA_IN, Scaler, co_props, delta_head, predict, subset, to_batch, train
from pipeline import featurize, fit_predict

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
ROOT = Path(__file__).parent
KAN_SEEDS, KAN_EPOCHS = [0, 1, 2, 3, 4], 107
GBM_T0 = 130
MAX_MEAN_DIFF = 0.02


def validate(path, ss):
    df = pd.read_csv(path)
    checks = {
        "header id,log_k": list(df.columns) == ["id", "log_k"],
        "5959 rows": len(df) == 5959,
        "no NaN": not df.isna().any().any(),
        "ids unique": df.id.is_unique,
        "ids == sample (order)": df.id.equals(ss.id),
        "log_k >= -2": bool((df.log_k >= LOGK_MIN).all()),
    }
    status = "OK " if all(checks.values()) else "FAIL"
    bad = [k for k, v in checks.items() if not v]
    print(f"  {status} {path.name:30s} range [{df.log_k.min():+.4f}, {df.log_k.max():+.4f}]  md5 {hashlib.md5(path.read_bytes()).hexdigest()[:10]}"
          + (f"  FAILED: {bad}" if bad else ""))
    return all(checks.values()), df


def main():
    tr, te, ss, X, Xte, ok, use, te_anc, has = prepare(verbose=False)

    # KAN-delta, exactly as in kan.py
    salts = sorted(set(tr.salt_name) | set(te.salt_name))
    sidx = {s: i for i, s in enumerate(salts)}
    for f, df in [(X, tr), (Xte, te)]:
        f[["eps_co", "lneta_co", "M_co"]] = co_props(df)
    sc = Scaler([X[use], Xte[has]], DELTA_IN)
    B = to_batch(sc(X), tr.salt_name.map(sidx).values, x_co=X.x_co, y=tr.delta.fillna(0))
    Bte = to_batch(sc(Xte), te.salt_name.map(sidx).values, x_co=Xte.x_co)
    kan = np.mean([predict(train([6, 1], len(DELTA_IN), len(salts), delta_head, subset(B, use.values), epochs=KAN_EPOCHS, seed=s)[0],
                           delta_head, subset(Bte, has)) for s in KAN_SEEDS], axis=0)

    # Phase 2 GBM, exactly as in pipeline.py's final fit
    Xg, okg = featurize(tr, GBM_T0)
    Xgte, _ = featurize(te, GBM_T0)
    gbm_k = np.clip(10 ** fit_predict(Xg[okg.values], tr.log_k[okg.values], np.ones(okg.sum()), Xgte), 0.01, None)
    gbm = np.log10(gbm_k)

    v = gbm.copy()
    v[has] = te_anc[has] + kan
    new = pd.DataFrame({"id": ss.id, "log_k": np.clip(v, LOGK_MIN, None)})

    old = pd.read_csv(ROOT / "submission_final.csv")
    d = (new.log_k - old.log_k).abs()
    print(f"rebuild vs existing submission_final.csv: max|diff| {d.max():.2e}, mean|diff| {d.mean():.2e} "
          f"(anchored {d[has].mean():.2e}, fallback {d[~has].mean():.2e})")
    if d.mean() > MAX_MEAN_DIFF:
        print(f"STOP: mean|diff| > {MAX_MEAN_DIFF}; nothing written.")
        sys.exit(1)

    new.to_csv(ROOT / "submission_final.csv", index=False)
    plus = new.assign(log_k=np.clip(new.log_k + 0.1, LOGK_MIN, None))
    plus.to_csv(ROOT / "submission_final_plus01.csv", index=False)

    print("\nvalidation:")
    files = ["submission_final.csv", "submission_final_plus01.csv", "submission_anchor.csv"]
    frames = {}
    for name in files:
        good, frames[name] = validate(ROOT / name, ss)
        if not good:
            sys.exit(1)

    print(f"\nrows per model (final): KAN-delta {has.sum()}, Phase 2 GBM {(~has).sum()}, total {len(new)}")
    print("per-salt mean log_k:")
    summ = pd.DataFrame({n.replace("submission_", "").replace(".csv", ""): f.log_k.groupby(te.salt_name.values).mean() for n, f in frames.items()})
    summ.loc["ALL"] = [f.log_k.mean() for f in frames.values()]
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()

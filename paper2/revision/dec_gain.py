"""paper2/revision/dec_gain.py: the learned DEC gain, traceable to numbers (hygiene; appdomain_plan.md, ec769ad).
delta log10 k vs pure PC predicted by the 5 final KAN-delta seeds (archived delta_s0-4: gauge.py, 107 epochs on all anchored rows,
the protocol of kan.py's finals1) for binary PC+DEC at x_DEC = 0.7, 1 mol/kg, T = 233, 273, 313 K, per test salt (kan.py's figure
construction). Run from the repo root with the pinned interpreter:
  /usr/bin/python3 paper2/revision/dec_gain.py > paper2/revision/dec_gain_output.txt"""
import contextlib
import io
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "paper2", REPO / "arch"):
    sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
import torch

import anchor as AN
from kan import DELTA_IN, KAN, LOGK_MIN, Scaler, co_props, delta_head, predict, to_batch
from pipeline import PROPS, TEST_SALTS
from stage2 import extract_verified, in_memory

TEMPS, XDEC, SEEDS = [233, 273, 313], 0.7, list(range(5))


def main():
    print("paper2/revision/dec_gain.py: learned DEC gain from the archived 5 final KAN-delta seeds (delta_s0-4)")
    tr, te, ss, X, Xte, ok, use, te_anc, has = AN.prepare(verbose=False)
    salts = sorted(set(tr.salt_name) | set(te.salt_name)); sidx = {s: i for i, s in enumerate(salts)}
    for f, df in [(X, tr), (Xte, te)]:
        f[["eps_co", "lneta_co", "M_co"]] = co_props(df)
    sc1 = Scaler([X[use], Xte[has]], DELTA_IN)                 # identical to kan.py / final.py
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        blobs = in_memory(extract_verified([f"delta_s{s}.pt" for s in SEEDS]))
    print("  inputs: delta_s0-4.pt SHA-256 verified against weights_manifest.txt, held in memory")
    nets = []
    for s in SEEDS:
        n = KAN(len(DELTA_IN), [6, 1], len(salts))
        n.load_state_dict({k: v.float() for k, v in torch.load(io.BytesIO(blobs[f"delta_s{s}.pt"]), weights_only=False).items()}, strict=False)
        nets.append(n.eval())
    # traceability check: the 5 seeds reproduce the committed submission's KAN-delta rows
    b_te = to_batch(sc1(Xte[has]), te.salt_name.map(sidx).values[has], x_co=Xte.x_co.values[has])
    kan_log = te_anc[has] + np.mean([predict(n, delta_head, b_te) for n in nets], 0)
    sub = pd.read_csv(REPO / "submission_kan.csv").log_k.values[has]
    dev = float(np.max(np.abs(np.clip(kan_log, LOGK_MIN, None) - sub)))
    print(f"  check: the 5 archived seeds vs submission_kan.csv on the {int(has.sum())} anchored test rows: max |difference| {dev:.2e} "
          f"-> {'REPRODUCED (float32 tolerance 1e-5)' if dev < 1e-5 else 'MISMATCH'}")
    pc, dec = PROPS["PC"], PROPS["DEC"]
    rows = []
    print(f"\n  delta log10 k vs pure PC, binary PC+DEC, x_DEC = {XDEC}, 1 mol/kg (x_co * h, as in kan.py's figure)")
    print("    salt                   T [K]   " + "   ".join(f"seed {s}" for s in SEEDS) + "     mean   min    max")
    for salt in TEST_SALTS:
        for T in TEMPS:
            f = pd.DataFrame({"x_co": [XDEC], "eps_co": dec[2], "lneta_co": np.log(dec[3]), "M_co": dec[0],
                              "mix_eps": (1 - XDEC) * pc[2] + XDEC * dec[2], "mix_lneta": (1 - XDEC) * np.log(pc[3]) + XDEC * np.log(dec[3]),
                              "invT": 1000 / T, "molal": 1.0})
            b = to_batch(sc1(f), np.full(1, sidx[salt]), x_co=[XDEC])
            v = np.array([float(predict(n, delta_head, b)[0]) for n in nets])
            rows.append(dict(salt=salt, T=T, seeds=v.tolist(), mean=float(v.mean()), lo=float(v.min()), hi=float(v.max())))
            print(f"    {salt:22s} {T:5d}   " + "  ".join(f"{x:+.3f}" for x in v) + f"   {v.mean():+.3f}  {v.min():+.3f}  {v.max():+.3f}")
    print("\n  SUMMARY STATISTIC the paper would quote: the 5-seed mean per salt (the figure's line), with its range over the test salts:")
    for T in TEMPS:
        m = [r["mean"] for r in rows if r["T"] == T]
        print(f"    {T} K: 5-seed mean delta log10 k from {min(m):+.2f} to {max(m):+.2f} over the {len(m)} test salts "
              f"(per-seed range {min(r['lo'] for r in rows if r['T'] == T):+.2f} to {max(r['hi'] for r in rows if r['T'] == T):+.2f})")


if __name__ == "__main__":
    main()

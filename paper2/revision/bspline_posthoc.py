"""paper2/revision/bspline_posthoc.py: POST-HOC follow-up to the B-spline replication (addendum ff5bd20; no verdict changes).
1. P2 path extension to gamma = 1e-8, 1e-9 (archive-17 weights); 2. P2 at mu = 1e-5 (new models); 3. the 7 non-certifying
independent P0/P1 models (descriptive). Run from the repo root with the pinned interpreter:
  /usr/bin/python3 paper2/revision/bspline_posthoc.py >> paper2/revision/bspline_output.txt"""
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "paper2", REPO / "arch", HERE, REPO / "tests"):
    sys.path.insert(0, str(p))

import numpy as np

import bspline as BP
import bspline_null as BN
import infimum as I

ARCH17 = "gauge_weights_2026-10-24q.tar.gz"
EXT_G = [(3, 1e-8), (4, 1e-9)]
SEEDS = list(range(8))
CONDS = [c for c, _ in BP.CONDS]
OUT = HERE / "bspline_posthoc_results.json"


def log(*a):
    print(*a, flush=True)


def verify_against_manifest(names):
    man = {l.split()[0].split("/")[-1]: l.split()[1] for l in (REPO / "weights_manifest.txt").read_text().splitlines()
           if l.startswith("gauge_weights/bsp_") and l.split()[-1] == ARCH17}
    bad = [n for n in names if n not in man or hashlib.sha256((I.W / n).read_bytes()).hexdigest() != man[n]]
    if bad:
        raise SystemExit(f"SHA-256 mismatch / not in archive 17: {bad[:5]}")
    return len(names)


def extend(key, wlog):
    """continue the P2 short path from gamma = 1e-7 (archive 17) to 1e-8 and 1e-9, the dose/Step 1 point protocol"""
    _, _, zin = BP.iflat(key)
    x = I.npx(I.cached(f"bsp_path_{key}_g2.pt", lambda: None)[0]["x"])
    pts = []
    for k, g in EXT_G:
        fg, _, _ = BP.iflat(key, "L2", g)
        st = I.W / f"bsp_path_{key}_g{k}_start.npy"
        if not st.exists():
            np.save(st, x)
        sp = dict(key=key, kind="L2", coef=g, start=str(st), name=f"bsp_path_{key}_g{k}.pt", ck=str(I.W / f"bsp_path_{key}_g{k}_ck.npz"),
                  gtol=I.GRAD_TOL, elapsed0=0.0)
        xk, inf_ = BP.run_point_bsp(sp, fg, wlog)
        r = BN.analyse_c(fg, fg.net, zin, xk, inf_, sp, f"bsp_path_{key}_g{k}", wlog, BP.run_point_bsp)
        r.update(coef=g); pts.append((r, xk)); x = xk
    return pts


def main():
    log("\n" + "#" * 100)
    log(f"paper2/revision/bspline_posthoc.py {time.strftime('%Y-%m-%d %H:%M:%S')}: POST-HOC follow-up (addendum ff5bd20; no verdict changes)")
    log("#" * 100)
    p2 = [f"{c}_P2_s{s}" for c in CONDS for s in SEEDS]
    ind = [f"independent_{l}_s{s}" for l in ("P0", "P1") for s in SEEDS]
    names = [n for k in p2 for n in (f"bsp_early_{k}.pt", f"bsp_pol_{k}.pt", f"bsp_path_{k}_g2.pt", f"bsp_rec_{k}.json")] + \
            [n for k in ind for n in (f"bsp_rec_{k}.json",)]
    log(f"  inputs: {verify_against_manifest(names)} archive-17 files, SHA-256 verified against weights_manifest.txt")
    res = {}

    # ---------------- 1. P2 path extension
    log("\n==================== 1. P2 path extension (gamma 1e-8, 1e-9; mu = 1e-6 as frozen) ====================")
    lines = []
    with ThreadPoolExecutor(4) as ex:
        ext = dict(zip(p2, ex.map(lambda k: extend(k, lines.append), p2)))
    rows = {}
    for k in p2:
        rec = json.loads((I.W / f"bsp_rec_{k}.json").read_text())
        f, _, _ = BP.iflat(k)
        xp = I.npx(I.cached(f"bsp_pol_{k}.pt", lambda: None)[0]["x"])
        (r8, x8), (r9, x9) = ext[k]
        t7 = rec["path"][2]["theta2"]
        nr = float(np.linalg.norm(x9) / np.linalg.norm(xp))
        rows[k] = dict(D_prime=r9["theta2"] / t7, norm_ratio=nr, dist=float(np.linalg.norm(x9 - xp) / np.linalg.norm(xp)),
                       rel_loss=(f.Lstar(x9) - f.Lstar(xp)) / abs(f.Lstar(xp)), theta2=[rec["path"][1]["theta2"], t7, r8["theta2"], r9["theta2"]],
                       cert=[r8["certified"], r9["certified"]], stop=[r8["stop"], r9["stop"]], degenerate=[r8.get("degenerate"), r9.get("degenerate")],
                       polish_certified=rec["certified"])
        v = rows[k]
        log(f"  {k:18s} ||theta*||^2 1e-5/1e-7/1e-8/1e-9 {', '.join(f'{t:.4g}' for t in v['theta2'])} | D' {v['D_prime']:.3f} | vs polish: "
            f"norm ratio {nr:.4f}, rel. distance {v['dist']:.3f}, rel. loss diff {v['rel_loss']:+.2e} | certified {v['cert']} | "
            f"{', '.join(s[:22] for s in v['stop'])}")
    pred1 = {}
    for c in CONDS:
        rr = [rows[f"{c}_P2_s{s}"] for s in SEEDS]
        mD, mN = float(np.median([r["D_prime"] for r in rr])), float(np.median([r["norm_ratio"] for r in rr]))
        pred1[c] = dict(D_prime_med=mD, norm_ratio_med=mN, D_ok=mD <= 1.5, norm_ok=0.9 <= mN <= 1.1,
                        rel_loss_med=float(np.median([r["rel_loss"] for r in rr])), dist_med=float(np.median([r["dist"] for r in rr])))
        log(f"  {c}: median D' {mD:.3f} (<= 1.5: {mD <= 1.5}); median norm ratio {mN:.4f} (in [0.9, 1.1]: {0.9 <= mN <= 1.1}); "
            f"median rel. distance {pred1[c]['dist_med']:.3f}; median rel. loss diff {pred1[c]['rel_loss_med']:+.2e}")
    allD, allN = all(v["D_ok"] for v in pred1.values()), all(v["norm_ok"] for v in pred1.values())
    read1 = ("BOTH HOLD: the path is bounded at mu = 1e-6; B2's D failure is a mu-versus-window effect" if allD and allN else
             "D' > 1.5: the window explanation fails; the branch-3 qualification stands unmodified" if not allD else
             "D' <= 1.5 BUT THE NORM APPROACH FAILS: a plateau elsewhere (see the loss differences; matching loss reads bounded)")
    log(f"  READING (check 1): {read1}")
    for l in lines:
        log("   " + l)
    res["check1"] = dict(rows=rows, cells=pred1, reading=read1)

    # ---------------- 2. P2 at mu = 1e-5
    log("\n==================== 2. P2 at the RBF study's mu = 1e-5 (16 new models, the Step 1 protocol end to end) ====================")
    keys = [f"{c}_P2m1e-05_s{s}" for c in CONDS for s in SEEDS]
    recs = BP.run_workers(keys, 4)
    pred2 = {}
    for c in CONDS:
        rr = [recs[f"{c}_P2m1e-05_s{s}"] for s in SEEDS]
        ncert = sum(r["certified"] for r in rr); mD = float(np.median([r["D"] for r in rr]))
        Dc = [r["D"] for r in rr if r["D_both_certified"]]
        reasons = {}
        for r in rr:
            if not r["certified"]:
                x = r["polish"].get("reason") or "PLATEAU by probe"; reasons[x] = reasons.get(x, 0) + 1
        pred2[c] = dict(certified=ncert, D_med=mD, D_cert_med=(float(np.median(Dc)) if Dc else None), reasons=reasons,
                        ok=ncert >= 7 and mD <= 1.5, D_per_seed=[r["D"] for r in rr], growth_med=float(np.median([r["growth"] for r in rr])))
        log(f"  {c} P2 (mu 1e-5): certified {ncert}/8 {reasons if reasons else ''}; median D {mD:.3f} (certified-only {I.fmt(pred2[c]['D_cert_med'], '.3f')}); "
            f"D per seed {', '.join(f'{d:.3f}' for d in pred2[c]['D_per_seed'])} -> prediction {'held' if pred2[c]['ok'] else 'FAILED'}")
    ok2 = all(v["ok"] for v in pred2.values()); ok1 = allD and allN
    read2 = ("PREDICTION HOLDS: the fix carries to B-splines at the main study's mu" + ("; with check 1, the D-leg failure reads as a calibration/window lesson" if ok1 else "")
             if ok2 else "FAILS WHILE CHECK 1 PASSES: boundedness stands, but D <= 1.5 is not portable across bases at a fixed mu (criterion sensitivity)" if ok1
             else "BOTH FAIL: the branch-3 qualification stands unmodified")
    log(f"  READING (check 2): {read2}")
    res["check2"] = dict(cells=pred2, reading=read2, models=recs)

    # ---------------- 3. the non-certifying independent P0/P1 models
    log("\n==================== 3. the non-certifying independent P0/P1 models (descriptive) ====================")
    rows3 = {}
    for k in ind:
        r = json.loads((I.W / f"bsp_rec_{k}.json").read_text())
        if r["certified"]:
            continue
        th = np.array(r["trust_theta2"], float); nrm = np.sqrt(th)
        short = len(nrm) <= 101
        a = nrm[0] if short else nrm[-101]
        g = float((nrm[-1] - a) / a)
        hc = r["hidden_corr"]["polished"]
        cls = ("runaway-like" if g > 0.01 and hc > 0.5 else "slow convergence" if g <= 0.01 and hc <= 0.5 else "mixed")
        rows3[k] = dict(growth_last100=g, short=short, n_iter=len(nrm) - 1, hidden_corr=hc, D=r["D"], cls=cls, reason=r["polish"].get("reason"),
                        theta2_last=float(th[-1]), theta2_early=float(th[0]))
        log(f"  {k:18s} {r['polish'].get('reason') or '':16s} {len(nrm) - 1:3d} it | ||theta|| growth over the last "
            f"{'all (short)' if short else '100'} iterations {100 * g:+.2f}% | ||theta||^2 {th[0]:.4g} -> {th[-1]:.4g} | largest hidden-unit corr. "
            f"{hc:.3f} | short-path D {r['D']:.3f} -> {cls}")
    cnt = {}
    for v in rows3.values():
        cnt[v["cls"]] = cnt.get(v["cls"], 0) + 1
    log(f"  classification counts: {cnt}")
    res["check3"] = dict(rows=rows3, counts=cnt)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    log("  wrote bspline_posthoc_results.json")


if __name__ == "__main__":
    main()

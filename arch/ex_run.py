"""arch/ex_run: EX-KAN (two-anchor excess model), see arch/ex_prereg.md (3938f8f).
Run from the repo root:
  python3 arch/ex_run.py step0 >> arch/ex_output.txt     (data checks only; no training)
Inputs: competition CSVs, SHA-256 logged and read once into memory by anchor.prepare."""
import hashlib
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

ARCH = Path(__file__).resolve().parent
REPO = ARCH.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(ARCH))

import numpy as np

import anchor as AN
from kan import co_props

warnings.filterwarnings("ignore")


def log(*a):
    print(*a, flush=True)


def comp(names, fracs):
    """Non-PC composition: {solvent: mole fraction in the full mixture}."""
    return {n: float(x) for n, x in zip(names.split(";"), fracs.split(";")) if n != "PC"}


def step0():
    log("#" * 110)
    log(f"arch/ex_run STEP 0 {time.strftime('%Y-%m-%d %H:%M:%S')} (prereg arch/ex_prereg.md, commit 3938f8f): data checks only, no training.")
    log("#" * 110)
    log("DISCREPANCIES / choices (files win; flags 1-11 are in the prereg):")
    log("  D1 'rows per co-solvent' = anchored training rows containing it (the LOCO rule); x_co = 1 - frac_PC (anchor.prepare), i.e. the TOTAL")
    log("     non-PC fraction, so for blends x_co includes the partner; the co-solvent's own mole fraction is reported alongside")
    log("  D2 distinct x_co values are counted after rounding to 4 decimals")
    log("  D3 a 'group' (for within-group x_co and molality drift) = source DOI, salt, T level (0.1 K), concentration unit and nominal")
    log("     concentration, and non-PC blend ratio (co-solvent fractions renormalised, rounded to 3 decimals); x_co varies inside a group")
    log("  D5 concentrations are continuous (e.g. EA: one source, one salt, 396 distinct concentrations in 635 rows), so D3 groups almost never")
    log("     repeat a concentration across x_co; a second grouping drops the concentration (source DOI, salt, T level, blend ratio) and")
    log("     reports distinct x_co there together with the overlap of their molality ranges (g is smooth in molality)")
    log("  D4 molality drift: (max - min) / median of molal within a mol/L group with >= 2 distinct x_co (molal = c / rho_mix there)")
    shas = {f: hashlib.sha256((AN.D / f).read_bytes()).hexdigest() for f in ["train.csv", "test.csv", "sample_submission.csv"]}
    log("inputs: " + ", ".join(f"{k} sha256 {v[:12]}" for k, v in shas.items()) + " (read once by anchor.prepare, held in memory)")
    tr, te, ss, X, Xte, ok, use, te_anc, has = AN.prepare(verbose=False)
    use = use.values
    t = tr[use].copy(); Xt = X[use]
    t["x_co"] = Xt.x_co.values; t["molal"] = Xt.molal.values
    t["comp"] = [comp(n, f) for n, f in zip(t.solvents, t.solvent_fracs_mol)]
    t["cos"] = [tuple(sorted(c)) for c in t.comp]
    tot = t.comp.apply(lambda c: sum(c.values()))
    t["blend"] = [tuple((n, round(c[n] / s, 3)) for n in sorted(c)) for c, s in zip(t.comp, tot)]
    t["grp2"] = list(zip(t.source_doi, t.salt_name, t.temperature_K.round(1), t.blend))
    t["grp"] = list(zip(t.source_doi, t.salt_name, t.temperature_K.round(1), t.conc_unit, t.conc_native.round(4), t.blend))
    cos = sorted({n for c in t.cos for n in c})
    log(f"\nanchored training rows: {len(t)}; co-solvents: {len(cos)}")
    R = dict(prereg="3938f8f", shas=shas, cosolvents={})

    # ---------------- 0a / 0b
    log("\n==================== STEP 0a/0b: per co-solvent ====================")
    for c in cos:
        m = t.cos.apply(lambda s: c in s).values
        d = t[m]
        xs = d.x_co.round(4)
        own = d.comp.apply(lambda q: q[c]).round(4)
        nd = int(xs.nunique())
        sole = int((d.cos.apply(len) == 1).sum())
        partners = Counter(tuple(n for n in s if n != c) for s in d.cos if len(s) > 1)
        # within-group x_co
        g = d.groupby("grp")
        per = g.x_co.apply(lambda v: v.round(4).nunique())
        drift = []
        for _, gg in g:
            if gg.conc_unit.iloc[0] == "mol/L" and gg.x_co.round(4).nunique() >= 2:
                drift.append(float((gg.molal.max() - gg.molal.min()) / gg.molal.median()))
        g2 = d.groupby("grp2")
        n2, ov = [], []
        for _, gg in g2:
            k = gg.x_co.round(4).nunique(); n2.append(k)
            if k >= 2:
                rng = gg.groupby(gg.x_co.round(4)).molal.agg(["min", "max"])
                ov.append(float(max(0.0, rng["max"].min() - rng["min"].max()) / max(gg.molal.max() - gg.molal.min(), 1e-12)))
        hi = d[d.x_co >= 0.9]
        rec = dict(rows=int(m.sum()), distinct_xco=nd, xco_min=float(d.x_co.min()), xco_max=float(d.x_co.max()), xco_sd=float(d.x_co.std(ddof=0)),
                   flag_le2=nd <= 2, sole_rows=sole, blend_only=sole == 0, partners={"+".join(k): v for k, v in partners.items()},
                   own_fracs=sorted(own.unique().tolist()), xco_values=sorted(xs.unique().tolist()),
                   groups=int(len(per)), groups_ge2=int((per >= 2).sum()), median_distinct_per_group=float(per.median()),
                   molL_groups=len(drift), drift_max=float(max(drift)) if drift else None, drift_median=float(np.median(drift)) if drift else None,
                   groups2=len(n2), groups2_ge2=int(sum(k >= 2 for k in n2)), median_distinct_group2=float(np.median(n2)),
                   molal_overlap_median=float(np.median(ov)) if ov else None, distinct_conc=int(d.conc_native.round(4).nunique()),
                   rows_ge09=int(len(hi)), rows_eq1=int((d.x_co >= 1 - 1e-9).sum()))
        R["cosolvents"][c] = rec
        log(f"\n  {c}: {rec['rows']} rows; distinct x_co {nd}; x_co range [{rec['xco_min']:.3f}, {rec['xco_max']:.3f}], SD {rec['xco_sd']:.3f}"
            f" -> {'FLAGGED (<= 2 distinct x_co)' if nd <= 2 else 'not flagged'}")
        log(f"    x_co values: {', '.join(f'{v:.3f}' for v in rec['xco_values'][:20])}{' ...' if nd > 20 else ''}")
        log(f"    sole co-solvent in {sole} rows" + ("  -> BLEND-ONLY" if sole == 0 else ""))
        if partners:
            log("    blend partners: " + "; ".join(f"{'+'.join(k)} ({v} rows)" for k, v in partners.most_common()))
            for k in partners:
                dd = d[d.cos.apply(lambda s: tuple(n for n in s if n != c) == k)]
                pairs = Counter(tuple((n, round(q[n], 3)) for n in sorted(q)) for q in dd.comp)
                log(f"      with {'+'.join(k)}: mole fractions (in the full mixture) " +
                    "; ".join(", ".join(f"{n} {x:.3f}" for n, x in p) + f" [{v}]" for p, v in sorted(pairs.items())))
        log(f"    groups (D3): {rec['groups']}; with >= 2 distinct x_co: {rec['groups_ge2']}; median distinct x_co per group {rec['median_distinct_per_group']:.0f}")
        log(f"    groups without concentration (D5): {rec['groups2']}; with >= 2 distinct x_co: {rec['groups2_ge2']}; median distinct x_co {rec['median_distinct_group2']:.0f}"
            + (f"; median shared molality range across x_co {rec['molal_overlap_median']:.2f} of the group's range" if ov else "")
            + f"; distinct concentrations {rec['distinct_conc']}")
        log(f"    mol/L groups with varying x_co: {rec['molL_groups']}" + (f"; molality drift max {rec['drift_max']:.3%}, median {rec['drift_median']:.3%}" if drift else ""))
        log(f"    0b: rows with x_co >= 0.9: {rec['rows_ge09']} (x_co = 1: {rec['rows_eq1']})" +
            (f"; at x_co {sorted(hi.x_co.round(3).unique().tolist())}" if len(hi) else ""))
    flagged = [c for c, r in R["cosolvents"].items() if r["flag_le2"]]
    unflagged = [c for c in cos if c not in flagged]
    log(f"\n  SUMMARY 0a: flagged (<= 2 distinct x_co): {flagged or 'none'}; unflagged: {unflagged}")
    log(f"  E3 needs >= 3 unflagged co-solvents: {len(unflagged)} -> {'OK' if len(unflagged) >= 3 else 'E3 would be INCONCLUSIVE'}")
    log(f"  E5 needs >= 1 flagged and >= 1 unflagged: {'OK' if flagged and unflagged else 'E5 would be INCONCLUSIVE'}")
    log(f"  blend-only co-solvents: {[c for c, r in R['cosolvents'].items() if r['blend_only']]}")
    R.update(flagged=flagged, unflagged=unflagged)

    # ---------------- 0c
    log("\n==================== STEP 0c: DEC test set, x_co distribution ====================")
    xte = Xte.x_co.values
    combos = Counter(te.combo)
    for nm, v in [("all test rows", xte), ("anchored test rows (KAN-delta rows)", xte[has])]:
        q = np.percentile(v, [0, 10, 25, 50, 75, 90, 100])
        log(f"  {nm}: n {len(v)}; distinct x_co {len(np.unique(v.round(4)))}; quantiles 0/10/25/50/75/90/100% " + " / ".join(f"{x:.3f}" for x in q))
        bins = np.histogram(v, bins=np.linspace(0, 1, 11))[0]
        log("    histogram (0.1-wide bins from 0 to 1): " + " ".join(str(b) for b in bins))
    log("  test combinations: " + "; ".join(f"{k} {v}" for k, v in combos.most_common()))
    log(f"  x_co >= 0.9 in test: {(xte >= 0.9).sum()} rows (anchored {(xte[has] >= 0.9).sum()}); x_co = 1: {(xte >= 1 - 1e-9).sum()}")
    R["dec_test"] = dict(n=int(len(xte)), n_anchored=int(has.sum()), quantiles=np.percentile(xte, [0, 25, 50, 75, 100]).tolist(),
                         quantiles_anchored=np.percentile(xte[has], [0, 25, 50, 75, 100]).tolist(), combos=dict(combos))

    # ---------------- 0d
    log("\n==================== STEP 0d: LOCO folds (as arch/sc_prereg.md flag 6) ====================")
    co_all = [[n for n in s.split(";") if n != "PC"] for s in tr.solvents]
    hoA = ((tr.combo == "EA+PC") & ok).values
    folds = {"A": (use & ~hoA, hoA & use)}
    for c in cos:
        tm = np.array([c in co_all[i] for i in range(len(tr))]) & use
        folds[c] = (use & ~tm, tm)
    rows = []
    for f, (a, b) in folds.items():
        rows.append(dict(fold=f, train=int(a.sum()), test=int(b.sum())))
        log(f"  {f:20s} train {a.sum():5d}  test {b.sum():5d}  {'>=20' if b.sum() >= 20 else '<20'}")
    sc0 = json.loads((ARCH / "sc_step0.json").read_text())["loco"]
    same = all(r["train"] == s["train"] and r["test"] == s["test"] for r, s in zip(rows, sc0))
    log(f"  identical to arch/sc Step 0b table (d898d73): {same}; fold A == EA fold: {bool((folds['A'][0] == folds['EA'][0]).all())}")
    R.update(loco=rows, loco_matches_sc=same)
    (ARCH / "ex_step0.json").write_text(json.dumps(R, indent=1, default=float))
    log("\nSTEP 0 done (data checks only). Waiting for the decision before Step 1.")


# ------------------------------------------------------------------ STEPS 1-2
SEP = ["EA", "EC", "Methylene chloride"]                     # Amendment 1, primary rule
LOW_EPS = ["EA", "2-MeTHF", "Methylene chloride", "THF", "DME", "Toluene", "3-Glyme", "2-Glyme"]
HIGH_EPS = ["EC", "Sulfolane"]
T25 = 298.15


def step12():
    import itertools
    import subprocess
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import sc_run as SR
    from ex_model import EXKAN, predict_ex, train_ex
    from kan import DELTA_IN, KAN, subset
    from pipeline import PROPS

    st = subprocess.run(["git", "status", "--porcelain", "arch/ex_prereg.md", "arch/ex_step0.json"], cwd=REPO, capture_output=True, text=True).stdout
    if st.strip() or "Amendment 1" not in (ARCH / "ex_prereg.md").read_text():
        raise SystemExit("REFUSED: prereg amendment / Step 0 not committed")
    log("\n" + "#" * 110)
    log(f"arch/ex_run STEPS 1-2 {time.strftime('%Y-%m-%d %H:%M:%S')} (prereg 3938f8f, Step 0 4c268c1, Amendment 1 fa6ae2a)")
    log("#" * 110)
    log("DISCREPANCIES / choices (files win):")
    log("  D6 baseline = cached scbase_* early models from arch/sc (SHA-256 re-checked against gauge_weights/sc_index.json, read into memory)")
    log("  D7 s at 25 C uses each co-solvent's PURE properties from pipeline.PROPS (eps, ln eta, M) and 1000/298.15, scaled by final.py's Scaler;")
    log("     values outside the scaler's [-1, 1] range are extrapolation and are marked")
    log("  D8 '25 C rows' for the observed-property report = rows with |T - 298.15| < 1 K (none exist if no T level is that close)")
    log("  D9 E3 spread for the descriptive ratio = SD over all 10 co-solvents of their seed-median s (Amendment 1)")
    S = SR.setup()
    n_salt = len(S["salts"])
    folds = dict(S["folds"]); folds["full"] = dict(train=S["use"], test=np.zeros_like(S["use"]))
    B = S["B"]
    R0 = json.loads((ARCH / "ex_step0.json").read_text())
    flag_old = R0["flagged"]; unflag_old = R0["unflagged"]
    cos = sorted(R0["cosolvents"])
    FLAG = [c for c in cos if c not in SEP]

    # ---------------- Step 1
    log("\n==================== STEP 1: training (EX-KAN, 10 seeds per fold; AdamW as final.py, 107 epochs; no polishing) ====================")
    idx = SR._index()
    ex, base = {}, {}
    for f in folds:
        src = "A" if f == "EA" else f
        b = subset(B, folds[src]["train"])
        t0 = time.time(); nets = []
        for s in range(10):
            obj, _ = SR.cached(f"ex_{src.replace(' ', '_')}_s{s}.pt", lambda: dict(early=train_ex(n_salt, b, 107, s).state_dict()))
            n = EXKAN(n_salt); n.load_state_dict(obj["early"]); nets.append(n.eval())
        ex[f] = nets
        if f != "full":
            names = [f"scbase_{src.replace(' ', '_')}_s{s}.pt" for s in range(10)]
            missing = [n for n in names if n not in idx]
            if missing:
                raise SystemExit(f"STOP: baseline cache missing {missing[:3]} (flag 9: reuse, never retrain)")
            base[f] = [SR.base_early(S, src, s) for s in range(10)]
        log(f"  fold {f:20s}: 10 EX models ({time.time() - t0:.1f} s){'' if f == 'full' else '; 10 baseline models reloaded (SHA ok)'}"
            + (" (= fold A, flag 10)" if f == "EA" else ""))

    # ---------------- Step 2: accuracy
    log("\n==================== STEP 2: measurements ====================")
    acc = {}
    for kind, M in [("ex", ex), ("base", base)]:
        per = {}
        for f in S["folds"]:
            bt = subset(B, folds[f]["test"])
            p = np.mean([(predict_ex(n, bt) if kind == "ex" else SR.base_pred(n, bt)) for n in M[f][:5]], 0)
            per[f] = dict(rmse=SR.rmse(p, bt["y"].numpy()), n=int(folds[f]["test"].sum()))
        lo = {f: v for f, v in per.items() if f != "A"}
        acc[kind] = dict(per=per, loco_unweighted=float(np.mean([v["rmse"] for v in lo.values()])),
                         loco_rowweighted=float(np.sqrt(sum(v["rmse"] ** 2 * v["n"] for v in lo.values()) / sum(v["n"] for v in lo.values()))),
                         loco_ge20=float(np.mean([v["rmse"] for v in lo.values() if v["n"] >= 20])), foldA=per["A"]["rmse"])
    log("  accuracy (5-seed ensemble, seeds 0-4; RMSE on delta = log k):")
    log(f"    {'fold':20s} {'n':>5s} {'EX':>8s} {'base':>8s}")
    for f in S["folds"]:
        log(f"    {f:20s} {acc['ex']['per'][f]['n']:5d} {acc['ex']['per'][f]['rmse']:8.4f} {acc['base']['per'][f]['rmse']:8.4f}")
    for k, nm in [("loco_unweighted", "LOCO mean, unweighted (E1)"), ("loco_rowweighted", "LOCO, row-weighted (pooled RMSE)"),
                  ("loco_ge20", "LOCO mean, folds >= 20 test rows"), ("foldA", "fold A (E2)")]:
        log(f"    {nm:38s} EX {acc['ex'][k]:.4f}  base {acc['base'][k]:.4f}  ratio {acc['ex'][k] / acc['base'][k]:.3f}")
    log("    note: fold A and the LOCO EA fold are the same rows (flag 10), so E2 and the EA term of E1 are not independent")

    # ---------------- pure-co-solvent offset s (full set)
    sc_ = S["sc"]
    def scale(j, v):
        return 2 * (v - sc_.lo[j]) / (sc_.hi[j] - sc_.lo[j]) - 1
    raw = {c: [PROPS[c][2], float(np.log(PROPS[c][3])), PROPS[c][0], 1000 / T25] for c in cos}
    cols = [DELTA_IN.index(k) for k in ["eps_co", "lneta_co", "M_co", "invT"]]
    Z = np.zeros((len(cos), len(DELTA_IN)))
    for r, c in enumerate(cos):
        for v, j in zip(raw[c], cols):
            Z[r, j] = scale(j, v)
    Zt = torch.tensor(Z, dtype=torch.float32)
    sv = np.array([n.s_of(Zt).detach().double().numpy() for n in ex["full"]])          # (10 seeds, n_co)
    med, sd = np.median(sv, 0), sv.std(0, ddof=0)
    spread = float(np.std(med, ddof=0))
    log("\n  pure-co-solvent offset s at 25 C (full set, 10 seeds; s = log k(pure co-solvent) - log k(PC) under the model):")
    log(f"    {'co-solvent':20s} {'eps':>6s} {'median s':>9s} {'seed SD':>8s}  rule(primary)  rule(<=2)  extrapolated inputs")
    for r, c in enumerate(cos):
        ext = [DELTA_IN[j] for j in cols if abs(Z[r, j]) > 1]
        log(f"    {c:20s} {PROPS[c][2]:6.2f} {med[r]:+9.3f} {sd[r]:8.3f}  {'separable' if c in SEP else 'flagged  '}      "
            f"{'flagged' if c in flag_old else 'unflag '}    {', '.join(ext) or '-'}")
    log(f"    spread of s across co-solvents (SD of seed medians, D9): {spread:.3f}")
    log("    separable co-solvents (descriptive, Amendment 1): " + "; ".join(f"{c} seed SD {sd[cos.index(c)]:.3f} = {sd[cos.index(c)] / spread:.2f} x spread" for c in SEP))
    # observed 25 C rows
    tr = S["tr"]
    near = np.abs(tr.temperature_K.values - T25) < 1.0
    obs = []
    for c in cos:
        m = S["use"] & near & np.array([c in str(x).split(";") for x in tr.solvents])
        if m.any():
            bx = B["x"][torch.tensor(m)]
            v = np.array([n.s_of(bx).detach().numpy() for n in ex["full"]])
            obs.append((c, int(m.sum()), float(np.median(v)), float(np.median(v.std(0)))))
    log("    s at observed (blend) properties on 25 C rows (D8): " + ("; ".join(f"{c} n={k} median {m_:+.3f} (seed SD {d:.3f})" for c, k, m_, d in obs) or "no rows within 1 K of 298.15 K"))

    def e3(group):
        ii = [cos.index(c) for c in group]
        cs = [float(np.corrcoef(sv[a, ii], sv[b, ii])[0, 1]) for a, b in itertools.combinations(range(10), 2)]
        return float(np.median(cs)), cs
    e3p, _ = e3(SEP); e3s, _ = e3(unflag_old)
    low = [cos.index(c) for c in LOW_EPS]; high = [cos.index(c) for c in HIGH_EPS]
    e4 = sv[:, low].mean(1) < sv[:, high].mean(1)
    e4_ec = sv[:, low].mean(1) < sv[:, cos.index("EC")]
    e4_sf = sv[:, low].mean(1) < sv[:, cos.index("Sulfolane")]
    def e5(flagged, sep):
        a = float(np.median(sd[[cos.index(c) for c in flagged]])); b = float(np.median(sd[[cos.index(c) for c in sep]]))
        return a, b, a / b
    e5p = e5(FLAG, SEP); e5s = e5(flag_old, unflag_old)
    log(f"\n  E3 primary (separable {SEP}): median pairwise corr over 45 seed pairs {e3p:+.3f}   [Methylene chloride marginal: 8/80 groups, ~4% molality drift]")
    log(f"  E3 secondary (<= 2 rule, 7 unflagged): {e3s:+.3f}")
    log(f"  E4: mean s(low eps) < mean s(EC, Sulfolane) in {int(e4.sum())}/10 seeds; vs EC alone {int(e4_ec.sum())}/10; vs Sulfolane alone {int(e4_sf.sum())}/10")
    log(f"      per seed: mean low {np.round(sv[:, low].mean(1), 3).tolist()}; EC {np.round(sv[:, cos.index('EC')], 3).tolist()}; Sulfolane {np.round(sv[:, cos.index('Sulfolane')], 3).tolist()}")
    log(f"  E5 primary: median seed-SD flagged (7) {e5p[0]:.3f} vs separable (3) {e5p[1]:.3f} -> ratio {e5p[2]:.2f}")
    log(f"  E5 secondary (<= 2 rule): flagged (3) {e5s[0]:.3f} vs unflagged (7) {e5s[1]:.3f} -> ratio {e5s[2]:.2f}")

    # ---------------- verdicts
    V = {}
    V["E1"] = (acc["ex"]["loco_unweighted"] <= 0.95 * acc["base"]["loco_unweighted"],
               f"EX {acc['ex']['loco_unweighted']:.4f} vs 0.95 x base {0.95 * acc['base']['loco_unweighted']:.4f} (ratio {acc['ex']['loco_unweighted'] / acc['base']['loco_unweighted']:.3f}; "
               f"row-weighted {acc['ex']['loco_rowweighted']:.4f}/{acc['base']['loco_rowweighted']:.4f}; >=20 rows {acc['ex']['loco_ge20']:.4f}/{acc['base']['loco_ge20']:.4f})")
    V["E2"] = (acc["ex"]["foldA"] <= 0.142, f"EX fold A {acc['ex']['foldA']:.4f} (base this run {acc['base']['foldA']:.4f}); not independent of E1's EA fold")
    V["E3"] = (e3p >= 0.9, f"primary {e3p:+.3f} over 3 separable (weak test; MeCl marginal); secondary {e3s:+.3f} over 7")
    V["E4"] = (int(e4.sum()) >= 8, f"{int(e4.sum())}/10 seeds (vs EC alone {int(e4_ec.sum())}/10, vs Sulfolane alone {int(e4_sf.sum())}/10)")
    V["E5"] = (e5p[2] >= 2, f"primary ratio {e5p[2]:.2f} ({e5p[0]:.3f}/{e5p[1]:.3f}); secondary {e5s[2]:.2f}")
    log("\n==================== VERDICTS (pre-registered E1-E5; E3/E5 by Amendment 1 primary rule) ====================")
    for k, (h, t) in V.items():
        log(f"  {k:3s} {'HELD' if h else 'FAILED':6s} {t}")

    # ---------------- figure
    FIG = ARCH / "figs"; FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    order = np.argsort(med)
    for k, r in enumerate(order):
        ax.scatter(np.full(10, k), sv[:, r], s=14, alpha=0.7, color="C0" if cos[r] in SEP else "C3")
        ax.plot([k - 0.3, k + 0.3], [med[r]] * 2, color="k")
    ax.set_xticks(range(len(cos))); ax.set_xticklabels([f"{cos[r]}\n(eps {PROPS[cos[r]][2]:.1f})" for r in order], fontsize=7)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("s at 25 C, pure properties"); ax.set_title("EX-KAN pure-co-solvent offset s, full set, 10 seeds (blue: separable, red: flagged; Amendment 1)")
    fig.tight_layout(); fig.savefig(FIG / "ex_offsets.png", dpi=110); plt.close(fig)

    R = dict(accuracy=acc, s=dict(cosolvents=cos, values=sv.tolist(), median=med.tolist(), sd=sd.tolist(), spread=spread, observed25=obs),
             E3=dict(primary=e3p, secondary=e3s), E4=dict(n=int(e4.sum()), ec=int(e4_ec.sum()), sulfolane=int(e4_sf.sum())),
             E5=dict(primary=e5p, secondary=e5s), verdicts={k: dict(held=bool(h), text=t) for k, (h, t) in V.items()})
    (ARCH / "ex_results.json").write_text(json.dumps(R, indent=1, default=float))


if __name__ == "__main__":
    {"step0": step0, "step12": step12}[sys.argv[1]]()

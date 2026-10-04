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


if __name__ == "__main__":
    {"step0": step0}[sys.argv[1]]()

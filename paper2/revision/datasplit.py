"""paper2/revision/datasplit.py: DESCRIPTIVE facts for Appendix H ("Data, splits and chronology"). No training, no model.
Reads the competition CSVs (git-ignored) through the committed data path (anchor.prepare, arch/sc_run.setup) and the git history.
Output: datasplit_output.txt, datasplit_results.json. Run from the repo root: /usr/bin/python3 paper2/revision/datasplit.py"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "arch"):
    sys.path.insert(0, str(p))

import numpy as np
import pandas as pd

import anchor as AN
import sc_run as SR
from kan import DELTA_IN

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.append(s)


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def anchor_how(series, c, T):
    """mirror of anchor.anchor_value, returning how the anchor was obtained (T: grid level or 1/T interpolation; c: grid point or interpolation)"""
    levels = np.array(sorted(series))
    near = np.abs(levels - T) < AN.T_TOL
    used = [levels[near][0]] if near.any() else [levels[np.searchsorted(levels, T) - 1], levels[np.searchsorted(levels, T)]]
    c_exact = all(np.any(np.isclose(series[L][0], c, rtol=1e-9, atol=0)) for L in used)
    return ("T level" if near.any() else "T interpolated (linear in 1/T)"), ("c on grid" if c_exact else "c interpolated (linear)")


def main():
    raw_tr = pd.read_csv(AN.D / "train.csv"); raw_te = pd.read_csv(AN.D / "test.csv")
    tr, te, ss, X, Xte, ok, use, te_anc, has = AN.prepare(verbose=False)
    S = SR.setup()
    res = {}
    log("#" * 100)
    log("paper2/revision/datasplit.py: DESCRIPTIVE data, split and chronology facts for Appendix H (no training)")
    log("#" * 100)
    # ---------------- b) target, unit, low conductivities
    res["target"] = dict(train_cols=raw_tr.columns.tolist(), test_cols=raw_te.columns.tolist(),
                         log10_identity=bool(np.allclose(raw_tr.log_k, np.log10(raw_tr.k))), k_min=float(raw_tr.k.min()),
                         n_k_nonpositive=int((raw_tr.k <= 0).sum()), n_below_0p01=int((raw_tr.k < 0.01).sum()),
                         filter="tr = tr[tr.k >= 0.01] (anchor.py, prepare)", logk_min_clip=AN.LOGK_MIN)
    log(f"\n[b] log_k == log10(k) on all train rows: {res['target']['log10_identity']}; k unit mS/cm (metaData.csv); k min {raw_tr.k.min():.3g}; "
        f"rows with k <= 0: {res['target']['n_k_nonpositive']}; rows with k < 0.01 mS/cm dropped: {res['target']['n_below_0p01']}; "
        f"submissions clipped at log_k >= {AN.LOGK_MIN}")
    # ---------------- d) inclusion counts
    pc_mix = ok & tr.has_pc & (tr.combo != "PC")
    series = {k: AN.build_series(g) for k, g in tr[tr.combo == "PC"].groupby(["source_doi", "salt_name"])}
    cand = pc_mix & np.array([(d, s) in series for d, s in zip(tr.source_doi, tr.salt_name)])
    cnt = dict(train_rows=len(raw_tr), test_rows=len(raw_te), after_k_filter=len(tr), known_solvents=int(ok.sum()),
               pure_pc_rows=int((ok & (tr.combo == "PC")).sum()), pc_mixture_rows=int(pc_mix.sum()), no_pc_rows=int((ok & ~tr.has_pc).sum()),
               with_same_source_pc_series=int(cand.sum()), anchored=int(use.sum()), not_anchored_outside_range=int((cand & ~use).sum()),
               pure_pc_series=len(series), test_with_pc=int(te.has_pc.sum()), test_anchored=int(has.sum()))
    res["counts"] = cnt
    log("\n[d] row inclusion (anchor.prepare): " + "; ".join(f"{k} {v}" for k, v in cnt.items()))
    # ---------------- a) anchor: how obtained, per row
    how = [anchor_how(series[(r.source_doi, r.salt_name)], r.conc_native, r.temperature_K) for r in tr[use].itertuples()]
    ht = pd.Series([h[0] for h in how]).value_counts().to_dict(); hc = pd.Series([h[1] for h in how]).value_counts().to_dict()
    res["anchor"] = dict(T=ht, c=hc, T_TOL=AN.T_TOL, test_rule="same-salt pure-PC series from SAME_SOURCE articles, mol/kg test rows only; mean over covering series",
                         test_sources=sorted(AN.SAME_SOURCE) if hasattr(AN, "SAME_SOURCE") else None)
    log(f"\n[a] anchored train rows by temperature handling: {ht}; by concentration handling: {hc} (T_TOL {AN.T_TOL} K; no extrapolation)")
    # ---------------- c) composition representation
    ua = tr[use]
    res["composition"] = dict(frac_basis_anchored=ua.frac_basis.value_counts().to_dict(), conc_unit_anchored=ua.conc_unit.value_counts().to_dict(),
                              n_solvents_anchored=ua.solvents.str.count(";").add(1).value_counts().sort_index().to_dict(),
                              n_solvents_test_anchored=te[has].solvents.str.count(";").add(1).value_counts().sort_index().to_dict(),
                              conc_unit_test_anchored=te[has].conc_unit.value_counts().to_dict())
    log(f"\n[c] anchored train rows: frac_basis {res['composition']['frac_basis_anchored']}; conc_unit {res['composition']['conc_unit_anchored']}; "
        f"solvents per mixture {res['composition']['n_solvents_anchored']}; DEC anchored test rows: solvents {res['composition']['n_solvents_test_anchored']}, "
        f"conc_unit {res['composition']['conc_unit_test_anchored']}")
    # ---------------- e) folds and fold A
    folds = {f: dict(n_test=int(m["test"].sum()), n_train=int(m["train"].sum())) for f, m in S["folds"].items()}
    res["folds"] = folds
    log("\n[e] folds (arch/sc_run.setup): " + "; ".join(f"{f} test {v['n_test']} / train {v['n_train']}" for f, v in folds.items()))
    A = tr[S["folds"]["A"]["test"]]
    xa = X.loc[S["folds"]["A"]["test"], "x_co"]
    srcA = A.source_doi.value_counts().to_dict(); saltA = A.salt_name.value_counts().to_dict()
    pairs = A.groupby(["source_doi", "salt_name"]).size()
    # do fold A's sources contribute training rows (other mixtures) or only anchors?
    trainA = tr[S["folds"]["A"]["train"]]
    src_in_train = {d: int((trainA.source_doi == d).sum()) for d in srcA}
    res["foldA"] = dict(n=len(A), combos=A.combo.value_counts().to_dict(), salts=saltA, sources=srcA, source_salt_pairs={f"{d} | {s}": int(n) for (d, s), n in pairs.items()},
                        T_range=[float(A.temperature_K.min()), float(A.temperature_K.max())], n_T_levels=int(A.temperature_K.round(1).nunique()),
                        x_co_range=[float(xa.min()), float(xa.max())], n_x_co=int(xa.round(9).nunique()),
                        frac_basis=A.frac_basis.value_counts().to_dict(), conc_unit=A.conc_unit.value_counts().to_dict(),
                        conc_range=[float(A.conc_native.min()), float(A.conc_native.max())],
                        training_rows_from_same_sources=src_in_train,
                        pure_pc_rows_same_sources=int(((tr.combo == "PC") & tr.source_doi.isin(list(srcA))).sum()))
    fa = res["foldA"]
    log(f"\n[e] fold A (EA+PC held out): {fa['n']} rows; salts {saltA}; sources {srcA}; T {fa['T_range'][0]}-{fa['T_range'][1]} K ({fa['n_T_levels']} levels); "
        f"x_co {fa['x_co_range'][0]:.4f}-{fa['x_co_range'][1]:.4f} ({fa['n_x_co']} values, mole fraction); frac_basis {fa['frac_basis']}; "
        f"conc_unit {fa['conc_unit']} ({fa['conc_range'][0]}-{fa['conc_range'][1]}); training rows (fold A train) from the same sources: {src_in_train}; "
        f"pure-PC rows of those sources (anchor series, not in any fold's test set): {fa['pure_pc_rows_same_sources']}")
    res["scaler"] = "kan.Scaler min-max to [-1, 1] over the anchored train rows AND the anchored DEC test rows' inputs (arch/sc_run.setup, as final.py); no labels"
    log(f"\n[a/e] input scaling: {res['scaler']}")
    # ---------------- f) DEC labels
    res["dec"] = dict(test_has_k="k" in raw_te.columns, test_has_log_k="log_k" in raw_te.columns, n_test=len(raw_te),
                      dec_in_train=int(raw_tr.solvents.str.split(";").apply(lambda l: "DEC" in l).sum()),
                      dec_in_test=int(raw_te.solvents.str.split(";").apply(lambda l: "DEC" in l).sum()))
    log(f"\n[f] test.csv has k: {res['dec']['test_has_k']}, log_k: {res['dec']['test_has_log_k']}; rows containing DEC: train {res['dec']['dec_in_train']}, "
        f"test {res['dec']['dec_in_test']} of {len(raw_te)}")
    # ---------------- g) chronology: committed outputs that evaluate on fold A
    rows = []
    files = [f for f in git("ls-files").splitlines() if f.endswith(("_output.txt",)) or f.endswith("prereg.md") or f == "electrolyte_case_study_summary.md"]
    for f in files:
        txt = (REPO / f).read_text(errors="ignore")
        if "fold A" in txt or "PC+EA" in txt or "EA+PC" in txt:
            c = git("log", "--diff-filter=A", "--follow", "--format=%h|%ad|%s", "--date=short", "--", f).splitlines()
            if c:
                h, d, s = c[-1].split("|", 2)
                rows.append(dict(file=f, commit=h, date=d, subject=s, n_mentions=txt.count("fold A") + txt.count("PC+EA") + txt.count("EA+PC")))
    rows.sort(key=lambda r: (r["date"], r["commit"]))
    res["foldA_chronology"] = rows
    log(f"\n[g] committed files that report on fold A (first commit), {len(rows)}:")
    for r in rows:
        log(f"  {r['date']} {r['commit']} {r['file']} ({r['n_mentions']} mentions)")
    (HERE / "datasplit_results.json").write_text(json.dumps(res, indent=1, default=str))
    (HERE / "datasplit_output.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()

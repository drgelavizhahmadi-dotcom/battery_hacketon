"""paper2/revision/uncertainty.py: DESCRIPTIVE uncertainty from committed results (no training, no optimisation).
  a) Wilson 95% intervals for the certification fractions quoted in the main text;
  b) composition-grouped bootstrap 95% intervals for fold A's RMSE (early; polished mu = 0, 1e-6, 1e-4), from archived
     weights (SHA-verified against weights_manifest.txt, held in memory); the logged RMSEs must reproduce or the script stops;
  c) stop reasons across all studies, solver errors and hangs separated from iteration and wall-clock caps.
Output: uncertainty_output.txt, uncertainty_results.json. Run from the repo root: /usr/bin/python3 paper2/revision/uncertainty.py"""
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "arch", REPO / "paper2"):
    sys.path.insert(0, str(p))

import numpy as np
import torch

LOG = []
Z = 1.959963984540054
B_RESAMPLES, B_SEED = 10_000, 0


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.append(s)


def J(rel):
    return json.loads((REPO / rel).read_text())


def wilson(k, n):
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def part_a():
    rows = []
    def add(study, cell, k, n, src):
        lo, hi = wilson(k, n)
        rows.append(dict(study=study, cell=cell, k=int(k), n=int(n), frac=k / n, lo=lo, hi=hi, source=src))
    sr = J("paper2/synth_results.json")["cells"]
    for c in ("independent", "correlated"):
        for l in ("P0", "P1", "P2"):
            add("synthetic 2x3 (RBF)", f"{c} {l}", sr[f"{c}|{l}"]["certified"], 8, "paper2/synth_results.json")
    add("synthetic 2x3 (RBF)", "correlated P0+P1", sr["correlated|P0"]["certified"] + sr["correlated|P1"]["certified"], 16, "paper2/synth_results.json")
    add("synthetic 2x3 (RBF)", "P2, both conditions", sr["independent|P2"]["certified"] + sr["correlated|P2"]["certified"], 16, "paper2/synth_results.json")
    add("synthetic 2x3 (RBF)", "independent P0+P1", sr["independent|P0"]["certified"] + sr["independent|P1"]["certified"], 16, "paper2/synth_results.json")
    dc = J("paper2/revision/dose_results.json")["cells"]
    for k, v in dc.items():
        add("dose-response", k.replace("_r", ", rho = "), v["certified"], 8, "paper2/revision/dose_results.json")
    bs = J("paper2/revision/bspline_results.json")["step1"]["cells"]
    for k, v in bs.items():
        add("B-spline (mu = 1e-6)", k.replace("_", " "), v["certified"], 8, "paper2/revision/bspline_results.json")
    add("B-spline (mu = 1e-6)", "correlated P0+P1", bs["correlated_P0"]["certified"] + bs["correlated_P1"]["certified"], 16, "paper2/revision/bspline_results.json")
    c2 = J("paper2/revision/bspline_posthoc_results.json")["check2"]["cells"]
    for k, v in c2.items():
        add("B-spline P2 at mu = 1e-5 (post hoc)", k, v["certified"], 8, "paper2/revision/bspline_posthoc_results.json")
    fr = J("paper2/revision/freeze_results.json")
    cert = lambda d: sum(bool(m.get("certified")) for m in d.values() if isinstance(m, dict))
    for key, lab in (("A-R", "layer 1 frozen, real"), ("A-S-P0", "layer 1 frozen, synthetic P0"), ("A-S-P1", "layer 1 frozen, synthetic P1"),
                     ("C", "one-layer KAN, correlated"), ("D", "L2 at polish time (Arm D)")):
        add("freeze test", lab, cert(fr[key]), len(fr[key]), "paper2/revision/freeze_results.json")
    add("freeze test", "layer 2 frozen, synthetic P0+P1", cert(fr["B-S-P0"]) + cert(fr["B-S-P1"]), len(fr["B-S-P0"]) + len(fr["B-S-P1"]), "paper2/revision/freeze_results.json")
    ce = J("certify_results.json")
    cand = {(m, t) for m, t in ce["candidates"]}
    for m, lab in (("1e-06", "1e-6"), ("1e-05", "1e-5"), ("0.0001", "1e-4")):
        add("real data, null-space weight (seeds at a numerical minimum)", f"mu = {lab}", sum((m, f"ind_{i}") in cand for i in range(6)), 6, "certify_results.json (candidates)")
    add("real data, null-space weight", "mu = 1e-4, second-order certificate (W1)", sum(ce["cert"][f"0.0001|ind_{i}"]["certified"] for i in range(6)), 6, "certify_results.json")
    sc = J("arch/sc_results.json")
    for kind, lab in (("base", "baseline KAN-delta, fold A"), ("sc", "shared-curve KAN, fold A")):
        add("structured alternatives", lab, sum(p["cert"]["certified"] for p in sc["polish"][kind]), len(sc["polish"][kind]), "arch/sc_results.json")
    log("\n==================== a) Wilson 95% intervals for certification fractions ====================")
    for r in rows:
        log(f"  {r['study']:58s} {r['cell']:42s} {r['k']:2d}/{r['n']:<2d}  {r['frac']:.3f}  [{r['lo']:.3f}, {r['hi']:.3f}]  ({r['source']})")
    return rows


def part_b():
    import sc_run as SR
    from close import FlatQ
    from kan import DELTA_IN, KAN, delta_head, predict, subset
    from stage2 import extract_verified, in_memory
    from trust import Flat
    S = SR.setup()
    tr = S["tr"]; fa = S["folds"]["A"]
    names = [f"scbase_A_s{s}.pt" for s in range(5)] + [f"scbase_A_s{s}_pol.pt" for s in range(5)] + \
            [f"close_foldA_1e-06_s{s}.pt" for s in range(5)] + [f"certify_foldA_0.0001_s{s}.pt" for s in range(5)]
    log("\n==================== b) fold A RMSE, composition-grouped bootstrap (archived weights) ====================")
    blobs = in_memory(extract_verified(names))
    bt, bh = subset(S["B"], fa["train"]), subset(S["B"], fa["test"])
    btd, bhd = SR.dbl(bt), SR.dbl(bh)
    used_f = np.unique(S["B"]["s"].numpy()[fa["train"]]).tolist()
    d, n_salt = len(DELTA_IN), len(S["salts"])
    y = tr.log_k.values[fa["test"]]; anc = tr.anchor.values[fa["test"]]
    P = {"early": [], "polished mu = 0": [], "polished mu = 1e-6": [], "polished mu = 1e-4": []}
    for s in range(5):
        sd = torch.load(io.BytesIO(blobs[f"scbase_A_s{s}.pt"]), weights_only=False)
        sd = sd["early"] if isinstance(sd, dict) and "early" in sd else sd
        net0 = KAN(d, [6, 1], n_salt); net0.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False); net0.eval()
        P["early"].append(predict(net0, delta_head, bh))
        pol = torch.load(io.BytesIO(blobs[f"scbase_A_s{s}_pol.pt"]), weights_only=False)["x"].numpy()
        net64 = KAN(d, [6, 1], n_salt); net64.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False); net64.eval()
        P["polished mu = 0"].append(SR.base_pred(Flat(net64, btd, used_f).to_kan(pol, d, n_salt), bhd))
        for mu, nm, key in (("1e-06", f"close_foldA_1e-06_s{s}.pt", "polished mu = 1e-6"), ("0.0001", f"certify_foldA_0.0001_s{s}.pt", "polished mu = 1e-4")):
            x = torch.load(io.BytesIO(blobs[nm]), weights_only=False)
            x = x.numpy() if torch.is_tensor(x) else np.asarray(x)
            P[key].append(predict(FlatQ(net0, btd, used_f, float(mu)).to_kan(x, d, n_salt).float(), delta_head, bh))
    LOGGED = {"early": 0.1420, "polished mu = 0": 0.8147, "polished mu = 1e-6": 1.0262, "polished mu = 1e-4": 0.3345}
    err = {k: anc + np.mean(v, 0) - y for k, v in P.items()}
    rm = {k: float(np.sqrt(np.mean(e ** 2))) for k, e in err.items()}
    log("  reconstructed ensemble RMSE: " + ", ".join(f"{k} {rm[k]:.4f} (logged {LOGGED[k]:.4f})" for k in rm))
    if not all(abs(rm[k] - LOGGED[k]) < 1e-4 for k in rm):
        raise SystemExit("STOP: the archived fold A ensembles do not reproduce the logged RMSEs")
    A = tr[fa["test"]]
    comp = A.solvent_fracs.astype(str) + "|" + A.conc_native.round(9).astype(str) + "|" + A.salt_name
    groups = {g: np.where(comp.values == g)[0] for g in sorted(set(comp))}
    keys = list(groups)
    n_src = A.source_doi.nunique()
    log(f"  fold A rows {len(A)}; compositions (solvent fractions x concentration x salt) {len(keys)}; rows per composition "
        f"{min(len(v) for v in groups.values())}-{max(len(v) for v in groups.values())}; source articles {n_src}")
    rng = np.random.default_rng(B_SEED)
    draws = rng.integers(0, len(keys), size=(B_RESAMPLES, len(keys)))
    out = {}
    for k, e in err.items():
        se = {g: (float(np.sum(e[idx] ** 2)), len(idx)) for g, idx in groups.items()}
        ssum = np.array([se[g][0] for g in keys]); cnt = np.array([se[g][1] for g in keys])
        bs = np.sqrt(ssum[draws].sum(1) / cnt[draws].sum(1))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[k] = dict(rmse=rm[k], logged=LOGGED[k], lo=float(lo), hi=float(hi))
        log(f"  {k:22s} RMSE {rm[k]:.4f}  composition bootstrap 95% [{lo:.4f}, {hi:.4f}]")
    src_note = ("not computed: fold A has a single source article, so a source-grouped bootstrap is degenerate" if n_src == 1 else "see results")
    log(f"  source-article bootstrap: {src_note}")
    return dict(ensembles=out, n_rows=len(A), n_compositions=len(keys), n_sources=int(n_src), resamples=B_RESAMPLES, seed=B_SEED,
                grouping="solvent_fracs x conc_native x salt_name", source_bootstrap=src_note)


STUDY_FILES = {"synthetic 2x3": "paper2/synth_results.json", "dose-response": "paper2/revision/dose_results.json",
               "B-spline Step 1": "paper2/revision/bspline_results.json", "B-spline post hoc": "paper2/revision/bspline_posthoc_results.json",
               "freeze test": "paper2/revision/freeze_results.json", "infimum test": "paper2/revision/infimum_results.json",
               "degeneracy follow-up": "paper2/revision/degeneracy_results.json", "shared-curve / baseline": "arch/sc_results.json"}


def classify(s):
    t = s.lower()
    if "hang" in t or "watchdog" in t:
        return "solver error: hang (killed)"
    if "killed" in t or "hard kill" in t:
        return "hard kill (wall-clock cap + 5 min)"
    if "solver error" in t or "cholesky" in t or "nan" in t:
        return "solver error"
    if "iteration cap" in t:
        return "iteration cap"
    if "wall-clock" in t or "wall clock" in t or "2 h" in t:
        return "wall-clock cap"
    if "certified" in t or "terminated" in t:
        return "certification-based termination"
    if "scipy termination" in t or "converged" in t or "stage2 criterion" in t or "normal completion" in t:
        return "optimiser termination (tolerance)"
    if "not reached" in t:
        return "point not reached (path stopped)"
    return "other: " + s[:40]


def walk_stops(o, acc):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "stop" and isinstance(v, str):
                acc.append(v)
            elif k == "status" and isinstance(v, str) and v.startswith("not reached"):
                acc.append(v)
            else:
                walk_stops(v, acc)
    elif isinstance(o, list):
        for v in o:
            walk_stops(v, acc)


def part_c():
    log("\n==================== c) stop reasons across studies (every 'stop' field, and every path point not reached, in the results files) ====================")
    table = {}
    for study, f in STUDY_FILES.items():
        acc = []
        walk_stops(J(f), acc)
        cnt = Counter(classify(s) for s in acc)
        table[study] = dict(file=f, n=len(acc), counts=dict(cnt))
        log(f"  {study:26s} ({f}): {len(acc)} records: " + "; ".join(f"{k} {v}" for k, v in sorted(cnt.items())))
    tot = Counter()
    for v in table.values():
        tot.update(v["counts"])
    log("  all studies: " + "; ".join(f"{k} {v}" for k, v in sorted(tot.items())))
    log("  note: records are per polish, per path point or per probe as each study stores them; a model can contribute several records")
    return dict(by_study=table, total=dict(tot))


def main():
    log("#" * 100)
    log("paper2/revision/uncertainty.py: DESCRIPTIVE uncertainty from committed results (no training)")
    log("#" * 100)
    res = dict(wilson=part_a(), foldA_bootstrap=part_b(), stop_reasons=part_c())
    (HERE / "uncertainty_results.json").write_text(json.dumps(res, indent=1, default=float))
    (HERE / "uncertainty_output.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()

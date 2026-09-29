"""rule: basis-size rule for co-solvent-determined inputs (see rule_prereg.md). Run: python3 rule.py
Step 0 (feasibility gate, no training) runs first; Steps 1-3 run only if the gate passes."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from gauge import setup
from kan import N_RBF
from support import design_matrix

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
FIG = ROOT / "figs" / "rule"
REL_TOL = 1e-2
MIN_REMOVED = 6
CONST_TOL = 1e-9
RESULTS = {}


def log(*a):
    print(*a, flush=True)


def rank(A):
    s = np.linalg.svd(A, compute_uv=False)
    return int((s / s[0] >= REL_TOL).sum())


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    tr, te, te_anc, D, _ = setup()
    names = D["cols"]
    Ztr = D["Z"][D["train"]]
    log("DISCREPANCIES / choices (files win; see rule_prereg.md):")
    log("  R1 'layer-1' = kan.py layers.0 (input->hidden)")
    log("  R2 co-solvent combination = distinct (eps_co, lneta_co, M_co) triple; eps_co etc. are mole-weighted over non-PC solvents")
    log("  anticipated before running (from support_output.txt): 30 distinct values each -> K = 8 = final.py -> gate stops")

    log("\n==================== STEP 0: feasibility gate (no training) ====================")
    distinct = {n: int(len(np.unique(Ztr[:, i].round(9)))) for i, n in enumerate(names)}
    log("  distinct values per continuous input: " + ", ".join(f"{n} {v}" for n, v in distinct.items()))
    key_cols = [names.index(n) for n in ["eps_co", "lneta_co", "M_co"]]
    keys = [tuple(r) for r in Ztr[:, key_cols].round(9)]
    combos = sorted(set(keys))
    n_co = len(combos)
    idx = np.array([combos.index(k) for k in keys])
    group = [n for i, n in enumerate(names)
             if all(np.ptp(Ztr[idx == c, i]) <= CONST_TOL for c in range(n_co))]
    log(f"  n_co (distinct co-solvent triples) = {n_co}")
    log(f"  inputs constant within every co-solvent combination (the group): {group}")
    others = [n for n in names if n not in group]
    for n in others:
        i = names.index(n)
        within = max(np.ptp(Ztr[idx == c, i]) for c in range(n_co))
        log(f"    not in group: {n:10s} max within-combination range {within:.3g} (scaled units)")
    M = design_matrix(Ztr)
    gcols = np.concatenate([np.arange(names.index(n) * (1 + N_RBF), (names.index(n) + 1) * (1 + N_RBF)) for n in group])
    log(f"  group block of M: {len(gcols)} columns, numerical rank {rank(M[:, gcols])} (rel tol {REL_TOL}); "
        f"full M rank {rank(M)} of {M.shape[1]}")
    g = len(group)
    K = max(1, (n_co - 1) // g - 1) if g else 0
    removed = (1 + N_RBF - (K + 1)) * g
    log(f"  rule: largest K with {g} x (K + 1) <= n_co - 1 = {n_co - 1}  ->  K = {K} RBF centres (final.py: {N_RBF})")
    log(f"  columns per group input: final.py {1 + N_RBF}, modified {K + 1}; columns removed = {removed}")
    gate = removed >= MIN_REMOVED
    log(f"  GATE (>= {MIN_REMOVED} columns removed): {'PASS' if gate else 'STOP'}")
    RESULTS["step0"] = dict(distinct=distinct, n_co=n_co, group=group, group_cols=int(len(gcols)),
                            group_rank=rank(M[:, gcols]), K=K, removed=removed, gate=gate)
    if not gate:
        log("\nSTOP: the rule does not change this model. Steps 1-4 not run; I1-I5 inconclusive.")
        RESULTS["verdicts"] = {f"I{k}": "inconclusive (gate stopped: rule does not change the model)" for k in range(1, 6)}
        (ROOT / "rule_results.json").write_text(json.dumps(RESULTS, indent=1))
        sys.exit(0)
    log("\nGATE PASSED: Steps 1-3 are implemented only after this point (see rule_prereg.md).")
    (ROOT / "rule_results.json").write_text(json.dumps(RESULTS, indent=1))


if __name__ == "__main__":
    main()

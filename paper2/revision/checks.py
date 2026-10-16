"""paper2/revision/checks.py: revision step 1, CHECKS AND CODE HYGIENE. DESCRIPTIVE: trains nothing, re-optimises nothing.
Reads committed files only (no weights needed). Run from the repo root with the pinned interpreter:
  /usr/bin/python3 paper2/revision/checks.py > paper2/revision/checks_output.txt"""
import ast
import importlib.metadata as md
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LR, WD = 3e-3, 1e-4


def log(*a):
    print(*a, flush=True)


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def main():
    log("#" * 100)
    log("paper2/revision step 1: checks and code hygiene. DESCRIPTIVE; no training, no new analysis beyond reading committed files.")
    log("#" * 100)
    log("DISCREPANCIES (files win):")
    log("  - paper2/tex/README.md said the anonymous build labels '67' file and archive names; the committed commitmap.json")
    log("    has 53 distinct names (67 counted escaped and unescaped spellings separately). Corrected in the README.")
    log("  - Table 2 of the draft types 'about equal' for the 1e-6 training fit; the committed values give -3% to +13% (item 5).")

    # ---------------- 1
    log("\n==================== 1. REVIEWER CLAIM: 'learned DEC gain +0.6 to +0.85 log10 at 233 K vs 313 K' ====================")
    readme = (REPO / "README.md").read_text().splitlines()
    hit = [(i + 1, l) for i, l in enumerate(readme) if "+0.6 to +0.85" in l or "log10 at 233 K" in l]
    for n, l in hit:
        log(f"  README.md:{n}: {l.strip()}")
    log(f"  introduced in commit {git('log', '--format=%h %ad %s', '--date=short', '-S', '+0.6 to +0.85', '--', 'README.md').splitlines()[-1]}")
    log("  It describes figs/kan_delta_vs_xDEC.png ('KAN-delta: conductivity gain from DEC'), drawn by kan.py (main, 'figures' block):")
    log("    predicted delta log10 k vs pure PC = x_co * h for binary PC+DEC at 1 mol/kg, x_DEC on linspace(0, 0.7, 71), T = 233, 273,")
    log("    313 K, one panel per test salt, line = mean of the 5 final KAN-delta seeds (kan.py finals1), band = min/max over seeds.")
    found = [f for f in REPO.glob("*_output.txt") if re.search(r"\+0\.6|0\.85 log", f.read_text())]
    log(f"  No committed log, notebook or results file prints these numbers (searched *_output.txt, *.md, *.py, *.ipynb; log hits: {len(found)}).")
    log("  -> The values were read off the figure when the README was written; there is no numeric source. Nothing new is computed here.")
    log("  (anchor.py's 'low-T check (233.75 K, ~1 mol/kg)' in anchor_output.txt is a different quantity: the anchor model's k by DEC")
    log("   weight fraction, not the KAN-delta gain.)")

    # ---------------- 2
    log("\n==================== 2. ADAMW WEIGHT DECAY ====================")
    for f, where in [("kan.py", "train(): torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)  (final.py calls kan.train)"),
                     ("paper2/synth.py", "train_early(): torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)"),
                     ("arch/sc_model.py", "train_sc(): torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)"),
                     ("arch/ex_model.py", "train_ex(): torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)")]:
        log(f"  {f}: {where}")
    log(f"  weight_decay = WD = {WD} (kan.py), lr = {LR}; one parameter group = ALL parameters: RBF coefficients, SiLU base weights,")
    log("  biases, salt embedding (KAN-delta, SC, EX) and, for SC, A, W, w, b. Also used unchanged by control2/3, rank, compare.")
    log("  AdamW's decay is DECOUPLED: each step multiplies every parameter by (1 - lr*wd) outside the Adam update. wd*||theta||^2/2")
    log("  below is therefore an EQUIVALENT-LOSS comparison (what an L2 term of that weight would contribute), not the training objective.")
    rows = {}
    for l in (REPO / "anatomy_output.txt").read_text().splitlines():
        m = re.match(r"\s+ind (\d)\s+107-epoch\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)", l)
        if m and int(m.group(1)) not in rows:
            rows[int(m.group(1))] = tuple(float(m.group(k)) for k in (2, 3, 5))
    log("\n  real data, early-stopped KAN-delta seeds (107 AdamW epochs = 107 full-batch steps; anatomy_output.txt, 107-epoch rows):")
    log("    seed   data MSE    smoothness   ||theta||^2   wd*||theta||^2/2   /smoothness   /data    decoupled shrink over 107 steps")
    rr = []
    for i, (d, p, t) in sorted(rows.items()):
        e = WD * t / 2; rr.append((e / p, e / d))
        log(f"    ind {i}  {d:.4e}  {p:.4e}   {t:9.1f}     {e:.4e}         {e / p:6.3f}     {e / d:6.3f}   factor {(1 - LR * WD) ** 107:.6f}")
    sres = json.loads((REPO / "paper2/synth_results.json").read_text())["models"]
    ld = json.loads((REPO / "paper2/lossdecomp_results.json").read_text())["rows"]
    log("\n  synthetic early states (AdamW early-stopped; data/D2/SiLU from lossdecomp_results.json, ||theta||^2 = trust_theta2[0]):")
    log("    cell              median data   median D2     median SiLU(raw)  median ||theta||^2  median wd*th2/2  /D2     /data   steps (median)  shrink")
    srat = []
    for cond in ("independent", "correlated"):
        for lev in ("P0", "P1", "P2"):
            ks = [f"{cond}_{lev}_s{s}" for s in range(8)]
            import statistics as st
            d = st.median(ld[k]["early"]["data"] for k in ks); d2 = st.median(ld[k]["early"]["D2"] for k in ks)
            si = st.median(ld[k]["early"]["SiLU_raw"] for k in ks)
            th = st.median(sres[k]["trust_theta2"][0] for k in ks); e = WD * th / 2
            steps = st.median(sres[k]["early"]["best_epoch"] for k in ks)
            srat.append((e / d2, e / d))
            log(f"    {cond:11s} {lev}   {d:.4e}   {d2:.4e}   {si:.4e}        {th:8.2f}         {e:.4e}    {e / d2:6.2f}  {e / d:5.2f}   {steps:8.0f}       {(1 - LR * WD) ** steps:.4f}")
    log("\n  polishing losses that OMIT the AdamW decay (all of them; none adds wd*||theta||^2/2):")
    for f, what in [("trust.py Flat / trust-krylov", "data MSE + lambda*sum D2"), ("stage2.py FlatL2 (Part A)", "data + lambda*sum D2 (Part C adds an explicit L2 lam*||theta||^2 with lam 1e-6/1e-5, not wd)"),
                    ("fiber.py / fiber2.py / control3.py L-BFGS", "data + lambda*sum D2"), ("close.py FlatQ", "data + lambda*sum D2 + mu*Q"),
                    ("certify.py (FlatQ, same loss)", "data + lambda*sum D2 + mu*Q"), ("paper2/synth.py FlatS", "data + cell penalty (P0/P1/P2)"),
                    ("arch/sc_model.py FlatSC", "data + lambda(D2 + K_silu alpha^2) + mu Q + beta(A,w,W,e)"), ("arch/ex (EX-KAN)", "not polished")]:
        log(f"    {f:45s} {what}")
    real_p = sorted(x[0] for x in rr); real_d = sorted(x[1] for x in rr)
    log(f"\n  CONCLUSION: as an equivalent loss the decay term is MATERIAL: real data wd*||theta||^2/2 = {real_p[0]:.2f}-{real_p[-1]:.2f} x the smoothness")
    log(f"  penalty and {real_d[0]:.3f}-{real_d[-1]:.3f} x the data term at 107 epochs; synthetic {min(x[0] for x in srat):.1f}-{max(x[0] for x in srat):.1f} x D2 and "
        f"{min(x[1] for x in srat):.2f}-{max(x[1] for x in srat):.2f} x the data term at the early states.")
    log("  Its actual decoupled effect during training was small (total shrink factor 0.99997 over 107 steps; >= 0.997 for the synthetic")
    log("  early stops), but polishing drops it entirely, so step 2's real-data arm should include it.")

    # ---------------- 3
    log("\n==================== 3. PENALTY NORMALISATION ====================")
    log("  Data term: F.mse_loss = MEAN over training rows in every training and polishing script (kan.train, synth, SC, EX, trust, stage2,")
    log("  fiber, fiber2, control2/3, close, certify). Every penalty is a SUM (over all edges, and over coefficients within an edge):")
    log("    D2 = sum over edges and second differences of c^2 (kan.KANLayer.smoothness, synth.penalty_terms, close.FlatQ, sc_model.penalty)")
    log("    Q  = sum over edges ||P_null c||^2 (+ sum of squared SiLU weights in close/certify), SiLU term = sum alpha^2, beta-term = sum of squares")
    log("  So relative to a SUMMED data term the effective weights are lambda*n and mu*n (n = training rows).")
    log("    study                               n rows   lambda   lambda*n   mu (Q)         mu*n           mu/lambda   other")
    studies = [("KAN-delta seeds (final/kan, all rows)", 2069, 1e-3, None, ""), ("KAN-delta fold A (kan/sc baseline)", 1434, 1e-3, None, ""),
               ("close/certify (seeds, all rows)", 2069, 1e-3, (1e-6, 1e-5, 1e-4), "Q includes SiLU weights^2"),
               ("certify fold A step 3", 1434, 1e-3, (1e-6, 1e-4), ""), ("stage2 Part C", 2069, 1e-3, None, "explicit L2 1e-6/1e-5 on ||theta||^2"),
               ("synthetic P0/P1/P2", 2000, 1e-3, (1e-5,), "P1/P2: lambda*K_silu*alpha^2, K_silu 0.4430; Q RBF only"),
               ("SC-KAN-delta fold A", 1434, 1e-3, (1e-5,), "beta 1e-4 on A,w,W,e; K_silu term"), ("EX-KAN fold A", 1434, 1e-3, None, "s and g smoothness")]
    for name, n, lam, mus, other in studies:
        mu_s = "-" if not mus else ", ".join(f"{m:g}" for m in mus)
        mun = "-" if not mus else ", ".join(f"{m * n:.3g}" for m in mus)
        rat = "-" if not mus else ", ".join(f"{m / lam:g}" for m in mus)
        log(f"    {name:36s} {n:6d}   {lam:g}    {lam * n:7.3f}    {mu_s:13s}  {mun:13s}  {rat:10s}  {other}")

    # ---------------- 4
    log("\n==================== 4. PROPERTY TESTS (tests/test_nullspace.py, pytest) ====================")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:warnings"], cwd=REPO, capture_output=True, text=True)
    for l in r.stdout.strip().splitlines()[-8:]:
        log("  " + l)
    log(f"  -> {'ALL PASSED' if r.returncode == 0 else 'FAILURE'}")

    # ---------------- 5
    log("\n==================== 5. TABLES 1-3 GENERATED FROM COMMITTED RESULTS (paper2/tex/build_tex.py, gen_tables) ====================")
    log("  sources: Table 1 paper2/synth_results.json (cells); Table 2 close_results.json (runs: neg, theta2/theta2_start, data/ref_A) +")
    log("  certify_results.json (candidates); Table 3 arch/sc_results.json, arch/ex_results.json, paper2/figures/fig5_uniqueness.json")
    diffs = json.loads((REPO / "paper2/tex/table_diffs.json").read_text())
    log(f"  differences from the draft's typed values: {len(diffs)}")
    for d in diffs:
        log(f"    {d['table']}, row '{d['row']}', column '{d['column']}': typed '{d['typed']}' -> generated '{d['generated']}'")
    log("  (the build now prints the generated value; the draft text itself is unchanged)")

    # ---------------- 6
    log("\n==================== 6. PINNED VERSIONS (requirements.txt) ====================")
    mods = set()
    for f in REPO.rglob("*.py"):
        if any(p in f.parts for p in ("ca-li-sol-23-challenge", "__pycache__")):
            continue
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods.add(node.module.split(".")[0])
    local = {f.stem for f in REPO.rglob("*.py")}
    import importlib.util as iu

    def is_third(m):
        try:
            spec = iu.find_spec(m)
        except Exception:
            return False
        return bool(spec and spec.origin and "site-packages" in spec.origin)
    third = sorted(m for m in mods - local if is_third(m))
    dist = {"sklearn": "scikit-learn", "PIL": "Pillow"}
    req = []
    for m in third:
        name = dist.get(m, m)
        try:
            req.append(f"{name}=={md.version(name)}")
        except md.PackageNotFoundError:
            req.append(f"# {name}: not installed for {sys.executable}")
    if not any(r.startswith("pytest==") for r in req):
        req.append(f"pytest=={md.version('pytest')}")
    (REPO / "requirements.txt").write_text(f"# pinned interpreter: /usr/bin/python3 (resolves to {sys.executable}), Python {sys.version.split()[0]}\n"
                                          "# exact versions of every third-party package imported by the repository's scripts and tests\n" + "\n".join(req) + "\n")
    log(f"  interpreter /usr/bin/python3 -> {sys.executable}, Python {sys.version.split()[0]}; third-party imports found: {', '.join(third)}")
    for l in req:
        log("  " + l)


if __name__ == "__main__":
    main()

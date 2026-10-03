"""arch/sc_run: SC-KAN-delta Steps 0-2 (see arch/sc_prereg.md, 0053d01).
Run from the repo root:
  python3 arch/sc_run.py step0 >> arch/sc_output.txt     (checks, LOCO table, beta rule, gauge check; then commit)
  python3 arch/sc_run.py step12 >> arch/sc_output.txt    (refuses to run unless the Step 0 log is committed)
Weights: gauge_weights/sc_* (git-ignored); every run is cached there with its SHA-256 and resumes from the cache."""
import copy
import hashlib
import io
import itertools
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

ARCH = Path(__file__).resolve().parent
REPO = ARCH.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(ARCH))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

import anchor as AN
from certify import certify_point
from kan import DELTA_IN, Scaler, co_props, delta_head, predict, subset, to_batch, train
from sc_model import (FlatSC, SCKAN, beta_norm, conventions, curve_scale_dirs, n_params, predict_sc, sc_loss, train_sc)
from stage2 import FlatL2, run_exact

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
W_DIR = REPO / "gauge_weights"
FIG = ARCH / "figs"
INDEX = W_DIR / "sc_index.json"
EPOCHS = 107
BETAS = [1e-5, 1e-4, 1e-3, 1e-2]
PILOT, SEEDS, ENS = [100, 101, 102], list(range(10)), list(range(5))
EIGREL = 1e-8


def log(*a):
    print(*a, flush=True)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ------------------------------------------------------------------ cache (SHA-256 indexed, read into memory)
def _index():
    return json.loads(INDEX.read_text()) if INDEX.exists() else {}


def cached(name, make):
    p = W_DIR / name
    idx = _index()
    if p.exists() and name in idx:
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != idx[name]:
            raise SystemExit(f"SHA-256 MISMATCH for cached {name}")
        return torch.load(io.BytesIO(data), weights_only=False), True
    obj = make()
    buf = io.BytesIO(); torch.save(obj, buf); data = buf.getvalue()
    p.write_bytes(data)
    idx = _index(); idx[name] = hashlib.sha256(data).hexdigest()
    INDEX.write_text(json.dumps(idx, indent=1))
    return obj, False


# ------------------------------------------------------------------ data (CSV inputs SHA-logged, read once into memory)
def setup():
    shas = {f: hashlib.sha256((AN.D / f).read_bytes()).hexdigest() for f in ["train.csv", "test.csv", "sample_submission.csv"]}
    tr, te, ss, X, Xte, ok, use, te_anc, has = AN.prepare(verbose=False)
    salts = sorted(set(tr.salt_name) | set(te.salt_name))
    sidx = {s: i for i, s in enumerate(salts)}
    for f, df in [(X, tr), (Xte, te)]:
        f[["eps_co", "lneta_co", "M_co"]] = co_props(df)
    sc = Scaler([X[use], Xte[has]], DELTA_IN)  # identical to final.py
    B = to_batch(sc(X), tr.salt_name.map(sidx).values, x_co=X.x_co, y=tr.delta.fillna(0))
    use = use.values
    co = [[n for n in s.split(";") if n != "PC"] for s in tr.solvents]
    cos = sorted({n for i in np.where(use)[0] for n in co[i]})
    loco = {c: np.array([c in co[i] for i in range(len(tr))]) & use for c in cos}
    hoA = ((tr.combo == "EA+PC") & ok).values
    folds = {"A": dict(train=use & ~hoA, test=hoA & use)}
    for c in cos:
        folds[c] = dict(train=use & ~loco[c], test=loco[c])
    return dict(tr=tr, X=X, sc=sc, B=B, salts=salts, use=use, folds=folds, cos=cos, shas=shas)


def anchor_slopes(tr, mask):
    """d anchor / d(1000/T) from the pure-PC series' piecewise-linear-in-1/T interpolant (see flags)."""
    series = {k: AN.build_series(g) for k, g in tr[tr.combo == "PC"].groupby(["source_doi", "salt_name"])}
    out = np.full(mask.sum(), np.nan)
    for o, i in enumerate(np.where(mask)[0]):
        r = tr.iloc[i]
        sr = series[(r.source_doi, r.salt_name)]
        lv = np.array(sorted(sr)); T, c = r.temperature_K, r.conc_native
        y = {L: AN.interp_c(sr[L], c) for L in lv}
        near = np.where(np.abs(lv - T) < AN.T_TOL)[0]
        if len(near):
            k = near[0]; cand = [(lv[k - 1] if k > 0 else None), (lv[k + 1] if k + 1 < len(lv) else None)]
            a, b = cand
            pts = [(a, b)] if a is not None and b is not None else [(a, lv[k])] if a is not None else [(lv[k], b)] if b is not None else []
        else:
            j = np.searchsorted(lv, T)
            pts = [(lv[j - 1], lv[j])] if 0 < j < len(lv) else []
        for lo, hi in pts:
            if not (np.isnan(y[lo]) or np.isnan(y[hi])):
                out[o] = (y[hi] - y[lo]) / (1000 / hi - 1000 / lo)
    return out


def dbl(b):
    return {k: (v.double() if v.is_floating_point() else v) for k, v in b.items()}


# ------------------------------------------------------------------ model runs (cached)
def sc_early(S, fold, seed, beta, tag="sc"):
    b = subset(S["B"], S["folds"][fold]["train"])
    obj, hit = cached(f"{tag}_{fold.replace(' ', '_')}_b{beta:g}_s{seed}.pt",
                      lambda: dict(early=train_sc(len(S["salts"]), b, beta, EPOCHS, seed).state_dict()))
    net = SCKAN(len(S["salts"])); net.load_state_dict(obj["early"])
    return net.eval()


def base_early(S, fold, seed):
    b = subset(S["B"], S["folds"][fold]["train"])
    obj, hit = cached(f"scbase_{fold.replace(' ', '_')}_s{seed}.pt",
                      lambda: dict(early=train([6, 1], len(DELTA_IN), len(S["salts"]), delta_head, b, epochs=EPOCHS, seed=seed)[0].state_dict()))
    from kan import KAN
    net = KAN(len(DELTA_IN), [6, 1], len(S["salts"])); net.load_state_dict(obj["early"])
    return net.eval()


def sc_polish(S, fold, seed, beta, net, tag="sc"):
    b = subset(S["B"], S["folds"][fold]["train"])

    def make():
        fl = FlatSC(net, b, beta)
        x, info, hist = run_exact(f"{tag} {fold} s{seed}", fl)
        cert, _, ev = certify_point(fl, x)
        return dict(x=torch.tensor(x), x0=torch.tensor(fl.x0), info={k: v for k, v in info.items() if k != "eig_checks"},
                    theta2=[float((fl.x0 ** 2).sum()), float((x ** 2).sum())], cert=cert)
    obj, hit = cached(f"{tag}_{fold.replace(' ', '_')}_b{beta:g}_s{seed}_pol.pt", make)
    fl = FlatSC(net, b, beta)
    return fl, obj["x"].numpy(), obj


def base_polish(S, fold, seed, net):
    m = S["folds"][fold]["train"]
    b = dbl(subset(S["B"], m))
    used = np.unique(S["B"]["s"][torch.tensor(m)].numpy()).tolist()

    def make():
        fl = FlatL2(copy.deepcopy(net), b, used, 0.0)
        x, info, hist = run_exact(f"baseline {fold} s{seed}", fl)
        cert, _, ev = certify_point(fl, x)
        return dict(x=torch.tensor(x), x0=torch.tensor(fl.x0), info={k: v for k, v in info.items() if k != "eig_checks"},
                    theta2=[float((fl.x0 ** 2).sum()), float((x ** 2).sum())], cert=cert)
    obj, hit = cached(f"scbase_{fold.replace(' ', '_')}_s{seed}_pol.pt", make)
    fl = FlatL2(copy.deepcopy(net), b, used, 0.0)
    return fl, obj["x"].numpy(), obj


def base_pred(net, b):
    with torch.no_grad():
        dt = next(net.parameters()).dtype
        bb = {k: (v.to(dt) if v.is_floating_point() else v) for k, v in b.items()}
        return delta_head(net, bb).double().numpy()


# ------------------------------------------------------------------ STEP 0
def step0():
    log("#" * 110)
    log(f"arch/sc_run STEP 0 {time.strftime('%Y-%m-%d %H:%M:%S')} (prereg arch/sc_prereg.md, commit 0053d01). Nothing in Step 0 is a result.")
    log("#" * 110)
    log("DISCREPANCIES / choices (files win; flags 1-11 are in the prereg):")
    log("  D1 polishing uses stage2.run_exact unchanged: trust-exact, float64, exact Hessian, cap 500 iterations AND stage2's 2 h wall-clock cap")
    log("  D2 certification = certify.py's certify_point unchanged (its null set is spectral: |eig| < 1e-8 x max); the frozen gauge set from")
    log("     Step 0 is used as a diagnostic: spectral-null eigenvalues outside the frozen set are reported next to every certificate")
    log("  D3 the LOCO EA fold has exactly fold A's training and test rows; the same 10 models (same rows, seeds, protocol) serve both")
    log("  D4 anchor temperature slope: anchor.anchor_value is constant in T within T_TOL = 0.3 K of a series level, so a finite difference of the")
    log("     anchor gives 0 on most rows; d anchor/d(1000/T) is taken from the pure-PC series' piecewise-linear-in-1/T interpolant between")
    log("     adjacent T levels (central at a level, else the bracketing pair); rows without two valid levels are excluded from S8/S9 (counted)")
    log("  D5 d log k / d(ln eta_mix) is the partial derivative in the mix_lneta feature (other features, incl. lneta_co, held fixed); anchor-free")
    log("  D6 SC initialisation: alpha ~ U(-1,1), c ~ 0.1 N(0,1) (as kan.KANLayer), nn.Linear defaults for A/b, W, head; nn.Embedding default for e")
    log("  D7 ||e||^2 includes all 12 salt rows (unused salts are pulled to 0 by beta); RMSEs unclipped, on delta (= log k, the anchor is fixed)")
    log("  D8 d_AB uses std of the fold A training-row delta targets (as certify.py)")
    log("  D9 the beta pilot evaluates the grid in ascending order and stops at the first passing value (= the smallest passing beta)")
    S = setup()
    log("\ninputs (read once by anchor.prepare, held in memory): " + ", ".join(f"{k} sha256 {v[:12]}" for k, v in S["shas"].items()))
    R = dict(prereg="0053d01", shas=S["shas"])

    # ---------------- a) unit tests
    log("\n==================== STEP 0a: unit tests ====================")
    bA = dbl(subset(S["B"], S["folds"]["A"]["train"]))
    torch.manual_seed(0)
    net = SCKAN(len(S["salts"])).double()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.3 * torch.randn_like(p))
    fl = FlatSC(net, bA, 1e-3)
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(5):
        v = rng.standard_normal(fl.n); v /= np.linalg.norm(v)
        g = fl.fun(fl.x0)[1] @ v
        fd = (fl.fun(fl.x0 + 1e-6 * v)[0] - fl.fun(fl.x0 - 1e-6 * v)[0]) / 2e-6
        errs.append(abs(g - fd) / max(abs(g), 1e-300))
    nP = n_params(net)
    net.set_center(bA["x"])
    ball = dbl(subset(S["B"], S["use"]))
    p0 = predict_sc(net, ball)
    pc = predict_sc(conventions(net, bA["x"]), ball)
    nflip_units = int((net.head.weight[0] < 0).sum())
    conv_err = float(np.abs(p0 - pc).max())
    ok_a = max(errs) < 1e-6 and nP == 169 and conv_err < 1e-12
    log(f"  finite-difference gradient check (full loss incl. centring, 5 random directions, h = 1e-6): max rel error {max(errs):.2e} -> {'ok' if max(errs) < 1e-6 else 'FAIL'}")
    log(f"  parameter count {nP} (expected 169) -> {'ok' if nP == 169 else 'FAIL'}")
    log(f"  conventions (curve sign, unit variance, hidden-unit sign; {nflip_units} units flipped): max |delta pred| over {ball['y'].shape[0]} anchored rows "
        f"{conv_err:.2e} -> {'ok' if conv_err < 1e-12 else 'FAIL'}")
    R["unit_tests"] = dict(fd_max_rel=max(errs), n_params=nP, conv_err=conv_err, ok=ok_a)
    if not ok_a:
        log("STOP: a unit test failed"); R["stopped"] = "unit tests"
        (ARCH / "sc_step0.json").write_text(json.dumps(R, indent=1, default=float)); return

    # ---------------- b) LOCO split
    log("\n==================== STEP 0b: LOCO split (fold c test = anchored rows containing c; train = anchored rows without c) ====================")
    tr = S["tr"]
    rows = []
    for f, m in S["folds"].items():
        ntr, nte = int(m["train"].sum()), int(m["test"].sum())
        nsalt = len(set(tr.salt_name[m["train"]])); nsalt_te = len(set(tr.salt_name[m["test"]]))
        unseen = sorted(set(tr.salt_name[m["test"]]) - set(tr.salt_name[m["train"]]))
        rows.append(dict(fold=f, train=ntr, test=nte, salts_train=nsalt, salts_test=nsalt_te, unseen_test_salts=unseen, ge20=nte >= 20))
        log(f"  {f:20s} train {ntr:5d}  test {nte:5d}  salts train/test {nsalt:2d}/{nsalt_te:2d}  {'>=20' if nte >= 20 else '<20 '}  test salts unseen in train: {unseen or '-'}")
    same = bool((S["folds"]["A"]["train"] == S["folds"]["EA"]["train"]).all() and (S["folds"]["A"]["test"] == S["folds"]["EA"]["test"]).all())
    log(f"  fold A == LOCO EA fold (train and test masks identical): {same}")
    log(f"  folds with >= 20 test rows: {sum(r['ge20'] for r in rows if r['fold'] != 'A')} of {len(S['cos'])}")
    R["loco"] = rows; R["foldA_equals_EA"] = same

    # ---------------- c) beta rule
    log("\n==================== STEP 0c: beta rule (pilot on fold A TRAINING rows, seeds 100-102; calibration only) ====================")
    calib, frozen = [], None
    for beta in BETAS:
        ratios, pts = [], []
        for s in PILOT:
            net = sc_early(S, "A", s, beta, tag="scpilot")
            fl, x, obj = sc_polish(S, "A", s, beta, net, tag="scpilot")
            e = float(beta_norm(fl.params(torch.tensor(fl.x0)))); p_ = float(beta_norm(fl.params(torch.tensor(x))))
            ratios.append(p_ / e); pts.append((fl, x, obj))
            c = obj["cert"]
            log(f"  beta {beta:g} seed {s}: beta-norm {e:.3f} -> {p_:.3f} (x{p_ / e:.3f}); ||theta||^2 x{obj['theta2'][1] / obj['theta2'][0]:.3f}; "
                f"{obj['info']['stop']} after {obj['info']['iters']} iters; certified {c['certified']} (grad {c['grad']:.1e}, min eig {c['mineig']:.1e}, "
                f"null {c['n_null']}, Newton {c['newton']:.1e})")
        med = float(np.median(ratios))
        calib.append(dict(beta=beta, ratios=ratios, median=med, passed=med <= 3))
        log(f"  beta {beta:g}: median (||A||^2+||w||^2+||W||^2+||e||^2) polished/early {med:.3f} -> {'PASS' if med <= 3 else 'fail'}")
        if med <= 3:
            frozen = beta; frozen_pts = pts
            break
    R["beta_calibration"] = calib
    if frozen is None:
        log("STOP: no beta in {1e-5, 1e-4, 1e-3, 1e-2} passed the rule; Step 1 not run (as pre-registered)")
        R["stopped"] = "beta rule"; R["beta"] = None
        (ARCH / "sc_step0.json").write_text(json.dumps(R, indent=1, default=float)); return
    log(f"  FROZEN: beta = {frozen:g}")
    R["beta"] = frozen

    # ---------------- d) gauge check: exact continuous gauge directions of the full loss
    log("\n==================== STEP 0d: gauge check (certification null space, frozen before Step 1) ====================")
    log("  candidates: curve scale (alpha_i, c_i) x s with A_.i / s (8, exact for the data term); constant shift of a curve (removed by centring");
    log("  only to the extent sum_k B_k is constant); salt-embedding GL(2)/translation (broken by tanh); unused-salt embeddings (data-free)")
    gauge = []
    for (fl, x, obj), s in zip(frozen_pts, PILOT):
        for state, xx in [("early", fl.x0), ("polished", x)]:
            H = fl.hessian(xx); ev = np.linalg.eigvalsh(H)
            J = fl.jac_pred(xx); sv = np.linalg.svd(J, compute_uv=False)
            nd = int((sv < 1e-8 * sv[0]).sum())
            dirs = curve_scale_dirs(fl, xx)
            jv = max(np.linalg.norm(J @ v) / (sv[0] * np.linalg.norm(v)) for v in dirs)
            curv = min(float(v @ H @ v) / (ev[-1] * float(v @ v)) for v in dirs)
            nnull = int((np.abs(ev) < EIGREL * ev[-1]).sum())
            gauge.append(dict(seed=s, state=state, n_hess_null=nnull, min_eig=float(ev[0]), max_eig=float(ev[-1]), n_data_null=nd,
                              curve_scale_data_change=float(jv), curve_scale_min_curv=curv))
            log(f"  pilot s{s} {state:8s}: Hessian eig [{ev[0]:.2e}, {ev[-1]:.2e}], |eig| < 1e-8 x max: {nnull}; data-Jacobian null dims (sigma < 1e-8 x max): {nd}; "
                f"curve-scale directions: max rel data change {jv:.1e}, min curvature/max eig {curv:.1e}")
    pol = [g for g in gauge if g["state"] == "polished"]
    empty = all(g["n_hess_null"] == 0 for g in pol)
    R["gauge"] = dict(points=gauge, frozen_set=[] if empty else "spectral null set at each point (certify_point)",
                      note="data-term symmetries (curve scale, unused salts) are broken by the lambda/mu/beta terms")
    log(f"  FROZEN certification gauge set: {'EMPTY (no exact continuous gauge direction of the full loss beyond sign flips)' if empty else 'NON-EMPTY: spectral null set used, see points'}")
    log("  baseline KAN-delta: as certify.py (E4: its constant-shift/rescaling directions are least-squares, not exact); frozen set empty")
    (ARCH / "sc_step0.json").write_text(json.dumps(R, indent=1, default=float))
    log("\nSTEP 0 done; commit arch/sc_output.txt and arch/sc_step0.json before Step 1.")


# ------------------------------------------------------------------ STEP 1-2
def committed():
    st = subprocess.run(["git", "status", "--porcelain", "arch/sc_step0.json", "arch/sc_output.txt"], cwd=REPO, capture_output=True, text=True).stdout
    lg = subprocess.run(["git", "log", "-1", "--format=%h", "--", "arch/sc_step0.json"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return st.strip() == "" and lg != "", lg


def step12():
    ok, c0 = committed()
    if not ok:
        raise SystemExit("REFUSED: the Step 0 log (arch/sc_step0.json, arch/sc_output.txt) is not committed")
    R0 = json.loads((ARCH / "sc_step0.json").read_text())
    if R0.get("stopped"):
        raise SystemExit(f"REFUSED: Step 0 stopped ({R0['stopped']})")
    beta = R0["beta"]
    log("\n" + "#" * 110)
    log(f"arch/sc_run STEPS 1-2 {time.strftime('%Y-%m-%d %H:%M:%S')}; Step 0 committed in {c0}; frozen beta = {beta:g}; frozen gauge set: "
        f"{R0['gauge']['frozen_set'] or 'empty'}")
    log("#" * 110)
    S = setup()
    FIG.mkdir(parents=True, exist_ok=True)
    folds = S["folds"]
    R = dict(beta=beta, step0_commit=c0)

    # ---------------- Step 1: training
    log("\n==================== STEP 1: training (10 seeds per fold, both models; AdamW as final.py, 107 epochs) ====================")
    nets = {"sc": {}, "base": {}}
    for f in folds:
        src = "A" if f == "EA" else f  # D3: identical rows
        t0 = time.time()
        nets["sc"][f] = [sc_early(S, src, s, beta) for s in SEEDS]
        nets["base"][f] = [base_early(S, src, s) for s in SEEDS]
        log(f"  fold {f:20s}: 10 SC + 10 baseline early models ({time.time() - t0:.1f} s{', = fold A models (D3)' if f == 'EA' else ''})")

    log("\n  fold A polishing (trust-exact, float64, cap 500) and certification (certify.py four-part criterion):")
    pol = {"sc": [], "base": []}
    for s in SEEDS:
        for kind in ["sc", "base"]:
            net = nets[kind]["A"][s]
            fl, x, obj = (sc_polish(S, "A", s, beta, net) if kind == "sc" else base_polish(S, "A", s, net))
            pol[kind].append((fl, x, obj))
            c = obj["cert"]; i = obj["info"]
            extra = f" beta-norm x{float(beta_norm(fl.params(torch.tensor(x)))) / float(beta_norm(fl.params(torch.tensor(fl.x0)))):.3f};" if kind == "sc" else ""
            log(f"  {kind:4s} s{s}: {i['stop']} after {i['iters']} iters ({i['wall_s']} s);{extra} ||theta||^2 x{obj['theta2'][1] / obj['theta2'][0]:.3f}; "
                f"certified {str(c['certified']):5s} grad {c['grad']:.1e} min eig {c['mineig']:.1e} max {c['maxeig']:.1e} spectral null {c['n_null']} "
                f"(outside frozen set: {c['n_null']}) Newton {c['newton']:.1e} g_null {c['gnull']:.1e} | " +
                " ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in c["crit"].items()))

    # ---------------- Step 2: measurements
    log("\n==================== STEP 2: measurements ====================")
    B = S["B"]
    acc = {}
    for kind in ["sc", "base"]:
        per = {}
        for f, m in folds.items():
            bt = subset(B, m["test"])
            p = np.mean([(predict_sc(n, bt) if kind == "sc" else base_pred(n, bt)) for n in nets[kind][f][:5]], 0)
            per[f] = dict(rmse=rmse(p, bt["y"].numpy()), n=int(m["test"].sum()))
        lo = {f: v for f, v in per.items() if f != "A"}
        acc[kind] = dict(per=per, loco_unweighted=float(np.mean([v["rmse"] for v in lo.values()])),
                         loco_rowweighted=float(np.sqrt(sum(v["rmse"] ** 2 * v["n"] for v in lo.values()) / sum(v["n"] for v in lo.values()))),
                         loco_ge20=float(np.mean([v["rmse"] for v in lo.values() if v["n"] >= 20])), foldA=per["A"]["rmse"])
    log("  accuracy (5-seed ensemble, seeds 0-4, early state; RMSE on delta = log k):")
    log(f"    {'fold':20s} {'n':>5s} {'SC':>8s} {'base':>8s}")
    for f in folds:
        log(f"    {f:20s} {acc['sc']['per'][f]['n']:5d} {acc['sc']['per'][f]['rmse']:8.4f} {acc['base']['per'][f]['rmse']:8.4f}")
    for k, nm in [("loco_unweighted", "LOCO mean, unweighted (S1)"), ("loco_rowweighted", "LOCO, row-weighted (pooled RMSE)"),
                  ("loco_ge20", "LOCO mean, folds >= 20 test rows"), ("foldA", "fold A")]:
        log(f"    {nm:38s} SC {acc['sc'][k]:.4f}  base {acc['base'][k]:.4f}  ratio {acc['sc'][k] / acc['base'][k]:.3f}")
    log("    note: fold A and the LOCO EA fold are the same rows (D3), so S2 and the EA term of S1 are not independent")

    # polished accuracy (fold A)
    bA_te = dbl(subset(B, folds["A"]["test"]))
    bA_tr = subset(B, folds["A"]["train"])
    pol_rmse = {}
    for kind in ["sc", "base"]:
        ps = []
        for fl, x, obj in pol[kind][:5]:
            n = fl.to_net(x) if kind == "sc" else fl.to_kan(x, len(DELTA_IN), len(S["salts"]))
            ps.append(predict_sc(n, bA_te) if kind == "sc" else base_pred(n, bA_te))
        pol_rmse[kind] = rmse(np.mean(ps, 0), bA_te["y"].numpy())
        log(f"  fold A polished ensemble RMSE {kind}: {pol_rmse[kind]:.4f} (early {acc[kind]['foldA']:.4f}; ratio {pol_rmse[kind] / acc[kind]['foldA']:.3f})")

    # seed agreement (fold A, early)
    sd_y = float(bA_tr["y"].std())
    dab = {}
    for kind in ["sc", "base"]:
        pr = {s: (predict_sc(n, bA_tr) if kind == "sc" else base_pred(n, bA_tr)) for s, n in enumerate(nets[kind]["A"])}
        dab[kind] = [rmse(pr[a], pr[b]) / sd_y for a, b in itertools.combinations(SEEDS, 2)]
        log(f"  seed agreement {kind}: median d_AB {np.median(dab[kind]):.4f} over 45 pairs (IQR {np.percentile(dab[kind], 25):.4f}-{np.percentile(dab[kind], 75):.4f})")

    # curve agreement (SC, fold A, early, after conventions)
    xtr = bA_tr["x"].double()
    lo, hi = xtr.min(0).values, xtr.max(0).values
    G = torch.stack([torch.linspace(float(lo[i]), float(hi[i]), 200, dtype=torch.float64) for i in range(8)], 1)
    conv = [conventions(n, xtr) for n in nets["sc"]["A"]]
    zc = [(c.curves(G) - c.center).detach().numpy() for c in conv]
    curve = []
    for i in range(8):
        cs = [float(np.corrcoef(zc[a][:, i], zc[b][:, i])[0, 1]) for a, b in itertools.combinations(SEEDS, 2)]
        curve.append(float(np.median(cs)))
    log("  curve agreement (median pairwise corr of z_i on a 200-point grid, after conventions): " +
        ", ".join(f"{DELTA_IN[i]} {curve[i]:.3f}" for i in range(8)) + f" -> {sum(c >= 0.90 for c in curve)} of 8 >= 0.90")

    # hidden-unit match (descriptive; conventions + Hungarian on contribution correlation)
    def contrib(c):
        with torch.no_grad():
            return (c.head.weight[0] * torch.tanh(c.hidden(xtr, bA_tr["s"]))).numpy()
    Cs = [contrib(c) for c in conv]
    um = []
    for a, b in itertools.combinations(SEEDS, 2):
        A_, B_ = Cs[a] - Cs[a].mean(0), Cs[b] - Cs[b].mean(0)
        C = (A_.T @ B_) / np.outer(np.linalg.norm(A_, axis=0), np.linalg.norm(B_, axis=0)).clip(1e-12)
        r, k = linear_sum_assignment(-C)
        um.append(float(np.median(C[r, k])))
    log(f"  (descriptive) hidden-unit match: median over pairs of the median matched contribution correlation {np.median(um):.3f}")

    # physics (SC, fold A, early, per seed)
    sl = anchor_slopes(S["tr"], folds["A"]["train"])
    valid = ~np.isnan(sl)
    invT = S["X"].invT.values[folds["A"]["train"]]
    xco = bA_tr["x_co"].double().numpy()
    sc_ = S["sc"]
    jT, jE = DELTA_IN.index("invT"), DELTA_IN.index("mix_lneta")
    phys = []
    for s, n in enumerate(nets["sc"]["A"]):
        n = copy.deepcopy(n).double()
        def dh(j, eps=1e-4):
            xp, xm = xtr.clone(), xtr.clone(); xp[:, j] += eps; xm[:, j] -= eps
            with torch.no_grad():
                return ((n(xp, bA_tr["s"]) - n(xm, bA_tr["s"])) / (2 * eps)).numpy() * 2 / (sc_.hi[j] - sc_.lo[j])
        dT = sl + xco * dh(jT)
        dE = xco * dh(jE)
        v = dT[valid]
        rho = float(spearmanr(invT[valid], np.abs(v)).correlation)
        phys.append(dict(seed=s, neg_T=float((v < 0).mean()), rho=rho, neg_eta=float((dE < 0).mean()), dT=dT, dE=dE))
        log(f"  physics s{s}: d log k/d(1000/T) < 0 on {phys[-1]['neg_T']:.3f} of {valid.sum()} rows; Spearman(1000/T, |slope|) {rho:+.3f}; "
            f"d log k/d(ln eta_mix) < 0 on {phys[-1]['neg_eta']:.3f} of {len(dE)} rows")
    log(f"  rows excluded from S8/S9 (no two valid anchor T levels, D4): {(~valid).sum()} of {len(sl)}; model-part-only d delta/d(1000/T) < 0 "
        f"(median over seeds) {np.median([float(((d - sl)[valid] < 0).mean()) for d in [p['dT'] for p in phys]]):.3f}")

    # stability (fold A)
    st = {}
    for kind in ["sc", "base"]:
        th = [o["theta2"][1] / o["theta2"][0] for _, _, o in pol[kind]]
        cert = sum(o["cert"]["certified"] for _, _, o in pol[kind])
        st[kind] = dict(theta2=th, certified=cert)
        if kind == "sc":
            st[kind]["beta_norm"] = [float(beta_norm(fl.params(torch.tensor(x)))) / float(beta_norm(fl.params(torch.tensor(fl.x0)))) for fl, x, _ in pol[kind]]
        log(f"  stability {kind}: certified {cert}/10; ||theta||^2 polished/early median {np.median(th):.3f} (range {min(th):.3f}-{max(th):.3f})"
            + (f"; beta-norm polished/early median {np.median(st[kind]['beta_norm']):.3f} (range {min(st[kind]['beta_norm']):.3f}-{max(st[kind]['beta_norm']):.3f})" if kind == "sc" else ""))

    # ---------------- verdicts
    V = {}
    V["S1"] = (acc["sc"]["loco_unweighted"] <= 1.05 * acc["base"]["loco_unweighted"],
               f"SC {acc['sc']['loco_unweighted']:.4f} vs 1.05 x base {1.05 * acc['base']['loco_unweighted']:.4f} (row-weighted {acc['sc']['loco_rowweighted']:.4f}/{acc['base']['loco_rowweighted']:.4f}; >=20 rows {acc['sc']['loco_ge20']:.4f}/{acc['base']['loco_ge20']:.4f})")
    V["S2"] = (acc["sc"]["foldA"] <= 0.160, f"SC fold A {acc['sc']['foldA']:.4f} (base {acc['base']['foldA']:.4f}); not independent of S1's EA fold")
    V["S3"] = (np.median(dab["sc"]) <= 0.5 * np.median(dab["base"]), f"SC {np.median(dab['sc']):.4f} vs 0.5 x base {0.5 * np.median(dab['base']):.4f}")
    V["S4"] = (sum(c >= 0.90 for c in curve) >= 6, f"{sum(c >= 0.90 for c in curve)} of 8 >= 0.90 (min {min(curve):.3f})")
    V["S5"] = (st["sc"]["certified"] >= 8, f"SC {st['sc']['certified']}/10 certified (base {st['base']['certified']}/10)")
    V["S6"] = (np.median(st["sc"]["beta_norm"]) <= 3, f"median {np.median(st['sc']['beta_norm']):.3f} (calibrated, flag 8)")
    V["S7"] = (pol_rmse["sc"] / acc["sc"]["foldA"] <= 1.5, f"SC {pol_rmse['sc'] / acc['sc']['foldA']:.3f} (base {pol_rmse['base'] / acc['base']['foldA']:.3f})")
    nT = sum(p["neg_T"] >= 0.95 for p in phys); nR = sum(p["rho"] > 0 for p in phys); nE = sum(p["neg_eta"] >= 0.90 for p in phys)
    V["S8"] = (nT >= 8, f"{nT}/10 seeds with >= 95% rows negative (median share {np.median([p['neg_T'] for p in phys]):.3f})")
    V["S9"] = (nR >= 8, f"{nR}/10 seeds with Spearman > 0 (median {np.median([p['rho'] for p in phys]):+.3f})")
    V["S10"] = (nE >= 8, f"{nE}/10 seeds with >= 90% rows negative (median share {np.median([p['neg_eta'] for p in phys]):.3f})")
    log("\n==================== VERDICTS (pre-registered S1-S10) ====================")
    for k, (h, txt) in V.items():
        log(f"  {k:4s} {'HELD' if h else 'FAILED':6s} {txt}")
    log("  note: fold A and the LOCO EA fold are the same rows, so S2 and the EA fold of S1 are not independent")

    # ---------------- figures
    names = DELTA_IN
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    for i, ax in enumerate(axs.ravel()):
        xr = sc_.inv(i, G[:, i].numpy())
        for s, z in enumerate(zc):
            ax.plot(xr, z[:, i], lw=1, alpha=0.7, label=f"s{s}" if i == 0 else None)
        ax.set_title(f"{names[i]} (median corr {curve[i]:.3f})"); ax.set_xlabel(names[i]); ax.set_ylabel("z_i (unit var, sign conv.)")
    axs[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle("SC-KAN-delta shared curves, fold A, early state, 10 seeds after conventions")
    fig.tight_layout(); fig.savefig(FIG / "sc_curves.png", dpi=110); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for p in phys:
        axs[0].scatter(invT[valid], p["dT"][valid], s=3, alpha=0.25)
        axs[1].scatter(invT, (p["dT"] - sl), s=3, alpha=0.25)
    axs[0].axhline(0, color="k", lw=0.8); axs[1].axhline(0, color="k", lw=0.8)
    axs[0].set_title("d log k / d(1000/T): anchor slope + model part (10 seeds)"); axs[1].set_title("model part only: x_co dh/d(1000/T)")
    for ax in axs:
        ax.set_xlabel("1000/T (1/K)")
    axs[0].set_ylabel("marginal effect")
    fig.tight_layout(); fig.savefig(FIG / "sc_temperature.png", dpi=110); plt.close(fig)

    R.update(accuracy=acc, polished_rmse=pol_rmse, dab={k: dict(median=float(np.median(v)), all=v) for k, v in dab.items()}, curve=curve, unit_match=um,
             physics=[{k: v for k, v in p.items() if k not in ("dT", "dE")} for p in phys], n_excluded_S8=int((~valid).sum()), stability=st,
             polish={k: [dict(info=o["info"], cert=o["cert"]) for _, _, o in pol[k]] for k in pol},
             verdicts={k: dict(held=bool(h), text=t) for k, (h, t) in V.items()})
    (ARCH / "sc_results.json").write_text(json.dumps(R, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else bool(o) if isinstance(o, np.bool_) else float(o)))


if __name__ == "__main__":
    {"step0": step0, "step12": step12}[sys.argv[1]]()

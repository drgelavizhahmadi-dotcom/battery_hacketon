"""paper2/revision/dose.py: revision step 4, dose-response over input correlation. See dose_prereg.md (1a18569).
Run from the repo root with the pinned interpreter:
  /usr/bin/python3 paper2/revision/dose.py --gate >> paper2/revision/dose_output.txt   (runtime checks + replication gate, rho = 0.95)
  /usr/bin/python3 paper2/revision/dose.py --grid >> paper2/revision/dose_output.txt   (the full grid, only after the gate passed)
Model, data, training, polish and certification as committed in paper2/synth.py (34cc321); probes, short path and point protocol
from paper2/revision/infimum.py. Every polish runs in its own child process (hard-killed at the cap + 5 min) and is cached in
gauge_weights/dose_* (SHA-indexed), so an interruption resumes. Inputs SHA-verified and held in memory. No weights in git."""
import argparse
import io
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "paper2", REPO / "arch", HERE):
    sys.path.insert(0, str(p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

import anatomy_synth as AN
import infimum as I
import synth as SY
from certify import certify_point

torch.set_num_threads(4)                       # as synth.py (34cc321)

RHOS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
GATE_RHO, CTRL_RHO = 0.95, 0.99
LEVS = ["P0", "P1"]
SEEDS = list(range(8))
PATH_G = [1e-3, 1e-5, 1e-7]
C_IDX = 1                                      # E2: the correlated cell's seeds (1000+1, ..., 5000+1) at every rho
HARD = 300
W = I.W
LOGDIR = W / "dose_logs"
SYN_ARCH = "gauge_weights_2026-10-03h.tar.gz"
FIG = HERE / "figs"
RES = HERE / "dose_results.json"


def log(*a):
    print(*a, flush=True)


def rtag(rho):
    return f"{rho:.2f}"


def key_of(lev, rho, s):
    return f"{lev}_r{rtag(rho)}_s{s}"


def parse(key):
    lev, r, s = key.split("_")
    return lev, float(r[1:]), int(s[1:])


# ------------------------------------------------------------------ data (synth.py's generator and label rule, paired seeds)
_DATA = {}


def teacher():
    if "T" not in _DATA:
        res = json.loads((REPO / "paper2/synth_results.json").read_text())
        _DATA["T"] = SY.build_teacher(res["teacher_scale"]); _DATA["res"] = res
    return _DATA["T"]


def data(rho):
    if rho not in _DATA:
        T = teacher()
        X, Xv, Xt = SY.make_inputs(rho, 1000 + C_IDX), SY.make_inputs(rho, 2000 + C_IDX), SY.make_inputs(rho, 3000 + C_IDX)
        Xo = SY.make_inputs(rho, 4000 + C_IDX, independent=True)
        y0 = SY.predict(T, X); sig = 0.05 * y0.std()
        rng = np.random.default_rng(5000 + C_IDX)
        y = y0 + sig * rng.standard_normal(SY.N); yv = SY.predict(T, Xv) + sig * rng.standard_normal(SY.N)
        _DATA[rho] = dict(X=X, Xv=Xv, Xt=Xt, Xo=Xo, y0=y0, y=y, yv=yv, sigma=float(sig))
    return _DATA[rho]


def mu_of(lev):
    teacher()
    return float(_DATA["res"]["mu"]) if lev == "P2" else 0.0


# ------------------------------------------------------------------ models
def early(key):
    """synth.train_early, cached (state dict float32 + info + trajectory)"""
    lev, rho, s = parse(key)
    d = data(rho)

    def make():
        net0, einfo, traj = SY.train_early(s, d["X"], d["y"], d["Xv"], d["yv"], lev, mu_of(lev))
        return dict(early=net0.state_dict(), info=einfo, traj=traj)
    obj, _ = I.cached(f"dose_early_{key}.pt", make)
    net = SY.SynthKAN(); net.load_state_dict(obj["early"]); net.eval()
    return net, obj["info"]


def iflat(key, kind=None, coef=None, x_net=None):
    """infimum.Flat with the committed cell loss (and the gamma*||theta||^2 term for kind 'L2'); net = early state by default"""
    lev, rho, s = parse(key)
    d = data(rho)
    net = (x_net if x_net is not None else early(key)[0]).double()
    full = dict(net.named_parameters())
    f = I.Flat(net, full, list(full), I.syn_parts(net, d["X"], d["y"], lev, mu_of(lev)), None, {})
    if kind is not None:
        f.reg = I.make_reg(kind, coef, f.sl)
    Xd = torch.tensor(d["X"], dtype=torch.float64)
    return f, net, (lambda prm: Xd)


# ------------------------------------------------------------------ child processes
def child(spec):
    if spec["mode"] == "trust":
        # the committed polish, unchanged: synth.run_trust on synth.FlatS from the early state; checkpoint at every Hessian call
        lev, rho, s = parse(spec["key"])
        d = data(rho)
        net0, _ = early(spec["key"])
        fl = SY.FlatS(net0, d["X"], d["y"], lev, mu_of(lev))
        ck, th = Path(spec["ck"]), []
        orig = fl.hessian

        def hess(x):
            th.append(float((np.asarray(x) ** 2).sum()))
            np.savez(ck.with_name(ck.stem + ".tmp.npz"), x=np.asarray(x), theta2=np.array(th)); os.replace(ck.with_name(ck.stem + ".tmp.npz"), ck)
            return orig(x)
        fl.hessian = hess
        I.cached(spec["name"], lambda: (lambda r: dict(x=torch.tensor(r[0]), info=r[1], hist=r[2]))(SY.run_trust(fl)))
    else:
        f, _, _ = iflat(spec["key"], spec["kind"], spec["coef"])
        x0 = np.load(spec["start"])
        I.cached(spec["name"], lambda: dict(zip(("x", "info"), (lambda r: (torch.tensor(r[0]), r[1]))(
            I.polish(f, x0, Path(spec["ck"]), spec["gtol"], spec["elapsed0"])))))


def run_child(spec, wlog):
    """cached result, else the child with a heartbeat and a hard kill at the cap + 5 min (cumulative, monotonic)"""
    name = spec["name"]
    if not I.has_cached(name):
        ck = Path(spec["ck"]); hb = W / (ck.stem + ".hb.json")
        if spec["mode"] == "trust":
            # synth.run_trust cannot resume: an interrupted committed polish restarts from the early state with its full caps,
            # so stale heartbeat / checkpoint files of an earlier attempt are discarded (crash record 1)
            e0 = 0.0
            for fp in (hb, ck):
                if fp.exists():
                    fp.unlink()
        else:
            e0 = float(json.loads(hb.read_text())["elapsed"]) if hb.exists() else 0.0
            if ck.exists():
                e0 = max(e0, float(np.load(ck)["elapsed"]))
        spec = dict(spec, elapsed0=e0)
        LOGDIR.mkdir(exist_ok=True)
        with open(LOGDIR / f"{Path(name).stem}.log", "a") as lf:
            pr = subprocess.Popen([sys.executable, str(Path(__file__)), "--child", json.dumps(spec)], cwd=REPO, stdout=lf, stderr=lf)
            t0 = time.monotonic(); killed = False
            while pr.poll() is None:
                try:
                    pr.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                el = e0 + time.monotonic() - t0
                hb.write_text(json.dumps(dict(elapsed=el)))
                if el > SY.WALL + HARD:
                    pr.kill(); pr.wait(); killed = True
        if not I.has_cached(name):
            if not killed:
                raise SystemExit(f"child for {name} exited with code {pr.returncode} without a result (see {LOGDIR})")
            z = np.load(ck) if ck.exists() else None
            if spec["mode"] == "trust":
                lev, rho, s = parse(spec["key"]); d = data(rho)
                fl = SY.FlatS(early(spec["key"])[0], d["X"], d["y"], lev, mu_of(lev))
                x = z["x"] if z is not None else fl.x0
                I.cached(name, lambda: dict(x=torch.tensor(x), info=dict(stop="2 h cap (hard kill; last checkpointed iterate)", message="killed",
                                                                        iters=int(len(z["theta2"])) if z is not None else 0),
                                            hist=dict(theta2=list(z["theta2"]) if z is not None else [])))
            else:
                f, _, _ = iflat(spec["key"], spec["kind"], spec["coef"])
                x = z["x"] if z is not None else np.load(spec["start"])
                cert, _, _ = certify_point(f, x)
                I.cached(name, lambda: dict(x=torch.tensor(x), info=dict(stop="wall-clock cap (2 h, cumulative; hard kill, last checkpointed iterate)",
                                                                        iters=int(z["iters"]) if z is not None else 0, wall_s=float(SY.WALL + HARD),
                                                                        message="killed", cert=cert, resumed=False)))
            wlog(f"    {name}: child hard-killed at the cap + 5 min; last checkpointed iterate used")
    obj, _ = I.cached(name, lambda: (_ for _ in ()).throw(RuntimeError(name)))
    return obj


def run_point_dose(spec, flat, wlog):
    """drop-in for infimum.run_point (used by infimum.analyse for the restart probe); spec carries the dose key"""
    obj = run_child(dict(spec, mode="point"), wlog)
    return I.npx(obj["x"]), obj["info"]


I.run_point = run_point_dose                   # infimum.analyse's restart probe runs through the dose child


# ------------------------------------------------------------------ one model (worker process)
def max_hidden_corr(net, X):
    _, s, _ = SY.parts(net, X)
    u = np.tanh(s)
    return float(max(abs(SY.corr(u[:, i], u[:, j])) for i in range(3) for j in range(i + 1, 3)))


def reason_trust(info, cert):
    s = info["stop"]
    if "iteration cap" in s:
        return "iteration cap"
    if "2 h" in s:
        return "wall-clock cap"
    names = dict(a="gradient criterion (a)", b="negative curvature (b)", c="Newton decrement (c)", d="null-space gradient (d)")
    return ", ".join(names[k] for k, ok in cert["crit"].items() if not ok)


def worker(key):
    lines = []
    wlog = lines.append
    lev, rho, s = parse(key)
    d = data(rho)
    net0, einfo = early(key)
    rec = dict(key=key, lev=lev, rho=rho, seed=s, early=einfo)
    # 1. polish (committed protocol), certification, degeneracy screen and probes
    t0 = time.monotonic()
    pol = run_child(dict(mode="trust", key=key, name=f"dose_pol_{key}.pt", ck=str(W / f"dose_pol_{key}_ck.npz")), wlog)
    xP, pinfo, hist = I.npx(pol["x"]), pol["info"], pol["hist"]
    fl = SY.FlatS(net0, d["X"], d["y"], lev, mu_of(lev))
    cert, _, _ = certify_point(fl, xP)
    netP = fl.to_net(xP)
    f, _, zin = iflat(key)
    info = dict(stop=pinfo["stop"], iters=pinfo["iters"], wall_s=time.monotonic() - t0, cert=cert, resumed=False)
    spec = dict(key=key, kind=None, coef=None, gtol=I.GRAD_TOL, elapsed0=0.0)
    a = I.analyse(f, f.net, zin, xP, info, spec, f"dose_pol_{key}", wlog)
    a.pop("wall_s", None)
    rec["polish"] = dict(pinfo=pinfo, **a)
    rec["polish"]["reason"] = None if a["certified"] else reason_trust(pinfo, cert)
    rec["uncertified"] = bool(not a["included"])
    e0d, ePd = SY.diagnostics(net0.double(), d["X"]), SY.diagnostics(netP, d["X"])
    rec["growth"] = ePd["theta2"] / e0d["theta2"]
    rec["growth_blocks"] = {b: ePd["norms"][b] / e0d["norms"][b] for b in e0d["norms"]}
    th = np.array(hist["theta2"], float)
    if rec["uncertified"] and len(th) >= 2:
        tail = th[-101:] if len(th) > 101 else th
        it = np.arange(len(tail))
        rec["runaway_rate"] = float(np.polyfit(it, np.log(tail), 1)[0]); rec["runaway_rate_short"] = bool(len(th) <= 101)
    else:
        rec["runaway_rate"] = None
    # 4. anatomy, 5. per-model diagnostics
    rec["anatomy"] = AN.delta_shares(net0.double(), netP)
    rec["affine_share"] = rec["anatomy"]["shares"]["L1 affine RBF"] + rec["anatomy"]["shares"]["L2 affine RBF"]
    rec["hidden_corr"] = dict(early=max_hidden_corr(net0.double(), d["X"]), polished=max_hidden_corr(netP, d["X"]))
    rec["cancel"] = dict(early=AN.cancellation(net0.double(), d["X"]), polished=AN.cancellation(netP, d["X"]))
    rec["Lstar_pol"] = float(f.Lstar(xP)); rec["Lstar_early"] = float(f.Lstar(f.x0))
    # 3. short regularisation path
    x = f.x0.copy(); path = []
    for k, g in enumerate(PATH_G):
        fg, _, _ = iflat(key, "L2", g)
        st = W / f"dose_path_{key}_g{k}_start.npy"
        if not st.exists():
            np.save(st, x)
        sp = dict(key=key, kind="L2", coef=g, start=str(st), name=f"dose_path_{key}_g{k}.pt", ck=str(W / f"dose_path_{key}_g{k}_ck.npz"),
                  gtol=I.GRAD_TOL, elapsed0=0.0)
        xk, inf_ = run_point_dose(sp, fg, wlog)
        r = I.analyse(fg, fg.net, zin, xk, inf_, sp, f"dose_path_{key}_g{k}", wlog)
        r.update(coef=g)
        path.append(r); x = xk
    rec["path"] = path
    rec["D"] = path[2]["theta2"] / path[1]["theta2"]
    rec["D_both_certified"] = bool(path[1]["included"] and path[2]["included"])
    rec["worker_log"] = lines
    (W / f"dose_rec_{key}.json").write_text(json.dumps(rec, indent=1, default=float))


def run_workers(keys, nproc):
    def one(k):
        if (W / f"dose_rec_{k}.json").exists():
            return
        LOGDIR.mkdir(exist_ok=True)
        with open(LOGDIR / f"worker_{k}.log", "a") as lf:
            r = subprocess.run([sys.executable, str(Path(__file__)), "--worker", k], cwd=REPO, stdout=lf, stderr=lf)
        if r.returncode != 0:
            raise SystemExit(f"worker {k} failed (code {r.returncode}); see {LOGDIR}/worker_{k}.log")
    with ThreadPoolExecutor(nproc) as ex:
        list(ex.map(one, keys))
    return {k: json.loads((W / f"dose_rec_{k}.json").read_text()) for k in keys}


# ------------------------------------------------------------------ runtime checks, gate, grid, report
def header(what):
    log("#" * 100)
    log(f"paper2/revision/dose.py {time.strftime('%Y-%m-%d %H:%M:%S')}: {what} (prereg dose_prereg.md, 1a18569)")
    log("#" * 100)


def choices():
    log("DISCREPANCIES / choices (files win):")
    log("  E1-E4 as recorded in the prereg (rho already a generator parameter, no edit; paired seeds c = 1 so rho = 0 is a new draw;")
    log("        synth.run_trust used unchanged inside a hard-killed child; teacher acceptance not repeated)")
    log("  G1 torch threads 4 and default BLAS threads, as synth.py; the committed polish (run_trust/FlatS) decides certification;")
    log("     the degeneracy screen, probes and short path use infimum.py's Flat/analyse/probes/polish on the same cell loss")
    log("  G2 runaway rate = least-squares slope of log ||theta||^2 per iteration over the last 100 polishing iterations (from")
    log("     run_trust's per-iteration ||theta||^2; after a hard kill: from the checkpointed accepted iterates)")
    log("  G3 hidden-unit correlation = largest |corr| between tanh(s_j) over unit pairs on the training rows")


def runtime_checks():
    log("\n==================== RUNTIME CHECKS (E1): realised correlations, marginals, paired draws ====================")
    out, ok = {}, True
    base = data(0.0)
    for rho in RHOS + ([CTRL_RHO] if CTRL_RHO not in RHOS else []):
        d = data(rho)
        row = {}
        for nm in ("X", "Xv", "Xt"):
            r12 = float(np.corrcoef(d[nm][:, 0], d[nm][:, 1])[0, 1]); r34 = float(np.corrcoef(d[nm][:, 2], d[nm][:, 3])[0, 1])
            row[nm] = dict(r12=r12, r34=r34, ok=bool(abs(r12 - rho) <= 0.02 and abs(r34 - rho) <= 0.02))
            ok &= row[nm]["ok"]
        X = d["X"]
        row["marginals"] = dict(mean=X.mean(0).tolist(), std=X.std(0).tolist(), clipped=(np.abs(X) >= 1).mean(0).tolist())
        row["paired_x1_x3_identical"] = bool(np.array_equal(X[:, [0, 2]], base["X"][:, [0, 2]]))
        ok &= row["paired_x1_x3_identical"]
        C = np.corrcoef(((X - X.mean(0)) / X.std(0)).T); ev = np.linalg.eigvalsh(C)
        row["gram"] = dict(cond=float(ev[-1] / ev[0]), min_eig=float(ev[0]), vif=np.diag(np.linalg.inv(C)).tolist())
        row["sigma"] = d["sigma"]
        out[rtag(rho)] = row
        log(f"  rho {rho:.2f}: corr(x1,x2)/corr(x3,x4) train {row['X']['r12']:.4f}/{row['X']['r34']:.4f}, val {row['Xv']['r12']:.4f}/{row['Xv']['r34']:.4f}, "
            f"test {row['Xt']['r12']:.4f}/{row['Xt']['r34']:.4f} (|r - rho| <= 0.02: {row['X']['ok'] and row['Xv']['ok'] and row['Xt']['ok']}) | "
            f"std {', '.join(f'{v:.4f}' for v in row['marginals']['std'])} | clipped {', '.join(f'{v:.4f}' for v in row['marginals']['clipped'])} | "
            f"x1, x3 identical to rho 0: {row['paired_x1_x3_identical']} | Gram cond {row['gram']['cond']:.2f}, min eig {row['gram']['min_eig']:.4f}, "
            f"VIF max {max(row['gram']['vif']):.2f} | sigma {d['sigma']:.4e}")
    log(f"  RUNTIME CHECKS: {'PASS' if ok else 'FAIL'}")
    return out, ok


def gate():
    header("RUNTIME CHECKS AND REPLICATION GATE (rho = 0.95, P0 and P1, 16 models)")
    choices()
    log("\n==================== inputs (SHA-256 verified, held in memory) ====================")
    I.blobs([f"synth_correlated_{l}_s{s}.pt" for l in LEVS for s in SEEDS])
    chk, ok_rt = runtime_checks()
    res = json.loads((REPO / "paper2/synth_results.json").read_text())["models"]
    sd = res["correlated_P0_s0"]
    log(f"  data check: committed early data term correlated_P0_s0 {sd['early_terms']['data']:.10e}")
    keys = [key_of(l, GATE_RHO, s) for l in LEVS for s in SEEDS]
    log("\n==================== REPLICATION GATE ====================")
    recs = run_workers(keys, 4)
    gate_ok, rows = True, []
    for k in keys:
        lev, rho, s = parse(k)
        com = I.ld(f"synth_correlated_{lev}_s{s}.pt")["early"]
        new = torch.load(io.BytesIO((W / f"dose_early_{k}.pt").read_bytes()), weights_only=False)["early"]
        dmax = max(float((com[n].double() - new[n].double()).abs().max()) for n in com)
        cm = res[f"correlated_{lev}_s{s}"]; r = recs[k]
        same = dict(early_weights=dmax == 0.0, certified=r["polish"]["certified"] == cm["certified"],
                    stop=r["polish"]["pinfo"]["stop"] == cm["polish"]["stop"], iters=r["polish"]["pinfo"]["iters"] == cm["polish"]["iters"],
                    growth=abs(r["growth"] - cm["growth"]) <= 1e-12 * max(1.0, abs(cm["growth"])), best_epoch=r["early"]["best_epoch"] == cm["early"]["best_epoch"])
        gate_ok &= same["early_weights"] and same["certified"]
        rows.append(dict(key=k, max_abs_diff_early=dmax, **same, growth_new=r["growth"], growth_committed=cm["growth"],
                         iters_new=r["polish"]["pinfo"]["iters"], iters_committed=cm["polish"]["iters"]))
        log(f"  {k}: early weights max |diff| {dmax:.1e} | best epoch {r['early']['best_epoch']} vs {cm['early']['best_epoch']} | "
            f"stop '{r['polish']['pinfo']['stop']}' vs '{cm['polish']['stop']}' | iters {r['polish']['pinfo']['iters']} vs {cm['polish']['iters']} | "
            f"certified {r['polish']['certified']} vs {cm['certified']} | growth x{r['growth']:.6f} vs x{cm['growth']:.6f}"
            + ("" if all(same.values()) else "  <-- DIFFERS: " + ", ".join(n for n, v in same.items() if not v)))
    log(f"  REPLICATION GATE: {'PASS' if gate_ok else 'FAIL'} (early weights identical and certification identical for all 16)"
        + ("" if all(all(v for kk, v in r.items() if isinstance(v, bool)) for r in rows) else "; other differences listed above (logged, files win)"))
    d = json.loads(RES.read_text()) if RES.exists() else {}
    d["runtime_checks"] = chk; d["gate"] = dict(rows=rows, passed=gate_ok, runtime_ok=ok_rt)
    RES.write_text(json.dumps(d, indent=1, default=float))
    if not gate_ok:
        log("  STOP: the replication gate failed; no other rho level is run (prereg).")


def inv_count(seq):
    return sum(1 for a, b in zip(seq, seq[1:]) if b < a)


def boot_slope(x, y, n=10000, seed=0):
    x, y = np.asarray(x), np.asarray(y)
    b = np.polyfit(x, y, 1)[0]
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0:
            bs.append(np.polyfit(x[i], y[i], 1)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(b), float(lo), float(hi)


def grid():
    d0 = json.loads(RES.read_text()) if RES.exists() else {}
    if not d0.get("gate", {}).get("passed"):
        raise SystemExit("the replication gate has not passed; the grid is not run")
    header("FULL GRID (rho 0, 0.5, 0.8, 0.9, 0.95, 0.99 x P0, P1; P2 control at 0.99)")
    choices()
    log("\n==================== inputs (SHA-256 verified, held in memory) ====================")
    I.blobs([f"synth_correlated_{l}_s{s}.pt" for l in LEVS for s in SEEDS])
    keys = [key_of(l, r, s) for r in RHOS for l in LEVS for s in SEEDS] + [key_of("P2", CTRL_RHO, s) for s in SEEDS]
    t0 = time.time()
    recs = run_workers(keys, 4)
    log(f"  all {len(keys)} models done ({(time.time() - t0) / 60:.1f} min this run)")
    chk = d0["runtime_checks"]
    log("\n==================== PER-MODEL ====================")
    for k in keys:
        r = recs[k]; p = r["polish"]
        log(f"  {k:12s} early ep {r['early']['best_epoch']:5d} | {p['pinfo']['stop'][:28]:28s} {p['pinfo']['iters']:3d} it | cert {str(p['certified'])[:5]:5s} "
            f"degen {('-' if p.get('degenerate') is None else str(p['degenerate'])[:5]):5s} {(p.get('class') or '-'):12s} | uncert {str(r['uncertified'])[:5]:5s} "
            f"{(p.get('reason') or ''):24s} | rate {I.fmt(r['runaway_rate'], '+.2e'):9s} | growth x{r['growth']:.2f} | affine {r['affine_share']:.3f} | "
            f"D {r['D']:.3f}{'' if r['D_both_certified'] else '*'} | path theta2 " + ", ".join(f"{q['theta2']:.3g}{'' if q['included'] else '*'}" for q in r["path"])
            + f" | hidden corr {r['hidden_corr']['polished']:.3f} | cancel L1 {r['cancel']['polished']['cross']:.2f} L2 {r['cancel']['polished']['layer2']:.2f}")
        for w in r.get("worker_log", []):
            log("   " + w)
    cells = {}
    for lev in LEVS + ["P2"]:
        for rho in (RHOS if lev != "P2" else [CTRL_RHO]):
            rr = [recs[key_of(lev, rho, s)] for s in SEEDS]
            Dc = [r["D"] for r in rr if r["D_both_certified"]]
            rates = [r["runaway_rate"] for r in rr if r["runaway_rate"] is not None]
            reasons, classes = {}, {}
            for r in rr:
                if r["uncertified"]:
                    rs = r["polish"].get("reason") or "PLATEAU by probe"; reasons[rs] = reasons.get(rs, 0) + 1
                if r["polish"].get("class"):
                    classes[r["polish"]["class"]] = classes.get(r["polish"]["class"], 0) + 1
            pathspread = []
            for k in range(3):
                th = [r["path"][k]["theta2"] for r in rr]
                xs = [I.npx(I.cached(f"dose_path_{key_of(lev, rho, s)}_g{k}.pt", lambda: None)[0]["x"]) for s in SEEDS]
                f, net, _ = iflat(key_of(lev, rho, 0))
                Xd = torch.tensor(data(rho)["X"], dtype=torch.float64)
                pr = []
                for x in xs:
                    with torch.no_grad():
                        pr.append(torch.func.functional_call(net, f.prm_at(x), (Xd,))[:, 0].numpy())
                Dd = [np.linalg.norm(a - b) / np.linalg.norm(a) for a, b in itertools.combinations(xs, 2)]
                P = [np.sqrt(np.mean((a - b) ** 2)) / a.std() for a, b in itertools.combinations(pr, 2)]
                pathspread.append(dict(coef=PATH_G[k], spread=(max(th) - min(th)) / float(np.median(th)), dist_med=float(np.median(Dd)),
                                       pred_med=float(np.median(P)), pred_max=float(max(P))))
            g = [r["growth"] for r in rr]
            cells[f"{lev}_r{rtag(rho)}"] = dict(
                lev=lev, rho=rho, certified=sum(1 for r in rr if not r["uncertified"]), uncert_frac=sum(r["uncertified"] for r in rr) / 8,
                reasons=reasons, classes=classes, D_med=float(np.median([r["D"] for r in rr])), D_cert_med=(float(np.median(Dc)) if Dc else None), n_D_cert=len(Dc),
                rate_med=(float(np.median(rates)) if rates else None), n_rates=len(rates), affine_med=float(np.median([r["affine_share"] for r in rr])),
                shares_med={kk: float(np.median([r["anatomy"]["shares"][kk] for r in rr])) for kk in rr[0]["anatomy"]["shares"]},
                hidden_corr_med=float(np.median([r["hidden_corr"]["polished"] for r in rr])),
                cancel_L1_med=float(np.median([r["cancel"]["polished"]["cross"] for r in rr])), cancel_L2_med=float(np.median([r["cancel"]["polished"]["layer2"] for r in rr])),
                growth_med=float(np.median(g)), growth_min=float(min(g)), growth_max=float(max(g)),
                Lstar_pol_med=float(np.median([r["Lstar_pol"] for r in rr])), gram=chk[rtag(rho)]["gram"], path_spread=pathspread)
    log("\n==================== PER-CELL TABLE (generated from dose_results.json) ====================")
    log("  cell        cert  uncert  reasons / probe classes                         median D (cert-only, n)   rate med    Gram cond  VIF max  hid.corr  cancel L1/L2  affine  growth med (range)        L* pol")
    for c, v in cells.items():
        rs = "; ".join(f"{a} {b}" for a, b in v["reasons"].items()) + (" | " + "; ".join(f"{a} {b}" for a, b in v["classes"].items()) if v["classes"] else "")
        log(f"  {c:10s}  {v['certified']}/8   {v['uncert_frac']:.3f}   {rs[:46]:46s}  {v['D_med']:8.3f} ({I.fmt(v['D_cert_med'], '.3f')}, {v['n_D_cert']})      "
            f"{I.fmt(v['rate_med'], '+.2e'):9s}  {v['gram']['cond']:9.2f}  {max(v['gram']['vif']):7.2f}  {v['hidden_corr_med']:.3f}     "
            f"{v['cancel_L1_med']:.2f}/{v['cancel_L2_med']:.2f}     {v['affine_med']:.3f}  x{v['growth_med']:.2f} (x{v['growth_min']:.2f}-x{v['growth_max']:.2f})  {v['Lstar_pol_med']:.4e}")
    log("\n  within-cell spread of the short paths (theta2 spread; raw pairwise distance median; prediction diff / std median (max)):")
    for c, v in cells.items():
        log(f"    {c:10s} " + " | ".join(f"g {q['coef']:.0e}: {q['spread']:.3f}, {q['dist_med']:.2f}, {q['pred_med']:.1e} ({q['pred_max']:.1e})" for q in v["path_spread"]))
    # verdicts
    ver = {}
    log("\n==================== VERDICTS (pre-registered R1-R4) ====================")
    for lev in LEVS:
        uf = [cells[f"{lev}_r{rtag(r)}"]["uncert_frac"] for r in RHOS]
        aff = [cells[f"{lev}_r{rtag(r)}"]["affine_med"] for r in RHOS]
        Dm = [cells[f"{lev}_r{rtag(r)}"]["D_med"] for r in RHOS]
        Dcm = [cells[f"{lev}_r{rtag(r)}"]["D_cert_med"] for r in RHOS]
        sp = float(spearmanr(RHOS, Dm).correlation)
        okc = all(v is not None for v in Dcm)
        spc = float(spearmanr(RHOS, Dcm).correlation) if okc else None
        R1 = inv_count(uf) <= 1; R3 = inv_count(aff) <= 1; R4 = sp >= 0.8; R4c = (spc is not None and spc >= 0.8)
        ver[lev] = dict(R1=R1, R3=R3, R4=R4, R4_cert=R4c, R4_sensitive=(R4 != R4c), uf=uf, aff=aff, D=Dm, D_cert=Dcm, spearman=sp, spearman_cert=spc,
                        inv_uf=inv_count(uf), inv_aff=inv_count(aff),
                        rho_star=next((r for r, u in zip(RHOS, uf) if u > 0.5), None), early_runaway=any(u > 0.5 for r, u in zip(RHOS, uf) if r <= 0.5))
        log(f"  {lev}: uncertified fraction over rho {', '.join(f'{u:.3f}' for u in uf)} ({inv_count(uf)} inversions) -> R1 {'HELD' if R1 else 'FAILED'}")
        log(f"  {lev}: median affine-RBF share {', '.join(f'{u:.3f}' for u in aff)} ({inv_count(aff)} inversions) -> R3 {'HELD' if R3 else 'FAILED'} (secondary)")
        log(f"  {lev}: median D (as reached) {', '.join(f'{u:.3f}' for u in Dm)}; Spearman {sp:+.3f} -> R4 {'HELD' if R4 else 'FAILED'}; "
            f"certified-only {', '.join(I.fmt(u, '.3f') for u in Dcm)}; Spearman {I.fmt(spc, '+.3f')} -> {'HELD' if R4c else 'FAILED'}"
            + ("  => R4 SENSITIVE TO CERTIFICATION" if R4 != R4c else ""))
    c2 = cells[f"P2_r{rtag(CTRL_RHO)}"]
    R2 = c2["certified"] >= 7 and c2["D_med"] <= 1.5
    ver["R2"] = dict(held=R2, certified=c2["certified"], D_med=c2["D_med"])
    log(f"  R2 {'HELD' if R2 else 'FAILED'}: P2 at rho 0.99 certified {c2['certified']}/8 (needs >= 7), median D {c2['D_med']:.3f} (needs <= 1.5)")
    br = []
    if not R2:
        br.append("R2 FAILS: the fix has a dose limit at rho = 0.99 with mu = 1e-5; qualify the paper's recommendation")
    P = ver
    if any(not P[l]["R1"] and P[l]["early_runaway"] for l in LEVS):
        br.append("R1 FAILS WITH THE RUNAWAY ALREADY AT rho <= 0.5 (" + ", ".join(l for l in LEVS if not P[l]["R1"] and P[l]["early_runaway"])
                  + "): correlation amplifies but is not necessary; qualify the paper's framing")
    elif any(not P[l]["R1"] for l in LEVS):
        br.append("R1 FAILS (non-monotone shape) for " + ", ".join(l for l in LEVS if not P[l]["R1"]) + ": report the shape as found; no threshold")
    elif all(P[l]["R1"] and P[l]["R4"] for l in LEVS) and R2:
        br.append("DOSE-RESPONSE ESTABLISHED: non-convergence and path divergence both increase with input correlation, and the null-space penalty closes the runaway at the strongest dose tested")
    elif all(P[l]["R1"] for l in LEVS):
        br.append("R1 HOLDS FOR BOTH BUT R4 FAILS FOR " + ", ".join(l for l in LEVS if not P[l]["R4"]) + ": the uncertified models may be budget-limited rather than diverging (per penalty)")
    if not br or (len(br) == 1 and br[0].startswith("R2 FAILS")):
        if any(P[l]["R1"] or P[l]["R4"] for l in LEVS) and not all(P[l]["R1"] and P[l]["R4"] for l in LEVS):
            br.append("PARTIAL: R1 or R4 holds for one penalty only; both penalties reported")
    log("  BRANCH: " + " | ".join(br) if br else "  BRANCH: none matched (reported as is)")
    # descriptive
    desc = {}
    for lev in LEVS:
        xs, ys = [], []
        for rho in RHOS:
            for s in SEEDS:
                r = recs[key_of(lev, rho, s)]
                if r["runaway_rate"] is not None:
                    xs.append(np.log(1 / (1 - rho))); ys.append(r["runaway_rate"])
        sl = boot_slope(xs, ys) if len(set(xs)) >= 2 else None
        Lm = [cells[f"{lev}_r{rtag(r)}"]["Lstar_pol_med"] for r in RHOS]
        desc[lev] = dict(rho_star=P[lev]["rho_star"], rate_slope=sl, n_rate=len(ys), Lstar_spearman=float(spearmanr(RHOS, Lm).correlation), Lstar_med=Lm)
        log(f"  DESCRIPTIVE {lev}: rho* (first uncertified fraction > 50%) {P[lev]['rho_star']}; runaway rate vs log(1/(1-rho)) slope "
            + (f"{sl[0]:+.3e} [{sl[1]:+.3e}, {sl[2]:+.3e}] over {len(ys)} uncertified models" if sl else f"not estimable ({len(ys)} uncertified models)")
            + f"; median polished L* {', '.join(f'{u:.4e}' for u in Lm)} (Spearman with rho {desc[lev]['Lstar_spearman']:+.3f})")
    ind = json.loads((REPO / "paper2/synth_results.json").read_text())["models"]
    for lev in LEVS:
        cc = sum(ind[f"independent_{lev}_s{s}"]["certified"] for s in SEEDS)
        log(f"  reference (E2): committed independent {lev} cell (seed 1000) certified {cc}/8; this study's rho = 0 cell (seed 1001) "
            f"certified {cells[f'{lev}_r0.00']['certified']}/8")
    d0.update(cells=cells, verdicts=ver, branch=br, descriptive=desc, models=recs)
    RES.write_text(json.dumps(d0, indent=1, default=float))
    figures(cells)
    log("\nwrote dose_results.json and figs/dose_response.png, figs/dose_anatomy.png")


def figures(cells):
    FIG.mkdir(exist_ok=True)
    xr = lambda r: np.log(1 / (1 - r))
    cols = {"P0": "#D55E00", "P1": "#0072B2"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for lev in LEVS:
        cc = [cells[f"{lev}_r{rtag(r)}"] for r in RHOS]
        axs[0].plot([xr(r) for r in RHOS], [c["uncert_frac"] for c in cc], marker="o", color=cols[lev], label=lev)
        axs[1].plot([xr(r) for r in RHOS], [c["D_med"] for c in cc], marker="o", color=cols[lev], label=f"{lev} (as reached)")
        axs[1].plot([xr(r) for r in RHOS], [c["D_cert_med"] if c["D_cert_med"] is not None else np.nan for c in cc], marker="s", mfc="none", ls=":", color=cols[lev], label=f"{lev} (certified only)")
    c2 = cells[f"P2_r{rtag(CTRL_RHO)}"]
    axs[0].scatter([xr(CTRL_RHO)], [c2["uncert_frac"]], marker="*", s=120, color="k", zorder=4, label="P2 control")
    axs[1].scatter([xr(CTRL_RHO)], [c2["D_med"]], marker="*", s=120, color="k", zorder=4, label="P2 control")
    for ax, lab in ((axs[0], "uncertified fraction (of 8)"), (axs[1], "median D = ||theta*(1e-7)||^2 / ||theta*(1e-5)||^2")):
        ax.set_xticks([xr(r) for r in RHOS]); ax.set_xticklabels([f"{r:g}" for r in RHOS]); ax.set_xlabel("rho (axis: log 1/(1 - rho))"); ax.set_ylabel(lab)
        ax.legend(fontsize=7)
    axs[1].set_yscale("log"); axs[1].axhline(1.5, color="grey", ls=":", lw=0.8)
    fig.tight_layout(); fig.savefig(FIG / "dose_response.png", dpi=110); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, lev in zip(axs, LEVS):
        cc = [cells[f"{lev}_r{rtag(r)}"] for r in RHOS]
        for kk in cc[0]["shares_med"]:
            ax.plot([xr(r) for r in RHOS], [c["shares_med"][kk] for c in cc], marker="o", label=kk)
        ax.set_xticks([xr(r) for r in RHOS]); ax.set_xticklabels([f"{r:g}" for r in RHOS]); ax.set_xlabel("rho"); ax.set_title(f"{lev}: median share of ||delta theta||^2")
    axs[0].set_ylabel("share"); axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "dose_anatomy.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--worker")
    ap.add_argument("--child")
    a = ap.parse_args()
    if a.child:
        child(json.loads(a.child))
    elif a.worker:
        worker(a.worker)
    elif a.gate:
        gate()
    elif a.grid:
        grid()

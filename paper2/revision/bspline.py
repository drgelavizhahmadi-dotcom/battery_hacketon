"""paper2/revision/bspline.py: revision step 5, B-spline replication of the synthetic 2 x 3 study. See bspline_prereg.md (5491481).
Run from the repo root with the pinned interpreter:
  /usr/bin/python3 paper2/revision/bspline.py --step0 >> paper2/revision/bspline_output.txt   (property gates, Step 0 (i) descriptive, (ii) gate)
  /usr/bin/python3 paper2/revision/bspline.py --mu    >> paper2/revision/bspline_output.txt   (mu calibration, after (ii) passed)
  /usr/bin/python3 paper2/revision/bspline.py --step1 >> paper2/revision/bspline_output.txt   (48 models; after the frozen mu is recorded)
Edges from bspline_basis.py (0ce6d00); data, training, polish (synth.run_trust / FlatS), certification and the infimum-test
probes / short path as committed. Every polish runs in its own child process (hard-killed at the cap + 5 min) and is cached in
gauge_weights/bsp_* (SHA-indexed). No weights in git."""
import argparse
import ast
import hashlib
import inspect
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
for p in (REPO, REPO / "paper2", REPO / "arch", HERE, REPO / "tests"):
    sys.path.insert(0, str(p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import anatomy_synth as AN
import bspline_basis as BS
import bspline_null as BN
import freeze as FZ
import infimum as I
import synth as SY
from certify import certify_point
from kan import WD

torch.set_num_threads(4)                       # as synth.py (34cc321)

CONDS = list(SY.CONDS.items())                 # [("independent", 0.0), ("correlated", 0.95)]
LEVS = ["P0", "P1", "P2"]
SEEDS = list(range(8))
PILOT = list(SY.PILOT)                         # 100-107, as committed
PATH_G = [1e-3, 1e-5, 1e-7]
HARD = 300
STALE = 600                                    # staleness watchdog (note to supplement 4b3f20a, 5bf5acd): no new accepted iterate for 10 min
W = I.W
LOGDIR = W / "bsp_logs"
RES = HERE / "bspline_results.json"
FIG = HERE / "figs"


def log(*a):
    print(*a, flush=True)


def load_res():
    return json.loads(RES.read_text()) if RES.exists() else {}


def save_res(key, obj):
    d = load_res(); d[key] = obj
    RES.write_text(json.dumps(d, indent=1, default=float))


# ------------------------------------------------------------------ teacher: the committed edge functions, refitted into the B-spline basis
def committed_edge_functions(s):
    """the L1 and L2 dicts of synth.build_teacher, taken from its source (files win), at frequency scale s"""
    src = inspect.getsource(SY.build_teacher)
    tree = ast.parse(src)
    ns = dict(np=np, s=s)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("L1", "L2"):
            exec(compile(ast.Module([node], []), "<build_teacher>", "exec"), ns)
    return ns["L1"], ns["L2"]


def build_teacher_bs(s):
    """mirrors synth.build_teacher: smoothness-penalised least squares on 401 points, basis [SiLU, B_0..B_7], R = LAM D'D + LAM K_SILU"""
    L1, L2 = committed_edge_functions(s)
    grid = np.linspace(-1, 1, 401)
    Bm = np.column_stack([grid / (1 + np.exp(-grid)), BS.basis(torch.tensor(grid)).numpy()])
    Dm = np.zeros((BS.N_B - 2, BS.N_B + 1))
    for k in range(BS.N_B - 2):
        Dm[k, 1 + k], Dm[k, 2 + k], Dm[k, 3 + k] = 1, -2, 1
    R = SY.LAM * (Dm.T @ Dm); R[0, 0] += SY.LAM * SY.K_SILU

    def fit(f):
        return np.linalg.solve(Bm.T @ Bm / len(grid) + R, Bm.T @ f(grid) / len(grid))
    T = BS.BSynthKAN().double()
    with torch.no_grad():
        for (i, j), f in L1.items():
            c = fit(f); T.layers[0].base.weight[j, i] = c[0]; T.layers[0].coef[j, i] = torch.tensor(c[1:])
        for j, f in L2.items():
            c = fit(f); T.layers[1].base.weight[0, j] = c[0]; T.layers[1].coef[0, j] = torch.tensor(c[1:])
        T.layers[0].base.bias.zero_(); T.layers[1].base.bias.zero_()
    return T.eval()


_D = {}


def teacher():
    if "T" not in _D:
        res = json.loads((REPO / "paper2/synth_results.json").read_text())
        _D["res"] = res; _D["scale"] = float(res["teacher_scale"])
        _D["T"] = build_teacher_bs(_D["scale"]); _D["T_rbf"] = SY.build_teacher(_D["scale"])
    return _D["T"]


def data(cond):
    if cond not in _D:
        T = teacher()
        ci = [c for c, _ in CONDS].index(cond); rho = dict(CONDS)[cond]
        X, Xv, Xt = SY.make_inputs(rho, 1000 + ci), SY.make_inputs(rho, 2000 + ci), SY.make_inputs(rho, 3000 + ci)
        Xo = SY.make_inputs(rho, 4000 + ci, independent=True)
        y0 = SY.predict(T, X); sig = 0.05 * y0.std()
        rng = np.random.default_rng(5000 + ci)
        y = y0 + sig * rng.standard_normal(SY.N); yv = SY.predict(T, Xv) + sig * rng.standard_normal(SY.N)
        _D[cond] = dict(X=X, Xv=Xv, Xt=Xt, Xo=Xo, y0=y0, y=y, yv=yv, sigma=float(sig))
    return _D[cond]


def frozen_mu():
    return float(load_res()["step0"]["mu_frozen"])


# ------------------------------------------------------------------ model keys: "<cond>_<lev>_s<seed>" (Step 1) or "pilot_m<mu>_s<seed>" (Step 0)
def parse(key):
    a, b, c = key.split("_")
    if a == "pilot":
        return "correlated", "P2", int(c[1:]), float(b[1:])
    return a, b, int(c[1:]), (frozen_mu() if b == "P2" else 0.0)


def early(key):
    cond, lev, s, mu = parse(key)
    d = data(cond)
    pen = lambda prm: SY.cell_penalty(prm, lev, mu)

    def make():
        net, info = FZ.train_early_generic(BS.BSynthKAN, s, d["X"], d["y"], d["Xv"], d["yv"], pen, WD)
        return dict(early=net.state_dict(), info=info)
    obj, _ = I.cached(f"bsp_early_{key}.pt", make)
    net = BS.BSynthKAN(); net.load_state_dict(obj["early"]); net.eval()
    return net, obj["info"]


def iflat(key, kind=None, coef=None):
    cond, lev, s, mu = parse(key)
    d = data(cond)
    net = early(key)[0].double()
    full = dict(net.named_parameters())
    f = I.Flat(net, full, list(full), I.syn_parts(net, d["X"], d["y"], lev, mu), None, {})
    if kind is not None:
        f.reg = I.make_reg(kind, coef, f.sl)
    Xd = torch.tensor(d["X"], dtype=torch.float64)
    return f, net, (lambda prm: Xd)


def flats(key):
    cond, lev, s, mu = parse(key)
    d = data(cond)
    return SY.FlatS(early(key)[0], d["X"], d["y"], lev, mu)


# ------------------------------------------------------------------ child processes (as dose.py after crash records 1 and 2)
def supervise(pr, ck, cap_s, stale_s, hb=None, e0=0.0, poll=5.0):
    """wait for child pr; kill it at the cumulative cap (monotonic, e0 already used) or when no new accepted iterate has been
    checkpointed (ck mtime) for stale_s seconds (counted from the later of the child's start and the last checkpoint).
    Returns None (exited), 'cap' or 'stale'."""
    t0m, t0w = time.monotonic(), time.time()
    while pr.poll() is None:
        try:
            pr.wait(timeout=poll)
        except subprocess.TimeoutExpired:
            pass
        if pr.poll() is not None:
            break
        el = e0 + time.monotonic() - t0m
        if hb is not None:
            hb.write_text(json.dumps(dict(elapsed=el)))
        last = max(t0w, ck.stat().st_mtime if ck.exists() else 0.0)
        if el > cap_s:
            pr.kill(); pr.wait(); return "cap"
        if time.time() - last > stale_s:
            pr.kill(); pr.wait(); return "stale"
    return None



def teacher_flat(spec):
    d = data(spec["cond"])
    return SY.FlatS(teacher(), d["X"], d["y"], spec["lev"], spec["mu"])


def child(spec):
    if spec["mode"] == "teacher":
        # Step 0 teacher polish (committed synth.run_trust, unchanged) in a hard-killed child; every accepted iterate checkpointed
        # with its loss, ||theta||^2 and wall-clock time (bspline crash record 1); non-finite subproblem caught (11e3663)
        fl = teacher_flat(spec)
        ck = Path(spec["ck"]); acc = dict(theta2=[], loss=[], t=[])

        def record(x):
            acc["theta2"].append(float((x ** 2).sum())); acc["loss"].append(float(fl.fun(x)[0])); acc["t"].append(time.time())
            tmp = ck.with_name(ck.stem + ".tmp.npz")
            np.savez(tmp, x=x, theta2=np.array(acc["theta2"]), loss=np.array(acc["loss"]), t=np.array(acc["t"])); os.replace(tmp, ck)
        if spec.get("terminate"):
            fl.hessian = BN.terminating_hessian(fl, null_aware=spec["lev"] != "P2", on_accept=record)
        else:
            orig = fl.hessian; last = dict(x=None)

            def hess(x):
                x = np.asarray(x)
                if last["x"] is None or not np.array_equal(last["x"], x):
                    last["x"] = x.copy(); record(x)
                return orig(x)
            fl.hessian = hess
        t0 = time.time()
        try:
            x, info, hist = SY.run_trust(fl); ended = "returned"
        except BN.CertifiedStop as e:
            x = e.x
            info = dict(stop="certified (terminated at the first accepted iterate that certifies)", message="certified", iters=len(acc["theta2"]) - 1)
            hist = dict(theta2=list(acc["theta2"])); ended = "certified"
        except (ValueError, np.linalg.LinAlgError) as e:
            z = np.load(ck); x = z["x"]
            info = dict(stop=f"solver error (trust-exact subproblem: {str(e)[:60]})", message=str(e), iters=int(len(z["theta2"]) - 1))
            hist = dict(theta2=list(z["theta2"])); ended = "solver error (caught)"
        I.cached(spec["name"], lambda: dict(x=torch.tensor(np.asarray(x)), info=info, hist=hist, ended=ended, wall_s=time.time() - t0,
                                            acc=dict(theta2=acc["theta2"], loss=acc["loss"], t=acc["t"])))
        return
    if spec["mode"] == "trust":
        fl = flats(spec["key"])
        ck, th = Path(spec["ck"]), []
        lev = parse(spec["key"])[1]

        def record(x):
            th.append(float((np.asarray(x) ** 2).sum()))
            np.savez(ck.with_name(ck.stem + ".tmp.npz"), x=np.asarray(x), theta2=np.array(th)); os.replace(ck.with_name(ck.stem + ".tmp.npz"), ck)
        fl.hessian = BN.terminating_hessian(fl, null_aware=lev != "P2", on_accept=record)

        def run():
            try:
                return SY.run_trust(fl)
            except BN.CertifiedStop as e:
                return e.x, dict(stop="certified (terminated at the first accepted iterate that certifies)", message="certified",
                                 iters=int(len(th) - 1)), dict(theta2=list(th))
            except (ValueError, np.linalg.LinAlgError) as e:
                z = np.load(ck)
                return z["x"], dict(stop=f"solver error (trust-exact subproblem: {str(e)[:60]})", message=str(e), iters=int(len(z["theta2"]) - 1)), \
                    dict(theta2=list(z["theta2"]))
        I.cached(spec["name"], lambda: (lambda r: dict(x=torch.tensor(r[0]), info=r[1], hist=r[2]))(run()))
    else:
        f, _, _ = iflat(spec["key"], spec["kind"], spec["coef"])
        x0 = np.load(spec["start"])
        na = spec["kind"] is None and parse(spec["key"])[1] != "P2"          # gamma = 0 objective of P0/P1: null-set-aware
        f.hessian = BN.terminating_hessian(f, null_aware=na, gtol=spec["gtol"])
        ck = Path(spec["ck"])

        def run():
            try:
                x, info = I.polish(f, x0, ck, spec["gtol"], spec["elapsed0"])
            except BN.CertifiedStop as e:
                z = np.load(ck) if ck.exists() else None
                x = e.x
                info = dict(stop="certified (terminated at the first accepted iterate that certifies)", iters=int(z["iters"]) if z is not None else 0,
                            wall_s=round(float(z["elapsed"]), 1) if z is not None else 0.0, message="certified", resumed=False)
            info["cert"] = BN.certify_na(f, x, na)[0]
            return torch.tensor(x), info
        I.cached(spec["name"], lambda: dict(zip(("x", "info"), run())))


def run_child(spec, wlog):
    name = spec["name"]
    if not I.has_cached(name):
        ck = Path(spec["ck"]); hb = W / (ck.stem + ".hb.json")
        if spec["mode"] == "trust":
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
            why = supervise(pr, ck, SY.WALL + HARD, STALE, hb=hb, e0=e0)
            killed = why is not None
        if not I.has_cached(name):
            if not killed:
                raise SystemExit(f"child for {name} exited with code {pr.returncode} without a result (see {LOGDIR})")
            z = np.load(ck) if ck.exists() else None
            if spec["mode"] == "trust":
                fl = flats(spec["key"]); x = z["x"] if z is not None else fl.x0
                I.cached(name, lambda: dict(x=torch.tensor(x), info=dict(stop=("solver error (hang); ended by the staleness watchdog (no new accepted iterate for 10 min)"
                                                                             if why == "stale" else "2 h cap (hard kill; last checkpointed iterate)"), message="killed",
                                                                        iters=int(len(z["theta2"])) if z is not None else 0),
                                            hist=dict(theta2=list(z["theta2"]) if z is not None else [])))
            else:
                f, _, _ = iflat(spec["key"], spec["kind"], spec["coef"]); x = z["x"] if z is not None else np.load(spec["start"])
                cert, _, _ = certify_point(f, x)
                I.cached(name, lambda: dict(x=torch.tensor(x), info=dict(stop=("solver error (hang); ended by the staleness watchdog (no new accepted iterate for 10 min)"
                                                                             if why == "stale" else "wall-clock cap (2 h, cumulative; hard kill, last checkpointed iterate)"),
                                                                        iters=int(z["iters"]) if z is not None else 0, wall_s=float(SY.WALL + HARD),
                                                                        message="killed", cert=cert, resumed=False)))
            wlog(f"    {name}: child killed ({'staleness watchdog' if why == 'stale' else 'cap + 5 min'}); last checkpointed iterate used")
    obj, _ = I.cached(name, lambda: (_ for _ in ()).throw(RuntimeError(name)))
    return obj


def run_teacher(spec, wlog):
    """teacher polish in a hard-killed child; returns dict(x, info, ended, wall_s, acc); a kill keeps the last checkpointed iterate"""
    name = spec["name"]
    if not I.has_cached(name):
        ck = Path(spec["ck"])
        if ck.exists():
            ck.unlink()
        LOGDIR.mkdir(exist_ok=True)
        with open(LOGDIR / f"{Path(name).stem}.log", "a") as lf:
            pr = subprocess.Popen([sys.executable, str(Path(__file__)), "--child", json.dumps(spec)], cwd=REPO, stdout=lf, stderr=lf)
            t0 = time.monotonic()
            why = supervise(pr, ck, SY.WALL + HARD, STALE)
            killed = why is not None
            el = time.monotonic() - t0
        if not I.has_cached(name):
            if not killed:
                raise SystemExit(f"teacher child {name} exited with code {pr.returncode} without a result (see {LOGDIR})")
            z = np.load(ck) if ck.exists() else None
            fl = teacher_flat(spec)
            x = z["x"] if z is not None else fl.x0
            acc = dict(theta2=z["theta2"].tolist(), loss=z["loss"].tolist(), t=z["t"].tolist()) if z is not None else dict(theta2=[], loss=[], t=[])
            stop_txt = ("solver error (hang); ended by the staleness watchdog (no new accepted iterate for 10 min)" if why == "stale"
                        else "solver error (hang); ended by the 2 h + 5 min child kill")
            I.cached(name, lambda: dict(x=torch.tensor(x), info=dict(stop=stop_txt, message="killed",
                                                                    iters=max(len(acc["theta2"]) - 1, 0)),
                                        hist=dict(theta2=acc["theta2"]), ended="killed", wall_s=el, acc=acc))
            wlog(f"    {name}: child killed ({'staleness watchdog' if why == 'stale' else '2 h + 5 min cap'}); last checkpointed accepted iterate kept")
    obj, _ = I.cached(name, lambda: (_ for _ in ()).throw(RuntimeError(name)))
    return obj


def classify_ending(obj):
    """supplement to cba2c07 (4b3f20a); 'CONVERGED' = terminated at the first certifying iterate (amendment 03638f8)"""
    if obj["ended"] == "returned":
        return "NORMAL COMPLETION"
    if obj["ended"] == "certified":
        return "CONVERGED"
    L, th = obj["acc"]["loss"], obj["acc"]["theta2"]
    n_acc = len(th) - 1
    if n_acc >= 10 and abs(L[-1] - L[-11]) / abs(L[-11]) < 1e-6 and abs(th[-1] - th[-11]) / abs(th[-11]) < 1e-6:
        return "STALLED-AT-MINIMUM"
    if n_acc >= 250:
        return "KILLED-SUBSTANTIAL"
    return "KILLED-IN-MOTION"


def run_point_bsp(spec, flat, wlog):
    obj = run_child(dict(spec, mode="point"), wlog)
    return I.npx(obj["x"]), obj["info"]


I.run_point = run_point_bsp                    # infimum.analyse's restart probe runs through this child


def reason(info, cert):
    s = info["stop"]
    if "iteration cap" in s:
        return "iteration cap"
    if "2 h" in s:
        return "wall-clock cap"
    if "solver error" in s:
        return "solver error"
    if "certified" in s:
        return None
    names = dict(a="criterion a (gradient)", b="criterion b (curvature)", c="criterion c (Newton decrement)", d="criterion d (null-space gradient)")
    return ", ".join(names[k] for k, ok in cert["crit"].items() if not ok)


def max_hidden_corr(net, X):
    _, s, _ = SY.parts(net, X)
    u = np.tanh(s)
    return float(max(abs(SY.corr(u[:, i], u[:, j])) for i in range(3) for j in range(i + 1, 3)))


# ------------------------------------------------------------------ one model (worker process)
def worker(key, with_path=True):
    lines = []
    wlog = lines.append
    cond, lev, s, mu = parse(key)
    d = data(cond)
    net0, einfo = early(key)
    rec = dict(key=key, cond=cond, lev=lev, seed=s, mu=mu, early=einfo)
    t0 = time.monotonic()
    pol = run_child(dict(mode="trust", key=key, name=f"bsp_pol_{key}.pt", ck=str(W / f"bsp_pol_{key}_ck.npz")), wlog)
    xP, pinfo, hist = I.npx(pol["x"]), pol["info"], pol["hist"]
    fl = flats(key)
    cert = BN.certify_na(fl, xP, null_aware=lev != "P2")[0]
    netP = fl.to_net(xP)
    f, _, zin = iflat(key)
    info = dict(stop=pinfo["stop"], iters=pinfo["iters"], wall_s=time.monotonic() - t0, cert=cert, resumed=False)
    spec = dict(key=key, kind=None, coef=None, gtol=I.GRAD_TOL, elapsed0=0.0)
    a = BN.analyse_c(f, f.net, zin, xP, info, spec, f"bsp_pol_{key}", wlog, run_point_bsp)
    a.pop("wall_s", None)
    rec["polish"] = dict(pinfo=pinfo, **a)
    rec["polish"]["reason"] = None if a["certified"] else reason(pinfo, cert)
    rec["certified"] = bool(a["included"])
    e0d, ePd = SY.diagnostics(net0.double(), d["X"]), SY.diagnostics(netP, d["X"])
    rec["growth"] = ePd["theta2"] / e0d["theta2"]
    rec["trust_theta2"] = hist["theta2"]
    rec["anatomy"] = AN.delta_shares(net0.double(), netP)
    rec["hidden_corr"] = dict(early=max_hidden_corr(net0.double(), d["X"]), polished=max_hidden_corr(netP, d["X"]))
    rec["cancel"] = dict(early=AN.cancellation(net0.double(), d["X"]), polished=AN.cancellation(netP, d["X"]))
    T = teacher()
    rec["rmse_off"] = dict(early=float(np.sqrt(np.mean((SY.predict(net0, d["Xo"]) - SY.predict(T, d["Xo"])) ** 2))),
                           polished=float(np.sqrt(np.mean((SY.predict(netP, d["Xo"]) - SY.predict(T, d["Xo"])) ** 2))))
    rec["rmse_in"] = dict(polished=float(np.sqrt(np.mean((SY.predict(netP, d["Xt"]) - SY.predict(T, d["Xt"])) ** 2))))
    if with_path:
        x = f.x0.copy(); path = []
        for k, g in enumerate(PATH_G):
            fg, _, _ = iflat(key, "L2", g)
            st = W / f"bsp_path_{key}_g{k}_start.npy"
            if not st.exists():
                np.save(st, x)
            sp = dict(key=key, kind="L2", coef=g, start=str(st), name=f"bsp_path_{key}_g{k}.pt", ck=str(W / f"bsp_path_{key}_g{k}_ck.npz"),
                      gtol=I.GRAD_TOL, elapsed0=0.0)
            xk, inf_ = run_point_bsp(sp, fg, wlog)
            r = BN.analyse_c(fg, fg.net, zin, xk, inf_, sp, f"bsp_path_{key}_g{k}", wlog, run_point_bsp)
            r.update(coef=g); path.append(r); x = xk
        rec["path"] = path
        rec["D"] = path[2]["theta2"] / path[1]["theta2"]
        rec["D_both_certified"] = bool(path[1]["included"] and path[2]["included"])
    rec["worker_log"] = lines
    (W / f"bsp_rec_{key}.json").write_text(json.dumps(rec, indent=1, default=float))


def run_workers(keys, nproc, path=True):
    def one(k):
        if (W / f"bsp_rec_{k}.json").exists():
            return
        LOGDIR.mkdir(exist_ok=True)
        with open(LOGDIR / f"worker_{k}.log", "a") as lf:
            r = subprocess.run([sys.executable, str(Path(__file__)), "--worker", k] + ([] if path else ["--nopath"]), cwd=REPO, stdout=lf, stderr=lf)
        if r.returncode != 0:
            raise SystemExit(f"worker {k} failed (code {r.returncode}); see {LOGDIR}/worker_{k}.log")
    with ThreadPoolExecutor(nproc) as ex:
        list(ex.map(one, keys))
    return {k: json.loads((W / f"bsp_rec_{k}.json").read_text()) for k in keys}


# ------------------------------------------------------------------ property gates, Step 0, Step 1
def header(what):
    log("#" * 100)
    log(f"paper2/revision/bspline.py {time.strftime('%Y-%m-%d %H:%M:%S')}: {what} (prereg bspline_prereg.md, 5491481; basis 0ce6d00)")
    log("#" * 100)


def choices():
    log("DISCREPANCIES / choices (files win):")
    log("  F1 mu calibration mirrors synth.py Step 0 exactly (grid 1e-6..1e-3 increasing; regularised teacher < 0.5 sigma AND certified in")
    log("     both conditions; pilot correlated-P2 seeds 100-107, max ||theta||^2 growth <= 3); the certification criterion (all 8 pilot")
    log("     models certified incl. certified-degenerate) is reported beside it; every mu is evaluated so both selections are known")
    log("  F2 P1 = P0 + LAM * K_SILU * sum w_silu^2 as committed; lambda kept, so the D2 strength relative to curvature differs from RBF")
    log("  F3 inputs clipped to [-1, 1] and layer-2 inputs tanh(s): no edge is evaluated outside [-1, 1]")
    log("  F4 no teacher smoothing rounds (stop if acceptance fails); committed frequency scale used")
    log("  F5 labels from the refitted B-spline teacher; inputs and noise draws as 34cc321")
    log("  H1 training = freeze.train_early_generic(BSynthKAN, ...) (synth.train_early with the model constructor as a parameter;")
    log("     identical otherwise, determinism-checked in revision step 2); polish = synth.run_trust on synth.FlatS, unchanged")


def gates():
    log("\n==================== PROPERTY TESTS (tests/test_bspline.py, committed 0ce6d00) ====================")
    import test_bspline as TB
    ok = True
    for name in ("test_a_partition_of_unity", "test_b_exact_affine_reproduction", "test_c_penalty_null_space", "test_d_null_basis_orthonormal"):
        try:
            getattr(TB, name)(); res = "PASS"
        except AssertionError as e:
            res = f"FAIL ({e})"; ok = False
        log(f"  {name}: {res}")
    pu = float((BS.basis(TB.XK).sum(-1) - 1).abs().max())
    xi = BS.greville(); Bm = BS.basis(TB.XK)
    ar = max(float((Bm @ (a + b * xi) - (a + b * TB.XK)).abs().max()) for a, b in [(0.0, 1.0), (1.0, 0.0), (-0.7, 2.3), (3.1, -1.9)])
    s = torch.linalg.svdvals(BS.d2_matrix())
    dev = TB.affinity_deviation()
    log(f"  (a) max |sum_j B_j(x) - 1| on [-1, 1] (10,001 points + knots): {pu:.2e}")
    log(f"  (b) Greville abscissae {', '.join(f'{v:.2f}' for v in xi.tolist())}; max affine-reproduction error {ar:.2e}")
    log(f"  (c) D2 singular values {', '.join(f'{v:.3e}' for v in s.tolist())} -> null-space dimension {BS.N_B - int((s > 1e-10 * s.max()).sum())}")
    log(f"  (d) ||NB'NB - I||_max {float((BS.null_basis().T @ BS.null_basis() - torch.eye(2, dtype=torch.float64)).abs().max()):.2e}; "
        f"NB equals synth.NB: {bool(torch.allclose(BS.null_basis(), SY.NB.double(), atol=1e-14))}")
    log(f"  (e) DESCRIPTIVE: null-space basis functions' max deviation from affinity on [-1, 1]: {dev[0]:.2e}, {dev[1]:.2e}")
    log(f"  PROPERTY GATES: {'PASS' if ok else 'FAIL'}")
    return ok, dict(pu=pu, affine=ar, d2_sv=s.tolist(), affinity_dev=dev)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def step0():
    header("PROPERTY GATES AND STEP 0 (teacher refit, acceptance, mu calibration; calibration only, no results)")
    choices()
    ok, props = gates()
    if not ok:
        log("STOP: a property gate failed (prereg)"); save_res("properties", dict(ok=False, **props)); return
    save_res("properties", dict(ok=True, **props))
    log("\n==================== DATA (34cc321 generator and seeds; labels from the refitted teacher, F5) ====================")
    T, Tr = teacher(), _D["T_rbf"]
    rows = {}
    for ci, (cond, rho) in enumerate(CONDS):
        d = data(cond)
        ref = dict(X=SY.make_inputs(rho, 1000 + ci), Xv=SY.make_inputs(rho, 2000 + ci), Xt=SY.make_inputs(rho, 3000 + ci), Xo=SY.make_inputs(rho, 4000 + ci, independent=True))
        same = {k: sha(d[k]) == sha(v) for k, v in ref.items()}
        y0r = SY.predict(Tr, d["X"]); sig_r = 0.05 * y0r.std()
        fid = float(np.sqrt(np.mean((d["y0"] - y0r) ** 2)))
        rows[cond] = dict(sha={k: sha(d[k]) for k in ref}, identical=same, fidelity_rms=fid, sigma=d["sigma"], sigma_rbf=float(sig_r), fid_ok=fid < 0.1 * d["sigma"])
        log(f"  {cond}: inputs X/Xv/Xt/Xo SHA-256 {', '.join(sha(d[k])[:12] for k in ref)} (identical to synth.make_inputs: {all(same.values())}); "
            f"sigma {d['sigma']:.4e} (RBF teacher {sig_r:.4e})")
    log("\n==================== STEP 0 (i): refit fidelity (gate WITHDRAWN as mis-specified, amendment cba2c07; DESCRIPTIVE) ====================")
    for cond, r in rows.items():
        log(f"  {cond}: RMS(B-spline teacher - committed RBF teacher) on the training inputs {r['fidelity_rms']:.3e} = {r['fidelity_rms'] / r['sigma']:.4f} sigma "
            f"(needs < 0.1 sigma) -> {'ok' if r['fid_ok'] else 'FAIL'}")
    with torch.no_grad():
        d2, si, q = SY.penalty_terms({k: v for k, v in T.named_parameters()})
    log(f"  refitted teacher: D2 term {float(d2):.3e}, SiLU term {float(si):.3e}, Q {float(q):.3f}")
    log("  per-edge RMS error against the analytic teacher functions on 401 points of [-1, 1] (DESCRIPTIVE; amendment cba2c07):")
    L1f, L2f = committed_edge_functions(_D["scale"])
    g = torch.linspace(-1, 1, 401, dtype=torch.float64); edge_err = []
    for (lay, key_, f) in [(0, k, f) for k, f in L1f.items()] + [(1, (j, 0), f) for j, f in L2f.items()]:
        i, j = key_
        tru = f(g.numpy())
        with torch.no_grad():
            eb = T.layers[lay].edge(i, g)[:, j].numpy(); er = Tr.double().layers[lay].edge(i, g)[:, j].numpy()
        edge_err.append(dict(layer=lay + 1, edge=[i, j], bspline=float(np.sqrt(np.mean((eb - tru) ** 2))), rbf=float(np.sqrt(np.mean((er - tru) ** 2)))))
        log(f"    layer {lay + 1} edge {key_}: B-spline {edge_err[-1]['bspline']:.2e}, RBF {edge_err[-1]['rbf']:.2e}")
    for ci, (cond, rho) in enumerate(CONDS):
        d = data(cond)
        ana = analytic_predict(d["X"])
        rows[cond]["analytic_rms_bspline"] = float(np.sqrt(np.mean((d["y0"] - ana) ** 2)))
        rows[cond]["analytic_rms_rbf"] = float(np.sqrt(np.mean((SY.predict(Tr, d["X"]) - ana) ** 2)))
        log(f"  {cond}: prediction RMS against the analytic-edge network: B-spline {rows[cond]['analytic_rms_bspline'] / d['sigma']:.4f} sigma, "
            f"RBF {rows[cond]['analytic_rms_rbf'] / d['sigma']:.4f} sigma")
    log("  STEP 0 (i): WITHDRAWN as mis-specified (amendment cba2c07); reported descriptively above")
    log("\n==================== STEP 0 (ii): polish stability (synth.run_trust under P0 and P1; hard-killed children, at most 2 in parallel) ====================")
    lines = []
    specs = [dict(mode="teacher", cond=c, lev=l, mu=0.0, name=f"bsp_teacher_{c}_{l}.pt", ck=str(W / f"bsp_teacher_{c}_{l}_ck.npz"))
             for c, _ in CONDS for l in ("P0", "P1")]
    with ThreadPoolExecutor(2) as ex:
        objs = list(ex.map(lambda s: run_teacher(s, lines.append), specs))
    stab, gate_ok, motion = {}, True, []
    for sp, obj in zip(specs, objs):
        d = data(sp["cond"]); fl = teacher_flat(sp); x = I.npx(obj["x"])
        dp = float(np.sqrt(np.mean((SY.predict(fl.to_net(x), d["X"]) - d["y0"]) ** 2)))
        rc, _, _ = certify_point(fl, x)
        cls = classify_ending(obj)
        ts = np.array(obj["acc"]["t"]); it_t = np.diff(ts) if len(ts) > 1 else np.array([])
        r = dict(ending=cls, ended=obj["ended"], stop=obj["info"]["stop"], n_accepted=len(obj["acc"]["theta2"]) - 1, wall_s=obj["wall_s"],
                 iter_time_median=(float(np.median(it_t)) if len(it_t) else None), iter_time_max=(float(it_t.max()) if len(it_t) else None),
                 dpred=dp, ok=dp < 0.5 * d["sigma"], certified=bool(rc["certified"]),
                 growth=(obj["acc"]["theta2"][-1] / obj["acc"]["theta2"][0]) if obj["acc"]["theta2"] else None)
        stab[f"{sp['cond']}_{sp['lev']}"] = r
        if cls == "KILLED-IN-MOTION":
            motion.append(f"{sp['cond']}_{sp['lev']}")
        else:
            gate_ok &= r["ok"]
        log(f"  {sp['cond']:11s} {sp['lev']}: ending {cls} ({r['stop'][:60]}); {r['n_accepted']} accepted iterations; wall {r['wall_s']:.0f} s; "
            f"per-iteration median {I.fmt(r['iter_time_median'], '.2f')} s (max {I.fmt(r['iter_time_max'], '.1f')} s) | RMS prediction change {dp:.2e} vs "
            f"0.5 sigma {0.5 * d['sigma']:.2e} -> {'ok' if r['ok'] else 'FAIL'}; certified {r['certified']}")
    for l in lines:
        log("   " + l)
    hung = [k for k, r in stab.items() if r["ended"] == "killed"]
    log(f"  HANG CENSUS: {len(hung)}/4 polishes hung (" + (", ".join(f"{k} after {stab[k]['n_accepted']} accepted iterations" for k in hung) if hung else "none") + ")")
    s0 = dict(data=rows, edge_errors=edge_err, stability=stab)
    if motion:
        s0["acceptance"] = False; save_res("step0", s0)
        log(f"  STOP: KILLED-IN-MOTION for {', '.join(motion)}: the polish-stability gate is uninformative (supplement 4b3f20a)"); return
    s0["acceptance"] = bool(gate_ok)
    log(f"  STEP 0 (ii) POLISH-STABILITY GATE: {'PASS' if gate_ok else 'FAIL'}")
    save_res("step0", s0)
    if not gate_ok:
        log("STOP: Step 0 (ii) failed; Step 1 not run (prereg)")


def recert():
    """amendment 03638f8 item 4: re-certify the two hung Step 0 (ii) records null-set-aware; CONVERGED-AT-KILL if they pass"""
    header("STEP 0 (ii) RE-CERTIFICATION (POST-HOC amendment 03638f8, item 4; no new polish)")
    s0 = load_res()["step0"]; stab = s0["stability"]
    gate_ok, motion = True, []
    for k, r in stab.items():
        cond, lev = k.split("_")
        sp = dict(mode="teacher", cond=cond, lev=lev, mu=0.0, name=f"bsp_teacher_{cond}_{lev}.pt")
        obj, _ = I.cached(sp["name"], lambda: (_ for _ in ()).throw(RuntimeError(sp["name"])))
        fl = teacher_flat(sp); x = I.npx(obj["x"])
        c_na = BN.certify_na(fl, x, True)[0]; c_pl = certify_point(fl, x)[0]
        r["cert_null_aware"] = c_na; r["certified_null_aware"] = bool(c_na["certified"])
        if r["ending"] == "KILLED-IN-MOTION" and c_na["certified"]:
            r["ending_amended"] = "CONVERGED-AT-KILL"
        else:
            r["ending_amended"] = r["ending"]
        if r["ending_amended"] == "KILLED-IN-MOTION":
            motion.append(k)
        else:
            gate_ok &= r["ok"]
        log(f"  {k:14s}: original ending {r['ending']:18s} -> {r['ending_amended']:18s} | null-set-aware certification {c_na['certified']} "
            f"(crit {c_na['crit']}; max|g| {c_na['grad']:.1e}; complement lambda_min/max {c_na['mineig']:+.2e}/{c_na['maxeig']:.2e}; complement near-null "
            f"{c_na['n_null']}; Newton {c_na['newton']:.1e}; ||g along null set|| {c_na['grad_along_null']:.1e}); plain certify_point {c_pl['certified']} "
            f"(n_null {c_pl['n_null']}) | RMS prediction change {r['dpred']:.2e} -> {'ok' if r['ok'] else 'FAIL'}")
    log("  the 10-iteration plateau window of supplement 4b3f20a is recorded as unable to detect quadratic convergence (amendment 03638f8)")
    s0["acceptance_amended"] = bool(gate_ok and not motion)
    log(f"  STEP 0 (ii) POLISH-STABILITY GATE (amended classification): {'PASS' if s0['acceptance_amended'] else 'FAIL'}"
        + (f" (still KILLED-IN-MOTION: {motion})" if motion else ""))
    save_res("step0", s0)


def validate():
    """amendment 03638f8 item 5: the two hung independent polishes re-run with certification-based termination"""
    header("VALIDATION OF CERTIFICATION-BASED TERMINATION (amendment 03638f8, item 5)")
    lines = []
    specs = [dict(mode="teacher", cond="independent", lev=l, mu=0.0, terminate=True, name=f"bsp_teacher_independent_{l}_term.pt",
                  ck=str(W / f"bsp_teacher_independent_{l}_term_ck.npz")) for l in ("P0", "P1")]
    with ThreadPoolExecutor(2) as ex:
        objs = list(ex.map(lambda s: run_teacher(s, lines.append), specs))
    out, ok = {}, True
    for sp, new, lim in zip(specs, objs, (48, 21)):
        old, _ = I.cached(f"bsp_teacher_independent_{sp['lev']}.pt", lambda: None)
        n = len(new["acc"]["theta2"])
        Lo, To = np.array(old["acc"]["loss"][:n]), np.array(old["acc"]["theta2"][:n])
        Ln, Tn = np.array(new["acc"]["loss"]), np.array(new["acc"]["theta2"])
        m = min(len(Lo), n)
        dev = float(max((np.abs(Ln[:m] - Lo[:m]) / np.maximum(1, np.abs(Lo[:m]))).max(), (np.abs(Tn[:m] - To[:m]) / np.maximum(1, np.abs(To[:m]))).max()))
        stop_it = n - 1
        cls = classify_ending(new)
        good = dev <= 1e-12 and m == n and stop_it <= lim and cls == "CONVERGED"
        ok &= good
        out[sp["lev"]] = dict(stop_iteration=stop_it, limit=lim, max_rel_dev=dev, ending=cls, stop=new["info"]["stop"], wall_s=new["wall_s"], pass_=good)
        log(f"  independent {sp['lev']}: stopped at accepted iteration {stop_it} (must be <= {lim}); ending {cls}; per-iterate loss and ||theta||^2 vs the "
            f"f28515c checkpoints: max relative deviation {dev:.1e} over {m} iterates (<= 1e-12 required); wall {new['wall_s']:.1f} s -> {'PASS' if good else 'FAIL'}")
    for l in lines:
        log("   " + l)
    log("  (the checkpoints record each accepted iterate's loss and ||theta||^2; the comparison is iterate by iterate on these)")
    log(f"  VALIDATION: {'PASS' if ok else 'FAIL'}")
    save_res("termination_validation", dict(rows=out, passed=ok))


def analytic_predict(X):
    """the network with the exact analytic teacher edge functions (descriptive reference)"""
    L1f, L2f = committed_edge_functions(_D["scale"])
    s = np.zeros((len(X), 3))
    for (i, j), f in L1f.items():
        s[:, j] += f(X[:, i])
    h = np.tanh(s)
    return sum(f(h[:, j]) for j, f in L2f.items())


def mu_calibration():
    s0 = load_res().get("step0", {})
    if not s0.get("acceptance"):
        raise SystemExit("Step 0 (ii) has not passed; no mu calibration")
    header("STEP 0: mu calibration (synth.py Step 0 mirrored exactly; teacher polishes in hard-killed children; every mu evaluated)")
    choices()
    T = teacher(); calib = []; lines = []
    for mu in SY.MU_GRID:
        row = dict(mu=mu); okt = True
        specs = [dict(mode="teacher", cond=c, lev="P2", mu=mu, terminate=True, name=f"bsp_teacher_{c}_P2_m{mu:g}.pt",
                      ck=str(W / f"bsp_teacher_{c}_P2_m{mu:g}_ck.npz")) for c, _ in CONDS]
        with ThreadPoolExecutor(2) as ex:
            objs = list(ex.map(lambda s: run_teacher(s, lines.append), specs))
        for sp, obj in zip(specs, objs):
            d = data(sp["cond"]); fl = teacher_flat(sp); x = I.npx(obj["x"])
            dp = float(np.sqrt(np.mean((SY.predict(fl.to_net(x), d["X"]) - d["y0"]) ** 2)))
            rc = BN.certify_na(fl, x, False)[0]
            row[f"{sp['cond']}_dpred"], row[f"{sp['cond']}_cert"], row[f"{sp['cond']}_ending"] = dp, bool(rc["certified"]), classify_ending(obj)
            okt &= dp < 0.5 * d["sigma"] and rc["certified"]
            log(f"  mu {mu:g} regularised teacher {sp['cond']:11s}: RMS change {dp:.2e} (0.5 sigma {0.5 * d['sigma']:.2e}); certified {rc['certified']}; "
                f"ending {classify_ending(obj)} ({obj['info']['stop'][:50]}), {len(obj['acc']['theta2']) - 1} accepted iterations")
        row["teacher_ok"] = bool(okt)
        if okt:
            keys = [f"pilot_m{mu:g}_s{sd}" for sd in PILOT]
            recs = run_workers(keys, 2, path=False)
            gr = [r["growth"] for r in recs.values()]
            ncert = sum(r["certified"] for r in recs.values())
            row.update(pilot_max_growth=float(max(gr)), pilot_growth=gr, pilot_certified=ncert,
                       growth_ok=bool(max(gr) <= 3), cert_ok=bool(ncert == len(PILOT)),
                       pilot_stops={k: r["polish"]["pinfo"]["stop"] for k, r in recs.items()})
            log(f"  mu {mu:g} pilot (seeds {PILOT[0]}-{PILOT[-1]}, correlated-P2): max ||theta||^2 growth x{max(gr):.2f} -> {'ok' if max(gr) <= 3 else 'FAIL'} "
                f"(committed rule); certified {ncert}/{len(PILOT)} incl. certified-degenerate (descriptive criterion: {'ok' if ncert == len(PILOT) else 'FAIL'})")
        else:
            row.update(growth_ok=False, cert_ok=False)
            log(f"  mu {mu:g}: regularised-teacher check failed; no pilot (as committed)")
        calib.append(row)
    for l in lines:
        log("   " + l)
    pick_g = next((r["mu"] for r in calib if r["teacher_ok"] and r["growth_ok"]), None)
    pick_c = next((r["mu"] for r in calib if r["teacher_ok"] and r["cert_ok"]), None)
    s0.update(calibration=calib, mu_growth=pick_g, mu_cert=pick_c, mu_frozen=pick_g, disagree=(pick_g != pick_c))
    save_res("step0", s0)
    if pick_g is None:
        log("STOP: no mu passed the committed calibration; Step 1 not run (prereg)"); return
    log(f"  FROZEN: mu = {pick_g:g} (committed growth criterion); the certification criterion would select {pick_c}"
        + ("  <-- THE TWO CRITERIA DISAGREE (the growth criterion decides, prereg F1)" if pick_g != pick_c else " (agree)"))


def step1():
    s0 = load_res().get("step0", {})
    if not s0.get("mu_frozen"):
        raise SystemExit("no frozen mu; run --step0 first")
    header(f"STEP 1 (48 models; frozen mu = {s0['mu_frozen']:g})")
    choices()
    keys = [f"{c}_{l}_s{s}" for c, _ in CONDS for l in LEVS for s in SEEDS]
    t0 = time.time()
    recs = run_workers(keys, 4)
    log(f"  all {len(keys)} models done ({(time.time() - t0) / 60:.1f} min this run)")
    log("\n==================== PER-MODEL ====================")
    for k in keys:
        r = recs[k]; p = r["polish"]
        log(f"  {k:20s} early ep {r['early']['best_epoch']:5d} | {p['pinfo']['stop'][:28]:28s} {p['pinfo']['iters']:3d} it | cert {str(p['certified'])[:5]:5s} "
            f"degen {('-' if p.get('degenerate') is None else str(p['degenerate'])[:5]):5s} {(p.get('class') or '-'):12s} | included {str(r['certified'])[:5]:5s} "
            f"{(p.get('reason') or ''):26s} | growth x{r['growth']:.2f} | D {r['D']:.3f}{'' if r['D_both_certified'] else '*'} | "
            f"hidden corr {r['hidden_corr']['polished']:.3f} | cancel L1 {r['cancel']['polished']['cross']:.2f} L2 {r['cancel']['polished']['layer2']:.2f} | "
            f"off-RMSE {r['rmse_off']['early']:.4f} -> {r['rmse_off']['polished']:.4f}")
        for w in r.get("worker_log", []):
            log("   " + w)
    cells = {}
    for cond, _ in CONDS:
        for lev in LEVS:
            rr = [recs[f"{cond}_{lev}_s{s}"] for s in SEEDS]
            reasons, classes = {}, {}
            for r in rr:
                if not r["certified"]:
                    x = r["polish"].get("reason") or "PLATEAU by probe"; reasons[x] = reasons.get(x, 0) + 1
                if r["polish"].get("class"):
                    classes[r["polish"]["class"]] = classes.get(r["polish"]["class"], 0) + 1
            Dc = [r["D"] for r in rr if r["D_both_certified"]]
            g = [r["growth"] for r in rr]
            cells[f"{cond}_{lev}"] = dict(
                certified=sum(r["certified"] for r in rr), reasons=reasons, classes=classes,
                D_med=float(np.median([r["D"] for r in rr])), D_cert_med=(float(np.median(Dc)) if Dc else None), n_D_cert=len(Dc),
                shares_med={kk: float(np.median([r["anatomy"]["shares"][kk] for r in rr])) for kk in rr[0]["anatomy"]["shares"]},
                hidden_corr_med=float(np.median([r["hidden_corr"]["polished"] for r in rr])),
                cancel_L1_med=float(np.median([r["cancel"]["polished"]["cross"] for r in rr])), cancel_L2_med=float(np.median([r["cancel"]["polished"]["layer2"] for r in rr])),
                growth_med=float(np.median(g)), growth_min=float(min(g)), growth_max=float(max(g)),
                rmse_off_med=float(np.median([r["rmse_off"]["polished"] for r in rr])))
    log("\n==================== PER-CELL TABLE (generated from bspline_results.json) ====================")
    log("  cell             cert  failure reasons / probe classes                     median D (cert-only, n)   affine (L1+L2)  hid.corr  cancel L1/L2  growth med (range)        off-RMSE")
    for c, v in cells.items():
        rs = "; ".join(f"{a} {b}" for a, b in v["reasons"].items()) + (" | " + "; ".join(f"{a} {b}" for a, b in v["classes"].items()) if v["classes"] else "")
        aff = v["shares_med"]["L1 affine RBF"] + v["shares_med"]["L2 affine RBF"]
        log(f"  {c:15s}  {v['certified']}/8   {rs[:50]:50s}  {v['D_med']:8.3f} ({I.fmt(v['D_cert_med'], '.3f')}, {v['n_D_cert']})      {aff:.3f}           "
            f"{v['hidden_corr_med']:.3f}     {v['cancel_L1_med']:.2f}/{v['cancel_L2_med']:.2f}     x{v['growth_med']:.2f} (x{v['growth_min']:.2f}-x{v['growth_max']:.2f})  {v['rmse_off_med']:.4f}")
    log("  (anatomy shares use anatomy_synth.delta_shares; its 'affine RBF' / 'curved RBF' labels are the affine (null-space) and curved")
    log("   parts of the B-spline coefficients here)")
    log("  NOTE (prereg F2): lambda is kept as committed, so the strength of the second-difference penalty relative to the curvature it")
    log("  measures differs between the B-spline and RBF bases.")
    cc = lambda c: cells[c]
    B1 = {c: cc(f"correlated_{c}")["certified"] <= 2 for c in ("P0", "P1")}
    B2 = {}
    for cond, _ in CONDS:
        v = cc(f"{cond}_P2")
        r_as = v["certified"] >= 7 and v["D_med"] <= 1.5
        r_c = v["certified"] >= 7 and (v["D_cert_med"] is not None and v["D_cert_med"] <= 1.5)
        B2[cond] = dict(held=r_as, held_cert=r_c, sensitive=(r_as != r_c), certified=v["certified"], D_med=v["D_med"], D_cert_med=v["D_cert_med"])
    B3 = {c: cc(f"independent_{c}")["certified"] >= 6 for c in ("P0", "P1")}
    log("\n==================== VERDICTS (pre-registered B1-B3) ====================")
    for c in ("P0", "P1"):
        log(f"  B1 correlated {c}: certified {cc('correlated_' + c)['certified']}/8 (needs <= 2) -> {'HELD' if B1[c] else 'FAILED'}")
    for cond, v in B2.items():
        log(f"  B2 {cond} P2: certified {v['certified']}/8 (needs >= 7), median D {v['D_med']:.3f} (certified-only {I.fmt(v['D_cert_med'], '.3f')}) (needs <= 1.5) -> "
            f"{'HELD' if v['held'] else 'FAILED'}" + ("  => SENSITIVE TO CERTIFICATION" if v["sensitive"] else ""))
    for c in ("P0", "P1"):
        log(f"  B3 independent {c}: certified {cc('independent_' + c)['certified']}/8 (needs >= 6) -> {'HELD' if B3[c] else 'FAILED'}")
    b1, b2, b3 = all(B1.values()), all(v["held"] for v in B2.values()), all(B3.values())
    br = []
    if not b1:
        br.append("1. B1 FAILS: the effect does not carry over at this setting; the paper's scope is RBF edges")
    if b1 and b2:
        br.append("2. B1 AND B2 HOLD: the runaway and its fix carry over to B-spline KANs")
    if b1 and not b2:
        legs = [f"{cond} ({'certification' if v['certified'] < 7 else ''}{' and ' if v['certified'] < 7 and v['D_med'] > 1.5 else ''}{'D' if v['D_med'] > 1.5 else ''})" for cond, v in B2.items() if not v["held"]]
        br.append("3. B1 HOLDS, B2 FAILS: the runaway carries over but the null-space fix at this mu does not close it; failed leg(s): " + ", ".join(legs))
    if not b3:
        br.append("4. B3 FAILS: the B-spline basis is less benign on independent inputs than the RBF basis")
    if not br:
        br.append("5. PARTIAL: per-cell results reported")
    for b in br:
        log("  BRANCH " + b)
    if s0.get("disagree"):
        log(f"  NOTE: the mu-calibration criteria disagreed (growth {s0['mu_growth']}, certification {s0['mu_cert']}); the growth criterion decided (F1)")
    log("  DESCRIPTIVE caveat: median D alone does not separate a distant finite minimum from a receding one (dose POST-HOC, 75b778b).")
    save_res("step1", dict(cells=cells, verdicts=dict(B1=B1, B2=B2, B3=B3), branch=br, models=recs))
    figures(recs)
    log("\nwrote bspline_results.json and figs/bspline_theta_trajectories.png, figs/bspline_offmanifold_vs_cancellation.png")


def figures(recs):
    FIG.mkdir(exist_ok=True)
    cols = {"P0": "#D55E00", "P1": "#E69F00", "P2": "#0072B2"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (cond, _) in zip(axs, CONDS):
        for lev in LEVS:
            for s in SEEDS:
                th = recs[f"{cond}_{lev}_s{s}"]["trust_theta2"]
                ax.plot(np.array(th) / th[0], color=cols[lev], lw=0.7, alpha=0.7, label=lev if s == 0 else None)
        ax.set_yscale("log"); ax.set_xlabel("trust-exact iteration"); ax.set_title(f"B-spline, {cond}")
    axs[0].set_ylabel("||theta||^2 / early"); axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "bspline_theta_trajectories.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for cond, mk in (("independent", "o"), ("correlated", "s")):
        for lev in LEVS:
            rr = [recs[f"{cond}_{lev}_s{s}"] for s in SEEDS]
            ax.scatter([r["cancel"]["polished"]["cross"] for r in rr], [r["rmse_off"]["polished"] for r in rr], marker=mk, color=cols[lev], s=18,
                       label=f"{cond} {lev}")
    ax.set_xlabel("cross-edge cancellation, layer 1 (polished)"); ax.set_ylabel("off-manifold RMSE vs teacher (polished)"); ax.set_yscale("log")
    ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / "bspline_offmanifold_vs_cancellation.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step0", action="store_true")
    ap.add_argument("--step1", action="store_true")
    ap.add_argument("--mu", action="store_true")
    ap.add_argument("--recert", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--worker")
    ap.add_argument("--nopath", action="store_true")
    ap.add_argument("--child")
    a = ap.parse_args()
    if a.child:
        child(json.loads(a.child))
    elif a.worker:
        worker(a.worker, with_path=not a.nopath)
    elif a.step0:
        step0()
    elif a.step1:
        step1()
    elif a.mu:
        mu_calibration()
    elif a.recert:
        recert()
    elif a.validate:
        validate()

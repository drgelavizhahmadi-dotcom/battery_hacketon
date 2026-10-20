"""paper2/revision/infimum.py: revision step 3, infimum test (certified L2 regularisation path). See infimum_prereg.md (179814f).
Run from the repo root with the pinned interpreter, one phase at a time, appending to the log:
  /usr/bin/python3 paper2/revision/infimum.py --phase 0 >> paper2/revision/infimum_output.txt   (POST-HOC flat-direction search)
  /usr/bin/python3 paper2/revision/infimum.py --phase 1 >> paper2/revision/infimum_output.txt   (synthetic paths)
  /usr/bin/python3 paper2/revision/infimum.py --phase 2 >> paper2/revision/infimum_output.txt   (real paths)
  /usr/bin/python3 paper2/revision/infimum.py --coldstart >> paper2/revision/infimum_output.txt (POST-HOC cold-start basin check)
  /usr/bin/python3 paper2/revision/infimum.py --arms    >> paper2/revision/infimum_output.txt   (secondary arms a, c; after phase 2)
  /usr/bin/python3 paper2/revision/infimum.py --control >> paper2/revision/infimum_output.txt   (POST-HOC real-data control arm)
Inputs SHA-verified against weights_manifest.txt and held in memory. Every path point runs in its own child process (hard-killed
at the cap + 5 min; crash record 2 of the degeneracy follow-up) and is cached in gauge_weights/inf_* (SHA-indexed), so an
interruption resumes. No weights in git."""
import argparse
import fcntl
import hashlib
import io
import json
import os
import signal
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
import torch.nn.functional as F
from scipy.optimize import minimize

import degeneracy as DG
import freeze as FZ
import synth as SY
import sc_run as SR
from certify import certify_point
from kan import DELTA_IN, KAN, subset
from stage2 import extract_verified, in_memory

torch.set_num_threads(int(os.environ.get("INF_THREADS", "2")))

GAMMAS = [10 ** (-2 - k / 2) for k in range(11)]
MU_GRID = [1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
ARM_SEEDS = ["R-ind0", "R-ind1", "R-ind2"]
MAXITER, WALL, HARD = 500, 2 * 3600, 300
GRAD_TOL, GRAD_TOL_RS, REL_TOL, EIG_TOL = 1e-7, 1e-9, 1e-12, 1e-8      # stage2.run_exact / trust.py; restart x100 tighter
NEAR, SAT, DEG_NEAR, DEG_SAT, REL = 1e-8, 3.0, 3, 0.10, 1e-6
CS = [1.01, 1.1, 1.5, 2, 5, 10]
STEPS = 10 ** np.linspace(-3, 0, 13)
NFLAT = 10
BUDGET = {"L2": 12 * 3600, "Q": 8 * 3600, "Qc": 8 * 3600, "L2mu": 12 * 3600}               # real models only; synthetic: per-point caps only
SCAN_N, SCAN_MAX = 50, 100.0
MU_CTL, CTL_SEEDS = 1e-4, ["R-ind0", "R-ind1"]                          # POST-HOC real-data control arm (closed loss)
NB = SY.NB.double()
W = Path(os.environ["INF_W"]) if os.environ.get("INF_W") else SR.W_DIR    # INF_W: separate cache dir (from-scratch re-runs)
INDEX = W / "sc_index.json" if os.environ.get("INF_W") else SR.INDEX
LOGDIR = W / "inf_logs"
SYN_ARCH = "gauge_weights_2026-10-03h.tar.gz"
B_ARCH = "gauge_weights_2026-10-17k.tar.gz"


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ SHA-indexed cache, safe for parallel processes
LOCK = W / ".inf_index.lock"


class _Locked:
    def __enter__(self):
        self.f = open(LOCK, "w"); fcntl.flock(self.f, fcntl.LOCK_EX); return self

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN); self.f.close()


def _index():
    return json.loads(INDEX.read_text()) if INDEX.exists() else {}


def cached(name, make):
    """as sc_run.cached (same index file), with the index read-modify-write under a file lock and atomic writes"""
    p = W / name
    with _Locked():
        idx = _index()
    if p.exists() and name in idx:
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != idx[name]:
            raise SystemExit(f"SHA-256 MISMATCH for cached {name}")
        return torch.load(io.BytesIO(data), weights_only=False), True
    obj = make()
    buf = io.BytesIO(); torch.save(obj, buf); data = buf.getvalue()
    tmp = p.with_suffix(".tmp"); tmp.write_bytes(data); os.replace(tmp, p)
    with _Locked():
        idx = _index(); idx[name] = hashlib.sha256(data).hexdigest()
        t = INDEX.with_suffix(".tmp"); t.write_text(json.dumps(idx, indent=1)); os.replace(t, INDEX)
    return obj, False


def has_cached(name):
    with _Locked():
        return (W / name).exists() and name in _index()


def npx(a):
    return a.numpy() if torch.is_tensor(a) else np.asarray(a)


# ------------------------------------------------------------------ inputs and data (per process, in memory)
_BLOBS, _DATA = {}, {}


def blobs(names):
    need = [n for n in names if n not in _BLOBS]
    if need:
        _BLOBS.update(in_memory(extract_verified(need)))
    return {n: _BLOBS[n] for n in names}


def ld(name):
    return torch.load(io.BytesIO(blobs([name])[name]), weights_only=False)


def syn_data():
    if "syn" not in _DATA:
        res = json.loads((REPO / "paper2/synth_results.json").read_text())
        teacher = SY.build_teacher(res["teacher_scale"])
        ci = list(SY.CONDS).index("correlated")
        X = SY.make_inputs(SY.CONDS["correlated"], 1000 + ci)
        y0 = SY.predict(teacher, X); sig = 0.05 * y0.std(); rng = np.random.default_rng(5000 + ci)
        y = y0 + sig * rng.standard_normal(SY.N)
        _DATA["syn"] = dict(X=X, y=y, mu=float(res["mu"]), res=res)
    return _DATA["syn"]


def real_data():
    if "real" not in _DATA:
        S = SR.setup()
        ball = SR.dbl(subset(S["B"], S["use"]))
        _DATA["real"] = dict(S=S, ball=ball, n_salt=len(S["salts"]), used=torch.unique(ball["s"]).sort().values)
    return _DATA["real"]


def input_names(model):
    cell, key = model.split("-")
    return [f"delta_s{key[3:]}.pt"] if cell == "R" else [f"synth_correlated_{cell}_s{key[1:]}.pt"]


# ------------------------------------------------------------------ flat parameter vector (free subset; real: used salt rows only)
class Flat:
    def __init__(self, net, full, free, parts, reg=None, rows=None):
        """parts(prm) -> dict of the committed loss terms (L = their sum); reg(prm, v) -> the regularisation term (scaled)"""
        self.net, self.parts, self.reg = net, parts, reg
        self.full = {k: v.detach().clone().double() for k, v in full.items()}
        self.rows = rows or {}
        self.meta, o, pieces = [], 0, []
        for n in free:
            t = self.full[n][self.rows[n]] if n in self.rows else self.full[n]
            self.meta.append((n, t.shape, o, o + t.numel())); o += t.numel(); pieces.append(t.reshape(-1))
        self.n = o
        self.x0 = torch.cat(pieces).numpy().copy()
        self.sl = {n: (a, b, shp) for n, shp, a, b in self.meta}
        self._cache = None

    def params(self, v):
        prm = dict(self.full)
        for n, shp, a, b in self.meta:
            t = v[a:b].reshape(shp)
            prm[n] = self.full[n].index_put((self.rows[n],), t) if n in self.rows else t
        return prm

    def L_t(self, v):
        return sum(self.parts(self.params(v)).values())

    def loss_t(self, v):
        prm = self.params(v)
        return sum(self.parts(prm).values()) + (self.reg(prm, v) if self.reg else 0.0)

    def fun(self, x):
        if self._cache is not None and np.array_equal(self._cache[0], x):
            return self._cache[1], self._cache[2]
        v = torch.tensor(x, requires_grad=True)
        L = self.loss_t(v)
        (g,) = torch.autograd.grad(L, v)
        self._cache = (x.copy(), float(L), g.numpy().copy())
        return self._cache[1], self._cache[2]

    def loss(self, x):
        with torch.no_grad():
            return float(self.loss_t(torch.tensor(x)))

    def Lstar(self, x):
        with torch.no_grad():
            return float(self.L_t(torch.tensor(x)))

    def terms(self, x):
        with torch.no_grad():
            v = torch.tensor(x); prm = self.params(v)
            d = {k: float(t) for k, t in self.parts(prm).items()}
            if self.reg:
                d["reg"] = float(self.reg(prm, v))
            return d

    def hessian(self, x):
        H = torch.autograd.functional.hessian(self.loss_t, torch.tensor(x)).numpy()
        return (H + H.T) / 2

    def prm_at(self, x):
        with torch.no_grad():
            return {k: t.detach().clone() for k, t in self.params(torch.tensor(x)).items()}


def syn_parts(net, X, y, lev, mu):
    """the committed cell loss (synth.cell_penalty): data + D2 (+ SiLU for P1/P2) (+ mu*Q for P2)"""
    Xt, yt = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)

    def parts(prm):
        pred = torch.func.functional_call(net, prm, (Xt,))[:, 0]
        d2, silu, q = SY.penalty_terms(prm)
        out = dict(data=F.mse_loss(pred, yt), D2=d2)
        if lev in ("P1", "P2"):
            out["SiLU"] = silu
        if lev == "P2":
            out["muQ"] = mu * q
        return out
    return parts


def q_close(prm):
    """close.py's Q: RBF null-space part of every edge + squared SiLU base weights (biases, embedding unpenalised)"""
    return (sum(((prm[c] @ NB) ** 2).sum() for c in ("layers.0.coef", "layers.1.coef"))
            + (prm["layers.0.base.weight"] ** 2).sum() + (prm["layers.1.base.weight"] ** 2).sum())


def make_reg(kind, coef, flat_sl):
    if kind in ("L2", "L2mu"):
        return lambda prm, v: coef * (v ** 2).sum()
    if kind == "Q":
        return lambda prm, v: coef * q_close(prm)
    if kind == "Qc":   # coercive Q: + biases + the used salt-embedding rows (the free part of emb.weight)
        def reg(prm, v):
            extra = (prm["layers.0.base.bias"] ** 2).sum() + (prm["layers.1.base.bias"] ** 2).sum()
            if "emb.weight" in flat_sl:
                a, b, _ = flat_sl["emb.weight"]; extra = extra + (v[a:b] ** 2).sum()
            return coef * (q_close(prm) + extra)
        return reg
    raise ValueError(kind)


def build(model, kind=None, coef=None):
    """-> flat (with the regulariser kind*coef, or none), net, zin(prm)"""
    cell, key = model.split("-")
    if cell == "R":
        rd = real_data()
        net = KAN(len(DELTA_IN), [6, 1], rd["n_salt"])
        miss, unexp = net.load_state_dict({n: t.float() for n, t in ld(f"delta_s{key[3:]}.pt").items()}, strict=False)
        assert not unexp and all(x.endswith(".grid") for x in miss)
        net = net.double().eval()
        full = dict(net.named_parameters())
        parts = FZ.real_parts(net, rd["ball"])
        rows = {"emb.weight": rd["used"]}
        ball = rd["ball"]
        zin = lambda prm: torch.cat([ball["x"], torch.tanh(prm["emb.weight"][ball["s"]])], 1)
    else:
        sd = syn_data()
        net = SY.SynthKAN().double()
        net.load_state_dict({k: v.double() for k, v in ld(f"synth_correlated_{cell}_s{key[1:]}.pt")["early"].items()})
        net.eval()
        full = dict(net.named_parameters())
        parts = syn_parts(net, sd["X"], sd["y"], cell, sd["mu"])
        rows = {}
        Xd = torch.tensor(sd["X"], dtype=torch.float64)
        zin = lambda prm: Xd
    if kind == "L2mu":                         # control arm: the closed loss final.py + 1e-4 Q is the loss L of this path
        open_parts = parts
        parts = lambda prm: dict(open_parts(prm), muQ=MU_CTL * q_close(prm))
    flat = Flat(net, full, list(full), parts, None, rows)
    if kind is not None:
        flat.reg = make_reg(kind, coef, flat.sl)
    return flat, net, zin


# ------------------------------------------------------------------ one polish (child process), cumulative caps, checkpoints
class WallClock(Exception):
    pass


def polish(flat, x_start, ck, gtol, elapsed0):
    """trust-exact as freeze.run (stage2 stopping rule, gtol 1e-30 in scipy). The 2 h cap and the 500-iteration cap are
    cumulative across checkpoint resumes: a resumed run continues from the checkpoint with the remaining budget."""
    st = dict(x=np.asarray(x_start, float).copy(), iters=0, losses=[], grads=[])
    if ck.exists():
        z = np.load(ck)
        st.update(x=z["x"].copy(), iters=int(z["iters"]), losses=list(z["losses"]), grads=list(z["grads"]))
        elapsed0 = max(elapsed0, float(z["elapsed"]))
    out = dict(stop=None, resumed=bool(ck.exists()))
    if elapsed0 >= WALL:
        out["stop"] = "wall-clock cap (2 h, cumulative)"
    if st["iters"] >= MAXITER:
        out["stop"] = "iteration cap (500)"
    if not st["losses"]:
        L0, g0 = flat.fun(st["x"]); st["losses"], st["grads"] = [L0], [float(np.abs(g0).max())]
    t0 = time.monotonic()
    el = lambda: elapsed0 + time.monotonic() - t0

    def save():
        tmp = ck.with_name(ck.stem + ".tmp.npz")
        np.savez(tmp, x=st["x"], iters=st["iters"], losses=np.array(st["losses"][-11:]), grads=np.array(st["grads"][-11:]), elapsed=el())
        os.replace(tmp, ck)

    def callback(intermediate_result):
        x = intermediate_result.x
        L, g = flat.fun(x)
        st["x"] = x.copy(); st["iters"] += 1; st["losses"].append(L); st["grads"].append(float(np.abs(g).max()))
        save()
        ls = st["losses"]
        if len(ls) >= 11 and st["grads"][-1] < gtol and abs(ls[-11] - L) / max(abs(ls[-11]), 1e-300) < REL_TOL:
            ev = np.linalg.eigvalsh(flat.hessian(x))
            if ev[0] >= -EIG_TOL * ev[-1]:
                out["stop"] = "converged (stage2 criterion)"
                raise StopIteration
        if el() > WALL:
            out["stop"] = "wall-clock cap (2 h, cumulative)"
            raise StopIteration

    msg = ""
    if out["stop"] is None:
        def on_alarm(signum, frame):
            raise WallClock()
        old = signal.signal(signal.SIGALRM, on_alarm)
        signal.setitimer(signal.ITIMER_REAL, max(WALL - elapsed0, 1) + 60)
        try:
            res = minimize(flat.fun, st["x"], jac=True, hess=flat.hessian, method="trust-exact", callback=callback,
                           options=dict(maxiter=MAXITER - st["iters"], gtol=1e-30))
            msg = str(res.message)
        except WallClock:
            out["stop"] = "wall-clock cap (2 h, cumulative; timer inside the trust-exact subproblem)"
        except (ValueError, np.linalg.LinAlgError) as e:
            # scipy's trust-exact subproblem can produce a non-finite Cholesky factor (dose-response crash record 2); the run stops
            # at the last accepted iterate, which is then certified like any other stop
            out["stop"] = f"solver error (trust-exact subproblem: {str(e)[:60]})"
            msg = str(e)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)
        if out["stop"] is None:
            out["stop"] = "iteration cap (500)" if st["iters"] >= MAXITER else "scipy termination"
    save()
    cert, _, _ = certify_point(flat, st["x"])
    return st["x"], dict(stop=out["stop"], iters=st["iters"], wall_s=round(el(), 1), message=msg, cert=cert, resumed=out["resumed"])


def child_point(spec):
    """child process: spec = {model, kind, coef, start (npy path), name (cache), ck (checkpoint), gtol, elapsed0}"""
    flat, _, _ = build(spec["model"], spec["kind"], spec["coef"])
    x0 = np.load(spec["start"])
    cached(spec["name"], lambda: dict(zip(("x", "info"), (lambda r: (torch.tensor(r[0]), r[1]))(
        polish(flat, x0, Path(spec["ck"]), spec["gtol"], spec["elapsed0"])))))


def run_point(spec, flat, wlog):
    """worker side: cached result, else run the child (heartbeat file, hard kill at the cumulative cap + 5 min)."""
    name = spec["name"]
    if not has_cached(name):
        ck, hb = Path(spec["ck"]), W / (Path(spec["ck"]).stem + ".hb.json")
        e0 = 0.0
        if hb.exists():
            e0 = float(json.loads(hb.read_text())["elapsed"])
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
                if el > WALL + HARD:
                    pr.kill(); pr.wait(); killed = True
        if not has_cached(name):
            if not killed:
                raise SystemExit(f"child for {name} exited with code {pr.returncode} without a result (see {LOGDIR})")
            z = np.load(ck) if ck.exists() else None
            x = z["x"] if z is not None else np.load(spec["start"])
            cert, _, _ = certify_point(flat, x)
            cached(name, lambda: dict(x=torch.tensor(x), info=dict(stop="wall-clock cap (2 h, cumulative; hard kill, last checkpointed iterate)",
                                                                    iters=int(z["iters"]) if z is not None else 0, wall_s=round(WALL + HARD, 1),
                                                                    message="killed", cert=cert, resumed=False)))
            wlog(f"    {name}: child hard-killed at the cap + 5 min; last checkpointed iterate certified separately")
    obj, _ = cached(name, lambda: (_ for _ in ()).throw(RuntimeError(name)))
    return npx(obj["x"]), obj["info"]


# ------------------------------------------------------------------ diagnostics and probes at a point
def block_norms(flat, x):
    prm = flat.prm_at(x)
    out = {}
    for li in (0, 1):
        c = prm[f"layers.{li}.coef"]; aff = float(((c @ NB) ** 2).sum()); tot = float((c ** 2).sum())
        out[f"L{li + 1} RBF affine"], out[f"L{li + 1} RBF curved"] = aff, tot - aff
        out[f"L{li + 1} SiLU"] = float((prm[f"layers.{li}.base.weight"] ** 2).sum())
        out[f"L{li + 1} bias"] = float((prm[f"layers.{li}.base.bias"] ** 2).sum())
    if "emb.weight" in flat.sl:
        a, b, _ = flat.sl["emb.weight"]; out["salt emb"] = float((x[a:b] ** 2).sum())
    return out


def affine_scaled(flat, x, c):
    a, b, shp = flat.sl["layers.0.coef"]
    y = x.copy(); cf = torch.tensor(x[a:b]).reshape(shp)
    y[a:b] = (cf + (c - 1) * ((cf @ NB) @ NB.T)).reshape(-1).numpy()
    return y


def l1_scaled(flat, x, c):
    y = x.copy()
    for n in FZ.L1:
        a, b, _ = flat.sl[n]; y[a:b] = c * x[a:b]
    return y


def probes(flat, x, ev, V, restart_fn):
    """restart, scaling and flat-direction probes on the (regularised) loss at this point; classification as fixed in the prereg"""
    tp = time.monotonic()
    L0, th0 = flat.loss(x), float((x ** 2).sum())
    rel = lambda L: (L - L0) / abs(L0)
    found = []

    def test(tag, y):
        th = float((y ** 2).sum()); r = rel(flat.loss(y))
        if th > th0 and r < -REL:
            found.append(dict(probe=tag, rel_loss=r, rel_theta=(th - th0) / th0))
        return r
    sc = {}
    for kind, fn in (("affine", affine_scaled), ("all_L1", l1_scaled)):
        sc[kind] = [test(f"scale {kind} c={c}", fn(flat, x, c)) for c in CS]
    order = np.argsort(np.abs(ev))[:NFLAT]
    nx = float(np.sqrt(th0))
    fs = []
    for j in order:
        for sg in (1, -1):
            fs.append(dict(eig=float(ev[j]), sign=sg, rel=[test(f"flat eig {ev[j]:+.1e} sign {sg:+d} s={s:.3g}", x + sg * s * nx * V[:, j]) for s in STEPS]))
    tr = time.monotonic()
    xs, info = restart_fn(x)
    tr = time.monotonic() - tr
    L1, th1 = flat.loss(xs), float((xs ** 2).sum())
    rs = dict(stop=info["stop"], iters=info["iters"], wall_s=info["wall_s"], rel_loss=rel(L1), rel_theta=(th1 - th0) / th0,
              certified=bool(info["cert"]["certified"]))
    rs["lowers_and_grows"] = bool(rs["rel_loss"] < -REL and rs["rel_theta"] > REL)
    capped = "cap" in rs["stop"]
    rs["stalled"] = bool(capped and abs(rs["rel_loss"]) < REL and abs(rs["rel_theta"]) < REL)
    rs["cap_type"] = ("iteration" if "iteration" in rs["stop"] else "wall-clock") if capped else None
    cls = "PLATEAU" if (rs["lowers_and_grows"] or found) else "FLAT MINIMUM"
    best = min(found, key=lambda d: d["rel_loss"]) if found else None
    return dict(cls=cls, restart=rs, scaling=sc, flat_search=fs, n_descents=len(found), best_descent=best,
                min_rel_scaling=float(min(min(v) for v in sc.values())), min_rel_flat=float(min(min(d["rel"]) for d in fs)),
                compute_s=time.monotonic() - tp - tr)


def saturation(flat, net, zin, x):
    prm = flat.prm_at(x)
    with torch.no_grad():
        s = DG.hidden(prm, net, zin(prm)).numpy()
    return float((np.abs(s) > SAT).mean())


def uncert_reason(info):
    s = info["stop"]
    if "iteration cap" in s:
        return "iteration cap"
    if "wall-clock" in s:
        return "wall-clock cap"
    if "solver error" in s:
        return "solver error (trust-exact subproblem)"
    names = dict(a="gradient criterion (a)", b="negative curvature (b)", c="Newton decrement (c)", d="null-space gradient (d)")
    return ", ".join(names[k] for k, ok in info["cert"]["crit"].items() if not ok)


# ------------------------------------------------------------------ one model's path (worker process)
def analyse(flat, net, zin, xk, info, spec, stem, wlog):
    """certified / degenerate / probes record of one polished point (the primary-path protocol); stem = cache-name stem"""
    rec = dict(status="reached", stop=info["stop"], iters=info["iters"], wall_s=info["wall_s"], certified=bool(info["cert"]["certified"]),
               crit=info["cert"]["crit"], grad=info["cert"]["grad"], resumed=info.get("resumed", False),
               theta2=float((xk ** 2).sum()), blocks=block_norms(flat, xk), Lstar=flat.Lstar(xk), terms=flat.terms(xk))
    if rec["certified"]:
        H = flat.hessian(xk); ev, V = np.linalg.eigh(H)
        near = np.abs(ev) < NEAR * ev[-1]
        rec.update(n_near_flat=int(near.sum()), eig_deciles=[float(q) for q in np.quantile(ev, np.linspace(0, 1, 11))],
                   min_max=float(ev[0] / ev[-1]), sat=saturation(flat, net, zin, xk))
        rec["degenerate"] = bool(rec["n_near_flat"] > DEG_NEAR or rec["sat"] > DEG_SAT)
        if rec["degenerate"]:
            def restart_fn(xc):
                rstart = W / f"{stem}_rs_start.npy"
                if not rstart.exists():
                    np.save(rstart, xc)
                sp = dict(spec, start=str(rstart), name=f"{stem}_rs.pt", ck=str(W / f"{stem}_rs_ck.npz"), gtol=GRAD_TOL_RS)
                return run_point(sp, flat, wlog)
            pr, _ = cached(f"{stem}_probes.pt", lambda: probes(flat, xk, ev, V, restart_fn))
            rec["probes"] = pr
            rec["class"] = pr["cls"]
            rec["included"] = pr["cls"] == "FLAT MINIMUM"
            if pr["cls"] == "PLATEAU":
                rec["reason"] = "PLATEAU by probe"
            rec["wall_s"] += pr["restart"]["wall_s"] + pr["compute_s"]
        else:
            rec["class"] = None; rec["included"] = True
    else:
        rec.update(degenerate=None, included=False, reason=uncert_reason(info)); rec["class"] = None
    return rec


def path_name(kind, model, k, suffix=""):
    return f"inf_{kind}_{model}_g{k}{suffix}"


def worker(kind, model):
    """runs one regularisation path sequentially; writes gauge_weights/inf_rec_<kind>_<model>.json"""
    lines = []
    wlog = lambda s: lines.append(s)
    grid = GAMMAS if kind in ("L2", "L2mu") else MU_GRID
    real = model.startswith("R-")
    budget = BUDGET[kind] if real else None
    flat0, net, zin = build(model, kind, 0.0) if kind == "L2mu" else build(model)
    x = flat0.x0.copy()
    used = 0.0
    recs, consec, stopped = [], 0, None
    LOGDIR.mkdir(exist_ok=True)
    for k, g in enumerate(grid):
        rec = dict(k=k, coef=g)
        if stopped:
            rec["status"] = "not reached (stopped)"; recs.append(rec); continue
        if budget is not None and used >= budget:
            rec["status"] = "not reached (budget)"; recs.append(rec); continue
        flat, _, _ = build(model, kind, g)
        start = W / f"{path_name(kind, model, k)}_start.npy"
        if not start.exists():
            np.save(start, x)
        x_in = np.load(start)
        spec = dict(model=model, kind=kind, coef=g, start=str(start), name=path_name(kind, model, k, ".pt"),
                    ck=str(W / f"{path_name(kind, model, k)}_ck.npz"), gtol=GRAD_TOL, elapsed0=0.0)
        if budget is not None and budget - used < WALL and not has_cached(spec["name"]):
            rec["status"] = "not reached (budget)"; rec["note"] = f"remaining budget {budget - used:.0f} s < the 2 h point cap"
            recs.append(rec); stopped = "budget"; continue
        xk, info = run_point(spec, flat, wlog)
        rec.update(analyse(flat, net, zin, xk, info, spec, path_name(kind, model, k), wlog))
        used += rec["wall_s"]
        if budget is not None and not rec["included"] and "wall-clock" in str(rec.get("reason", "")) and used > budget:
            rec["status"] = "not reached (budget)"; rec["note"] = "point cut by the model budget"
        recs.append(rec)
        consec = 0 if rec["included"] else consec + 1
        x = xk
        if consec >= 2:
            prev = [r for r in recs if r.get("status") == "reached"]
            stopped = "two consecutive uncertified points"
            rec["path_stop"] = dict(reason=stopped, at_k=k, at_coef=g,
                                    norm_still_growing=bool(len(prev) >= 2 and prev[-1]["theta2"] > prev[-2]["theta2"]))
    reached = [r for r in recs if r.get("status") == "reached"]
    out = dict(kind=kind, model=model, theta2_start=float((flat0.x0 ** 2).sum()), Lstar_start=flat0.Lstar(flat0.x0),
               blocks_start=block_norms(flat0, flat0.x0), n_params=flat0.n, path=recs, budget_used_s=used, worker_log=lines)
    # secondary arm (b): straight-line scan along the final drift direction (runaway models, primary path)
    if kind == "L2" and not model.startswith("P2-") and reached:
        xs = [flat0.x0] + [npx(cached(path_name(kind, model, r["k"], ".pt"), lambda: None)[0]["x"]) for r in reached]
        xe, xp = xs[-1], xs[-2]
        d = xe - xp; d = d / np.linalg.norm(d)
        ne = float(np.linalg.norm(xe)); td = float(xe @ d)
        tmax = -td + np.sqrt(td ** 2 + (SCAN_MAX ** 2 - 1) * ne ** 2)
        ts = np.logspace(np.log10(1e-3 * ne), np.log10(tmax), SCAN_N)
        L0 = flat0.Lstar(xe)
        Ls = [flat0.Lstar(xe + t * d) for t in ts]
        nr = [float(np.linalg.norm(xe + t * d) / ne) for t in ts]
        j = next((i for i, L in enumerate(Ls) if L > L0), None)
        out["scan"] = dict(L0=L0, t=ts.tolist(), L=Ls, norm_ratio=nr, turning_index=j,
                           s_star_norm_ratio=(nr[j] if j is not None else None), min_rel=float(min((L - L0) / abs(L0) for L in Ls)),
                           from_points=[reached[-2]["k"] if len(reached) >= 2 else "early", reached[-1]["k"]])
    (W / f"inf_rec_{kind}_{model}.json").write_text(json.dumps(out, indent=1, default=float))


def run_workers(kind, models, nproc):
    def one(m):
        if (W / f"inf_rec_{kind}_{m}.json").exists():
            return
        LOGDIR.mkdir(exist_ok=True)
        with open(LOGDIR / f"worker_{kind}_{m}.log", "a") as lf:
            r = subprocess.run([sys.executable, str(Path(__file__)), "--worker", kind, m], cwd=REPO, stdout=lf, stderr=lf)
        if r.returncode != 0:
            raise SystemExit(f"worker {kind} {m} failed (code {r.returncode}); see {LOGDIR}/worker_{kind}_{m}.log")
    with ThreadPoolExecutor(nproc) as ex:
        list(ex.map(one, models))
    return {m: json.loads((W / f"inf_rec_{kind}_{m}.json").read_text()) for m in models}


# ------------------------------------------------------------------ predictions
def included(rec):
    return sorted([r for r in rec["path"] if r.get("status") == "reached" and r["included"]], key=lambda r: -r["coef"])


def inversions(seq, up):
    return sum(1 for a, b in zip(seq, seq[1:]) if (b <= a if up else b >= a))


def fit_alpha(pts):
    """log theta2 = a - alpha log gamma over the smallest four included gamma values; residual bootstrap 10,000, seed 0"""
    p4 = sorted(pts, key=lambda r: r["coef"])[:4]
    lg, lt = np.log(np.array([r["coef"] for r in p4])), np.log(np.array([r["theta2"] for r in p4]))
    A = np.column_stack([np.ones(4), -lg])
    coef = np.linalg.lstsq(A, lt, rcond=None)[0]
    fit = A @ coef; res = lt - fit
    rng = np.random.default_rng(0)
    bs = np.array([np.linalg.lstsq(A, fit + rng.choice(res, 4, replace=True), rcond=None)[0][1] for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(alpha=float(coef[1]), lo=float(lo), hi=float(hi), n=4, via_probes=sum(1 for r in p4 if r.get("class") == "FLAT MINIMUM"),
                gammas=[r["coef"] for r in p4])


def evaluate(recs, runaway, control, need_run, need_ctl):
    rows = {}
    for m, rec in recs.items():
        pts = included(rec)
        th, Ls = [r["theta2"] for r in pts], [r["Lstar"] for r in pts]
        d = dict(n_included=len(pts), via_probes=sum(1 for r in pts if r.get("class") == "FLAT MINIMUM"),
                 smallest=min([r["coef"] for r in pts]) if pts else None)
        d["I1"] = len(pts) >= 3 and inversions(th, True) <= 1
        d["I4"] = len(pts) >= 3 and inversions(Ls, False) <= 1
        d["fit"] = fit_alpha(pts) if len(pts) >= 4 else None
        d["I2"] = bool(d["fit"] and d["fit"]["alpha"] > 0.2 and d["fit"]["lo"] > 0)
        p2 = sorted(pts, key=lambda r: r["coef"])[:2]
        d["ratio"] = p2[0]["theta2"] / p2[1]["theta2"] if len(p2) == 2 else None
        d["I3"] = d["ratio"] is not None and d["ratio"] <= 1.5
        d["three_decades"] = d["smallest"] is not None and d["smallest"] <= 1e-5 * (1 + 1e-9)
        d["plateau_like"] = bool(d["fit"] is not None and d["ratio"] is not None and d["ratio"] <= 1.5 and (d["fit"]["alpha"] <= 0.2 or d["fit"]["lo"] <= 0))
        rows[m] = d
    cnt = lambda ms, key: sum(1 for m in ms if rows[m][key])
    v = dict(I1=cnt(runaway, "I1") >= need_run, I2=cnt(runaway, "I2") >= need_run, I4=cnt(runaway, "I4") >= need_run,
             I3=(cnt(control, "I3") >= need_ctl) if control else None,
             counts=dict(I1=cnt(runaway, "I1"), I2=cnt(runaway, "I2"), I4=cnt(runaway, "I4"), I3=cnt(control, "I3") if control else None,
                         three_decades=cnt(runaway, "three_decades"), plateau_like=cnt(runaway, "plateau_like"), n_run=len(runaway), n_ctl=len(control)))
    return rows, v


def branch(v, i3, nrun, need_run):
    c = v["counts"]
    if nrun - c["three_decades"] > nrun / 2:
        return "INCONCLUSIVE: fewer than three decades of gamma reached (certified, incl. certified-degenerate) for most runaway models"
    if i3 is False:
        return "I3 FAILS: the test does not discriminate; report it and do not use it in the paper"
    if v["I1"] and v["I2"] and v["I4"] and i3:
        return "RECEDING MINIMUM over the tested range: the regularised minimisers diverge as gamma -> 0 (I1, I2, I4 and I3 hold)"
    if c["plateau_like"] >= need_run:
        return "FINITE BUT DISTANT MINIMUM: the runaway models' paths plateau (alpha ~ 0 and ratio <= 1.5)"
    return "PARTIAL: no pre-registered branch applies; per-model results reported"


# ------------------------------------------------------------------ reporting
def fmt(x, f=".3e"):
    return "-" if x is None else format(x, f)


def report_paths(recs, title):
    log(f"\n==================== {title}: per-model paths ====================")
    for m, rec in recs.items():
        log(f"  {m}  (start ||theta||^2 {rec['theta2_start']:.4e}, L {rec['Lstar_start']:.6e}, {rec['n_params']} parameters)")
        log("     coef      status/stop                                       it  wall_s   cert  degen  near sat    class         ||theta*||^2   L*            reason / probes")
        for r in rec["path"]:
            if r.get("status") != "reached":
                log(f"     {r['coef']:.2e}  {r['status']}" + (f" ({r['note']})" if r.get("note") else "")); continue
            pr = r.get("probes")
            pinfo = ""
            if pr:
                rs = pr["restart"]
                pinfo = (f"restart {rs['stop']} {rs['iters']} it dL {rs['rel_loss']:+.1e} dth {rs['rel_theta']:+.1e}"
                         + (f" STALLED ({rs['cap_type']} cap)" if rs["stalled"] else "") + f"; scaling min dL {pr['min_rel_scaling']:+.1e}; flat min dL {pr['min_rel_flat']:+.1e}; descents {pr['n_descents']}")
            log(f"     {r['coef']:.2e}  {r['stop'][:48]:48s} {r['iters']:4d} {r['wall_s']:7.0f}  {str(r['certified'])[:5]:5s}  "
                f"{('-' if r.get('degenerate') is None else str(r['degenerate'])[:5]):5s}  {('-' if r.get('n_near_flat') is None else r['n_near_flat']):>4} "
                f"{fmt(r.get('sat'), '.3f'):5s}  {(r.get('class') or '-'):12s}  {r['theta2']:.6e}  {r['Lstar']:.6e}  {r.get('reason', '')} {pinfo}")
            if r.get("path_stop"):
                ps = r["path_stop"]
                log(f"     PATH STOPPED at coef {ps['at_coef']:.2e}: {ps['reason']}; norm still growing at the stop: {ps['norm_still_growing']}")
        if rec.get("scan"):
            s = rec["scan"]
            log(f"     arm (b) straight-line scan from points {s['from_points']}: turning point "
                + (f"at ||theta|| x{s['s_star_norm_ratio']:.3g}" if s["turning_index"] is not None else "none up to 100x")
                + f"; largest relative decrease of L along the line {s['min_rel']:+.2e}")
        for w in rec.get("worker_log", []):
            log("   " + w)


def report_blocks(recs):
    log("\n  per-block ||theta||^2 at the start and at the last reached point (primary path):")
    for m, rec in recs.items():
        rr = [r for r in rec["path"] if r.get("status") == "reached"]
        if not rr:
            continue
        b0, b1 = rec["blocks_start"], rr[-1]["blocks"]
        log(f"    {m:8s} coef {rr[-1]['coef']:.1e}: " + "; ".join(f"{k} {b0[k]:.3g}->{b1[k]:.3g}" for k in b0))


def report_eval(rows, v, runaway, control, br, title, i3_note=""):
    log(f"\n==================== {title}: fitted alpha (unconditional) and per-model prediction checks ====================")
    log("  model     incl  via probes  smallest coef  alpha    95% CI               pts (via probes)  I1    I2    I4    ratio(2 smallest)  I3")
    for m, d in rows.items():
        f = d["fit"]
        log(f"  {m:8s}  {d['n_included']:4d}  {d['via_probes']:10d}  {fmt(d['smallest'], '.1e'):13s}  "
            + (f"{f['alpha']:+.3f}  [{f['lo']:+.3f}, {f['hi']:+.3f}]  {f['n']} ({f['via_probes']})          " if f else "-        -                    -                 ")
            + f"{str(d['I1'])[:5]:5s} {str(d['I2'])[:5]:5s} {str(d['I4'])[:5]:5s} {fmt(d['ratio'], '.3f'):17s}  {str(d['I3'])[:5] if m in control else '-'}")
    c = v["counts"]
    log(f"\n==================== {title}: VERDICTS (pre-registered I1-I4) ====================")
    log(f"  I1 {'HELD' if v['I1'] else 'FAILED':6s} {c['I1']}/{c['n_run']} runaway models: ||theta*||^2 increases as gamma decreases (one inversion allowed)")
    log(f"  I2 {'HELD' if v['I2'] else 'FAILED':6s} {c['I2']}/{c['n_run']} runaway models: alpha > 0.2 with the bootstrap 95% interval excluding 0")
    if v["I3"] is not None:
        log(f"  I3 {'HELD' if v['I3'] else 'FAILED':6s} {c['I3']}/{c['n_ctl']} control models: ratio between the two smallest certified gamma <= 1.5")
    else:
        log(f"  I3 (no control models in this phase) {i3_note}")
    log(f"  I4 {'HELD' if v['I4'] else 'FAILED':6s} {c['I4']}/{c['n_run']} runaway models: L* decreases as gamma decreases (one inversion allowed)")
    log(f"  runaway models with >= three decades reached: {c['three_decades']}/{c['n_run']}; plateau-like (ratio <= 1.5 and alpha <= 0.2 or CI incl. 0): {c['plateau_like']}/{c['n_run']}")
    log(f"  BRANCH: {br}")


def report_plateau_share(recs, grid):
    log("\n  PLATEAU share per gamma (degenerate certified points classified PLATEAU / points reached), and points entered via probes:")
    for k, g in enumerate(grid):
        rr = [rec["path"][k] for rec in recs.values() if rec["path"][k].get("status") == "reached"]
        pl = sum(1 for r in rr if r.get("class") == "PLATEAU"); fm = sum(1 for r in rr if r.get("class") == "FLAT MINIMUM")
        ce = sum(1 for r in rr if r["certified"])
        log(f"    coef {g:.2e}: reached {len(rr):2d}, certified {ce:2d}, degenerate {pl + fm:2d}, PLATEAU {pl:2d} ({(pl / len(rr) if rr else 0):.2f}), entered via probes (FLAT MINIMUM) {fm:2d}")


def figure(recs, groups, fname, title):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for grp, (ms, col) in groups.items():
        for m in ms:
            rr = [r for r in recs[m]["path"] if r.get("status") == "reached"]
            if not rr:
                continue
            g = [r["coef"] for r in rr]
            for ax, key in ((axs[0], "theta2"), (axs[1], "Lstar")):
                ax.plot(g, [r[key] for r in rr], color=col, lw=0.8, alpha=0.7, label=grp if m == ms[0] else None)
                for r in rr:
                    mk = ("x" if r.get("class") == "PLATEAU" else "s" if r.get("class") == "FLAT MINIMUM" else "o" if r["certified"] else "v")
                    if mk == "s":
                        ax.scatter([r["coef"]], [r[key]], marker=mk, s=30, facecolors="none", edgecolors=col, zorder=3)
                    else:
                        ax.scatter([r["coef"]], [r[key]], marker=mk, s=(30 if mk == "x" else 12), color=col, zorder=3)
    for ax, lab in ((axs[0], "||theta*||^2"), (axs[1], "L* (loss without the gamma term)")):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis(); ax.set_xlabel("gamma"); ax.set_ylabel(lab)
    axs[0].legend(fontsize=7)
    axs[0].set_title("o certified, s certified-degenerate (flat minimum), x plateau, v uncertified", fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(); FZ.FIG.mkdir(exist_ok=True); fig.savefig(FZ.FIG / fname, dpi=110); plt.close(fig)


def save_results(key, obj):
    p = HERE / "infimum_results.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d[key] = obj
    p.write_text(json.dumps(d, indent=1, default=float))


def header(phase):
    log("#" * 100)
    log(f"paper2/revision/infimum.py {time.strftime('%Y-%m-%d %H:%M:%S')}: {phase} (prereg infimum_prereg.md, 179814f)")
    log("#" * 100)


def choices():
    log("DISCREPANCIES / choices (files win):")
    log("  D1-D4 as recorded in the prereg (Phase 0 inputs from archive 11; no committed synthetic P2 near-flat count above 0;")
    log("        the 0.37 output-bias share is independent P0 s0; the null-space-gradient criterion added to the reasons)")
    log("  H1 theta = all loss-relevant parameters: synthetic all 139; real the 609 of trust.Flat (the unused salt-embedding rows are")
    log("     fixed, as everywhere since trust.py; they do not enter the loss and are not part of theta)")
    log("  H2 'stalled restart' is applied by its fixed definition (cap AND |relative change| < 1e-6 in BOTH loss and ||theta||^2).")
    log("     By that definition the degeneracy study's ind0/ind2/ind3 restarts are NOT stalled (their ||theta||^2 changed by")
    log("     +9.5e-4, +4.6e-6, -4.6e-4 relative); the prereg's 'as ind0, ind2 and ind3 did' does not match its own definition")
    log("  H3 relative norm changes use ||theta||^2 (as the degeneracy study); 'a direction that increases the norm' = ||y|| > ||x||")
    log("  H4 a point whose remaining model budget is below the 2 h point cap is not started ('not reached (budget)'); a point cut by")
    log("     the budget is 'not reached (budget)', never uncertified (real models only)")
    log("  H5 I1/I4 need >= 3 included points, I2 >= 4, I3 >= 2, else 'not met'; 'three decades reached' = an included point at")
    log("     gamma <= 1e-5; branch precedence: inconclusive > I3 fails > receding > finite distant > partial")
    log("  H6 arm (b): direction = difference of the last two reached path points (or early state -> first point), 50 log-spaced")
    log("     points from 1e-3 ||theta_end|| to ||theta|| = 100 ||theta_end||; L = the committed loss without the gamma term")
    log("  H7 stage2 stopping rule resumes with the checkpointed last 11 losses; per-point caps cumulative (heartbeat + checkpoint)")


# ------------------------------------------------------------------ Phase 0
def phase0():
    header("PHASE 0, POST-HOC retroactive flat-direction search on the six real Arm B endpoints of step 2")
    choices()
    names = [f"delta_s{i}.pt" for i in range(6)] + [f"freeze_B-R_ind{i}.pt" for i in range(6)]
    log("\n==================== Phase 0 inputs (SHA-256 verified, held in memory) ====================")
    blobs(names)
    deg = {r["key"]: r for r in json.loads((HERE / "degeneracy_results.json").read_text())["rows"] if r["arm"] == "B" and r["cell"] == "R"}
    rd = real_data()
    out = []
    for i in range(6):
        key = f"ind{i}"
        flat0, net, _ = build(f"R-{key}")
        full = dict(net.named_parameters())
        st = ld(f"freeze_B-R_{key}.pt"); xe = npx(st["x"])
        fb = FZ.FlatFree(net, full, FZ.L1, FZ.real_parts(net, rd["ball"]))
        L0 = float(sum(fb.terms(xe).values())); n0 = float(np.linalg.norm(xe))
        ev, V = np.linalg.eigh(fb.hessian(xe))
        order = np.argsort(np.abs(ev))[:NFLAT]
        found, dirs = [], []
        for j in order:
            for sg in (1, -1):
                rels = []
                for s in STEPS:
                    y = xe + sg * s * n0 * V[:, j]
                    r = (float(sum(fb.terms(y).values())) - L0) / abs(L0); rels.append(r)
                    if np.linalg.norm(y) > n0 and r < -REL:
                        found.append(dict(eig=float(ev[j]), sign=sg, s=float(s), rel_loss=r))
                dirs.append(dict(eig=float(ev[j]), sign=sg, rel=rels))
        d = deg[key]; rs = d["restart"]
        base = float(d["scaling"]["affine"][0])
        sc_desc = [(k, c, (v - base) / abs(base)) for k in ("affine", "all") for c, v in zip(CS, d["scaling"][k][1:]) if (v - base) / abs(base) < -REL]
        lowers_grows = bool(rs["rel_loss"] < -REL and rs["rel_theta"] > REL)
        capped = "cap" in rs["stop"]
        stalled = bool(capped and abs(rs["rel_loss"]) < REL and abs(rs["rel_theta"]) < REL)
        cls = "PLATEAU" if (lowers_grows or sc_desc or found) else "FLAT MINIMUM"
        r = dict(key=key, step2_certified=d["certified"], old_class=d["class"], new_class=cls, restart_lowers_and_grows=lowers_grows,
                 restart_stalled=stalled, restart_cap=("iteration" if "iteration" in rs["stop"] else "wall-clock") if capped else None,
                 scaling_descents=sc_desc, flat_descents=found, n_flat_descents=len(found), min_rel_flat=float(min(min(x["rel"]) for x in dirs)),
                 flattest_eigs=[float(ev[j]) for j in order], n_free=fb.n, dirs=dirs)
        out.append(r)
        log(f"  B-R {key}: step-2 certified {d['certified']}; degeneracy class {d['class']} -> UPDATED CLASS {cls} | flat-direction search "
            f"({len(order)} eigenvectors, |lambda| {abs(ev[order[0]]):.1e}-{abs(ev[order[-1]]):.1e}, both signs, 13 steps): {len(found)} descents along "
            f"norm-increasing directions (largest relative loss change {r['min_rel_flat']:+.2e}); scaling descents {len(sc_desc)}; "
            f"restart lowers+grows {lowers_grows}; restart stalled {stalled}" + (f" ({r['restart_cap']} cap)" if capped else ""))
        if found:
            b = min(found, key=lambda z: z["rel_loss"])
            log(f"      strongest descent: eig {b['eig']:+.2e}, sign {b['sign']:+d}, s = {b['s']:.3g} x ||theta||, relative loss change {b['rel_loss']:+.2e}")
    save_results("phase0", out)
    stamp = time.strftime("%Y-%m-%d")
    with open(HERE / "degeneracy_output.txt", "a") as f:
        f.write(f"\n==================== ADDENDUM {stamp} (POST-HOC; infimum test Phase 0, prereg infimum_prereg.md 179814f) ====================\n")
        f.write("Retroactive flat-direction search on the six real Arm B endpoints (archive 11): the up to 10 flattest Hessian eigenvectors\n"
                "(smallest |lambda|, layer-1 space of the Arm B loss), both signs, 13 log-spaced steps 1e-3..1 x ||theta_free||; a descent = relative\n"
                "loss decrease > 1e-6 at a norm-increasing step. Classes updated by the infimum prereg's rule (restart lowers the loss AND raises\n"
                "the norm, or any probe finds a descent along a norm-increasing direction -> PLATEAU). The text above is unchanged.\n")
        for r in out:
            f.write(f"  {r['key']}: {r['old_class']} -> {r['new_class']} (flat-direction descents {r['n_flat_descents']}, largest relative loss change "
                    f"{r['min_rel_flat']:+.2e}; scaling descents {len(r['scaling_descents'])}; restart lowers+grows {r['restart_lowers_and_grows']}; "
                    f"restart stalled {r['restart_stalled']})\n")
        f.write("  Stalled-restart flag by its fixed definition (cap AND |relative change| < 1e-6 in both loss and ||theta||^2); see infimum_output.txt H2.\n")
    log(f"\n  appended the Phase 0 addendum ({stamp}) to paper2/revision/degeneracy_output.txt")



# ------------------------------------------------------------------ path diversity (Phase 2 report; also used by the cold-start arm)
def predict_at(model, flat, net, x):
    prm = flat.prm_at(x)
    with torch.no_grad():
        if model.startswith("R-"):
            b = real_data()["ball"]
            return torch.func.functional_call(net, prm, (b["x"], b["s"]))[:, 0].numpy()
        return torch.func.functional_call(net, prm, (torch.tensor(syn_data()["X"], dtype=torch.float64),))[:, 0].numpy()


def path_x(kind, model, k):
    return npx(cached(path_name(kind, model, k, ".pt"), lambda: (_ for _ in ()).throw(RuntimeError("missing")))[0]["x"])


def report_diversity(recs, kind="L2"):
    """do the per-model paths stay distinct or collapse? theta2 spread, raw pairwise distance, prediction difference"""
    import itertools
    ms = list(recs)
    flat, net, _ = build(ms[0])
    log("\n  path diversity across models (points reached by all models; raw parameter distance, no symmetry alignment):")
    log("    coef      theta2 spread (max-min)/median   pairwise ||x_i - x_j|| / ||x_i|| median (max)   prediction RMS diff / std median (max)")
    out = []
    for k, g in enumerate(GAMMAS if kind == "L2" else MU_GRID):
        if not all(recs[m]["path"][k].get("status") == "reached" for m in ms):
            continue
        xs = [path_x(kind, m, k) for m in ms]
        th = [float((x ** 2).sum()) for x in xs]
        D = [np.linalg.norm(a - b) / np.linalg.norm(a) for a, b in itertools.combinations(xs, 2)]
        pr = [predict_at(ms[0], flat, net, x) for x in xs]
        P = [np.sqrt(np.mean((a - b) ** 2)) / a.std() for a, b in itertools.combinations(pr, 2)]
        d = dict(coef=g, spread=(max(th) - min(th)) / float(np.median(th)), dist_med=float(np.median(D)), dist_max=float(max(D)),
                 pred_med=float(np.median(P)), pred_max=float(max(P)))
        out.append(d)
        log(f"    {g:.2e}  {d['spread']:.4f}                             {d['dist_med']:.2e} ({d['dist_max']:.2e})                      "
            f"{d['pred_med']:.2e} ({d['pred_max']:.2e})")
    return out


# ------------------------------------------------------------------ POST-HOC robustness arm: cold-start basin check
COLD = {"P0": [2, 6, 10], "P1": [2, 6, 10], "P2": [10]}          # indices into GAMMAS: 1e-3, 1e-5, 1e-7
JIT_SCALE, JIT_SEED = 1e-6, 20261018


def symmetric_units(flat, net, zin, x):
    """all hidden units identical (pre-activations equal on every training row, relative 1e-6)"""
    prm = flat.prm_at(x)
    with torch.no_grad():
        s = DG.hidden(prm, net, zin(prm)).numpy()
    scale = max(float(np.abs(s).max()), 1e-300)
    return bool(max(float(np.abs(s[:, i] - s[:, j]).max()) for i in range(s.shape[1]) for j in range(i + 1, s.shape[1])) < 1e-6 * scale)


def cold_one(cell, k, start, wlog):
    g = GAMMAS[k]
    model = f"{cell}-s0"                  # the cell loss does not depend on the seed; only the start vector differs
    flat, net, zin = build(model, "L2", g)
    stem = f"inf_cold_{cell}_{start}_g{k}"
    sfile = W / f"{stem}_start.npy"
    if not sfile.exists():
        if start == "zero":
            x0 = np.zeros(flat.n)
        elif start == "jitter":
            x0 = np.random.default_rng(JIT_SEED).normal(0.0, JIT_SCALE, flat.n)
        else:
            x0 = build(f"{cell}-{start}")[0].x0
        np.save(sfile, x0)
    x0 = np.load(sfile)
    spec = dict(model=model, kind="L2", coef=g, start=str(sfile), name=f"{stem}.pt", ck=str(W / f"{stem}_ck.npz"), gtol=GRAD_TOL, elapsed0=0.0)
    xk, info = run_point(spec, flat, wlog)
    rec = dict(cell=cell, k=k, coef=g, start=start, **analyse(flat, net, zin, xk, info, spec, stem, wlog))
    warm = {s: path_x("L2", f"{cell}-s{s}", k) for s in range(8)}
    rd = {s: float(np.linalg.norm(xk - w) / np.linalg.norm(w)) for s, w in warm.items()}
    pk = predict_at(model, flat, net, xk)
    pdist = {s: float(np.sqrt(np.mean((pk - predict_at(model, flat, net, w)) ** 2)) / predict_at(model, flat, net, w).std()) for s, w in warm.items()}
    th_w = [float((w ** 2).sum()) for w in warm.values()]
    rec.update(dist_own=(rd[int(start[1:])] if start.startswith("s") else None), dist_min_warm=min(rd.values()),
               pred_diff_min_warm=min(pdist.values()), pred_diff_own=(pdist[int(start[1:])] if start.startswith("s") else None),
               theta2_warm_min=min(th_w), theta2_warm_max=max(th_w), theta2_warm_median=float(np.median(th_w)),
               Lstar_warm_median=float(np.median([flat.Lstar(w) for w in warm.values()])),
               moved=float(np.linalg.norm(xk - x0)), symmetric=symmetric_units(flat, net, zin, xk))
    return rec


def basin(r):
    """descriptive grouping (POST-HOC, fixed before reading the cold-start results)"""
    if not r["certified"]:
        return "uncertified"
    if r["theta2"] < 0.5 * r["theta2_warm_min"]:
        return "small-norm (theta2 < 0.5 x warm-path minimum)"
    if r["pred_diff_min_warm"] < 0.02 and 0.9 * r["theta2_warm_min"] <= r["theta2"] <= 1.1 * r["theta2_warm_max"]:
        return "warm-path branch (prediction within 2% of the output std of a warm-path point, theta2 within the warm range +-10%)"
    return "other"


def coldstart():
    header("POST-HOC ROBUSTNESS ARM (amendment 2026-10-18): cold-start basin check, synthetic P0/P1 (+ P2 at 1e-7)")
    log("  POST-HOC: designed after the Phase 1 results; reported separately; cannot change any pre-registered verdict.")
    log("  C1 starts: the 8 committed early-stopped weights + the all-zeros vector; each (start, gamma) polished independently")
    log("     (no warm-start chain), primary-path protocol (trust-exact, caps, certify_point, degenerate check and probes)")
    log("  C2 jittered start (only where the zero start does not move or ends with all hidden units identical): all-zeros + iid")
    log(f"     N(0, {JIT_SCALE:g}^2) noise, numpy default_rng({JIT_SEED}), the same vector for every (cell, gamma)")
    log("  C3 distances: ||x - x_warm|| / ||x_warm|| (raw, no symmetry alignment) to the warm-started path points at that gamma;")
    log("     prediction RMS difference / std on the training inputs; basin groups as defined in basin() (descriptive)")
    blobs(sorted({f"synth_correlated_{c}_s{s}.pt" for c in COLD for s in range(8)}))
    lines = []
    jobs = [(c, k, st) for c, ks in COLD.items() for k in ks for st in [f"s{s}" for s in range(8)] + ["zero"]]
    with ThreadPoolExecutor(6) as ex:
        recs = list(ex.map(lambda j: cold_one(*j, lines.append), jobs))
    jit = [(r["cell"], r["k"], "jitter") for r in recs if r["start"] == "zero" and (r["moved"] < 1e-8 or r["symmetric"])]
    with ThreadPoolExecutor(6) as ex:
        recs += list(ex.map(lambda j: cold_one(*j, lines.append), jit))
    for r in recs:
        r["basin"] = basin(r)
    zero_art = {}
    for c, k, _ in jit:
        z = next(r for r in recs if r["cell"] == c and r["k"] == k and r["start"] == "zero")
        j = next(r for r in recs if r["cell"] == c and r["k"] == k and r["start"] == "jitter")
        esc = j["basin"].startswith("warm-path")
        zero_art[f"{c}_g{k}"] = esc
        if esc:
            z["basin"] = "symmetric-manifold artefact of the exactly-zero start"
    log("\n  cell gamma     start   cert   degen class         stop                           it   ||theta*||^2  warm range           L*            "
        "dist own  dist min  pred min  moved     symm   basin")
    for r in recs:
        log(f"  {r['cell']}   {r['coef']:.1e}  {r['start']:6s}  {str(r['certified'])[:5]:5s}  {('-' if r['degenerate'] is None else str(r['degenerate'])[:5]):5s} "
            f"{(r['class'] or '-'):12s}  {r['stop'][:30]:30s} {r['iters']:4d}  {r['theta2']:.5e}  {r['theta2_warm_min']:.3e}-{r['theta2_warm_max']:.3e}  {r['Lstar']:.6e}  "
            f"{fmt(r['dist_own'], '.2e'):8s}  {r['dist_min_warm']:.2e}  {r['pred_diff_min_warm']:.2e}  {r['moved']:.2e}  {str(r['symmetric'])[:5]:5s}  {r['basin']}"
            + (f"  [{r['reason']}]" if r.get("reason") else ""))
    log("\n  endpoints grouped by basin:")
    for c, ks in COLD.items():
        for k in ks:
            rr = [r for r in recs if r["cell"] == c and r["k"] == k]
            grp = {}
            for r in rr:
                grp.setdefault(r["basin"], []).append(r["start"])
            log(f"    {c} gamma {GAMMAS[k]:.1e}: " + "; ".join(f"{b}: {', '.join(v)}" for b, v in grp.items()))
    for l in lines:
        log("   " + l)
    save_results("coldstart_posthoc", dict(rows=recs, jitter_escape=zero_art))
    log("\nwrote infimum_results.json (coldstart_posthoc)")



# ------------------------------------------------------------------ POST-HOC real-data control arm (closed loss, mu = 1e-4 fixed)
def control():
    header("POST-HOC REAL-DATA CONTROL ARM (amendment 2026-10-19): closed loss final.py + 1e-4 Q, gamma path 1e-2..1e-7, ind0/ind1")
    log("  POST-HOC: designed after the Phase 2 results; outcomes fixed in the prereg before running; changes no pre-registered verdict.")
    log("  K1 L* = the closed loss (final.py loss + 1e-4 * close.py Q) without the gamma term; the bare data term is reported beside it")
    log("  K2 same polishing, caps, certify_point, degenerate check and probes as the primary path; 12 h budget per seed")
    blobs(sorted({n for m in CTL_SEEDS for n in input_names(m)}))
    recs = run_workers("L2mu", CTL_SEEDS, 2)
    report_paths(recs, "CONTROL ARM (closed loss)")
    report_blocks(recs)
    p2 = json.loads((HERE / "infimum_results.json").read_text())["phase2"]["models"]
    log("\n  closed (mu = 1e-4) vs open (Phase 2 primary) paths, same seeds:")
    log("    model   coef      closed: ||theta*||^2  L*(closed)    data          cert  | open: ||theta*||^2  L*(open)      data          cert")
    for m in CTL_SEEDS:
        for rc, ro in zip(recs[m]["path"], p2[m]["path"]):
            c = (f"{rc['theta2']:.4e}  {rc['Lstar']:.6e}  {rc['terms']['data']:.6e}  {str(rc['certified'])[:5]:5s}" if rc.get("status") == "reached" else f"{rc['status']:60s}")
            o = (f"{ro['theta2']:.4e}  {ro['Lstar']:.6e}  {ro['terms']['data']:.6e}  {str(ro['certified'])[:5]:5s}" if ro.get("status") == "reached" else ro["status"])
            log(f"    {m}  {rc['coef']:.2e}  {c} | {o}")
    rows, _ = evaluate(recs, CTL_SEEDS, [], 2, None)
    log("\n  fixed outcome rule (prereg amendment 2026-10-19):")
    for m, d in rows.items():
        f = d["fit"]
        log(f"    {m}: included {d['n_included']} (via probes {d['via_probes']}), smallest coef {fmt(d['smallest'], '.1e')}, ratio(2 smallest) {fmt(d['ratio'], '.3f')}, "
            + (f"alpha {f['alpha']:+.3f} [{f['lo']:+.3f}, {f['hi']:+.3f}]" if f else "alpha -"))
    plateau = all(rows[m]["ratio"] is not None and rows[m]["ratio"] <= 1.5 for m in CTL_SEEDS)
    diverge = any(rows[m]["fit"] is not None and rows[m]["fit"]["alpha"] > 0.2 and rows[m]["fit"]["lo"] > 0 for m in CTL_SEEDS)
    out = ("PLATEAU (both seeds ratio <= 1.5): the real-data contrast is complete" if plateau and not diverge else
           "DIVERGENCE (alpha > 0.2, interval excluding 0, for at least one seed): mu = 1e-4 does not bound the real path over the tested range" if diverge and not plateau else
           "BOTH CONDITIONS MET (reported as is)" if plateau and diverge else "NEITHER (reported as is)")
    log(f"  CONTROL-ARM OUTCOME (POST-HOC): {out}")
    save_results("control_posthoc", dict(models=recs, rows=rows, outcome=out))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    cols = {"R-ind0": "#000000", "R-ind1": "#D55E00"}
    for m in CTL_SEEDS:
        for src, ls, lab in ((recs[m], "-", "closed (mu = 1e-4)"), (p2[m], "--", "open (Phase 2)")):
            rr = [r for r in src["path"] if r.get("status") == "reached"]
            g = [r["coef"] for r in rr]
            for ax, val in ((axs[0], lambda r: r["theta2"]), (axs[1], lambda r: r["Lstar"]), (axs[2], lambda r: r["terms"]["data"])):
                ax.plot(g, [val(r) for r in rr], color=cols[m], ls=ls, lw=0.9, label=f"{m} {lab}")
                for r in rr:
                    mk = ("x" if r.get("class") == "PLATEAU" else "s" if r.get("class") == "FLAT MINIMUM" else "o" if r["certified"] else "v")
                    if mk == "s":
                        ax.scatter([r["coef"]], [val(r)], marker=mk, s=30, facecolors="none", edgecolors=cols[m], zorder=3)
                    else:
                        ax.scatter([r["coef"]], [val(r)], marker=mk, s=(30 if mk == "x" else 12), color=cols[m], zorder=3)
    for ax, lab in ((axs[0], "||theta*||^2"), (axs[1], "L* (path loss without the gamma term)"), (axs[2], "data term (MSE)")):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis(); ax.set_xlabel("gamma"); ax.set_ylabel(lab)
    axs[0].legend(fontsize=7)
    axs[0].set_title("o certified, s certified-degenerate (flat minimum), x plateau, v uncertified", fontsize=8)
    fig.suptitle("POST-HOC real-data control: closed loss (final.py + 1e-4 Q) vs open loss, L2 path gamma = 1e-2 ... 1e-7", fontsize=9)
    fig.tight_layout(); fig.savefig(FZ.FIG / "infimum_control.png", dpi=110); plt.close(fig)
    log("\nwrote infimum_results.json (control_posthoc) and figs/infimum_control.png")


# ------------------------------------------------------------------ Phase 1 / 2 / arms
def phase(n):
    if n == 1:
        header("PHASE 1, synthetic correlated P0/P1 (runaway) and P2 (control), primary L2 path")
        run = [f"{c}-s{s}" for c in ("P0", "P1") for s in range(8)]; ctl = [f"P2-s{s}" for s in range(8)]
        models, need_run, need_ctl = run + ctl, 13, 7
    else:
        header("PHASE 2, real KAN-delta seeds, primary L2 path")
        run, ctl = [f"R-ind{i}" for i in range(6)], []
        models, need_run, need_ctl = run, 5, None
    choices()
    log("\n==================== inputs (SHA-256 verified, held in memory) ====================")
    blobs(sorted({n for m in models for n in input_names(m)}))
    if n == 1:
        sd = syn_data()
        log(f"  synthetic correlated data regenerated (seeds 1001/5001); mu (P2) = {sd['mu']:g}; check: committed early data term of "
            f"correlated_P0_s0 {sd['res']['models']['correlated_P0_s0']['early_terms']['data']:.6e} vs here "
            f"{build('P0-s0')[0].terms(build('P0-s0')[0].x0)['data']:.6e}")
    else:
        f0 = build("R-ind0")[0]
        log(f"  real data: {len(real_data()['ball']['y'])} anchored rows; free parameters {f0.n}")
    t0 = time.time()
    recs = run_workers("L2", models, 6 if n == 1 else 6)
    log(f"\n  all {len(models)} paths done ({(time.time() - t0) / 60:.1f} min this run)")
    report_paths(recs, f"PHASE {n}")
    report_blocks(recs)
    report_plateau_share(recs, GAMMAS)
    div = report_diversity(recs) if n == 2 else None
    rows, v = evaluate(recs, run, ctl, need_run, need_ctl)
    if n == 1:
        br = branch(v, v["I3"], len(run), need_run)
        report_eval(rows, v, run, ctl, br, "PHASE 1")
    else:
        p1 = json.loads((HERE / "infimum_results.json").read_text()).get("phase1", {}).get("verdicts", {})
        br = branch(v, p1.get("I3"), len(run), need_run)
        report_eval(rows, v, run, ctl, br, "PHASE 2", i3_note=f"-> Phase 1's I3 is used as the discrimination check: {p1.get('I3')}")
    save_results(f"phase{n}", dict(models=recs, rows=rows, verdicts=v, branch=br, diversity=div))
    groups = ({"P0 (runaway)": ([m for m in run if m.startswith("P0")], "#D55E00"), "P1 (runaway)": ([m for m in run if m.startswith("P1")], "#E69F00"),
               "P2 (control)": (ctl, "#0072B2")} if n == 1 else {"real KAN-delta (runaway)": (run, "#000000")})
    figure(recs, groups, f"infimum_phase{n}.png", f"Infimum test, phase {n}: certified L2 path (gamma = 1e-2 ... 1e-7)")
    log(f"\nwrote infimum_results.json (phase{n}) and figs/infimum_phase{n}.png")


def arms():
    header("SECONDARY ARMS (a) mu*Q and (c) coercive Q, real seeds ind0-ind2 (descriptive, n = 3; no verdicts)")
    choices()
    blobs(sorted({n for m in ARM_SEEDS for n in input_names(m)}))
    import close as CL
    log(f"  close.py null basis vs synth.NB: max |diff| {float((CL.NB.double() - NB).abs().max()):.1e}")
    # coercivity check for arm (c): Hessian of the committed smoothness penalty + the extended Q (unscaled), all 609 parameters
    for m in ARM_SEEDS:
        f, _, _ = build(m, "Qc", 1.0)
        f.parts = lambda prm: dict(D2=FZ.SMOOTH * sum((FZ.D2(prm[c]) ** 2).sum() for c in ("layers.0.coef", "layers.1.coef")))
        ev = np.linalg.eigvalsh(f.hessian(f.x0))
        fq, _, _ = build(m, "Q", 1.0)
        fq.parts = f.parts
        evq = np.linalg.eigvalsh(fq.hessian(fq.x0))
        log(f"  {m}: smallest eigenvalue of the penalty Hessian, smoothness + coercive Q {ev[0]:.3e} (positive definite: {bool(ev[0] > 0)}); "
            f"smoothness + close.py Q {evq[0]:.3e} ({int((evq < 1e-12).sum())} zero directions)")
    out = {}

    def one(kind_title):
        kind, title = kind_title
        return kind, title, run_workers(kind, ARM_SEEDS, 3)
    with ThreadPoolExecutor(2) as ex:          # the two arms run side by side; each has its own 8 h budget per model
        res = list(ex.map(one, (("Q", "ARM (a): R = mu * Q (close.py's Q)"), ("Qc", "ARM (c): coercive Q (+ biases + salt embedding)"))))
    for kind, title, recs in res:
        report_paths(recs, title)
        report_blocks(recs)
        out[kind] = recs
    log("\n  arms (a) and (c) are descriptive, n = 3 (ind0, ind1, ind2); no verdicts depend on them")
    save_results("arms", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int)
    ap.add_argument("--arms", action="store_true")
    ap.add_argument("--coldstart", action="store_true")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--worker", nargs=2)
    ap.add_argument("--child")
    a = ap.parse_args()
    if a.child:
        child_point(json.loads(a.child))
    elif a.worker:
        worker(*a.worker)
    elif a.arms:
        arms()
    elif a.coldstart:
        coldstart()
    elif a.control:
        control()
    elif a.phase == 0:
        phase0()
    elif a.phase in (1, 2):
        phase(a.phase)

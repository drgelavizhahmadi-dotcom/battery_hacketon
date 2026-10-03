"""paper2/synth: synthetic teacher-student test of the runaway mechanism, 2 x 3 design (see paper2/synth_prereg.md).
Run from the repo root: python3 paper2/synth.py >> paper2/synth_output.txt
Weights: gauge_weights/synth_* (git-ignored). Data generated in memory from fixed seeds."""
import copy
import itertools
import json
import math
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment, minimize
from scipy.stats import spearmanr
from torch.func import functional_call

from certify import certify_point
from kan import LR, N_RBF, SMOOTH, WD, KANLayer

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
FIG = ROOT / "figs"
W_DIR = REPO / "gauge_weights"
LAM = SMOOTH  # 1e-3
K_SILU = (30 + math.pi ** 2) / 90
N = 2000
CONDS = {"independent": 0.0, "correlated": 0.95}
LEVELS = ["P0", "P1", "P2"]
MU_GRID = [1e-6, 1e-5, 1e-4, 1e-3]
SEEDS, PILOT = list(range(8)), list(range(100, 108))
PATIENCE, MAX_EP = 100, 10000
MAXITER, WALL = 500, 2 * 3600
GTOL, REL_TOL, EIG_TOL = 1e-7, 1e-12, 1e-8
RESULTS = {}
NB = torch.tensor(np.linalg.qr(np.column_stack([np.ones(N_RBF), np.arange(N_RBF, dtype=float)]))[0])
D2 = lambda c: c[..., 2:] - 2 * c[..., 1:-1] + c[..., :-2]


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ model and penalties
class SynthKAN(nn.Module):
    """KAN [4, 3, 1]: two kan.KANLayers with tanh between them, no salt embedding."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([KANLayer(4, 3), KANLayer(3, 1)])

    def forward(self, x):
        return self.layers[1](torch.tanh(self.layers[0](x)))


def penalty_terms(prm):
    c0, c1 = prm["layers.0.coef"], prm["layers.1.coef"]
    d2 = LAM * ((D2(c0) ** 2).sum() + (D2(c1) ** 2).sum())
    silu = LAM * K_SILU * ((prm["layers.0.base.weight"] ** 2).sum() + (prm["layers.1.base.weight"] ** 2).sum())
    q = ((c0 @ NB.to(c0.dtype)) ** 2).sum() + ((c1 @ NB.to(c1.dtype)) ** 2).sum()
    return d2, silu, q


def cell_penalty(prm, level, mu):
    d2, silu, q = penalty_terms(prm)
    return d2 + (silu if level in ("P1", "P2") else 0.0) + (mu * q if level == "P2" else 0.0)


class FlatS:
    """All parameters of a SynthKAN as one float64 vector, with the cell's loss."""

    def __init__(self, net, X, y, level, mu):
        self.net = copy.deepcopy(net).double()
        self.meta = [(n, p.shape, p.numel()) for n, p in self.net.named_parameters()]
        self.x0 = torch.cat([p.detach().reshape(-1) for p in self.net.parameters()]).numpy().copy()
        self.X, self.y, self.level, self.mu = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64), level, mu
        self.n = self.x0.size
        self._cache = None

    def params(self, v):
        out, o = {}, 0
        for n, shp, k in self.meta:
            out[n] = v[o:o + k].reshape(shp); o += k
        return out

    def loss_t(self, v):
        prm = self.params(v)
        pred = functional_call(self.net, prm, (self.X,))[:, 0]
        return F.mse_loss(pred, self.y) + cell_penalty(prm, self.level, self.mu)

    def fun(self, x):
        if self._cache is not None and np.array_equal(self._cache[0], x):
            return self._cache[1], self._cache[2]
        v = torch.tensor(x, requires_grad=True)
        L = self.loss_t(v)
        (g,) = torch.autograd.grad(L, v)
        self._cache = (x.copy(), float(L), g.numpy().copy())
        return self._cache[1], self._cache[2]

    def hessian(self, x):
        H = torch.autograd.functional.hessian(self.loss_t, torch.tensor(x)).numpy()
        return (H + H.T) / 2

    def to_net(self, x):
        net = copy.deepcopy(self.net)
        with torch.no_grad():
            for (n, p), (_, t) in zip(net.named_parameters(), self.params(torch.tensor(x)).items()):
                p.copy_(t)
        return net.eval()


def run_trust(flat):
    """trust-exact, float64, cap 500; same criteria as stage2/close.py; also tracks ||theta||^2 per iteration."""
    x0 = flat.x0.copy()
    L0, g0 = flat.fun(x0)
    hist = dict(loss=[L0], grad=[float(np.abs(g0).max())], theta2=[float((x0 ** 2).sum())])
    st = dict(prev=x0.copy(), t0=time.time(), stop="scipy termination")

    def callback(intermediate_result):
        x = intermediate_result.x
        L, g = flat.fun(x)
        hist["loss"].append(L); hist["grad"].append(float(np.abs(g).max())); hist["theta2"].append(float((x ** 2).sum()))
        st["prev"] = x.copy()
        k = len(hist["loss"]) - 1
        if k >= 10 and hist["grad"][k] < GTOL and abs(hist["loss"][k - 10] - L) / hist["loss"][k - 10] < REL_TOL:
            ev = np.linalg.eigvalsh(flat.hessian(x))
            if ev[0] >= -EIG_TOL * ev[-1]:
                st["stop"] = "converged (stage2 criterion)"
                raise StopIteration
        if time.time() - st["t0"] > WALL:
            st["stop"] = "2 h cap"
            raise StopIteration

    res = minimize(flat.fun, x0, jac=True, hess=flat.hessian, method="trust-exact", callback=callback,
                   options=dict(maxiter=MAXITER, gtol=1e-30))
    if st["stop"] == "scipy termination" and len(hist["loss"]) - 1 >= MAXITER:
        st["stop"] = "iteration cap (500)"
    return st["prev"], dict(stop=st["stop"], message=str(res.message), iters=len(hist["loss"]) - 1), hist


def train_early(seed, X, y, Xv, yv, level, mu):
    """AdamW as final.py (lr 3e-3, wd 1e-4, full batch), early-stopped on the validation data term."""
    torch.manual_seed(seed)
    net = SynthKAN()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    Xt, yt, Xvt, yvt = (torch.tensor(a, dtype=torch.float32) for a in (X, y, Xv, yv))
    best, best_ep, best_state, traj = float("inf"), 0, None, []
    for ep in range(1, MAX_EP + 1):
        opt.zero_grad()
        prm = dict(net.named_parameters())
        loss = F.mse_loss(net(Xt)[:, 0], yt) + cell_penalty(prm, level, mu)
        loss.backward()
        opt.step()
        with torch.no_grad():
            v = float(F.mse_loss(net(Xvt)[:, 0], yvt))
            if ep % 10 == 0:
                traj.append((ep, float(sum((p ** 2).sum() for p in net.parameters()))))
        if v < best:
            best, best_ep, best_state = v, ep, copy.deepcopy(net.state_dict())
        elif ep - best_ep >= PATIENCE:
            break
    net.load_state_dict(best_state)
    return net.eval(), dict(best_epoch=best_ep, stopped_epoch=ep, val=best), traj


# ------------------------------------------------------------------ teacher
def edge_basis(u):
    g = np.linspace(-1, 1, N_RBF); h = 2 / (N_RBF - 1)
    return np.column_stack([u / (1 + np.exp(-u))] + [np.exp(-(((u - gk) / h) ** 2)) for gk in g])


def build_teacher(s):
    L1 = {(0, 0): lambda x: 0.8 * np.sin(1.5 * s * x), (1, 0): lambda x: 0.5 * (s * x) ** 2, (2, 0): lambda x: 0.6 * np.tanh(2 * s * x), (3, 0): lambda x: -0.4 * x,
          (0, 1): lambda x: 0.5 * (s * x) ** 2, (1, 1): lambda x: -0.7 * np.sin(1.2 * s * x), (2, 1): lambda x: 0.4 * x, (3, 1): lambda x: 0.6 * np.tanh(1.5 * s * x),
          (0, 2): lambda x: -0.5 * np.tanh(2 * s * x), (1, 2): lambda x: 0.4 * x, (2, 2): lambda x: 0.6 * np.sin(1.5 * s * x), (3, 2): lambda x: -0.5 * (s * x) ** 2}
    L2 = {0: lambda u: u, 1: lambda u: 0.8 * np.sin(1.5 * s * u), 2: lambda u: 0.6 * (s * u) ** 2}
    grid = np.linspace(-1, 1, 401); B = edge_basis(grid)
    Dm = np.zeros((N_RBF - 2, 9))
    for k in range(N_RBF - 2):
        Dm[k, 1 + k], Dm[k, 2 + k], Dm[k, 3 + k] = 1, -2, 1
    R = LAM * (Dm.T @ Dm); R[0, 0] += LAM * K_SILU

    def fit(f):
        return np.linalg.solve(B.T @ B / len(grid) + R, B.T @ f(grid) / len(grid))

    T = SynthKAN().double()
    with torch.no_grad():
        for (i, j), f in L1.items():
            c = fit(f); T.layers[0].base.weight[j, i] = c[0]; T.layers[0].coef[j, i] = torch.tensor(c[1:])
        for j, f in L2.items():
            c = fit(f); T.layers[1].base.weight[0, j] = c[0]; T.layers[1].coef[0, j] = torch.tensor(c[1:])
        T.layers[0].base.bias.zero_(); T.layers[1].base.bias.zero_()
    return T.eval()


def make_inputs(rho, seed, independent=False):
    z = np.random.default_rng(seed).standard_normal((N, 4))
    if rho and not independent:
        z[:, 1] = rho * z[:, 0] + math.sqrt(1 - rho ** 2) * z[:, 1]
        z[:, 3] = rho * z[:, 2] + math.sqrt(1 - rho ** 2) * z[:, 3]
    return np.clip(z / 3, -1, 1)


def predict(net, X):
    with torch.no_grad():
        p = next(net.parameters())
        return net(torch.tensor(X, dtype=p.dtype))[:, 0].double().numpy()


# ------------------------------------------------------------------ diagnostics and gauge (embedding-free)
def parts(net, X):
    net = net.double()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float64)
        L0, L1 = net.layers
        phi = torch.stack([L0.edge(i, Xt[:, i]) for i in range(4)])          # (4, n, 3)
        s = phi.sum(0) + L0.base.bias
        ps = torch.stack([L1.edge(j, torch.tanh(s[:, j]))[:, 0] for j in range(3)], 1)
    return phi.numpy(), s.numpy(), ps.numpy()


def diagnostics(net, X):
    phi, s, _ = parts(net, X)
    rms = np.sqrt((phi ** 2).mean(1))                       # (4, 3)
    tot = np.sqrt((phi.sum(0) ** 2).mean(0))                # (3,)
    sd = dict(net.named_parameters())
    blocks = {"layer-1 RBF": ["layers.0.coef"], "layer-2 RBF": ["layers.1.coef"], "SiLU base": ["layers.0.base.weight", "layers.1.base.weight"],
              "biases": ["layers.0.base.bias", "layers.1.base.bias"]}
    norms = {b: float(sum((sd[k].detach() ** 2).sum() for k in ks)) for b, ks in blocks.items()}
    return dict(cross_edge=float(np.median(rms.sum(0) / tot)), saturated=float((np.abs(s) > 3).mean()), norms=norms, theta2=sum(norms.values()))


def eig_summary(flat, x):
    ev = np.linalg.eigvalsh(flat.hessian(x))
    return dict(min=float(ev[0]), max=float(ev[-1]), n_neg=int((ev < -EIG_TOL * ev[-1]).sum()),
                n_null=int((np.abs(ev) < EIG_TOL * ev[-1]).sum()), deciles=[float(q) for q in np.quantile(ev, np.linspace(0, 1, 11))])


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


class Gauge:
    """gauge.py steps 1-4 for the embedding-free network, on the training rows."""

    def __init__(self, net, X):
        phi, s, ps = parts(net, X)
        self.m, self.mu, self.sd = phi.mean(1), s.mean(0), s.std(0) + 1e-12
        sh = (s - self.mu) / self.sd
        self.sign = np.array([1.0 if corr(sh[:, j], ps[:, j]) >= 0 else -1.0 for j in range(3)])
        self.c = ps - ps.mean(0)
        self.perm = np.arange(3)

    def fixed(self, phi):
        return (self.sign * (phi - self.m[:, None, :]) / self.sd)[..., self.perm]


def align(ref, g):
    A, B = ref.c[:, ref.perm], g.c
    A, B = A - A.mean(0), B - B.mean(0)
    C = (A.T @ B) / np.outer(np.sqrt((A * A).sum(0)), np.sqrt((B * B).sum(0))).clip(1e-12)
    r, k = linear_sum_assignment(1 - np.abs(C))
    g.perm = k[np.argsort(r)]
    return np.abs(C[np.arange(3), g.perm])


def match(net, ref, X):
    gR, gS = Gauge(ref, X), Gauge(net, X)
    um = align(gR, gS)
    fR = gR.fixed(parts(ref, X)[0])
    var = fR.var(1)
    active = var >= 0.05 * var.max(0, keepdims=True)
    lo, hi = X.min(0), X.max(0)
    med = np.median(X, 0)
    edges = []
    for i in range(4):
        G = np.tile(med, (100, 1)); G[:, i] = np.linspace(lo[i], hi[i], 100)
        eR, eS = gR.fixed(parts(ref, G)[0])[i], gS.fixed(parts(net, G)[0])[i]
        edges += [abs(corr(eS[:, j], eR[:, j])) for j in range(3) if active[i, j]]
    return um, float(np.median(edges))


# ------------------------------------------------------------------ main
def main():
    FIG.mkdir(parents=True, exist_ok=True); W_DIR.mkdir(exist_ok=True)
    log("#" * 100)
    log(f"paper2/synth run {time.strftime('%Y-%m-%d %H:%M:%S')} (see paper2/synth_prereg.md)")
    log("DISCREPANCIES / choices (files win):")
    log("  kan.KAN always adds a salt embedding -> SynthKAN = two kan.KANLayers with tanh between them, no embedding")
    log("  inputs N(0,1)/3 then clipped to the grid range [-1,1] (clipping N(0,1) directly would put ~32% on the boundary)")
    log("  an extra 2000-row validation split for early stopping (the brief's test rows stay untouched)")
    log("  P2's Q holds only the RBF null-space part (SiLU already penalised in P1), unlike close.py's Q; P2 uses mu*Q in BOTH stages")
    log(f"  K_silu = (30 + pi^2)/90 = {K_SILU:.5f} (real-line integral; over [-1,1] it would be 0.373)")

    data = {}
    for ci, (cond, rho) in enumerate(CONDS.items()):
        data[cond] = dict(X=make_inputs(rho, 1000 + ci), Xv=make_inputs(rho, 2000 + ci), Xt=make_inputs(rho, 3000 + ci),
                          Xo=make_inputs(rho, 4000 + ci, independent=True))
        log(f"  {cond}: realised corr(x1,x2) {np.corrcoef(data[cond]['X'][:, 0], data[cond]['X'][:, 1])[0, 1]:.3f}, "
            f"corr(x3,x4) {np.corrcoef(data[cond]['X'][:, 2], data[cond]['X'][:, 3])[0, 1]:.3f}; "
            f"clipped {np.mean(np.abs(data[cond]['X']) >= 1):.4f}")

    # ---------------- Step 0: teacher acceptance
    log("\n==================== STEP 0: teacher, acceptance, mu calibration (calibration only; no verdicts) ====================")
    teacher, s, refs = None, 1.0, {}
    for rnd in range(11):
        T = build_teacher(s)
        ok_all, polishes = True, {}
        for ci, cond in enumerate(CONDS):
            dd = data[cond]
            y0 = predict(T, dd["X"]); sig = 0.05 * y0.std()
            rng = np.random.default_rng(5000 + ci)
            dd.update(y=y0 + sig * rng.standard_normal(N), yv=predict(T, dd["Xv"]) + sig * rng.standard_normal(N), sigma=sig, y0=y0)
            for lev in ["P0", "P1"]:
                fl = FlatS(T, dd["X"], dd["y"], lev, 0.0)
                x, info, hist = run_trust(fl)
                net = fl.to_net(x)
                dp = float(np.sqrt(np.mean((predict(net, dd["X"]) - y0) ** 2)))
                rc, _, _ = certify_point(fl, x)
                polishes[(cond, lev)] = dict(net=net, cert=rc["certified"], dpred=dp, ok=dp < 0.5 * sig, info=info,
                                             growth=hist["theta2"][-1] / hist["theta2"][0])
                ok_all &= dp < 0.5 * sig
                log(f"  round {rnd} (frequency scale {s:.3f}) {cond:11s} {lev}: RMS prediction change {dp:.2e} vs 0.5 sigma {0.5 * sig:.2e} -> "
                    f"{'ok' if dp < 0.5 * sig else 'FAIL'}; {info['stop']} after {info['iters']} iters; certified {rc['certified']}; "
                    f"||theta||^2 growth x{hist['theta2'][-1] / hist['theta2'][0]:.2f}")
        if ok_all:
            teacher = T
            break
        s *= 0.8
    if teacher is None:
        log("STOP: teacher acceptance failed after 10 smoothing rounds")
        (ROOT / "synth_results.json").write_text(json.dumps(dict(stopped="teacher acceptance"), indent=1)); return
    with torch.no_grad():
        d2, silu, q = penalty_terms({k: v for k, v in teacher.named_parameters()})
    log(f"  teacher accepted at frequency scale {s:.3f}: D2 term {float(d2):.3e}, SiLU curvature term {float(silu):.3e}, Q {float(q):.3f}; "
        + "; ".join(f"{c}: std(output) {data[c]['y0'].std():.3f}, sigma {data[c]['sigma']:.3e}" for c in CONDS))
    for k, v in polishes.items():
        refs[k] = v["net"] if v["cert"] else teacher

    # ---------------- mu calibration
    mu_frozen, calib = None, []
    for mu in MU_GRID:
        ok, row = True, dict(mu=mu)
        reg = {}
        for cond in CONDS:
            dd = data[cond]
            fl = FlatS(teacher, dd["X"], dd["y"], "P2", mu)
            x, info, _ = run_trust(fl)
            net = fl.to_net(x)
            dp = float(np.sqrt(np.mean((predict(net, dd["X"]) - dd["y0"]) ** 2)))
            rc, _, _ = certify_point(fl, x)
            reg[cond] = (net, rc["certified"])
            row[f"{cond}_dpred"], row[f"{cond}_cert"] = dp, rc["certified"]
            ok &= dp < 0.5 * dd["sigma"] and rc["certified"]
            log(f"  mu {mu:g} regularised teacher {cond:11s}: RMS change {dp:.2e} (0.5 sigma {0.5 * dd['sigma']:.2e}); certified {rc['certified']}")
        if ok:
            dd = data["correlated"]; ratios = []
            for sd in PILOT:
                net0, _, _ = train_early(sd, dd["X"], dd["y"], dd["Xv"], dd["yv"], "P2", mu)
                fl = FlatS(net0, dd["X"], dd["y"], "P2", mu)
                x, info, hist = run_trust(fl)
                ratios.append(hist["theta2"][-1] / hist["theta2"][0])
            row["pilot_max_growth"] = float(max(ratios))
            ok &= max(ratios) <= 3
            log(f"  mu {mu:g} pilot (calibration only, seeds {PILOT[0]}-{PILOT[-1]}, correlated-P2): max ||theta||^2 growth x{max(ratios):.2f} -> {'ok' if max(ratios) <= 3 else 'FAIL'}")
        calib.append(row)
        if ok:
            mu_frozen = mu
            for cond in CONDS:
                refs[(cond, "P2")] = reg[cond][0]
            break
    RESULTS.update(teacher_scale=s, calibration=calib, mu=mu_frozen,
                   references={f"{c}|{l}": ("polished teacher (certified)" if l != "P2" and polishes[(c, l)]["cert"] else
                                            "regularised teacher (certified)" if l == "P2" else "original teacher") for c in CONDS for l in LEVELS})
    if mu_frozen is None:
        log("STOP: no mu passed calibration; Step 1 not run; all predictions inconclusive")
        (ROOT / "synth_results.json").write_text(json.dumps(RESULTS, indent=1, default=float)); return
    log(f"  FROZEN: mu = {mu_frozen:g}; teacher frequency scale {s:.3f}; data seeds 1000-5001")
    log("  references: " + ", ".join(f"{k}: {v}" for k, v in RESULTS["references"].items()))

    # ---------------- Step 1
    log("\n==================== STEP 1: training (48 models) ====================")
    models = {}
    for cond in CONDS:
        dd = data[cond]
        for lev in LEVELS:
            mu = mu_frozen if lev == "P2" else 0.0
            for sd in SEEDS:
                key = f"{cond}_{lev}_s{sd}"
                net0, einfo, traj = train_early(sd, dd["X"], dd["y"], dd["Xv"], dd["yv"], lev, mu)
                fl = FlatS(net0, dd["X"], dd["y"], lev, mu)
                x, pinfo, hist = run_trust(fl)
                netP = fl.to_net(x)
                rc, _, _ = certify_point(fl, x)
                rec = dict(early=einfo, polish=pinfo, certified=rc["certified"], crit=rc["crit"], grad=rc["grad"],
                           early_diag=diagnostics(net0.double(), dd["X"]), pol_diag=diagnostics(netP, dd["X"]),
                           early_eig=eig_summary(fl, fl.x0), pol_eig=eig_summary(fl, x), traj=traj, trust_theta2=hist["theta2"])
                for tag, net in [("early", net0), ("pol", netP)]:
                    with torch.no_grad():
                        prm = dict(net.double().named_parameters())
                        d2_, si_, q_ = penalty_terms(prm)
                    rec[f"{tag}_terms"] = dict(data=float(np.mean((predict(net, dd["X"]) - dd["y"]) ** 2)), d2=float(d2_), silu=float(si_), muQ=float(mu * q_))
                    rec[f"{tag}_rmse_in"] = float(np.sqrt(np.mean((predict(net, dd["Xt"]) - predict(teacher, dd["Xt"])) ** 2)))
                    rec[f"{tag}_rmse_off"] = float(np.sqrt(np.mean((predict(net, dd["Xo"]) - predict(teacher, dd["Xo"])) ** 2)))
                rec["growth"] = rec["pol_diag"]["theta2"] / rec["early_diag"]["theta2"]
                torch.save(dict(early=net0.state_dict(), polished=netP.state_dict()), W_DIR / f"synth_{key}.pt")
                models[key] = (rec, netP)
                log(f"  {key:22s} early ep {einfo['best_epoch']:5d} | {pinfo['stop']} {pinfo['iters']:3d} it | certified {str(rc['certified']):5s} | "
                    f"||theta||^2 {rec['early_diag']['theta2']:9.1f} -> {rec['pol_diag']['theta2']:10.1f} (x{rec['growth']:.2f}) | "
                    f"cross-edge {rec['early_diag']['cross_edge']:.2f} -> {rec['pol_diag']['cross_edge']:.2f} | sat {rec['pol_diag']['saturated']:.3f} | "
                    f"off-RMSE {rec['early_rmse_off']:.4f} -> {rec['pol_rmse_off']:.4f} | in-RMSE {rec['early_rmse_in']:.4f} -> {rec['pol_rmse_in']:.4f}")

    # ---------------- Step 2
    log("\n==================== STEP 2: identifiability and extrapolation ====================")
    cells = {}
    for cond in CONDS:
        dd = data[cond]; sd_t = dd["y0"].std()
        for lev in LEVELS:
            keys = [f"{cond}_{lev}_s{s_}" for s_ in SEEDS]
            ref = refs[(cond, lev)]
            ums, eds, pcs = [], [], []
            for k in keys:
                um, ed = match(models[k][1], ref, dd["X"])
                ums += um.tolist(); eds.append(ed); pcs.append(corr(predict(models[k][1], dd["X"]), predict(ref, dd["X"])))
            preds = {k: predict(models[k][1], dd["X"]) for k in keys}
            dab = [float(np.sqrt(np.mean((preds[a] - preds[b]) ** 2)) / sd_t) for a, b in itertools.combinations(keys, 2)]
            R_ = [models[k][0] for k in keys]
            c = dict(growth=float(np.median([r["growth"] for r in R_])), certified=int(sum(r["certified"] for r in R_)),
                     cross=float(np.median([r["pol_diag"]["cross_edge"] for r in R_])),
                     off_ratio=float(np.median([r["pol_rmse_off"] / r["early_rmse_off"] for r in R_])),
                     unit=float(np.median(ums)), edge=float(np.median(eds)), pred_corr=float(np.median(pcs)), dAB=float(np.median(dab)),
                     reference=RESULTS["references"][f"{cond}|{lev}"])
            cells[(cond, lev)] = c
            log(f"  {cond:11s} {lev}: growth x{c['growth']:.2f} | certified {c['certified']}/8 | cross-edge {c['cross']:.2f} | off-RMSE ratio {c['off_ratio']:.2f} | "
                f"unit {c['unit']:.3f} | edge {c['edge']:.3f} | pred corr {c['pred_corr']:.5f} | d_AB {c['dAB']:.4f} | reference: {c['reference']}")

    corr_keys = [k for k in models if k.startswith("correlated")]
    rho, pval = spearmanr([models[k][0]["pol_diag"]["cross_edge"] for k in corr_keys], [models[k][0]["pol_rmse_off"] for k in corr_keys])
    C = lambda c, l: cells[(c, l)]
    v = {}
    v["X1"] = ("held" if C("correlated", "P0")["growth"] >= 10 and C("correlated", "P1")["growth"] >= 10 and
               all(C("independent", l)["growth"] <= 3 for l in LEVELS) and C("correlated", "P2")["growth"] <= 3 else "failed") + \
        " (" + ", ".join(f"{c[:4]}-{l} x{C(c, l)['growth']:.2f}" for c in CONDS for l in LEVELS) + ")"
    v["X2"] = ("held" if C("correlated", "P0")["certified"] <= 2 and C("correlated", "P1")["certified"] <= 2 and
               all(C(c, l)["certified"] >= 6 for c, l in [("independent", "P0"), ("independent", "P1"), ("independent", "P2"), ("correlated", "P2")]) else "failed") + \
        " (" + ", ".join(f"{c[:4]}-{l} {C(c, l)['certified']}/8" for c in CONDS for l in LEVELS) + ")"
    v["X3"] = ("held" if C("correlated", "P0")["cross"] >= 2 * C("correlated", "P2")["cross"] and C("correlated", "P1")["cross"] >= 2 * C("correlated", "P2")["cross"] else "failed") + \
        f" (corr P0 {C('correlated', 'P0')['cross']:.2f}, P1 {C('correlated', 'P1')['cross']:.2f}, P2 {C('correlated', 'P2')['cross']:.2f})"
    v["X4"] = ("held" if C("correlated", "P0")["off_ratio"] >= 3 and C("correlated", "P1")["off_ratio"] >= 3 and C("correlated", "P2")["off_ratio"] <= 1.5 else "failed") + \
        f" (corr P0 {C('correlated', 'P0')['off_ratio']:.2f}, P1 {C('correlated', 'P1')['off_ratio']:.2f}, P2 {C('correlated', 'P2')['off_ratio']:.2f})"
    v["X5"] = ("held" if rho >= 0.6 else "failed") + f" (rho {rho:+.3f}, p {pval:.3f}, n = {len(corr_keys)})"
    v["X6"] = ("held" if all(C(c, l)["unit"] >= 0.85 for c, l in [("independent", "P0"), ("independent", "P1"), ("independent", "P2"), ("correlated", "P2")])
               and C("correlated", "P0")["unit"] < 0.7 and C("correlated", "P1")["unit"] < 0.7 else "failed") + \
        " (" + ", ".join(f"{c[:4]}-{l} {C(c, l)['unit']:.3f}" for c in CONDS for l in LEVELS) + ")"
    log("\nVERDICTS: " + "; ".join(f"{k}: {x}" for k, x in v.items()))
    RESULTS.update(cells={f"{c}|{l}": x for (c, l), x in cells.items()}, models={k: r for k, (r, _) in models.items()},
                   x5=dict(rho=float(rho), p=float(pval)), verdicts=v)
    (ROOT / "synth_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))

    fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    for a, cond in enumerate(CONDS):
        for b, lev in enumerate(LEVELS):
            ax = axs[a, b]
            for sd in SEEDS:
                r = models[f"{cond}_{lev}_s{sd}"][0]
                e = [t for _, t in r["traj"]]
                xs = list(range(len(e))) + [len(e) + k / 10 for k in range(len(r["trust_theta2"]))]
                ax.semilogy(xs, e + r["trust_theta2"], lw=0.8, alpha=0.8)
            ax.set_title(f"{cond} {lev} (median growth x{C(cond, lev)['growth']:.1f})"); ax.set_xlabel("AdamW epoch/10, then trust-exact iteration/10")
        axs[a, 0].set_ylabel("||theta||^2")
    fig.tight_layout(); fig.savefig(FIG / "theta_trajectories.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for lev, mk in zip(LEVELS, "osD"):
        kk = [k for k in corr_keys if f"_{lev}_" in k]
        ax.scatter([models[k][0]["pol_diag"]["cross_edge"] for k in kk], [models[k][0]["pol_rmse_off"] for k in kk], marker=mk, s=40, label=f"correlated {lev}")
    ax.set_xlabel("cross-edge cancellation ratio (polished)"); ax.set_ylabel("off-manifold RMSE vs teacher (polished)"); ax.set_yscale("log")
    ax.legend(); ax.set_title(f"X5: Spearman rho {rho:+.2f} (p {pval:.2f}, n = {len(corr_keys)})")
    fig.tight_layout(); fig.savefig(FIG / "offmanifold_vs_cancellation.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()

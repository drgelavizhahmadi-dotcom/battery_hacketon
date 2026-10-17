"""paper2/revision/freeze.py: revision step 2, is composition necessary? (plus weight decay). See freeze_prereg.md (a6d752b).
Run from the repo root with the pinned interpreter:
  /usr/bin/python3 paper2/revision/freeze.py >> paper2/revision/freeze_output.txt
Inputs SHA-verified against weights_manifest.txt and held in memory; every run is cached in gauge_weights/freeze_* (SHA-indexed)
so an interruption resumes. No weights in git."""
import copy
import io
import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "paper2", REPO / "arch"):
    sys.path.insert(0, str(p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from torch.func import functional_call

import synth as SY
import sc_run as SR
from certify import certify_point
from kan import DELTA_IN, KAN, LR, SMOOTH, KANLayer, delta_head, predict, subset
from stage2 import extract_verified, in_memory

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
MAXITER, WALL, LOG_EVERY = 500, 2 * 3600, 25
GRAD_TOL, REL_TOL, EIG_TOL = 1e-7, 1e-12, 1e-8          # as stage2.run_exact / trust.py
WD = 1e-4
FIG = HERE / "figs"
RES = {}


def log(*a):
    print(*a, flush=True)


D2 = lambda c: c[..., 2:] - 2 * c[..., 1:-1] + c[..., :-2]


# ------------------------------------------------------------------ generic flattener over a subset of free parameters
class FlatFree:
    def __init__(self, net, full, free, loss_parts, extra=None):
        """full: {name: float64 tensor}; free: names polished; loss_parts(prm) -> dict of terms (sum = loss); extra(prm) -> added term"""
        self.net, self.full, self.free = net, {k: v.detach().clone().double() for k, v in full.items()}, list(free)
        self.parts, self.extra = loss_parts, extra
        self.meta, o = [], 0
        for n in self.free:
            k = self.full[n].numel(); self.meta.append((n, self.full[n].shape, o, o + k)); o += k
        self.n = o
        self.x0 = torch.cat([self.full[n].reshape(-1) for n in self.free]).numpy().copy()
        self._cache = None

    def params(self, v):
        prm = dict(self.full)
        for n, shp, a, b in self.meta:
            prm[n] = v[a:b].reshape(shp)
        return prm

    def loss_t(self, v):
        prm = self.params(v)
        t = sum(self.parts(prm).values())
        return t + (self.extra(prm) if self.extra else 0.0)

    def terms(self, x):
        with torch.no_grad():
            prm = self.params(torch.tensor(x))
            d = {k: float(v) for k, v in self.parts(prm).items()}
            if self.extra:
                d["extra_L2"] = float(self.extra(prm))
            return d

    def base_loss(self, x):
        return self.terms(x)["data"]

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

    def full_at(self, x):
        with torch.no_grad():
            return {k: v.detach().clone() for k, v in self.params(torch.tensor(x)).items()}


def block_norms(prm, blocks):
    return {b: float(sum((prm[n] ** 2).sum() for n in names if n in prm)) for b, names in blocks.items()}


def run(flat, blocks):
    """trust-exact (float64, exact Hessian, cap 500), stage2 stopping rule; logs block norms, eigen extremes and terms every 25 iterates"""
    x0 = flat.x0.copy()
    L0, g0 = flat.fun(x0)
    hist = dict(loss=[L0], grad=[float(np.abs(g0).max())])
    logged = []

    def snap(k, x):
        ev = np.linalg.eigvalsh(flat.hessian(x))
        logged.append(dict(it=k, min_eig=float(ev[0]), max_eig=float(ev[-1]), norms=block_norms(flat.full_at(x), blocks), terms=flat.terms(x)))
    snap(0, x0)
    st = dict(prev=x0.copy(), t0=time.time(), stop="scipy termination")

    def callback(intermediate_result):
        x = intermediate_result.x
        L, g = flat.fun(x)
        hist["loss"].append(L); hist["grad"].append(float(np.abs(g).max()))
        st["prev"] = x.copy()
        k = len(hist["loss"]) - 1
        if k % LOG_EVERY == 0:
            snap(k, x)
        if k >= 10 and hist["grad"][k] < GRAD_TOL and abs(hist["loss"][k - 10] - L) / max(abs(hist["loss"][k - 10]), 1e-300) < REL_TOL:
            ev = np.linalg.eigvalsh(flat.hessian(x))
            if ev[0] >= -EIG_TOL * ev[-1]:
                st["stop"] = "converged (stage2 criterion)"
                raise StopIteration
        if time.time() - st["t0"] > WALL:
            st["stop"] = "2 h wall-clock cap"
            raise StopIteration

    res = minimize(flat.fun, x0, jac=True, hess=flat.hessian, method="trust-exact", callback=callback, options=dict(maxiter=MAXITER, gtol=1e-30))
    k = len(hist["loss"]) - 1
    if st["stop"] == "scipy termination" and k >= MAXITER:
        st["stop"] = "iteration cap (500)"
    x = st["prev"]
    if not logged or logged[-1]["it"] != k:
        snap(k, x)
    cert, _, _ = certify_point(flat, x)
    return x, dict(stop=st["stop"], message=str(res.message), iters=k, wall_s=round(time.time() - st["t0"], 1), cert=cert, logged=logged)


# ------------------------------------------------------------------ models and losses
REAL_BLOCKS = {"L1 RBF": ["layers.0.coef"], "L1 SiLU": ["layers.0.base.weight"], "L1 bias": ["layers.0.base.bias"],
               "L2 RBF": ["layers.1.coef"], "L2 SiLU": ["layers.1.base.weight"], "L2 bias": ["layers.1.base.bias"], "salt emb": ["emb.weight"]}
SYN_BLOCKS = {k: v for k, v in REAL_BLOCKS.items() if k != "salt emb"}
L1 = ["layers.0.coef", "layers.0.base.weight", "layers.0.base.bias"]
L2 = ["layers.1.coef", "layers.1.base.weight", "layers.1.base.bias"]


def real_parts(net, b):
    def parts(prm):
        pred = b["x_co"] * functional_call(net, prm, (b["x"], b["s"]))[:, 0]
        return dict(data=F.mse_loss(pred, b["y"]), D2=SMOOTH * sum((D2(prm[c]) ** 2).sum() for c in ("layers.0.coef", "layers.1.coef")))
    return parts


def syn_parts(net, X, y, lev):
    Xt, yt = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    def parts(prm):
        pred = functional_call(net, prm, (Xt,))[:, 0]
        d2, silu, q = SY.penalty_terms(prm)
        out = dict(data=F.mse_loss(pred, yt), D2=d2)
        if lev in ("P1", "P2"):
            out["SiLU"] = silu
        return out
    return parts


class OneLayer(nn.Module):
    """Arm C: a single kan.KANLayer [4, 1], linear output, no tanh"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([KANLayer(4, 1)])

    def forward(self, x):
        return self.layers[0](x)


def one_parts(net, X, y):
    Xt, yt = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    def parts(prm):
        pred = functional_call(net, prm, (Xt,))[:, 0]
        return dict(data=F.mse_loss(pred, yt), D2=SY.LAM * (D2(prm["layers.0.coef"]) ** 2).sum())
    return parts


def train_early_generic(ctor, seed, X, y, Xv, yv, penalty, wd):
    """identical to synth.train_early except the model constructor, the penalty function and weight_decay"""
    torch.manual_seed(seed)
    net = ctor()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=wd)
    Xt, yt, Xvt, yvt = (torch.tensor(a, dtype=torch.float32) for a in (X, y, Xv, yv))
    best, best_ep, best_state = float("inf"), 0, None
    for ep in range(1, SY.MAX_EP + 1):
        opt.zero_grad()
        prm = dict(net.named_parameters())
        loss = F.mse_loss(net(Xt)[:, 0], yt) + penalty(prm)
        loss.backward()
        opt.step()
        with torch.no_grad():
            v = float(F.mse_loss(net(Xvt)[:, 0], yvt))
        if v < best:
            best, best_ep, best_state = v, ep, copy.deepcopy(net.state_dict())
        elif ep - best_ep >= SY.PATIENCE:
            break
    net.load_state_dict(best_state)
    return net.eval(), dict(best_epoch=best_ep, stopped_epoch=ep, val=best)


def kan_train_wd(n_salt, b, epochs, seed, wd):
    """identical to kan.train (no validation) except weight_decay"""
    torch.manual_seed(seed)
    net = KAN(len(DELTA_IN), [6, 1], n_salt)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(delta_head(net, b), b["y"]) + SMOOTH * net.smoothness()
        loss.backward()
        opt.step()
    return net.eval()


def theta2(sd):
    return float(sum((v.double() ** 2).sum() for k, v in sd.items() if not k.endswith("grid")))


def to_cache(obj):
    return {k: (torch.tensor(v) if isinstance(v, np.ndarray) else v) for k, v in obj.items()}


# ------------------------------------------------------------------ main
def main():
    FIG.mkdir(exist_ok=True)
    log("#" * 100)
    log(f"paper2/revision/freeze.py {time.strftime('%Y-%m-%d %H:%M:%S')}: revision step 2 (prereg freeze_prereg.md, a6d752b)")
    log("#" * 100)
    log("DISCREPANCIES / choices (files win):")
    log("  F1 Arm A real loss = trust.Flat's final.py loss (MSE on delta + 1e-3*sum D2 over both layers), all 2069 anchored rows,")
    log("     no weight decay (prereg decision); layer 1 AND the salt embedding are frozen (the embedding feeds layer 1)")
    log("  F2 Arm B: layer 2 frozen; layer 1 (coef, SiLU weights, biases) polished; the salt embedding stays frozen")
    log("  F3 growth for Z2 = squared norm of all layer-2 parameters (RBF coef + SiLU weights + bias), polished / early")
    log("  F4 Arm D adds 1e-4 * ||theta||^2 over all parameters, as specified (the AdamW-equivalent would be 5e-5 * ||theta||^2)")
    log("  F5 Arm E wd = 1e-4 references are the committed early states; a determinism check retrains seed 0 with wd = 1e-4")
    log("  F6 per-iterate logs (every 25 iterates + final: block norms, min/max Hessian eigenvalue, terms) are in freeze_results.json")

    names = ([f"delta_s{i}.pt" for i in range(6)] + [f"synth_correlated_{l}_s{s}.pt" for l in ("P0", "P1") for s in range(8)]
             + [f"scbase_A_s{s}.pt" for s in range(5)])
    log("\n==================== inputs (SHA-256 verified, held in memory) ====================")
    blobs = in_memory(extract_verified(names))
    ld = lambda n: torch.load(io.BytesIO(blobs[n]), weights_only=False)

    # data
    S = SR.setup()
    n_salt = len(S["salts"])
    ball = SR.dbl(subset(S["B"], S["use"]))
    res = json.loads((REPO / "paper2/synth_results.json").read_text())
    teacher = SY.build_teacher(res["teacher_scale"])
    rho = SY.CONDS["correlated"]; ci = list(SY.CONDS).index("correlated")
    X, Xv, Xt, Xo = SY.make_inputs(rho, 1000 + ci), SY.make_inputs(rho, 2000 + ci), SY.make_inputs(rho, 3000 + ci), SY.make_inputs(rho, 4000 + ci, independent=True)
    y0 = SY.predict(teacher, X); sig = 0.05 * y0.std()
    rng = np.random.default_rng(5000 + ci)
    y = y0 + sig * rng.standard_normal(SY.N); yv = SY.predict(teacher, Xv) + sig * rng.standard_normal(SY.N)
    yt_, yo_ = SY.predict(teacher, Xt), SY.predict(teacher, Xo)
    log(f"  synthetic correlated data regenerated (seeds 1001-5001); check: committed early data term of correlated_P0_s0 "
        f"{res['models']['correlated_P0_s0']['early_terms']['data']:.6e}")

    def kan_from(sd):
        k = KAN(len(DELTA_IN), [6, 1], n_salt)
        miss, unexp = k.load_state_dict({n: t.float() for n, t in sd.items()}, strict=False)
        assert not unexp and all(x.endswith(".grid") for x in miss)
        return k.double().eval()

    def syn_from(sd):
        n = SY.SynthKAN().double(); n.load_state_dict({k: v.double() for k, v in sd.items()}); return n.eval()

    def summarise(arm, key, flat, x, info, blocks, growth_names):
        p0, p1 = flat.full_at(flat.x0), flat.full_at(x)
        n0, n1 = block_norms(p0, {"g": growth_names})["g"], block_norms(p1, {"g": growth_names})["g"]
        mins = [l["min_eig"] / l["max_eig"] for l in info["logged"]]
        r = dict(arm=arm, key=key, certified=bool(info["cert"]["certified"]), crit=info["cert"]["crit"], grad=info["cert"]["grad"],
                 stop=info["stop"], iters=info["iters"], growth=n1 / n0, min_eig_rel=float(min(mins)), logged=info["logged"],
                 terms0=flat.terms(flat.x0), terms1=flat.terms(x))
        RES.setdefault(arm, {})[key] = r
        log(f"  {arm} {key:24s} {info['stop']:30s} it {info['iters']:3d} | certified {str(r['certified']):5s} grad {info['cert']['grad']:.1e} "
            f"| growth x{r['growth']:.3f} | min eig/max eig over logged iterates {r['min_eig_rel']:+.2e} | data {r['terms0']['data']:.4e} -> {r['terms1']['data']:.4e}")
        return r

    def cached_run(arm, key, make_flat, blocks, growth_names):
        flat = make_flat()
        obj, hit = SR.cached(f"freeze_{arm}_{key}.pt", lambda: to_cache(dict(zip(("x", "info"), run(flat, blocks)))))
        x = obj["x"].numpy() if torch.is_tensor(obj["x"]) else obj["x"]
        return summarise(arm, key, flat, x, obj["info"], blocks, growth_names)

    # ---------------- Arm A
    log("\n==================== ARM A: frozen layer 1 (and salt embedding), polish layer 2 only ====================")
    for i in range(6):
        net = kan_from(ld(f"delta_s{i}.pt"))
        full = {k: v for k, v in net.named_parameters()}
        cached_run("A-R", f"ind{i}", lambda: FlatFree(net, full, L2, real_parts(net, ball)), REAL_BLOCKS, L2)
    for lev in ("P0", "P1"):
        for s in range(8):
            net = syn_from(ld(f"synth_correlated_{lev}_s{s}.pt")["early"])
            full = {k: v for k, v in net.named_parameters()}
            cached_run(f"A-S-{lev}", f"s{s}", lambda: FlatFree(net, full, L2, syn_parts(net, X, y, lev)), SYN_BLOCKS, L2)

    # ---------------- Arm C
    log("\n==================== ARM C: one-layer control KANLayer [4,1], correlated synthetic data, P0 loss ====================")
    pen1 = lambda prm: SY.LAM * (D2(prm["layers.0.coef"]) ** 2).sum()
    for s in range(8):
        obj, _ = SR.cached(f"freeze_C_early_s{s}.pt", lambda: (lambda r: dict(early=r[0].state_dict(), info=r[1]))(train_early_generic(OneLayer, s, X, y, Xv, yv, pen1, WD)))
        net = OneLayer().double(); net.load_state_dict({k: v.double() for k, v in obj["early"].items()}); net.eval()
        full = {k: v for k, v in net.named_parameters()}
        r = cached_run("C", f"s{s}", lambda: FlatFree(net, full, list(full), one_parts(net, X, y)), {"L1 RBF": ["layers.0.coef"], "L1 SiLU": ["layers.0.base.weight"], "L1 bias": ["layers.0.base.bias"]}, list(full))
        r["early_info"] = obj["info"]

    # ---------------- Arm D
    log("\n==================== ARM D: correlated P0, polish-time L2 = 1e-4 * ||theta||^2 added to the P0 loss ====================")
    for s in range(8):
        net = syn_from(ld(f"synth_correlated_P0_s{s}.pt")["early"])
        full = {k: v for k, v in net.named_parameters()}
        l2 = lambda prm: 1e-4 * sum((v ** 2).sum() for v in prm.values())
        cached_run("D", f"s{s}", lambda: FlatFree(net, full, list(full), syn_parts(net, X, y, "P0"), extra=l2), SYN_BLOCKS, list(full))

    # ---------------- Arm E
    log("\n==================== ARM E: retrain with AdamW weight_decay = 0 (early-stopped state), against wd = 1e-4 ====================")
    pen0 = lambda prm: SY.cell_penalty(prm, "P0", 0.0)
    E = {"i": [], "ii": []}
    for s in range(8):
        obj, _ = SR.cached(f"freeze_E_wd0_syn_s{s}.pt", lambda: (lambda r: dict(early=r[0].state_dict(), info=r[1]))(train_early_generic(SY.SynthKAN, s, X, y, Xv, yv, pen0, 0.0)))
        n0 = syn_from(obj["early"]); n4 = syn_from(ld(f"synth_correlated_P0_s{s}.pt")["early"])
        row = {}
        for tag, n in (("wd0", n0), ("wd1e-4", n4)):
            row[tag] = dict(theta2=theta2(n.state_dict()), data=float(np.mean((SY.predict(n, X) - y) ** 2)),
                            rmse_in=float(np.sqrt(np.mean((SY.predict(n, Xt) - yt_) ** 2))), rmse_off=float(np.sqrt(np.mean((SY.predict(n, Xo) - yo_) ** 2))))
        row["best_epoch_wd0"] = obj["info"]["best_epoch"]; row["best_epoch_wd1e-4"] = res["models"][f"correlated_P0_s{s}"]["early"]["best_epoch"]
        E["i"].append(row)
        log(f"  E(i) s{s}: ||theta||^2 wd0 {row['wd0']['theta2']:.3f} vs wd1e-4 {row['wd1e-4']['theta2']:.3f} (x{row['wd0']['theta2'] / row['wd1e-4']['theta2']:.3f}); "
            f"data {row['wd0']['data']:.4e} / {row['wd1e-4']['data']:.4e}; in-RMSE {row['wd0']['rmse_in']:.4f} / {row['wd1e-4']['rmse_in']:.4f}; "
            f"off-RMSE {row['wd0']['rmse_off']:.4f} / {row['wd1e-4']['rmse_off']:.4f}; best epoch {row['best_epoch_wd0']} / {row['best_epoch_wd1e-4']}")
    det_s = SR.cached("freeze_E_det_syn_s0.pt", lambda: dict(early=train_early_generic(SY.SynthKAN, 0, X, y, Xv, yv, pen0, WD)[0].state_dict()))[0]["early"]
    ref = ld("synth_correlated_P0_s0.pt")["early"]
    det_syn = max(float((det_s[k].double() - ref[k].double()).abs().max()) for k in ref)
    log(f"  E(i) determinism check: seed 0 retrained with wd = 1e-4 vs committed early state: max |diff| {det_syn:.2e}")
    bA = subset(S["B"], S["folds"]["A"]["train"]); bAt = subset(S["B"], S["folds"]["A"]["test"])
    preds = {"wd0": [], "wd1e-4": []}
    for s in range(5):
        obj, _ = SR.cached(f"freeze_E_wd0_foldA_s{s}.pt", lambda: dict(early=kan_train_wd(n_salt, bA, 107, s, 0.0).state_dict()))
        row = {}
        for tag, sd in (("wd0", obj["early"]), ("wd1e-4", ld(f"scbase_A_s{s}.pt")["early"])):
            n = kan_from(sd).float()
            row[tag] = dict(theta2=theta2(n.state_dict()), data=float(np.mean((predict(n, delta_head, bA) - bA["y"].numpy()) ** 2)))
            preds[tag].append(predict(n, delta_head, bAt))
        E["ii"].append(row)
        log(f"  E(ii) s{s}: ||theta||^2 wd0 {row['wd0']['theta2']:.3f} vs wd1e-4 {row['wd1e-4']['theta2']:.3f} (x{row['wd0']['theta2'] / row['wd1e-4']['theta2']:.4f}); "
            f"train data {row['wd0']['data']:.4e} / {row['wd1e-4']['data']:.4e}")
    ens = {t: float(np.sqrt(np.mean((np.mean(preds[t], 0) - bAt["y"].numpy()) ** 2))) for t in preds}
    det_r = SR.cached("freeze_E_det_foldA_s0.pt", lambda: dict(early=kan_train_wd(n_salt, bA, 107, 0, WD).state_dict()))[0]["early"]
    refr = ld("scbase_A_s0.pt")["early"]
    det_real = max(float((det_r[k].double() - refr[k].double()).abs().max()) for k in refr if not k.endswith("grid"))
    log(f"  E(ii) fold A 5-seed ensemble RMSE: wd0 {ens['wd0']:.4f} vs wd1e-4 {ens['wd1e-4']:.4f} (reference 0.1420); "
        f"determinism check seed 0 wd = 1e-4 vs committed: max |diff| {det_real:.2e}")
    RES["E"] = dict(rows=E, ensemble=ens, det_syn=det_syn, det_real=det_real)

    # ---------------- Arm B (exploratory, slowest last)
    log("\n==================== ARM B (EXPLORATORY): frozen layer 2, polish layer 1 only (salt embedding frozen) ====================")
    for lev in ("P0", "P1"):
        for s in range(8):
            net = syn_from(ld(f"synth_correlated_{lev}_s{s}.pt")["early"])
            full = {k: v for k, v in net.named_parameters()}
            cached_run(f"B-S-{lev}", f"s{s}", lambda: FlatFree(net, full, L1, syn_parts(net, X, y, lev)), SYN_BLOCKS, L1)
    for i in range(6):
        net = kan_from(ld(f"delta_s{i}.pt"))
        full = {k: v for k, v in net.named_parameters()}
        cached_run("B-R", f"ind{i}", lambda: FlatFree(net, full, L1, real_parts(net, ball)), REAL_BLOCKS, L1)

    report()


def med(v):
    return float(np.median(v))


def report():
    log("\n==================== SUMMARY TABLES ====================")
    log("  arm/cell     n  certified  growth median (range)        min eig/max eig (worst logged)  stop reasons")
    cells = [k for k in RES if k != "E"]
    for c in cells:
        rr = list(RES[c].values())
        g = [r["growth"] for r in rr]
        stops = {}
        for r in rr:
            stops[r["stop"]] = stops.get(r["stop"], 0) + 1
        log(f"  {c:10s} {len(rr):2d}  {sum(r['certified'] for r in rr):2d}/{len(rr):<2d}     x{med(g):.3f} (x{min(g):.3f}-x{max(g):.3f})    "
            f"{min(r['min_eig_rel'] for r in rr):+.2e}                      {stops}")
    E = RES["E"]
    for sub, key in (("E(i) synthetic correlated P0", "i"), ("E(ii) real fold A", "ii")):
        ratio = [r["wd0"]["theta2"] / r["wd1e-4"]["theta2"] for r in E["rows"][key]]
        log(f"  {sub}: median ||theta||^2 ratio wd0/wd1e-4 x{med(ratio):.4f} (x{min(ratio):.4f}-x{max(ratio):.4f}); "
            f"median data {med([r['wd0']['data'] for r in E['rows'][key]]):.4e} vs {med([r['wd1e-4']['data'] for r in E['rows'][key]]):.4e}"
            + (f"; median in-RMSE {med([r['wd0']['rmse_in'] for r in E['rows'][key]]):.4f} vs {med([r['wd1e-4']['rmse_in'] for r in E['rows'][key]]):.4f}; "
               f"off-RMSE {med([r['wd0']['rmse_off'] for r in E['rows'][key]]):.4f} vs {med([r['wd1e-4']['rmse_off'] for r in E['rows'][key]]):.4f}" if key == "i" else
               f"; fold A ensemble RMSE {E['ensemble']['wd0']:.4f} vs {E['ensemble']['wd1e-4']:.4f}"))

    # verdicts
    log("\n==================== VERDICTS (pre-registered Z1-Z6) ====================")
    A = {c: list(RES[c].values()) for c in ("A-R", "A-S-P0", "A-S-P1")}
    need = {"A-R": 5, "A-S-P0": 7, "A-S-P1": 7}
    z1 = {c: sum(r["certified"] for r in rr) >= need[c] for c, rr in A.items()}
    z2 = {c: med([r["growth"] for r in rr]) <= 3 for c, rr in A.items()}
    z3 = all(r["min_eig_rel"] >= -1e-8 for rr in A.values() for r in rr)
    C = list(RES["C"].values())
    z4 = sum(r["certified"] for r in C) == 8 and med([r["growth"] for r in C]) <= 3
    Dd = list(RES["D"].values()); dc, dg = sum(r["certified"] for r in Dd), med([r["growth"] for r in Dd])
    z5 = "bounds the runaway" if dc >= 6 and dg <= 3 else "does not bound it" if dc <= 2 else "partial effect"
    z6 = {}
    for key in ("i", "ii"):
        rt = med([r["wd0"]["theta2"] / r["wd1e-4"]["theta2"] for r in E["rows"][key]])
        z6[key] = "decay materially bounded the early-stopped models" if rt >= 2 else "early stopping alone; decay incidental" if rt <= 1.25 and rt >= 1 / 1.25 else "partial effect"
    for c in A:
        log(f"  Z1 {c:7s} {'HELD' if z1[c] else 'FAILED':6s} certified {sum(r['certified'] for r in A[c])}/{len(A[c])} (needs >= {need[c]})")
    for c in A:
        log(f"  Z2 {c:7s} {'HELD' if z2[c] else 'FAILED':6s} median layer-2 growth x{med([r['growth'] for r in A[c]]):.3f}")
    log(f"  Z3         {'HELD' if z3 else 'FAILED':6s} worst min eig/max eig over all logged Arm A iterates {min(r['min_eig_rel'] for rr in A.values() for r in rr):+.2e}")
    log(f"  Z4         {'HELD' if z4 else 'FAILED':6s} Arm C certified {sum(r['certified'] for r in C)}/8, median growth x{med([r['growth'] for r in C]):.3f}")
    log(f"  Z5         branch: {z5} (certified {dc}/8, median growth x{dg:.3f})")
    log(f"  Z6 E(i)    branch: {z6['i']}")
    log(f"  Z6 E(ii)   branch: {z6['ii']}")
    allz1, allz2 = all(z1.values()), all(z2.values())
    if allz1 and allz2 and z4:
        br = "Z1, Z2, Z4 hold -> composition is necessary for the runaway; correlated inputs alone are not sufficient; the one-layer problem is well posed"
    elif not allz2 or not z4:
        br = "Z2 or Z4 fails -> a runaway without composition: the mechanism needs revision (stop and report)"
    elif not allz1 and allz2 and z3:
        br = "Z1 fails while Z2, Z3 hold -> convex but too ill-conditioned to certify in 500 iterations; mechanism unaffected"
    else:
        br = "no pre-registered branch matches exactly; reported as is"
    log(f"  INTERPRETATION: {br}")
    RES["verdicts"] = dict(Z1=z1, Z2=z2, Z3=z3, Z4=z4, Z5=z5, Z6=z6, branch=br)
    (HERE / "freeze_results.json").write_text(json.dumps(RES, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    figure()


def figure():
    cells = [k for k in RES if k not in ("E", "verdicts")]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for j, c in enumerate(cells):
        rr = list(RES[c].values())
        g = [r["growth"] for r in rr]
        axs[0].scatter(np.full(len(g), j) + np.random.default_rng(j).uniform(-0.15, 0.15, len(g)), g, s=12,
                       c=["#0072B2" if r["certified"] else "#D55E00" for r in rr])
        axs[0].plot([j - 0.25, j + 0.25], [med(g)] * 2, color="k")
    axs[0].axhline(3, color="grey", ls=":", lw=0.8)
    axs[0].set_yscale("log"); axs[0].set_xticks(range(len(cells))); axs[0].set_xticklabels(cells, fontsize=7, rotation=30)
    axs[0].set_ylabel("squared-norm growth of the polished block"); axs[0].set_title("growth (blue certified, vermillion not)")
    for c, col in (("A-R", "#000000"), ("A-S-P0", "#E69F00"), ("A-S-P1", "#56B4E9")):
        for r in RES[c].values():
            it = [l["it"] for l in r["logged"]]
            axs[1].plot(it, [max(l["min_eig"] / l["max_eig"], -1) for l in r["logged"]], color=col, lw=0.8, alpha=0.7)
    axs[1].axhline(-1e-8, color="grey", ls=":", lw=0.8); axs[1].set_yscale("symlog", linthresh=1e-12)
    axs[1].set_xlabel("iteration"); axs[1].set_ylabel("min eig / max eig"); axs[1].set_title("Arm A: Hessian spectrum (black R, orange P0, blue P1)")
    fig.tight_layout(); fig.savefig(FIG / "freeze_summary.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()

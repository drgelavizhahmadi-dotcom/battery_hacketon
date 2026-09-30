"""trust: trust-region Newton (scipy trust-krylov, exact Hessian-vector products) from the capped weights.
See trust_prereg.md. Run: python3 trust.py    Weights: gauge_weights/trust_* (not committed)   Figures: figs/trust/"""
import hashlib
import itertools
import json
import tarfile
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment, minimize
from torch.func import functional_call

from gauge import Gauge, align, corr, grid_inputs, parts, setup
from kan import KAN, SMOOTH, delta_head, predict, subset, to_batch
from fiber import affine_fits, residual_ss

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
ROOT = Path(__file__).parent
FIG = ROOT / "figs" / "trust"
ARCH = ROOT.parent / "battery_hacketon_weights"
W_DIR = ROOT / "gauge_weights"
MAXITER, WALL = 500, 6 * 3600
GRAD_TOL, REL_TOL, EIG_TOL = 1e-7, 1e-12, 1e-8
IND = list(range(6))
STU = [115, 104, 114]
RESULTS = {}


def log(*a):
    print(*a, flush=True)


def extract_verified(names):
    """Extract named files from the archives into a temp dir and verify SHA-256 against the manifest."""
    man = {}
    for line in (ROOT / "weights_manifest.txt").read_text().splitlines():
        parts_ = line.split()
        if line.startswith("gauge_weights/"):
            man[parts_[0]] = (parts_[1], parts_[4] if len(parts_) > 4 else "gauge_weights_2026-09-29.tar.gz")
    tmp = Path(tempfile.mkdtemp(prefix="trust_weights_"))
    out = {}
    for n in names:
        key = f"gauge_weights/{n}"
        sha, arch = man[key]
        with tarfile.open(ARCH / arch) as tf:
            tf.extract(tf.getmember(key), tmp)
        p = tmp / key
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != sha:
            raise SystemExit(f"SHA-256 MISMATCH for {key}: manifest {sha[:12]} vs archive {got[:12]}")
        out[n] = p
        log(f"  verified {key} ({arch}) sha256 {sha[:12]}")
    return out


class Flat:
    """The 609 loss-relevant parameters of a KAN as one float64 vector (unused salt embeddings fixed)."""

    def __init__(self, net, bd, used):
        self.net = net.double()
        self.used = torch.tensor(used)
        self.named = [(n, p.detach().clone()) for n, p in self.net.named_parameters()]
        self.meta, pieces = [], []
        for n, p in self.named:
            v = p[self.used] if n == "emb.weight" else p
            pieces.append(v.reshape(-1)); self.meta.append((n, v.shape, v.numel()))
        self.x0 = torch.cat(pieces).numpy().copy()
        self.full = dict(self.named)
        self.bd = bd
        off, self.slices = 0, {}
        for n, shp, k in self.meta:
            self.slices[n] = (off, off + k, shp); off += k
        self.n = off
        self._cache = None

    def params(self, v):
        prm = {}
        for n, shp, k in self.meta:
            a, b_, _ = self.slices[n]
            t = v[a:b_].reshape(shp)
            prm[n] = self.full[n].index_put((self.used,), t) if n == "emb.weight" else t
        return prm

    def loss_t(self, v):
        prm = self.params(v)
        pred = functional_call(self.net, prm, (self.bd["x"], self.bd["s"]))[:, 0]
        pen = sum(((c[..., 2:] - 2 * c[..., 1:-1] + c[..., :-2]) ** 2).sum() for c in [prm["layers.0.coef"], prm["layers.1.coef"]])
        return F.mse_loss(self.bd["x_co"] * pred, self.bd["y"]) + SMOOTH * pen

    def fun(self, x):
        if self._cache is not None and np.array_equal(self._cache[0], x):
            return self._cache[1], self._cache[2]
        v = torch.tensor(x, requires_grad=True)
        L = self.loss_t(v)
        (g,) = torch.autograd.grad(L, v)
        out = (float(L), g.numpy().copy())
        self._cache = (x.copy(), *out)
        return out

    def hessp(self, x, p):
        v = torch.tensor(x, requires_grad=True)
        (g,) = torch.autograd.grad(self.loss_t(v), v, create_graph=True)
        (hp,) = torch.autograd.grad(g @ torch.tensor(p), v)
        return hp.numpy().copy()

    def hessian(self, x):
        H = torch.autograd.functional.hessian(self.loss_t, torch.tensor(x)).numpy()
        return (H + H.T) / 2

    def to_kan(self, x, d, n_salt):
        K = KAN(d, [6, 1], n_salt).double()
        missing, unexpected = K.load_state_dict({n: t.detach() for n, t in self.params(torch.tensor(x)).items()}, strict=False)
        assert not unexpected and all(k.endswith(".grid") for k in missing), (missing, unexpected)
        return K.eval()

    def block_masks(self, d):
        """Index masks for layer-1 (continuous + bias), salt (used emb + emb-edge entries of layer 0), layer-2."""
        n = self.n
        L1, S, L2 = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
        a, b_, _ = self.slices["emb.weight"]; S[a:b_] = True
        a, b_, _ = self.slices["layers.0.base.bias"]; L1[a:b_] = True
        for name, per in [("layers.0.coef", 8), ("layers.0.base.weight", 1)]:
            a, b_, shp = self.slices[name]
            idx = np.arange(a, b_).reshape(shp[0], shp[1], -1) if per > 1 else np.arange(a, b_).reshape(shp[0], shp[1], 1)
            L1[idx[:, :d, :].ravel()] = True
            S[idx[:, d:, :].ravel()] = True
        for name in ["layers.1.coef", "layers.1.base.weight", "layers.1.base.bias"]:
            a, b_, _ = self.slices[name]; L2[a:b_] = True
        assert (L1.astype(int) + S + L2 == 1).all()
        return dict(layer1=L1, salt=S, layer2=L2)


def run_model(tag, flat):
    x0 = flat.x0.copy()
    L0, g0 = flat.fun(x0)
    ev0 = np.linalg.eigvalsh(flat.hessian(x0))
    # Q5 diagnostics: hessp vs full Hessian and vs finite difference of the gradient
    H0 = flat.hessian(x0)
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(3):
        v = rng.normal(size=flat.n); v /= np.linalg.norm(v)
        hp = flat.hessp(x0, v)
        fd = (flat.fun(x0 + 1e-6 * v)[1] - flat.fun(x0 - 1e-6 * v)[1]) / 2e-6
        errs.append((np.linalg.norm(hp - H0 @ v) / np.linalg.norm(hp), np.linalg.norm(hp - fd) / np.linalg.norm(hp)))
    hist = dict(loss=[L0], grad=[float(np.abs(g0).max())], step=[0.0], t=[0.0])
    state = dict(prev=x0.copy(), t0=time.time(), stop="scipy termination", eig_checks=[])

    def criteria_ab():
        k = len(hist["loss"]) - 1
        if k < 10:
            return False
        rel = abs(hist["loss"][k - 10] - hist["loss"][k]) / hist["loss"][k - 10]
        return hist["grad"][k] < GRAD_TOL and rel < REL_TOL

    def callback(intermediate_result):
        x = intermediate_result.x
        L, g = flat.fun(x)
        hist["loss"].append(L); hist["grad"].append(float(np.abs(g).max()))
        hist["step"].append(float(np.linalg.norm(x - state["prev"]))); hist["t"].append(time.time() - state["t0"])
        state["prev"] = x.copy()
        if criteria_ab():
            ev = np.linalg.eigvalsh(flat.hessian(x))
            ok = ev[0] >= -EIG_TOL * ev[-1]
            state["eig_checks"].append((len(hist["loss"]) - 1, float(ev[0]), float(ev[-1]), bool(ok)))
            if ok:
                state["stop"] = "converged"
                raise StopIteration
        if time.time() - state["t0"] > WALL:
            state["stop"] = "6 h wall-clock cap"
            raise StopIteration

    res = minimize(flat.fun, x0, jac=True, hessp=flat.hessp, method="trust-krylov", callback=callback,
                   options=dict(maxiter=MAXITER, gtol=1e-30))
    if state["stop"] == "scipy termination" and len(hist["loss"]) - 1 >= MAXITER:
        state["stop"] = "iteration cap (500)"
    x = state["prev"]
    L, g = flat.fun(x)
    ev = np.linalg.eigvalsh(flat.hessian(x))
    k = len(hist["loss"]) - 1
    rel10 = abs(hist["loss"][max(0, k - 10)] - hist["loss"][k]) / hist["loss"][max(0, k - 10)]
    conv = bool(np.abs(g).max() < GRAD_TOL and rel10 < REL_TOL and k >= 10 and ev[0] >= -EIG_TOL * ev[-1])
    rel50 = (hist["loss"][max(0, k - 50)] - hist["loss"][k]) / hist["loss"][max(0, k - 50)]
    maxstep50 = max(hist["step"][max(1, k - 49):k + 1]) if k >= 1 else 0.0
    neg = ev[0] < -EIG_TOL * ev[-1]
    if conv:
        cls = "CONVERGED"
    elif rel50 > 1e-10 and neg:
        cls = "DESCENDING-SADDLE"
    elif abs(rel50) < 1e-12 and np.abs(g).max() >= GRAD_TOL and maxstep50 < 1e-10:
        cls = "COLLAPSED"
    else:
        cls = "OTHER"
    info = dict(tag=tag, iters=k, stop=state["stop"], scipy_message=str(res.message), wall_s=round(hist["t"][-1], 1),
                loss0=L0, loss=L, loss_ratio=L / L0, grad0=hist["grad"][0], grad=float(np.abs(g).max()),
                rel10=rel10, rel50=rel50, maxstep50=maxstep50, neg0=int((ev0 < -EIG_TOL * ev0[-1]).sum()),
                neg=int((ev < -EIG_TOL * ev[-1]).sum()), mineig0=float(ev0[0]), maxeig0=float(ev0[-1]),
                mineig=float(ev[0]), maxeig=float(ev[-1]), converged=conv, cls=cls, eig_checks=state["eig_checks"],
                hessp_err=[list(map(float, e)) for e in errs], hist=hist)
    log(f"  {tag}: {info['stop']} after {k} iters ({info['wall_s']} s); loss {L0:.10e} -> {L:.10e} (ratio {L / L0:.6f}); "
        f"max|grad| {info['grad0']:.2e} -> {info['grad']:.2e}; rel10 {rel10:.1e}, rel50 {rel50:.1e}, max step (last 50) {maxstep50:.1e}; "
        f"neg eigs {info['neg0']} -> {info['neg']} (min {ev[0]:.2e}, max {ev[-1]:.2e}); class {cls}")
    log(f"      hessp rel error vs full Hessian {max(e[0] for e in errs):.1e}, vs finite-difference gradient {max(e[1] for e in errs):.1e}; "
        f"scipy: {res.message}")
    return x, info


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    tr, te, te_anc, D, _ = setup()
    d = D["d_num"]
    m = D["train"]
    Ztr, str_ = D["Z"][m], D["s"][m]
    b = subset(D["batch"], m)
    bd = {k: (v.double() if v.is_floating_point() else v) for k, v in b.items()}
    used = np.unique(str_).tolist()
    log("DISCREPANCIES / deviations (files win; see trust_prereg.md):")
    log("  Q6 DEVIATION: scipy 1.13 trust-krylov exposes only (x, fun) to the callback, not the trust radius ->")
    log("     COLLAPSED uses the max step length ||x_k - x_{k-1}|| over the last 50 iterations < 1e-10 as a proxy")
    log("  Q2 609 parameters; unused salt embeddings fixed")

    log("\n==================== inputs: extract from archives and verify SHA-256 ====================")
    files = extract_verified([f"fiber2_ind_{i}.pt" for i in IND] + [f"fiber2_stu_{s}.pt" for s in STU] + ["delta_s0.pt"])

    def load(p):
        k = KAN(d, [6, 1], D["n_salt"]); k.load_state_dict(torch.load(p)); return k.eval()

    T = load(files["delta_s0.pt"])
    yT = predict(T, delta_head, to_batch(Ztr, str_, x_co=b["x_co"].numpy()))
    fiber2_meta = {**json.loads((W_DIR / "fiber2_ind_meta.json").read_text()), **{f"s{k}": v for k, v in json.loads((W_DIR / "fiber2_stu_meta.json").read_text()).items()}}

    log("\n==================== trust-region Newton ====================")
    finals, infos = {}, {}
    for tag, fname, target, mkey in [(f"ind {i}", f"fiber2_ind_{i}.pt", bd["y"], str(i)) for i in IND] + \
                                     [(f"stu {s}", f"fiber2_stu_{s}.pt", torch.tensor(yT, dtype=torch.float64), f"s{s}") for s in STU]:
        bb = dict(bd); bb["y"] = target
        flat = Flat(load(files[fname]), bb, used)
        log(f"  {tag}: {flat.n} parameters; capped loss recomputed {flat.fun(flat.x0)[0]:.10e} vs fiber2 record {fiber2_meta[mkey]['loss']:.10e}")
        x, info = run_model(tag, flat)
        finals[tag], infos[tag] = (flat, x), info
        torch.save({n: t.detach().clone() for n, t in flat.params(torch.tensor(x)).items()}, W_DIR / f"trust_{tag.replace(' ', '_')}.pt")

    conv_ind = [i for i in IND if infos[f"ind {i}"]["converged"]]
    conv_stu = [s for s in STU if infos[f"stu {s}"]["converged"]]
    log(f"\n  converged: baseline {len(conv_ind)}/6 {conv_ind}; students {len(conv_stu)}/3 {conv_stu}")
    log("  classes: " + ", ".join(f"{t} {i['cls']}" for t, i in infos.items()))

    # ---------------- after convergence
    log("\n==================== after convergence (converged models only) ====================")
    sd_y = bd["y"].std().item()
    out = {}
    if len(conv_ind) >= 2:
        pairs = list(itertools.combinations(conv_ind, 2))
        dl = {i: predict(finals[f"ind {i}"][0].to_kan(finals[f"ind {i}"][1], d, D["n_salt"]).float(), delta_head, b) for i in conv_ind}
        dc = {i: predict(load(files[f"fiber2_ind_{i}.pt"]), delta_head, b) for i in conv_ind}
        dAB = [float(np.sqrt(np.mean((dl[A] - dl[B]) ** 2)) / sd_y) for A, B in pairs]
        dAB0 = [float(np.sqrt(np.mean((dc[A] - dc[B]) ** 2)) / sd_y) for A, B in pairs]
        out["dAB"], out["dAB_capped"] = dAB, dAB0
        log(f"  baseline d_AB on delta over {len(pairs)} converged pairs: median {np.median(dAB):.4f} (capped, same pairs {np.median(dAB0):.4f})")
    if conv_stu:
        sweeps = grid_inputs(D)
        gT = Gauge(T, Ztr, str_, d)
        varT = gT.fixed(parts(T, Ztr, str_, d)[0]).var(1)
        active = varT >= 0.05 * varT.max(0, keepdims=True)
        edgeT = [gT.fixed(parts(T, *sweeps[i], d)[0])[i] for i in range(d)]
        stu_rows = []
        for s in STU:
            states = [("capped", load(files[f"fiber2_stu_{s}.pt"]))]
            if s in conv_stu:
                fl, x = finals[f"stu {s}"]
                states.append(("converged", fl.to_kan(x, d, D["n_salt"]).float()))
            for state, net in states:
                g = Gauge(net, Ztr, str_, d)
                um = np.abs(align(gT, g))
                pe = [abs(corr(g.fixed(parts(net, *sweeps[i], d)[0])[i][:, j], edgeT[i][:, j])) for i in range(d) for j in range(6) if active[i, j]]
                eA = np.stack([parts(T, *sweeps[i], d)[0][i] for i in range(d)])
                eB = np.stack([parts(net, *sweeps[i], d)[0][i] for i in range(d)])
                mA, mB = parts(T, Ztr, str_, d)[0].mean(1), parts(net, Ztr, str_, d)[0].mean(1)
                rii = []
                for j in range(6):
                    kk = g.perm[j]
                    A_, B_ = eA[:, :, j], eB[:, :, kk]
                    ss = residual_ss(A_, B_, mA[:d, j][:, None], mB[:d, kk][:, None], affine_fits(A_, B_))
                    rii.append(float(np.sqrt(ss[1] / ((B_ - mB[:d, kk][:, None]) ** 2).sum(0).mean())))
                pc = corr(predict(net, delta_head, b), yT)
                stu_rows.append(dict(student=s, state=state, pred_corr=pc, unit=float(np.median(um)), units=um.tolist(),
                                     edge=float(np.median(pe)), rii=float(np.median(rii))))
                log(f"  student {s} [{state}]: delta corr with teacher {pc:.6f}; unit match median {np.median(um):.3f} {np.round(um, 3).tolist()}; "
                    f"edge |corr| median {np.median(pe):.3f}; shared-affine residual R_ii median {np.median(rii):.3f}")
        out["students"] = stu_rows
    zm = {}
    for tag, (flat, x) in finals.items():
        if not infos[tag]["converged"]:
            continue
        ev, V = np.linalg.eigh(flat.hessian(x))
        z = np.abs(ev) < EIG_TOL * ev[-1]
        masks = flat.block_masks(d)
        share = {k: float((V[mk][:, z] ** 2).sum() / max(z.sum(), 1)) for k, mk in masks.items()}
        zm[tag] = dict(n_zero=int(z.sum()), share=share)
        log(f"  {tag}: zero modes (|eig| < 1e-8 x max) {int(z.sum())}; energy share layer-1 {share['layer1']:.3f}, salt {share['salt']:.3f}, layer-2 {share['layer2']:.3f}")
    out["zero_modes"] = zm

    # ---------------- verdicts
    v = {"T1": ("held" if len(conv_ind) >= 4 else "failed") + f" ({len(conv_ind)}/6)",
         "T2": ("held" if len(conv_stu) >= 2 else "failed") + f" ({len(conv_stu)}/3)"}
    if conv_ind:
        rat = {i: infos[f"ind {i}"]["loss_ratio"] for i in conv_ind}
        v["T3"] = ("held" if all(r <= 0.95 for r in rat.values()) else "failed") + " (" + ", ".join(f"{i}: {r:.4f}" for i, r in rat.items()) + ")"
    else:
        v["T3"] = "inconclusive (no baseline seed converged)"
    v["T4"] = (("held" if np.median(out["dAB"]) <= 0.5 * np.median(out["dAB_capped"]) else "failed") +
               f" (median {np.median(out['dAB']):.4f} vs 0.5 x {np.median(out['dAB_capped']):.4f})") if len(conv_ind) >= 2 else \
        f"inconclusive ({len(conv_ind)} baseline seed(s) converged)"
    if conv_stu:
        um = [u for r in out["students"] if r["state"] == "converged" for u in r["units"]]
        v["T5"] = ("held" if np.median(um) >= 0.8 else "failed") + f" (median {np.median(um):.3f})"
    else:
        v["T5"] = "inconclusive (no student converged)"
    log("\nVERDICTS: " + "; ".join(f"{k}: {x}" for k, x in v.items()))
    RESULTS.update(models={t: {k: val for k, val in i.items() if k != "hist"} for t, i in infos.items()},
                   after=out, verdicts=v, conv_ind=conv_ind, conv_stu=conv_stu)
    (ROOT / "trust_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    for t, i in infos.items():
        h = i["hist"]
        axs[0].semilogy(np.array(h["loss"]) / h["loss"][0], label=t, lw=0.9)
        axs[1].semilogy(h["grad"], label=t, lw=0.9)
    axs[0].set_title("loss / capped loss"); axs[0].set_xlabel("trust-krylov iteration"); axs[0].legend(fontsize=7)
    axs[1].axhline(GRAD_TOL, color="r", ls="--"); axs[1].set_title("max |grad|"); axs[1].set_xlabel("iteration")
    fig.tight_layout(); fig.savefig(FIG / "trust_curves.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()

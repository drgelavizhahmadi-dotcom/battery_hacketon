"""stage2: trust-exact optimiser check (A), runaway diagnostic (B), coercivity intervention (C). See stage2_prereg.md.
Run: python3 stage2.py    Weights: gauge_weights/stage2_* (not committed)   Figures: figs/stage2/"""
import hashlib
import io
import itertools
import json
import shutil
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize

from fiber import affine_fits, residual_ss
from gauge import Gauge, align, corr, grid_inputs, parts, setup
from kan import KAN, SMOOTH, delta_head, predict, subset, to_batch
from trust import EIG_TOL, GRAD_TOL, IND, REL_TOL, STU, Flat, extract_verified

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
ROOT = Path(__file__).parent
FIG = ROOT / "figs" / "stage2"
W_DIR = ROOT / "gauge_weights"
MAXITER, WALL = 500, 2 * 3600
LAMBDAS = [1e-6, 1e-5]
BLOCKS = {"layer-1 RBF": ["layers.0.coef"], "layer-2 RBF": ["layers.1.coef"],
          "SiLU base": ["layers.0.base.weight", "layers.1.base.weight"],
          "biases": ["layers.0.base.bias", "layers.1.base.bias"], "salt embedding": ["emb.weight"]}
RESULTS = {}


def log(*a):
    print(*a, flush=True)


class FlatL2(Flat):
    def __init__(self, net, bd, used, lam=0.0):
        super().__init__(net, bd, used)
        self.lam = lam

    def loss_t(self, v):
        return Flat.loss_t(self, v) + self.lam * (v ** 2).sum()

    def base_loss(self, x):
        with torch.no_grad():
            return float(Flat.loss_t(self, torch.tensor(x)))


def in_memory(paths):
    """Read verified inputs into memory, re-check SHA-256 against the manifest, delete the extraction dir.
    Nothing is re-read from $TMPDIR later (macOS's nightly cleanup deleted lazily re-read files: restart 2)."""
    man = {l.split()[0].split("/")[-1]: l.split()[1] for l in (ROOT / "weights_manifest.txt").read_text().splitlines()
           if l.startswith("gauge_weights/")}
    blobs = {}
    for n, p in paths.items():
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != man[n]:
            raise SystemExit(f"SHA-256 MISMATCH (in-memory re-check) for {n}")
        blobs[n] = data
    for root in {p.parents[1] for p in paths.values()}:
        shutil.rmtree(root)
    log(f"  {len(blobs)} inputs held in memory (SHA-256 re-checked from memory); extraction directory removed")
    return blobs


def run_exact(tag, flat):
    x0 = flat.x0.copy()
    L0, g0 = flat.fun(x0)
    ev0 = np.linalg.eigvalsh(flat.hessian(x0))
    hist = dict(loss=[L0], grad=[float(np.abs(g0).max())], step=[0.0], t=[0.0])
    st = dict(prev=x0.copy(), t0=time.time(), stop="scipy termination", eig_checks=[])

    def callback(intermediate_result):  # this exact name makes scipy pass the OptimizeResult, not bare x
        x = intermediate_result.x
        L, g = flat.fun(x)
        hist["loss"].append(L); hist["grad"].append(float(np.abs(g).max()))
        hist["step"].append(float(np.linalg.norm(x - st["prev"]))); hist["t"].append(time.time() - st["t0"])
        st["prev"] = x.copy()
        k = len(hist["loss"]) - 1
        if k >= 10 and hist["grad"][k] < GRAD_TOL and abs(hist["loss"][k - 10] - L) / hist["loss"][k - 10] < REL_TOL:
            ev = np.linalg.eigvalsh(flat.hessian(x))
            ok = ev[0] >= -EIG_TOL * ev[-1]
            st["eig_checks"].append((k, float(ev[0]), float(ev[-1]), bool(ok)))
            if ok:
                st["stop"] = "converged"
                raise StopIteration
        if time.time() - st["t0"] > WALL:
            st["stop"] = "2 h wall-clock cap"
            raise StopIteration

    res = minimize(flat.fun, x0, jac=True, hess=flat.hessian, method="trust-exact", callback=callback,
                   options=dict(maxiter=MAXITER, gtol=1e-30))
    k = len(hist["loss"]) - 1
    if st["stop"] == "scipy termination" and k >= MAXITER:
        st["stop"] = "iteration cap (500)"
    x = st["prev"]
    L, g = flat.fun(x)
    ev = np.linalg.eigvalsh(flat.hessian(x))
    rel10 = abs(hist["loss"][max(0, k - 10)] - L) / hist["loss"][max(0, k - 10)]
    conv = bool(k >= 10 and np.abs(g).max() < GRAD_TOL and rel10 < REL_TOL and ev[0] >= -EIG_TOL * ev[-1])
    steps = np.array(hist["step"][1:]) if k else np.array([0.0])
    info = dict(tag=tag, iters=k, stop=st["stop"], message=str(res.message), wall_s=round(hist["t"][-1], 1),
                loss0=L0, loss=L, loss_ratio=L / L0, base_loss=flat.base_loss(x), grad0=hist["grad"][0], grad=float(np.abs(g).max()),
                rel10=rel10, neg0=int((ev0 < -EIG_TOL * ev0[-1]).sum()), neg=int((ev < -EIG_TOL * ev[-1]).sum()),
                mineig=float(ev[0]), maxeig=float(ev[-1]), converged=conv, step_median=float(np.median(steps)),
                step_max=float(steps.max()), n_rejected=int((steps == 0).sum()), eig_checks=st["eig_checks"])
    return x, info, hist


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    tr, te, te_anc, D, _ = setup()
    d = D["d_num"]
    m = D["train"]
    Ztr, str_ = D["Z"][m], D["s"][m]
    b = subset(D["batch"], m)
    bd = {k: (v.double() if v.is_floating_point() else v) for k, v in b.items()}
    used = np.unique(str_).tolist()
    log("DISCREPANCIES / choices (files win; see stage2_prereg.md):")
    log("  S2 students' AdamW-stage weights (control3_R) were trained 530-2680 epochs, not 107; students have no 'fiber' state")
    log("  S4 smoothness penalty acts on layers.0.coef and layers.1.coef only; no weight decay in any loss (AdamW's was decoupled)")
    log("  S5 scipy trust-exact callback exposes only (x, fun): step length = ||x_k - x_{k-1}||, 0 on rejected steps")
    log("  S6 Part C convergence criteria apply to the coercive loss (incl. the L2 term)")

    log("\n==================== inputs (SHA-256 verified) ====================")
    names = ([f"delta_s{i}.pt" for i in IND] + [f"fiber_polished_s{i}.pt" for i in IND] + [f"fiber2_ind_{i}.pt" for i in IND] +
             [f"trust_ind_{i}.pt" for i in IND] + [f"control3_R_s{s}.pt" for s in STU] + [f"fiber2_stu_{s}.pt" for s in STU] +
             [f"trust_stu_{s}.pt" for s in STU])
    files = in_memory(extract_verified(names))

    def load(p):
        k = KAN(d, [6, 1], D["n_salt"])
        miss, unexp = k.load_state_dict({n: t.float() for n, t in torch.load(io.BytesIO(p)).items()}, strict=False)
        assert not unexp and all(x.endswith(".grid") for x in miss), (miss, unexp)
        return k.eval()

    T = load(files["delta_s0.pt"])
    yT = torch.tensor(predict(T, delta_head, to_batch(Ztr, str_, x_co=b["x_co"].numpy())), dtype=torch.float64)
    models = [(f"ind {i}", f"fiber2_ind_{i}.pt", bd["y"]) for i in IND] + [(f"stu {s}", f"fiber2_stu_{s}.pt", yT) for s in STU]

    # S4 check: the flat penalty equals kan.py's smoothness()
    fl = Flat(load(files["fiber2_ind_0.pt"]), bd, used)
    with torch.no_grad():
        prm = fl.params(torch.tensor(fl.x0))
        pen_flat = sum(((c[..., 2:] - 2 * c[..., 1:-1] + c[..., :-2]) ** 2).sum() for c in [prm["layers.0.coef"], prm["layers.1.coef"]])
        pen_kan = load(files["fiber2_ind_0.pt"]).double().smoothness()
    log(f"  S4 check: flat penalty {float(pen_flat):.10e} vs kan.smoothness() {float(pen_kan):.10e}")

    # ---------------- Parts A and C
    runs = [("A", 0.0)] + [(f"C{lam:g}", lam) for lam in LAMBDAS]
    finals, infos, hists = {}, {}, {}
    for part, lam in runs:
        log(f"\n==================== PART {part[0]}{'' if part == 'A' else f' (lambda = {lam:g})'}: trust-exact ====================")
        for tag, fname, y in models:
            key = f"{part}|{tag}"
            bb = dict(bd); bb["y"] = y
            flat = FlatL2(load(files[fname]), bb, used, lam)
            cache = W_DIR / f"stage2_{part}_{tag.replace(' ', '_')}.pt"
            meta_p = W_DIR / f"stage2_{part}_{tag.replace(' ', '_')}.json"
            cached = cache.exists() and meta_p.exists()
            if cached:
                x = torch.load(cache).numpy(); info, hist = json.loads(meta_p.read_text()).values()
            else:
                x, info, hist = run_exact(tag, flat)
                info["capped_loss"] = Flat.fun(Flat(load(files[fname]), bb, used), flat.x0)[0]
                info["base_ratio"] = info["base_loss"] / info["capped_loss"]
                torch.save(torch.tensor(x), cache)
                meta_p.write_text(json.dumps(dict(info=info, hist=hist), default=float))
            finals[key], infos[key], hists[key] = (flat, x), info, hist
            log(f"  {tag}: {info['stop']} after {info['iters']} iters ({info['wall_s']} s) | scipy: {info['message']}" + ("  [cached]" if cached else ""))
            log(f"      loss ratio {info['loss_ratio']:.6f}" + ("" if part == "A" else f", without L2 {info['base_ratio']:.6f}") +
                f"; max|grad| {info['grad0']:.2e} -> {info['grad']:.2e}; neg eigs {info['neg0']} -> {info['neg']} "
                f"(min {info['mineig']:.2e}, max {info['maxeig']:.2e}); steps median {info['step_median']:.2e}, max {info['step_max']:.2e}, "
                f"rejected {info['n_rejected']}; converged {info['converged']}")

    conv = {part: [t for t, _, _ in models if infos[f"{part}|{t}"]["converged"]] for part, _ in runs}
    log("\n  converged: " + "; ".join(f"{p}: {len(c)}/9 {c}" for p, c in conv.items()))

    # ---------------- Part B
    log("\n==================== PART B: runaway diagnostic (block norms) ====================")
    state_files = {}
    for i in IND:
        state_files[f"ind {i}"] = [("AdamW-107", f"delta_s{i}.pt"), ("fiber", f"fiber_polished_s{i}.pt"),
                                   ("fiber2 capped", f"fiber2_ind_{i}.pt"), ("trust final", f"trust_ind_{i}.pt")]
    for s in STU:
        state_files[f"stu {s}"] = [("AdamW-stage (530-2680 ep)", f"control3_R_s{s}.pt"), ("fiber", None),
                                   ("fiber2 capped", f"fiber2_stu_{s}.pt"), ("trust final", f"trust_stu_{s}.pt")]

    def norms(K):
        sd = dict(K.named_parameters())
        out = {}
        for blk, keys in BLOCKS.items():
            v = torch.cat([(sd[k][used] if k == "emb.weight" else sd[k]).detach().reshape(-1) for k in keys])
            out[blk] = float(v.norm())
        return out

    partB = {}
    hdr = "  " + f"{'model':8s} {'state':26s} " + " ".join(f"{blk:>15s}" for blk in BLOCKS)
    log(hdr)
    for tag, _, _ in models:
        partB[tag] = {}
        flat_a, xa = finals[f"A|{tag}"]
        rows = [(st, load(files[f]) if f else None) for st, f in state_files[tag]] + [("Part A final", flat_a.to_kan(xa, d, D["n_salt"]).float())]
        for st, K in rows:
            if K is None:
                log(f"  {tag:8s} {st:26s} " + " ".join(f"{'N/A':>15s}" for _ in BLOCKS))
                continue
            nm = norms(K); partB[tag][st] = nm
            log(f"  {tag:8s} {st:26s} " + " ".join(f"{nm[blk]:15.4f}" for blk in BLOCKS))
        dx = xa - flat_a.x0
        tot = float((dx ** 2).sum())
        share = {}
        for blk, keys in BLOCKS.items():
            sq = sum(float((dx[flat_a.slices[k][0]:flat_a.slices[k][1]] ** 2).sum()) for k in keys)
            share[blk] = sq / tot if tot > 0 else 0.0
        partB[tag]["share of (Part A final - capped)"] = share
        log(f"  {tag:8s} {'share of ||A - capped||^2':26s} " + " ".join(f"{share[blk]:15.3f}" for blk in BLOCKS) + f"   (||delta|| = {np.sqrt(tot):.3e})")

    # ---------------- after convergence (Part C)
    log("\n==================== Part C converged models ====================")
    sd_y = bd["y"].std().item()
    after = {}
    sweeps = grid_inputs(D)
    gT = Gauge(T, Ztr, str_, d)
    varT = gT.fixed(parts(T, Ztr, str_, d)[0]).var(1)
    active = varT >= 0.05 * varT.max(0, keepdims=True)
    edgeT = [gT.fixed(parts(T, *sweeps[i], d)[0])[i] for i in range(d)]
    for part, lam in runs[1:]:
        ci = [int(t.split()[1]) for t in conv[part] if t.startswith("ind")]
        cs = [int(t.split()[1]) for t in conv[part] if t.startswith("stu")]
        a = dict(conv_ind=ci, conv_stu=cs)
        if len(ci) >= 2:
            pairs = list(itertools.combinations(ci, 2))
            dl = {i: predict(finals[f"{part}|ind {i}"][0].to_kan(finals[f"{part}|ind {i}"][1], d, D["n_salt"]).float(), delta_head, b) for i in ci}
            dc = {i: predict(load(files[f"fiber2_ind_{i}.pt"]), delta_head, b) for i in ci}
            a["dAB"] = [float(np.sqrt(np.mean((dl[A] - dl[B]) ** 2)) / sd_y) for A, B in pairs]
            a["dAB_capped"] = [float(np.sqrt(np.mean((dc[A] - dc[B]) ** 2)) / sd_y) for A, B in pairs]
            log(f"  {part}: d_AB over {len(pairs)} converged pairs median {np.median(a['dAB']):.4f} (capped, same pairs {np.median(a['dAB_capped']):.4f})")
        rows = []
        for s in cs:
            fl_, x = finals[f"{part}|stu {s}"]
            net = fl_.to_kan(x, d, D["n_salt"]).float()
            g = Gauge(net, Ztr, str_, d)
            um = np.abs(align(gT, g))
            pe = [abs(corr(g.fixed(parts(net, *sweeps[i], d)[0])[i][:, j], edgeT[i][:, j])) for i in range(d) for j in range(6) if active[i, j]]
            pc = corr(predict(net, delta_head, b), yT.numpy())
            rows.append(dict(student=s, pred_corr=pc, unit=float(np.median(um)), units=um.tolist(), edge=float(np.median(pe))))
            log(f"  {part} student {s}: delta corr {pc:.6f}; unit match median {np.median(um):.3f} {np.round(um, 3).tolist()}; edge |corr| median {np.median(pe):.3f}")
        a["students"] = rows
        after[part] = a
        if not ci and not cs:
            log(f"  {part}: no converged models")

    # ---------------- verdicts
    nA = sum(t.startswith("ind") for t in conv["A"])
    c5 = conv["C1e-05"]; c6 = conv["C1e-06"]
    n5i, n5s = sum(t.startswith("ind") for t in c5), sum(t.startswith("stu") for t in c5)
    v = {"U1": ("held" if nA >= 4 else "failed") + f" ({nA}/6 baseline)",
         "U2": ("held" if n5i >= 4 and n5s >= 2 else "failed") + f" ({n5i}/6 baseline, {n5s}/3 students)",
         "U3": ("held" if len(c5) > len(c6) else "failed") + f" ({len(c5)} at 1e-5 vs {len(c6)} at 1e-6)"}
    cr = [(p, t, infos[f"{p}|{t}"]["base_ratio"]) for p in ["C1e-06", "C1e-05"] for t in conv[p]]
    v["U4"] = ("inconclusive (no Part C model converged)" if not cr else
               ("held" if all(r <= 1.05 for *_, r in cr) else "failed") + " (" + ", ".join(f"{p} {t}: {r:.4f}" for p, t, r in cr) + ")")
    a5 = after.get("C1e-05", {})
    v["U5"] = (("held" if np.median(a5["dAB"]) <= 0.5 * np.median(a5["dAB_capped"]) else "failed") +
               f" (median {np.median(a5['dAB']):.4f} vs 0.5 x {np.median(a5['dAB_capped']):.4f})") if "dAB" in a5 else \
        f"inconclusive ({n5i} baseline seed(s) converged at 1e-5)"
    log("\nVERDICTS: " + "; ".join(f"{k}: {x}" for k, x in v.items()))
    RESULTS.update(infos=infos, conv=conv, partB=partB, after=after, verdicts=v)
    (ROOT / "stage2_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))

    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, (part, _) in zip(axs, runs):
        for tag, _, _ in models:
            h = hists[f"{part}|{tag}"]
            ax.semilogy(h["grad"], lw=0.9, label=tag)
        ax.axhline(GRAD_TOL, color="r", ls="--"); ax.set_title(f"{part}: max|grad|"); ax.set_xlabel("trust-exact iteration")
    axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "stage2_grad.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()

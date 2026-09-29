"""fiber2: certified convergence, on-fiber test on delta, and teacher-known fiber analysis (see fiber2_prereg.md).
Run: python3 fiber2.py     Weights: gauge_weights/fiber2_* (not committed)   Figures: figs/fiber2/"""
import copy
import itertools
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares, linear_sum_assignment

from fiber import affine_fits, psi_unit, residual_ss
from gauge import W_DIR, corr, corr_matrix, grid_inputs, knn_density, parts, setup
from kan import KAN, SMOOTH, delta_head, predict, subset, to_batch

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
ROOT = Path(__file__).parent
FIG = ROOT / "figs" / "fiber2"
IND_SEEDS = list(range(6))
STUDENTS = list(range(100, 120))
STEP, CAP, GRAD_TOL, REL_TOL = 50, 20000, 1e-7, 1e-12
DEF_A, DEF_B = 0.05, 0.25
GOOD_MATCH, POOR_MATCH = 0.9, 0.8
RESULTS = {}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ Step 0
def certify(net, b):
    net = copy.deepcopy(net).double()
    bd = {k: (v.double() if v.is_floating_point() else v) for k, v in b.items()}

    def loss():
        return F.mse_loss(delta_head(net, bd), bd["y"]) + SMOOTH * net.smoothness()

    def make():
        return torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=STEP, history_size=100, line_search_fn="strong_wolfe",
                                 tolerance_grad=1e-15, tolerance_change=1e-20)

    opt = make()

    def closure():
        opt.zero_grad()
        L = loss()
        L.backward()
        return L

    def grad_max():
        net.zero_grad()
        L = loss()
        L.backward()
        return float(L), max(float(p.grad.abs().max()) for p in net.parameters())

    L_prev, g = grad_max()
    iters, status, just_reset, trace, rel = 0, "cap (20000)", False, [(0, L_prev, g)], float("nan")
    while iters < CAP:
        opt.param_groups[0]["max_iter"] = min(STEP, CAP - iters)
        before = opt.state[opt._params[0]].get("n_iter", 0)
        opt.step(closure)
        taken = opt.state[opt._params[0]].get("n_iter", 0) - before
        if taken == 0:
            if just_reset:
                status = "stalled"
                break
            opt, just_reset = make(), True
            continue
        just_reset = False
        iters += taken
        L, g = grad_max()
        rel = abs(L_prev - L) / L_prev
        trace.append((iters, L, g))
        if g < GRAD_TOL and rel < REL_TOL:
            status = "converged"
            break
        L_prev = L
    return net, dict(status=status, iters=int(iters), loss=L_prev if status != "converged" else L, grad=g, rel=rel, trace=trace)


def certified(tag, ids, make_net, b):
    W_DIR.mkdir(exist_ok=True)
    meta_p = W_DIR / f"fiber2_{tag}_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    nets = {}
    for i in ids:
        p = W_DIR / f"fiber2_{tag}_{i}.pt"
        if p.exists() and str(i) in meta:
            net = make_net(i).double()
            net.load_state_dict(torch.load(p))
        else:
            t0 = time.time()
            net, info = certify(make_net(i), b)
            info["seconds"] = round(time.time() - t0, 1)
            torch.save(net.state_dict(), p)
            meta[str(i)] = info
            meta_p.write_text(json.dumps(meta))
            log(f"    {tag} {i}: {info['status']} after {info['iters']} iters, loss {info['loss']:.10e}, max|grad| {info['grad']:.2e}, "
                f"rel {info['rel']:.1e} ({info['seconds']} s)")
        nets[i] = net.float().eval()
    return nets, meta


# ------------------------------------------------------------------ Step 2 per pair (fiber.py definitions)
def edges_of(net, D, sweeps, Ztr, str_, Zte, ste, salt_pts):
    d = D["d_num"]
    phi, s_rows, ps, _ = parts(net, Ztr, str_, d)
    return dict(grid=np.stack([parts(net, *sweeps[i], d)[0][i] for i in range(d)]), rows=phi[:d],
                test=parts(net, Zte, ste, d)[0][:d], mean_rows=phi.mean(1), salt=parts(net, *salt_pts, d)[0][d],
                bias0=net.layers[0].base.bias.detach().numpy(), s_rows=s_rows, c=ps - ps.mean(0))


def analyse_pair(nA, nB, eA, eB, dense, sparse, cont):
    d = len(cont)
    C = corr_matrix(eA["c"], eB["c"])
    r, k = linear_sum_assignment(1 - np.abs(C))
    perm = k[np.argsort(r)]
    recs = []
    for j in range(C.shape[0]):
        kk = int(perm[j])
        rec = dict(j=j, k=kk, match=float(C[j, kk]))
        gA, gB = eA["grid"][:, :, j], eB["grid"][:, :, kk]
        muA, muB = eA["mean_rows"][:d, j][:, None], eB["mean_rows"][:d, kk][:, None]
        fit_g = affine_fits(gA, gB)
        ss = residual_ss(gA, gB, muA, muB, fit_g)
        rec["grid"] = dict(i=ss[0], ii=ss[1], iii=ss[2], den=((gB - muB) ** 2).sum(0).mean())
        a, cs, ai, cf = fit_g
        rec["grid_edge"] = {}
        for i, nm in enumerate(cont):
            den_e = max(((gB[i] - muB[i]) ** 2).mean(), 1e-300)
            rec["grid_edge"][nm] = dict(i=float(np.sqrt(((gB[i] - muB[i] - (gA[i] - muA[i])) ** 2).mean() / den_e)),
                                        ii=float(np.sqrt(((gB[i] - a * gA[i] - cs[i]) ** 2).mean() / den_e)),
                                        iii=float(np.sqrt(((gB[i] - ai[i] * gA[i] - cf[i]) ** 2).mean() / den_e)))
        rA, rB = eA["rows"][:, :, j], eB["rows"][:, :, kk]
        fit_r = affine_fits(rA, rB)
        den_r = ((rB - muB) ** 2).sum(0).mean()
        for reg, (XA, XB) in {"train": (rA, rB), "dense": (rA[:, dense], rB[:, dense]), "sparse": (rA[:, sparse], rB[:, sparse]),
                              "test": (eA["test"][:, :, j], eB["test"][:, :, kk])}.items():
            ss = residual_ss(XA, XB, muA, muB, fit_r)
            rec[reg] = dict(i=ss[0], ii=ss[1], iii=ss[2], den=den_r)
        a_r, cs_r = fit_r[0], fit_r[1]
        c_salt = eB["mean_rows"][d, kk] - a_r * eA["mean_rows"][d, j]
        b_jk = float(cs_r.sum() + c_salt + eB["bias0"][kk] - a_r * eA["bias0"][j])
        u = eB["s_rows"][:, kk]
        psiB = psi_unit(nB, kk, u)
        den_p = np.sqrt(np.mean((psiB - psiB.mean()) ** 2))

        def res(p):
            rr = psiB - psi_unit(nA, j, (u - p[1]) / p[0])
            return rr - rr.mean()

        rec["cross"] = float(np.sqrt(np.mean(res([a_r, b_jk]) ** 2)) / den_p) if abs(a_r) > 1e-12 else np.nan
        try:
            sol = least_squares(res, x0=[a_r if abs(a_r) > 1e-6 else 1.0, b_jk])
            rec["direct"] = float(np.sqrt(np.mean(sol.fun ** 2)) / den_p)
        except Exception:
            rec["direct"] = np.nan
        rec["a"], rec["b_jk"] = a_r, b_jk
        recs.append(rec)
    return recs, C, perm


def split_test(cT, cS, C, perm, W):
    """Exploratory unit splitting, both directions, for matches with |corr| < POOR_MATCH."""
    out = []
    for j in range(W):
        kk = perm[j]
        if abs(C[j, kk]) >= POOR_MATCH:
            continue
        for direction, tgt, src, single in [("student<-2 teacher", cS[:, kk], cT, np.abs(C[:, kk]).max()),
                                            ("teacher<-2 student", cT[:, j], cS, np.abs(C[j, :]).max())]:
            best_sum = max(abs(corr(tgt, src[:, a] + src[:, b])) for a, b in itertools.combinations(range(W), 2))
            best_reg = 0.0
            for a, b in itertools.combinations(range(W), 2):
                Xr = np.column_stack([src[:, a], src[:, b], np.ones(len(tgt))])
                coef, *_ = np.linalg.lstsq(Xr, tgt, rcond=None)
                best_reg = max(best_reg, abs(corr(Xr @ coef, tgt)))
            out.append(dict(direction=direction, matched=float(abs(C[j, kk])), single=float(single), sum2=best_sum, reg2=best_reg))
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    tr, te, te_anc, D, _ = setup()
    log("DISCREPANCIES (files win; see fiber2_prereg.md):")
    log("  1. control-3 R students have no polished weights (0 L-BFGS iters) -> start from control3_R_s*.pt")
    log("  2. d_AB normalised by std of the DATA target delta (fiber.py used the ensemble mean)")
    log("  3. 'h weighted by x_co' = delta literally; reported instead: x_co-weighted RMS of (h_A - h_B)")
    log("  4. teacher = delta_s0.pt unpolished, as in control 3")
    m = D["train"]
    Ztr, str_ = D["Z"][m], D["s"][m]
    b_data = subset(D["batch"], m)
    xco, y = b_data["x_co"].numpy(), b_data["y"].numpy()
    d, W = D["d_num"], D["widths"][0]

    def kan_from(path):
        n = KAN(d, D["widths"], D["n_salt"])
        n.load_state_dict(torch.load(path))
        return n.eval()

    T = kan_from(W_DIR / "delta_s0.pt")
    yT = predict(T, delta_head, to_batch(Ztr, str_, x_co=xco))
    b_teacher = to_batch(Ztr, str_, x_co=xco, y=yT)

    # ---------------- Step 0
    log("\n==================== STEP 0: certified convergence (float64) ====================")
    ind, meta_i = certified("ind", IND_SEEDS, lambda i: kan_from(W_DIR / f"fiber_polished_s{i}.pt"), b_data)
    stu, meta_s = certified("stu", STUDENTS, lambda i: kan_from(W_DIR / f"control3_R_s{i}.pt"), b_teacher)
    log(f"\n  {'model':>12s} {'status':>12s} {'iters':>6s} {'final loss':>18s} {'max|grad|':>10s} {'rel(50 it)':>11s}")
    for tag, meta in [("ind", meta_i), ("stu", meta_s)]:
        for k_, x in meta.items():
            log(f"  {tag + ' ' + k_:>12s} {x['status']:>12s} {x['iters']:6d} {x['loss']:18.12e} {x['grad']:10.2e} {x['rel']:11.1e}")
    conv_i = [i for i in IND_SEEDS if meta_i[str(i)]["status"] == "converged"]
    conv_s = [i for i in STUDENTS if meta_s[str(i)]["status"] == "converged"]
    log(f"\n  converged: independent {len(conv_i)}/6 {conv_i}; R students {len(conv_s)}/20 {conv_s}")
    RESULTS["step0"] = dict(ind={k_: {kk: v for kk, v in x.items() if kk != "trace"} for k_, x in meta_i.items()},
                            stu={k_: {kk: v for kk, v in x.items() if kk != "trace"} for k_, x in meta_s.items()},
                            conv_ind=conv_i, conv_stu=conv_s)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for tag, meta, col in [("independent", meta_i, "tab:blue"), ("R student", meta_s, "tab:orange")]:
        for n_, x in enumerate(meta.values()):
            tr_ = np.array(x["trace"])
            ax.semilogy(tr_[:, 0], tr_[:, 2], color=col, alpha=0.6, lw=0.8, label=tag if n_ == 0 else None)
    ax.axhline(GRAD_TOL, color="r", ls="--", label="max|grad| threshold 1e-7")
    ax.set_xlabel("L-BFGS iterations"); ax.set_ylabel("max |grad|"); ax.legend(); ax.set_title("Step 0: gradient during certification")
    fig.tight_layout(); fig.savefig(FIG / "convergence.png", dpi=110); plt.close(fig)

    # ---------------- Step 1
    log("\n==================== STEP 1: on-fiber test on delta (converged independent seeds) ====================")
    sd_y = y.std()
    Hs = {i: parts(ind[i], Ztr, str_, d)[3] for i in conv_i}
    res_rms = {i: float(np.sqrt(np.mean((Hs[i] * xco - y) ** 2))) for i in conv_i}
    pairs = list(itertools.combinations(conv_i, 2))
    rows1 = []
    for A, B in pairs:
        diff = (Hs[A] - Hs[B]) * xco
        rms = float(np.sqrt(np.mean(diff ** 2)))
        hbar = (Hs[A] + Hs[B]) / 2
        wmean = np.sum(xco * hbar) / xco.sum()
        dhw = float(np.sqrt(np.sum(xco * (Hs[A] - Hs[B]) ** 2) / xco.sum()) / np.sqrt(np.sum(xco * (hbar - wmean) ** 2) / xco.sum()))
        thrB = DEF_B * min(res_rms[A], res_rms[B])
        rows1.append(dict(A=A, B=B, d=rms / sd_y, rms=rms, thrB=thrB, onA=rms / sd_y < DEF_A, onB=rms < thrB, dhw=dhw))
        log(f"  pair ({A},{B}): d_AB {rms / sd_y:.4f} (A: {'ON' if rms / sd_y < DEF_A else 'off'}); RMS diff {rms:.4f} vs 0.25 x min resid "
            f"{thrB:.4f} (B: {'ON' if rms < thrB else 'off'}); x_co-weighted h distance {dhw:.4f}")
    if conv_i:
        log("  data residual RMS of converged seeds: " + ", ".join(f"{i}: {res_rms[i]:.4f}" for i in conv_i))
    nA_ = sum(r["onA"] for r in rows1); nB_ = sum(r["onB"] for r in rows1)
    if pairs:
        log(f"  on-fiber: Definition A {nA_}/{len(pairs)}, Definition B {nB_}/{len(pairs)}")
    RESULTS["step1"] = dict(pairs=rows1, res_rms=res_rms)
    if rows1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter([r["d"] for r in rows1], [r["rms"] / r["thrB"] for r in rows1])
        ax.axvline(DEF_A, color="r", ls="--", label="Definition A threshold")
        ax.axhline(1, color="g", ls="--", label="Definition B threshold")
        ax.set_xlabel("d_AB = RMS(delta diff) / std(target)"); ax.set_ylabel("RMS(delta diff) / (0.25 x min residual)")
        ax.legend(); ax.set_title("Step 1: converged independent seed pairs")
        fig.tight_layout(); fig.savefig(FIG / "on_fiber.png", dpi=110); plt.close(fig)

    # ---------------- Step 2
    log("\n==================== STEP 2: converged R students vs teacher ====================")
    recs, splits, heat, before_after = [], [], [], []
    if conv_s:
        sweeps = grid_inputs(D)
        Zte, ste = D["Zte"][D["test"]], D["ste"][D["test"]]
        _, _, dense, sparse = knn_density(D)
        cont = D["cols"]
        eT = edges_of(T, D, sweeps, Ztr, str_, Zte, ste, sweeps[-1])
        for sd in conv_s:
            eS = edges_of(stu[sd], D, sweeps, Ztr, str_, Zte, ste, sweeps[-1])
            rs, C, perm = analyse_pair(T, stu[sd], eT, eS, dense, sparse, cont)
            for r in rs:
                r["student"] = sd
            recs += rs
            heat.append(np.abs(C[np.arange(W), perm]))
            splits += split_test(eT["c"], eS["c"], C, perm, W)
            pre = kan_from(W_DIR / f"control3_R_s{sd}.pt")
            cp = parts(pre, Ztr, str_, d)[2]
            Cp = corr_matrix(eT["c"], cp - cp.mean(0))
            rp, kp = linear_sum_assignment(1 - np.abs(Cp))
            before_after.append((float(np.median(np.abs(Cp[rp, kp]))), float(np.median(np.abs(C[np.arange(W), perm])))))
            hS = parts(stu[sd], Ztr, str_, d)[3]
            log(f"  student {sd}: delta corr with teacher {corr(hS * xco, yT):.6f}; unit |r| median before certification "
                f"{before_after[-1][0]:.3f} -> after {before_after[-1][1]:.3f}")
        match = np.abs([r["match"] for r in recs])
        log(f"\n  unit match |corr| (teacher unit -> student unit), {len(recs)} unit pairs: median {np.median(match):.3f}; "
            f">= {GOOD_MATCH}: {np.mean(match >= GOOD_MATCH):.1%}; < {POOR_MATCH}: {np.mean(match < POOR_MATCH):.1%}")
        hm = np.array(heat)
        log("  per teacher unit (median |corr| over students): " + ", ".join(f"u{j} {np.median(hm[:, j]):.3f}" for j in range(W)))
        ba = np.array(before_after)
        log(f"  unit match median before -> after certification: {np.median(ba[:, 0]):.3f} -> {np.median(ba[:, 1]):.3f}")
        good = [r for r in recs if abs(r["match"]) >= GOOD_MATCH]
        log(f"  unit set for G4/G5 (|corr| >= {GOOD_MATCH}): {len(good)} unit pairs")

        def R(rs, reg, q):
            return np.array([np.sqrt(r[reg][q] / r[reg]["den"]) for r in rs])

        log(f"\n  residuals (median pooled residual per unit pair):  {'set':12s} {'domain':7s} {'(i)':>8s} {'(ii)':>8s} {'(iii)':>8s}  explained (ii)/(iii)")
        ladder = {}
        for name, rs in [("|r|>=0.9", good), ("all", recs)]:
            if not rs:
                continue
            for reg in ["grid", "train", "dense", "sparse", "test"]:
                si = sum(r[reg]["i"] for r in rs)
                ex = (1 - sum(r[reg]["ii"] for r in rs) / si, 1 - sum(r[reg]["iii"] for r in rs) / si)
                vals = [float(np.median(R(rs, reg, q))) for q in ["i", "ii", "iii"]]
                ladder[f"{name}|{reg}"] = vals + list(ex)
                log(f"  {'':49s}{name:12s} {reg:7s} {vals[0]:8.4f} {vals[1]:8.4f} {vals[2]:8.4f}  {ex[0]:.3f} / {ex[1]:.3f}")
        if good:
            log("  per-edge grid residuals (|r|>=0.9): input (i) -> (ii) -> (iii)")
            for nm in cont:
                v = [r["grid_edge"][nm] for r in good]
                log(f"    {nm:10s} {np.median([x['i'] for x in v]):.3f} -> {np.median([x['ii'] for x in v]):.3f} -> {np.median([x['iii'] for x in v]):.3f}")
            cr = np.array([r["cross"] for r in good]); di = np.array([r["direct"] for r in good])
            log(f"  outgoing cross-check (|r|>=0.9): cross {np.nanmedian(cr):.4f}, direct free-affine {np.nanmedian(di):.4f}")
        if splits:
            for direction in ["student<-2 teacher", "teacher<-2 student"]:
                sp = [x for x in splits if x["direction"] == direction]
                if sp:
                    log(f"  EXPLORATORY split ({direction}, {len(sp)} units with match < {POOR_MATCH}): matched {np.median([x['matched'] for x in sp]):.3f}, "
                        f"best single {np.median([x['single'] for x in sp]):.3f}, best 2-sum {np.median([x['sum2'] for x in sp]):.3f}, "
                        f"best 2-regression {np.median([x['reg2'] for x in sp]):.3f}; regression beats single by > 0.1 in "
                        f"{np.mean([x['reg2'] > x['single'] + 0.1 for x in sp]):.0%}")
        RESULTS["step2"] = dict(ladder=ladder, n_unit_pairs=len(recs), n_good=len(good), match_median=float(np.median(match)),
                                before_after=before_after, splits=splits,
                                per_teacher_unit=[float(np.median(hm[:, j])) for j in range(W)])

        # figures
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
        im = axs[0].imshow(hm, vmin=0, vmax=1, aspect="auto", cmap="viridis")
        axs[0].set_yticks(range(len(conv_s))); axs[0].set_yticklabels(conv_s, fontsize=7)
        axs[0].set_xlabel("teacher unit"); axs[0].set_ylabel("converged student"); axs[0].set_title("match |corr| after certification")
        fig.colorbar(im, ax=axs[0])
        axs[1].scatter(ba[:, 0], ba[:, 1]); axs[1].plot([0, 1], [0, 1], "k:")
        axs[1].set_xlabel("median unit |corr| before (control 3)"); axs[1].set_ylabel("after certification")
        axs[1].set_title("does the unit mismatch survive convergence?")
        fig.tight_layout(); fig.savefig(FIG / "unit_match.png", dpi=110); plt.close(fig)
        if good:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            regs = ["grid", "dense", "sparse", "test"]
            x = np.arange(len(regs))
            for o, lab in enumerate(["(i) centred", "(ii) shared affine", "(iii) free affine"]):
                ax.bar(x + (o - 1) * 0.27, [ladder[f"|r|>=0.9|{r_}"][o] for r_ in regs], 0.27, label=lab)
            ax.set_xticks(x); ax.set_xticklabels(regs); ax.set_yscale("log"); ax.legend()
            ax.set_ylabel("median residual / centred-target RMS"); ax.set_title(f"Teacher -> converged students, |corr| >= 0.9 (n={len(good)})")
            fig.tight_layout(); fig.savefig(FIG / "residual_ladder.png", dpi=110); plt.close(fig)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(di, cr, s=10)
            lim = np.nanmax(np.r_[cr, di]) * 1.05
            ax.plot([0, lim], [0, 2 * lim], "r--", label="cross = 2 x direct"); ax.plot([0, lim], [0, lim], "k:", label="cross = direct")
            ax.set_xlabel("direct free-affine fit"); ax.set_ylabel("cross-check (a, b_jk)"); ax.legend(); ax.set_title("outgoing-edge cross-check")
            fig.tight_layout(); fig.savefig(FIG / "outgoing_crosscheck.png", dpi=110); plt.close(fig)
    else:
        good = []

    # ---------------- verdicts
    v = {"G1": ("held" if len(conv_i) >= 4 and len(conv_s) >= 10 else "failed") + f" (independent {len(conv_i)}/6, students {len(conv_s)}/20)"}
    if len(conv_i) < 2:
        v["G2"] = v["G3"] = f"inconclusive (only {len(conv_i)} independent seed(s) converged)"
    else:
        v["G2"] = ("held" if nA_ >= 1 else "failed") + f" ({nA_}/{len(pairs)} pairs on-fiber under A)"
        v["G3"] = ("held" if nB_ >= 0.5 * len(pairs) else "failed") + f" ({nB_}/{len(pairs)} under B)"
    if not good:
        v["G4"] = v["G5"] = "inconclusive (no converged students or no unit pairs with |corr| >= 0.9)"
    else:
        Ri, Rii, Riii = R(good, "dense", "i"), R(good, "dense", "ii"), R(good, "dense", "iii")
        red = float(np.median(1 - Rii / Ri))
        v["G4"] = ("held" if red >= 0.8 else "failed") + f" (median reduction {red:.3f})"
        v["G5"] = ("held" if np.median(Rii) <= 1.5 * np.median(Riii) else "failed") + \
            f" (median (ii) {np.median(Rii):.4f} vs 1.5 x (iii) {1.5 * np.median(Riii):.4f})"
    log("\nVERDICTS: " + "; ".join(f"{k}: {x}" for k, x in v.items()))
    RESULTS["verdicts"] = v
    (ROOT / "fiber2_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))


if __name__ == "__main__":
    main()

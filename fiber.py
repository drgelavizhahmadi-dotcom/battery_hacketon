"""Completeness test of the additive affine gauge on KAN-delta's realised fiber (see fiber_prereg.md).
Run: python3 fiber.py      Weights: gauge_weights/fiber_*  (not committed)   Figures: figs/fiber/"""
import copy
import itertools
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares, linear_sum_assignment

from gauge import W_DIR, corr, corr_matrix, grid_inputs, knn_density, parts, setup
from kan import KAN, SMOOTH, delta_head, subset

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
ROOT = Path(__file__).parent
FIG = ROOT / "figs" / "fiber"
SEEDS = list(range(20))
LBFGS_MAX, LBFGS_STEP, REL_TOL = 2000, 20, 1e-8
ON_FIBER = 0.05
GOOD_MATCH, POOR_MATCH = 0.9, 0.8
RESULTS = {}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ Step 1: polish
def polish(net, b):
    net = copy.deepcopy(net).double()
    bd = {k: (v.double() if v.is_floating_point() else v) for k, v in b.items()}
    lb = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=LBFGS_STEP, line_search_fn="strong_wolfe",
                           tolerance_grad=1e-12, tolerance_change=1e-14)

    def parts_loss():
        mse = F.mse_loss(delta_head(net, bd), bd["y"])
        pen = SMOOTH * net.smoothness()
        return mse, pen

    def closure():
        lb.zero_grad()
        mse, pen = parts_loss()
        (mse + pen).backward()
        return mse + pen

    with torch.no_grad():
        m0, p0 = (float(v) for v in parts_loss())
    prev, iters, reason = m0 + p0, 0, "iteration cap (2000)"
    while iters < LBFGS_MAX:
        lb.param_groups[0]["max_iter"] = min(LBFGS_STEP, LBFGS_MAX - iters)
        lb.step(closure)
        new_it = lb.state[lb._params[0]]["n_iter"]
        with torch.no_grad():
            m1, p1 = (float(v) for v in parts_loss())
        if new_it == iters:
            reason = "optimizer stalled (no iteration taken)"
            break
        iters = new_it
        if abs(prev - (m1 + p1)) / prev < REL_TOL:
            reason = "relative loss change < 1e-8"
            break
        prev = m1 + p1
    return net.float().eval(), dict(loss0=m0 + p0, mse0=m0, loss=m1 + p1, mse=m1, pen=p1, iters=int(iters), reason=reason)


def load_polished(D, b):
    W_DIR.mkdir(exist_ok=True)
    meta_p = W_DIR / "fiber_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    nets = []
    for s in SEEDS:
        p = W_DIR / f"fiber_polished_s{s}.pt"
        net = KAN(D["d_num"], D["widths"], D["n_salt"])
        if p.exists() and str(s) in meta:
            net.load_state_dict(torch.load(p)); net.eval()
        else:
            raw = KAN(D["d_num"], D["widths"], D["n_salt"])
            raw.load_state_dict(torch.load(W_DIR / f"delta_s{s}.pt")); raw.eval()
            net, info = polish(raw, b)
            torch.save(net.state_dict(), p)
            meta[str(s)] = info
            meta_p.write_text(json.dumps(meta))
        nets.append(net)
    return nets, meta


# ------------------------------------------------------------------ helpers
@torch.no_grad()
def psi_unit(net, j, u):
    return net.layers[1].edge(j, torch.tanh(torch.tensor(u, dtype=torch.float32)))[:, 0].numpy()


def affine_fits(A, B):
    """A, B: (E, P). Shared scale a (+ per-edge constants) and free per-edge scales."""
    Am, Bm = A.mean(1, keepdims=True), B.mean(1, keepdims=True)
    Ac, Bc = A - Am, B - Bm
    a = float((Ac * Bc).sum() / max((Ac * Ac).sum(), 1e-300))
    ai = (Ac * Bc).sum(1) / np.maximum((Ac * Ac).sum(1), 1e-300)
    return a, Bm - a * Am, ai, Bm - ai[:, None] * Am


def residual_ss(A, B, muA, muB, fit):
    """Mean-over-points sum-over-edges squared residuals for (i), (ii), (iii) and the centred-target SS."""
    a, cs, ai, cf = fit
    ss_i = (((B - muB) - (A - muA)) ** 2).sum(0).mean()
    ss_ii = ((B - (a * A + cs)) ** 2).sum(0).mean()
    ss_iii = ((B - (ai[:, None] * A + cf)) ** 2).sum(0).mean()
    return ss_i, ss_ii, ss_iii


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    tr, te, te_anc, D, _ = setup()
    log("DISCREPANCIES (files win; see fiber_prereg.md):")
    salts_tr = [D["salts"][i] for i in np.unique(D["s"][D["train"]])]
    log(f"  1. brief says 3 salt points; KAN-delta training rows contain {len(salts_tr)} salts {salts_tr} -> salt edges at all "
        f"{len(salts_tr)}, 3-test-salt subset reported too")
    log("  2. brief defines on-fiber on h; model is fitted on delta = x_co*h -> primary on h (as specified), delta also reported")
    log("  3. b_jk = sum of fitted constants + salt constant + layer-0 bias terms (needed for s_k^B = a s_j^A + b_jk)")

    m = D["train"]
    Ztr, str_ = D["Z"][m], D["s"][m]
    b = subset(D["batch"], m)
    xco = b["x_co"].numpy()
    d, W = D["d_num"], D["widths"][0]
    cont = D["cols"]  # 8 continuous inputs
    Zte, ste = D["Zte"][D["test"]], D["ste"][D["test"]]

    # ---------------- Step 1
    log("\n==================== STEP 1: polish the 20 saved seeds ====================")
    nets, meta = load_polished(D, b)
    log(f"  {'seed':>4s} {'loss before':>12s} {'loss after':>12s} {'mse after':>11s} {'penalty':>9s} {'iters':>6s}  stop reason")
    for s in SEEDS:
        x = meta[str(s)]
        log(f"  {s:4d} {x['loss0']:12.6e} {x['loss']:12.6e} {x['mse']:11.4e} {x['pen']:9.2e} {x['iters']:6d}  {x['reason']}")
    P = {s: parts(n, Ztr, str_, d) for s, n in zip(SEEDS, nets)}  # phi, s, psi, h on training rows
    H = np.array([P[s][3] for s in SEEDS])
    sd_h = H.mean(0).std()
    sd_delta = (H.mean(0) * xco).std()
    pairs = list(itertools.combinations(SEEDS, 2))
    dh = {p: float(np.sqrt(np.mean((H[p[0]] - H[p[1]]) ** 2)) / sd_h) for p in pairs}
    dd = {p: float(np.sqrt(np.mean(((H[p[0]] - H[p[1]]) * xco) ** 2)) / sd_delta) for p in pairs}
    on = [p for p in pairs if dh[p] < ON_FIBER]
    on_delta = [p for p in pairs if dd[p] < ON_FIBER]
    log(f"  d_AB(h): median {np.median(list(dh.values())):.4f}, range [{min(dh.values()):.4f}, {max(dh.values()):.4f}]; "
        f"on-fiber (d<{ON_FIBER}): {len(on)}/{len(pairs)} = {len(on) / len(pairs):.1%}")
    log(f"  d_AB(delta): median {np.median(list(dd.values())):.4f}; on-fiber: {len(on_delta)}/{len(pairs)} = {len(on_delta) / len(pairs):.1%}")
    RESULTS["step1"] = dict(meta=meta, dh_median=float(np.median(list(dh.values()))), on_fiber=len(on), n_pairs=len(pairs),
                            on_fiber_delta=len(on_delta), dd_median=float(np.median(list(dd.values()))))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(list(dh.values()), bins=40, alpha=0.7, label="d_AB on h (primary)")
    ax.hist(list(dd.values()), bins=40, alpha=0.5, label="d_AB on delta")
    ax.axvline(ON_FIBER, color="r", ls="--", label=f"on-fiber threshold {ON_FIBER}")
    ax.set_xlabel("RMS(h_A - h_B) / std(h), training rows"); ax.set_ylabel("seed pairs"); ax.legend()
    ax.set_title(f"Polished KAN-delta seeds: {len(on)}/{len(pairs)} pairs on-fiber")
    fig.tight_layout(); fig.savefig(FIG / "on_fiber_pairs.png", dpi=110); plt.close(fig)
    if not on:
        log("\nNo on-fiber pairs -> Steps 2-4 not run; F2-F5 inconclusive.")
        (ROOT / "fiber_results.json").write_text(json.dumps(RESULTS, indent=1))
        return

    # ---------------- precompute edges per seed
    sweeps = grid_inputs(D)
    salt_pts = sweeps[-1]
    test_salt_mask = np.isin(salt_pts[1], ste)
    E = {}
    for s, net in zip(SEEDS, nets):
        phi_rows = P[s][0]  # (d+1, n, W)
        E[s] = dict(grid=np.stack([parts(net, *sweeps[i], d)[0][i] for i in range(d)]),  # (d, 100, W)
                    rows=phi_rows[:d], test=parts(net, Zte, ste, d)[0][:d],
                    mean_rows=phi_rows.mean(1),  # (d+1, W)
                    salt=parts(net, *salt_pts, d)[0][d],  # (n_salt_pts, W)
                    bias0=net.layers[0].base.bias.detach().numpy(), s_rows=P[s][1],
                    c=P[s][2] - P[s][2].mean(0))
    _, _, dense, sparse = knn_density(D)

    # ---------------- Steps 2-4 over on-fiber pairs
    log("\n==================== STEP 2-4 over on-fiber pairs ====================")
    recs = []
    for A, B in on:
        C = corr_matrix(E[A]["c"], E[B]["c"])
        r, k = linear_sum_assignment(1 - np.abs(C))
        perm = k[np.argsort(r)]
        for j in range(W):
            kk = perm[j]
            mc = float(C[j, kk])
            eA, eB = E[A], E[B]
            rec = dict(A=A, B=B, j=j, k=int(kk), match=mc)
            # grid: fit and evaluate on grid; centring by training-row means
            gA, gB = eA["grid"][:, :, j], eB["grid"][:, :, kk]
            muA, muB = eA["mean_rows"][:d, j][:, None], eB["mean_rows"][:d, kk][:, None]
            fit_g = affine_fits(gA, gB)
            ss_i, ss_ii, ss_iii = residual_ss(gA, gB, muA, muB, fit_g)
            den_g = ((gB - muB) ** 2).sum(0).mean()
            rec["grid"] = dict(i=ss_i, ii=ss_ii, iii=ss_iii, den=den_g)
            rec["grid_edge"] = {}
            a, cs, ai, cf = fit_g
            for i, nm in enumerate(cont):
                den_e = max(((gB[i] - muB[i]) ** 2).mean(), 1e-300)
                rec["grid_edge"][nm] = dict(i=float(np.sqrt(((gB[i] - muB[i] - (gA[i] - muA[i])) ** 2).mean() / den_e)),
                                            ii=float(np.sqrt(((gB[i] - a * gA[i] - cs[i]) ** 2).mean() / den_e)),
                                            iii=float(np.sqrt(((gB[i] - ai[i] * gA[i] - cf[i]) ** 2).mean() / den_e)))
            # regions: fit on all training rows, evaluate per region, shared denominator
            rA, rB = eA["rows"][:, :, j], eB["rows"][:, :, kk]
            fit_r = affine_fits(rA, rB)
            den_r = ((rB - muB) ** 2).sum(0).mean()
            tA, tB = eA["test"][:, :, j], eB["test"][:, :, kk]
            for reg, (XA, XB) in {"train": (rA, rB), "dense": (rA[:, dense], rB[:, dense]),
                                  "sparse": (rA[:, sparse], rB[:, sparse]), "test": (tA, tB)}.items():
                ss = residual_ss(XA, XB, muA, muB, fit_r)
                rec[reg] = dict(i=ss[0], ii=ss[1], iii=ss[2], den=den_r)
            # salt edge: (i) centred by training-row mean, (ii) the continuous-row a, (iii) free
            a_r, cs_r = fit_r[0], fit_r[1]
            sa, sb = eA["salt"][:, j], eB["salt"][:, kk]
            smA, smB = eA["mean_rows"][d, j], eB["mean_rows"][d, kk]
            c_salt = smB - a_r * smA
            for tag, msk in [("salt4", np.ones(len(sa), bool)), ("salt3", test_salt_mask)]:
                xa, xb = sa[msk], sb[msk]
                den_s = max(((xb - smB) ** 2).mean(), 1e-300)
                af = np.polyfit(xa, xb, 1) if len(xa) >= 2 and np.ptp(xa) > 0 else (0.0, xb.mean())
                rec[tag] = dict(i=float(np.sqrt(((xb - smB - (xa - smA)) ** 2).mean() / den_s)),
                                ii=float(np.sqrt(((xb - a_r * xa - c_salt) ** 2).mean() / den_s)),
                                iii=float(np.sqrt(((xb - np.polyval(af, xa)) ** 2).mean() / den_s)))
            # Step 4: outgoing edge cross-check
            b_jk = float(cs_r.sum() + c_salt + eB["bias0"][kk] - a_r * eA["bias0"][j])
            u = eB["s_rows"][:, kk]
            psiB = psi_unit(nets[B], kk, u)
            den_p = np.sqrt(np.mean((psiB - psiB.mean()) ** 2))

            def res(p):
                rr = psiB - psi_unit(nets[A], j, (u - p[1]) / p[0])
                return rr - rr.mean()

            cross = float(np.sqrt(np.mean(res([a_r, b_jk]) ** 2)) / den_p) if abs(a_r) > 1e-12 else np.nan
            try:
                sol = least_squares(res, x0=[a_r if abs(a_r) > 1e-6 else 1.0, b_jk])
                direct = float(np.sqrt(np.mean(sol.fun ** 2)) / den_p)
            except Exception:
                direct = np.nan
            rec.update(a=a_r, b_jk=b_jk, cross=cross, direct=direct,
                       s_fit_rel=float(np.sqrt(np.mean((u - (a_r * eA["s_rows"][:, j] + b_jk)) ** 2)) / u.std()))
            recs.append(rec)

    match = np.array([r["match"] for r in recs])
    log(f"  matched unit pairs: {len(recs)} (from {len(on)} on-fiber seed pairs x {W} units)")
    log(f"  match |corr|: median {np.median(np.abs(match)):.3f}; >= {GOOD_MATCH}: {np.mean(np.abs(match) >= GOOD_MATCH):.1%}; "
        f"< {POOR_MATCH}: {np.mean(np.abs(match) < POOR_MATCH):.1%}")
    by_unit = {}
    for r in recs:
        by_unit.setdefault(r["j"], []).append(abs(r["match"]))
    log("  per source-unit index (median |corr| [min]): " +
        ", ".join(f"u{j} {np.median(v):.3f} [{min(v):.3f}]" for j, v in sorted(by_unit.items())))
    good = [r for r in recs if abs(r["match"]) >= GOOD_MATCH]
    log(f"  unit set for F2-F5 (|corr| >= {GOOD_MATCH}): {len(good)} unit pairs")

    def R(rs, reg, q):
        return np.array([np.sqrt(r[reg][q] / r[reg]["den"]) for r in rs])

    def explained(rs, reg):
        si = sum(r[reg]["i"] for r in rs)
        return 1 - sum(r[reg]["ii"] for r in rs) / si, 1 - sum(r[reg]["iii"] for r in rs) / si

    log("\n  STEP 3 residuals (median pooled residual per unit pair; / centred-target RMS):")
    log(f"  {'set':12s} {'domain':7s} {'(i)':>8s} {'(ii)':>8s} {'(iii)':>8s}  explained by (ii) / (iii)")
    ladder = {}
    for name, rs in [("|r|>=0.9", good), ("all matched", recs)]:
        if not rs:
            continue
        for reg in ["grid", "train", "dense", "sparse", "test"]:
            ri, rii, riii = (float(np.median(R(rs, reg, q))) for q in ["i", "ii", "iii"])
            ex = explained(rs, reg)
            ladder[(name, reg)] = (ri, rii, riii, ex)
            log(f"  {name:12s} {reg:7s} {ri:8.4f} {rii:8.4f} {riii:8.4f}  {ex[0]:.3f} / {ex[1]:.3f}")
    log("\n  per-edge grid residuals (median over |r|>=0.9 unit pairs): input (i) -> (ii) -> (iii)")
    for nm in cont:
        v = [r["grid_edge"][nm] for r in good]
        if v:
            log(f"    {nm:10s} {np.median([x['i'] for x in v]):.3f} -> {np.median([x['ii'] for x in v]):.3f} -> {np.median([x['iii'] for x in v]):.3f}")
    for tag in ["salt4", "salt3"]:
        v = [r[tag] for r in good]
        if v:
            log(f"  salt edge ({tag[-1]} salts, reported separately): (i) {np.median([x['i'] for x in v]):.3f}, "
                f"(ii) {np.median([x['ii'] for x in v]):.3f}, (iii) {np.median([x['iii'] for x in v]):.3f}")
    cr = np.array([r["cross"] for r in good]); di = np.array([r["direct"] for r in good])
    sf = np.array([r["s_fit_rel"] for r in good])
    if len(good):
        log(f"\n  STEP 4 (|r|>=0.9): cross-check residual median {np.nanmedian(cr):.4f}, direct free-affine fit {np.nanmedian(di):.4f}; "
            f"s_k^B vs a*s_j^A+b_jk relative RMS {np.median(sf):.4f}")

    # exploratory: unit splitting
    split = []
    for A, B in on:
        C = corr_matrix(E[A]["c"], E[B]["c"])
        r_, k_ = linear_sum_assignment(1 - np.abs(C))
        for j, kk in zip(r_, k_):
            if abs(C[j, kk]) >= POOR_MATCH:
                continue
            tgt = E[B]["c"][:, kk]
            best1 = float(np.abs(C[:, kk]).max())
            best_sum = max(abs(corr(tgt, E[A]["c"][:, j1] + E[A]["c"][:, j2])) for j1, j2 in itertools.combinations(range(W), 2))
            best_reg = 0.0
            for j1, j2 in itertools.combinations(range(W), 2):
                Xr = np.column_stack([E[A]["c"][:, j1], E[A]["c"][:, j2], np.ones(len(tgt))])
                coef, *_ = np.linalg.lstsq(Xr, tgt, rcond=None)
                best_reg = max(best_reg, abs(corr(Xr @ coef, tgt)))
            split.append((abs(C[j, kk]), best1, best_sum, best_reg))
    if split:
        sp = np.array(split)
        log(f"\n  EXPLORATORY unit splitting ({len(sp)} target units with match |corr| < {POOR_MATCH}): median matched {np.median(sp[:, 0]):.3f}, "
            f"best single {np.median(sp[:, 1]):.3f}, best 2-unit sum {np.median(sp[:, 2]):.3f}, best 2-unit regression {np.median(sp[:, 3]):.3f}; "
            f"sum beats best single in {np.mean(sp[:, 2] > sp[:, 1]):.0%}, regression beats it by >0.1 in {np.mean(sp[:, 3] > sp[:, 1] + 0.1):.0%}")

    # ---------------- verdicts (as pre-registered)
    v = {"F1": ("held" if len(on) / len(pairs) >= 0.5 else "failed") + f" ({len(on)}/{len(pairs)} = {len(on) / len(pairs):.1%})"}
    if not good:
        for f in ["F2", "F3", "F4", "F5"]:
            v[f] = "inconclusive (no unit pairs with |corr| >= 0.9)"
    else:
        Ri, Rii, Riii = R(good, "dense", "i"), R(good, "dense", "ii"), R(good, "dense", "iii")
        red = float(np.median(1 - Rii / Ri))
        v["F2"] = ("held" if red >= 0.8 else "failed") + f" (median reduction {red:.3f})"
        v["F3"] = ("held" if np.median(Rii) <= 1.5 * np.median(Riii) else "failed") + \
            f" (median (ii) {np.median(Rii):.4f} vs 1.5 x (iii) {1.5 * np.median(Riii):.4f})"
        v["F4"] = ("held" if np.nanmedian(cr) <= 2 * np.nanmedian(di) else "failed") + \
            f" (cross {np.nanmedian(cr):.4f} vs 2 x direct {2 * np.nanmedian(di):.4f})"
        md, ms, mt = (float(np.median(R(good, reg, "ii"))) for reg in ["dense", "sparse", "test"])
        v["F5"] = ("held" if ms >= 2 * md and mt > max(md, ms) else "failed") + f" (dense {md:.4f}, sparse {ms:.4f}, test {mt:.4f})"
    log("\nVERDICTS: " + "; ".join(f"{k}: {x}" for k, x in v.items()))
    RESULTS.update(verdicts=v, n_unit_pairs=len(recs), n_good=len(good),
                   ladder={f"{a}|{b_}": list(x[:3]) + list(x[3]) for (a, b_), x in ladder.items()},
                   step4=dict(cross=float(np.nanmedian(cr)) if len(good) else None, direct=float(np.nanmedian(di)) if len(good) else None),
                   match=match.tolist(), split=np.array(split).tolist() if split else [])
    (ROOT / "fiber_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))

    # ---------------- figures
    fig, ax = plt.subplots(figsize=(10, 4.5))
    regs = ["grid", "dense", "sparse", "test"]
    x = np.arange(len(regs))
    for o, (q, lab) in enumerate([(0, "(i) centred"), (1, "(ii) shared affine"), (2, "(iii) free affine")]):
        ax.bar(x + (o - 1) * 0.27, [ladder[("|r|>=0.9", r_)][q] if ("|r|>=0.9", r_) in ladder else np.nan for r_ in regs], 0.27, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(regs); ax.set_yscale("log"); ax.set_ylabel("median residual / centred-target RMS")
    ax.set_title(f"Residual ladder, unit pairs with match |corr| >= 0.9 (n={len(good)})"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "residual_ladder.png", dpi=110); plt.close(fig)

    # reference seed = most on-fiber partners
    partners = {s: [p for p in on if s in p] for s in SEEDS}
    ref = max(SEEDS, key=lambda s: (len(partners[s]), -np.mean([dh[p] for p in pairs if s in p])))
    iT = cont.index("invT")
    xs = D["sc"].inv(iT, sweeps[iT][0][:, iT])
    fig, axs = plt.subplots(2, W, figsize=(3.3 * W, 6.5), sharex=True)
    ref_rows = []
    for p in partners[ref]:
        o = p[1] if p[0] == ref else p[0]
        C = corr_matrix(E[ref]["c"], E[o]["c"])
        r_, k_ = linear_sum_assignment(1 - np.abs(C))
        perm = k_[np.argsort(r_)]
        ref_rows.append((o, np.abs(C[np.arange(W), perm])))
        for j in range(W):
            kk = perm[j]
            fit = affine_fits(E[ref]["grid"][:, :, j], E[o]["grid"][:, :, kk])
            gB = E[o]["grid"][iT, :, kk] - E[o]["mean_rows"][iT, kk]
            axs[0, j].plot(xs, gB, lw=0.8, alpha=0.5)
            axs[1, j].plot(xs, gB / fit[0] if abs(fit[0]) > 1e-12 else gB, lw=0.8, alpha=0.5)
    for j in range(W):
        gA = E[ref]["grid"][iT, :, j] - E[ref]["mean_rows"][iT, j]
        for rrow in (0, 1):
            axs[rrow, j].plot(xs, gA, color="k", lw=2)
        axs[0, j].set_title(f"ref unit {j}", fontsize=9); axs[1, j].set_xlabel("1000/T (1/K)")
    axs[0, 0].set_ylabel("(i) centred"); axs[1, 0].set_ylabel("(ii) / shared scale a")
    fig.suptitle(f"1000/T edges: reference seed {ref} (black) and its {len(partners[ref])} on-fiber partners, matched units")
    fig.tight_layout(); fig.savefig(FIG / "edge_overlay_T.png", dpi=110); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    if len(good):
        axs[0].scatter(di, cr, s=10)
        lim = np.nanmax(np.r_[cr, di]) * 1.05
        axs[0].plot([0, lim], [0, 2 * lim], "r--", label="cross = 2 x direct")
        axs[0].plot([0, lim], [0, lim], "k:", label="cross = direct")
        axs[0].set_xlabel("direct free-affine fit residual"); axs[0].set_ylabel("cross-check residual (a, b_jk from Step 3)")
        axs[0].legend(); axs[0].set_title(f"Step 4, unit pairs |corr| >= 0.9 (n={len(good)})")
        ex = good[int(np.argmin(np.abs(cr - np.nanmedian(cr))))]
        u = E[ex["B"]]["s_rows"][:, ex["k"]]
        o_ = np.argsort(u)
        pb = psi_unit(nets[ex["B"]], ex["k"], u)
        pa = psi_unit(nets[ex["A"]], ex["j"], (u - ex["b_jk"]) / ex["a"])
        axs[1].plot(u[o_], (pb - pb.mean())[o_], "k", lw=2, label=f"psi_k (seed {ex['B']}, unit {ex['k']})")
        axs[1].plot(u[o_], (pa - pa.mean())[o_], "tab:orange", label=f"psi_j((u-b)/a) (seed {ex['A']}, unit {ex['j']})")
        axs[1].set_xlabel("u = s_k"); axs[1].legend(fontsize=8); axs[1].set_title(f"median example: cross {ex['cross']:.3f}")
    fig.tight_layout(); fig.savefig(FIG / "outgoing_crosscheck.png", dpi=110); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    if ref_rows:
        Mx = np.array([r_[1] for r_ in ref_rows])
        im = axs[0].imshow(Mx, vmin=0, vmax=1, aspect="auto", cmap="viridis")
        axs[0].set_yticks(range(len(ref_rows))); axs[0].set_yticklabels([r_[0] for r_ in ref_rows], fontsize=7)
        axs[0].set_xlabel(f"unit of reference seed {ref}"); axs[0].set_ylabel("on-fiber partner seed")
        axs[0].set_title("match |corr| per unit and seed")
        fig.colorbar(im, ax=axs[0])
    axs[1].hist(np.abs(match), bins=40)
    axs[1].axvline(GOOD_MATCH, color="g", ls="--"); axs[1].axvline(POOR_MATCH, color="r", ls="--")
    axs[1].set_xlabel("match |corr|, all on-fiber pairs x units"); axs[1].set_title(f"{len(recs)} matched unit pairs")
    fig.tight_layout(); fig.savefig(FIG / "unit_match.png", dpi=110); plt.close(fig)
    log(f"\nfigures in figs/fiber/ (reference seed {ref}); results in fiber_results.json")


if __name__ == "__main__":
    main()

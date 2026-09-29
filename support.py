"""support: near-null space of the layer-1 design matrix, where KAN-delta models differ, and the Hessian.
See support_prereg.md. Run: python3 support.py    Figures: figs/support/"""
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
from scipy.optimize import linear_sum_assignment
from torch.func import functional_call

from gauge import W_DIR, corr_matrix, parts, setup
from kan import KAN, N_RBF, SMOOTH, delta_head, predict, subset, to_batch

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
ROOT = Path(__file__).parent
FIG = ROOT / "figs" / "support"
REL_NULL = 1e-2
GROUP = {"eps_co", "lneta_co", "M_co", "mix_eps", "mix_lneta", "x_co"}
IND = list(range(6))
STUDENTS = list(range(100, 120))
N_FLAT = 10
MATCH_SPLIT = 0.8
RESULTS = {}


def log(*a):
    print(*a, flush=True)


def silu(x):
    return x / (1 + np.exp(-x))


def design_matrix(Z):
    g, h = np.linspace(-1, 1, N_RBF), 2 / (N_RBF - 1)
    cols = []
    for i in range(Z.shape[1]):
        z = Z[:, i]
        cols.append(silu(z))
        cols.extend(np.exp(-(((z - gk) / h) ** 2)) for gk in g)
    return np.column_stack(cols)


def classify(energy, names):
    e = energy / energy.sum()
    big = [names[i] for i in np.nonzero(e >= 0.2)[0]]
    if e.max() >= 0.9:
        return "WITHIN-EDGE", big
    if len(big) >= 2:
        return "REDISTRIBUTION", big
    return "OTHER", big


def theta(net, j, d):
    """Unit j's layer-1 continuous coefficients, input-major: [base_w, rbf_1..8] per input."""
    bw = net.layers[0].base.weight.detach().numpy()
    cf = net.layers[0].coef.detach().numpy()
    return np.concatenate([np.r_[bw[j, i], cf[j, i, :]] for i in range(d)])


def load(path, D):
    n = KAN(D["d_num"], D["widths"], D["n_salt"])
    n.load_state_dict(torch.load(path))
    return n.eval()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    tr, te, te_anc, D, _ = setup()
    names, d = D["cols"], D["d_num"]
    m = D["train"]
    Ztr, str_ = D["Z"][m], D["s"][m]
    log("DISCREPANCIES / choices (files win; see support_prereg.md):")
    log("  D1 brief 'layer-1 edges' = kan.py layers.0 (input->hidden)")
    log("  D2 16 embedding params of 8 salts absent from training rows have zero gradient/Hessian -> excluded; Hessian over 609 params")
    log("  D3 committed names: control 2 = C1-C3 (no C4); control 3 = C1', C2', C4 (no C3'/C4')")
    log("  D4 the '5-7x' loss ratio is measured below, not assumed")

    # ================= STEP 1 (no trained weights loaded yet) =================
    log("\n==================== STEP 1: near-null space of M (NO trained weights loaded) ====================")
    M = design_matrix(Ztr)
    log(f"  M shape {M.shape} (expected (2069, 72)){'' if M.shape[1] == 72 else '  <-- FLAG: column count differs'}")
    _, S, Vt = np.linalg.svd(M, full_matrices=False)
    rel = S / S[0]
    VN = Vt[rel < REL_NULL].T
    dimN = VN.shape[1]
    log("  relative singular values (sorted): " + " ".join(f"{x:.1e}" for x in rel))
    log(f"  dim(N) = {dimN} (rel sv < {REL_NULL}); random expectation dim(N)/72 = {dimN / 72:.3f}")
    classes = []
    for q in range(dimN):
        en = (VN[:, q].reshape(d, 1 + N_RBF) ** 2).sum(1)
        cls, big = classify(en, names)
        grp = cls == "REDISTRIBUTION" and len(GROUP & set(big)) >= 2
        classes.append(dict(cls=cls, big=big, group=grp, energy=(en / en.sum()).tolist(), rel_sv=float(rel[rel < REL_NULL][q])))
        log(f"   N{q:02d} rel sv {classes[-1]['rel_sv']:.1e}  {cls:15s} inputs>=20%: {big}  " +
            " ".join(f"{names[i]}:{en[i] / en.sum():.2f}" for i in np.argsort(-en)[:3]))
    cnt = {c: sum(x["cls"] == c for x in classes) for c in ["REDISTRIBUTION", "WITHIN-EDGE", "OTHER"]}
    n_grp = sum(x["group"] for x in classes)
    log(f"  classes: {cnt}; REDISTRIBUTION involving the correlated group: {n_grp}/{dimN}")
    C = np.corrcoef(Ztr.T)
    log("  input correlation matrix:")
    log("    " + " ".join(f"{n[:8]:>8s}" for n in names))
    for i, n in enumerate(names):
        log(f"    {n[:8]:8s} " + " ".join(f"{C[i, j]:8.3f}" for j in range(d)))
    log("  constant shifts: the per-unit layer-0 bias represents a constant EXACTLY (absorbed by c in Step 2).")
    for i, n in enumerate(names):
        Bi = M[:, i * 9:(i + 1) * 9]
        coef, *_ = np.linalg.lstsq(Bi, np.ones(len(M)), rcond=None)
        log(f"    edge {n:10s}: relative residual of fitting the constant 1 with its 9 columns = "
            f"{np.linalg.norm(Bi @ coef - 1) / np.sqrt(len(M)):.2e}  (distinct values on training rows: {len(np.unique(Ztr[:, i].round(6)))})")
    Sc = np.linalg.svd(M - M.mean(0), compute_uv=False)
    log(f"  sensitivity (no verdict): dim(N) of column-centred M = {int((Sc / Sc[0] < REL_NULL).sum())}")
    RESULTS["step1"] = dict(rel_sv=rel.tolist(), dimN=dimN, classes=classes, counts=cnt, n_group=n_grp, corr=C.tolist())
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))
    axs[0].semilogy(rel, "o-", ms=3); axs[0].axhline(REL_NULL, color="r", ls="--")
    axs[0].set_title(f"M: relative singular values (dim N = {dimN})"); axs[0].set_xlabel("index")
    if dimN:
        im = axs[1].imshow(np.array([x["energy"] for x in classes]), aspect="auto", cmap="magma")
        axs[1].set_xticks(range(d)); axs[1].set_xticklabels(names, rotation=45, fontsize=8)
        axs[1].set_ylabel("N vector"); axs[1].set_title("energy per input edge"); fig.colorbar(im, ax=axs[1])
    im = axs[2].imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
    axs[2].set_xticks(range(d)); axs[2].set_xticklabels(names, rotation=45, fontsize=8)
    axs[2].set_yticks(range(d)); axs[2].set_yticklabels(names, fontsize=8); axs[2].set_title("input correlations"); fig.colorbar(im, ax=axs[2])
    fig.tight_layout(); fig.savefig(FIG / "step1_null_space.png", dpi=110); plt.close(fig)
    log("  --- N is now fixed; trained weights are loaded from here on ---")

    # ================= STEP 2 =================
    log("\n==================== STEP 2: where do models differ? ====================")
    xco = subset(D["batch"], m)["x_co"].numpy()
    y_data = tr.delta.values[m]
    T = load(W_DIR / "delta_s0.pt", D)
    yT = predict(T, delta_head, to_batch(Ztr, str_, x_co=xco))
    # sanity: M theta_j reproduces the continuous part of unit j's pre-activation
    phi = parts(T, Ztr, str_, d)[0]
    err = max(np.abs(M @ theta(T, j, d) - phi[:d, :, j].sum(0)).max() for j in range(6))
    log(f"  check: max |M theta_j - sum of continuous layer-1 edges| on the teacher = {err:.2e}")

    def compare(nA, nB):
        cA, cB = parts(nA, Ztr, str_, d)[2], parts(nB, Ztr, str_, d)[2]
        Cm = corr_matrix(cA - cA.mean(0), cB - cB.mean(0))
        r, k = linear_sum_assignment(1 - np.abs(Cm))
        out = []
        for j, kk in zip(r, k):
            tA, tB = theta(nA, j, d), theta(nB, kk, d)
            X = np.column_stack([M @ tA, np.ones(len(M))])
            (a, c), *_ = np.linalg.lstsq(X, M @ tB, rcond=None)
            dlt = tB - a * tA
            ap = tA @ tB / (tA @ tA)
            dp = tB - ap * tA
            fr = lambda v: float(np.sum((VN.T @ v) ** 2) / np.sum(v ** 2)) if dimN else 0.0
            out.append(dict(j=int(j), k=int(kk), match=float(abs(Cm[j, kk])), frac=fr(dlt), frac_param=fr(dp), a=float(a)))
        return out

    def summary(tag, recs):
        f = np.array([r["frac"] for r in recs]); mt = np.array([r["match"] for r in recs])
        hi, lo = f[mt >= MATCH_SPLIT], f[mt < MATCH_SPLIT]
        fp = np.median([r["frac_param"] for r in recs])
        log(f"  {tag}: {len(recs)} unit pairs; median fraction in N {np.median(f):.3f} (random {dimN / 72:.3f}); "
            f"|r|>=0.8: {np.median(hi) if len(hi) else float('nan'):.3f} (n={len(hi)}), |r|<0.8: {np.median(lo) if len(lo) else float('nan'):.3f} (n={len(lo)}); "
            f"param-space scale (sensitivity) {fp:.3f}; match |r| median {np.median(mt):.3f}")
        return float(np.median(f))

    ind = {i: load(W_DIR / f"fiber2_ind_{i}.pt", D) for i in IND}
    setA = []
    for A_, B_ in itertools.combinations(IND, 2):
        for r in compare(ind[A_], ind[B_]):
            r.update(A=A_, B=B_)
            setA.append(r)
    medA = summary("set A (independent, capped)", setA)

    def own_loss(net, y):
        with torch.no_grad():
            b = to_batch(Ztr, str_, x_co=xco, y=y)
            return float(F.mse_loss(delta_head(net, b), b["y"]) + SMOOTH * net.smoothness())

    setB = {"control3": [], "capped": []}
    per_student, loss_ratio = {}, []
    for s in STUDENTS:
        per_student[s] = {}
        for state, path in [("control3", f"control3_R_s{s}.pt"), ("capped", f"fiber2_stu_{s}.pt")]:
            net = load(W_DIR / path, D)
            recs = compare(T, net)
            for r in recs:
                r["student"] = s
            setB[state] += recs
            per_student[s][state] = dict(median_frac=float(np.median([r["frac"] for r in recs])),
                                         matches=[round(r["match"], 4) for r in sorted(recs, key=lambda r: r["j"])],
                                         loss=own_loss(net, yT))
        loss_ratio.append(per_student[s]["control3"]["loss"] / per_student[s]["capped"]["loss"])
    medB0 = summary("set B, control-3 state", setB["control3"])
    medB1 = summary("set B, capped state   ", setB["capped"])
    log("  per-student unit match |corr| to the teacher (teacher units 0-5), control-3 -> capped:")
    for s in STUDENTS:
        log(f"    {s}: {per_student[s]['control3']['matches']} -> {per_student[s]['capped']['matches']}   "
            f"median frac in N {per_student[s]['control3']['median_frac']:.3f} -> {per_student[s]['capped']['median_frac']:.3f}")
    lr = np.array(loss_ratio)
    log(f"  loss ratio control-3 state / capped state (own training loss): median {np.median(lr):.2f}, range [{lr.min():.2f}, {lr.max():.2f}]")
    n_up = sum(per_student[s]["capped"]["median_frac"] > per_student[s]["control3"]["median_frac"] for s in STUDENTS)
    log(f"  H6 count: students with higher median fraction in N when capped: {n_up}/20")
    RESULTS["step2"] = dict(setA=setA, setB=setB, per_student=per_student, medA=medA, medB_control3=medB0, medB_capped=medB1,
                            loss_ratio=lr.tolist(), h6_count=int(n_up))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for lab, recs in [("A independent (capped)", setA), ("B control-3 state", setB["control3"]), ("B capped", setB["capped"])]:
        axs[0].hist([r["frac"] for r in recs], bins=30, alpha=0.5, label=lab, range=(0, 1))
    axs[0].axvline(dimN / 72, color="k", ls="--", label="random dim(N)/72")
    axs[0].set_xlabel("fraction of gauge-removed layer-1 difference in N"); axs[0].legend(fontsize=8)
    x0 = [per_student[s]["control3"]["median_frac"] for s in STUDENTS]; x1 = [per_student[s]["capped"]["median_frac"] for s in STUDENTS]
    axs[1].scatter(x0, x1); lim = max(x0 + x1) * 1.05 + 1e-9
    axs[1].plot([0, lim], [0, lim], "k:"); axs[1].set_xlabel("control-3 state"); axs[1].set_ylabel("capped state")
    axs[1].set_title(f"H6: per-student median fraction in N ({n_up}/20 above diagonal)")
    fig.tight_layout(); fig.savefig(FIG / "step2_differences.png", dpi=110); plt.close(fig)

    # ================= STEP 3 =================
    log("\n==================== STEP 3: full Hessian (float64, 609 params) ====================")
    meta = json.loads((W_DIR / "fiber2_stu_meta.json").read_text())
    by_grad = sorted(STUDENTS, key=lambda s: meta[str(s)]["grad"])
    picks = [by_grad[0], by_grad[9], by_grad[-1]]
    log(f"  students by final max|grad|: lowest {picks[0]} ({meta[str(picks[0])]['grad']:.2e}), 10th {picks[1]} "
        f"({meta[str(picks[1])]['grad']:.2e}), highest {picks[2]} ({meta[str(picks[2])]['grad']:.2e})")
    used = np.unique(str_)
    models = [(f"ind {i}", W_DIR / f"fiber2_ind_{i}.pt", y_data) for i in IND] + \
             [(f"stu {s}", W_DIR / f"fiber2_stu_{s}.pt", yT) for s in picks]
    hess_rows, flat_all = [], []
    xd = torch.tensor(Ztr, dtype=torch.float64); sd_ = torch.tensor(str_, dtype=torch.long)
    xc = torch.tensor(xco, dtype=torch.float64)
    for tag, path, y in models:
        net = load(path, D).double()
        P0 = {k: v.detach().clone() for k, v in net.named_parameters()}
        keys = ["emb_used", "layers.0.coef", "layers.0.base.weight", "layers.0.base.bias",
                "layers.1.coef", "layers.1.base.weight", "layers.1.base.bias"]
        pieces = [P0["emb.weight"][used]] + [P0[k] for k in keys[1:]]
        shapes = [p.shape for p in pieces]
        sizes = [p.numel() for p in pieces]
        flat0 = torch.cat([p.reshape(-1) for p in pieces])
        yt = torch.tensor(y, dtype=torch.float64)

        def unflat(v):
            out, o = {}, 0
            for k, shp, n in zip(keys, shapes, sizes):
                out[k] = v[o:o + n].reshape(shp); o += n
            emb = P0["emb.weight"].clone()
            emb = emb.index_put((torch.tensor(used),), out.pop("emb_used"))
            out["emb.weight"] = emb
            return out

        def loss(v):
            prm = unflat(v)
            pred = xc * functional_call(net, prm, (xd, sd_))[:, 0]
            pen = 0.0
            for k in ["layers.0.coef", "layers.1.coef"]:
                c = prm[k]
                pen = pen + ((c[..., 2:] - 2 * c[..., 1:-1] + c[..., :-2]) ** 2).sum()
            return F.mse_loss(pred, yt) + SMOOTH * pen

        with torch.no_grad():
            b = to_batch(Ztr, str_, x_co=xco, y=y)
            direct = float(F.mse_loss(delta_head(net, {k: (v.double() if v.is_floating_point() else v) for k, v in b.items()}),
                                      torch.tensor(y, dtype=torch.float64)) + SMOOTH * net.smoothness())
        H = torch.autograd.functional.hessian(loss, flat0).numpy()
        H = (H + H.T) / 2
        ev, evec = np.linalg.eigh(H)
        # index map of the layer-1 continuous block: offsets of layers.0.coef and layers.0.base.weight
        o_coef = sizes[0]
        o_bw = sizes[0] + sizes[1]
        W, I = 6, d + 2
        rows = []
        for q in range(N_FLAT):
            v = evec[:, q]
            blocks = []
            for j in range(W):
                blk = np.concatenate([np.r_[v[o_bw + j * I + i], v[o_coef + (j * I + i) * N_RBF:o_coef + (j * I + i + 1) * N_RBF]] for i in range(d)])
                blocks.append(blk)
            Bm = np.array(blocks)
            tot = (Bm ** 2).sum()
            pooled = float(sum(np.sum((VN.T @ bl) ** 2) for bl in blocks) / tot) if dimN and tot > 0 else 0.0
            per_unit = [float(np.sum((VN.T @ bl) ** 2) / np.sum(bl ** 2)) if dimN and np.sum(bl ** 2) > 0 else 0.0 for bl in blocks]
            en = (Bm.reshape(W, d, 1 + N_RBF) ** 2).sum((0, 2))
            cls, big = classify(en, names)
            rows.append(dict(model=tag, rank=q, eig=float(ev[q]), pooled=pooled, per_unit=per_unit, block_share=float(tot), cls=cls, big=big))
        hess_rows += rows
        log(f"  {tag}: loss via Hessian fn {float(loss(flat0)):.10e} (direct {direct:.10e}); eigenvalues: {int((ev < 0).sum())} negative, "
            f"min {ev[0]:.2e}, max {ev[-1]:.2e}; 10 smallest: " + " ".join(f"{x:.1e}" for x in ev[:N_FLAT]))
        log(f"      flattest-10 pooled overlap with N: " + " ".join(f"{r['pooled']:.2f}" for r in rows) +
            f"; layer-1 block share of ||v||^2: " + " ".join(f"{r['block_share']:.2f}" for r in rows))
        log(f"      classes: " + ", ".join(f"{r['cls'][:5]}{'(' + '/'.join(b_[:6] for b_ in r['big']) + ')' if r['big'] else ''}" for r in rows))
    pooled_all = np.array([r["pooled"] for r in hess_rows])
    log(f"  median pooled overlap over {len(pooled_all)} flattest directions: {np.median(pooled_all):.3f}")
    RESULTS["step3"] = dict(rows=hess_rows, picks=picks, median_pooled=float(np.median(pooled_all)))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for tag in dict.fromkeys(r["model"] for r in hess_rows):
        rr = [r for r in hess_rows if r["model"] == tag]
        axs[0].semilogy(range(N_FLAT), [abs(r["eig"]) for r in rr], "o-", ms=3, label=tag)
        axs[1].scatter([r["block_share"] for r in rr], [r["pooled"] for r in rr], s=14, label=tag)
    axs[0].set_xlabel("rank (smallest first)"); axs[0].set_ylabel("|eigenvalue|"); axs[0].legend(fontsize=7); axs[0].set_title("10 flattest Hessian directions")
    axs[1].axhline(0.5, color="r", ls="--"); axs[1].axhline(dimN / 72, color="k", ls=":")
    axs[1].set_xlabel("share of ||v||^2 in layer-1 continuous block"); axs[1].set_ylabel("pooled overlap with N")
    fig.tight_layout(); fig.savefig(FIG / "step3_hessian.png", dpi=110); plt.close(fig)

    # ================= verdicts =================
    v = {"H1": ("held" if dimN >= 5 else "failed") + f" (dim N = {dimN})"}
    v["H2"] = ("inconclusive (N empty)" if not dimN else ("held" if n_grp >= 0.5 * dimN else "failed") + f" ({n_grp}/{dimN} group REDISTRIBUTION)")
    rnd = dimN / 72
    for h, med in [("H3", medA), ("H4", medB1)]:
        if 3 * rnd > 1:
            v[h] = f"inconclusive (3 x random = {3 * rnd:.2f} > 1, untestable; median {med:.3f})"
        else:
            v[h] = ("held" if med >= 0.5 and med >= 3 * rnd else "failed") + f" (median {med:.3f}; need >= 0.5 and >= {3 * rnd:.3f})"
    v["H5"] = ("held" if np.median(pooled_all) >= 0.5 else "failed") + f" (median pooled overlap {np.median(pooled_all):.3f})"
    v["H6"] = ("held" if n_up >= 15 else "failed") + f" ({n_up}/20)"
    log("\nVERDICTS: " + "; ".join(f"{k}: {x}" for k, x in v.items()))
    RESULTS["verdicts"] = v
    (ROOT / "support_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))


if __name__ == "__main__":
    main()

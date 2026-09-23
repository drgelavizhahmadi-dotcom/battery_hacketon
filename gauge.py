"""Identifiability study of the CALiSol KAN models (see gauge_prereg.md). Research only: writes no submissions.

python3 gauge.py              train/load models -> control -> (only if control passes) Part A + Part B
python3 gauge.py partb        Part B only (seed spread vs error)
Weights: gauge_weights/   Figures: figs/gauge/   Results: gauge_results.json
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from anchor import prepare, rmse
from kan import DELTA_IN, KAN, N_RBF, Scaler, co_props, delta_head, predict, subset, to_batch, train
from kanfull import FULL_IN, full_head

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
ROOT = Path(__file__).parent
W_DIR = ROOT / "gauge_weights"
FIG = ROOT / "figs" / "gauge"
SEEDS = list(range(20))
EPOCHS = 107
K_NN = 10
N_GRID = 100
ACTIVE_FRAC = 0.05
CONTROL_PASS = 0.8
H_SANITY = 0.95
RESULTS = {}


def log(*a):
    print(*a, flush=True)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def corr_matrix(A, B):
    """Column correlations (n, p) x (n, q) -> (p, q); zero where a column is constant."""
    A, B = A - A.mean(0), B - B.mean(0)
    na, nb = np.sqrt((A * A).sum(0)), np.sqrt((B * B).sum(0))
    C = (A.T @ B) / np.outer(na, nb).clip(1e-12)
    C[(na < 1e-12)[:, None] | (nb < 1e-12)[None, :]] = 0.0
    return C


# ------------------------------------------------------------------ data / models
def setup():
    tr, te, ss, X, Xte, ok, use, te_anc, has = prepare(verbose=False)
    salts = sorted(set(tr.salt_name) | set(te.salt_name))
    sidx = {s: i for i, s in enumerate(salts)}
    for f, df in [(X, tr), (Xte, te)]:
        f[["eps_co", "lneta_co", "M_co"]] = co_props(df)
        f["x_cyclic"] = f.frac_high_eps
        f["x_linear"] = 1 - f.frac_high_eps
    s_tr, s_te = tr.salt_name.map(sidx).values, te.salt_name.map(sidx).values
    sc = Scaler([X[use], Xte[has]], DELTA_IN)  # identical to final.py
    D = dict(tag="delta", name="KAN-delta", cols=DELTA_IN, widths=[6, 1], head=delta_head, sc=sc,
             Z=sc(X), s=s_tr, train=use.values, Zte=sc(Xte), ste=s_te, test=has,
             batch=to_batch(sc(X), s_tr, x_co=X.x_co, y=tr.delta.fillna(0)),
             batch_te=to_batch(sc(Xte), s_te, x_co=Xte.x_co))
    Xf, Xfte = X.copy(), Xte.copy()
    for f in (Xf, Xfte):
        f[["eps_co", "lneta_co", "M_co"]] = f[["eps_co", "lneta_co", "M_co"]].fillna(0.0)
    scf = Scaler([Xf[ok], Xfte], FULL_IN)  # identical to kanfull.py
    Fm = dict(tag="full", name="KAN-full", cols=FULL_IN, widths=[8, 1], head=full_head, sc=scf,
              Z=scf(Xf), s=s_tr, train=ok.values, Zte=scf(Xfte), ste=s_te, test=has,
              batch=to_batch(scf(Xf), s_tr, y=tr.log_k))
    for M in (D, Fm):
        M["n_salt"], M["d_num"], M["salts"] = len(salts), len(M["cols"]), salts
        M["names"] = M["cols"] + ["salt"]
    return tr, te, te_anc, D, Fm


def get_models(tag, M, batch, mask, seeds=SEEDS):
    W_DIR.mkdir(exist_ok=True)
    nets, n_new = [], 0
    for s in seeds:
        p = W_DIR / f"{tag}_s{s}.pt"
        net = KAN(M["d_num"], M["widths"], M["n_salt"])
        if p.exists():
            net.load_state_dict(torch.load(p))
        else:
            net = train(M["widths"], M["d_num"], M["n_salt"], M["head"], subset(batch, mask), epochs=EPOCHS, seed=s)[0]
            torch.save(net.state_dict(), p)
            n_new += 1
        nets.append(net.eval())
    log(f"  {tag}: {len(nets)} models ({n_new} trained, {len(nets) - n_new} loaded)")
    return nets


# ------------------------------------------------------------------ decomposition
@torch.no_grad()
def layer0_edges(net, Z, sidx, d_num):
    """(I, n, W): numeric edges phi_ij(z_i) and the salt edge (sum of the two embedding edges); bias excluded."""
    Zt, L0 = torch.tensor(Z, dtype=torch.float32), net.layers[0]
    out = [L0.edge(i, Zt[:, i]) for i in range(d_num)]
    e = torch.tanh(net.emb(torch.tensor(sidx, dtype=torch.long)))
    out.append(L0.edge(d_num, e[:, 0]) + L0.edge(d_num + 1, e[:, 1]))
    return torch.stack(out).numpy()


@torch.no_grad()
def psi(net, s):
    """psi_j(s_j) for each unit (layer-1 edge applied to tanh(s_j)); bias excluded."""
    st, L1 = torch.tanh(torch.tensor(s, dtype=torch.float32)), net.layers[1]
    return np.stack([L1.edge(j, st[:, j])[:, 0].numpy() for j in range(s.shape[1])], 1)


def parts(net, Z, sidx, d_num):
    phi = layer0_edges(net, Z, sidx, d_num)
    s = phi.sum(0) + net.layers[0].base.bias.detach().numpy()
    ps = psi(net, s)
    return phi, s, ps, ps.sum(1) + float(net.layers[1].base.bias)


class Gauge:
    """Steps 1-3 of the gauge fixing, from training rows; step 4 (perm) is set by align()."""

    def __init__(self, net, Z, sidx, d_num):
        phi, s, ps, _ = parts(net, Z, sidx, d_num)
        self.m = phi.mean(1)
        self.mu, self.sd = s.mean(0), s.std(0) + 1e-12
        shat = (s - self.mu) / self.sd
        self.sign = np.array([1.0 if corr(shat[:, j], ps[:, j]) >= 0 else -1.0 for j in range(s.shape[1])])
        self.psi_mean = ps.mean(0)
        self.perm = np.arange(s.shape[1])
        self.contrib_train = ps - self.psi_mean

    def centred(self, phi):
        return phi - self.m[:, None, :]

    def fixed(self, phi):
        return (self.sign * self.centred(phi) / self.sd)[..., self.perm]

    def contrib(self, ps):
        return (ps - self.psi_mean)[:, self.perm]


def align(ref, g):
    C = corr_matrix(ref.contrib_train[:, ref.perm], g.contrib_train)
    r, k = linear_sum_assignment(1 - np.abs(C))
    g.perm = k[np.argsort(r)]
    return C[np.arange(len(g.perm)), g.perm]


def refit_check(net, g, Z, sidx, d_num):
    """How far is step 2 from an exact symmetry in the fixed-grid basis: refit psi_hat(u) into the layer-1 basis."""
    _, s, ps, h = parts(net, Z, sidx, d_num)
    grid = np.linspace(-1, 1, N_RBF)
    hw = 2 / (N_RBF - 1)
    fit = np.zeros_like(ps)
    for j in range(s.shape[1]):
        u = np.tanh(g.sign[j] * (s[:, j] - g.mu[j]) / g.sd[j])
        Bm = np.column_stack([u / (1 + np.exp(-u)), np.exp(-(((u[:, None] - grid) / hw) ** 2)), np.ones_like(u)])
        coef, *_ = np.linalg.lstsq(Bm, ps[:, j], rcond=None)
        fit[:, j] = Bm @ coef
    dh = fit.sum(1) - ps.sum(1)
    return float(np.abs(dh).mean()), float(np.abs(dh).max()), float(np.sqrt(np.mean(dh ** 2)) / (h.std() + 1e-12))


# ------------------------------------------------------------------ measures
def grid_inputs(M):
    Ztr, str_ = M["Z"][M["train"]], M["s"][M["train"]]
    med = np.median(Ztr, 0)
    modal = np.bincount(str_).argmax()
    sweeps = []
    for i in range(M["d_num"]):
        Zg = np.tile(med, (N_GRID, 1))
        Zg[:, i] = np.linspace(Ztr[:, i].min(), Ztr[:, i].max(), N_GRID)
        sweeps.append((Zg, np.full(N_GRID, modal)))
    present = np.unique(str_)
    sweeps.append((np.tile(med, (len(present), 1)), present))
    return sweeps


def x_co_of(M, Z):
    return M["sc"].inv(M["cols"].index("x_co"), Z[:, M["cols"].index("x_co")]) if "x_co" in M["cols"] else None


def grid_measures(M, nets, gauges):
    """Per input i: arrays (seeds, points, W) for a/b/c/d and (seeds, points) for h (+delta)."""
    out = []
    for i, (Zg, sg) in enumerate(grid_inputs(M)):
        a, b, c, d, h = [], [], [], [], []
        for net, g in zip(nets, gauges):
            phi, s, ps, hh = parts(net, Zg, sg, M["d_num"])
            a.append(phi[i]); b.append(g.centred(phi)[i]); c.append(g.fixed(phi)[i]); d.append(g.contrib(ps)); h.append(hh)
        r = {k: np.array(v) for k, v in zip("abcd", (a, b, c, d))}
        r["e_h"] = np.array(h)[..., None]
        xc = x_co_of(M, Zg)
        if xc is not None:
            r["e_delta"] = (np.array(h) * xc)[..., None]
        out.append(r)
    return out


def row_measures(M, nets, gauges, Z, sidx):
    a, b, c, d, h = [], [], [], [], []
    for net, g in zip(nets, gauges):
        phi, s, ps, hh = parts(net, Z, sidx, M["d_num"])
        a.append(phi); b.append(g.centred(phi)); c.append(g.fixed(phi)); d.append(g.contrib(ps)); h.append(hh)
    # edges: (S, I, n, W) -> (S, n, I*W)
    ed = lambda v: np.transpose(np.array(v), (0, 2, 1, 3)).reshape(len(v), Z.shape[0], -1)
    r = {"a": ed(a), "b": ed(b), "c": ed(c), "d": np.array(d), "e_h": np.array(h)[..., None]}
    xc = x_co_of(M, Z)
    if xc is not None:
        r["e_delta"] = (np.array(h) * xc)[..., None]
    return r


def nv_parts(F):
    F = F.reshape(F.shape[0], F.shape[1], -1)
    return float(F.var(0).mean(0).sum()), float(F.mean(0).var(0).sum())


def nv_pooled(list_of_F):
    num = den = 0.0
    for F in list_of_F:
        n, d = nv_parts(F)
        num, den = num + n, den + d
    return num / den if den > 0 else np.nan


def nv_region(Ftrain, Fregion):
    """mean over region rows of cross-seed var, over the global (all training rows) variance of the seed mean."""
    den = float(Ftrain.mean(0).var(0).sum())
    return float(Fregion.var(0).mean(0).sum()) / den


def knn_density(M):
    Ztr = M["Z"][M["train"]]
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-12
    Str, Ste = (Ztr - mu) / sd, (M["Zte"][M["test"]] - mu) / sd
    s_tr, s_te = M["s"][M["train"]], M["ste"][M["test"]]
    dtr, dte = np.full(len(Str), np.nan), np.full(len(Ste), np.nan)
    for s in np.unique(s_tr):
        m = s_tr == s
        k = min(K_NN, m.sum() - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Str[m])
        dtr[m] = nn.kneighbors(Str[m])[0][:, 1:].mean(1)
        mt = s_te == s
        if mt.any():
            dte[mt] = nn.kneighbors(Ste[mt], n_neighbors=k)[0].mean(1)
    q1, q3 = np.quantile(dtr, [0.25, 0.75])
    return dtr, dte, dtr <= q1, dtr >= q3


def gauge_all(M, nets, ref=None):
    Ztr, str_ = M["Z"][M["train"]], M["s"][M["train"]]
    gauges = [Gauge(n, Ztr, str_, M["d_num"]) for n in nets]
    ref = ref or gauges[0]
    matches = np.array([align(ref, g) for g in gauges])
    return gauges, matches


def check_decomposition(M, net):
    Ztr, str_ = M["Z"][M["train"]], M["s"][M["train"]]
    _, _, _, h = parts(net, Ztr, str_, M["d_num"])
    with torch.no_grad():
        ref = net(torch.tensor(Ztr, dtype=torch.float32), torch.tensor(str_, dtype=torch.long))[:, 0].numpy()
    return float(np.abs(h - ref).max())


# ------------------------------------------------------------------ control
def smooth_teacher(M, real_delta, seed=12345):
    torch.manual_seed(seed)
    T = KAN(M["d_num"], M["widths"], M["n_salt"])
    with torch.no_grad():
        for L in T.layers:
            walk = torch.randn(L.coef.shape).cumsum(-1)
            walk = walk - walk.mean(-1, keepdim=True)
            L.coef.copy_(walk / walk.std() * 0.5)
            L.base.weight.copy_(torch.randn_like(L.base.weight) * 0.5)
            L.base.bias.zero_()
        b = subset(M["batch"], M["train"])
        dT = delta_head(T, b).numpy()
        a = real_delta.std() / dT.std()
        L1 = T.layers[1]
        L1.coef.mul_(a); L1.base.weight.mul_(a); L1.base.bias.mul_(a)
    return T.eval()


def run_control(M, real_nets, tr):
    log("\n==================== CONTROL: teacher-student recovery ====================")
    b = subset(M["batch"], M["train"])
    y = tr.delta.values[M["train"]]
    ens = np.mean([predict(n, delta_head, b) for n in real_nets], 0)
    sigma = float((y - ens).std())
    T = smooth_teacher(M, y)
    yT = predict(T, delta_head, b)
    rng = np.random.default_rng(0)
    y_ctrl = np.zeros(len(tr))
    y_ctrl[M["train"]] = yT + rng.normal(0, sigma, len(yT))
    bc = dict(M["batch"])
    bc["y"] = torch.tensor(y_ctrl, dtype=torch.float32)
    log(f"  noise std (real KAN-delta in-sample residual, 20 seeds) = {sigma:.4f}; teacher delta std = {yT.std():.4f}; "
        f"SNR (var) = {yT.var() / sigma ** 2:.2f}")
    students = get_models("student", M, bc, M["train"])

    Ztr, str_ = M["Z"][M["train"]], M["s"][M["train"]]
    gT = Gauge(T, Ztr, str_, M["d_num"])
    gS, matches = gauge_all(M, students, ref=gT)
    _, _, dense, sparse = knn_density(M)
    phiT = gT.fixed(parts(T, Ztr, str_, M["d_num"])[0])  # (I, n, W)
    varT = phiT.var(1)  # (I, W)
    active = varT >= ACTIVE_FRAC * varT.max(0, keepdims=True)
    hT = parts(T, Ztr, str_, M["d_num"])[3]
    rec = {"all": [], "dense": [], "sparse": []}
    per_edge = np.zeros((len(students),) + varT.shape)
    h_corr = []
    for k, (net, g) in enumerate(zip(students, gS)):
        phiS = g.fixed(parts(net, Ztr, str_, M["d_num"])[0])
        h_corr.append(corr(parts(net, Ztr, str_, M["d_num"])[3], hT))
        for i in range(varT.shape[0]):
            for j in range(varT.shape[1]):
                per_edge[k, i, j] = corr(phiS[i, dense, j], phiT[i, dense, j])
                if active[i, j]:
                    rec["all"].append(corr(phiS[i, :, j], phiT[i, :, j]))
                    rec["dense"].append(per_edge[k, i, j])
                    rec["sparse"].append(corr(phiS[i, sparse, j], phiT[i, sparse, j]))
    med = {k: float(np.median(v)) for k, v in rec.items()}
    frac90 = float(np.mean(np.array(rec["dense"]) >= 0.9))
    passed = med["dense"] >= CONTROL_PASS
    h_ok = float(np.median(h_corr)) >= H_SANITY
    log(f"  active teacher edges: {int(active.sum())} of {active.size}")
    log(f"  student-teacher unit match |corr| (contributions): median {np.median(np.abs(matches)):.3f}, "
        f"min {np.abs(matches).min():.3f}; per unit (median over seeds): {np.round(np.median(np.abs(matches), 0), 3).tolist()}")
    log(f"  prediction recovery corr(h_student, h_teacher): median {np.median(h_corr):.4f}, min {np.min(h_corr):.4f} "
        f"-> sanity {'OK' if h_ok else 'FAILED (training, not gauge)'}")
    log(f"  gauge-fixed edge recovery corr, active edges x 20 seeds: median all {med['all']:.3f}, dense {med['dense']:.3f}, "
        f"sparse {med['sparse']:.3f}; fraction >= 0.9 (dense) {frac90:.2f}")
    log("  per-input median dense recovery over active edges:")
    for i, n in enumerate(M["names"]):
        v = per_edge[:, i, :][:, active[i]]
        log(f"    {n:10s} n_active={int(active[i].sum())}  median corr {np.median(v) if v.size else np.nan:.3f}")
    log(f"  CONTROL {'PASSED' if passed else 'FAILED'} (criterion: median dense recovery >= {CONTROL_PASS})")
    RESULTS["control"] = dict(sigma=sigma, median=med, frac90_dense=frac90, h_corr_median=float(np.median(h_corr)),
                              match_abs_median=float(np.median(np.abs(matches))), passed=bool(passed), h_ok=bool(h_ok),
                              n_active=int(active.sum()))

    # figure: teacher vs recovered edges on the grid for the 6 largest active teacher edges
    FIG.mkdir(parents=True, exist_ok=True)
    sweeps = grid_inputs(M)
    order = np.dstack(np.unravel_index(np.argsort(-(varT * active), axis=None), varT.shape))[0][:6]
    fig, axs = plt.subplots(2, 4, figsize=(18, 7.5))
    for ax, (i, j) in zip(axs.flat[:6], order):
        Zg, sg = sweeps[i]
        xs = M["sc"].inv(i, Zg[:, i]) if i < M["d_num"] else np.arange(len(sg))
        for net, g in zip(students, gS):
            ax.plot(xs, g.fixed(parts(net, Zg, sg, M["d_num"])[0])[i][:, j], color="tab:blue", alpha=0.25, lw=0.8)
        ax.plot(xs, gT.fixed(parts(T, Zg, sg, M["d_num"])[0])[i][:, j], color="k", lw=2, label="teacher")
        ax.set_title(f"{M['names'][i]} -> unit {j}  (median corr {np.median(per_edge[:, i, j]):.2f})", fontsize=9)
    axs.flat[0].legend()
    ax = axs.flat[6]
    ax.hist(rec["dense"], bins=30, color="tab:blue")
    ax.axvline(CONTROL_PASS, color="r", ls="--")
    ax.set_title("recovery corr, active edges (dense rows)")
    ax = axs.flat[7]
    ax.hist(np.abs(matches[:, :]).ravel(), bins=30, color="tab:green")
    ax.set_title("student-teacher unit match |corr|")
    fig.suptitle(f"Control: teacher vs 20 gauge-fixed students ({'PASSED' if passed else 'FAILED'})")
    fig.tight_layout(); fig.savefig(FIG / "control_recovery.png", dpi=110); plt.close(fig)
    return passed and h_ok


# ------------------------------------------------------------------ Part A (real data)
def part_a(M, nets, tr):
    log(f"\n==================== PART A: {M['name']} ====================")
    log(f"  decomposition check: max |h(parts) - net(x)| = {check_decomposition(M, nets[0]):.2e}")
    gauges, matches = gauge_all(M, nets)
    Ztr, str_ = M["Z"][M["train"]], M["s"][M["train"]]
    rf = np.array([refit_check(n, g, Ztr, str_, M["d_num"]) for n, g in zip(nets, gauges)])
    log("  step 2 applied functionally -> prediction change exactly 0 by construction.")
    log(f"  if psi_hat is re-fitted into the fixed RBF basis instead: mean|dh| {rf[:, 0].mean():.2e}, "
        f"max|dh| {rf[:, 1].max():.2e}, rmse(dh)/std(h) {rf[:, 2].mean():.2e} (mean over seeds)")
    m = np.abs(matches[1:])
    log(f"  unit match |corr| vs seed 0 (seeds 1-19): per unit median {np.round(np.median(m, 0), 3).tolist()}, "
        f"min {np.round(m.min(0), 3).tolist()}")
    n_low = int((m.min(1) < 0.8).sum())
    log(f"  seeds with at least one unit |corr| < 0.8: {n_low}/19")

    G = grid_measures(M, nets, gauges)
    keys = ["a", "b", "c", "d", "e_h"] + (["e_delta"] if "e_delta" in G[0] else [])
    ladder = {k: nv_pooled([g[k] for g in G]) for k in keys}
    log("  normalised cross-seed variance, grid (pooled over inputs):")
    for k in keys:
        log(f"    {k:8s} {ladder[k]:.4f}")
    per_input_c = {M["names"][i]: nv_pooled([G[i]["c"]]) for i in range(len(G))}
    per_input_b = {M["names"][i]: nv_pooled([G[i]["b"]]) for i in range(len(G))}
    log("  per input NV (b) centred -> (c) fixed:")
    for n in M["names"]:
        log(f"    {n:10s} {per_input_b[n]:.4f} -> {per_input_c[n]:.4f}")

    dtr, dte, dense, sparse = knn_density(M)
    log(f"  kNN distance: train Q1 {np.quantile(dtr, 0.25):.3f}, median {np.median(dtr):.3f}, Q3 {np.quantile(dtr, 0.75):.3f}; "
        f"test (DEC) median {np.nanmedian(dte):.3f}, fraction of test beyond train Q3 {np.nanmean(dte >= np.quantile(dtr, 0.75)):.2f}")
    Rtr = row_measures(M, nets, gauges, Ztr, str_)
    Rte = row_measures(M, nets, gauges, M["Zte"][M["test"]], M["ste"][M["test"]])
    region = {}
    for k in keys:
        region[k] = dict(train=nv_region(Rtr[k], Rtr[k]), dense=nv_region(Rtr[k], Rtr[k][:, dense]),
                         sparse=nv_region(Rtr[k], Rtr[k][:, sparse]), test=nv_region(Rtr[k], Rte[k]))
    log("  normalised cross-seed variance by region (row-based, denominator = all training rows):")
    log(f"    {'measure':8s} {'train':>9s} {'dense':>9s} {'sparse':>9s} {'test':>9s}")
    for k in keys:
        r = region[k]
        log(f"    {k:8s} {r['train']:9.4f} {r['dense']:9.4f} {r['sparse']:9.4f} {r['test']:9.4f}")
    RESULTS[M["tag"]] = dict(ladder=ladder, region=region, per_input_c=per_input_c, per_input_b=per_input_b,
                            match_median=np.median(m, 0).tolist(), match_min=m.min(0).tolist(), n_low=n_low, refit=rf.mean(0).tolist())
    return dict(G=G, gauges=gauges, matches=matches, ladder=ladder, region=region)


def p5_shared(D, Dres, Fm, Fres):
    shared = ["invT", "mix_lneta", "lneta_co", "mix_eps", "eps_co", "molal"]
    dv = [Dres["G"][D["names"].index(n)]["c"] for n in shared]
    fv = [Fres["G"][Fm["names"].index(n)]["c"] for n in shared]
    # salt on the 4 salts present in KAN-delta's training rows
    salts4 = np.unique(D["s"][D["train"]])
    Zmed = np.median(Fm["Z"][Fm["train"]], 0)
    Zs = np.tile(Zmed, (len(salts4), 1))
    fsalt = np.array([g.fixed(parts(n, Zs, salts4, Fm["d_num"])[0])[-1] for n, g in zip(Fres["nets"], Fres["gauges"])])
    dv.append(Dres["G"][-1]["c"]); fv.append(fsalt)
    nd, nf = nv_pooled(dv), nv_pooled(fv)
    log(f"\n  P5 shared inputs, NV(c) pooled: KAN-delta {nd:.4f}, KAN-full {nf:.4f} (ratio {nf / nd:.2f})")
    RESULTS["p5"] = dict(delta=nd, full=nf)
    return nd, nf


# ------------------------------------------------------------------ Part B
def part_b(D, tr):
    log("\n==================== PART B: does seed spread flag errors? (KAN-delta) ====================")
    y = tr.log_k.values
    out = {}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    covs = [1.0, 0.9, 0.8, 0.7, 0.5]
    rng = np.random.default_rng(0)
    for ax, (fold, combo) in zip(axs, [("A", "EA+PC"), ("B", "EC+PC")]):
        ho = (tr.combo == combo).values & D["train"]
        dt = D["train"] & ~ho
        nets = get_models(f"partb{fold}", D, D["batch"], dt)
        P = np.array([tr.anchor.values[ho] + predict(n, delta_head, subset(D["batch"], ho)) for n in nets])
        pm, sd = P.mean(0), P.std(0)
        err = np.abs(pm - y[ho])
        rho = float(spearmanr(sd, err).correlation)
        order = np.argsort(sd)
        rows = []
        for c in covs:
            n = int(round(c * len(err)))
            keep = order[:n]
            rnd = [rmse(pm[idx], y[ho][idx]) for idx in (rng.choice(len(err), n, replace=False) for _ in range(100))]
            rows.append(dict(coverage=c, std_rmse=rmse(pm[keep], y[ho][keep]), random_mean=float(np.mean(rnd)),
                             random_p5=float(np.percentile(rnd, 5)), random_p95=float(np.percentile(rnd, 95))))
        out[fold] = dict(n=int(ho.sum()), spearman=rho, curve=rows)
        log(f"  fold {fold} ({combo}, n={ho.sum()}): Spearman(seed std, |err|) = {rho:+.3f}")
        for r in rows:
            log(f"    coverage {r['coverage']:.0%}: RMSE std-abstain {r['std_rmse']:.4f} | random mean {r['random_mean']:.4f} "
                f"[5-95%: {r['random_p5']:.4f}-{r['random_p95']:.4f}]")
        cc = [r["coverage"] for r in rows]
        ax.plot(cc, [r["std_rmse"] for r in rows], "o-", label="abstain on highest seed std")
        ax.plot(cc, [r["random_mean"] for r in rows], "s--", color="gray", label="random abstention (mean)")
        ax.fill_between(cc, [r["random_p5"] for r in rows], [r["random_p95"] for r in rows], color="gray", alpha=0.2, label="random 5-95%")
        ax.invert_xaxis(); ax.set_xlabel("coverage"); ax.set_ylabel("RMSE of retained rows (log10)")
        ax.set_title(f"fold {fold} ({combo}), Spearman {rho:+.2f}")
    axs[0].legend(fontsize=8)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG / "retention.png", dpi=110); plt.close(fig)
    RESULTS["partb"] = out


# ------------------------------------------------------------------ figures (Part A)
def figures(D, Dres, Fm, Fres):
    # variance ladder
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, (M, R) in zip(axs, [(D, Dres), (Fm, Fres)]):
        keys = list(R["ladder"])
        x = np.arange(len(keys))
        vals = {"grid": [R["ladder"][k] for k in keys]}
        for reg in ["dense", "sparse", "test"]:
            vals[reg] = [R["region"][k][reg] for k in keys]
        for o, (lab, v) in enumerate(vals.items()):
            ax.bar(x + (o - 1.5) * 0.2, v, 0.2, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(["(a) raw", "(b) centred", "(c) fixed", "(d) contrib", "(e) h", "(e) delta"][:len(keys)])
        ax.set_yscale("log"); ax.set_title(M["name"]); ax.set_ylabel("normalised cross-seed variance")
    axs[0].legend()
    fig.suptitle("Cross-seed variance ladder (grid = 1-D sweeps; dense/sparse/test = row-based, shared denominator)")
    fig.tight_layout(); fig.savefig(FIG / "variance_ladder.png", dpi=110); plt.close(fig)

    # edges before / after for 1000/T (KAN-delta)
    i = D["names"].index("invT")
    Zg, sg = grid_inputs(D)[i]
    xs = D["sc"].inv(i, Zg[:, i])
    W = D["widths"][0]
    fig, axs = plt.subplots(2, W, figsize=(3.2 * W, 6.4), sharex=True)
    for net, g in zip(Dres["nets"], Dres["gauges"]):
        phi = parts(net, Zg, sg, D["d_num"])[0]
        for j in range(W):
            axs[0, j].plot(xs, g.centred(phi)[i][:, j], alpha=0.4, lw=0.8)
            axs[1, j].plot(xs, g.fixed(phi)[i][:, j], alpha=0.4, lw=0.8)
    med = np.median(np.abs(Dres["matches"][1:]), 0)
    for j in range(W):
        axs[0, j].set_title(f"unit index {j} (as trained)", fontsize=9)
        axs[1, j].set_title(f"aligned unit {j} (median |r| {med[j]:.2f})", fontsize=9)
        axs[1, j].set_xlabel("1000/T (1/K)")
    axs[0, 0].set_ylabel("(b) centred only"); axs[1, 0].set_ylabel("(c) gauge-fixed + aligned")
    fig.suptitle("KAN-delta layer-1 edge for 1000/T, 20 seeds")
    fig.tight_layout(); fig.savefig(FIG / "edges_before_after.png", dpi=110); plt.close(fig)

    # unit matching heatmaps
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (M, R) in zip(axs, [(D, Dres), (Fm, Fres)]):
        m = np.abs(R["matches"][1:])
        im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xlabel("aligned unit (reference = seed 0)"); ax.set_ylabel("seed"); ax.set_yticks(range(19)); ax.set_yticklabels(range(1, 20), fontsize=7)
        ax.set_title(f"{M['name']}: |corr| of matched unit contributions")
        for (r, c), v in np.ndenumerate(m):
            if v < 0.8:
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6, color="w")
    fig.colorbar(im, ax=axs, shrink=0.8)
    fig.savefig(FIG / "unit_matching.png", dpi=110); plt.close(fig)


def main():
    tr, te, te_anc, D, Fm = setup()
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "partb":
        part_b(D, tr)
        (ROOT / "gauge_results.json").write_text(json.dumps(RESULTS, indent=1))
        return
    log("training / loading models")
    Dnets = get_models("delta", D, D["batch"], D["train"])
    Fnets = get_models("full", Fm, Fm["batch"], Fm["train"])
    # reproducibility: seeds 0-4 of KAN-delta must reproduce submission_final.csv on anchored rows
    import pandas as pd
    fin = pd.read_csv(ROOT / "submission_final.csv").log_k.values
    k5 = te_anc[D["test"]] + np.mean([predict(n, delta_head, subset(D["batch_te"], D["test"])) for n in Dnets[:5]], 0)
    log(f"  reproducibility: seeds 0-4 vs submission_final.csv anchored rows, max|diff| = {np.abs(np.clip(k5, -2, None) - fin[D['test']]).max():.2e}")

    ok = run_control(D, Dnets, tr)
    (ROOT / "gauge_results.json").write_text(json.dumps(RESULTS, indent=1))
    if not ok:
        log("\nSTOP: control failed -> Part A/B on real data are not interpretable with this procedure. Not run.")
        sys.exit(2)
    Dres = part_a(D, Dnets, tr); Dres["nets"] = Dnets
    Fres = part_a(Fm, Fnets, tr); Fres["nets"] = Fnets
    p5_shared(D, Dres, Fm, Fres)
    figures(D, Dres, Fm, Fres)
    part_b(D, tr)
    (ROOT / "gauge_results.json").write_text(json.dumps(RESULTS, indent=1))
    log("\nsaved gauge_results.json, figures in figs/gauge/")


if __name__ == "__main__":
    main()

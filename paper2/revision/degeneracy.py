"""paper2/revision/degeneracy.py: POST-HOC, DESCRIPTIVE follow-up to revision step 2 (no verdict changes, no pre-registration).
Are the real Arm B "certified" endpoints genuine finite minima or plateaus on the way to infinity? Does the boundary follow
convex (A, C) vs non-convex (B)? Run from the repo root with the pinned interpreter:
  /usr/bin/python3 paper2/revision/degeneracy.py >> paper2/revision/degeneracy_output.txt
Inputs: archive 11 (step-2 weights) + the early states, SHA-verified against weights_manifest.txt, held in memory. Replays and
restarts are cached in gauge_weights/degen_* (SHA-indexed) so an interruption resumes."""
import io
import json
import sys
import time
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

import freeze as FZ
import synth as SY
import sc_run as SR
from kan import DELTA_IN, KAN, subset
from stage2 import extract_verified, in_memory

NEAR = 1e-8
CS = [1.01, 1.1, 1.5, 2, 5, 10]
NB = SY.NB.double()
OUT = {}


def log(*a):
    print(*a, flush=True)


def hidden(prm, net, z_in):
    """pre-activations s of layer 1 (n x units) from a parameter dict"""
    L = net.layers[0]
    rbf = L.rbf(z_in)
    return F.silu(z_in) @ prm["layers.0.base.weight"].T + prm["layers.0.base.bias"] + torch.einsum("bik,oik->bo", rbf, prm["layers.0.coef"])


WALL_HARD = FZ.WALL + 300


def restart_child(key):
    """separate process: restart trust-exact at the step-2 endpoint of real Arm B `key` with GRAD_TOL 1e-9; checkpoints the
    last accepted iterate (every Hessian evaluation) to gauge_weights/degen_ckpt_<key>.npy; saves the result via SR.cached."""
    i = int(key[3:])
    blobs = in_memory(extract_verified([f"delta_s{i}.pt", f"freeze_B-R_{key}.pt"]))
    ld = lambda n: torch.load(io.BytesIO(blobs[n]), weights_only=False)
    S = SR.setup(); n_salt = len(S["salts"]); ball = SR.dbl(subset(S["B"], S["use"]))
    k = KAN(len(DELTA_IN), [6, 1], n_salt); k.load_state_dict({n: t.float() for n, t in ld(f"delta_s{i}.pt").items()}, strict=False); net = k.double().eval()
    parts = FZ.real_parts(net, ball)
    full = {n: v for n, v in net.named_parameters()}
    st = ld(f"freeze_B-R_{key}.pt"); xe = st["x"].numpy() if torch.is_tensor(st["x"]) else st["x"]
    pe = FZ.FlatFree(net, full, FZ.L1, parts).full_at(xe)
    flat_e = FZ.FlatFree(net, pe, FZ.L1, parts)
    ck = REPO / "gauge_weights" / f"degen_ckpt_{key}.npy"
    orig = flat_e.hessian

    def hess(x):
        np.save(ck, np.asarray(x)); return orig(x)
    flat_e.hessian = hess
    FZ.GRAD_TOL = 1e-9
    SR.cached(f"degen_restart_B-R_{key}.pt", lambda: FZ.to_cache(dict(zip(("x", "info"), FZ.run(flat_e, FZ.REAL_BLOCKS)))))


def restart_probe(key, flat_e):
    """parent side: cached result, else run the child with a hard timeout; on timeout use the last checkpoint"""
    import subprocess
    from certify import certify_point
    name = f"degen_restart_B-R_{key}.pt"
    if not (REPO / "gauge_weights" / name).exists():
        try:
            subprocess.run([sys.executable, str(Path(__file__)), "--restart", key], cwd=REPO, timeout=WALL_HARD,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            ck = REPO / "gauge_weights" / f"degen_ckpt_{key}.npy"
            x = np.load(ck) if ck.exists() else flat_e.x0.copy()
            cert, _, _ = certify_point(flat_e, x)
            SR.cached(name, lambda: FZ.to_cache(dict(x=x, info=dict(stop="2 h cap (hard kill of the probe process; last checkpointed iterate)",
                                                                    iters=-1, cert=cert, logged=[]))))
    obj, _ = SR.cached(name, lambda: (_ for _ in ()).throw(RuntimeError("restart result missing")))
    return obj


def main():
    log("#" * 100)
    log(f"paper2/revision/degeneracy.py {time.strftime('%Y-%m-%d %H:%M:%S')}: POST-HOC, DESCRIPTIVE follow-up to revision step 2 (190544d).")
    log("No pre-registration (the step-2 data have been seen); nothing here changes a verdict.")
    log("#" * 100)
    log("DISCREPANCIES / choices (files win):")
    log("  G1 step 2 logged the data term only every 25 iterates, so 'first 10 iterations' (item 4) comes from a deterministic REPLAY of")
    log("     the first 10 trust-exact iterations from the early state (same code); for runs that ended within 10 iterations the replay")
    log("     endpoint is checked against the cached endpoint")
    log("  G2 a hidden unit counts as 'saturated' when |s| > 3 on >= 10% of the training rows at the endpoint (fixed before running)")
    log("  G3 'parameters feeding saturated units': Arm B = that unit's layer-1 RBF coefficients, SiLU weights and bias; Arm A = the")
    log("     layer-2 RBF coefficients and SiLU weight of that unit's output edge (the layer-2 bias is not unit-specific: 'rest')")
    log("  G4 'layer-1 affine RBF' share = energy of each near-flat eigenvector's layer-1 coefficient part projected on span{1, k}")
    log("     (per edge), over the eigenvector's total energy; it is 0 by construction in Arm A (layer 1 frozen)")
    log("  G5 restart probe: same trust-exact runner and caps (500 iterations, 2 h, timer), convergence gradient tolerance 1e-9 instead")
    log("     of 1e-7 (x100 tighter), started at the step-2 endpoint; ||theta||^2 = all parameters")
    log("  G7 (after crash record 2) each real Arm B restart probe runs in a separate process killed at the 2 h cap + 5 min; the last")
    log("     accepted iterate is checkpointed at every Hessian evaluation and used if the probe is killed (iterations then reported as -1)")
    log("  G6 scaling probes use the Arm B loss (data + 1e-3 * sum D2); 'non-increasing up to c = 10' = loss(c) never rises over")
    log("     c = 1, 1.01, 1.1, 1.5, 2, 5, 10")
    log("  references for item 2: certified mu = 1e-4 minima n_null median 1 (range 0-3); SC-KAN-delta 0 (certify_results.json, arch/sc)")

    man = [l.split()[0].split("/")[-1] for l in (REPO / "weights_manifest.txt").read_text().splitlines()
           if l.startswith("gauge_weights/freeze_") and l.split()[-1] == "gauge_weights_2026-10-17k.tar.gz"]
    want = [n for n in man if n.startswith(("freeze_A-", "freeze_B-", "freeze_D_"))]
    names = want + [f"delta_s{i}.pt" for i in range(6)] + [f"synth_correlated_{l}_s{s}.pt" for l in ("P0", "P1") for s in range(8)]
    log("\n==================== inputs (SHA-256 verified, held in memory) ====================")
    blobs = in_memory(extract_verified(names))
    ld = lambda n: torch.load(io.BytesIO(blobs[n]), weights_only=False)

    S = SR.setup(); n_salt = len(S["salts"]); ball = SR.dbl(subset(S["B"], S["use"]))
    res = json.loads((REPO / "paper2/synth_results.json").read_text())
    teacher = SY.build_teacher(res["teacher_scale"])
    ci = list(SY.CONDS).index("correlated")
    X = SY.make_inputs(SY.CONDS["correlated"], 1000 + ci); Xv = SY.make_inputs(SY.CONDS["correlated"], 2000 + ci)
    y0 = SY.predict(teacher, X); sig = 0.05 * y0.std(); rng = np.random.default_rng(5000 + ci); y = y0 + sig * rng.standard_normal(SY.N)
    Xd = torch.tensor(X, dtype=torch.float64)

    def kan_from(sd):
        k = KAN(len(DELTA_IN), [6, 1], n_salt)
        k.load_state_dict({n: t.float() for n, t in sd.items()}, strict=False)
        return k.double().eval()

    def syn_from(sd):
        n = SY.SynthKAN().double(); n.load_state_dict({k: v.double() for k, v in sd.items()}); return n.eval()

    def model(arm, cell, key):
        if cell == "R":
            i = int(key[3:]); net = kan_from(ld(f"delta_s{i}.pt"))
            parts = FZ.real_parts(net, ball)
            zin = lambda prm: torch.cat([ball["x"], torch.tanh(prm["emb.weight"][ball["s"]])], 1)
        else:
            lev = cell; s = int(key[1:]); net = syn_from(ld(f"synth_correlated_{lev}_s{s}.pt")["early"])
            parts = FZ.syn_parts(net, X, y, lev)
            zin = lambda prm: Xd
        full = {k: v for k, v in net.named_parameters()}
        free = FZ.L2 if arm == "A" else FZ.L1 if arm == "B" else list(full)
        extra = (lambda prm: 1e-4 * sum((v ** 2).sum() for v in prm.values())) if arm == "D" else None
        return net, full, free, parts, zin, extra

    rows = []
    groups = [("A", "R", [f"ind{i}" for i in range(6)]), ("A", "P0", [f"s{s}" for s in range(8)]), ("A", "P1", [f"s{s}" for s in range(8)]),
              ("B", "R", [f"ind{i}" for i in range(6)]), ("B", "P0", [f"s{s}" for s in range(8)]), ("B", "P1", [f"s{s}" for s in range(8)]),
              ("D", "P0", [f"s{s}" for s in range(8)])]
    for arm, cell, keys in groups:
        log(f"\n==================== Arm {arm}, cell {cell} ====================")
        for key in keys:
            fname = f"freeze_{arm}-R_{key}.pt" if cell == "R" else (f"freeze_{arm}-S-{cell}_{key}.pt" if arm in "AB" else f"freeze_D_{key}.pt")
            st = ld(fname); xe = st["x"].numpy() if torch.is_tensor(st["x"]) else st["x"]; info = st["info"]
            net, full, free, parts, zin, extra = model(arm, cell, key)
            flat = FZ.FlatFree(net, full, free, parts, extra)
            p0, pe = flat.full_at(flat.x0), flat.full_at(xe)
            r = dict(arm=arm, cell=cell, key=key, certified=bool(info["cert"]["certified"]), stop=info["stop"], iters=info["iters"])
            # 8: eigenvalue at the endpoint vs worst along the logged trajectory
            H = flat.hessian(xe); ev, V = np.linalg.eigh(H)
            r["endpoint_min_max"] = float(ev[0] / ev[-1])
            r["worst_logged_min_max"] = float(min(l["min_eig"] / l["max_eig"] for l in info["logged"]))
            r["flag_certified_negative"] = bool(r["certified"] and r["endpoint_min_max"] < -NEAR)
            if arm == "D":
                rows.append(r)
                log(f"  D {key}: certified {r['certified']}; endpoint min/max {r['endpoint_min_max']:+.2e}; worst along trajectory {r['worst_logged_min_max']:+.2e}"
                    + ("  <-- CERTIFIED with min/max < -1e-8" if r["flag_certified_negative"] else ""))
                continue
            # 1: saturation
            with torch.no_grad():
                s0, se = hidden(p0, net, zin(p0)).numpy(), hidden(pe, net, zin(pe)).numpy()
            r["sat_early"], r["sat_end"] = float((np.abs(s0) > 3).mean()), float((np.abs(se) > 3).mean())
            sat_units = [j for j in range(se.shape[1]) if (np.abs(se[:, j]) > 3).mean() >= 0.10]
            r["saturated_units"] = sat_units
            # 2: near-flat directions
            near = np.abs(ev) < NEAR * ev[-1]
            r["n_near_flat"] = int(near.sum()); r["eig_deciles"] = [float(q) for q in np.quantile(ev, np.linspace(0, 1, 11))]
            # 3: overlap of near-flat eigenvectors
            feed = np.zeros(flat.n, bool)
            aff_w = None
            for n, shp, a, b in flat.meta:
                idx = np.arange(a, b).reshape(shp)
                for j in sat_units:
                    if arm == "B" and n in ("layers.0.coef", "layers.0.base.weight", "layers.0.base.bias"):
                        feed[idx[j].ravel()] = True
                    if arm == "A" and n in ("layers.1.coef", "layers.1.base.weight"):
                        feed[idx[0, j].ravel()] = True
            if near.any():
                Vn = V[:, near]; w = (Vn ** 2).sum(1) / near.sum()
                r["share_feeding_saturated"] = float(w[feed].sum())
                aff = 0.0
                for n, shp, a, b in flat.meta:
                    if n == "layers.0.coef":
                        c = torch.tensor(Vn[a:b].T.reshape(-1, *shp))          # (n_near, out, in, K)
                        aff = float(((c @ NB) ** 2).sum()) / near.sum()
                r["share_L1_affine"] = aff
            else:
                r["share_feeding_saturated"] = r["share_L1_affine"] = None
            # 4: data-term improvement in the first 10 iterations (deterministic replay)
            FZ.MAXITER = 10
            rep, _ = SR.cached(f"degen_replay_{arm}-{cell}_{key}.pt", lambda: FZ.to_cache(dict(zip(("x", "info"), FZ.run(flat, FZ.SYN_BLOCKS)))))
            FZ.MAXITER = 500
            xr = rep["x"].numpy() if torch.is_tensor(rep["x"]) else rep["x"]
            d0, d10, de = flat.terms(flat.x0)["data"], flat.terms(xr)["data"], flat.terms(xe)["data"]
            r["data0"], r["data10"], r["data_end"] = d0, d10, de
            r["share_first10"] = float((d0 - d10) / (d0 - de)) if d0 - de > 0 else None
            if info["iters"] <= 10:
                r["replay_check_max_abs_dx"] = float(np.abs(xr - xe).max())
            # 5-7: probes, real Arm B only
            if arm == "B" and cell == "R":
                Lfun = lambda prm: float(sum(parts(prm).values()))
                base = Lfun(pe)
                probes = {}
                for kind in ("affine", "all"):
                    vals = [base]
                    for c in CS:
                        q = {k: v.clone() for k, v in pe.items()}
                        if kind == "affine":
                            cf = q["layers.0.coef"]; A = (cf @ NB) @ NB.T
                            q["layers.0.coef"] = cf + (c - 1) * A
                        else:
                            for n in FZ.L1:
                                q[n] = q[n] * c
                        with torch.no_grad():
                            vals.append(Lfun(q))
                    probes[kind] = vals
                r["scaling"] = probes
                r["scaling_nonincreasing"] = {k: bool(all(v[i + 1] <= v[i] for i in range(len(v) - 1))) for k, v in probes.items()}
                flat_e = FZ.FlatFree(net, pe, free, parts)
                rs = restart_probe(key, flat_e)
                xs = rs["x"].numpy() if torch.is_tensor(rs["x"]) else rs["x"]
                L0r, L1r = flat_e.fun(flat_e.x0)[0], flat_e.fun(xs)[0]
                th0 = float(sum((v ** 2).sum() for v in pe.values())); th1 = float(sum((v ** 2).sum() for v in flat_e.full_at(xs).values()))
                r["restart"] = dict(stop=rs["info"]["stop"], iters=rs["info"]["iters"], loss0=L0r, loss1=L1r, rel_loss=(L1r - L0r) / abs(L0r),
                                    theta0=th0, theta1=th1, rel_theta=(th1 - th0) / th0, certified=bool(rs["info"]["cert"]["certified"]))
                plateau = (r["restart"]["rel_loss"] < -1e-6 and r["restart"]["rel_theta"] > 1e-6) or any(r["scaling_nonincreasing"].values())
                r["class"] = "PLATEAU" if plateau else "FLAT MINIMUM"
            rows.append(r)
            line = (f"  {arm}-{cell} {key}: cert {str(r['certified']):5s} | sat |s|>3 {r['sat_early']:.3f} -> {r['sat_end']:.3f} (units {sat_units}) | near-flat {r['n_near_flat']:3d} | "
                    f"overlap sat-feed {('-' if r['share_feeding_saturated'] is None else format(r['share_feeding_saturated'], '.3f'))}, L1-affine "
                    f"{('-' if r['share_L1_affine'] is None else format(r['share_L1_affine'], '.3f'))} | data improvement in first 10 it "
                    f"{('n/a (data rose)' if r['share_first10'] is None else format(r['share_first10'], '.3f'))} | endpoint min/max {r['endpoint_min_max']:+.2e} "
                    f"(worst logged {r['worst_logged_min_max']:+.2e})")
            if "replay_check_max_abs_dx" in r:
                line += f" | replay check {r['replay_check_max_abs_dx']:.1e}"
            log(line)
            if "class" in r:
                rr = r["restart"]
                log(f"      restart (gtol 1e-9): {rr['stop']}, {rr['iters']} it, loss {rr['rel_loss']:+.2e} rel, ||theta||^2 {rr['rel_theta']:+.2e} rel, certified {rr['certified']}")
                for k in ("affine", "all"):
                    v = r["scaling"][k]
                    log(f"      scaling {k:6s} c=1..10: " + " ".join(f"{x:.4e}" for x in v) + f"  non-increasing: {r['scaling_nonincreasing'][k]}")
                log(f"      CLASS: {r['class']}")
            if r["flag_certified_negative"]:
                log("      <-- FLAG: CERTIFIED endpoint with min/max < -1e-8")
    OUT["rows"] = rows
    (HERE / "degeneracy_results.json").write_text(json.dumps(OUT, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))

    # figure: scaling probes
    br = [r for r in rows if r["arm"] == "B" and r["cell"] == "R"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, kind, title in ((axs[0], "affine", "(i) layer-1 affine RBF component x c"), (axs[1], "all", "(ii) all layer-1 weights and biases x c")):
        for r in br:
            v = np.array(r["scaling"][kind]); ax.plot([1] + CS, v / v[0], marker="o", ms=3, label=f"{r['key']} ({'cert' if r['certified'] else 'not cert'})")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("scale factor c"); ax.set_ylabel("loss / loss at the endpoint"); ax.set_title(title, fontsize=9)
    axs[0].legend(fontsize=7)
    fig.suptitle("POST-HOC: real Arm B endpoints, scaling probes", fontsize=9)
    fig.tight_layout(); FZ.FIG.mkdir(exist_ok=True); fig.savefig(FZ.FIG / "degeneracy_scaling.png", dpi=110); plt.close(fig)
    log("\nwrote degeneracy_results.json and figs/degeneracy_scaling.png")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--restart":
        restart_child(sys.argv[2])
    else:
        main()

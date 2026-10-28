"""paper2/revision/review4.py: review-4 analysis bundle (POST-HOC, DESCRIPTIVE; plan review4_plan.md). No training, no optimisation.
1. attribution of the fold A damage to the blind-direction change; 2. physics consistency of the learned derivatives;
3. residual edge disagreement at certified minima. Archived weights SHA-verified and held in memory.
Output: review4_output.txt, review4_results.json. Run from the repo root: /usr/bin/python3 paper2/revision/review4.py"""
import io
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "arch", REPO / "paper2"):
    sys.path.insert(0, str(p))

import numpy as np
import torch

import anchor as AN
import gauge as GA
import sc_run as SR
from certify import certify_point
from close import FlatQ
from kan import DELTA_IN, KAN, N_RBF, delta_head, predict, subset
from stage2 import extract_verified, in_memory
from trust import Flat

LOG, RES = [], {}
LOGGED = {"early": 0.1420, "mu = 0": 0.8147, "mu = 1e-6": 1.0262, "mu = 1e-4": 0.3345}
Qb, _ = np.linalg.qr(np.column_stack([np.ones(N_RBF), np.arange(N_RBF, dtype=float)]))
P = Qb @ Qb.T                                                           # close.py null-space projector (8 x 8)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.append(s)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def net_from_sd(sd, d, ns):
    n = KAN(d, [6, 1], ns).double()
    miss, unexp = n.load_state_dict({k: v.double() for k, v in sd.items()}, strict=False)
    assert not unexp and all(x.endswith(".grid") for x in miss), (miss, unexp)
    return n.eval()


def params(net):
    return {k: v.detach().double().clone() for k, v in net.state_dict().items() if not k.endswith(".grid")}


# ------------------------------------------------------------------ fold A ensembles (items 1, 2)
def foldA_nets(S, blobs):
    fa = S["folds"]["A"]
    bt = subset(S["B"], fa["train"]); btd = SR.dbl(bt)
    used = np.unique(S["B"]["s"].numpy()[fa["train"]]).tolist()
    d, ns = len(DELTA_IN), len(S["salts"])
    nets = {k: [] for k in LOGGED}
    for s in range(5):
        sd = torch.load(io.BytesIO(blobs[f"scbase_A_s{s}.pt"]), weights_only=False)
        sd = sd["early"] if isinstance(sd, dict) and "early" in sd else sd
        early = lambda: net_from_sd(sd, d, ns).float()
        nets["early"].append(net_from_sd(sd, d, ns))
        x0 = torch.load(io.BytesIO(blobs[f"scbase_A_s{s}_pol.pt"]), weights_only=False)["x"].numpy()
        nets["mu = 0"].append(net_from_sd(Flat(early(), btd, used).to_kan(x0, d, ns).state_dict(), d, ns))
        for mu, nm, st in (("1e-06", f"close_foldA_1e-06_s{s}.pt", "mu = 1e-6"), ("0.0001", f"certify_foldA_0.0001_s{s}.pt", "mu = 1e-4")):
            xv = torch.load(io.BytesIO(blobs[nm]), weights_only=False)
            xv = xv.numpy() if torch.is_tensor(xv) else np.asarray(xv)
            nets[st].append(net_from_sd(FlatQ(early(), btd, used, float(mu)).to_kan(xv, d, ns).state_dict(), d, ns))
    return nets


def pred(net, b):
    return SR.base_pred(net, SR.dbl(b))


def item1(S, nets):
    log("\n==================== 1. attribution of the fold A damage (POST-HOC) ====================")
    tr, fa = S["tr"], S["folds"]["A"]
    rows = {"foldA": fa["test"], "train": fa["train"]}
    bs = {k: subset(S["B"], m) for k, m in rows.items()}
    y = {k: tr.log_k.values[m] for k, m in rows.items()}; anc = {k: tr.anchor.values[m] for k, m in rows.items()}
    d, ns = len(DELTA_IN), len(S["salts"])
    out = {}
    for st in ("mu = 0", "mu = 1e-6"):
        P_seed = {lab: {k: [] for k in rows} for lab in ("early", "early + blind", "early + remainder", "polished")}
        maxrec = 0.0
        for e_net, p_net in zip(nets["early"], nets[st]):
            pe, pp = params(e_net), params(p_net)
            blind, rem = {}, {}
            for k in pe:
                dlt = pp[k] - pe[k]
                if k.endswith(".coef"):
                    bl = dlt @ torch.tensor(P)                          # edge-wise projection over the 8 centre indices
                    blind[k], rem[k] = bl, dlt - bl
                elif k.endswith("base.weight"):
                    blind[k], rem[k] = dlt, torch.zeros_like(dlt)
                else:                                                   # base.bias, emb.weight
                    blind[k], rem[k] = torch.zeros_like(dlt), dlt
            recon = {k: pe[k] + blind[k] + rem[k] for k in pe}
            maxrec = max(maxrec, max(float((recon[k] - pp[k]).abs().max()) for k in pe))
            states = {"early": pe, "early + blind": {k: pe[k] + blind[k] for k in pe},
                      "early + remainder": {k: pe[k] + rem[k] for k in pe}, "polished": pp}
            for lab, sd in states.items():
                n = net_from_sd(sd, d, ns)
                for k in rows:
                    P_seed[lab][k].append(pred(n, bs[k]))
        assert maxrec < 1e-10, maxrec
        R = {lab: {k: dict(ensemble=rmse(anc[k] + np.mean(v[k], 0), y[k]), per_seed=[rmse(anc[k] + p, y[k]) for p in v[k]]) for k in rows}
             for lab, v in P_seed.items()}
        def frac(k, key="ensemble", i=None):
            g = (lambda lab: R[lab][k][key]) if i is None else (lambda lab: R[lab][k]["per_seed"][i])
            tot = g("polished") - g("early")
            fb, fr = (g("early + blind") - g("early")) / tot, (g("early + remainder") - g("early")) / tot
            return dict(total=tot, f_blind=fb, f_rem=fr, interaction=1 - fb - fr)
        fr_ = {k: dict(ensemble=frac(k), per_seed=[frac(k, i=i) for i in range(5)]) for k in rows}
        ens = R
        dtrain_blind = ens["early + blind"]["train"]["ensemble"] - ens["early"]["train"]["ensemble"]
        fb_A = fr_["foldA"]["ensemble"]["f_blind"]
        free = abs(dtrain_blind) < 0.01 and fb_A > 0.5
        nonadd = {k: abs(fr_[k]["ensemble"]["interaction"]) > 0.25 for k in rows}
        out[st] = dict(rmse=R, fractions=fr_, reconstruction_max_abs=maxrec, train_change_blind=dtrain_blind,
                       nearly_free_on_train_and_costly_on_foldA=free, not_additive=nonadd)
        log(f"\n  {st} (early + blind + remainder = polished to {maxrec:.1e})")
        log(f"  {'state':20s} {'fold A RMSE (ens; seeds)':48s} {'training RMSE (ens; seeds)':48s}")
        for lab in P_seed:
            f = lambda k: f"{R[lab][k]['ensemble']:.4f} ({', '.join(f'{x:.3f}' for x in R[lab][k]['per_seed'])})"
            log(f"  {lab:20s} {f('foldA'):48s} {f('train'):48s}")
        for k in rows:
            e = fr_[k]["ensemble"]
            ps = fr_[k]["per_seed"]
            log(f"  {k:6s} ensemble: total damage {e['total']:+.4f}; blind alone {e['f_blind']:.3f}, remainder alone {e['f_rem']:.3f}, "
                f"interaction {e['interaction']:+.3f}" + ("  -> the effect is not additive" if nonadd[k] else "")
                + " | per seed f_blind " + ", ".join("%.2f" % p["f_blind"] for p in ps))
        log(f"  reading: early + blind changes the training RMSE by {dtrain_blind:+.4f} and carries f_blind = {fb_A:.3f} of the fold A damage -> "
            + ("the blind-direction change is nearly free on training rows and costly on fold A (plan wording)" if free
               else "the plan's condition for 'nearly free on training rows and costly on fold A' is not met; reported as found"))
    RES["item1"] = out


def anchor_slopes(tr, mask):
    series = {k: AN.build_series(g) for k, g in tr[tr.combo == "PC"].groupby(["source_doi", "salt_name"])}
    sl = np.full(mask.sum(), np.nan)
    for n, r in enumerate(tr[mask].itertuples()):
        ser = series[(r.source_doi, r.salt_name)]
        lv = np.array(sorted(ser)); j = int(np.argmin(np.abs(lv - r.temperature_K)))
        a0 = AN.interp_c(ser[lv[j]], r.conc_native)
        sec = []
        for jj in (j - 1, j + 1):
            if 0 <= jj < len(lv):
                a1 = AN.interp_c(ser[lv[jj]], r.conc_native)
                if not np.isnan(a1) and not np.isnan(a0):
                    sec.append((a1 - a0) / (1000 / lv[jj] - 1000 / lv[j]))
        if sec:
            sl[n] = np.mean(sec)
    return sl


def item2(S, nets):
    log("\n==================== 2. physics consistency (POST-HOC; a consistency check, not a test of a transport law) ====================")
    tr, fa, sc = S["tr"], S["folds"]["A"], S["sc"]
    ci = {c: DELTA_IN.index(c) for c in ("invT", "mix_lneta")}
    fac = {c: 2.0 / (sc.hi[i] - sc.lo[i]) for c, i in ci.items()}
    log(f"  scaler factors (positive, so signs are unit-free): d/d(1000/T) x {fac['invT']:.4g}, d/d(ln eta_mix) x {fac['mix_lneta']:.4g}")
    out = {}
    rng = np.random.default_rng(0)
    for rs, m in (("train", fa["train"]), ("foldA", fa["test"])):
        b = SR.dbl(subset(S["B"], m))
        xco = b["x_co"].numpy(); T = tr.temperature_K.values[m]
        aslope = anchor_slopes(tr, m)
        n_noslope = int(np.isnan(aslope).sum())
        terc = {"x_co": np.quantile(xco, [1 / 3, 2 / 3]), "T": np.quantile(T, [1 / 3, 2 / 3])}
        bins = {k: np.digitize(v, terc[k], right=True) for k, v in (("x_co", xco), ("T", T))}
        out[rs] = {"n": int(m.sum()), "anchor_slope_missing": n_noslope, "terciles": {k: v.tolist() for k, v in terc.items()},
                   "tercile_counts": {k: np.bincount(v, minlength=3).tolist() for k, v in bins.items()}}
        for st, ens in nets.items():
            ders = {c: [] for c in ci}; fd_err = 0.0
            for net in ens:
                x = b["x"].clone().requires_grad_(True)
                dl = b["x_co"] * net(x, b["s"])[:, 0]
                g = torch.autograd.grad(dl.sum(), x)[0].numpy()
                for c, i in ci.items():
                    ders[c].append(g[:, i] * fac[c])
                    idx = rng.choice(len(xco), 50, replace=False)
                    with torch.no_grad():
                        xp, xm = b["x"].clone(), b["x"].clone(); xp[idx, i] += 1e-4; xm[idx, i] -= 1e-4
                        fd = ((b["x_co"] * net(xp, b["s"])[:, 0] - b["x_co"] * net(xm, b["s"])[:, 0]) / 2e-4).numpy()[idx]
                    fd_err = max(fd_err, float(np.max(np.abs(fd - g[idx, i]) / np.maximum(np.abs(g[idx, i]), 1e-8))))
            model_T = np.mean(ders["invT"], 0)
            res = {}
            for name, per_seed in (("dlogk_dinvT_total", [aslope + d_ for d_ in ders["invT"]]), ("dlogk_dinvT_model_part", ders["invT"]),
                                   ("dlogk_dlneta_mix", ders["mix_lneta"])):
                ensd = np.mean(per_seed, 0)
                ok = ~np.isnan(ensd)
                viol = (ensd > 1e-9) & ok
                r = dict(share_violating=float(viol.sum() / ok.sum()), n=int(ok.sum()), median=float(np.median(ensd[ok])),
                         per_seed_share=[float(((p > 1e-9) & ~np.isnan(p)).sum() / (~np.isnan(p)).sum()) for p in per_seed],
                         by_tercile={k: [float(viol[(bins[k] == t) & ok].sum() / max(((bins[k] == t) & ok).sum(), 1)) for t in range(3)] for k in bins})
                res[name] = r
            res["fd_check_max_rel_err"] = fd_err
            out[rs][st] = res
            log(f"  {rs:6s} {st:10s}: share violating (ensemble; seed range) | median:")
            for name in ("dlogk_dinvT_total", "dlogk_dinvT_model_part", "dlogk_dlneta_mix"):
                r = res[name]
                log(f"     {name:24s} {r['share_violating']:.3f} ({min(r['per_seed_share']):.3f}-{max(r['per_seed_share']):.3f}) | {r['median']:+.4f}"
                    f" | x_co terciles {', '.join(f'{v:.2f}' for v in r['by_tercile']['x_co'])} | T terciles {', '.join(f'{v:.2f}' for v in r['by_tercile']['T'])}"
                    + ("   (no expected sign for the correction alone)" if name.endswith("model_part") else ""))
            log(f"     autograd vs central difference (50 rows per input and seed): max relative disagreement {fd_err:.1e}")
        log(f"  {rs}: {out[rs]['n']} rows; anchor slope unavailable for {n_noslope}; tercile row counts x_co {out[rs]['tercile_counts']['x_co']}, "
            f"T {out[rs]['tercile_counts']['T']}")
    RES["item2"] = out


# ------------------------------------------------------------------ item 3
def curved_edges(net, Z, sidx, d):
    """(d+1, n, W): curved part RBF(z) (I - P) c of every layer-1 edge (blind component removed); salt edge = sum of the two embedding edges"""
    with torch.no_grad():
        L0 = net.layers[0]
        C = L0.coef.detach().double().numpy() @ (np.eye(N_RBF) - P)    # (W, d_in, 8)
        Zt = torch.tensor(Z, dtype=torch.float64)
        e = torch.tanh(net.emb(torch.tensor(sidx, dtype=torch.long))).double()
        cols = [Zt[:, i] for i in range(d)] + [e[:, 0], e[:, 1]]
        rb = [L0.rbf(cc).double().numpy() for cc in cols]              # (n, 8)
        out = [rb[i] @ C[:, i, :].T for i in range(d)]
        out.append(rb[d] @ C[:, d, :].T + rb[d + 1] @ C[:, d + 1, :].T)
    return np.stack(out)


def gfix(g, phi):
    return (g.sign * (phi - phi.mean(1, keepdims=True)) / g.sd)[..., g.perm]


def item3(blobs):
    log("\n==================== 3. residual edge disagreement (POST-HOC) ====================")
    tr, te, te_anc, D, _ = GA.setup()
    d, m = D["d_num"], D["train"]
    Ztr, str_ = D["Z"][m], D["s"][m]
    b = subset(D["batch"], m)
    bd = {k: (v.double() if v.is_floating_point() else v) for k, v in b.items()}
    used = np.unique(str_).tolist()
    sd_y = bd["y"].std().item()
    def kan_from(blob):
        k = KAN(d, [6, 1], D["n_salt"])
        miss, unexp = k.load_state_dict({n: t.float() for n, t in torch.load(io.BytesIO(blob)).items()}, strict=False)
        assert not unexp and all(x.endswith(".grid") for x in miss)
        return k.eval()
    sets = {"early-stopped": [kan_from(blobs[f"delta_s{i}.pt"]) for i in range(6)], "certified mu = 1e-4": []}
    ncert = 0
    for i in range(6):
        flat = FlatQ(kan_from(blobs[f"delta_s{i}.pt"]), bd, used, 1e-4)
        x = torch.load(io.BytesIO(blobs[f"close_0.0001_ind_{i}.pt"])).numpy()
        ncert += bool(certify_point(flat, x)[0]["certified"])
        sets["certified mu = 1e-4"].append(flat.to_kan(x, d, D["n_salt"]).float().eval())
    preds = {k: [predict(n, delta_head, b) for n in v] for k, v in sets.items()}
    dab = {k: [float(np.sqrt(np.mean((p[a] - p[c]) ** 2)) / sd_y) for a, c in itertools.combinations(range(6), 2)] for k, p in preds.items()}
    log(f"  integrity: certify_point re-certifies {ncert}/6 of the mu = 1e-4 points; their d_AB median {np.median(dab['certified mu = 1e-4']):.4f} "
        f"(committed 0.0771); early-stopped d_AB median {np.median(dab['early-stopped']):.4f} (computed here; no all-rows value is committed)")
    if ncert != 6 or abs(np.median(dab["certified mu = 1e-4"]) - 0.0771) > 1e-4:
        raise SystemExit("STOP: the certified points do not reproduce the committed certification / d_AB")
    names = D["cols"] + ["salt (embedding)"]
    out = {}
    for k, ns_ in sets.items():
        gs = [GA.Gauge(n, Ztr, str_, d) for n in ns_]
        full = [GA.parts(n, Ztr, str_, d)[0] for n in ns_]
        curv = [curved_edges(n, Ztr, str_, d) for n in ns_]
        rec = {"full": [], "curved": []}; excl = {"full": 0, "curved": 0}
        for a, c in itertools.combinations(range(6), 2):
            gA, gB = gs[a], gs[c]
            GA.align(gA, gB)                                             # sets gB.perm against gA (A is the reference)
            for ver, src in (("full", full), ("curved", curv)):
                fa_, fb_ = gfix(gA, src[a]), gfix(gB, src[c])
                for i in range(d + 1):
                    for j in range(fa_.shape[2]):
                        u, v = fa_[i, :, j], fb_[i, :, j]
                        if u.std() < 1e-8 or v.std() < 1e-8:
                            excl[ver] += 1; continue
                        rec[ver].append(dict(pair=[a, c], input=names[i], unit=j, corr=float(np.corrcoef(u, v)[0, 1])))
            gB.perm = np.arange(len(gB.perm))
        summ = {}
        for ver in rec:
            cont = [r["corr"] for r in rec[ver] if r["input"] != "salt (embedding)"]
            salt = [r["corr"] for r in rec[ver] if r["input"] == "salt (embedding)"]
            per_in = {nm: float(np.median([r["corr"] for r in rec[ver] if r["input"] == nm])) for nm in names if any(r["input"] == nm for r in rec[ver])}
            summ[ver] = dict(continuous=dict(median=float(np.median(cont)), min=float(min(cont)), max=float(max(cont)), n=len(cont)),
                             salt=dict(median=float(np.median(salt)), min=float(min(salt)), max=float(max(salt)), n=len(salt)) if salt else None,
                             per_input_median=per_in, excluded=excl[ver])
        out[k] = dict(summary=summ, dAB=dict(median=float(np.median(dab[k])), min=float(min(dab[k])), max=float(max(dab[k])),
                                             source="certify_output.txt (committed 0.0771)" if k.startswith("certified") else "computed here (committed definition)"),
                      records=rec)
        for ver in ("full", "curved"):
            s_ = summ[ver]
            log(f"  {k:20s} {('edges' if ver == 'full' else 'blind part removed'):18s}: continuous inputs median {s_['continuous']['median']:+.3f} "
                f"(range {s_['continuous']['min']:+.3f} to {s_['continuous']['max']:+.3f}, n {s_['continuous']['n']}, excluded {s_['excluded']}); "
                + (f"salt edge median {s_['salt']['median']:+.3f}" if s_["salt"] else "salt edge: none"))
            log("      per input: " + ", ".join(f"{nm} {v:+.2f}" for nm, v in s_["per_input_median"].items()))
        log(f"  {k:20s} prediction-level d_AB median {out[k]['dAB']['median']:.4f} (range {out[k]['dAB']['min']:.4f}-{out[k]['dAB']['max']:.4f}; {out[k]['dAB']['source']})")
    RES["item3"] = out


def main():
    log("#" * 100)
    log("paper2/revision/review4.py: review-4 analysis bundle (POST-HOC, DESCRIPTIVE; plan review4_plan.md); no training")
    log("#" * 100)
    log("DISCREPANCIES with the plan (files win):")
    log("  D1 item 3: the committed decomposition (gauge.layer0_edges) sums the two salt-embedding edges into one salt edge; the plan's")
    log("     'two salt-embedding coordinates reported separately' is therefore the one combined salt edge, reported apart from the 8 inputs")
    log("  D2 item 1: on training rows the total change (polished minus early) is negative (polishing lowers training RMSE), so the")
    log("     training-row fractions divide by a small negative number and are not interpretable as shares; they are printed as computed")
    log("  D3 item 2: the ensemble's violating share can lie outside the per-seed range (the ensemble derivative is the seed mean)")
    names = [f"scbase_A_s{s}.pt" for s in range(5)] + [f"scbase_A_s{s}_pol.pt" for s in range(5)] + \
            [f"close_foldA_1e-06_s{s}.pt" for s in range(5)] + [f"certify_foldA_0.0001_s{s}.pt" for s in range(5)] + \
            [f"delta_s{i}.pt" for i in range(6)] + [f"close_0.0001_ind_{i}.pt" for i in range(6)]
    blobs = in_memory(extract_verified(names))
    S = SR.setup()
    nets = foldA_nets(S, blobs)
    tr, fa = S["tr"], S["folds"]["A"]
    bh = subset(S["B"], fa["test"]); y = tr.log_k.values[fa["test"]]; anc = tr.anchor.values[fa["test"]]
    rm = {k: rmse(anc + np.mean([pred(n, bh) for n in v], 0), y) for k, v in nets.items()}
    log("  integrity: fold A ensemble RMSE " + ", ".join(f"{k} {rm[k]:.4f} (logged {LOGGED[k]:.4f})" for k in rm))
    if not all(abs(rm[k] - LOGGED[k]) < 1e-4 for k in rm):
        raise SystemExit("STOP: fold A ensembles do not reproduce the logged RMSEs")
    RES["integrity_foldA_rmse"] = rm
    item1(S, nets)
    item2(S, nets)
    item3(blobs)
    (HERE / "review4_results.json").write_text(json.dumps(RES, indent=1, default=float))
    (HERE / "review4_output.txt").write_text("\n".join(LOG) + "\n")
    log("wrote review4_output.txt, review4_results.json")


if __name__ == "__main__":
    main()

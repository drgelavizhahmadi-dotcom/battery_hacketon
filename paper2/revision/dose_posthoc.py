"""paper2/revision/dose_posthoc.py: POST-HOC, DESCRIPTIVE check on the dose-response study (no training, no verdict changes).
At the doses where the unregularised polish certified, is the short regularised path approaching that certified finite minimum,
or does it move elsewhere? Weights only, from archive 16 (gauge_weights_2026-10-21p.tar.gz), SHA-verified against the manifest
and held in memory. Run from the repo root with the pinned interpreter:
  /usr/bin/python3 paper2/revision/dose_posthoc.py >> paper2/revision/dose_output.txt"""
import io
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "paper2", REPO / "arch", HERE):
    sys.path.insert(0, str(p))

import numpy as np
import torch

import dose as D
import infimum as I
import synth as SY
from stage2 import extract_verified, in_memory

ARCH16 = "gauge_weights_2026-10-21p.tar.gz"
TOL_N, TOL_D, TOL_L = 0.10, 0.10, 1e-3       # fixed before computing (author's brief, 2026-10-21)


def log(*a):
    print(*a, flush=True)


def classify(nr, dist, rl):
    if abs(nr - 1) <= TOL_N and dist < TOL_D and abs(rl) <= TOL_L:
        return "APPROACHING"
    if abs(rl) <= TOL_L:
        return "ELSEWHERE / matching loss (FINITE)"
    if rl > TOL_L:
        return "ELSEWHERE / higher loss (FINITE, NOT YET REACHED)"
    if rl < -TOL_L and nr > 1 + TOL_N:
        return "ELSEWHERE / lower loss, outward (RECEDING)"
    return "UNCLASSIFIED"


def main():
    log("\n" + "#" * 100)
    log(f"paper2/revision/dose_posthoc.py {time.strftime('%Y-%m-%d %H:%M:%S')}: POST-HOC, DESCRIPTIVE check (no training; changes no verdict)")
    log("#" * 100)
    log("  Q: where the unregularised polish certified, does the short path (gamma 1e-5, 1e-7) approach that certified minimum?")
    log(f"  rules fixed before computing: APPROACHING = |norm ratio - 1| <= {TOL_N}, relative distance < {TOL_D}, |relative loss diff| <= {TOL_L}")
    log("  at gamma = 1e-7; otherwise ELSEWHERE: matching loss (|rel| <= 1e-3) -> FINITE; higher (> +1e-3) -> FINITE, NOT YET REACHED;")
    log("  lower (< -1e-3) with norm ratio > 1.1 -> RECEDING; anything else UNCLASSIFIED. Trend flag: relative distance shrank 1e-5 -> 1e-7.")
    log("  'certified' = the polish was included (certified, incl. certified-degenerate FLAT MINIMUM); L = the committed cell loss")
    log("  (no gamma term) at both points. 'Mostly' = more than half of the certified models pooled over rho and P0/P1 (fixed now).")
    man = {l.split()[0].split("/")[-1] for l in (REPO / "weights_manifest.txt").read_text().splitlines()
           if l.startswith("gauge_weights/dose_") and l.split()[-1] == ARCH16}
    keys = [D.key_of(l, r, s) for r in D.RHOS for l in D.LEVS for s in D.SEEDS]
    names = [n for k in keys for n in (f"dose_early_{k}.pt", f"dose_pol_{k}.pt", f"dose_path_{k}_g1.pt", f"dose_path_{k}_g2.pt", f"dose_rec_{k}.json")]
    missing = [n for n in names if n not in man]
    if missing:
        raise SystemExit(f"not in archive 16's manifest: {missing[:5]}")
    log(f"\n  inputs: {len(names)} files from {ARCH16} (SHA-256 verified, held in memory)")
    import contextlib, os
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        blobs = in_memory(extract_verified(names))
    log(f"  {len(blobs)} inputs held in memory (SHA-256 re-checked from memory)")
    ld = lambda n: torch.load(io.BytesIO(blobs[n]), weights_only=False)
    rows = []
    for k in keys:
        lev, rho, s = D.parse(k)
        rec = json.loads(blobs[f"dose_rec_{k}.json"])
        net = SY.SynthKAN(); net.load_state_dict(ld(f"dose_early_{k}.pt")["early"]); net = net.double().eval()
        d = D.data(rho)
        full = dict(net.named_parameters())
        f = I.Flat(net, full, list(full), I.syn_parts(net, d["X"], d["y"], lev, D.mu_of(lev)), None, {})
        xp = I.npx(ld(f"dose_pol_{k}.pt")["x"])
        x5, x7 = I.npx(ld(f"dose_path_{k}_g1.pt")["x"]), I.npx(ld(f"dose_path_{k}_g2.pt")["x"])
        Lp = f.Lstar(xp); tp = float((xp ** 2).sum()); npn = float(np.linalg.norm(xp))
        r = dict(key=k, lev=lev, rho=rho, seed=s, certified=not rec["uncertified"], L_pol=Lp, theta2_pol=tp)
        for tag, x in (("g5", x5), ("g7", x7)):
            r[tag] = dict(norm_ratio=float((x ** 2).sum()) / tp, dist=float(np.linalg.norm(x - xp)) / npn, rel_loss=(f.Lstar(x) - Lp) / abs(Lp))
        if r["certified"]:
            g = r["g7"]
            r["class"] = classify(g["norm_ratio"], g["dist"], g["rel_loss"])
            r["trend_shrinks"] = bool(r["g7"]["dist"] < r["g5"]["dist"])
        rows.append(r)
    log("\n  per model (certified polishes):")
    log("    key            norm ratio 1e-5 / 1e-7     rel. distance 1e-5 / 1e-7    rel. loss diff 1e-5 / 1e-7     shrinks  class")
    for r in rows:
        if r["certified"]:
            a, b = r["g5"], r["g7"]
            log(f"    {r['key']:13s}  {a['norm_ratio']:9.3f} / {b['norm_ratio']:9.3f}    {a['dist']:8.3f} / {b['dist']:8.3f}         "
                f"{a['rel_loss']:+.2e} / {b['rel_loss']:+.2e}     {str(r['trend_shrinks'])[:5]:5s}    {r['class']}")
    cls_names = ["APPROACHING", "ELSEWHERE / matching loss (FINITE)", "ELSEWHERE / higher loss (FINITE, NOT YET REACHED)",
                 "ELSEWHERE / lower loss, outward (RECEDING)", "UNCLASSIFIED"]
    log("\n  per rho and penalty (certified polishes): n | APPROACHING | FINITE (valley) | FINITE not yet reached | RECEDING | unclassified "
        "| trend shrinks | medians at 1e-5 -> 1e-7 of norm ratio, rel. distance, rel. loss diff")
    cells = {}
    for lev in D.LEVS:
        for rho in D.RHOS:
            rr = [r for r in rows if r["lev"] == lev and r["rho"] == rho and r["certified"]]
            uu = [r for r in rows if r["lev"] == lev and r["rho"] == rho and not r["certified"]]
            c = {n: sum(1 for r in rr if r["class"] == n) for n in cls_names}
            med = lambda tag, q: float(np.median([r[tag][q] for r in rr])) if rr else None
            cells[f"{lev}_r{D.rtag(rho)}"] = dict(n=len(rr), counts=c, trend=sum(r["trend_shrinks"] for r in rr),
                                                  med={t: {q: med(t, q) for q in ("norm_ratio", "dist", "rel_loss")} for t in ("g5", "g7")},
                                                  uncert_norm_ratio_g7=[r["g7"]["norm_ratio"] for r in uu])
            v = cells[f"{lev}_r{D.rtag(rho)}"]
            fm = lambda t, q, f_: "-" if v["med"][t][q] is None else format(v["med"][t][q], f_)
            log(f"    {lev} rho {rho:.2f}: n {len(rr)} | {c[cls_names[0]]} | {c[cls_names[1]]} | {c[cls_names[2]]} | {c[cls_names[3]]} | {c[cls_names[4]]} | "
                f"{v['trend']}/{len(rr)} | norm {fm('g5', 'norm_ratio', '.3f')} -> {fm('g7', 'norm_ratio', '.3f')}, dist {fm('g5', 'dist', '.3f')} -> "
                f"{fm('g7', 'dist', '.3f')}, loss {fm('g5', 'rel_loss', '+.2e')} -> {fm('g7', 'rel_loss', '+.2e')}")
    log("\n  uncertified polishes, descriptive: ||theta*(1e-7)||^2 / polished endpoint ||theta||^2, median (range, n) per cell")
    for c, v in cells.items():
        u = v["uncert_norm_ratio_g7"]
        if u:
            log(f"    {c}: {np.median(u):.3f} ({min(u):.3f}-{max(u):.3f}, n {len(u)})")
    cert = [r for r in rows if r["certified"]]
    tot = {n: sum(1 for r in cert if r["class"] == n) for n in cls_names}
    fin = tot[cls_names[0]] + tot[cls_names[1]] + tot[cls_names[2]]
    rec_ = tot[cls_names[3]]
    log(f"\n  pooled over rho and P0/P1: certified {len(cert)}; APPROACHING {tot[cls_names[0]]}, FINITE (valley) {tot[cls_names[1]]}, "
        f"FINITE not yet reached {tot[cls_names[2]]}, RECEDING {rec_}, unclassified {tot[cls_names[4]]}; trend shrinks {sum(r['trend_shrinks'] for r in cert)}")
    if fin > len(cert) / 2:
        reading = "A"
    elif rec_ > len(cert) / 2:
        reading = "B"
    else:
        reading = "NEITHER"
    log(f"  READING (rule fixed in the brief): {'mostly APPROACHING or FINITE -> sentence A' if reading == 'A' else 'mostly RECEDING -> sentence B' if reading == 'B' else 'neither majority -> no sentence; the two-decade D does not settle the question at low rho'}")
    out = dict(rows=rows, cells=cells, pooled=dict(certified=len(cert), counts=tot, finite_total=fin), reading=reading)
    (HERE / "dose_posthoc_results.json").write_text(json.dumps(out, indent=1, default=float))
    log("  wrote dose_posthoc_results.json")


if __name__ == "__main__":
    main()

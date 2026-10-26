"""paper2/tex/build_tex.py: generate the LaTeX body of paper 2 from COMMITTED sources (no new analysis).

  body.tex       <- paper2/draft/paper2_draft.md (frozen source text), converted faithfully; figures 1-5 inserted
  appendixA.tex  <- committed verdict tables (paper2/synth_prereg.md, electrolyte_case_study_summary.md, arch/sc_prereg.md, arch/ex_prereg.md)
  appendixB.tex  <- post-hoc analyses: files + commit subjects from git
  appendixC.tex  <- reproducibility map from paper2/figures/MANIFEST.md (+ the draft's tables)
  appendixE.tex  <- supplementary figures S1-S3
Captions are drafted from the figure sidecar JSONs and marked "% CAPTION-DRAFT: review".
Run from the repo root: /usr/bin/python3 paper2/tex/build_tex.py (or `make` in paper2/tex)"""
import json
import re
import subprocess
from pathlib import Path

TEX = Path(__file__).resolve().parent
REPO = TEX.parent.parent
FIGS = REPO / "paper2" / "figures"


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def first_commit(path):
    """commit that added `path` (follows renames, e.g. the figure renumbering)"""
    return git("log", "--follow", "--diff-filter=A", "--format=%h", "--", path).splitlines()[-1]


def commit_with(path, text):
    """earliest commit whose version of `path` contains `text`"""
    out = git("log", "--format=%h", "-S", text, "--", path).splitlines()
    return out[-1] if out else "?"


# ------------------------------------------------------------------ inline conversion (markdown + unicode -> LaTeX)
FORMULAS = [  # exact draft substrings -> LaTeX math (applied before escaping)
    ("K\\_silu·α²", r"$K_{\mathrm{silu}}\,\alpha^{2}$"),
    ("K\\_silu", r"$K_{\mathrm{silu}}$"),
    ("s = φ₁(x₁) + φ₂(x₂)", r"$s = \varphi_1(x_1) + \varphi_2(x_2)$"),
    ("t·a·(x₁ − x₂)", r"$t\,a\,(x_1 - x_2)$"),
    ("t·a·x₁", r"$t\,a\,x_1$"), ("t·a·x₂", r"$t\,a\,x_2$"),
    ("t·ℓ", r"$t\,\ell$"), ("ℓ\\_k", r"$\ell_k$"),
    ("x₁ − x₂", r"$x_1 - x_2$"), ("x₁/x₂", r"$x_1/x_2$"), ("x₃/x₄", r"$x_3/x_4$"),
    ("P\\_null", r"$P_{\mathrm{null}}$"),
    ("log k = log k\\_PC + x\\_co·h(z)", r"$\log k = \log k_{\mathrm{PC}} + x_{\mathrm{co}}\, h(z)$"),
    ("N(0, 1)/3", r"$\mathcal{N}(0, 1)/3$"),
    ("x(1 − x)", r"$x(1 - x)$"),
    ("μQ", r"$\mu Q$"),
    ("\\[4, 3, 1\\]", "[4, 3, 1]"), ("\\[10, 6, 1\\]", "[10, 6, 1]"),
    # revision draft (rev 55) and revision appendix text
    ("θ\\*", r"$\theta^{*}$"), ("θ*", r"$\theta^{*}$"), ("L\\*", r"$L^{*}$"), ("L*", r"$L^{*}$"), ("ρ\\*", r"$\rho^{*}$"), ("ρ*", r"$\rho^{*}$"),
    ("D = ((λ + 2γ₁)/(λ + 2γ₂))²", r"$D = \bigl((\lambda + 2\gamma_1)/(\lambda + 2\gamma_2)\bigr)^{2}$"),
    ("x\\_co·h(z)", r"$x_{\mathrm{co}}\, h(z)$"), ("x\\_co·h", r"$x_{\mathrm{co}}\, h$"), ("x\\_co", r"$x_{\mathrm{co}}$"),
    ("L + γR", r"$L + \gamma R$"), ("γ → 0", r"$\gamma \to 0$"),
]
CITES = [  # draft author-year text -> natbib
    ("(Liu et al., 2025)", r"\citep{liu2025kan}"),
    ("(Li, 2024)", r"\citep{li2024fastkan}"),
    ("(Eilers and Marx, 1996)", r"\citep{eilers1996psplines}"),
    ("Marra and Wood (2011)", r"\citet{marra2011double}"),
    ("Bagrow (2026)", r"\citet{bagrow2026curvature}"),
    ("(Cang et al., 2024; see also the free-knots KAN of 2025)", r"(\citealp{cang2024kanwork}; see also the free-knots KAN of \citealp{zheng2025freeknots})"),
    ("Siems et al. (2023)", r"\citet{siems2023concurvity}"),
    ("(Soudry et al., 2018)", r"\citep{soudry2018implicit}"),
    ("(Sussmann, 1992; Kůrková and Neruda, 1994)", r"\citep{sussmann1992uniqueness,kurkova1994uniqueness}"),
    ("(de Blasio et al., 2024)", r"\citep{deblasio2024calisol}"),
    ("(Agarwal et al., 2021)", r"\citep{agarwal2021nam}"),
    ("Wood (2017)", r"\citet{wood2017gam}"),
    ("(Ramakrishnan et al., 2015)", r"\citep{ramakrishnan2015delta}"),
    ("(Yao et al., 2007; Ali et al., 2019)", r"\citep{yao2007early,ali2019continuous}"),
]
SUP = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")
SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
GREEK = {"γ": r"\gamma", "φ": r"\varphi", "θ": r"\theta", "μ": r"\mu", "λ": r"\lambda", "α": r"\alpha", "ρ": r"\rho", "β": r"\beta", "η": r"\eta",
         "ε": r"\varepsilon", "δ": r"\delta", "Δ": r"\Delta", "ψ": r"\psi", "σ": r"\sigma", "ℓ": r"\ell"}
UNI = {"≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$", "−": r"$-$", "→": r"$\to$", "≈": r"$\approx$", "·": r"$\cdot$",
       "–": "--", "—": "---", "’": "'", "‘": "`", "“": "``", "”": "''", "‖": r"$\|$", "²": r"$^{2}$", "³": r"$^{3}$",
       "ů": r"\r{u}", "á": r"\'a", "é": r"\'e", "č": r"\v{c}", "ć": r"\'c", "±": r"$\pm$", "∈": r"$\in$", "Σ": r"$\Sigma$", "∑": r"$\sum$", "̄": "", "π": r"$\pi$", "τ": r"$\tau$", "ω": r"$\omega$", "∂": r"$\partial$", "∇": r"$\nabla$", "∀": r"$\forall$", "≫": r"$\gg$", "≪": r"$\ll$", "′": "'", "∞": r"$\infty$", "≠": r"$\neq$", "√": r"$\surd$", "…": r"\ldots{}", "✓": r"$\checkmark$"}


def inline(s, extra=()):
    ph = []

    def keep(x):
        ph.append(x)
        return f"\x00{len(ph) - 1}\x00"
    for a, b in list(extra) + FORMULAS + CITES:
        s = s.replace(a, keep(b))
    s = re.sub(r"`([^`]+)`", lambda m: keep(r"\texttt{" + tex_escape(m.group(1)).replace(r"\_", r"\_\allowbreak{}").replace("/", r"/\allowbreak{}") + "}"), s)
    s = s.replace("\\_", "_").replace("\\[", "[").replace("\\]", "]").replace("\\|", "|")
    s = tex_escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
    s = re.sub(r"(\d+)([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: f"${m.group(1)}^{{{m.group(2).translate(SUP)}}}$", s)
    s = re.sub(r"([A-Za-zγφθμλαρβηεδψσℓ])([₀₁₂₃₄₅₆₇₈₉]+)", lambda m: f"${GREEK.get(m.group(1), m.group(1))}_{{{m.group(2).translate(SUB)}}}$", s)
    for k, v in GREEK.items():
        s = s.replace(k, f"${v}$")
    for k, v in UNI.items():
        s = s.replace(k, v)
    s = re.sub(r"\$\$", "", s)  # merge adjacent inline-math pieces
    s = s.replace(r"$\to$", r"$\to$\allowbreak{}")
    s = s.replace("KAN-delta", r"KAN-$\delta$")
    s = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)e-0*(\d+)(?![\w])", lambda m: (r"$10^{-" + m.group(2) + "}$") if m.group(1) == "1" else (f"${m.group(1)}" + r"\times10^{-" + m.group(2) + "}$"), s)
    for i, x in enumerate(ph):
        s = s.replace(f"\x00{i}\x00", x)
    s = s.replace("$$", "")  # merge inline-math pieces that meet a restored formula (e.g. "‖" + "θ*")
    bad = [c for c in s if ord(c) > 127]
    if bad:
        raise SystemExit(f"unmapped unicode {sorted(set(bad))} in: {s[:120]}")
    return s


def tex_escape(s):
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
             .replace("_", r"\_").replace("{", r"\{").replace("}", r"\}").replace("~", r"\textasciitilde{}").replace("^", r"\^{}")
             .replace(r"\textbackslash\{\}", r"\textbackslash{}"))


def cells(line):
    return [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


# ------------------------------------------------------------------ figures and captions (from sidecars)
def side(name):
    return json.loads((FIGS / f"{name}.json").read_text())["values"]


def pct(x):
    return f"{100 * x:.1f}\\%"


MU = {"0": "$\\mu = 0$", "1e-06": "$10^{-6}$", "1e-05": "$10^{-5}$", "0.0001": "$10^{-4}$"}
GROUP = {"real Part A": "real, further polishing", "synth. certified": "synthetic, certified", "synth. not cert.": "synthetic, not certified",
         "real capped": "real, capped"}


def pow10(x):
    """0.001 -> $10^{-3}$, 2e-07 -> $2\\times10^{-7}$"""
    m, e = f"{x:.0e}".split("e")
    e = int(e)
    return f"$10^{{{e}}}$" if m == "1" else f"${m}\\times10^{{{e}}}$"


def rng2(v, nd=2):
    lo, hi = f"{min(v):.{nd}f}", f"{max(v):.{nd}f}"
    return lo if lo == hi else f"{lo}--{hi}"


def captions():
    f1, f2, f3, f4, f5 = (side(n) for n in ["fig1_mechanism", "fig2_synthetic", "fig8_dose_response", "fig3_where", "fig9_uniqueness"])
    fa, fp, fd, fb = (side(n) for n in ["fig4_applicability", "fig5_paths", "fig6_correlation_dose", "fig7_bspline"])
    d = f1["a_deviation_relative_to_range"]; c = f1["c_rms_per_unit_t"]
    cap = {}
    cap["fig:mechanism"] = (
        "fig1_mechanism",
        "Mechanism (computed illustration) for the edge type of Section~\\ref{sec:2} (8 Gaussian RBFs on $[-1, 1]$). "
        f"\\textbf{{(a)}} Edge function produced by a unit linear coefficient pattern and its best straight line; the shaded band is the central 80\\% of the range, "
        f"where the maximum deviation is {pct(d['c80_max'])} of the function's range ({pct(d['full_max'])} over the full range). "
        "\\textbf{(b)} Penalty as a function of a tilt $t$ applied as $+t$ to edge 1 and $-t$ to edge 2: the second-difference penalty (P0) and P1 are exactly zero for every $t$; "
        f"only P2's term $\\mu Q$ grows ($\\lambda = $ {pow10(f1['b_lambda_mu_Ksilu'][0])}, $\\mu = $ {pow10(f1['b_lambda_mu_Ksilu'][1])}). "
        f"\\textbf{{(c)}} RMS change in the unit's pre-activation per unit tilt: {c['correlated']:.4f} on correlated inputs ($\\rho = 0.95$) against {c['independent']:.4f} on independent inputs "
        f"({c['ratio_independent_over_correlated']:.2f}$\\times$).")
    cells_ = [("independent", "P0"), ("independent", "P1"), ("independent", "P2"), ("correlated", "P0"), ("correlated", "P1"), ("correlated", "P2")]
    g = "; ".join(f"{a} {b}: {f2[f'{a}_{b}']['certified']}/8 certified, $\\times${f2[f'{a}_{b}']['median_growth']:.1f}" for a, b in cells_)
    cap["fig:synthetic"] = (
        "fig2_synthetic",
        "Synthetic $2\\times3$ study: $\\|\\theta\\|^2$ over exact-Hessian trust-region polishing iterations (log scale), one line per seed, "
        "coloured certified (blue) or not certified (vermillion). Rows: independent and correlated inputs; columns: P0, P1, P2. "
        f"Per panel, certified seeds and median growth (polished over early-stopped): {g}.")
    b = f3["b_growth"]; cc = f3["c_data_rel_runaway"]; dd = f3["d_foldA"]
    cap["fig:dose"] = (
        "fig8_dose_response",
        "Real-data dose-response in the null-space weight $\\mu$ (6 KAN-$\\delta$ seeds; $\\mu = 0$: polished without a null-space penalty). "
        "\\textbf{(a)} Seeds at a numerical minimum (all of them certified): "
        + "; ".join(f"{MU[m]}, {f3['a_numerical_minimum'][m]} of 6" for m in ['0', '1e-06', '1e-05', '0.0001']) + ". "
        f"\\textbf{{(b)}} $\\|\\theta\\|^2$ relative to the 107-epoch state, median (range): "
        + "; ".join(f"{MU[m]}, {b[m]['median']:.0f} ({b[m]['min']:.0f}--{b[m]['max']:.0f})" for m in ['0', '1e-06', '1e-05', '0.0001']) + ". "
        f"\\textbf{{(c)}} Training data term relative to the runaway, median: "
        + "; ".join(f"{MU[m]}, {cc[m]['median']:.3f}" for m in ['1e-06', '1e-05', '0.0001']) + ". "
        f"\\textbf{{(d)}} Fold A RMSE: early-stopped {dd['early_stopped']} (dashed line); polished 5-seed ensemble {dd['ensemble']['1e-06']:.3f} at $\\mu = 10^{{-6}}$ "
        f"and {dd['ensemble']['0.0001']:.3f} at $\\mu = 10^{{-4}}$; grey points are single seeds.")
    a = f4["a_mean_shares"]; ks = list(a)
    def sh(g, k): return f"{a[g][k]:.2f}"
    hb = f4["b_max_hidden_corr"]
    cap["fig:where"] = (
        "fig3_where",
        "Where the runaway lives. \\textbf{(a)} Mean share of $\\|\\Delta\\theta\\|^2$ by parameter block. Real data, 107 epochs$\\to$capped: layer-1 affine RBF "
        f"{sh(ks[0], 'layer-1 affine RBF')}, layer-2 affine RBF {sh(ks[0], 'layer-2 affine RBF')}, SiLU weights {sh(ks[0], 'SiLU weights')}; capped$\\to$further trust-region polishing: "
        f"{sh(ks[1], 'layer-1 affine RBF')}, {sh(ks[1], 'layer-2 affine RBF')}, {sh(ks[1], 'SiLU weights')}; synthetic correlated P0: layer-2 affine RBF "
        f"{sh(ks[2], 'layer-2 affine RBF')}, SiLU weights {sh(ks[2], 'SiLU weights')}; correlated P1: layer-2 affine RBF {sh(ks[3], 'layer-2 affine RBF')}. "
        "The real-data split was evaluated post hoc from archived weights (Appendix~\\ref{app:repro}). "
        "\\textbf{(b)} Largest absolute correlation between hidden-unit activations on the training rows, median: "
        + "; ".join(f"{GROUP.get(k, k)} {v['median']:.3f} ($n={v['n']}$)" for k, v in hb.items()) + ". Blue: certified; vermillion: not certified or runaway.")
    nn = f5["a_n_null"]; db = f5["b_dAB"]
    nnk = list(nn); dbk = list(db)
    def rng(v): return f"{min(v)}--{max(v)}"
    def zero_all(v):
        assert all(x == 0 for x in v), v
        return f"0 in all {len(v)} models"
    cap["fig:uniqueness"] = (
        "fig9_uniqueness",
        "Convergence is not uniqueness. \\textbf{(a)} Near-null Hessian eigenvalues ($|\\lambda| < 10^{-8}\\lambda_{\\max}$) at the end of polishing: "
        f"KAN-$\\delta$ baseline on fold A, median {nn[nnk[0]]['median']:.0f} (range {rng(nn[nnk[0]]['values'])}); null-space penalty $\\mu = 10^{{-4}}$ on all rows, "
        f"median {nn[nnk[1]]['median']:.0f} (range {rng(nn[nnk[1]]['values'])}); SC-KAN-$\\delta$ on fold A, {zero_all(nn[nnk[2]]['values'])}. "
        "\\textbf{The baseline counts are taken at non-stationary points}: none of the 10 baseline models was certified, and all stopped at the 500-iteration cap. "
        "\\textbf{(b)} Seed disagreement $d_{AB}$ (median, interquartile range): "
        + "; ".join(f"{k.replace('*', '$^*$').replace('1e-4', '$10^{-4}$')} {v['median']:.3f}" for k, v in db.items()) + ". "
        "Groups are evaluated on different row sets (fold A training rows or all rows), so values are not strictly comparable across row sets. "
        "Groups marked $^*$ (polished fold A) are not in any committed result: they were evaluated post hoc from the archived polished weights "
        "(SHA-verified; predictions on the 1434 fold A training rows), without any re-optimisation.")
    r = fa["ensemble_rmse_reproduced"]; cpt = fa["c_points"]
    cap["fig:applicability"] = (
        "fig4_applicability",
        "Applicability (descriptive; analysis plan committed before computing). \\textbf{(a, b)} Fold A (ethyl acetate held out, "
        f"{fa['a_1e-06']['n']} rows): change in each row's absolute error from the early-stopped 5-seed ensemble to the polished ensemble, against the "
        f"co-solvent fraction $x_{{\\mathrm{{co}}}}$ ({fa['a_1e-06']['n_distinct_xco']} distinct values); fold A RMSE {r['early']:.4f}$\\to${r['1e-06']:.4f} at "
        f"$\\mu = 10^{{-6}}$ (a) and {r['early']:.4f}$\\to${r['0.0001']:.4f} at $\\mu = 10^{{-4}}$ (b). Per-row values are evaluated post hoc from "
        "SHA-verified archived weights, with the logged RMSEs reproduced (Appendix~\\ref{app:repro}). "
        "\\textbf{(c)} Co-solvent descriptors $\\varepsilon_{\\mathrm{co}}$ and $\\ln\\eta_{\\mathrm{co}}$ in the committed min--max scaling: "
        f"the {cpt['n_train_rows']} anchored training rows ({cpt['n_train_distinct']} distinct points), fold A's co-solvent (off-pattern distance "
        f"{cpt['foldA_offpattern']:.3f}) and the {cpt['n_dec_rows']} anchored DEC test rows ({cpt['dec_offpattern']:.3f}); each held-out set occupies a single point.")
    a0 = [x["alpha"] for x in fp["synthetic_P0"]["alpha"]]; a1 = [x["alpha"] for x in fp["synthetic_P1"]["alpha"]]; a2 = [x["alpha"] for x in fp["synthetic_P2"]["alpha"]]
    ar = [v["alpha"] for v in fp["real_open"]["alpha"].values()]; cr = fp["real_closed_mu1e-4"]["rows"]
    cap["fig:paths"] = (
        "fig5_paths",
        "Certified $L_2$ regularisation paths, $\\gamma = 10^{-2}$ to $10^{-7}$ in half-decade steps (warm-started), one line per seed. "
        "\\textbf{(a, c)} Synthetic correlated inputs, 8 seeds per penalty level: $\\|\\theta^*\\|^2$ grows as a power of $1/\\gamma$ with exponent "
        f"{rng2(a0)} (P0) and {rng2(a1)} (P1), and {rng2(a2, 3)} with the null space closed (P2). "
        f"\\textbf{{(b, d)}} Real data: the 6 KAN-$\\delta$ seeds with the blind spot open (exponents {rng2(ar)}) and seeds ind0 and ind1 with the null-space "
        "penalty at $\\mu = 10^{-4}$ (post-hoc control, outcomes fixed in advance; ratio between the two smallest $\\gamma$ "
        + " and ".join(f"{cr[k]['ratio']:.2f}" for k in sorted(cr)) + "). "
        "$L^*$ is the loss without the $\\gamma$ term (for the closed paths, including $\\mu Q$). Filled markers: points included in the pre-registered fits; "
        "open markers: reached but not included; absent: not reached.")
    P2d = fd["P2"]
    cap["fig:corrdose"] = (
        "fig6_correlation_dose",
        "Correlation dose-response (synthetic [4, 3, 1], Gaussian inputs; 8 seeds per cell; the thresholds are synthetic, not real-data thresholds). "
        "Horizontal axis: $\\rho$ on a $\\log 1/(1-\\rho)$ scale. \\textbf{(a)} Share of models not certified (dashed: one half). "
        "\\textbf{(b)} Median short-path $D = \\|\\theta^*(10^{-7})\\|^2/\\|\\theta^*(10^{-5})\\|^2$ (log scale; dashed: 1.5). "
        "\\textbf{(c)} Largest hidden-unit correlation, cell median. Diamonds: P2 at $\\rho = 0.99$, "
        f"{list(P2d['certified'].values())[0]} of 8 certified, median $D = {P2d['D_med'][0]:.2f}$.")
    bb, bc = fb["b_independent"], fb["c_correlated"]
    cap["fig:bspline"] = (
        "fig7_bspline",
        "B-spline replication. \\textbf{(a)} Certified models (of 8) per cell: the Gaussian-RBF study (frozen $\\mu = 10^{-5}$) and the cubic B-spline "
        "replication (frozen $\\mu = 10^{-6}$). \\textbf{(b, c)} P2 regularisation paths at $\\mu = 10^{-6}$, continued to $\\gamma = 10^{-9}$ (post hoc, "
        "predictions fixed in advance): $\\|\\theta^*(\\gamma)\\|$ relative to each seed's certified polish minimum; shaded: the pre-fixed band [0.9, 1.1]. "
        f"Median ratio at $\\gamma = 10^{{-9}}$: {bb['median_norm_ratio_1e9']:.3f} (independent) and {bc['median_norm_ratio_1e9']:.3f} (correlated).")
    cap["fig:S1"] = ("figS1_lossdecomp", supp_caption("figS1_lossdecomp"))
    cap["fig:S2"] = ("figS2_gaugecheck", supp_caption("figS2_gaugecheck"))
    cap["fig:S3"] = ("figS3_loco", supp_caption("figS3_loco"))
    return cap


DELTA = "$\\delta$"
S2NAME = {"const_L1 ind": "constant shift in layer 1, real-data seeds", "const_L1 stu": "constant shift in layer 1, students (trained on a frozen teacher)",
          "const_L2 ind": "constant shift in layer 2, real-data seeds", "const_L2 stu": "constant shift in layer 2, students",
          "rescale ind": "unit rescaling, real-data seeds", "rescale stu": "unit rescaling, students"}


def supp_caption(name):
    v = side(name)
    if name == "figS1_lossdecomp":
        p = v["pooled_shares"]
        return ("Post hoc loss decomposition (synthetic, early$\\to$polished): share of the total-loss decrease from each term, pooled over the 8 seeds of each cell. "
                + "; ".join(f"{k}: data {d['data']:.2f}, D2 {d['D2']:.2f} (median total change {100 * d['median_total_change_rel']:+.1f}\\%)" for k, d in p.items()) + ".")
    if name == "figS2_gaugecheck":
        m = v["muQ_share"]
        return ("Post hoc gauge check at the $\\mu = 10^{-4}$ certified minima: share of the curvature along each least-squares gauge direction contributed by $\\mu Q$. Medians: "
                + "; ".join(f"{S2NAME[k]} {d['median']:.2f} ($n = {d['n']}$)" for k, d in m.items()) + ".")
    if name == "figS3_loco":
        m = v["means"]
        return ("Leave-one-co-solvent-out RMSE per fold (5-seed ensembles, early state) for the KAN-$\\delta$ baseline, SC-KAN-$\\delta$ and EX-KAN. Unweighted means: "
                + "; ".join(k.replace("δ", DELTA) + f" {d['unweighted']:.4f}" for k, d in m.items()) + ". The EA fold uses the fold A rows.")


def figure(label, name, cap, width="\\linewidth"):
    for k, v in GREEK.items():
        cap = cap.replace(k, f"${v}$")
    cap = cap.replace("$$", "")
    assert all(ord(c) < 128 for c in cap), [c for c in cap if ord(c) > 127]
    return (f"\n\\begin{{figure}}[t]\n  \\centering\n  \\includegraphics[width={width}]{{{name}.pdf}}\n"
            f"  % CAPTION-DRAFT: review (numbers from paper2/figures/{name}.json)\n  \\caption{{{cap}}}\n  \\label{{{label}}}\n\\end{{figure}}\n")


# ------------------------------------------------------------------ draft -> body.tex
FIG_REFS = [  # (exact draft substring, replacement); each inserted reference is listed in README.md
    ("it deviates from a straight line by at most 2% of its range", "it deviates from a straight line by at most 2% of its range \x01fig:mechanism|a\x01"),
    ("so the penalty is unchanged for every t.", "so the penalty is unchanged for every t \x01fig:mechanism|b\x01."),
    ("so predictions change little even for large t.", "so predictions change little even for large t \x01fig:mechanism|c\x01."),
    ("but left 0 of 8 models certified.", "but left 0 of 8 models certified \x01fig:synthetic|\x01."),
    ("was 0.110 (P0) and 0.024 (P1).", "was 0.110 (P0) and 0.024 (P1) \x01fig:where|a\x01."),
    ("(largest pairwise correlation 0.70, against 0.05 for certified models)", "(largest pairwise correlation 0.70, against 0.05 for certified models; \x02fig:where|b\x02)"),
    ("all six seeds at μ = 10⁻⁴ were certified.", "all six seeds at μ = 10⁻⁴ were certified \x01fig:dose|\x01."),
    ("their median prediction difference fell by only 26%.", "their median prediction difference fell by only 26% \x01fig:uniqueness|b\x01."),
    ("at non-stationary points), and it was less accurate.", "at non-stationary points; \x02fig:uniqueness|a\x02), and it was less accurate."),
    # revision figures (draft rev 55): two placeholders replaced by a reference; Sections 5 and 8 have no placeholder, so a reference is inserted
    ("so any change polishing makes to h is magnified where x\\_co is large (descriptive).",
     "so any change polishing makes to h is magnified where x\\_co is large (descriptive; \x02fig:applicability|a, b\x02)."),
    ("Fold A's co-solvent lies outside the training data's descriptor hull,", "Fold A's co-solvent lies outside the training data's descriptor hull \x01fig:applicability|c\x01,"),
    ("\\[Figure: regularisation paths, open vs closed, synthetic and real.\\]", "\x01fig:paths|\x01"),
    ("\\[Figure: share uncertified and median D against ρ.\\]", "\x01fig:corrdose|\x01"),
    ("none of the 16 P0 and P1 models certified (prediction held).", "none of the 16 P0 and P1 models certified (prediction held; \x02fig:bspline|a\x02)."),
    ("Extended to γ = 10⁻⁹, the paths were bounded,", "Extended to γ = 10⁻⁹, the paths were bounded \x01fig:bspline|b, c\x01,"),
]
FIG_AFTER = {  # place the figure after the paragraph that starts with (order = order of first reference = figure number)
    "fig:mechanism": "**Step 2: a direction the data cannot see.**",
    "fig:paths": "**A certified regularisation path.**",
    "fig:corrdose": "**Correlation drives non-convergence.**",
    "fig:bspline": "**The fix carries over, with a lesson about calibration.**",
    "fig:uniqueness": "**Convergence is not uniqueness.**",
}
TABLE_CAPTIONS = [  # CAPTION-DRAFT: short titles only; the draft's own notes stay in the text
    ("tab:synthetic", "Synthetic $2\\times3$ study, per cell."),
    ("tab:dose", "Dose-response in $\\mu$ on real data (seeds)."),
    ("tab:alternatives", "Structured alternatives against the early-stopped baseline."),
]


# ------------------------------------------------------------------ main-text tables generated from committed results (revision step 1)
def jload(rel):
    return json.loads((REPO / rel).read_text())


def gen_tables():
    """Each table: list of body rows (cell strings in the draft's notation) computed from committed JSON."""
    out = {}
    sr = jload("paper2/synth_results.json")["cells"]
    rows = []
    for cond, lab in [("independent", "Independent"), ("correlated", "Correlated")]:
        for lev in ("P0", "P1", "P2"):
            c = sr[f"{cond}|{lev}"]
            g = c["growth"]
            rows.append([f"{lab}, {lev}", (f"{g:.2f}×" if g < 2 else f"{g:.1f}×"), f"{c['certified']}/8", f"{c['unit']:.3f}"])
    out["tab:synthetic"] = rows
    cl, ce = jload("close_results.json"), jload("certify_results.json")
    cand = {(m, t) for m, t in ce["candidates"]}
    rows = []
    for m, lab in [("1e-06", "10⁻⁶"), ("1e-05", "10⁻⁵"), ("0.0001", "10⁻⁴")]:
        runs = [cl["runs"][f"{m}|ind {i}"] for i in range(6)]
        nmin = sum((m, f"ind_{i}") in cand for i in range(6))
        neg = max(r["neg"] for r in runs)
        gr = [r["theta2"] / r["theta2_start"] for r in runs]
        fit = [100 * (r["data"] / cl["ref_A"][str(i)] - 1) for i, r in enumerate(runs)]
        fmt = lambda x: f"{x:.1f}" if x < 2 else f"{x:.0f}"
        fit_s = (f"{min(fit):.0f}–{max(fit):.0f}% worse" if min(fit) >= 0 else f"{min(fit):+.0f}% to {max(fit):+.0f}%".replace("-", "−"))
        rows.append([lab, f"{nmin} of 6", ("none" if neg == 0 else f"up to {neg} eigenvalues"), f"{fmt(min(gr))}–{fmt(max(gr))}×", fit_s])
    out["tab:dose"] = rows
    sc, ex = jload("arch/sc_results.json"), jload("arch/ex_results.json")
    f5 = json.loads((FIGS / "fig9_uniqueness.json").read_text())["values"]["b_dAB"]
    cert = lambda k: sum(p["cert"]["certified"] for p in sc["polish"][k])
    out["tab:alternatives"] = [
        ["Baseline KAN-delta", None, f"{sc['accuracy']['base']['loco_unweighted']:.3f}", f"{cert('base')}/10", f"{sc['dab']['base']['median']:.3f}"],
        ["Shared-curve KAN", None, f"{sc['accuracy']['sc']['loco_unweighted']:.3f}", f"{cert('sc')}/10",
         f"{sc['dab']['sc']['median']:.3f} early; {f5['SC-KAN-δ fold A polished*']['median']:.3f} polished"],
        ["Two-anchor excess model", None, f"{ex['accuracy']['ex']['loco_unweighted']:.3f}", "not tested", "not tested"],
    ]
    return out


TABLE_DIFFS = []


def merge_table(label, typed_rows):
    """generated values replace the typed ones; text columns (None) keep the draft's wording; every difference is recorded"""
    gen = gen_tables()[label]
    head = typed_rows[0]
    typed = [cells(r) for r in typed_rows[2:]]
    assert len(gen) == len(typed), (label, len(gen), len(typed))
    merged = []
    for g, t in zip(gen, typed):
        row = []
        for j, (gv, tv) in enumerate(zip(g, t)):
            v = tv if gv is None else gv
            if gv is not None and gv != tv:
                TABLE_DIFFS.append(dict(table=label, row=t[0], column=cells(head)[j], typed=tv, generated=gv))
            row.append(v)
        merged.append("| " + " | ".join(row) + " |")
    return [head, typed_rows[1]] + merged


IN_E = ["fig:synthetic", "fig:where", "fig:applicability", "fig:dose"]   # draft rev 58: moved to Appendix E (supplementary figures), in this order


def fref(lab, suffix):
    f = f"Figure~\\ref{{{lab}}}{suffix}"
    return f"Appendix~\\ref{{app:supp}}, {f}" if lab in IN_E else f


def refs(s):
    s = re.sub("\x01([^|]+)\\|([^\x01]*)\x01", lambda m: f"({fref(m.group(1), m.group(2))})", s)
    return re.sub("\x02([^|]+)\\|([^\x02]*)\x02", lambda m: fref(m.group(1), m.group(2)), s)


def table_tex(rows, label, caption):
    head, body = rows[0], rows[2:]
    n = len(head)
    spec = "l" + "l" * (n - 1) if n <= 4 else "l" + "p{0.2\\linewidth}" * (n - 1)
    if label == "tab:alternatives":
        spec = "l p{0.24\\linewidth} l l p{0.16\\linewidth}"
    if label == "tab:dose":
        spec = "l l p{0.18\\linewidth} l p{0.16\\linewidth}"
    out = [f"\\begin{{table}}[{'h' if label == 'tab:alternatives' else 't'}]\n  \\centering\n  % CAPTION-DRAFT: review\n  \\caption{{{caption}}}\n  \\label{{{label}}}\n  \\small\n  \\begin{{tabular}}{{{spec}}}\n  \\toprule"]
    out.append("  " + " & ".join(inline(c) for c in cells(head)) + " \\\\\n  \\midrule")
    for r in body:
        out.append("  " + " & ".join(inline(c) for c in cells(r)) + " \\\\")
    out.append("  \\bottomrule\n  \\end{tabular}\n\\end{table}")
    return "\n".join(out)


def build_body(cap):
    md = (REPO / "paper2" / "draft" / "paper2_draft.md").read_text(encoding="utf-8")
    for a, b in FIG_REFS:
        assert md.count(a) == 1, f"figure-ref anchor not unique/absent: {a!r}"
        md = md.replace(a, b)
    lines = md.split("\n")
    title = lines[0].lstrip("# ").strip()
    out, abstract, i, tcount, placed = [], [], 0, 0, set()
    mode = None
    paras = []
    # split into blocks
    blocks, cur = [], []
    for ln in lines[1:]:
        if ln.strip() == "":
            if cur:
                blocks.append(cur); cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append(cur)
    in_abs, done = False, False
    for blk in blocks:
        first = blk[0]
        if first.startswith("Oct ") and "·" in first:
            continue                                  # byline: dropped (author handled by the tmlr \author block)
        if first.startswith("## References"):
            done = True
        if first.startswith("## Appendix F"):
            done = False
            out.append("\x03APPF\x03")
            out.append(f"\\section{{{inline(first.split(':', 1)[1].strip())}}}\\label{{app:F}}\n")
            continue
        if done:
            continue
        mf = re.match(r"\*\*F\.(\d) (.+?)\.\*\*\s*(.*)", " ".join(blk), re.S)
        if mf:                                        # "**F.n Title.** text" -> subsection F.n
            out.append(f"\\subsection{{{inline(mf.group(2))}}}\\label{{app:F{mf.group(1)}}}\n")
            if mf.group(3).strip():
                out.append(refs(inline(mf.group(3).strip())) + "\n")
            continue
        if first.startswith("## Abstract"):
            in_abs = True; continue
        m = re.match(r"## (\d+) (.+)", first)
        if m:
            if in_abs:
                out.append("\\end{abstract}\n"); in_abs = False
            out.append(f"\\section{{{inline(m.group(2))}}}\\label{{sec:{m.group(1)}}}\n")
            continue
        if first.startswith("```latex"):
            body = "\n".join(blk[1:-1])
            out.append(f"\\begin{{equation}}\n  {body}\n\\end{{equation}}\n")
            continue
        if first.startswith("|"):
            lab, c = TABLE_CAPTIONS[tcount]; tcount += 1
            out.append(table_tex(merge_table(lab, blk), lab, c) + "\n")
            continue
        if re.match(r"(- |\d+\. )", first):
            env = "itemize" if first.startswith("- ") else "enumerate"
            items = [re.sub(r"^(- |\d+\. )", "", x) for x in blk]
            out.append(f"\\begin{{{env}}}\n" + "\n".join("  \\item " + refs(inline(x)) for x in items) + f"\n\\end{{{env}}}\n")
        else:
            txt = refs(inline(" ".join(blk)))
            out.append(txt + "\n")
        if in_abs and len(out) == 0:
            pass
        for lab, start in FIG_AFTER.items():
            if lab not in placed and any(x.startswith(start) for x in blk):
                name, c = cap[lab]
                out.append(figure(lab, name, c)); placed.add(lab)
    body = "\n".join(out)
    body, appf = body.split("\x03APPF\x03") if "\x03APPF\x03" in body else (body, "")
    body = "\\begin{abstract}\n" + body if not body.startswith("\\begin{abstract}") else body
    assert placed == set(FIG_AFTER), placed
    return title, body, appf


# ------------------------------------------------------------------ appendix A: pre-registered predictions (generated)
def md_tables(text):
    tabs, cur = [], []
    for ln in text.splitlines() + [""]:
        if ln.startswith("|"):
            cur.append(ln)
        elif cur:
            tabs.append(cur); cur = []
    return tabs


def longtable(rows, colspec, header):
    out = [f"{{\\footnotesize\\setlength{{\\tabcolsep}}{{3pt}}\n\\begin{{longtable}}{{{colspec}}}\n\\toprule", " & ".join(header) + " \\\\\n\\midrule\n\\endhead"]
    for r in rows:
        out.append(" & ".join(r) + " \\\\")
    out.append("\\bottomrule\n\\end{longtable}}")
    return "\n".join(out)


def md_section(path, heading, level="## "):
    """committed text under a markdown heading (up to the next heading of the same level)"""
    out, on = [], False
    for ln in (REPO / path).read_text().splitlines():
        if ln.startswith(level):
            if on:
                break
            on = ln[len(level):].startswith(heading)
            continue
        if on:
            out.append(ln)
    return "\n".join(out).strip()


def h1(path):
    return (REPO / path).read_text().splitlines()[0].lstrip("# ").strip()


def described(title, path, paras):
    """study description: the pre-registration's own title and the committed paragraph(s) given (question, theory, model)"""
    txt = " ".join(" ".join(x.strip().lstrip("- ") for x in p.splitlines() if x.strip()) for p in paras if p)
    return (f"\\subsection{{{title}}}\n\\noindent\\textit{{Pre-registration:}} ``{inline(h1(path))}''. " + inline(txt) + "\n")


def pred_bullets(path, heading="## Pre-registered predictions"):
    """{ID: committed prediction wording} from '- **ID.** text' bullets"""
    sec = md_section(path, heading.lstrip("# "))
    out, cur = {}, None
    for ln in sec.splitlines():
        m = re.match(r"- \*\*([A-Z]\d)\.\*\* (.*)", ln)
        if m:
            cur = m.group(1); out[cur] = m.group(2).strip()
        elif cur and ln.startswith("  ") and not ln.strip().startswith("-"):
            out[cur] += " " + ln.strip()
        elif not ln.startswith("  "):
            cur = None
    return out


def verdict_word(s):
    s = s.replace("*", "")
    for w in ("HELD", "FAILED", "held", "failed", "inconclusive", "pending"):
        if w in s:
            return w.lower()
    return s


A_COLS = "R{0.07\\linewidth} R{0.28\\linewidth} R{0.35\\linewidth} R{0.09\\linewidth} R{0.11\\linewidth}"
A_HEAD = ["ID", "Prediction (threshold)", "Result", "Verdict", "Commits"]
STUDY = {"rule_prereg.md": "Rule", "rank_prereg.md": "Rank", "trust_prereg.md": "Trust", "stage2_prereg.md": "Stage 2",
         "close_prereg.md": "Close", "certify_prereg.md": "Certify"}   # the case-study summary's own study names


def readable(s):
    return re.sub(r"\s*\(`[^`]+`\)", "", s)


def appendix_a():
    o = ["\\section{All pre-registered predictions}\\label{app:prereg}\n",
         "Generated by \\texttt{paper2/tex/build\\_tex.py} from the committed verdict sections; the cell text is the committed wording. "
         "Each study opens with its pre-registration's title and its committed question (or theory). "
         "Commits are given as pre-registration $\\to$ results (in the anonymous version, as neutral labels S-NN). Thresholds are part of each prediction's wording.\n"]
    cm = lambda a, b: f"{a}$\\to$\\allowbreak {b}"
    # ---- synthetic
    p = "paper2/synth_prereg.md"; txt = (REPO / p).read_text()
    vt = [t for t in md_tables(txt.split("## Verdicts")[1]) if t[0].startswith("| Prediction")][0]
    pre, res = first_commit(p), commit_with(p, "**X2** certified")
    rows = []
    for r in vt[2:]:
        c = cells(r)
        mid = re.match(r"\*\*(X\d)\*\* (.*)", c[0])
        rows.append([inline(mid.group(1)), inline(mid.group(2)), inline(c[2]), inline(c[1]), cm(pre, res)])
    o.append(described("Synthetic controlled study", p, [md_section(p, "Design").split("\n- **Penalty")[0]]))
    o.append(longtable(rows, A_COLS, A_HEAD))
    # ---- real data: case-study summary tables
    p = "electrolyte_case_study_summary.md"; txt = (REPO / p).read_text()
    setting = re.search(r"\*\*Setting\.\*\* (.*?)\n\n", txt, re.S).group(1)
    tabs = md_tables(txt)
    rows = []
    t1 = [t for t in tabs if t[0].startswith("| Study | Prediction")][0]
    study = ""
    for r in t1[2:]:
        c = cells(r)
        study = c[0] or study
        rows.append([inline(readable(study)), inline(c[1]), inline(c[3]), inline(c[2]), inline(c[4].replace(" → ", "→"))])
    rule = re.search(r"\*\*0 columns were removed and the gate stopped;\*\* (I1–I5) are inconclusive\s*\(prereg (\w+) → results (\w+)\)", txt)
    rows.append([inline(STUDY["rule_prereg.md"]), inline(rule.group(1) + " (gate)"), inline("0 columns removed; gate stopped"), "inconclusive",
                 cm(rule.group(2), rule.group(3))])
    rk = [t for t in tabs if t[0].startswith("| Prediction | Arm G")][0]
    rc = cm(first_commit("rank_prereg.md"), first_commit("rank_results.json"))
    for r in rk[2:]:
        c = cells(r)
        rows.append([inline(STUDY["rank_prereg.md"]), inline(c[0]), inline("arm G: " + c[1] + ("; arm F: " + c[2] if c[2] else "")), "see result", rc])
    ts = [t for t in tabs if t[0].startswith("| Prediction | Verdict | Key number |")][0]
    for r in ts[2:]:
        c = cells(r)
        which = "trust_prereg.md" if c[0].startswith("T") else "stage2_prereg.md"
        res_file = "trust_results.json" if which.startswith("trust") else "stage2_results.json"
        rows.append([inline(STUDY[which]), inline(c[0]), inline(c[2]), inline(c[1]), cm(first_commit(which), first_commit(res_file))])
    cv = [t for t in tabs if t[0].startswith("| Prediction | Verdict | Key number | Prereg")][0]
    for r in cv[2:]:
        c = cells(r)
        which = "close_prereg.md" if c[0].startswith("V") else "certify_prereg.md"
        rows.append([inline(STUDY[which]), inline(c[0]), inline(c[2]), inline(c[1]), inline(c[3].replace(" → ", "→"))])
    o.append(f"\\subsection{{Real-data case study}}\n\\noindent\\textit{{From the case-study close-out:}} {inline(setting)}\n")
    o.append("Rows are as committed in the case-study summary (one row may cover several predictions, e.g.\\ F2--F5). "
             "The rank intervention's verdicts are given per arm in the result column.\n")
    o.append(longtable(rows, "R{0.14\\linewidth} R{0.26\\linewidth} R{0.29\\linewidth} R{0.10\\linewidth} R{0.11\\linewidth}",
                       ["Study", "Prediction (threshold)", "Result", "Verdict", "Commits"]))
    tally = [t for t in tabs if t[0].startswith("| Study | Held")][0]
    trow = [[inline(readable(x)) for x in cells(r)] for r in tally[2:]]
    o.append("\\noindent Committed tally of the case study:\n")
    o.append(longtable(trow, "R{0.55\\linewidth} r r r", ["Study", "Held", "Failed", "Inconclusive"]))
    # ---- structured alternatives
    for p, nm, model in [("arch/sc_prereg.md", "Shared-curve KAN", "Model: SC"), ("arch/ex_prereg.md", "Two-anchor excess model", "Model: EX")]:
        txt = (REPO / p).read_text()
        vt = [t for t in md_tables(txt.split("## Verdicts")[-1]) if t[0].startswith("| | Prediction")][0]
        key = cells(vt[2])[0]
        res = commit_with(p, f"| {key} |")
        rows = [[inline(c[0]), inline(c[1]), inline(c[3]), inline(c[2].replace("*", "").lower()), cm(first_commit(p), res)] for c in map(cells, vt[2:])]
        msec = [x for x in md_section(p, model).splitlines() if x.strip()]
        o.append(described(nm, p, ["\n".join(msec[:2])]))
        o.append("Run after the case-study close-out; not part of the 65 predictions above.\n")
        o.append(longtable(rows, A_COLS, A_HEAD))
    o.append("\\paragraph{Revision studies.} The studies below were run for the revision, each with its own pre-registration.\n")
    # ---- freeze test
    p = "paper2/revision/freeze_prereg.md"; txt = (REPO / p).read_text()
    pre = first_commit(p)
    rows = []
    for t, anchor in ((md_tables(txt.split("### Verdicts, part 1")[1].split("###")[0])[0], "| Z1 | Arm A certified"),
                      (md_tables(txt.split("### Verdicts, part 2")[1].split("###")[0])[0], "| Z5 | Arm D branch")):
        res = commit_with(p, anchor)
        for r in t[2:]:
            c = cells(r)
            if verdict_word(c[2]) == "pending":
                continue                                 # superseded by part 2
            rows.append([inline(c[0]), inline(c[1]), inline(c[3]), inline(c[2].replace("**", "")), cm(pre, res)])
    o.append(described("Freeze test: is composition necessary? (and weight decay)", p, [md_section(p, "Theory being tested")]))
    o.append(longtable(rows, A_COLS, A_HEAD))
    # ---- infimum test
    p = "paper2/revision/infimum_prereg.md"; txt = (REPO / p).read_text()
    pre = first_commit(p)
    rows = []
    for ph, anchor in (("### PHASE 1", "### PHASE 1 (recorded"), ("### PHASE 2", "### PHASE 2 (recorded")):
        t = md_tables(txt.split(ph)[1].split("\n### ")[0])[0]
        res = commit_with(p, anchor)
        for r in t[2:]:
            c = cells(r)
            rows.append([inline(c[0]), inline(f"{ph[4:].title()}: " + c[1]), inline(c[3]), inline(c[2].replace("**", "")), cm(pre, res)])
    amend = txt.split("### POST-HOC AMENDMENT (2026-10-19")[1].split("\n### ")[0]
    plateau = re.search(r"- \*\*Plateau \(expected\):\*\* (.*)", amend).group(1)
    ctrl = txt.split("### POST-HOC REAL-DATA CONTROL ARM")[1].split("\n### ")[0]
    outcome = re.search(r"- \*\*Outcome: PLATEAU, by the fixed rule\.\*\* (.*)", ctrl).group(1)
    rows.append([inline("Control"), inline("Post hoc, outcomes fixed before running. Plateau: " + plateau), inline(outcome), "plateau",
                 cm(commit_with(p, "### POST-HOC AMENDMENT (2026-10-19"), commit_with(p, "### POST-HOC REAL-DATA CONTROL ARM"))])
    o.append(described("Infimum test: a certified regularisation path", p, [md_section(p, "Question")]))
    o.append("Predictions I1--I4 apply per phase (Phase 1 synthetic, Phase 2 real data); the real-data control arm was added after Phase 2 "
             "with its outcomes fixed before it ran.\n")
    o.append(longtable(rows, A_COLS, A_HEAD))
    # ---- dose-response
    p = "paper2/revision/dose_prereg.md"; txt = (REPO / p).read_text()
    pre, wording = first_commit(p), pred_bullets(p)
    vsec = txt.split("### Verdicts (recorded")[1].split("\n### ")[0]
    res = commit_with(p, "### Verdicts (recorded 2026-10-21")
    rows = []
    for r in md_tables(vsec)[0][2:]:
        c = cells(r)
        m = re.match(r"\*\*(R\d)\*\*", c[0])
        if not m:
            rows.append(["", inline(c[0]), inline(f"P0: {c[1]}; P1: {c[2]}"), inline(verdict_word(c[1]) + " / " + verdict_word(c[2])), cm(pre, res)])
            continue
        rows.append([m.group(1), inline(wording[m.group(1)]), inline(f"P0: {c[1]}; P1: {c[2]}".replace("**", "")),
                     inline(verdict_word(c[1]) + " / " + verdict_word(c[2])), cm(pre, res)])
    r2 = re.search(r"- \*\*R2 HELD\.\*\* (.*)", vsec).group(1)
    rows.insert(1, ["R2", inline(wording["R2"]), inline(r2), "held", cm(pre, res)])
    o.append(described("Correlation dose-response", p, [md_section(p, "Question")]))
    o.append("Verdict column: P0 / P1. The thresholds are synthetic ([4, 3, 1], Gaussian inputs), not real-data thresholds.\n")
    o.append(longtable(rows, A_COLS, A_HEAD))
    # ---- B-spline replication and its post-hoc follow-up
    p = "paper2/revision/bspline_prereg.md"; txt = (REPO / p).read_text()
    pre, wording = first_commit(p), pred_bullets(p)
    vsec = txt.split("### Verdicts, Step 1")[1].split("\n### ")[0]
    res = commit_with(p, "### Verdicts, Step 1")
    rows, cur = [], ""
    for r in md_tables(vsec)[0][2:]:
        c = cells(r)
        m = re.match(r"\*\*(B\d)\*\*", c[0])
        if m:
            cur = m.group(1)
        rows.append([cur if m else "", inline(wording[cur]) if m else "", inline(f"{c[1]}: {c[2]}"), inline(c[3].replace("**", "")), cm(pre, res)])
    add = txt.split("### POST-HOC ADDENDUM (2026-10-24")[1].split("\n### ")[0]
    out = txt.split("### POST-HOC results (recorded 2026-10-24")[1].split("\n### ")[0]
    apre, ares = commit_with(p, "### POST-HOC ADDENDUM (2026-10-24"), commit_with(p, "### POST-HOC results (recorded 2026-10-24")
    p1 = re.search(r"\*\*Predictions \(fixed now\):\*\*\n((?:  - .*\n)+)", add).group(1)
    p1 = "; ".join(x.strip()[2:] for x in p1.splitlines())
    t1 = md_tables(out.split("**Check 1")[1].split("**Check 2")[0])[0]
    r1 = "; ".join(f"{cells(x)[0]}: D' {cells(x)[1]}, norm ratio {cells(x)[2]}" for x in t1[2:])
    rd1 = re.search(r"- \*\*Reading: (BOTH HOLD)\.\*\*", out).group(1).lower()
    rows.append(["Check 1", inline("Post hoc, predictions fixed in advance; P2 path extended to γ = 10⁻⁹: " + p1), inline(r1), inline(rd1), cm(apre, ares)])
    p2 = re.search(r"2\. P2 at the RBF study's μ = 1e-5.*?- \*\*Prediction \(fixed now\):\*\* (.*?)\n", add, re.S).group(1)
    t2 = md_tables(out.split("**Check 2")[1].split("**Check 3")[0])[0]
    r2 = "; ".join(f"{cells(x)[0]}: {cells(x)[1]} certified, median D {cells(x)[2]}" for x in t2[2:])
    rows.append(["Check 2", inline("Post hoc, prediction fixed in advance; P2 at μ = 10⁻⁵: " + p2), inline(r2),
                 inline(cells(t2[2])[3] + " / " + cells(t2[3])[3]), cm(apre, ares)])
    o.append(described("B-spline replication", p, [md_section(p, "Question")]))
    o.append("Step 1 verdicts at the frozen $\\mu = 10^{-6}$, then the post-hoc follow-up checks whose predictions and readings were fixed before computing "
             "(check 3 is descriptive; Appendix~\\ref{app:posthoc}).\n")
    o.append(longtable(rows, A_COLS, A_HEAD))
    return "\n".join(o)


# ------------------------------------------------------------------ appendix B: post-hoc analyses (generated from files + git)
IP, DP, BP = "paper2/revision/infimum_prereg.md", "paper2/revision/dose_prereg.md", "paper2/revision/bspline_prereg.md"
POSTHOC = [  # (label, name, files shown, commits: list of (path, committed text) or None = the commits that added the files)
    ("Post hoc", "Anatomy of the real-data runaway", ["anatomy_plan.md", "anatomy.py", "anatomy_output.txt"], None),
    ("Post hoc", "Gauge check and regularised teacher (exploratory)", ["gaugecheck.py", "gaugecheck_notes.md", "gaugecheck_output.txt"], None),
    ("Post hoc", "Anatomy of the synthetic runaway", ["paper2/anatomy_synth.py", "paper2/anatomy_synth_output.txt"], None),
    ("Post hoc", "Loss decomposition along the runaway", ["paper2/lossdecomp.py", "paper2/lossdecomp_output.txt"], None),
    ("Post hoc", "Descriptive checks (hidden-unit correlation, tilt affinity, references)", ["paper2/checks/plan.md", "paper2/checks/checks.py", "paper2/checks/checks_output.txt"], None),
    ("Post hoc", "Degeneracy follow-up: restart, scaling and flat-direction probes of the freeze test's endpoints",
     ["paper2/revision/degeneracy.py", "paper2/revision/degeneracy_output.txt"], None),
    ("Post hoc", "Infimum test, Phase 0: re-classification of the freeze test's degenerate endpoints", [IP], [(IP, "### PHASE 0 (POST-HOC")]),
    ("Post hoc", "Infimum test: cold-start basin check", [IP], [(IP, "### POST-HOC cold-start results")]),
    ("Post hoc, outcomes fixed in advance", "Infimum test: real-data control arm (null-space penalty at $\\mu = 10^{-4}$)", [IP],
     [(IP, "### POST-HOC AMENDMENT (2026-10-19"), (IP, "### POST-HOC REAL-DATA CONTROL ARM")]),
    ("Post hoc, rule fixed in advance", "Dose-response settling check (short path against the certified polished minimum)",
     ["paper2/revision/dose_posthoc.py", "paper2/revision/dose_posthoc_results.json"], None),
    ("Post-outcome diagnostic, criteria fixed in advance", "B-spline null-space diagnostic (constant-shift symmetry)",
     ["paper2/revision/bspline_nullspace_diag.py", "paper2/revision/bspline_nullspace_diag.json"], None),
    ("Post hoc, predictions fixed in advance", "B-spline follow-up: path extension, $\\mu = 10^{-5}$, the non-certifying independent models",
     [BP, "paper2/revision/bspline_posthoc.py", "paper2/revision/bspline_posthoc_results.json"],
     [(BP, "### POST-HOC ADDENDUM (2026-10-24"), (BP, "### POST-HOC results (recorded 2026-10-24")]),
    ("Descriptive, plan committed first", "Applicability analysis (no predictions)",
     ["paper2/revision/appdomain_plan.md", "paper2/revision/appdomain.py", "paper2/revision/appdomain_output.txt"], None),
    ("Post hoc", "Applicability: artefact check (the ensembles' own errors against distance)", ["paper2/revision/appdomain_results.json"],
     [("paper2/revision/appdomain_results.json", "analysis2_posthoc_artefact_check")]),
]


def appendix_b():
    o = ["\\section{Post-hoc analyses}\\label{app:posthoc}\n",
         "Every analysis below except one was defined after seeing results; the exception, the applicability analysis, is descriptive, with its plan "
         "committed before computing. Each row's label states how the analysis was defined, as recorded in its own files; none changes a pre-registered "
         "verdict, and all are reported as exploratory. The description is the subject line of the commit that added the analysis (plan and results "
         "commits where both exist).\n",
         "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\\begin{longtable}{R{0.28\\linewidth} R{0.52\\linewidth} R{0.10\\linewidth}}\n\\toprule\nAnalysis & Commit subject (as committed) & Commit \\\\\n\\midrule\n\\endhead"]
    for label, name, files, how in POSTHOC:
        if how is None:
            cs = [first_commit(files[0])] + ([first_commit(files[-1])] if len(files) > 1 else [])
        else:
            cs = [commit_with(p, t) for p, t in how]
        cs = list(dict.fromkeys(cs))
        head = f"\\textbf{{{label}.}} {inline(name) if '$' not in name else name} \\newline \\footnotesize{{\\texttt{{{tex_escape(', '.join(files)).replace(chr(92) + '_', chr(92) + '_' + chr(92) + 'allowbreak{}').replace('/', '/' + chr(92) + 'allowbreak{}')}}}}}"
        for j, c in enumerate(cs):
            o.append(f"{head if j == 0 else ''} & {inline(git('log', '-1', '--format=%s', c))} & {c} \\\\")
    rt = git("log", "--format=%h %s", "--grep", "regularised-teacher").splitlines()
    for ln in rt:
        h, s = ln.split(" ", 1)
        o.append(f"\\textbf{{Post hoc.}} {inline('Regularised teacher (weights pinned)')} & {inline(s)} & {h} \\\\")
    o.append("\\bottomrule\n\\end{longtable}}")
    return "\n".join(o)


# ------------------------------------------------------------------ appendix C: reproducibility map (from MANIFEST.md)
def appendix_c():
    man = (FIGS / "MANIFEST.md").read_text()
    t = [x for x in md_tables(man) if x[0].startswith("| Figure")][0]
    rows = []
    for r in t[2:]:
        c = cells(r)
        lab = {"fig1": "fig:mechanism", "fig2": "fig:synthetic", "fig3": "fig:where", "fig4": "fig:applicability", "fig5": "fig:paths",
               "fig6": "fig:corrdose", "fig7": "fig:bspline", "fig8": "fig:dose", "fig9": "fig:uniqueness",
               "figS1": "fig:S1", "figS2": "fig:S2", "figS3": "fig:S3"}[c[0].split("_")[0]]
        fname = tex_escape(c[0]).replace("\\_", "\\_\\allowbreak{}")
        rows.append([f"Figure~\\ref{{{lab}}} \\figname{{{fname}}}", inline(c[2]), inline(c[3]), inline(c[4])])   # c[1]: paper number (the \\ref gives it)
    o = ["\\section{Reproducibility map}\\label{app:repro}\n",
         "Generated from \\texttt{paper2/figures/MANIFEST.md}. Every figure is rebuilt by its script from committed results; "
         "each figure's sidecar JSON lists the data files (path, commit, SHA-256), any archived weights (SHA-256, archive) and every plotted number. "
         "\\REPOSTATEMENT\n",
         longtable(rows, "R{0.15\\linewidth} R{0.13\\linewidth} R{0.42\\linewidth} R{0.19\\linewidth}", ["Figure", "Script", "Data files (commit)", "Archived weights"])]
    tabs = [("Table~\\ref{tab:synthetic}", "typed in the draft from", "\\texttt{paper2/synth\\_results.json}, \\texttt{paper2/synth\\_output.txt}", first_commit("paper2/synth_results.json")),
            ("Table~\\ref{tab:dose}", "typed in the draft from", "\\texttt{close\\_results.json}, \\texttt{certify\\_results.json}, \\texttt{stage2\\_results.json}",
             ", ".join(first_commit(f) for f in ["close_results.json", "certify_results.json", "stage2_results.json"])),
            ("Table~\\ref{tab:alternatives}", "typed in the draft from", "\\texttt{arch/sc\\_results.json}, \\texttt{arch/ex\\_results.json}, \\texttt{paper2/figures/fig9\\_uniqueness.json}",
             ", ".join(first_commit(f) for f in ["arch/sc_results.json", "arch/ex_results.json", "paper2/figures/fig9_uniqueness.json"]))]
    o.append("\\noindent Tables in the main text (no script; values typed in the draft):\n")
    o.append(longtable([[a, b, c, d] for a, b, c, d in tabs], "R{0.14\\linewidth} R{0.14\\linewidth} R{0.45\\linewidth} R{0.16\\linewidth}",
                       ["Table", "Source", "Data files", "Commits"]))
    notes = man.split("## Panels that use deterministic evaluations of archived weights")[1].split("## Not plotted")[0]
    o.append("\\noindent Panels evaluated post hoc from archived weights (from \\texttt{MANIFEST.md}):\n")
    items = [x.strip()[2:] for x in notes.splitlines() if x.strip().startswith("- ")]
    o.append("\\begin{itemize}\n" + "\n".join("  \\item " + inline(x) for x in items) + "\n\\end{itemize}")
    return "\n".join(o)


# ------------------------------------------------------------------ appendix D: research process (prose = DRAFT-TEXT; records generated from git)
def appendix_d():
    o = ["\\section{Research process}\\label{app:process}\n",
         "% DRAFT-TEXT: review (new prose, not in the source draft; it states only the protocol followed, no results). First and last paragraphs: author-supplied wording (2026-10-05)",
         "\\paragraph{Pre-registration.} Every study committed a pre-registration file (predictions, thresholds, interpretation rules) "
         "before the script that tested it existed. Verdicts were appended below the committed text; the pre-registered text was never edited, "
         "and corrections to reported numbers were made in separate, labelled commits.",
         "\\paragraph{Amendments.} A change to a design after its pre-registration was committed as a dated amendment section before any affected model was trained, "
         "with its reason; the original text was kept. Choices forced by the code or data at run time were logged as numbered discrepancies at the top of each output log.",
         "\\paragraph{Files win.} Instructions to the agent stated that committed files take precedence over any number or description in a prompt; "
         "discrepancies between a brief and the files were flagged rather than resolved silently.",
         "\\paragraph{Data and weights.} Model weights were never committed: they were archived outside the repository with a manifest of SHA-256 hashes and producing commits, "
         "and every later analysis verified the hashes and held its inputs in memory. Post-hoc analyses were labelled as such in their files and changed no verdict.",
         "\\paragraph{AI assistance.} Code, runs and logs were produced by an AI coding agent working under this protocol. "
         "Study designs, briefs and the manuscript were developed with an AI assistant. The author set the research questions, reviewed and approved "
         "every brief and amendment, decided each next stage, and takes responsibility for all content. Crashes were recorded in the logs and fixed "
         "in separate commits labelled as implementation-only.\n",
         "\\noindent Pre-registrations, amendments and crash records (from the commit history):\n"]
    rows = []
    for ln in git("log", "--reverse", "--format=%h|%ad|%s", "--date=short").splitlines():
        h, d, subj = ln.split("|", 2)
        low = subj.lower()
        if any(k in low for k in ["pre-register", "amendment", "crash", "implementation only", "as first run"]):
            kind = "amendment" if "amendment" in low else "crash record / fix" if any(k in low for k in ["crash", "implementation only", "as first run"]) else "pre-registration"
            rows.append([h, d, kind, inline(subj)])
    o.append(longtable(rows, "R{0.08\\linewidth} R{0.10\\linewidth} R{0.14\\linewidth} R{0.58\\linewidth}", ["Commit", "Date", "Kind", "Commit subject"]))
    o.append(after_outcome())
    return "\n".join(o)


def grab(path, start, kind="bullet"):
    """committed text that starts with `start`: a bullet with its continuation lines, or a paragraph up to the next blank line"""
    lines = (REPO / path).read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.lstrip().startswith(start))
    ind = len(lines[i]) - len(lines[i].lstrip())
    out = [lines[i].strip()]
    for ln in lines[i + 1:]:
        if not ln.strip():
            break
        if kind == "bullet" and (len(ln) - len(ln.lstrip()) <= ind or ln.lstrip().startswith("- ")):
            break
        out.append(ln.strip())
    t = " ".join(out)
    return t[2:] if t.startswith("- ") else t


def cdate(c):
    return git("log", "-1", "--format=%ad", "--date=short", c)


def after_outcome():
    """every change made after an outcome was known, quoted from the committed files (date and commit of the quoted text)"""
    SP, FP, AP = "paper2/synth_prereg.md", "paper2/revision/freeze_prereg.md", "paper2/revision/appdomain_plan.md"
    rows = []
    def add(what, path, start, kind="bullet"):
        c = commit_with(path, start)
        rows.append([inline(what), inline(grab(path, start, kind)), cdate(c), c])
    # B-spline calibration step: the committed list of amendments made after outcomes were known
    txt = (REPO / BP).read_text().split("### Verdicts, Step 1")[1]
    lst = re.search(r"- \*\*Amendments to Step 0 made after outcomes were known:\*\*\n((?:  - .*\n)+)", txt).group(1)
    for ln in lst.splitlines():
        item = ln.strip()[2:].rstrip(";.")
        h = re.search(r"\((\w{7})\)$", item).group(1)
        rows.append([inline("B-spline replication, calibration step (Step 0)"), inline(item[:item.rindex("(")].strip()), cdate(h), h])
    add("Dose-response: runtime correlation tolerance", DP, "- **(a)** The runtime correlation tolerance")
    add("Dose-response: added stop reason", DP, "- **(b)** \"Solver error (trust-exact subproblem)\"")
    add("B-spline replication: added stop reason (hang)", BP, "- A polish killed after hanging records")
    add("B-spline replication: added stop reason (watchdog)", BP, "- **Recorded as** \"solver error (hang); ended by the staleness watchdog\"")
    add("Synthetic study: X6, pre-registered reading did not fit", SP, "- **X6's rule maps failure")
    add("Freeze test: Z2, branch taken and run resumed", FP, "By the fixed interpretation, Z2 failing", kind="para")
    add("Freeze test: Z2, pre-registered reading did not fit", FP, "- **Z2's branch premise did not occur.**")
    add("Applicability analysis: predictive follow-up dropped", AP, "The parked predictive study is DROPPED", kind="para")
    return ("\\subsection*{Changes made after an outcome was known}\n"
            "% DRAFT-TEXT: review (one provenance sentence; the table is generated)\n"
            "Generated from the committed files: each entry quotes the committed wording, with the date and commit of the quoted text.\n"
            + longtable(rows, "R{0.18\\linewidth} R{0.54\\linewidth} R{0.10\\linewidth} R{0.08\\linewidth}", ["Change", "Committed wording", "Date", "Commit"]))


def appendix_e(cap):
    o = ["\\section{Supplementary figures}\\label{app:supp}\n"]
    for lab in IN_E:                                  # moved from the main text in draft rev 58; captions and sidecars unchanged
        name, c = cap[lab]
        o.append(figure(lab, name, c).replace("[t]", "[htbp]"))
    for lab in ["fig:S1", "fig:S2", "fig:S3"]:
        name, c = cap[lab]
        o.append(figure(lab, name, c, width="0.5\\linewidth" if lab == "fig:S2" else "\\linewidth").replace("[t]", "[h]"))
    o.append("\\clearpage  % flush Appendix E's figures before Appendix F")
    return "\n".join(o)


# ------------------------------------------------------------------ appendix H: data, splits and chronology (from datasplit_results.json + committed code and logs)
def code_line(path, start):
    """the committed source line that starts with `start` (stripped), with its line number"""
    for i, ln in enumerate((REPO / path).read_text().splitlines(), 1):
        if ln.strip().startswith(start):
            return i, ln.strip()
    raise SystemExit(f"code line not found: {path}: {start}")


def appendix_h():
    DP_ = "paper2/revision/datasplit_results.json"
    d = jload(DP_); c = d["counts"]; a = d["anchor"]; cm = d["composition"]; fa = d["foldA"]; t = d["target"]
    dc = first_commit(DP_)
    tt = lambda s: "\\texttt{" + tex_escape(s).replace("\\_", "\\_\\allowbreak{}").replace("/", "/\\allowbreak{}") + "}"
    q = lambda path, start: (lambda n, s: f"{tt(path)}:{n} ({first_commit(path)}): {tt(s)}")(*code_line(path, start))
    fmt = lambda dct: ", ".join(f"{inline(str(k))} {v}" for k, v in dct.items())
    o = ["\\section{Data, splits and chronology}\\label{app:data}\n",
         "% DRAFT-TEXT: review (connecting prose; every value is generated from the committed files named, mainly "
         "paper2/revision/datasplit_results.json, written by paper2/revision/datasplit.py without training)",
         f"Values below come from {tt('paper2/revision/datasplit.py')} ({dc}; descriptive, no training), which reads the competition files "
         "through the committed data path, and from the committed code lines quoted. The competition files themselves are not redistributed.\n",
         "\\paragraph{(a) The PC anchor.} The anchor is the measured conductivity of a pure-PC electrolyte with the same salt, from the same source "
         "article, at the row's concentration and temperature; it is not neat PC solvent. "
         f"Per row it is looked up by (source article, salt) among the {c['pure_pc_series']} pure-PC series of the training file, then "
         f"interpolated: temperature at the series' grid level within {a['T_TOL']} K, otherwise linearly in $1/T$ between levels "
         f"(anchored rows: {fmt(a['T'])}); concentration linearly in the article's native unit ({fmt(a['c'])}); never extrapolated. "
         f"Code: {q('anchor.py', 'def anchor_value')}; {q('anchor.py', 'T_TOL = ')}. "
         "Anchor information and held-out rows: the pure-PC rows only supply anchors and are in no fold's training or test set. "
         f"For fold A, the {fa['pure_pc_rows_same_sources']} pure-PC rows of fold A's own source article supply its anchors, so a held-out row's anchor "
         "comes from the same article as the row. DEC test anchors use same-salt pure-PC series from a fixed list of six training articles, for mol/kg "
         f"test rows only, averaged where several cover ({q('anchor.py', 'te_series = ')}). The input scaler is a min--max map fitted on the anchored "
         "training rows and the anchored DEC test rows' inputs (no labels).\n",
         f"\\paragraph{{(b) Target.}} $\\log_{{10}}$ of the conductivity $k$ in mS/cm (\\texttt{{log\\_k}} equals $\\log_{{10}} k$ on every training row: "
         f"{t['log10_identity']}). No conductivity is zero or negative (minimum {t['k_min']:.3g} mS/cm); the {t['n_below_0p01']} training rows below "
         f"0.01 mS/cm are dropped ({q('anchor.py', 'tr = tr[tr.k >= 0.01]')}), and submitted predictions are clipped at $\\log_{{10}} k \\geq {t['logk_min_clip']}$.\n",
         "\\paragraph{(c) Composition.} $x_{\\mathrm{co}}$ is one minus the PC mole fraction, from the file's mole-fraction column; the competition file "
         f"converts weight- and volume-basis sources to mole fractions (anchored training rows by source basis: {fmt(cm['frac_basis_anchored'])}). "
         "Mixtures with more than one co-solvent enter through mole-weighted co-solvent descriptors (dielectric constant, log viscosity, molar mass of the "
         f"non-PC part) and mole-weighted mixture descriptors (anchored rows by number of solvents: {fmt(cm['n_solvents_anchored'])}). Molality is native "
         f"for mol/kg sources and $c/\\rho_{{\\mathrm{{mix}}}}$ for mol/L sources, with an ideal-mixing density that neglects the salt "
         f"({q('pipeline.py', '# molality')}); anchored training rows by unit: {fmt(cm['conc_unit_anchored'])}; anchored DEC test rows: "
         f"{fmt(cm['conc_unit_test_anchored'])}, all binary.\n",
         "\\paragraph{(d) Row inclusion.}\n",
         longtable([[inline(k.replace("_", " ")), str(v)] for k, v in c.items()], "R{0.6\\linewidth} r", ["Step (anchor.prepare)", "Rows"]),
         "\\paragraph{(e) Folds.} Fold A holds out the ethyl-acetate mixtures (EA+PC); the ten co-solvent folds each hold out every anchored row "
         "containing that co-solvent, so a row with several co-solvents is in several folds.\n",
         longtable([[inline(k), str(v["n_test"]), str(v["n_train"])] for k, v in d["folds"].items()], "R{0.4\\linewidth} r r", ["Fold", "Test rows", "Training rows"]),
         f"Fold A: {fa['n']} rows; salts {fmt(fa['salts'])}; source articles {fmt(fa['sources'])}; temperature {fa['T_range'][0]}--{fa['T_range'][1]} K "
         f"({fa['n_T_levels']} levels); $x_{{\\mathrm{{co}}}}$ {fa['x_co_range'][0]:.4f}--{fa['x_co_range'][1]:.4f} ({fa['n_x_co']} values); "
         f"concentration {fa['conc_range'][0]:.3f}--{fa['conc_range'][1]:.3f} {inline(list(fa['conc_unit'])[0])}; training rows from the same article: "
         f"{sum(fa['training_rows_from_same_sources'].values())}. "
         + ("\\textbf{Fold A is a single salt from a single source article, so salt, source and co-solvent are confounded in it.}\n"
            if len(fa["salts"]) == 1 and len(fa["sources"]) == 1 else "\n"),
         f"\\paragraph{{(f) DEC.}} The test file has no conductivity column (\\texttt{{k}}: {d['dec']['test_has_k']}, \\texttt{{log\\_k}}: {d['dec']['test_has_log_k']}); "
         f"all {d['dec']['n_test']} test rows contain DEC and no training row does. The labels are held by the challenge and were never available to us; this is "
         f"recorded in the competition's metadata file (\\texttt{{k}}: ``Absent from test.csv'') and in {tt('paper2/revision/appdomain_plan.md')} "
         f"({first_commit('paper2/revision/appdomain_plan.md')}): ``The DEC test labels are hidden''. No DEC number in this paper is an accuracy.\n"]
    # (g) selection chronology
    ko = (REPO / "kan_output.txt").read_text().splitlines()
    eps = [ln.strip() for ln in ko if "best epochs" in ln]
    ln_es = code_line("kan.py", "for s in SEEDS:  # early stop on this fold")
    o.append("\\paragraph{(g) Selection chronology.} The 107 epochs used by every KAN-$\\delta$ model were chosen on held-out folds: in "
             f"{tt('kan.py')} ({first_commit('kan.py')}) each of 5 seeds was early-stopped on fold A and on a second fold (EC+PC) using that fold's own "
             f"held-out rows ({tt('kan.py')}:{ln_es[0]}: {tt(ln_es[1])}), and the median of the 10 best epochs was fixed for refits. The committed log "
             f"({tt('kan_output.txt')}, {first_commit('kan_output.txt')}): " + "; ".join(tt(e) for e in eps) + ". "
             "\\textbf{Fold A was therefore reused during development: it informed the epoch count, and the committed studies below evaluated on it "
             "before the paper's final comparisons.} The list is every committed output or pre-registration that reports on fold A, by the commit that "
             "added it.\n")
    rows = [[r["date"], r["commit"], tt(r["file"]), inline(r["subject"][:110] + ("…" if len(r["subject"]) > 110 else ""))] for r in d["foldA_chronology"]]
    o.append(longtable(rows, "R{0.10\\linewidth} R{0.08\\linewidth} R{0.26\\linewidth} R{0.46\\linewidth}", ["Date", "Commit", "File", "Commit subject"]))
    return "\n".join(o)


# ------------------------------------------------------------------ appendix H: data, splits and chronology (from datasplit_results.json + committed code and logs)
def code_line(path, start):
    """the committed source line that starts with `start` (stripped), with its line number"""
    for i, ln in enumerate((REPO / path).read_text().splitlines(), 1):
        if ln.strip().startswith(start):
            return i, ln.strip()
    raise SystemExit(f"code line not found: {path}: {start}")


def appendix_h():
    DP_ = "paper2/revision/datasplit_results.json"
    d = jload(DP_); c = d["counts"]; a = d["anchor"]; cm = d["composition"]; fa = d["foldA"]; t = d["target"]
    dc = first_commit(DP_)
    tt = lambda s: "\\texttt{" + tex_escape(s).replace("\\_", "\\_\\allowbreak{}").replace("/", "/\\allowbreak{}") + "}"
    q = lambda path, start: (lambda n, s: f"{tt(path)}:{n} ({first_commit(path)}): {tt(s)}")(*code_line(path, start))
    fmt = lambda dct: ", ".join(f"{inline(str(k))} {v}" for k, v in dct.items())
    o = ["\\section{Data, splits and chronology}\\label{app:data}\n",
         "% DRAFT-TEXT: review (connecting prose; every value is generated from the committed files named, mainly "
         "paper2/revision/datasplit_results.json, written by paper2/revision/datasplit.py without training)",
         f"Values below come from {tt('paper2/revision/datasplit.py')} ({dc}; descriptive, no training), which reads the competition files "
         "through the committed data path, and from the committed code lines quoted. The competition files themselves are not redistributed.\n",
         "\\paragraph{(a) The PC anchor.} The anchor is the measured conductivity of a pure-PC electrolyte with the same salt, from the same source "
         "article, at the row's concentration and temperature; it is not neat PC solvent. "
         f"Per row it is looked up by (source article, salt) among the {c['pure_pc_series']} pure-PC series of the training file, then "
         f"interpolated: temperature at the series' grid level within {a['T_TOL']} K, otherwise linearly in $1/T$ between levels "
         f"(anchored rows: {fmt(a['T'])}); concentration linearly in the article's native unit ({fmt(a['c'])}); never extrapolated. "
         f"Code: {q('anchor.py', 'def anchor_value')}; {q('anchor.py', 'T_TOL = ')}. "
         "Anchor information and held-out rows: the pure-PC rows only supply anchors and are in no fold's training or test set. "
         f"For fold A, the {fa['pure_pc_rows_same_sources']} pure-PC rows of fold A's own source article supply its anchors, so a held-out row's anchor "
         "comes from the same article as the row. DEC test anchors use same-salt pure-PC series from a fixed list of six training articles, for mol/kg "
         f"test rows only, averaged where several cover ({q('anchor.py', 'te_series = ')}). The input scaler is a min--max map fitted on the anchored "
         "training rows and the anchored DEC test rows' inputs (no labels).\n",
         f"\\paragraph{{(b) Target.}} $\\log_{{10}}$ of the conductivity $k$ in mS/cm (\\texttt{{log\\_k}} equals $\\log_{{10}} k$ on every training row: "
         f"{t['log10_identity']}). No conductivity is zero or negative (minimum {t['k_min']:.3g} mS/cm); the {t['n_below_0p01']} training rows below "
         f"0.01 mS/cm are dropped ({q('anchor.py', 'tr = tr[tr.k >= 0.01]')}), and submitted predictions are clipped at $\\log_{{10}} k \\geq {t['logk_min_clip']}$.\n",
         "\\paragraph{(c) Composition.} $x_{\\mathrm{co}}$ is one minus the PC mole fraction, from the file's mole-fraction column; the competition file "
         f"converts weight- and volume-basis sources to mole fractions (anchored training rows by source basis: {fmt(cm['frac_basis_anchored'])}). "
         "Mixtures with more than one co-solvent enter through mole-weighted co-solvent descriptors (dielectric constant, log viscosity, molar mass of the "
         f"non-PC part) and mole-weighted mixture descriptors (anchored rows by number of solvents: {fmt(cm['n_solvents_anchored'])}). Molality is native "
         f"for mol/kg sources and $c/\\rho_{{\\mathrm{{mix}}}}$ for mol/L sources, with an ideal-mixing density that neglects the salt "
         f"({q('pipeline.py', '# molality')}); anchored training rows by unit: {fmt(cm['conc_unit_anchored'])}; anchored DEC test rows: "
         f"{fmt(cm['conc_unit_test_anchored'])}, all binary.\n",
         "\\paragraph{(d) Row inclusion.}\n",
         longtable([[inline(k.replace("_", " ")), str(v)] for k, v in c.items()], "R{0.6\\linewidth} r", ["Step (anchor.prepare)", "Rows"]),
         "\\paragraph{(e) Folds.} Fold A holds out the ethyl-acetate mixtures (EA+PC); the ten co-solvent folds each hold out every anchored row "
         "containing that co-solvent, so a row with several co-solvents is in several folds.\n",
         longtable([[inline(k), str(v["n_test"]), str(v["n_train"])] for k, v in d["folds"].items()], "R{0.4\\linewidth} r r", ["Fold", "Test rows", "Training rows"]),
         f"Fold A: {fa['n']} rows; salts {fmt(fa['salts'])}; source articles {fmt(fa['sources'])}; temperature {fa['T_range'][0]}--{fa['T_range'][1]} K "
         f"({fa['n_T_levels']} levels); $x_{{\\mathrm{{co}}}}$ {fa['x_co_range'][0]:.4f}--{fa['x_co_range'][1]:.4f} ({fa['n_x_co']} values); "
         f"concentration {fa['conc_range'][0]:.3f}--{fa['conc_range'][1]:.3f} {inline(list(fa['conc_unit'])[0])}; training rows from the same article: "
         f"{sum(fa['training_rows_from_same_sources'].values())}. "
         + ("\\textbf{Fold A is a single salt from a single source article, so salt, source and co-solvent are confounded in it.}\n"
            if len(fa["salts"]) == 1 and len(fa["sources"]) == 1 else "\n"),
         f"\\paragraph{{(f) DEC.}} The test file has no conductivity column (\\texttt{{k}}: {d['dec']['test_has_k']}, \\texttt{{log\\_k}}: {d['dec']['test_has_log_k']}); "
         f"all {d['dec']['n_test']} test rows contain DEC and no training row does. The labels are held by the challenge and were never available to us; this is "
         f"recorded in the competition's metadata file (\\texttt{{k}}: ``Absent from test.csv'') and in {tt('paper2/revision/appdomain_plan.md')} "
         f"({first_commit('paper2/revision/appdomain_plan.md')}): ``The DEC test labels are hidden''. No DEC number in this paper is an accuracy.\n"]
    # (g) selection chronology
    ko = (REPO / "kan_output.txt").read_text().splitlines()
    eps = [ln.strip() for ln in ko if "best epochs" in ln]
    ln_es = code_line("kan.py", "for s in SEEDS:  # early stop on this fold")
    o.append("\\paragraph{(g) Selection chronology.} The 107 epochs used by every KAN-$\\delta$ model were chosen on held-out folds: in "
             f"{tt('kan.py')} ({first_commit('kan.py')}) each of 5 seeds was early-stopped on fold A and on a second fold (EC+PC) using that fold's own "
             f"held-out rows ({tt('kan.py')}:{ln_es[0]}: {tt(ln_es[1])}), and the median of the 10 best epochs was fixed for refits. The committed log "
             f"({tt('kan_output.txt')}, {first_commit('kan_output.txt')}): " + "; ".join(tt(e) for e in eps) + ". "
             "\\textbf{Fold A was therefore reused during development: it informed the epoch count, and the committed studies below evaluated on it "
             "before the paper's final comparisons.} The list is every committed output or pre-registration that reports on fold A, by the commit that "
             "added it.\n")
    rows = [[r["date"], r["commit"], tt(r["file"]), inline(r["subject"][:110] + ("…" if len(r["subject"]) > 110 else ""))] for r in d["foldA_chronology"]]
    o.append(longtable(rows, "R{0.10\\linewidth} R{0.08\\linewidth} R{0.26\\linewidth} R{0.46\\linewidth}", ["Date", "Commit", "File", "Commit subject"]))
    return "\n".join(o)


# ------------------------------------------------------------------ appendix G: methods and definitions (quoted from the committed files)
GDEFS = [  # (term, path, start of the committed bullet/line, kind: bullet | code)
    ("Edge function (Gaussian RBF + SiLU)", "paper2/synth_prereg.md", "- Same edge type:", "bullet"),
    ("P0 (second differences)", "paper2/synth_prereg.md", "- **P0, \"D2 only\":**", "bullet"),
    ("P1 (adds the SiLU-weight term)", "paper2/synth_prereg.md", "- **P1, \"curvature penalty, Bagrow-style\":**", "bullet"),
    ("P2 (null-space penalty)", "paper2/synth_prereg.md", "- **P2, \"null space closed\":**", "bullet"),
    ("Null-space penalty on real data", "paper2/revision/infimum_prereg.md", "- Loss: the final.py loss + 1e-4", "bullet"),
    ("Polishing", "paper2/synth_prereg.md", "- **\"polished\":**", "bullet"),
    ("Certified (definition)", "paper2/synth_prereg.md", "- **CERTIFIED**", "bullet"),
    ("Certification tolerances (code)", "certify.py", "GTOL, EIGREL, NEWTON, GNULL =", "code"),
    ("Certification criteria (code)", "certify.py", "crit = dict(", "code"),
    ("Near-null directions (code)", "certify.py", "null = np.abs(ev) < EIGREL * lmax", "code"),
    ("Exact symmetries (B-spline null set)", "paper2/revision/bspline_prereg.md", "- The null set is the 15-dimensional constant-shift space", "bullet"),
    ("Null-set-aware certification", "paper2/revision/bspline_prereg.md", "- **Certification of the γ = 0 training objectives**", "bullet"),
    ("Degenerate certified point", "paper2/revision/infimum_prereg.md", "- **DEGENERATE.**", "bullet"),
    ("Probes at degenerate points", "paper2/revision/infimum_prereg.md", "- **Three probes at every degenerate point.**", "bullet"),
    ("PLATEAU / FLAT MINIMUM", "paper2/revision/infimum_prereg.md", "- **PLATEAU** if either of these holds:", "bullet"),
    ("Stalled restart", "paper2/revision/infimum_prereg.md", "- **Stalled restart.**", "bullet"),
    ("Path exponent (I2)", "paper2/revision/infimum_prereg.md", "- **I2.**", "bullet"),
    ("Plateau ratio (I3)", "paper2/revision/infimum_prereg.md", "- **I3.**", "bullet"),
    ("Short-path statistic D", "paper2/revision/dose_prereg.md", "- D = ‖θ*(1e-7)‖² / ‖θ*(1e-5)‖²", "bullet"),
    ("Seed disagreement d_AB", "paper2/synth_prereg.md", "- d_AB between", "bullet"),
    ("Unit match", "paper2/synth_prereg.md", "- **unit match:**", "bullet"),
    ("Cross-edge cancellation ratio", "paper2/synth_prereg.md", "- **Logged at both states:**", "bullet"),
]


def appendix_g():
    tt = lambda s: "\\texttt{" + tex_escape(s).replace("\\_", "\\_\\allowbreak{}").replace("/", "/\\allowbreak{}") + "}"
    rows = []
    for term, path, start, kind in GDEFS:
        if kind == "code":
            n, txt = code_line(path, start)
            body, c = tt(txt), first_commit(path)
            src = f"{tt(path)}:{n}"
        else:
            body, c = inline(grab(path, start)), commit_with(path, start)
            src = tt(path)
        rows.append([inline(term), body, src, c])
    return ("\\section{Methods and definitions}\\label{app:methods}\n"
            "% DRAFT-TEXT: review (one framing sentence; the table quotes the committed definitions)\n"
            "Each term is given in the wording of the file that fixed it, with that file and the commit of the quoted text; code lines are quoted "
            "verbatim. \\emph{Certified} means that all four criteria hold at the point: a numerically certified local minimum under these tolerances, "
            "not a global or unique one. Stop reasons, with solver errors and hangs separated from caps, are tabulated in "
            "\\texttt{paper2/revision/uncertainty\\_output.txt}.\n"
            + longtable(rows, "R{0.16\\linewidth} R{0.50\\linewidth} R{0.16\\linewidth} R{0.08\\linewidth}", ["Term", "Committed definition", "File", "Commit"]))


def anonymise(texts):
    """replace every commit hash of this repository by a neutral label S-NN (chronological order; same hash -> same label)"""
    chrono = git("log", "--reverse", "--format=%h %H").splitlines()
    short = {}
    for ln in chrono:
        h, full = ln.split()
        short[h] = full
    used = []
    pat = re.compile(r"(?<![0-9a-zA-Z])([0-9a-f]{7,40})(?![0-9a-zA-Z])")
    for t in texts.values():
        for m in pat.finditer(t):
            x = m.group(1)
            full = next((f for h, f in short.items() if f.startswith(x)), None)
            if full and full not in used:
                used.append(full)
    order = [f for _, f in [ln.split() for ln in chrono] if f in used]
    label = {f: f"S-{i + 1:02d}" for i, f in enumerate(order)}
    def sub(m):
        x = m.group(1)
        full = next((f for f in label if f.startswith(x)), None)
        return label[full] if full else x
    out = {k: pat.sub(sub, v) for k, v in texts.items()}
    # repository-specific file and archive names -> F-NN / W-NN (order of first appearance; same name -> same label)
    fpat = re.compile(r"(?<![A-Za-z0-9\\/_-])((?:[A-Za-z0-9-]+(?:\\_[A-Za-z0-9-]+)*/)*[A-Za-z0-9-]+(?:\\_[A-Za-z0-9.-]+)*\.(?:tar\.gz|md|py|json|txt|tex|pdf|png|csv|pt|zip|bib|bst|sty))(?![A-Za-z0-9])")
    fmap = {}
    def fsub(m):
        name = m.group(1)
        if name not in fmap:
            kind = "W" if name.endswith(".tar.gz") else "F"
            fmap[name] = f"{kind}-{sum(1 for v in fmap.values() if v[0] == kind) + 1:02d}"
        return fmap[name]
    out = {k: fpat.sub(fsub, v) for k, v in out.items()}
    (TEX / "commitmap.json").write_text(json.dumps({"commits": {label[f]: f for f in order},
                                                     "files": {v: k.replace("\\_", "_") for k, v in fmap.items()}}, indent=1))
    print(f"anonymised {len(fmap)} file/archive names")
    return out, len(order)


def main():
    cap = captions()
    title, body, appf = build_body(cap)
    (TEX / "title.tex").write_text(f"\\title{{{inline(title)}}}\n")
    (TEX / "body.tex").write_text("% GENERATED by build_tex.py from paper2/draft/paper2_draft.md -- do not edit by hand\n" + body)
    apps = {"appendixA": appendix_a(), "appendixB": appendix_b(), "appendixC": appendix_c(), "appendixD": appendix_d(), "appendixE": appendix_e(cap), "appendixH": appendix_h()}
    anon, n = anonymise(apps)
    for k in apps:
        (TEX / f"{k}.tex").write_text("% GENERATED by build_tex.py (preprint: real commit hashes)\n" + apps[k])
        (TEX / f"{k}_anon.tex").write_text("% GENERATED by build_tex.py (anonymous: commit hashes replaced by labels, see commitmap.json)\n" + anon[k])
    print(f"anonymised {n} distinct commit hashes")
    assert not re.search(r"(?<![0-9a-zA-Z])[0-9a-f]{7,40}(?![0-9a-zA-Z])", appf), "Appendix F must carry no commit hashes (one file serves both builds)"
    (TEX / "appendixF.tex").write_text("% GENERATED by build_tex.py from paper2/draft/paper2_draft.md (Appendix F) -- do not edit by hand\n" + appf)
    (TEX / "table_diffs.json").write_text(json.dumps(TABLE_DIFFS, indent=1, ensure_ascii=False))
    print(f"tables generated from committed results; {len(TABLE_DIFFS)} differences from the draft's typed values (table_diffs.json)")
    print("wrote title.tex body.tex appendixA.tex appendixB.tex appendixC.tex appendixD.tex appendixE.tex appendixF.tex")


if __name__ == "__main__":
    main()

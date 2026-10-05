"""paper2/tex/build_tex.py: generate the LaTeX body of paper 2 from COMMITTED sources (no new analysis).

  body.tex       <- paper2/draft/paper2_draft.md (frozen source text), converted faithfully; figures 1-5 inserted
  appendixA.tex  <- committed verdict tables (paper2/synth_prereg.md, electrolyte_case_study_summary.md, arch/sc_prereg.md, arch/ex_prereg.md)
  appendixB.tex  <- post-hoc analyses: files + commit subjects from git
  appendixC.tex  <- reproducibility map from paper2/figures/MANIFEST.md (+ the draft's tables)
  appendixE.tex  <- supplementary figures S1-S3
Captions are drafted from the figure sidecar JSONs and marked "% CAPTION-DRAFT: review".
Run from the repo root: python3 paper2/tex/build_tex.py"""
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
    return git("log", "--diff-filter=A", "--format=%h", "--", path).splitlines()[-1]


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
]
SUP = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")
SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
GREEK = {"φ": r"\varphi", "θ": r"\theta", "μ": r"\mu", "λ": r"\lambda", "α": r"\alpha", "ρ": r"\rho", "β": r"\beta", "η": r"\eta",
         "ε": r"\varepsilon", "δ": r"\delta", "Δ": r"\Delta", "ψ": r"\psi", "σ": r"\sigma", "ℓ": r"\ell"}
UNI = {"≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$", "−": r"$-$", "→": r"$\to$", "≈": r"$\approx$", "·": r"$\cdot$",
       "–": "--", "—": "---", "’": "'", "‘": "`", "“": "``", "”": "''", "‖": r"$\|$", "²": r"$^{2}$", "³": r"$^{3}$",
       "ů": r"\r{u}", "á": r"\'a", "é": r"\'e", "č": r"\v{c}", "ć": r"\'c", "±": r"$\pm$", "…": r"\ldots{}", "✓": r"$\checkmark$"}


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
    s = re.sub(r"([A-Za-zφθμλαρβηεδψσℓ])([₀₁₂₃₄₅₆₇₈₉]+)", lambda m: f"${GREEK.get(m.group(1), m.group(1))}_{{{m.group(2).translate(SUB)}}}$", s)
    for k, v in GREEK.items():
        s = s.replace(k, f"${v}$")
    for k, v in UNI.items():
        s = s.replace(k, v)
    s = re.sub(r"\$\$", "", s)  # merge adjacent inline-math pieces
    s = s.replace(r"$\to$", r"$\to$\allowbreak{}")
    for i, x in enumerate(ph):
        s = s.replace(f"\x00{i}\x00", x)
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


def captions():
    f1, f2, f3, f4, f5 = (side(n) for n in ["fig1_mechanism", "fig2_synthetic", "fig3_dose_response", "fig4_where", "fig5_uniqueness"])
    d = f1["a_deviation_relative_to_range"]; c = f1["c_rms_per_unit_t"]
    cap = {}
    cap["fig:mechanism"] = (
        "fig1_mechanism",
        "Mechanism (computed illustration) for the edge type of Section~\\ref{sec:2} (8 Gaussian RBFs on $[-1, 1]$). "
        f"\\textbf{{(a)}} Edge function produced by a unit linear coefficient pattern and its best straight line; the shaded band is the central 80\\% of the range, "
        f"where the maximum deviation is {pct(d['c80_max'])} of the function's range ({pct(d['full_max'])} over the full range). "
        "\\textbf{(b)} Penalty as a function of a tilt $t$ applied as $+t$ to edge 1 and $-t$ to edge 2: the second-difference penalty (P0) and P1 are exactly zero for every $t$; "
        f"only P2's term $\\mu Q$ grows ($\\lambda = {f1['b_lambda_mu_Ksilu'][0]:g}$, $\\mu = {f1['b_lambda_mu_Ksilu'][1]:g}$). "
        f"\\textbf{{(c)}} RMS change in the unit's pre-activation per unit tilt: {c['correlated']:.4f} on correlated inputs ($\\rho = 0.95$) against {c['independent']:.4f} on independent inputs "
        f"({c['ratio_independent_over_correlated']:.2f}$\\times$).")
    cells_ = [("independent", "P0"), ("independent", "P1"), ("independent", "P2"), ("correlated", "P0"), ("correlated", "P1"), ("correlated", "P2")]
    g = "; ".join(f"{a} {b}: {f2[f'{a}_{b}']['certified']}/8 certified, $\\times${f2[f'{a}_{b}']['median_growth']:.1f}" for a, b in cells_)
    cap["fig:synthetic"] = (
        "fig2_synthetic",
        "Synthetic $2\\times3$ study: $\\|\\theta\\|^2$ over trust-exact polishing iterations (log scale), one line per seed, "
        "coloured certified (blue) or not certified (vermillion). Rows: independent and correlated inputs; columns: P0, P1, P2. "
        f"Per panel, certified seeds and median growth (polished over early-stopped): {g}.")
    b = f3["b_growth"]; cc = f3["c_data_rel_runaway"]; dd = f3["d_foldA"]
    cap["fig:dose"] = (
        "fig3_dose_response",
        "Real-data dose-response in the null-space weight $\\mu$ (6 KAN-delta seeds; $\\mu = 0$ is the runaway reference, stage-2 Part~A). "
        "\\textbf{(a)} Seeds at a numerical minimum / certified: "
        + "; ".join(f"{m}: {f3['a_numerical_minimum'][m]}/{f3['a_certified'][m]}" for m in ['0', '1e-06', '1e-05', '0.0001']) + ". "
        f"\\textbf{{(b)}} $\\|\\theta\\|^2$ relative to the 107-epoch state, median (range): "
        + "; ".join(f"{m}: {b[m]['median']:.1f} ({b[m]['min']:.1f}--{b[m]['max']:.1f})" for m in ['0', '1e-06', '1e-05', '0.0001']) + ". "
        f"\\textbf{{(c)}} Training data term relative to the runaway, median: "
        + "; ".join(f"{m}: {cc[m]['median']:.3f}" for m in ['1e-06', '1e-05', '0.0001']) + ". "
        f"\\textbf{{(d)}} Fold A RMSE: early-stopped {dd['early_stopped']} (dashed line); polished 5-seed ensemble {dd['ensemble']['1e-06']:.3f} at $\\mu = 10^{{-6}}$ "
        f"and {dd['ensemble']['0.0001']:.3f} at $\\mu = 10^{{-4}}$; grey points are single seeds.")
    a = f4["a_mean_shares"]; ks = list(a)
    def sh(g, k): return f"{a[g][k]:.2f}"
    hb = f4["b_max_hidden_corr"]
    cap["fig:where"] = (
        "fig4_where",
        "Where the runaway lives. \\textbf{(a)} Mean share of $\\|\\Delta\\theta\\|^2$ by parameter block. Real data, 107 epochs$\\to$capped: layer-1 affine RBF "
        f"{sh(ks[0], 'layer-1 affine RBF')}, layer-2 affine RBF {sh(ks[0], 'layer-2 affine RBF')}, SiLU weights {sh(ks[0], 'SiLU weights')}; capped$\\to$Part~A: "
        f"{sh(ks[1], 'layer-1 affine RBF')}, {sh(ks[1], 'layer-2 affine RBF')}, {sh(ks[1], 'SiLU weights')}; synthetic correlated P0: layer-2 affine RBF "
        f"{sh(ks[2], 'layer-2 affine RBF')}, SiLU weights {sh(ks[2], 'SiLU weights')}; correlated P1: layer-2 affine RBF {sh(ks[3], 'layer-2 affine RBF')}. "
        "The real-data split was evaluated post hoc from archived weights (Appendix~\\ref{app:repro}). "
        "\\textbf{(b)} Largest absolute correlation between hidden-unit activations on the training rows, median: "
        + "; ".join(f"{k} {v['median']:.3f} ($n={v['n']}$)" for k, v in hb.items()) + ". Blue: certified; vermillion: not certified or runaway.")
    nn = f5["a_n_null"]; db = f5["b_dAB"]
    nnk = list(nn); dbk = list(db)
    def rng(v): return f"{min(v)}--{max(v)}"
    cap["fig:uniqueness"] = (
        "fig5_uniqueness",
        "Convergence is not uniqueness. \\textbf{(a)} Near-null Hessian eigenvalues ($|\\lambda| < 10^{-8}\\lambda_{\\max}$) at the end of polishing: "
        f"KAN-delta baseline on fold A, median {nn[nnk[0]]['median']:.0f} (range {rng(nn[nnk[0]]['values'])}); null-space penalty $\\mu = 10^{{-4}}$ on all rows, "
        f"median {nn[nnk[1]]['median']:.0f} (range {rng(nn[nnk[1]]['values'])}); SC-KAN-delta on fold A, {rng(nn[nnk[2]]['values'])}. "
        "\\textbf{The baseline counts are taken at non-stationary points}: none of the 10 baseline models was certified, and all stopped at the 500-iteration cap. "
        "\\textbf{(b)} Seed disagreement $d_{AB}$ (median, interquartile range): "
        + "; ".join(f"{k.replace('*', '$^*$')} {v['median']:.3f}" for k, v in db.items()) + ". "
        "Groups marked $^*$ (polished fold A) are not in any committed result: they were evaluated post hoc from the archived polished weights "
        "(SHA-verified; predictions on the 1434 fold A training rows), without any re-optimisation.")
    cap["fig:S1"] = ("figS1_lossdecomp", supp_caption("figS1_lossdecomp"))
    cap["fig:S2"] = ("figS2_gaugecheck", supp_caption("figS2_gaugecheck"))
    cap["fig:S3"] = ("figS3_loco", supp_caption("figS3_loco"))
    return cap


def supp_caption(name):
    v = side(name)
    if name == "figS1_lossdecomp":
        p = v["pooled_shares"]
        return ("Post hoc loss decomposition (synthetic, early$\\to$polished): share of the total-loss decrease from each term, pooled over the 8 seeds of each cell. "
                + "; ".join(f"{k}: data {d['data']:.2f}, D2 {d['D2']:.2f} (median total change {100 * d['median_total_change_rel']:+.1f}\\%)" for k, d in p.items()) + ".")
    if name == "figS2_gaugecheck":
        m = v["muQ_share"]
        return ("Post hoc gauge check at the $\\mu = 10^{-4}$ certified minima: share of the curvature along each least-squares gauge direction contributed by $\\mu Q$. Medians: "
                + "; ".join(f"{k.replace('_', ' ')} {d['median']:.2f} ($n = {d['n']}$)" for k, d in m.items()) + ".")
    if name == "figS3_loco":
        m = v["means"]
        return ("Leave-one-co-solvent-out RMSE per fold (5-seed ensembles, early state) for the KAN-delta baseline, SC-KAN-delta and EX-KAN. Unweighted means: "
                + "; ".join(f"{k.replace('δ', '$\\delta$')} {d['unweighted']:.4f}" for k, d in m.items()) + ". The EA fold uses the fold A rows.")


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
]
FIG_AFTER = {  # place the figure after the paragraph that starts with
    "fig:mechanism": "**Step 2: a direction the data cannot see.**",
    "fig:synthetic": "- **Independent inputs are not fully safe.**",
    "fig:where": "**Where the runaway lives.**",
    "fig:dose": "Under a second-order certificate",
    "fig:uniqueness": "**Convergence is not uniqueness.**",
}
TABLE_CAPTIONS = [  # CAPTION-DRAFT: short titles only; the draft's own notes stay in the text
    ("tab:synthetic", "Synthetic $2\\times3$ study, per cell."),
    ("tab:dose", "Dose-response in $\\mu$ on real data (seeds)."),
    ("tab:alternatives", "Structured alternatives against the early-stopped baseline."),
]


def refs(s):
    s = re.sub("\x01([^|]+)\\|([^\x01]*)\x01", lambda m: f"(Figure~\\ref{{{m.group(1)}}}{m.group(2)})", s)
    return re.sub("\x02([^|]+)\\|([^\x02]*)\x02", lambda m: f"Figure~\\ref{{{m.group(1)}}}{m.group(2)}", s)


def table_tex(rows, label, caption):
    head, body = rows[0], rows[2:]
    n = len(head)
    spec = "l" + "l" * (n - 1) if n <= 4 else "l" + "p{0.2\\linewidth}" * (n - 1)
    if label == "tab:alternatives":
        spec = "l p{0.24\\linewidth} l l p{0.16\\linewidth}"
    if label == "tab:dose":
        spec = "l l p{0.18\\linewidth} l p{0.16\\linewidth}"
    out = [f"\\begin{{table}}[t]\n  \\centering\n  % CAPTION-DRAFT: review\n  \\caption{{{caption}}}\n  \\label{{{label}}}\n  \\small\n  \\begin{{tabular}}{{{spec}}}\n  \\toprule"]
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
        if done:
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
            out.append(table_tex(blk, lab, c) + "\n")
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
    body = "\\begin{abstract}\n" + body if not body.startswith("\\begin{abstract}") else body
    assert placed == set(FIG_AFTER), placed
    return title, body


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


def appendix_a():
    o = ["\\section{All pre-registered predictions}\\label{app:prereg}\n",
         "Generated by \\texttt{paper2/tex/build\\_tex.py} from the committed verdict sections; the cell text is the committed wording. "
         "Commits are given as pre-registration $\\to$ results (in the anonymous version, as neutral labels S-NN). Thresholds are part of each prediction's wording.\n"]
    # synthetic
    p = "paper2/synth_prereg.md"; txt = (REPO / p).read_text()
    vt = [t for t in md_tables(txt.split("## Verdicts")[1]) if t[0].startswith("| Prediction")][0]
    pre, res = first_commit(p), commit_with(p, "**X2** certified")
    rows = []
    for r in vt[2:]:
        c = cells(r)
        mid = re.match(r"\*\*(X\d)\*\* (.*)", c[0])
        rows.append([inline(mid.group(1)), inline(mid.group(2)), inline(c[2]), inline(c[1]), f"{pre}$\\to$\\allowbreak {res}"])
    o.append("\\subsection{Synthetic study (\\texttt{paper2/synth\\_prereg.md})}\n")
    o.append(longtable(rows, "R{0.05\\linewidth} R{0.29\\linewidth} R{0.36\\linewidth} R{0.09\\linewidth} R{0.11\\linewidth}", ["ID", "Prediction (threshold)", "Result", "Verdict", "Commits"]))
    # real data: case-study summary tables
    p = "electrolyte_case_study_summary.md"; txt = (REPO / p).read_text()
    tabs = md_tables(txt)
    rows = []
    t1 = [t for t in tabs if t[0].startswith("| Study | Prediction")][0]
    study = ""
    for r in t1[2:]:
        c = cells(r)
        study = c[0] or study
        rows.append([inline(study), inline(c[1]), inline(c[3]), inline(c[2]), inline(c[4].replace(" → ", "→"))])
    rule = re.search(r"\*\*0 columns were removed and the gate stopped;\*\* (I1–I5) are inconclusive\s*\(prereg (\w+) → results (\w+)\)", txt)
    rows.append([inline("Rule (`rule_prereg.md`)"), inline(rule.group(1) + " (gate)"), inline("0 columns removed; gate stopped"), "inconclusive",
                 f"{rule.group(2)}$\\to$\\allowbreak {rule.group(3)}"])
    rk = [t for t in tabs if t[0].startswith("| Prediction | Arm G")][0]
    rc = f"{first_commit('rank_prereg.md')}$\\to$\\allowbreak {first_commit('rank_results.json')}"
    for r in rk[2:]:
        c = cells(r)
        rows.append([inline("Rank (`rank_prereg.md`)"), inline(c[0]), inline("arm G: " + c[1] + ("; arm F: " + c[2] if c[2] else "")), "see result", rc])
    ts = [t for t in tabs if t[0].startswith("| Prediction | Verdict | Key number |")][0]
    for r in ts[2:]:
        c = cells(r)
        which = "trust_prereg.md" if c[0].startswith("T") else "stage2_prereg.md"
        res_file = "trust_results.json" if which.startswith("trust") else "stage2_results.json"
        rows.append([inline(f"`{which}`"), inline(c[0]), inline(c[2]), inline(c[1]), f"{first_commit(which)}$\\to$\\allowbreak {first_commit(res_file)}"])
    cv = [t for t in tabs if t[0].startswith("| Prediction | Verdict | Key number | Prereg")][0]
    for r in cv[2:]:
        c = cells(r)
        which = "`close_prereg.md`" if c[0].startswith("V") else "`certify_prereg.md`"
        rows.append([inline(which), inline(c[0]), inline(c[2]), inline(c[1]), inline(c[3].replace(" → ", "→"))])
    o.append("\\subsection{Real-data case study (\\texttt{electrolyte\\_case\\_study\\_summary.md})}\n")
    o.append("Rows are as committed in the case-study summary (one row may cover several predictions, e.g.\\ F2--F5). "
             "The rank intervention's verdicts are given per arm in the result column.\n")
    o.append(longtable(rows, "R{0.14\\linewidth} R{0.26\\linewidth} R{0.29\\linewidth} R{0.10\\linewidth} R{0.11\\linewidth}",
                       ["Study", "Prediction (threshold)", "Result", "Verdict", "Commits"]))
    tally = [t for t in tabs if t[0].startswith("| Study | Held")][0]
    trow = [[inline(x) for x in cells(r)] for r in tally[2:]]
    o.append("\\noindent Committed tally of the case study:\n")
    o.append(longtable(trow, "R{0.55\\linewidth} r r r", ["Study", "Held", "Failed", "Inconclusive"]))
    # structured alternatives (arch)
    for p, nm in [("arch/sc_prereg.md", "Shared-curve KAN (\\texttt{arch/sc\\_prereg.md})"), ("arch/ex_prereg.md", "Two-anchor excess model (\\texttt{arch/ex\\_prereg.md})")]:
        txt = (REPO / p).read_text()
        vt = [t for t in md_tables(txt.split("## Verdicts")[-1]) if t[0].startswith("| | Prediction")][0]
        key = cells(vt[2])[0]
        res = commit_with(p, f"| {key} |")
        rows = [[inline(c[0]), inline(c[1]), inline(c[3]), inline(c[2].replace("*", "").lower()), f"{first_commit(p)}$\\to$\\allowbreak {res}"] for c in map(cells, vt[2:])]
        o.append(f"\\subsection{{{nm}}}\n")
        o.append("Run after the case-study close-out; not part of the 65 predictions above.\n")
        o.append(longtable(rows, "R{0.05\\linewidth} R{0.26\\linewidth} R{0.39\\linewidth} R{0.09\\linewidth} R{0.11\\linewidth}", ["ID", "Prediction (threshold)", "Result", "Verdict", "Commits"]))
    return "\n".join(o)


# ------------------------------------------------------------------ appendix B: post-hoc analyses (generated from files + git)
POSTHOC = [  # (name, files, how located)
    ("Anatomy of the real-data runaway", ["anatomy_plan.md", "anatomy.py", "anatomy_output.txt"]),
    ("Gauge check and regularised teacher (exploratory)", ["gaugecheck.py", "gaugecheck_notes.md", "gaugecheck_output.txt"]),
    ("Anatomy of the synthetic runaway", ["paper2/anatomy_synth.py", "paper2/anatomy_synth_output.txt"]),
    ("Loss decomposition along the runaway", ["paper2/lossdecomp.py", "paper2/lossdecomp_output.txt"]),
    ("Descriptive checks (hidden-unit correlation, tilt affinity, references)", ["paper2/checks/plan.md", "paper2/checks/checks.py", "paper2/checks/checks_output.txt"]),
]


def appendix_b():
    o = ["\\section{Post-hoc analyses}\\label{app:posthoc}\n",
         "Every analysis below was defined after seeing results. Each is labelled \\textbf{post hoc} in its own files, changes no pre-registered verdict, "
         "and is reported as exploratory. The description is the subject line of the commit that added the analysis.\n",
         "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\\begin{longtable}{R{0.28\\linewidth} R{0.52\\linewidth} R{0.10\\linewidth}}\n\\toprule\nAnalysis & Commit subject (as committed) & Commit \\\\\n\\midrule\n\\endhead"]
    for name, files in POSTHOC:
        c = first_commit(files[-1] if "plan" not in files[0] else files[0])
        subj = git("log", "-1", "--format=%s", c)
        o.append(f"\\textbf{{Post hoc.}} {inline(name)} \\newline \\footnotesize{{\\texttt{{{tex_escape(', '.join(files))}}}}} & {inline(subj)} & {c} \\\\")
        if "plan" in files[0] and len(files) > 1:
            c2 = first_commit(files[-1]); o.append(f" & {inline(git('log', '-1', '--format=%s', c2))} & {c2} \\\\")
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
        lab = {"fig1": "fig:mechanism", "fig2": "fig:synthetic", "fig3": "fig:dose", "fig4": "fig:where", "fig5": "fig:uniqueness",
               "figS1": "fig:S1", "figS2": "fig:S2", "figS3": "fig:S3"}[c[0].split("_")[0]]
        rows.append([f"Figure~\\ref{{{lab}}} \\figname{{{tex_escape(c[0]).replace(chr(92) + "_", chr(92) + "_" + chr(92) + "allowbreak{}")}}}", inline(c[1]), inline(c[2]), inline(c[3])])
    o = ["\\section{Reproducibility map}\\label{app:repro}\n",
         "Generated from \\texttt{paper2/figures/MANIFEST.md}. Every figure is rebuilt by its script from committed results; "
         "each figure's sidecar JSON lists the data files (path, commit, SHA-256), any archived weights (SHA-256, archive) and every plotted number. "
         "\\REPOSTATEMENT\n",
         longtable(rows, "R{0.15\\linewidth} R{0.13\\linewidth} R{0.42\\linewidth} R{0.19\\linewidth}", ["Figure", "Script", "Data files (commit)", "Archived weights"])]
    tabs = [("Table~\\ref{tab:synthetic}", "typed in the draft from", "\\texttt{paper2/synth\\_results.json}, \\texttt{paper2/synth\\_output.txt}", first_commit("paper2/synth_results.json")),
            ("Table~\\ref{tab:dose}", "typed in the draft from", "\\texttt{close\\_results.json}, \\texttt{certify\\_results.json}, \\texttt{stage2\\_results.json}",
             ", ".join(first_commit(f) for f in ["close_results.json", "certify_results.json", "stage2_results.json"])),
            ("Table~\\ref{tab:alternatives}", "typed in the draft from", "\\texttt{arch/sc\\_results.json}, \\texttt{arch/ex\\_results.json}, \\texttt{paper2/figures/fig5\\_uniqueness.json}",
             ", ".join(first_commit(f) for f in ["arch/sc_results.json", "arch/ex_results.json", "paper2/figures/fig5_uniqueness.json"]))]
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
         "% DRAFT-TEXT: review (new prose, not in the source draft; it states only the protocol followed, no results)",
         "\\paragraph{Pre-registration.} Every study committed a pre-registration file (predictions, thresholds, interpretation rules) "
         "before the script that tested it existed. Verdicts were appended below the committed text; the pre-registered text and recorded verdicts were never edited.",
         "\\paragraph{Amendments.} A change to a design after its pre-registration was committed as a dated amendment section before any affected model was trained, "
         "with its reason; the original text was kept. Choices forced by the code or data at run time were logged as numbered discrepancies at the top of each output log.",
         "\\paragraph{Files win.} Instructions to the agent stated that committed files take precedence over any number or description in a prompt; "
         "discrepancies between a brief and the files were flagged rather than resolved silently.",
         "\\paragraph{Data and weights.} Model weights were never committed: they were archived outside the repository with a manifest of SHA-256 hashes and producing commits, "
         "and every later analysis verified the hashes and held its inputs in memory. Post-hoc analyses were labelled as such in their files and changed no verdict.",
         "\\paragraph{AI coding agent.} Code, runs and write-ups were produced by an AI coding agent working under this protocol, with the human author setting the briefs, "
         "approving amendments and deciding each next stage. Crashes were recorded in the logs and fixed in separate commits labelled as implementation-only.\n",
         "\\noindent Pre-registrations, amendments and crash records (from the commit history):\n"]
    rows = []
    for ln in git("log", "--reverse", "--format=%h|%ad|%s", "--date=short").splitlines():
        h, d, subj = ln.split("|", 2)
        low = subj.lower()
        if any(k in low for k in ["pre-register", "amendment", "crash", "implementation only", "as first run"]):
            kind = "amendment" if "amendment" in low else "crash record / fix" if any(k in low for k in ["crash", "implementation only", "as first run"]) else "pre-registration"
            rows.append([h, d, kind, inline(subj)])
    o.append(longtable(rows, "R{0.08\\linewidth} R{0.10\\linewidth} R{0.14\\linewidth} R{0.58\\linewidth}", ["Commit", "Date", "Kind", "Commit subject"]))
    return "\n".join(o)


def appendix_e(cap):
    o = ["\\section{Supplementary figures}\\label{app:supp}\n"]
    for lab in ["fig:S1", "fig:S2", "fig:S3"]:
        name, c = cap[lab]
        o.append(figure(lab, name, c, width="0.5\\linewidth" if lab == "fig:S2" else "\\linewidth").replace("[t]", "[h]"))
    return "\n".join(o)


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
    title, body = build_body(cap)
    (TEX / "title.tex").write_text(f"\\title{{{inline(title)}}}\n")
    (TEX / "body.tex").write_text("% GENERATED by build_tex.py from paper2/draft/paper2_draft.md -- do not edit by hand\n" + body)
    apps = {"appendixA": appendix_a(), "appendixB": appendix_b(), "appendixC": appendix_c(), "appendixD": appendix_d(), "appendixE": appendix_e(cap)}
    anon, n = anonymise(apps)
    for k in apps:
        (TEX / f"{k}.tex").write_text("% GENERATED by build_tex.py (preprint: real commit hashes)\n" + apps[k])
        (TEX / f"{k}_anon.tex").write_text("% GENERATED by build_tex.py (anonymous: commit hashes replaced by labels, see commitmap.json)\n" + anon[k])
    print(f"anonymised {n} distinct commit hashes")
    print("wrote title.tex body.tex appendixA.tex appendixB.tex appendixC.tex appendixE.tex")


if __name__ == "__main__":
    main()

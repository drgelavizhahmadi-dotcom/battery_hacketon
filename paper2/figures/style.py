"""paper2/figures/style: shared style, data registry and sidecar writer for every paper-2 figure.
Figures are built from COMMITTED results only (no training, no re-optimisation). Where a panel needs a
deterministic evaluation of archived weights, the weights are SHA-verified against weights_manifest.txt and
held in memory, and the sidecar says so."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "paper2", REPO / "arch"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FULL, HALF = 6.5, 3.25
# Okabe-Ito colour-blind-safe palette; one fixed colour per condition in every figure
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2",
          vermillion="#D55E00", purple="#CC79A7", grey="#999999")
C = dict(runaway=OI["vermillion"], certified=OI["blue"], early=OI["grey"],
         P0=OI["orange"], P1=OI["sky"], P2=OI["green"],
         independent=OI["black"], correlated=OI["purple"],
         baseline=OI["black"], sc=OI["purple"], ex=OI["orange"], certify=OI["blue"])
BLOCK_C = {"layer-1 affine RBF": OI["vermillion"], "layer-1 curved RBF": OI["orange"], "layer-2 affine RBF": OI["blue"],
           "layer-2 curved RBF": OI["sky"], "SiLU weights": OI["green"], "biases": OI["grey"], "salt embedding": OI["purple"]}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 100,
})


def letter(ax, s, dx=-0.16, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def git_commit(path):
    return subprocess.run(["git", "log", "-1", "--format=%h", "--", str(path)], cwd=REPO, capture_output=True, text=True).stdout.strip()


class Sidecar:
    def __init__(self, name):
        self.name, self.files, self.weights, self.values, self.notes = name, [], {}, {}, []

    def json(self, rel):
        p = REPO / rel
        data = p.read_bytes()
        st = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=REPO, capture_output=True, text=True).stdout.strip()
        if st:
            raise SystemExit(f"REFUSED: {rel} has uncommitted changes (figures use committed results only)")
        self.files.append(dict(path=rel, commit=git_commit(rel), sha256=hashlib.sha256(data).hexdigest()))
        return json.loads(data)

    def text(self, rel):
        p = REPO / rel
        self.files.append(dict(path=rel, commit=git_commit(rel), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        return p.read_text()

    def load_weights(self, names):
        from stage2 import extract_verified, in_memory
        blobs = in_memory(extract_verified(names))
        man = {l.split()[0].split("/")[-1]: l.split() for l in (REPO / "weights_manifest.txt").read_text().splitlines() if l.startswith("gauge_weights/")}
        for n in names:
            m = man[n]
            self.weights[n] = dict(sha256=m[1], archive=m[4] if len(m) > 4 else "gauge_weights_2026-09-29.tar.gz")
        self.files.append(dict(path="weights_manifest.txt", commit=git_commit("weights_manifest.txt"),
                               sha256=hashlib.sha256((REPO / "weights_manifest.txt").read_bytes()).hexdigest()))
        return blobs

    def put(self, key, value):
        self.values[key] = value

    def note(self, s):
        self.notes.append(s)

    def save(self, fig):
        for ext, kw in [("pdf", {}), ("png", dict(dpi=300))]:
            fig.savefig(HERE / f"{self.name}.{ext}", **kw)
        plt.close(fig)
        (HERE / f"{self.name}.json").write_text(json.dumps(dict(figure=self.name, data_files=self.files, weights=self.weights,
                                                                notes=self.notes, values=self.values), indent=1, default=_conv))
        print(f"wrote {self.name}.pdf/.png/.json")


def _conv(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))

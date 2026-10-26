"""paper2/tex/make_review_archive.py: build (do not upload) the anonymised review archive and scan it.

Contents: every tracked file except the preprint build, commitmap.json and the git machinery, plus the weight archives
(repacked with owner names stripped from the tar headers; member bytes unchanged, so the manifest SHA-256 values hold).
Text files are anonymised: author name, account, e-mail, absolute paths, remote URLs and the repository name are replaced,
and every commit hash is replaced by a neutral label (the anonymous PDF's S-NN labels where they exist, then S-NN continued
in chronological order). The competition data are not included (not redistributed); README_REVIEW.md says how to obtain them.
The scan decompresses every member (and every member of the nested tar archives) and searches for the same identifiers.
Usage (repo root): /usr/bin/python3 paper2/tex/make_review_archive.py <output.zip>"""
import gzip
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

TEX = Path(__file__).resolve().parent
REPO = TEX.parent.parent
WEIGHTS = REPO.parent / "battery_hacketon_weights"
EXCLUDE = {"paper2/tex/preprint.tex", "paper2/tex/preprint.pdf", "paper2/tex/commitmap.json"}
EXCLUDE_PREFIX = ("paper2/tex/submission_tmlr/",)
TERMS = ["ahmadi", "gelavizh", "drgelavizhahmadi", "battery_hacketon", "hacketon", "github.com", "berlin", "orcid", "/users/",
         "/private/tmp", "claude.ai/", "@gmail", "dr.gelavizh"]
SUBS = [  # (pattern, replacement), applied in order to every text member
    (r"/Users/gelavizhahmadi/Projects/battery_hacketon_weights", "<WEIGHTS_DIR>"),
    (r"/Users/gelavizhahmadi/Projects/battery_hacketon", "<REPO>"),
    (r"/Users/gelavizhahmadi[^\s'\"]*", "<HOME>"), (r"/private/tmp/[^\s'\"]*", "<TMP>"),
    (r"https?://github\.com/[^\s)'\"]*", "<repository URL removed>"), (r"github\.com/[^\s)'\"]*", "<repository URL removed>"),
    (r"https?://claude\.ai/[^\s)'\"]*", "<link removed>"),
    (r"dr\.gelavizh\.ahmadi@gmail\.com", "anonymous@example.org"), (r"[A-Za-z0-9._%+-]+@gmail\.com", "anonymous@example.org"),
    (r"drgelavizhahmadi-dotcom", "anonymous"), (r"(?i)@?gelavizh\s+ahmadi", "Anonymous"), (r"(?i)gelavizh", "anonymous"), (r"(?i)ahmadi", "anonymous"),
    (r"battery_hacketon", "repo"), (r"(?i)hacketon", "repo"), (r"(?i)\bberlin\b", "<city>"), (r"(?i)orcid[^\s]*", "<id removed>"),
]


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout


def labels():
    chrono = [ln.split() for ln in git("log", "--reverse", "--format=%H %h").splitlines()]
    cm = json.loads((TEX / "commitmap.json").read_text())["commits"]
    lab = {full: l for l, full in cm.items()}
    k = max(int(l.split("-")[1]) for l in cm)
    for full, _ in chrono:
        if full not in lab:
            k += 1; lab[full] = f"S-{k:02d}"
    return lab, [f for f, _ in chrono]


def anonymise_text(s, lab, fulls):
    for pat, rep in SUBS:
        s = re.sub(pat, rep, s)
    def h(m):
        x = m.group(1)
        if len(x) > 40:
            return x
        cand = [f for f in fulls if f.startswith(x)]
        return lab[cand[0]] if len(cand) == 1 else x
    return re.sub(r"(?<![0-9a-zA-Z])([0-9a-f]{7,64})(?![0-9a-zA-Z])", h, s)


def is_text(b):
    if b"\x00" in b[:8192]:
        return False
    try:
        b.decode("utf-8"); return True
    except UnicodeDecodeError:
        return False


def repack_tar(path, lab, fulls, changed):
    """same members, owner names and ids stripped; text members anonymised (their manifest SHA-256 then no longer matches,
    and each is listed in weights/ANONYMISED_MEMBERS.txt); binary members (weights) byte-identical"""
    out = io.BytesIO()
    with tarfile.open(path) as src, tarfile.open(fileobj=out, mode="w:gz") as dst:
        for m in src.getmembers():
            data = src.extractfile(m).read() if m.isfile() else None
            if data is not None and is_text(data):
                new = anonymise_text(data.decode("utf-8"), lab, fulls).encode("utf-8")
                if new != data:
                    changed.append((f"{path.name}:{m.name}", hashlib.sha256(data).hexdigest()))
                    data = new; m.size = len(data)
            m.uid = m.gid = 0; m.uname = m.gname = ""; m.mtime = 0
            dst.addfile(m, io.BytesIO(data) if data is not None else None)
    return out.getvalue()


def scan(zpath, hashes):
    hits = []
    def check(name, b):
        if not is_text(b) and not name.endswith((".pdf", ".json", ".md", ".txt", ".py")):
            t = b.decode("latin-1").lower()
        else:
            t = b.decode("utf-8", errors="replace").lower()
        if name.endswith(".pdf"):
            for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", b, re.S):
                try:
                    t += zlib_text(m.group(1))
                except Exception:
                    pass
        for w in TERMS:
            if w in t:
                hits.append(f"{name}: '{w}'")
        for hh in hashes:
            if re.search(r"(?<![0-9a-f])" + hh + r"(?![0-9a-f])", t):
                hits.append(f"{name}: commit hash {hh}")
    with zipfile.ZipFile(zpath) as z:
        n = 0
        for info in z.infolist():
            b = z.read(info); n += 1
            check(info.filename, b)
            if info.filename.endswith(".tar.gz"):
                with tarfile.open(fileobj=io.BytesIO(b)) as tf:
                    for m in tf.getmembers():
                        if m.uname or m.gname:
                            hits.append(f"{info.filename}:{m.name}: tar owner '{m.uname}/{m.gname}'")
                        if m.isfile():
                            n += 1; check(f"{info.filename}:{m.name}", tf.extractfile(m).read())
    return n, hits


def zlib_text(b):
    import zlib
    return zlib.decompress(b).decode("latin-1").lower()


def main(out):
    out = Path(out).resolve()
    assert REPO not in out.parents, "write the archive outside the repository (it contains weights)"
    lab, fulls = labels()
    files = [f for f in git("ls-files").splitlines() if f not in EXCLUDE and not f.startswith(EXCLUDE_PREFIX)]
    manifest, n_text = [], 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            b = (REPO / f).read_bytes()
            if is_text(b):
                b = anonymise_text(b.decode("utf-8"), lab, fulls).encode("utf-8"); n_text += 1
            name = anonymise_text(f, lab, fulls)
            z.writestr(f"code/{name}", b); manifest.append((f"code/{name}", len(b)))
        changed = []
        for a in sorted(WEIGHTS.glob("*.tar.gz")):
            b = repack_tar(a, lab, fulls, changed)
            z.writestr(f"weights/{a.name}", b); manifest.append((f"weights/{a.name}", len(b)))
        z.writestr("weights/ANONYMISED_MEMBERS.txt", "# text members of the weight archives anonymised for review (original SHA-256 as in weights_manifest.txt)\n"
                   + "".join(f"{n}\t{h}\n" for n, h in changed))
        print(f"anonymised {len(changed)} text member(s) inside the weight archives: " + ", ".join(n for n, _ in changed))
        readme = ("# Review archive (anonymised)\n\n"
                  "- `code/`: the repository at the reviewed state (tracked files; commit hashes replaced by neutral labels S-NN, the anonymous\n"
                  "  PDF's labels where they appear there; author, paths and URLs removed).\n"
                  "- `weights/`: the archived model weights. Place the `.tar.gz` files in a directory next to the code named `repo_weights`\n"
                  "  (trust.py: ARCH = ROOT.parent / \"repo_weights\" after anonymisation); every member is verified against\n"
                  "  `code/weights_manifest.txt` (SHA-256) before use. Tar headers were rewritten without owner names; member bytes are unchanged.\n"
                  "- Competition data (not redistributed): download the CALiSol-23 challenge files and unpack them into\n"
                  "  `code/ca-li-sol-23-challenge/` (train.csv, test.csv, sample_submission.csv, metaData.csv, solvent_properties.csv).\n"
                  "- Python: the pinned interpreter of `code/paper2/tex/README.md` (Python 3.9.6; numpy 1.26.4, matplotlib 3.9.4, torch 2.8.0,\n"
                  "  scipy 1.13.1, pandas 2.3.3); `code/requirements.txt`.\n"
                  "- Figures: `code/paper2/figures/make_*.py` (see `code/paper2/figures/MANIFEST.md`); manuscript: `code/paper2/tex/` (`make`).\n")
        z.writestr("README_REVIEW.md", readme); manifest.append(("README_REVIEW.md", len(readme)))
        z.writestr("MANIFEST.txt", "\n".join(f"{n}\t{s}" for n, s in manifest) + "\n")
    hashes = sorted(set(git("log", "--format=%h").split()) | set(git("log", "--format=%H").split()))
    n, hits = scan(out, hashes)
    size = out.stat().st_size
    print(f"wrote {out} ({size / 1e6:.1f} MB): {len(files)} code files ({n_text} text, anonymised), "
          f"{len(list(WEIGHTS.glob('*.tar.gz')))} weight archives; sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
    print(f"scan: {n} members (including tar members) -> " + ("OK" if not hits else f"{len(hits)} FINDINGS"))
    for h in hits[:60]:
        print("  " + h)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

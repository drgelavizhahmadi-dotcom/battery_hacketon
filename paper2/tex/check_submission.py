"""paper2/tex/check_submission.py: verify that the TMLR submission folder contains only anonymised material.
Fails (exit 1) if the folder holds commitmap.json, the preprint, or any file whose text, metadata or decompressed
PDF streams contain the author's name, the repository owner/name/URL, an affiliation/ORCID marker, any commit hash
of this repository, or any real repository file/archive name listed in commitmap.json."""
import json
import re
import subprocess
import sys
import zlib
from pathlib import Path

TEX = Path(__file__).resolve().parent
REPO = TEX.parent.parent
ALLOWED = {"anonymous.pdf", "code_anonymised.zip", "MANIFEST.txt"}   # code archive: to be added later


def contents(p):
    raw = p.read_bytes()
    parts = [raw]
    if p.suffix == ".pdf":
        for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.S):
            try:
                parts.append(zlib.decompress(m.group(1)))
            except Exception:
                pass
        parts.append(subprocess.run(["pdftotext", str(p), "-"], capture_output=True).stdout)
        parts.append(subprocess.run(["pdfinfo", str(p)], capture_output=True).stdout)
    return b"\n".join(parts).decode("latin-1").lower()


def main(folder):
    folder = Path(folder)
    cmap = json.loads((TEX / "commitmap.json").read_text())
    hashes = subprocess.run(["git", "log", "--format=%h %H"], cwd=REPO, capture_output=True, text=True).stdout.split()
    terms = ["ahmadi", "gelavizh", "drgelavizhahmadi", "battery_hacketon", "hacketon", "github.com/drg", "berlin", "orcid"]
    files = sorted({v for v in cmap["files"].values()} | {Path(v).name for v in cmap["files"].values()}, key=len, reverse=True)
    bad, n = [], 0
    for p in sorted(folder.rglob("*")):
        if p.is_dir():
            continue
        n += 1
        if p.name not in ALLOWED or p.name == "commitmap.json" or "preprint" in p.name:
            bad.append(f"{p.name}: not an allowed submission file")
            continue
        t = contents(p)
        bad += [f"{p.name}: contains '{w}'" for w in terms if w in t]
        bad += [f"{p.name}: contains commit hash {h}" for h in hashes if re.search(r"(?<![0-9a-f])" + h + r"(?![0-9a-f])", t)]
        bad += [f"{p.name}: contains repository file name '{f}'" for f in files if len(Path(f).name) > 6 and Path(f).name.lower() in t]
    print(f"checked {n} file(s) in {folder}: " + ("OK" if not bad and n else "FAILED"))
    for b in bad:
        print("  " + b)
    sys.exit(1 if bad or not n else 0)


if __name__ == "__main__":
    main(sys.argv[1])

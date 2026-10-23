"""Tests for the staleness watchdog of paper2/revision/bspline.py (note to supplement 4b3f20a, 5bf5acd)."""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "paper2" / "revision"), str(ROOT / "paper2"), str(ROOT / "arch"), str(ROOT), str(ROOT / "tests")]
import bspline as BP


def child(code):
    return subprocess.Popen([sys.executable, "-c", code])


def test_does_not_fire_on_a_progressing_child():
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "ck.npz"
        pr = child(f"import time, pathlib\nfor i in range(16):\n    pathlib.Path({str(ck)!r}).write_text(str(i)); time.sleep(0.25)\n")
        t0 = time.monotonic()
        why = BP.supervise(pr, ck, cap_s=60, stale_s=2.0, poll=0.1)
        assert why is None and pr.returncode == 0 and time.monotonic() - t0 >= 3.5


def test_fires_on_a_stalled_child():
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "ck.npz"
        pr = child(f"import time, pathlib\npathlib.Path({str(ck)!r}).write_text('0'); time.sleep(60)\n")
        t0 = time.monotonic()
        why = BP.supervise(pr, ck, cap_s=60, stale_s=2.0, poll=0.1)
        assert why == "stale" and pr.returncode is not None and time.monotonic() - t0 < 10


def test_fires_on_a_child_that_never_checkpoints():
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "ck.npz"
        pr = child("import time; time.sleep(60)")
        why = BP.supervise(pr, ck, cap_s=60, stale_s=2.0, poll=0.1)
        assert why == "stale"


def test_cap_still_applies():
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "ck.npz"
        pr = child(f"import time, pathlib\nwhile True:\n    pathlib.Path({str(ck)!r}).write_text('x'); time.sleep(0.2)\n")
        why = BP.supervise(pr, ck, cap_s=2.0, stale_s=30, poll=0.1)
        assert why == "cap"

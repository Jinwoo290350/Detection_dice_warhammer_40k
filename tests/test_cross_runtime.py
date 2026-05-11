"""Cross-runtime sanity test: PyTorch CPU vs OpenVINO CPU.

For each video clip we run the full pipeline twice (once per runtime) and
compare snapshot-by-snapshot. We tolerate small differences (FP16 vs FP32
quantization can flip 1 die's classification on borderline cases) but flag
anything large.

Run from project root:
    python tests/test_cross_runtime.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PY = PROJECT_ROOT / ".venv" / "bin" / "python"
COUNT = PROJECT_ROOT / "src" / "count_dice.py"
DET_PT = PROJECT_ROOT / "models" / "detector.pt"
CLS_PT = PROJECT_ROOT / "models" / "classifier.pt"
DET_OV = PROJECT_ROOT / "models" / "detector_openvino_model"
CLS_OV = PROJECT_ROOT / "models" / "classifier_openvino_model"

CLIPS = [
    "Data/2546.mp4",
    "Data/2547.mp4",
    "Data/2548.mp4",
    "Data/WIN_25690506_20_41_32_Pro.mp4",
]

TOLERANCE_DICE = 1   # max acceptable |dice diff| per snapshot
TOLERANCE_PIPS = 5   # max acceptable |pip diff| per snapshot


def run_pipeline(video: str, det: Path, cls: Path, out: Path) -> dict:
    cmd = [
        str(PY), str(COUNT),
        "--video", video,
        "--output", str(out),
        "--detector", str(det),
        "--classifier", str(cls),
    ]
    print(f"  running with {det.name}...")
    subprocess.run(cmd, check=True, capture_output=True)
    return json.loads(out.read_text())


def compare(pt: dict, ov: dict, label: str) -> int:
    """Returns number of snapshot-pair failures."""
    pt_snaps = pt["snapshots"]
    ov_snaps = ov["snapshots"]
    if len(pt_snaps) != len(ov_snaps):
        print(f"  ⚠️ {label}: snapshot count mismatch — pt={len(pt_snaps)} ov={len(ov_snaps)}")
        return abs(len(pt_snaps) - len(ov_snaps))

    fails = 0
    for i, (a, b) in enumerate(zip(pt_snaps, ov_snaps)):
        dd = abs(a["total_dice"] - b["total_dice"])
        dp = abs(a["total_pips"] - b["total_pips"])
        ok = dd <= TOLERANCE_DICE and dp <= TOLERANCE_PIPS
        marker = "✓" if ok else "✗"
        print(
            f"  [{i}] frame {a['frame_idx']:>5}  pt={a['total_dice']}d/{a['total_pips']}p"
            f"  ov={b['total_dice']}d/{b['total_pips']}p  Δd={dd} Δp={dp}  {marker}"
        )
        if not ok:
            fails += 1
    return fails


def main() -> None:
    tmp = PROJECT_ROOT / "tests" / "_tmp"
    tmp.mkdir(exist_ok=True)
    total_fails = 0
    for video in CLIPS:
        name = Path(video).stem
        print(f"\n=== {name} ===")
        pt = run_pipeline(video, DET_PT, CLS_PT, tmp / f"{name}_pt.json")
        ov = run_pipeline(video, DET_OV, CLS_OV, tmp / f"{name}_ov.json")
        fails = compare(pt, ov, name)
        total_fails += fails

    print()
    if total_fails == 0:
        print("✅ all snapshots within tolerance")
        sys.exit(0)
    else:
        print(f"⚠️  {total_fails} snapshots over tolerance "
              f"(>{TOLERANCE_DICE} dice or >{TOLERANCE_PIPS} pips diff)")
        sys.exit(1)


if __name__ == "__main__":
    main()

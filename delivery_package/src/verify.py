"""Standalone verification script for the dice counter.

Run on the client's hardware to confirm the pipeline produces the same
results across machines. Output is a JSON report that can be diffed against
the Mac dev run.

Self-contained: only depends on `openvino`, `opencv-python`, `numpy`,
`ultralytics` (which provides high-level model loading and post-processing).

Usage (from delivery_package/):
    python verify.py --video your_clip.mp4 --output verify.json

Send the resulting JSON back; we'll compare it against the Mac reference.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DETECTOR = SCRIPT_DIR.parent / "models" / "detector_openvino_model"
DEFAULT_CLASSIFIER = SCRIPT_DIR.parent / "models" / "classifier_openvino_model"

# Try sibling-folder first (delivery_package layout)
if not DEFAULT_DETECTOR.exists():
    DEFAULT_DETECTOR = SCRIPT_DIR / "models" / "detector_openvino_model"
    DEFAULT_CLASSIFIER = SCRIPT_DIR / "models" / "classifier_openvino_model"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--diff-threshold", type=float, default=2.5)
    parser.add_argument("--stable-seconds", type=float, default=0.5)
    parser.add_argument("--motion-threshold", type=float, default=4.0)
    args = parser.parse_args()

    # Late imports so the script's --help works without the heavy deps installed
    from ultralytics import YOLO

    sys.path.insert(0, str(SCRIPT_DIR))
    from stability import StabilityDetector
    from pipeline import STAT_KEYS, compute_dice_stats, empty_stats

    # Inline DiceCounter (kept here so verify.py is more self-contained)
    detector = YOLO(str(args.detector), task="detect")
    classifier = YOLO(str(args.classifier), task="classify")
    cls_names = classifier.names

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stability = StabilityDetector(
        fps=fps,
        diff_threshold=args.diff_threshold,
        stable_seconds=args.stable_seconds,
        motion_threshold=args.motion_threshold,
    )

    snapshots: list[dict] = []
    timing = {"detect_ms": [], "classify_ms": []}
    frame_idx = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if not stability.feed(frame):
            continue

        # Detect
        t0 = time.perf_counter()
        det = detector.predict(frame, imgsz=640, verbose=False)[0]
        t1 = time.perf_counter()
        timing["detect_ms"].append((t1 - t0) * 1000)

        if len(det.boxes) == 0:
            snapshots.append({
                "frame_idx": frame_idx,
                "timestamp": frame_idx / fps,
                **empty_stats(),
                "dice": [],
            })
            continue

        # Crop + classify
        h, w = frame.shape[:2]
        dice = []
        cls_t0 = time.perf_counter()
        for box, conf in zip(det.boxes.xyxy.cpu().numpy(), det.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            pad = 0.08 * max(bw, bh)
            xa = max(0, int(x1 - pad))
            ya = max(0, int(y1 - pad))
            xb = min(w, int(x2 + pad))
            yb = min(h, int(y2 + pad))
            if xb - xa < 8 or yb - ya < 8:
                continue
            crop = frame[ya:yb, xa:xb]
            r = classifier.predict(crop, imgsz=64, verbose=False)[0]
            pip = int(cls_names[int(r.probs.top1)])
            dice.append({
                "bbox": [int(xa), int(ya), int(xb), int(yb)],
                "pip": pip,
                "det_conf": float(conf),
                "cls_conf": float(r.probs.top1conf),
            })
        timing["classify_ms"].append((time.perf_counter() - cls_t0) * 1000)

        stats = compute_dice_stats(dice)
        snapshots.append({
            "frame_idx": frame_idx,
            "timestamp": frame_idx / fps,
            **stats,
            "dice": dice,
        })
        print(f"  snapshot @ frame {frame_idx} (t={frame_idx/fps:.2f}s): "
              f"count={stats['dice_count']}  sum={stats['total_dice_sum']}  "
              f"D3 sum={stats['D3_total_dice_sum']}")

    cap.release()

    avg = lambda xs: round(sum(xs) / len(xs), 2) if xs else 0.0  # noqa: E731
    totals: dict[str, int] = {k: sum(s[k] for s in snapshots) for k in STAT_KEYS}
    totals["snapshot_count"] = len(snapshots)

    args.output.write_text(json.dumps({
        "video": str(args.video),
        "fps": fps,
        "resolution": [width, height],
        "frame_count": frame_idx + 1,
        "totals": totals,
        "snapshots": snapshots,
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "timing_ms": {
            "detect_avg": avg(timing["detect_ms"]),
            "detect_total_ms": round(sum(timing["detect_ms"]), 1),
            "classify_avg_per_snapshot": avg(timing["classify_ms"]),
            "classify_total_ms": round(sum(timing["classify_ms"]), 1),
        },
    }, indent=2))
    print(f"\nwrote {args.output} ({len(snapshots)} snapshots)")
    print(f"avg detect: {avg(timing['detect_ms'])} ms")
    print(f"avg classify per snapshot: {avg(timing['classify_ms'])} ms")


if __name__ == "__main__":
    main()

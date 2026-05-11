"""CLI: count dice in a video using the snapshot pipeline.

Streams the video, waits for the dice to settle (StabilityDetector), runs
the two-stage detector + classifier (DiceCounter) once per stable window,
writes a JSON summary, and optionally renders an annotated MP4 with bboxes
and pip values.

Usage:
    python src/count_dice.py --video Data/2546.mp4 --output result_2546.json
    python src/count_dice.py --video Data/2548.mp4 --output result.json --save-overlay overlay.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from pipeline import DiceCounter, STAT_KEYS, render_overlay
from stability import StabilityDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DETECTOR = PROJECT_ROOT / "models" / "detector_openvino_model"
DEFAULT_CLASSIFIER = PROJECT_ROOT / "models" / "classifier_openvino_model"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON path")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--save-overlay", type=Path, help="optional annotated MP4")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--diff-threshold", type=float, default=2.5)
    parser.add_argument("--stable-seconds", type=float, default=0.5)
    parser.add_argument("--motion-threshold", type=float, default=4.0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    counter = DiceCounter(args.detector, args.classifier, det_conf=args.det_conf)
    stability = StabilityDetector(
        fps=fps,
        diff_threshold=args.diff_threshold,
        stable_seconds=args.stable_seconds,
        motion_threshold=args.motion_threshold,
    )

    overlay_writer: cv2.VideoWriter | None = None
    if args.save_overlay:
        overlay_writer = cv2.VideoWriter(
            str(args.save_overlay),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    snapshots: list[dict] = []
    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        is_stable = stability.feed(frame)
        if is_stable:
            result = counter.count(frame)
            entry = {
                "frame_idx": frame_idx,
                "timestamp": frame_idx / fps,
                **result,
            }
            snapshots.append(entry)
            print(
                f"  snapshot @ frame {frame_idx} (t={frame_idx/fps:.2f}s): "
                f"count={result['dice_count']}  sum={result['total_dice_sum']}  "
                f"D3 sum={result['D3_total_dice_sum']}"
            )

        if overlay_writer is not None:
            # render every frame with the most recent count, if any
            last_result = snapshots[-1] if snapshots else None
            if last_result and is_stable:
                overlay = render_overlay(frame, last_result)
            else:
                overlay = frame
            overlay_writer.write(overlay)

    cap.release()
    if overlay_writer is not None:
        overlay_writer.release()

    # Aggregate totals across all snapshots — stats are linear, just sum each key
    totals: dict[str, int] = {k: sum(s[k] for s in snapshots) for k in STAT_KEYS}
    totals["snapshot_count"] = len(snapshots)

    args.output.write_text(json.dumps({
        "video": str(args.video),
        "fps": fps,
        "frame_count": frame_idx + 1,
        "totals": totals,
        "snapshots": snapshots,
    }, indent=2))

    print(f"\nwrote {args.output}")
    print("--- summary across all snapshots ---")
    print(f"  snapshots         : {totals['snapshot_count']}")
    print(f"  dice_count        : {totals['dice_count']}")
    print(f"  total_dice_sum    : {totals['total_dice_sum']}")
    print(f"  D3_total_dice_sum : {totals['D3_total_dice_sum']}")
    print(f"  per-face counts   : "
          + "  ".join(f"{f}: {totals[f'dice_{f}_count']}" for f in range(1, 7)))
    print(f"  per-face N+ count : "
          + "  ".join(f"{f}+: {totals[f'dice_{f}_plus_count']}" for f in range(1, 7)))


if __name__ == "__main__":
    main()

"""Extract stable (non-rolling) frames from dice videos.

Detects "stable" windows by checking when consecutive frame difference stays
below a threshold for a duration measured in **seconds** (not frame count) so
the logic stays correct on variable-frame-rate footage like screen recordings.

One representative frame per stable window is written out — duplicates within
the same window are skipped.

Usage:
    python src/extract_stable_frames.py --video Data/2548.mp4
    python src/extract_stable_frames.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Texture mapping per CLAUDE.md Data Layout.
TEXTURE_MAP: dict[str, str] = {
    "2546.mp4": "marble",
    "2547.mp4": "plain",
    "2548.mp4": "mixed",
    "WIN_25690506_20_41_32_Pro.mp4": "plain",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
FRAMES_DIR = PROJECT_ROOT / "frames"


def extract_stable_frames(
    video_path: Path,
    output_dir: Path,
    *,
    diff_threshold: float = 2.0,
    stable_seconds: float = 1.0,
    motion_threshold: float = 4.0,
    downscale_for_diff: int = 320,
) -> int:
    """Walk through `video_path` and dump one frame per stable window.

    State machine: emit at most one frame per "rolling → resting" cycle. After
    emitting, require a frame whose mean abs-diff exceeds `motion_threshold`
    (i.e. the dice were picked up or rolled again) before re-arming.

    Args:
        diff_threshold: max mean abs-diff (0-255) between consecutive grayscale
            frames to count as "still". Tune based on noise level of source.
        stable_seconds: how long the scene must stay still before we emit.
        motion_threshold: mean abs-diff above which we consider the scene
            actively moving — used to re-arm after an emit. Should be > diff_threshold.
        downscale_for_diff: width to downscale to for diff computation; speed
            optimization, doesn't affect saved frames.

    Returns:
        Number of stable frames written.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    prev_small: np.ndarray | None = None
    stable_run_sec = 0.0
    armed = True  # may emit when next stable window completes
    emitted = 0
    last_frame: np.ndarray | None = None
    last_frame_idx = -1

    dt = 1.0 / fps

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        h, w = frame.shape[:2]
        scale = downscale_for_diff / w
        small = cv2.resize(frame, (downscale_for_diff, int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if prev_small is None:
            prev_small = gray
            last_frame = frame
            last_frame_idx = frame_idx
            continue

        diff = float(cv2.absdiff(gray, prev_small).mean())
        prev_small = gray

        if diff < diff_threshold:
            stable_run_sec += dt
            last_frame = frame
            last_frame_idx = frame_idx
            if armed and stable_run_sec >= stable_seconds and last_frame is not None:
                out_path = output_dir / f"{stem}_f{last_frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), last_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                emitted += 1
                armed = False
                stable_run_sec = 0.0
        else:
            stable_run_sec = 0.0
            if diff >= motion_threshold:
                armed = True  # real motion observed → may emit on next stable window

    cap.release()
    return emitted


def resolve_output_dir(video_path: Path) -> Path:
    texture = TEXTURE_MAP.get(video_path.name)
    if texture is None:
        raise ValueError(
            f"unknown video {video_path.name!r}; add it to TEXTURE_MAP in this file"
        )
    return FRAMES_DIR / texture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="single video file under Data/")
    parser.add_argument("--all", action="store_true", help="run on every video in TEXTURE_MAP")
    parser.add_argument("--diff-threshold", type=float, default=2.0)
    parser.add_argument("--stable-seconds", type=float, default=1.0)
    parser.add_argument("--motion-threshold", type=float, default=4.0)
    args = parser.parse_args()

    if not args.video and not args.all:
        parser.error("pass --video PATH or --all")

    targets: list[Path]
    if args.all:
        targets = [DATA_DIR / name for name in TEXTURE_MAP]
    else:
        v: Path = args.video
        targets = [v if v.is_absolute() else DATA_DIR / v.name]

    total = 0
    for video in targets:
        if not video.exists():
            print(f"skip (missing): {video}")
            continue
        out = resolve_output_dir(video)
        n = extract_stable_frames(
            video,
            out,
            diff_threshold=args.diff_threshold,
            stable_seconds=args.stable_seconds,
            motion_threshold=args.motion_threshold,
        )
        print(f"{video.name:<40} -> {out.relative_to(PROJECT_ROOT)}/  {n} frames")
        total += n
    print(f"total: {total} frames")


if __name__ == "__main__":
    main()

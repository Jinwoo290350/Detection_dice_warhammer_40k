"""Sample one frame every N seconds from a video.

Complements `extract_stable_frames.py`. Stable extraction emits one snapshot
per "rolling → resting" cycle, which gives a small set when the source video
has only a handful of throws. Periodic sampling fills in the gap so we have
a richer unlabeled pool for the auto-label workflow in Phase 2.

Output filenames are tagged `_p` so they don't collide with stable extracts.

Usage:
    python src/sample_periodic_frames.py --video Data/WIN_25690506_20_41_32_Pro.mp4 --interval 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from extract_stable_frames import TEXTURE_MAP, FRAMES_DIR, DATA_DIR, PROJECT_ROOT


def sample_periodic(
    video_path: Path,
    output_dir: Path,
    *,
    interval_seconds: float = 5.0,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    step = max(1, int(round(fps * interval_seconds)))

    emitted = 0
    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % step != 0:
            continue
        out_path = output_dir / f"{stem}_p{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        emitted += 1

    cap.release()
    return emitted


def resolve_output_dir(video_path: Path) -> Path:
    texture = TEXTURE_MAP.get(video_path.name)
    if texture is None:
        raise ValueError(f"unknown video {video_path.name!r}")
    return FRAMES_DIR / texture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between samples")
    args = parser.parse_args()

    video: Path = args.video if args.video.is_absolute() else DATA_DIR / args.video.name
    if not video.exists():
        raise SystemExit(f"missing: {video}")

    out = resolve_output_dir(video)
    n = sample_periodic(video, out, interval_seconds=args.interval)
    print(f"{video.name:<40} -> {out.relative_to(PROJECT_ROOT)}/  {n} frames")


if __name__ == "__main__":
    main()

"""Auto-classify crops in `classifier_dataset/_unsorted/` using N3VERS4YDIE
pretrained detector (6 per-pip classes), to bootstrap the manual review.

The pretrained model detects pip faces directly. We run it on each crop, and
if it detects a die in the crop with confidence above threshold, we move the
crop into the corresponding face folder (1/.../6/). Crops with no detection
or with multiple disagreeing detections stay in `_unsorted/` for the user to
sort manually.

The user then opens each face folder and removes any mis-classified crops back
to `_unsorted/` (faster than starting from scratch — N3VERS4YDIE is reasonably
accurate on individual dice).

Usage:
    python src/auto_classify_crops.py
    python src/auto_classify_crops.py --conf 0.5 --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "pretrained" / "n3v_best.pt"
DEFAULT_DATASET = PROJECT_ROOT / "classifier_dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    unsorted_dir = args.dataset_dir / "_unsorted"
    crops = sorted(unsorted_dir.glob("*.jpg"))
    if not crops:
        raise SystemExit(f"no crops in {unsorted_dir}")

    print(f"loading model {args.model}")
    model = YOLO(str(args.model))
    # Pretrained classes: 0..5 = dice_1..dice_6
    print(f"running on {len(crops)} crops (conf>={args.conf})")

    moved: Counter[int] = Counter()
    skipped: Counter[str] = Counter()

    for i in tqdm(range(0, len(crops), args.batch)):
        batch = crops[i : i + args.batch]
        results = model.predict(
            [str(p) for p in batch],
            conf=args.conf,
            verbose=False,
        )
        for path, r in zip(batch, results):
            if len(r.boxes) == 0:
                skipped["no_detection"] += 1
                continue
            # Use the highest-confidence detection (face value 1..6 = class+1)
            best_idx = int(r.boxes.conf.argmax().item())
            cls0 = int(r.boxes.cls[best_idx].item())
            face = cls0 + 1
            if not 1 <= face <= 6:
                skipped["unexpected_class"] += 1
                continue
            moved[face] += 1
            if not args.dry_run:
                dst = args.dataset_dir / str(face) / path.name
                dst.parent.mkdir(exist_ok=True)
                shutil.move(str(path), str(dst))

    print()
    print(f"{'face':<6} {'count':>8}")
    for face in range(1, 7):
        print(f"  {face:<4} {moved[face]:>8}")
    print(f"  total moved : {sum(moved.values())}")
    print(f"  left unsorted:")
    for reason, n in skipped.items():
        print(f"    {reason}: {n}")


if __name__ == "__main__":
    main()

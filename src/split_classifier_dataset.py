"""Split `classifier_dataset/{1..6}/` into train/val/test for Ultralytics
classification training. Output goes to `classifier_split/{train,val,test}/{1..6}/`.

Original folders are left intact (we copy, not move).

Usage:
    python src/split_classifier_dataset.py
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "classifier_dataset"
DEFAULT_OUTPUT = PROJECT_ROOT / "classifier_split"
FACES = ("1", "2", "3", "4", "5", "6")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.output.exists():
        shutil.rmtree(args.output)

    counts = {"train": 0, "val": 0, "test": 0}

    for face in FACES:
        src_dir = args.source / face
        files = sorted(src_dir.glob("*.jpg"))
        rng.shuffle(files)

        n = len(files)
        n_train = int(round(n * args.train))
        n_val = int(round(n * args.val))
        # Whatever's left goes to test
        train_files = files[:n_train]
        val_files = files[n_train : n_train + n_val]
        test_files = files[n_train + n_val :]

        for split, group in (
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ):
            dst_dir = args.output / split / face
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src in group:
                shutil.copy2(src, dst_dir / src.name)
            counts[split] += len(group)
            print(f"  {split}/{face}: {len(group)}")

    print()
    print(f"total → train: {counts['train']}, val: {counts['val']}, test: {counts['test']}")
    print(f"output: {args.output.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()

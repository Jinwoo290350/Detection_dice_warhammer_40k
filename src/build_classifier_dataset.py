"""Build per-pip-face classifier dataset by cropping bboxes from a YOLO dataset.

Input: a YOLO-format detector dataset (the one we annotate in Roboflow), e.g.
    dataset/
        images/{train,valid,test}/*.jpg
        labels/{train,valid,test}/*.txt

Each label line is: `class_id cx cy w h` in normalized coords. We crop each
bbox (with light padding) and dump the crops into:
    classifier_dataset/_unsorted/<image_stem>_b<idx>.jpg

The user then visually sorts the crops into folders `1/` ... `6/` by face value
(use Finder Quick Look with arrow keys → drag into folder; small batch). After
sorting, `_unsorted/` should be empty (or contain rejects).

Usage:
    python src/build_classifier_dataset.py
    python src/build_classifier_dataset.py --dataset-dir path/to/dataset --padding 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT = PROJECT_ROOT / "classifier_dataset"
SPLITS = ("train", "valid", "val", "test")


def crop_bboxes(
    dataset_dir: Path,
    output_dir: Path,
    *,
    padding: float = 0.08,
) -> int:
    unsorted_dir = output_dir / "_unsorted"
    unsorted_dir.mkdir(parents=True, exist_ok=True)
    # also pre-create face folders so the user has an obvious target
    for face in range(1, 7):
        (output_dir / str(face)).mkdir(exist_ok=True)

    emitted = 0

    def split_dirs(split: str) -> tuple[Path, Path] | None:
        # Roboflow YOLOv8 export layout: dataset/{split}/{images,labels}/
        # Alternative layout: dataset/{images,labels}/{split}/
        candidates = [
            (dataset_dir / split / "images", dataset_dir / split / "labels"),
            (dataset_dir / "images" / split, dataset_dir / "labels" / split),
        ]
        for img_dir, lbl_dir in candidates:
            if img_dir.exists() and lbl_dir.exists():
                return img_dir, lbl_dir
        return None

    found_any = False
    for split in SPLITS:
        dirs = split_dirs(split)
        if dirs is None:
            continue
        found_any = True
        images_dir, labels_dir = dirs
        for img_path in sorted(images_dir.glob("*.jpg")):
            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            with label_path.open() as fh:
                for idx, raw in enumerate(fh):
                    parts = raw.strip().split()
                    if len(parts) < 5:
                        continue
                    coords = [float(x) for x in parts[1:]]
                    if len(coords) == 4:
                        # YOLO detection: cx cy w h (normalized)
                        cx, cy, bw_n, bh_n = coords
                        x1n, y1n = cx - bw_n / 2, cy - bh_n / 2
                        x2n, y2n = cx + bw_n / 2, cy + bh_n / 2
                    elif len(coords) >= 6 and len(coords) % 2 == 0:
                        # YOLO segmentation polygon: x1 y1 x2 y2 ...
                        xs = coords[0::2]
                        ys = coords[1::2]
                        x1n, y1n, x2n, y2n = min(xs), min(ys), max(xs), max(ys)
                    else:
                        continue
                    bw_px = (x2n - x1n) * w
                    bh_px = (y2n - y1n) * h
                    pad = padding * max(bw_px, bh_px)
                    x1 = max(0, int(x1n * w - pad))
                    y1 = max(0, int(y1n * h - pad))
                    x2 = min(w, int(x2n * w + pad))
                    y2 = min(h, int(y2n * h + pad))
                    if x2 - x1 < 8 or y2 - y1 < 8:  # tiny / degenerate bbox
                        continue
                    crop = img[y1:y2, x1:x2]
                    out_path = unsorted_dir / f"{img_path.stem}_b{idx:02d}.jpg"
                    cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    emitted += 1

    if not found_any:
        raise SystemExit(
            f"no recognized YOLO layout found under {dataset_dir} — "
            "expected either {split}/images,labels/ (Roboflow) or images,labels/{split}/"
        )
    return emitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--padding", type=float, default=0.08,
        help="extra margin around bbox as a fraction of the larger side (default 0.08)",
    )
    args = parser.parse_args()

    n = crop_bboxes(args.dataset_dir, args.output_dir, padding=args.padding)
    unsorted = args.output_dir / "_unsorted"
    print(f"wrote {n} crops to {unsorted.relative_to(PROJECT_ROOT)}/")
    print("next: open the folder in Finder, Quick Look each crop, drag into 1/.../6/")


if __name__ == "__main__":
    main()

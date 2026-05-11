"""Kaggle training script for the dice face-value classifier (Phase 3 step 2).

Pairs with `02_train_detector_kaggle.py`. The detector finds dice positions
(class `die`); this classifier reads the pip face value (1..6) on each crop.

To use:
  1. Upload `classifier_split.zip` (built by `src/split_classifier_dataset.py`)
     as a Kaggle Dataset, e.g. `frank290350/warhammer-40k-dice-classifier`
  2. New Kaggle Notebook → enable GPU (T4 x2 or P100) → Add Data → your dataset
  3. Paste each `# %% cell` block below into a notebook cell
  4. Run all → download `/kaggle/working/classifier.pt` when done

Architecture: `yolov8n-cls.pt` — 1.4M params, ImageNet-pretrained, exports
cleanly to OpenVINO IR. Small enough for Intel NPU inference.
"""

# %% cell 1 — install
# In a Kaggle notebook cell, use:
#     !pip install -q ultralytics==8.4.47
import subprocess
subprocess.run(["pip", "install", "-q", "ultralytics==8.4.47"], check=True)


# %% cell 2 — train
import glob
from pathlib import Path
import shutil

from ultralytics import YOLO

# Auto-locate split dataset (Kaggle mount path varies)
_train_dirs = glob.glob("/kaggle/input/**/train", recursive=True)
assert _train_dirs, "no train/ folder under /kaggle/input/ — is the dataset attached?"
DATASET_ROOT = Path(_train_dirs[0]).parent
print("dataset root:", DATASET_ROOT)
# Sanity: should contain train/, val/, test/ each with subfolders 1..6
for split in ("train", "val", "test"):
    for face in range(1, 7):
        d = DATASET_ROOT / split / str(face)
        assert d.exists(), f"missing: {d}"

PROJECT_DIR = "/kaggle/working/runs"
RUN_NAME = "classifier-yolov8n-cls"
WEIGHTS_OUT = Path("/kaggle/working/classifier.pt")

# yolov8n-cls is small (~3M params) and exports cleanly to OpenVINO
model = YOLO("yolov8n-cls.pt")

results = model.train(
    data=str(DATASET_ROOT),
    epochs=50,
    imgsz=64,                # small input — crops are tiny anyway
    batch=128,
    patience=15,
    save_period=10,
    project=PROJECT_DIR,
    name=RUN_NAME,
    pretrained=True,
    seed=42,
    # Augmentation — moderate since we already have ~1700 crops
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4,
    degrees=15,              # dice can land at any rotation
    translate=0.1,
    scale=0.3,
    fliplr=0.5,
    flipud=0.0,              # vertical flip would corrupt the face value
)

best_src = next(Path(PROJECT_DIR).rglob("best.pt"))
shutil.copy(best_src, WEIGHTS_OUT)
print(f"saved: {WEIGHTS_OUT}")


# %% cell 3 — eval on test split
metrics = model.val(data=str(DATASET_ROOT), split="test")
print(f"top1 accuracy: {metrics.top1:.4f}")
print(f"top5 accuracy: {metrics.top5:.4f}")


# %% cell 4 — confusion matrix on test (helpful for spotting which faces confuse each other)
from collections import Counter
import os

test_root = DATASET_ROOT / "test"
confusion: dict[tuple[int, int], int] = {}
for face_dir in sorted(test_root.iterdir()):
    if not face_dir.is_dir():
        continue
    true_face = int(face_dir.name)
    paths = sorted(face_dir.glob("*.jpg"))
    if not paths:
        continue
    preds = model.predict([str(p) for p in paths], verbose=False)
    for r in preds:
        pred_face = int(r.probs.top1) + 1  # classes 0..5 → faces 1..6
        confusion[(true_face, pred_face)] = confusion.get((true_face, pred_face), 0) + 1

print("\nConfusion matrix (rows=true, cols=pred):")
print("       " + "  ".join(f"{p:>4}" for p in range(1, 7)))
for t in range(1, 7):
    row = [confusion.get((t, p), 0) for p in range(1, 7)]
    print(f"  {t}:   " + "  ".join(f"{v:>4}" for v in row))

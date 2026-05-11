"""Kaggle training script for the YOLO detector (Phase 3).

This is a flat .py file so it stays diff-friendly. To use:

  1. Upload `dataset_for_kaggle.zip` (built locally — contains the YOLO dataset
     from Roboflow plus `pretrained/n3v_best.pt` for warm-start) as a private
     Kaggle Dataset, e.g. `frankhupu/warhammer-dice-detector`
  2. New Kaggle Notebook → enable GPU (T4 x2 or P100)
     Add Data → your dataset
  3. Replace KAGGLE_SLUG below with your slug
  4. Paste the cells below (split per `# %% cell` marker) into notebook cells
  5. Run all → download `/kaggle/working/best.pt` when done

Resume-friendly: `save_period=25` writes a checkpoint every 25 epochs so a
9-hour Kaggle session timeout doesn't lose progress; rerun with `resume=True`
on the latest checkpoint to continue.

Init weights: we warm-start from N3VERS4YDIE/dice-recognition `best.pt` (YOLOv8n
trained on Roboflow Universe `rune-zkmvl/rune55`, 7239 dice images, 6 per-pip
classes). Ultralytics will auto-reset the detection head from 6→1 classes when
fine-tuning against our `data.yaml`. Backbone features (die-shape detection)
transfer; head learns our single-class scenario from scratch.
"""

# %% cell 1 — install (some Kaggle GPU images don't have ultralytics)
# In a Kaggle notebook cell, use the line magic instead:
#     !pip install -q ultralytics==8.4.47
import subprocess
subprocess.run(["pip", "install", "-q", "ultralytics==8.4.47"], check=True)


# %% cell 2 — train
import glob
from pathlib import Path
import shutil

from ultralytics import YOLO  # noqa: E402

# Auto-locate the dataset under /kaggle/input/ — Kaggle's mount path varies
# (sometimes `/kaggle/input/<slug>/`, sometimes `/kaggle/input/datasets/<user>/<slug>/`).
# Find data.yaml directly so we don't hardcode the prefix.
_data_yamls = glob.glob("/kaggle/input/**/data.yaml", recursive=True)
assert _data_yamls, "data.yaml not found under /kaggle/input/ — is the dataset attached?"
DATASET_ROOT = Path(_data_yamls[0]).parent
print("dataset root:", DATASET_ROOT)

DATA_YAML = str(DATASET_ROOT / "data.yaml")
INIT_WEIGHTS = str(DATASET_ROOT / "pretrained" / "n3v_best.pt")
PROJECT_DIR = "/kaggle/working/runs"
RUN_NAME = "baseline-yolov8n"
WEIGHTS_OUT = Path("/kaggle/working/best.pt")

# Warm-start from N3VERS4YDIE pretrained dice detector (head auto-reset to 1 class)
# Fallback to plain yolov8n.pt if you want a clean COCO baseline for comparison.
model = YOLO(INIT_WEIGHTS)

results = model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    batch=16,
    patience=30,             # early stop if val mAP plateaus
    save_period=25,          # checkpoint every 25 epochs (resume-friendly)
    project=PROJECT_DIR,
    name=RUN_NAME,
    pretrained=True,
    seed=42,
    # --- augmentation tuned for occlusion + texture variety, light on motion ---
    mosaic=0.5,              # mosaic helps with crowded-scene generalization
    mixup=0.1,               # very light mixup
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    shear=2,
    flipud=0.0,              # never vertical-flip — would corrupt pip orientation
    fliplr=0.5,              # horizontal flip is safe for D6
    # we run inference on static snapshots so we don't need heavy motion blur
)

# Copy best.pt to a known path for download
best_src = next(Path(PROJECT_DIR).rglob("best.pt"))
shutil.copy(best_src, WEIGHTS_OUT)
print(f"saved: {WEIGHTS_OUT}")


# %% cell 3 — quick sanity check on test split (per-texture if Roboflow exported tags)
metrics = model.val(data=DATA_YAML, split="test")
print(metrics.box.map)       # mAP@0.5
print(metrics.box.r)         # recall — the metric we care about most


# %% cell 4 — (optional) export to OpenVINO IR for Phase 5
# Comment out if you want to keep training time short; export can also be done locally.
# model.export(format="openvino", half=True)

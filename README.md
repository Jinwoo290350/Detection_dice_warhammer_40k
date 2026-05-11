# Warhammer 40k Dice Detection

ระบบ **detect + นับแต้มลูกเต๋า D6** จากวิดีโอที่ทอย 20–30 ลูกพร้อมกัน — สร้างขึ้นสำหรับผู้เล่น Warhammer 40k ที่ต้องนับผลทอยจำนวนเยอะอย่างรวดเร็ว

ใช้ **YOLOv8n detector + YOLOv8n-cls classifier** (2-stage) + **snapshot logic** (รอลูกเต๋านิ่งแล้วค่อยรันโมเดล — แก้ปัญหา "กรอบกระพริบ" ของ approach real-time tracking) ส่งมอบเป็น **OpenVINO IR** สำหรับ Intel Core Ultra + Arc GPU + NPU

## ผลลัพธ์

| Metric | Value |
|---|---|
| Detector recall (val) | **0.964** ← แก้ pain point "นับไม่ครบ" |
| Detector mAP@0.5 | 0.974 |
| Classifier top-1 (test) | **0.988** |
| Cross-runtime sanity (PyTorch ↔ OpenVINO) | 28/31 snapshots within ±1 die / ±5 pips |

## Quick start

### สำหรับลูกค้า / ผู้ใช้งาน (Windows + Intel)

ใช้ [`delivery_package/`](delivery_package/) — เปิด [`delivery_package/README.md`](delivery_package/README.md) สำหรับวิธีติดตั้งและรัน หรือใช้ build:

```bash
cd delivery_package && zip -r ../delivery_package.zip . && cd ..
# ส่ง delivery_package.zip ให้ลูกค้า
```

### สำหรับ developer / reproducer

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# Phase 1: extract snapshot frames from customer videos
python src/extract_stable_frames.py --all
python src/sample_periodic_frames.py --video Data/WIN_25690506_20_41_32_Pro.mp4 --interval 5

# Phase 4: run pipeline on a clip
python src/count_dice.py --video Data/2548.mp4 --output result.json --save-overlay overlay.mp4
```

ดู workflow เต็มใน [`CLAUDE.md`](CLAUDE.md)

## Layout

```
.
├── CLAUDE.md                       # technical context — อ่านก่อนเริ่มทำงานบน repo
├── src/                            # pipeline + helper scripts
│   ├── stability.py                # snapshot detector (frame stability state machine)
│   ├── pipeline.py                 # DiceCounter: detect → crop → classify
│   ├── count_dice.py               # main CLI
│   ├── verify.py                   # client-side verification script
│   ├── extract_stable_frames.py    # Phase 1
│   ├── sample_periodic_frames.py   # Phase 1
│   ├── build_classifier_dataset.py # Phase 2: crop bboxes for classifier
│   ├── auto_classify_crops.py      # Phase 2: bootstrap sort with N3VERS4YDIE
│   ├── sort_crops_web.py           # Phase 2: web GUI for manual crop sort
│   └── split_classifier_dataset.py # Phase 3: train/val/test split for Ultralytics
├── notebooks/                      # Kaggle training scripts (paste-into-cell format)
│   ├── 02_train_detector_kaggle.py
│   └── 03_train_classifier_kaggle.py
├── docs/
│   └── annotation-guide.md         # bbox rules + occlusion handling
├── tests/
│   └── test_cross_runtime.py       # PyTorch vs OpenVINO sanity comparison
├── models/                         # *.pt weights (gitignored) + OpenVINO IR (tracked)
│   ├── detector_openvino_model/    # FP16 IR for delivery
│   └── classifier_openvino_model/
└── delivery_package/               # final bundle ส่งลูกค้า
    ├── README.md                   # ลูกค้าอ่านอันนี้
    ├── requirements.txt
    ├── src/                        # subset ที่ลูกค้าต้องใช้
    └── models/                     # OpenVINO IR
```

## ที่ไม่ track ใน git (อยู่ใน `.gitignore`)

- `Data/` — วิดีโอลูกค้า (ไม่ commit เพราะใหญ่ + privacy)
- `frames/`, `dataset/`, `classifier_dataset/`, `classifier_split/` — datasets ที่ generate ได้ใหม่
- `models/*.pt`, `models/pretrained/` — PyTorch weights (download from Kaggle หรือ retrain ตาม Phase 3)
- `*.zip` — build artifacts
- `.venv/` — virtual environment

## Reproduce แบบเต็ม

ดู `## สถานะปัจจุบัน` ใน [`CLAUDE.md`](CLAUDE.md) สำหรับ:
- ลำดับการรัน script ทุก phase
- Kaggle dataset slugs
- N3VERS4YDIE warm-start weights source
- 3-tier testing strategy (Mac functional → Kaggle sanity → ลูกค้า verify)

## License & Attribution

- Pretrained detector warm-start: [N3VERS4YDIE/dice-recognition](https://github.com/N3VERS4YDIE/dice-recognition) (CC BY 4.0, dataset `rune-zkmvl/rune55`)
- Customer data + delivery weights: private — ไม่เผยแพร่

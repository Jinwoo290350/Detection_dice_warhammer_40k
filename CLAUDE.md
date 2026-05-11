# CLAUDE.md — Warhammer 40k Dice Detection

## Project Overview

โปรเจกต์รับจ้างอิสระ (Freelance / Hobbyist project) สำหรับลูกค้าผู้เล่นบอร์ดเกม **Warhammer 40k** เป้าหมายคือสร้างระบบ **detect + นับแต้มลูกเต๋า D6 จำนวนมาก (ปกติ 20–30 ลูกพร้อมกัน)** จากวิดีโอที่ลูกค้าทอย

ประเด็นทางเทคนิคหลักคือ **occlusion** (ลูกเต๋าซ้อน/เบียดกัน) และจำนวนลูกเต๋าที่เยอะกว่า scenario ทั่วไปของโมเดล open-source ในตลาด

## Pain Points (จากลูกค้า)

ลูกค้าลองใช้โค้ด open source [`N3VERS4YDIE/dice-recognition`](https://github.com/N3VERS4YDIE/dice-recognition) แล้วเจอปัญหา:

1. **กรอบ detect กระพริบ** ระหว่างที่ลูกเต๋ากำลังกลิ้ง — โมเดลพยายาม track real-time ทำให้ผลไม่นิ่ง
2. **จับลูกเต๋าได้ไม่ครบ** เพราะเบียด/ซ้อนทับกัน (occlusion) + เคลื่อนไหวเร็ว
3. โมเดลเดิม **ไม่ได้ train สำหรับเคส 20–30 ลูกพร้อมกัน** — dataset ต้นทางเน้นลูกเต๋าน้อยลูก กระจายตัวดี

## แนวทางแก้ทางเทคนิค (Proposed Solution)

### 1. Snapshot / Static Count (เปลี่ยน logic)

ยกเลิก real-time tracking ระหว่างที่ลูกเต๋ากำลังกลิ้ง → **รอจนลูกเต๋าหยุดนิ่งแล้วค่อยรันโมเดลครั้งเดียว** ใน frame นั้น

- แก้ปัญหา "กรอบกระพริบ" ตรงต้นเหตุ
- ลด compute (ไม่ต้องรัน inference ทุก frame)
- ตรวจจับ "นิ่ง" ด้วย frame difference / optical flow threshold

### 2. Fine-tune YOLO (ปรับโมเดล)

ใช้ object detection model ขนาดเบา (เช่น YOLOv8n / YOLOv11n) แล้ว fine-tune ด้วย dataset เพิ่มเติมที่:

- มีลูกเต๋าจำนวนมากในเฟรมเดียว (cluster)
- มีเคส occlusion / partial visibility
- cover ทั้ง texture ปกติ และ marble (ดูข้อ 3)

### 3. รองรับ Texture หลายแบบ

ลูกเต๋าของลูกค้ามี **2 texture**:
- **Plain** (ลายปกติ) — สีพื้นเรียบ
- **Marble** (ลายหินอ่อน) — มี noise/texture สูงบนพื้นผิว เสี่ยงโดน confuse กับจุดแต้ม (pip)

Dataset ที่ fine-tune ต้อง cover ทั้งสอง texture และเคส **mixed** (ทอยรวมกัน) ตอนประเมินโมเดลควรแยก test set ตาม texture เพื่อตรวจ bias

## Tech Stack Decisions

| สิ่งที่เลือก | เหตุผล |
|---|---|
| **Class scheme: B (2-stage)** | YOLO detect 1 class (`die`) → crop → classifier (1–6) — annotate detector ง่ายกว่า, classifier swap ได้, รองรับลายอื่นในอนาคต |
| **Detector:** YOLOv8n หรือ YOLOv11n (pretrained COCO) | ขนาดเบา → inference เร็วบนเครื่องลูกค้า |
| **Classifier:** MobileNet / EfficientNet-lite | เล็ก เร็ว เหมาะกับ NPU |
| **Annotation:** Roboflow + auto-label | manual seed 30–50 ภาพ → train baseline → predict ส่วนที่เหลือ → review |
| **Training:** Kaggle (Tesla P100 / T4 ฟรี ~30 ชม./สัปดาห์) | ไม่มี local GPU; Kaggle session 9 ชม. → ออกแบบ training resume ได้ |
| **Inference runtime ที่ลูกค้า:** **OpenVINO** | ลูกค้ามี Intel Arc GPU + NPU 4 (ไม่มี CUDA); Ultralytics export YOLO → OpenVINO IR ได้ตรงๆ |
| **Format ส่งลูกค้า:** OpenVINO IR (`.xml` + `.bin`) | install ง่าย: `pip install openvino` ไม่ต้อง CUDA toolkit |

### Target Hardware (เครื่องลูกค้า)

- **CPU:** Intel Core Ultra 9 285H (16 cores, 5.4 GHz) + **Intel AI Boost NPU**
- **GPU:** Intel Arc Graphics (integrated)
- **RAM:** 32 GB LPDDR5X
- **OS:** Windows (เห็นจากชื่อไฟล์ `WIN_25690506_*.mp4`)
- **Inference target ที่ตั้งใจ:** OpenVINO `device=AUTO` (ให้ runtime เลือก NPU/GPU/CPU เอง)

### Dev Environment

- **Local:** MacBook Pro M2 (Apple Silicon, ARM64)
- **OpenVINO บน Mac:** ใช้ได้ แต่ **CPU only** (ไม่แตะ Apple Silicon GPU)
- **ใช้ Mac ทำได้:** functional/correctness testing, snapshot logic dev, video preprocessing
- **ใช้ Mac ทำไม่ได้:** benchmark performance ที่ลูกค้าจะเห็นจริง (ต้องอาศัยลูกค้า run script verify หรือ Intel Tiber Cloud)

## Testing Strategy (3 ชั้น)

ไม่มี Intel device ที่ dev — ต้องแยก correctness ออกจาก performance

### ชั้น 1 — Functional/Accuracy (Mac, ตลอด dev)

- รัน inference ด้วย OpenVINO CPU บน Mac
- Input: 4 คลิปลูกค้า
- **Ground truth:** manual count (4 คลิปไหวอยู่)
- Metrics: total pip count error, % ลูกที่ detect ครบ (recall), false positive rate

### ชั้น 2 — Cross-runtime sanity (Kaggle)

- รัน inference ของ weights ตัวเดียวกันบน PyTorch GPU (Kaggle)
- เทียบผลกับ OpenVINO CPU บน Mac
- ผลต่างเยอะ → bug ใน export/quantization

### ชั้น 3 — Real-device verification (ลูกค้ารันให้)

- เขียน `verify.py` ที่รัน pipeline แล้ว spit JSON: `{video, snapshots: [{frame_idx, total, breakdown}]}`
- ส่งให้ลูกค้ารันบนเครื่อง Intel ของเขา → ส่ง JSON กลับ → เทียบกับผลบน Mac
- ทำที่ Phase 5 (mid-checkpoint) และก่อน final delivery

**Backup:** Intel Tiber AI Cloud มี Core Ultra ฟรี trial — ใช้เฉพาะ final benchmark ถ้าจำเป็น

## Data Layout

```
Detection_dice_warhammer_40k/
└── Data/
    ├── 2546.mp4                       (~5.7 MB)   marble ล้วน
    ├── 2547.mp4                       (~3.0 MB)   plain ล้วน
    ├── 2548.mp4                       (~2.9 MB)   mixed — marble + plain คละกัน
    ├── WIN_25690506_20_41_32_Pro.mp4  (~386 MB)   plain ล้วน (screen recording ขนาดใหญ่)
    └── Result_custumer/
        └── 2543.mp4                   (~8.5 MB)   ผลลัพธ์จากโมเดลของลูกค้าเอง — baseline ปัญหา
```

**สำคัญ:**
- `Result_custumer/2543.mp4` คือ output ที่โมเดลเดิมของลูกค้ารัน — ใช้ **ดูปัญหา** (กระพริบ / นับไม่ครบ) **ไม่ใช่ ground truth**
- โฟลเดอร์สะกด `Result_custumer` (ไม่ใช่ `Result_customer`) — **อย่าเปลี่ยนชื่อ** เพื่อไม่ให้ path ลูกค้าพัง
- ยังไม่มี ground truth labels — ต้องสร้าง annotation เอง (Roboflow / CVAT / Label Studio)

## Budget & Timeline

| | |
|---|---|
| **ราคา** | **5,000 บาท** (เรทพิเศษสำหรับงานอดิเรก เพื่อให้ลูกค้าตัดสินใจง่าย) |
| **ระยะเวลา** | **1–2 สัปดาห์** |
| **ลักษณะงาน** | Freelance, hobbyist scope — ไม่ใช่งาน production-grade |

ผลของ scope แคบ:
- ไม่ต้องทำ deployment / API server / UI สวย
- เน้นใช้งานได้กับวิดีโอที่ลูกค้าให้ + วิดีโอใหม่ในลักษณะเดียวกัน
- ไม่ต้องทำ retraining pipeline ให้ลูกค้าเอง

## สถานะปัจจุบัน — เสร็จทุก phase ✅

- ✅ Phase 1: env + 103 frames extracted
- ✅ Phase 2: annotated 103 images on Roboflow (`deep-xnqut/warhammer-dice-detector`) + classifier dataset 1,725 crops user manually sorted
- ✅ Phase 3: detector (mAP@0.5=0.974, recall=0.964) + classifier (top1=0.988) — trained on Kaggle, weights ใน `models/`
- ✅ Phase 4: snapshot pipeline (stability + pipeline + CLI) verified end-to-end ทุกคลิป
- ✅ Phase 5: OpenVINO IR exported + cross-runtime sanity (28/31 snapshots within tolerance)
- ✅ Phase 6: `delivery_package.zip` (7.9 MB) พร้อมส่งลูกค้า

### Phase 3+ Artifacts

| Path | Purpose |
|---|---|
| `models/detector.pt` | YOLOv8n trained on 103 images (1,890 train bboxes) |
| `models/classifier.pt` | YOLOv8n-cls trained on 1,725 sorted crops |
| `models/detector_openvino_model/` | FP16 OpenVINO IR for delivery |
| `models/classifier_openvino_model/` | FP16 OpenVINO IR for delivery |
| `models/pretrained/n3v_best.pt` | N3VERS4YDIE warm-start weights (kept for reproducibility) |
| `src/stability.py` | StabilityDetector class (state machine, time-based) |
| `src/pipeline.py` | DiceCounter class (detect → crop → classify) |
| `src/count_dice.py` | main CLI |
| `src/verify.py` | client-side verification script |
| `src/auto_classify_crops.py` | bootstrap classifier sort with N3VERS4YDIE |
| `src/sort_crops_web.py` | web GUI for manual crop sorting (port 8765) |
| `src/split_classifier_dataset.py` | train/val/test splitter |
| `src/upload_to_roboflow.py` | bulk upload (REST API; not used — Roboflow web UI was simpler) |
| `notebooks/02_train_detector_kaggle.py` | Kaggle detector training |
| `notebooks/03_train_classifier_kaggle.py` | Kaggle classifier training |
| `tests/test_cross_runtime.py` | PyTorch vs OpenVINO sanity comparison |
| `delivery_package/` | final bundle: src + models + requirements + README |
| `delivery_package.zip` | 7.9 MB zip ready to send |

### Phase 1 Artifacts

| Path | Purpose |
|---|---|
| `.venv/` | Python 3.11.14 venv — torch 2.11, ultralytics 8.4, openvino 2026.1, opencv 4.13 |
| `requirements.txt` | runtime-only deps (สำหรับ deliverable) |
| `requirements-dev.txt` | dev deps (training + notebooks) |
| `src/extract_stable_frames.py` | state-machine stability detector (1 frame ต่อ rolling cycle) |
| `src/sample_periodic_frames.py` | time-interval sampler (สำหรับเติม unlabeled pool) |
| `frames/marble/` | 29 frames (จาก 2546.mp4) |
| `frames/plain/` | 60 frames (2547.mp4 + WIN_*.mp4) |
| `frames/mixed/` | 14 frames (2548.mp4) |

**Frame budget per Phase 2:** ~30–50 manual seed labels (กระจาย texture) → train baseline → auto-label ที่เหลือ → review

### Video Metadata (จาก ffprobe)

| File | Resolution | FPS | Duration | Notes |
|---|---|---|---|---|
| `2546.mp4` | 640×368 (portrait via rotation) | 30 | 58.9s | phone, marble |
| `2547.mp4` | 640×368 (portrait via rotation) | 30 | 31.9s | phone, plain |
| `2548.mp4` | 640×368 (portrait via rotation) | 30 | 29.6s | phone, mixed |
| `WIN_25690506_*.mp4` | 1920×1080 | ~19.92 (VFR 239/12) | 199.9s | screen recording, plain in pink/black tray, ~30 dice — primary plain training source |

## Conventions / Notes สำหรับ Claude

- **อย่า `cat` หรือ read ไฟล์ `.mp4` เข้า context** — ขนาดใหญ่มาก (โดยเฉพาะ `WIN_25690506_*.mp4` 386 MB) ใช้ `ffprobe` ดู metadata หรือ `ffmpeg` extract frame เป็น `.jpg` แล้วใช้ Read แทน
- **ภาษาที่ใช้สื่อสาร:** ผู้ใช้พิมพ์ **ไทยผสมอังกฤษ** (ศัพท์เทคนิคเป็นอังกฤษ บริบทเป็นไทย) — ตอบในสไตล์เดียวกัน
- **อย่าแก้ path `Result_custumer`** (ตามหมายเหตุข้างบน)
- **Reference โค้ด open source:** [`N3VERS4YDIE/dice-recognition`](https://github.com/N3VERS4YDIE/dice-recognition) — ใช้เป็นจุดเริ่ม / ดู approach เดิมของลูกค้า ไม่ใช่ของที่ต้อง maintain
- **อย่าใช้ CUDA หรือ MPS เป็น runtime สำหรับ deliverable** — เครื่องลูกค้าเป็น Intel; final inference ต้องผ่าน OpenVINO เท่านั้น
- **อย่าใช้ Mac MPS ใน final pipeline** — Mac เป็น dev env เท่านั้น; ทุก script ที่ส่งลูกค้าต้องรันบน Windows + Intel ได้

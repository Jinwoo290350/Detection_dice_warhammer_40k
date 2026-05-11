# Warhammer 40k Dice Counter

ระบบตรวจจับและนับลูกเต๋า D6 จากวิดีโอ — รันบน Intel CPU/GPU/NPU ผ่าน OpenVINO

## Hardware ที่รองรับ

ออกแบบมาสำหรับ **Intel Core Ultra 9 285H + Intel Arc Graphics + AI Boost NPU** (Windows) — แต่รันได้ทุกเครื่องที่ติดตั้ง Python + OpenVINO

## Install

ต้องการ Python 3.10–3.12 (ไม่รองรับ 3.13+)

```cmd
:: 1. สร้าง virtual environment (แนะนำ)
python -m venv .venv
.venv\Scripts\activate

:: 2. ติดตั้ง dependencies
pip install -r requirements.txt
```

ครั้งแรกจะใช้เวลาประมาณ 2–5 นาที (download torch ~800 MB)

## Usage

### Mode 1 — CLI: count dice ในวิดีโอ → JSON

```cmd
python src\count_dice.py --video your_clip.mp4 --output result.json
```

เพิ่ม `--save-overlay overlay.mp4` ถ้าอยากได้วิดีโอที่ขึ้นกรอบ + ผล count/sum/D3

### Mode 2 — HTTP API (สำหรับ Angular / web frontend)

```cmd
python src\api.py --port 5000
```

จะเห็น:
```
loading detector: ...models\detector_openvino_model
loading classifier: ...models\classifier_openvino_model
models loaded — ready
 * Running on http://0.0.0.0:5000
```

**Endpoints:**

| Method | Path | Body | Returns |
|---|---|---|---|
| GET  | `/health` | — | `{ok, detector, classifier}` |
| POST | `/count`  | multipart `video=<file>` (+ optional form fields `diff_threshold`, `stable_seconds`, `motion_threshold`) | full JSON (same schema as CLI output) |

CORS เปิดที่ `*` — Angular dev server (`localhost:4200`) เรียกได้ตรงๆ ไม่มี proxy

**Angular example:**

```typescript
// component.ts
async upload(file: File) {
  const form = new FormData();
  form.append("video", file);
  const result = await firstValueFrom(
    this.http.post<DiceResult>("http://localhost:5000/count", form)
  );
  console.log("total dice:", result.totals.dice_count);
  console.log("total sum:", result.totals.total_dice_sum);
  console.log("D3 sum:", result.totals.D3_total_dice_sum);
  console.log("4+:", result.totals.dice_4_plus_count);
}
```

**curl test:**

```cmd
curl http://localhost:5000/health
curl -X POST -F "video=@your_clip.mp4" http://localhost:5000/count
```

### Mode 3 — Verify บนเครื่องคุณ (ส่ง JSON กลับให้ผมเทียบ)

```cmd
python src\verify.py --video your_clip.mp4 --output verify.json
```

## Output schema (JSON)

```json
{
  "video": "your_clip.mp4",
  "fps": 30.0,
  "frame_count": 889,
  "totals": {
    "snapshot_count": 6,
    "dice_count": 104,
    "total_dice_sum": 367,
    "D3_dice_count": 104,
    "D3_total_dice_sum": 203,
    "dice_1_count": 9,  "dice_1_plus_count": 104,
    "dice_2_count": 26, "dice_2_plus_count": 95,
    "dice_3_count": 15, "dice_3_plus_count": 69,
    "dice_4_count": 24, "dice_4_plus_count": 54,
    "dice_5_count": 15, "dice_5_plus_count": 30,
    "dice_6_count": 15, "dice_6_plus_count": 15
  },
  "snapshots": [
    {
      "frame_idx": 80,
      "timestamp": 2.67,
      "dice_count": 14, "total_dice_sum": 49, "D3_total_dice_sum": 26,
      "dice_1_count": 0, "dice_1_plus_count": 14,
      "dice_2_count": 5, "dice_2_plus_count": 14,
      "dice_3_count": 2, "dice_3_plus_count": 9,
      "dice_4_count": 4, "dice_4_plus_count": 7,
      "dice_5_count": 1, "dice_5_plus_count": 3,
      "dice_6_count": 2, "dice_6_plus_count": 2,
      "dice": [
        { "bbox": [298, 253, 368, 326], "pip": 2, "det_conf": 0.96, "cls_conf": 0.99 },
        ...
      ]
    },
    ...
  ]
}
```

แต่ละ "snapshot" คือ frame ที่ลูกเต๋าหยุดนิ่ง (รอบทอย 1 รอบ) — ระบบ detect การเปลี่ยน rolling → resting อัตโนมัติ

### Fields

| Field | คำอธิบาย |
|---|---|
| `dice_count` | จำนวนลูกเต๋าที่ detect ได้ |
| `total_dice_sum` | ผลรวมแต้ม D6 |
| `D3_dice_count` | จำนวนลูก (เหมือน `dice_count` — แสดงในมุมมอง D3) |
| `D3_total_dice_sum` | ผลรวม D3 จาก D6 — `ceil(pip/2)` (1,2→1; 3,4→2; 5,6→3) ใช้ใน Warhammer mechanic |
| `dice_N_count` | จำนวนลูกที่ออกแต้ม N exactly (N=1..6) |
| `dice_N_plus_count` | จำนวนลูกที่ออกแต้ม ≥ N — Warhammer to-hit/to-wound (e.g. `dice_4_plus_count` = "wound on 4+") |
| `totals` | สรุปรวมทั้งวิดีโอ (รวมทุก snapshot) — stats linear ก็แค่ summed |

## ตัวเลือกที่ปรับได้

| Flag | Default | คำอธิบาย |
|---|---|---|
| `--det-conf` | 0.25 | confidence threshold ของ detector — ลดถ้าจับลูกเต๋าไม่ครบ, เพิ่มถ้ามี false positive |
| `--diff-threshold` | 2.5 | ความนิ่งของภาพถึงเรียกว่า "ไม่ขยับ" (mean abs-diff 0–255) — ลดถ้าจับ snapshot ยากเกิน |
| `--stable-seconds` | 0.5 | ลูกเต๋าต้องนิ่งกี่วินาทีก่อน count |
| `--motion-threshold` | 4.0 | ต้องเห็น motion ถึงค่านี้ก่อนจะ rearm สำหรับรอบถัดไป |

ตัวอย่าง: ถ้าจับ snapshot ไม่ค่อยได้ (วิดีโอมี shake เยอะ):

```cmd
python src\count_dice.py --video your_clip.mp4 --output result.json --diff-threshold 4.0 --stable-seconds 0.3
```

## Inference device

OpenVINO เลือก device อัตโนมัติ (`AUTO`) — จะใช้ NPU > GPU > CPU ตามลำดับ ถ้าต้องการ force device ใดเป็นหลัก แก้ environment variable:

```cmd
set OV_DEVICE=GPU    :: หรือ NPU, CPU
```

## โครงสร้างไฟล์

```
delivery_package/
├── README.md                    ← ไฟล์นี้
├── requirements.txt
├── src/
│   ├── stability.py             ← stability detector
│   ├── pipeline.py              ← detect → crop → classify pipeline
│   ├── count_dice.py            ← CLI หลัก
│   └── verify.py                ← test script
└── models/
    ├── detector_openvino_model/ ← YOLOv8n detector (FP16)
    │   ├── detector.xml
    │   ├── detector.bin
    │   └── metadata.yaml
    └── classifier_openvino_model/  ← YOLOv8n-cls (FP16)
        ├── classifier.xml
        ├── classifier.bin
        └── metadata.yaml
```

## Performance ที่คาดหวัง (Intel Core Ultra 9 285H)

- **Detector** (YOLOv8n, 640×640): ~10–30 ms/frame บน Arc GPU
- **Classifier** (YOLOv8n-cls, 64×64): <5 ms/crop บน NPU
- **End-to-end** ต่อ snapshot ที่มี 30 ลูกเต๋า: ~150–300 ms

ระบบรัน inference เฉพาะตอน "stable snapshot" เท่านั้น — ระหว่างที่ลูกเต๋ากลิ้งจะ skip → ไม่กินทรัพยากร

## Troubleshooting

### "ไม่มี snapshot เกิดขึ้นเลย"

ลด `--diff-threshold` หรือ `--stable-seconds`:

```cmd
python src\count_dice.py --video clip.mp4 --output r.json --diff-threshold 4.0 --stable-seconds 0.3
```

### "ลูกเต๋าบางลูก detect ไม่เจอ"

ลด `--det-conf`:

```cmd
python src\count_dice.py --video clip.mp4 --output r.json --det-conf 0.15
```

### "นับ pip ผิด (เช่น 5 อ่านเป็น 4)"

ดู `cls_conf` ใน JSON — ถ้าต่ำ < 0.7 บนลูกที่ผิด แสดงว่า classifier ไม่มั่นใจ — อาจเก็บข้อมูลเพิ่มของแต้มนั้นมา fine-tune classifier ใหม่

## Limitations

- ระบบนี้รองรับ **D6 ลูกเต๋ามาตรฐาน** (plain หรือ marble texture) — D8/D10/D20 ไม่รองรับ
- คุณภาพสูงสุดได้กับวิดีโอที่ลูกเต๋าหยุดนิ่ง > 0.5 วินาที — ถ้าทอยแล้วเก็บทันที model อาจไม่ทัน trigger snapshot
- ลูกเต๋าตะแคงอ่านยาก (เห็น 2 หน้า) — ระบบจะเดาหน้าที่เห็นเด่นชัดสุด

---

ติดต่อกลับมาถ้าเจอปัญหา หรือต้องการ fine-tune model เพิ่มเติม

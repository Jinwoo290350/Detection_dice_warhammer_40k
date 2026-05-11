# Annotation Guide — Manual Seed (Phase 2 Step 1)

เป้าหมาย: annotate **30–50 ภาพแรก** ใน Roboflow ให้กระจายทั้ง 3 textures (marble / plain / mixed) เพื่อใช้ train baseline detector ที่จะใช้ auto-label ภาพที่เหลือ

## Class scheme

ใช้ **1 class เดียว: `die`** (ตาม Class Scheme B ใน [CLAUDE.md](../CLAUDE.md))

อย่า annotate แต้ม (1–6) ในขั้นนี้ — แต้มจะถูกจัดการโดย classifier แยกต่างหากใน Phase 2 step 6

## Bbox rules

### 1. ลูกเต๋านิ่งสมบูรณ์, ไม่ซ้อน
- bbox **tight** ครอบขอบลูกพอดี
- ไม่กิน shadow, ไม่กิน whitespace มาก

### 2. ลูกเต๋าซ้อนกัน (occlusion)
- ครอบ **เฉพาะส่วนที่มองเห็น** ไม่เดาส่วนที่ถูกบัง
- ถ้าเห็น **น้อยกว่า ~30%** ของลูก → **skip** ลูกนั้น
- เห็น 30–100% → annotate ตามขอบจริงที่เห็น

### 3. ลูกเต๋าครึ่งโผล่ออกขอบเฟรม
- เห็น ≥ 50% → annotate ตามส่วนที่อยู่ในเฟรม
- เห็น < 30% → skip

### 4. ลูกเต๋าตะแคง (เห็นหน้าข้าง)
- annotate ปกติ — classifier ค่อย handle
- bbox ครอบทั้งลูกที่ตะแคง

### 5. Blurry / out-of-focus / motion blur
- ถ้าแต้มอ่านไม่ออกชัด → **skip** ทั้งลูก
- ถ้าโครงรูปลูกเต๋ายังเห็นชัด → annotate ก็ได้

## Texture coverage target

ตอน annotate manual seed (~30–50 ภาพ) เลือกให้กระจาย:
- **marble** ~10–15 ภาพ (จาก `frames/marble/`, `frames/mixed/`)
- **plain** ~15–25 ภาพ (จาก `frames/plain/`)
- **mixed** ~5–10 ภาพ (จาก `frames/mixed/`)

ใน Roboflow ใช้ filter ที่ **tag** ของแต่ละภาพ (script upload ติด tag = ชื่อ folder ให้แล้ว) เพื่อ batch annotate ทีละ texture

## Edge cases ที่ควรระวัง

| สถานการณ์ | วิธีจัดการ |
|---|---|
| ลูกเต๋าติดกัน 2 ลูก แต่ขอบยังแยกได้ | annotate **2 boxes** แยก ครอบเฉพาะที่เห็นของแต่ละลูก |
| ลูก 3 ลูกซ้อนกันเป็น stack | annotate ลูกบนสุด (เห็นเต็ม) + ลูกที่เห็น ≥ 30% ส่วนที่เห็น |
| Reflection ของลูกเต๋าบนพื้น (เงา) | **อย่า** annotate เงาเป็น die |
| ลูกเต๋าใน tray (WIN_* clip) | ครอบเฉพาะลูก ไม่รวมขอบ tray |
| มีของอื่นในเฟรม (มือ, เหรียญ, ปืน figurine) | **skip** ไม่ annotate |

## Quality bar (ก่อน export)

ก่อนส่ง dataset ไป train ตรวจ:
- ทุก die ที่เห็น > 30% มี bbox
- ไม่มี "ghost annotation" (bbox ที่ครอบของอื่น)
- bbox ไม่หลวมเกินไป (กิน whitespace > 20%)
- Class field = `die` ทุก instance

## After manual seed

หลัง annotate เสร็จ + export:
1. ผม download dataset YOLO format
2. Upload ขึ้น Kaggle เป็น Kaggle Dataset
3. Train YOLOv8n baseline ~50 epochs
4. รัน predict กับภาพที่เหลือใน `frames/` (~50–70 ภาพ)
5. Upload predictions กลับ Roboflow เป็น **model-assisted annotations**
6. คุณ review/แก้ใน Roboflow UI (เร็วกว่า annotate from scratch)

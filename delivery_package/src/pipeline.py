"""Two-stage detect→classify pipeline (Class Scheme B in CLAUDE.md).

`DiceCounter` runs the YOLO detector on a frame to get one bbox per die,
crops each bbox, runs the classifier on each crop to read the pip face,
and returns the per-die breakdown plus the totals.

Runtime defaults to PyTorch for development convenience. OpenVINO IR
weights (Phase 5 export) are loaded transparently if you point to a `.xml`
file or to an OpenVINO export directory; ultralytics handles either.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from ultralytics import YOLO


class DieResult(TypedDict):
    bbox: list[int]      # [x1, y1, x2, y2] in pixels
    pip: int             # 1..6
    det_conf: float
    cls_conf: float


# Output schema is a flat dict (Angular/JSON-friendly). Top-level keys:
#   STAT_KEYS (16 stat fields, all int) + "dice": list[DieResult]
# All stats are linear so totals across snapshots = Σ snapshot[stat].
STAT_KEYS = (
    "dice_count",
    "total_dice_sum",
    "D3_dice_count",
    "D3_total_dice_sum",
    "dice_1_count", "dice_1_plus_count",
    "dice_2_count", "dice_2_plus_count",
    "dice_3_count", "dice_3_plus_count",
    "dice_4_count", "dice_4_plus_count",
    "dice_5_count", "dice_5_plus_count",
    "dice_6_count", "dice_6_plus_count",
)


def d3_from_d6(pip: int) -> int:
    """Warhammer D3 result derived from a D6 roll: 1,2→1; 3,4→2; 5,6→3."""
    return (pip + 1) // 2


def compute_dice_stats(dice: list[DieResult]) -> dict[str, int]:
    """Build the flat stats dict from a list of detected dice."""
    counts = {face: 0 for face in range(1, 7)}
    for d in dice:
        counts[d["pip"]] += 1
    plus_counts = {face: sum(counts[f] for f in range(face, 7)) for face in range(1, 7)}
    total_dice = sum(counts.values())
    total_sum = sum(face * n for face, n in counts.items())
    total_d3 = sum(d3_from_d6(face) * n for face, n in counts.items())
    out: dict[str, int] = {
        "dice_count": total_dice,
        "total_dice_sum": total_sum,
        "D3_dice_count": total_dice,
        "D3_total_dice_sum": total_d3,
    }
    for face in range(1, 7):
        out[f"dice_{face}_count"] = counts[face]
        out[f"dice_{face}_plus_count"] = plus_counts[face]
    return out


def empty_stats() -> dict[str, int]:
    return {k: 0 for k in STAT_KEYS}


class DiceCounter:
    def __init__(
        self,
        detector_path: str | Path,
        classifier_path: str | Path,
        *,
        det_conf: float = 0.25,
        det_imgsz: int = 640,
        cls_imgsz: int = 64,
        crop_padding: float = 0.08,
    ) -> None:
        # Pass explicit task so Ultralytics doesn't have to guess from path —
        # OpenVINO IR exports lose the metadata otherwise.
        self._detector = YOLO(str(detector_path), task="detect")
        self._classifier = YOLO(str(classifier_path), task="classify")
        self.det_conf = det_conf
        self.det_imgsz = det_imgsz
        self.cls_imgsz = cls_imgsz
        self.crop_padding = crop_padding

    def count(self, frame: np.ndarray) -> dict:
        """Run detect + classify on a single frame.

        Returns a flat dict with the 16 stat fields (see STAT_KEYS) plus a
        `dice` list of per-die details.
        """
        h, w = frame.shape[:2]
        det = self._detector.predict(
            frame,
            conf=self.det_conf,
            imgsz=self.det_imgsz,
            verbose=False,
        )[0]
        if len(det.boxes) == 0:
            return {**empty_stats(), "dice": []}

        # Crop with padding, batch the crops through the classifier in one call
        crops: list[np.ndarray] = []
        bboxes_px: list[tuple[int, int, int, int]] = []
        det_confs: list[float] = []
        for box, conf in zip(det.boxes.xyxy.cpu().numpy(), det.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            pad = self.crop_padding * max(bw, bh)
            xa = max(0, int(x1 - pad))
            ya = max(0, int(y1 - pad))
            xb = min(w, int(x2 + pad))
            yb = min(h, int(y2 + pad))
            if xb - xa < 8 or yb - ya < 8:
                continue
            crops.append(frame[ya:yb, xa:xb])
            bboxes_px.append((xa, ya, xb, yb))
            det_confs.append(float(conf))

        if not crops:
            return {**empty_stats(), "dice": []}

        # Predict crops one-at-a-time for OpenVINO compatibility (static batch=1).
        # Classifier inference is ~1ms/crop, so this isn't a bottleneck.
        cls_results = [
            self._classifier.predict(crop, imgsz=self.cls_imgsz, verbose=False)[0]
            for crop in crops
        ]
        names = self._classifier.names

        dice: list[DieResult] = []
        for bbox, det_c, r in zip(bboxes_px, det_confs, cls_results):
            top1 = int(r.probs.top1)
            top1conf = float(r.probs.top1conf)
            pip = int(names[top1])
            dice.append({
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "pip": pip,
                "det_conf": det_c,
                "cls_conf": top1conf,
            })

        stats = compute_dice_stats(dice)
        return {**stats, "dice": dice}


def render_overlay(frame: np.ndarray, result: dict) -> np.ndarray:
    """Draw bbox + pip face value over a frame copy for visual debugging."""
    out = frame.copy()
    for d in result["dice"]:
        x1, y1, x2, y2 = d["bbox"]
        color = (0, 220, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d['pip']} ({d['cls_conf']:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    summary = (
        f"count={result['dice_count']}  sum={result['total_dice_sum']}  "
        f"D3 sum={result['D3_total_dice_sum']}"
    )
    cv2.putText(out, summary, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)
    return out

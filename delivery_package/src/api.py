"""HTTP API wrapper around the dice counter — for Angular / web frontends.

Endpoints:
  GET  /health            → { ok: true, ... }
  POST /count             → upload a video file, get back the same JSON schema
                            count_dice.py produces (top-level `totals` +
                            per-snapshot stats).

Run from project root (or delivery_package/):
    python src/api.py
    # → listens on http://0.0.0.0:5000

Then from Angular:
    const form = new FormData();
    form.append("video", fileBlob);
    this.http.post("http://<server>:5000/count", form).subscribe(...)

CORS is open by default (`*`) so the Angular dev server can hit it across
origins. Lock it down for production.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import cv2
from flask import Flask, jsonify, request
from flask_cors import CORS

from pipeline import DiceCounter, STAT_KEYS
from stability import StabilityDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DETECTOR = PROJECT_ROOT / "models" / "detector_openvino_model"
DEFAULT_CLASSIFIER = PROJECT_ROOT / "models" / "classifier_openvino_model"
# Allow override via env var for deployments
DETECTOR = Path(os.environ.get("DICE_DETECTOR", DEFAULT_DETECTOR))
CLASSIFIER = Path(os.environ.get("DICE_CLASSIFIER", DEFAULT_CLASSIFIER))

app = Flask(__name__)
CORS(app)  # allow any origin — tighten for production

# Load models once at startup — Ultralytics + OpenVINO compile is slow,
# we don't want it per-request.
print(f"loading detector: {DETECTOR}")
print(f"loading classifier: {CLASSIFIER}")
COUNTER = DiceCounter(DETECTOR, CLASSIFIER)
print("models loaded — ready")


def run_video(video_path: Path, params: dict) -> dict:
    """Run the full snapshot pipeline on a video file. Returns the JSON dict."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stability = StabilityDetector(
        fps=fps,
        diff_threshold=params.get("diff_threshold", 2.5),
        stable_seconds=params.get("stable_seconds", 0.5),
        motion_threshold=params.get("motion_threshold", 4.0),
    )

    snapshots: list[dict] = []
    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if not stability.feed(frame):
            continue
        result = COUNTER.count(frame)
        snapshots.append({
            "frame_idx": frame_idx,
            "timestamp": frame_idx / fps,
            **result,
        })
    cap.release()

    totals: dict[str, int] = {k: sum(s[k] for s in snapshots) for k in STAT_KEYS}
    totals["snapshot_count"] = len(snapshots)
    return {
        "video": video_path.name,
        "fps": fps,
        "frame_count": frame_idx + 1,
        "totals": totals,
        "snapshots": snapshots,
    }


@app.get("/health")
def health() -> tuple:
    return jsonify({
        "ok": True,
        "detector": str(DETECTOR),
        "classifier": str(CLASSIFIER),
    }), 200


@app.post("/count")
def count() -> tuple:
    file = request.files.get("video")
    if file is None:
        return jsonify({"error": "missing 'video' file in multipart form"}), 400

    # Optional tuning params from query string or form
    def _f(name: str, default: float) -> float:
        v = request.values.get(name)
        return float(v) if v is not None else default

    params = {
        "diff_threshold": _f("diff_threshold", 2.5),
        "stable_seconds": _f("stable_seconds", 0.5),
        "motion_threshold": _f("motion_threshold", 4.0),
    }

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = Path(tmp.name)

    try:
        result = run_video(tmp_path, params)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500
    finally:
        tmp_path.unlink(missing_ok=True)

    return jsonify(result), 200


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

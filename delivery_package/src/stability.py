"""Stability detector for the snapshot pipeline.

Streams frames in via `feed(frame)` and emits one "stable snapshot" event per
"rolling → resting" cycle. The state machine matches the offline extractor in
`extract_stable_frames.py`:

  - while consecutive mean abs-diff stays under `diff_threshold` for at least
    `stable_seconds`, emit the latest frame
  - after emitting, require a frame whose diff exceeds `motion_threshold` (the
    dice were picked up / rolled again) before re-arming
"""

from __future__ import annotations

import cv2
import numpy as np


class StabilityDetector:
    def __init__(
        self,
        *,
        fps: float = 30.0,
        diff_threshold: float = 2.5,
        stable_seconds: float = 0.5,
        motion_threshold: float = 4.0,
        downscale_for_diff: int = 320,
    ) -> None:
        self.fps = fps if fps > 0 else 30.0
        self.diff_threshold = diff_threshold
        self.stable_seconds = stable_seconds
        self.motion_threshold = motion_threshold
        self.downscale_for_diff = downscale_for_diff
        self._dt = 1.0 / self.fps
        self._prev: np.ndarray | None = None
        self._stable_run = 0.0
        self._armed = True
        self.last_diff: float = 0.0

    def feed(self, frame: np.ndarray) -> bool:
        """Returns True if `frame` is a stable snapshot (emit-once per cycle)."""
        h, w = frame.shape[:2]
        scale = self.downscale_for_diff / w
        small = cv2.resize(frame, (self.downscale_for_diff, max(1, int(h * scale))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev is None:
            self._prev = gray
            return False

        diff = float(cv2.absdiff(gray, self._prev).mean())
        self._prev = gray
        self.last_diff = diff

        if diff < self.diff_threshold:
            self._stable_run += self._dt
            if self._armed and self._stable_run >= self.stable_seconds:
                self._armed = False
                self._stable_run = 0.0
                return True
        else:
            self._stable_run = 0.0
            if diff >= self.motion_threshold:
                self._armed = True
        return False

    def reset(self) -> None:
        self._prev = None
        self._stable_run = 0.0
        self._armed = True

"""Tk GUI for fast manual sorting of dice crops into face-value folders.

Shows one crop at a time. Press 1..6 to move it into the matching face folder.
Press SPACE to skip (re-queue). Press D to delete (corrupt/junk crop). Press
R to restart from the start of the queue. Press Q to quit.

Counter on top shows current per-folder counts so you know when you've hit
the target (default ~50 per face).

Usage:
    python src/sort_crops_tk.py
"""

from __future__ import annotations

import shutil
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont

from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "classifier_dataset"
UNSORTED = DATASET_DIR / "_unsorted"


class SortApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.queue: list[Path] = sorted(UNSORTED.glob("*.jpg"))
        self.idx = 0

        root.title("Sort dice crops — 1..6 = face value, SPACE = skip, D = delete, Q = quit")
        root.geometry("520x620")

        big = tkfont.Font(family="Helvetica", size=14)
        small = tkfont.Font(family="Helvetica", size=11)

        self.counts_lbl = tk.Label(root, font=big)
        self.counts_lbl.pack(pady=8)

        self.img_lbl = tk.Label(root, bg="#222")
        self.img_lbl.pack(pady=8)

        self.name_lbl = tk.Label(root, font=small)
        self.name_lbl.pack()

        self.queue_lbl = tk.Label(root, font=small)
        self.queue_lbl.pack()

        for f in "123456":
            root.bind(f, lambda _e, face=int(f): self._move(face))
        root.bind("<space>", lambda _e: self._skip())
        root.bind("d", lambda _e: self._delete())
        root.bind("q", lambda _e: root.destroy())

        for face in range(1, 7):
            (DATASET_DIR / str(face)).mkdir(exist_ok=True)

        self._refresh()

    def _counts(self) -> dict[int, int]:
        return {f: len(list((DATASET_DIR / str(f)).glob("*.jpg"))) for f in range(1, 7)}

    def _refresh(self) -> None:
        counts = self._counts()
        self.counts_lbl.config(
            text="  ".join(f"{f}: {n}" for f, n in counts.items())
            + f"   total: {sum(counts.values())}"
        )
        if self.idx >= len(self.queue):
            self.img_lbl.config(image="", text="DONE", fg="#0f0", bg="#000",
                                width=40, height=20, font=("Helvetica", 24))
            self.name_lbl.config(text="")
            self.queue_lbl.config(text=f"queue empty ({len(self.queue)} processed)")
            return
        path = self.queue[self.idx]
        try:
            img = Image.open(path)
            img.thumbnail((400, 400))
            self._photo = ImageTk.PhotoImage(img)
            self.img_lbl.config(image=self._photo, text="", width=400, height=400)
        except Exception as exc:  # noqa: BLE001
            self.img_lbl.config(image="", text=f"failed: {exc}")
        self.name_lbl.config(text=path.name[:64])
        self.queue_lbl.config(text=f"{self.idx + 1} / {len(self.queue)} in queue")

    def _move(self, face: int) -> None:
        if self.idx >= len(self.queue):
            return
        path = self.queue[self.idx]
        if path.exists():
            shutil.move(str(path), str(DATASET_DIR / str(face) / path.name))
        self.idx += 1
        self._refresh()

    def _skip(self) -> None:
        self.idx += 1
        self._refresh()

    def _delete(self) -> None:
        if self.idx >= len(self.queue):
            return
        path = self.queue[self.idx]
        if path.exists():
            path.unlink()
        self.idx += 1
        self._refresh()


def main() -> None:
    if not UNSORTED.exists() or not any(UNSORTED.glob("*.jpg")):
        raise SystemExit(f"no crops in {UNSORTED}")
    root = tk.Tk()
    SortApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

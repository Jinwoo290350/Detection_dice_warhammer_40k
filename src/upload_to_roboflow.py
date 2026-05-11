"""Upload `frames/` to a Roboflow Object Detection project.

Reads credentials from environment variables or a project-root `.env` file:
    ROBOFLOW_API_KEY=...
    ROBOFLOW_WORKSPACE=...
    ROBOFLOW_PROJECT=...

Each frame's parent folder name (`marble` / `plain` / `mixed`) is attached as
a Roboflow tag so we can split test sets per-texture later.

Usage:
    python src/upload_to_roboflow.py --dry-run   # list what would be uploaded
    python src/upload_to_roboflow.py             # actual upload
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = PROJECT_ROOT / "frames"
TEXTURES = ("marble", "plain", "mixed")


def load_dotenv() -> None:
    """Minimal .env loader — avoids extra dependency."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def collect_frames() -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for texture in TEXTURES:
        for img in sorted((FRAMES_DIR / texture).glob("*.jpg")):
            out.append((img, texture))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-name", default="phase1-frames")
    args = parser.parse_args()

    load_dotenv()
    frames = collect_frames()
    if not frames:
        print(f"no jpg files found under {FRAMES_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"found {len(frames)} frames")
    by_texture: dict[str, int] = {}
    for _, t in frames:
        by_texture[t] = by_texture.get(t, 0) + 1
    for t, n in by_texture.items():
        print(f"  {t}: {n}")

    if args.dry_run:
        for path, texture in frames[:5]:
            print(f"  would upload {path.relative_to(PROJECT_ROOT)} (tag={texture})")
        print("  ...")
        return

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    workspace = os.environ.get("ROBOFLOW_WORKSPACE")
    project_id = os.environ.get("ROBOFLOW_PROJECT")
    if not (api_key and workspace and project_id):
        print(
            "missing one of ROBOFLOW_API_KEY / ROBOFLOW_WORKSPACE / ROBOFLOW_PROJECT",
            file=sys.stderr,
        )
        sys.exit(2)

    import requests

    upload_url = f"https://api.roboflow.com/dataset/{workspace}/{project_id}/upload"

    failed: list[Path] = []
    for i, (path, texture) in enumerate(frames, 1):
        try:
            with path.open("rb") as fh:
                resp = requests.post(
                    upload_url,
                    params={
                        "api_key": api_key,
                        "name": path.name,
                        "split": "train",
                        "batch": args.batch_name,
                        "tag": texture,
                    },
                    files={"file": (path.name, fh, "image/jpeg")},
                    timeout=60,
                )
            if resp.ok and resp.json().get("success") is not False:
                print(f"[{i}/{len(frames)}] {path.name} (tag={texture})")
            else:
                print(
                    f"[{i}/{len(frames)}] FAILED {path.name}: "
                    f"HTTP {resp.status_code} {resp.text[:200]}",
                    file=sys.stderr,
                )
                failed.append(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(frames)}] FAILED {path.name}: {exc}", file=sys.stderr)
            failed.append(path)

    print(f"done. uploaded {len(frames) - len(failed)}/{len(frames)}")
    if failed:
        print("failures:")
        for p in failed:
            print(f"  {p}")
        sys.exit(3)


if __name__ == "__main__":
    main()

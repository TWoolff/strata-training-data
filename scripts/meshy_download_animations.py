"""Download withSkin animation FBXs for all rigged Meshy characters.

Each rigging task includes basic walking + running animations with the mesh
baked in (withSkin). These deform correctly unlike retargeted animations.

Usage:
    python scripts/meshy_download_animations.py

Output structure:
    /Volumes/TAMWoolff/data/raw/meshy_cc0_rigged/
      rigged_{task_id}/
        rigged_{task_id}.fbx              # character rest pose (already exists)
        walking_withSkin.fbx              # walking animation with mesh
        running_withSkin.fbx              # running animation with mesh
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

API_BASE = "https://api.meshy.ai/openapi/v1"
RIGGED_DIR = Path("/Volumes/TAMWoolff/data/raw/meshy_cc0_rigged")
LOG_PATH = RIGGED_DIR / "download_log.json"


def get_api_key() -> str:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("MESHY_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    key = os.environ.get("MESHY_KEY", "")
    if not key:
        raise RuntimeError("Set MESHY_KEY in .env or environment")
    return key


def download_file(url: str, output_path: Path) -> bool:
    if output_path.exists():
        return True  # already downloaded
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(resp.content)
        return True
    except Exception as exc:
        print(f"    Download failed: {exc}")
        return False


def main() -> None:
    api_key = get_api_key()

    # Load task IDs from download log
    log_data = json.loads(LOG_PATH.read_text())
    task_ids = [
        entry["id"] for entry in log_data
        if entry.get("status") == "SUCCEEDED"
    ]
    print(f"Found {len(task_ids)} succeeded rigging tasks")

    downloaded = 0
    skipped = 0
    failed = 0

    for i, task_id in enumerate(task_ids):
        folder = RIGGED_DIR / f"rigged_{task_id}"
        walking_path = folder / "walking_withSkin.fbx"
        running_path = folder / "running_withSkin.fbx"

        # Skip if both already downloaded
        if walking_path.exists() and running_path.exists():
            skipped += 1
            continue

        # Query API for animation URLs
        try:
            resp = requests.get(
                f"{API_BASE}/rigging/{task_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"[{i+1}/{len(task_ids)}] {task_id}: API error: {exc}")
            failed += 1
            continue

        result = data.get("result", {})
        anims = result.get("basic_animations", {})

        walking_url = anims.get("walking_fbx_url", "")
        running_url = anims.get("running_fbx_url", "")

        if not walking_url and not running_url:
            print(f"[{i+1}/{len(task_ids)}] {task_id}: no animation URLs")
            failed += 1
            continue

        ok = True
        if walking_url and not walking_path.exists():
            ok = ok and download_file(walking_url, walking_path)
        if running_url and not running_path.exists():
            ok = ok and download_file(running_url, running_path)

        if ok:
            downloaded += 1
            if (downloaded) % 25 == 0:
                print(f"[{i+1}/{len(task_ids)}] Downloaded {downloaded} so far...")
        else:
            failed += 1

        # Brief pause to be polite to API
        time.sleep(0.2)

    print(f"\nDone: {downloaded} downloaded, {skipped} already had, {failed} failed")
    print(f"Total characters with animations: {downloaded + skipped}")


if __name__ == "__main__":
    main()

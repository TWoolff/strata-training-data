"""Download already-rigged models from Meshy API (no credits needed).

Fetches all completed rigging tasks and downloads the rigged FBX + GLB files.

Usage:
    python scripts/meshy_download_rigged.py \
        --output-dir /Volumes/TAMWoolff/data/raw/meshy_cc0_rigged
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests

API_BASE = "https://api.meshy.ai/openapi/v1"


def get_api_key() -> str:
    key = os.environ.get("MESHY_API_KEY", "") or os.environ.get("MESHY_KEY", "")
    if not key:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("MESHY_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"')
                    break
    if not key:
        raise RuntimeError("Set MESHY_API_KEY env var or add MESHY_KEY to .env")
    return key


def list_all_tasks(api_key: str) -> list[dict]:
    """Paginate through all rigging tasks."""
    headers = {"Authorization": f"Bearer {api_key}"}
    all_tasks = []
    page = 1
    while True:
        resp = requests.get(
            f"{API_BASE}/rigging?page_num={page}&page_size=50",
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_tasks.extend(data)
        if len(data) < 50:
            break
        page += 1
    return all_tasks


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    output_path.write_bytes(resp.content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download already-rigged models from Meshy API"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for rigged files",
    )
    args = parser.parse_args()

    api_key = get_api_key()
    print("Fetching rigging task list...")
    tasks = list_all_tasks(api_key)
    succeeded = [t for t in tasks if t["status"] == "SUCCEEDED"]
    print(f"Found {len(succeeded)} completed rigging tasks ({len(tasks)} total)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    failed = 0

    for i, task in enumerate(succeeded):
        task_id = task["id"]
        # Use task ID as folder name (we don't have original model names)
        folder_name = f"rigged_{task_id}"
        result = task.get("result", {})
        fbx_url = result.get("rigged_character_fbx_url", "")
        glb_url = result.get("rigged_character_glb_url", "")

        fbx_path = args.output_dir / folder_name / f"{folder_name}.fbx"
        glb_path = args.output_dir / folder_name / f"{folder_name}.glb"

        if fbx_path.exists():
            skipped += 1
            continue

        print(f"[{i+1}/{len(succeeded)}] Downloading {task_id}...", end=" ")

        try:
            if fbx_url:
                download_file(fbx_url, fbx_path)
            if glb_url:
                download_file(glb_url, glb_path)
            downloaded += 1
            size_mb = fbx_path.stat().st_size / 1024 / 1024 if fbx_path.exists() else 0
            print(f"OK ({size_mb:.1f} MB)")
        except Exception as exc:
            failed += 1
            print(f"FAIL: {exc}")

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed")

    # Save task metadata for reference
    log_path = args.output_dir / "download_log.json"
    log_data = [
        {"id": t["id"], "status": t["status"], "created_at": t.get("created_at")}
        for t in tasks
    ]
    log_path.write_text(json.dumps(log_data, indent=2) + "\n")
    print(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()

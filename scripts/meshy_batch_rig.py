"""Batch rig GLB models via the Meshy Auto-Rigging API.

Converts unrigged GLB files to rigged FBX by:
1. Uploading each GLB as a data URI
2. Submitting a rigging task
3. Polling until complete
4. Downloading the rigged FBX

Usage:
    export MESHY_API_KEY='msy_...'
    python scripts/meshy_batch_rig.py \
        --input-dir /Volumes/TAMWoolff/data/raw/meshy_cc0_glb \
        --output-dir /Volumes/TAMWoolff/data/raw/meshy_cc0_rigged \
        --limit 200
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

API_BASE = "https://api.meshy.ai/openapi/v1"
POLL_INTERVAL = 5  # seconds between status checks
MAX_POLL_TIME = 300  # 5 min max per model
MAX_CONCURRENT = 5  # concurrent rigging tasks


@dataclass
class RigResult:
    name: str
    status: str  # "succeeded", "failed", "skipped"
    task_id: str = ""
    error: str = ""


def get_api_key() -> str:
    key = os.environ.get("MESHY_API_KEY", "") or os.environ.get("MESHY_KEY", "")
    if not key:
        # Try reading from .env file
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("MESHY_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"')
                    break
    if not key:
        raise RuntimeError("Set MESHY_API_KEY env var or add MESHY_KEY to .env")
    return key


def glb_to_data_uri(glb_path: Path) -> str:
    raw = glb_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:application/octet-stream;base64,{b64}"


def submit_rig(api_key: str, glb_path: Path, height: float = 1.7) -> str:
    """Submit a rigging task. Returns task_id."""
    data_uri = glb_to_data_uri(glb_path)

    resp = requests.post(
        f"{API_BASE}/rigging",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model_url": data_uri,
            "height_meters": height,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["result"]


def poll_task(api_key: str, task_id: str) -> dict:
    """Poll until task completes or fails."""
    start = time.time()
    while time.time() - start < MAX_POLL_TIME:
        resp = requests.get(
            f"{API_BASE}/rigging/{task_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status", "")
        if status == "SUCCEEDED":
            return data
        if status in ("FAILED", "EXPIRED"):
            raise RuntimeError(f"Task {task_id} {status}: {data.get('task_error', {})}")

        progress = data.get("progress", 0)
        print(f"    ... {status} {progress}%", end="\r")
        time.sleep(POLL_INTERVAL)

    raise TimeoutError(f"Task {task_id} timed out after {MAX_POLL_TIME}s")


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    output_path.write_bytes(resp.content)


def process_one(api_key: str, glb_path: Path, output_dir: Path) -> RigResult:
    name = glb_path.parent.name
    rigged_fbx = output_dir / name / f"{name}_rigged.fbx"
    rigged_glb = output_dir / name / f"{name}_rigged.glb"

    if rigged_fbx.exists():
        return RigResult(name=name, status="skipped")

    # Submit
    try:
        task_id = submit_rig(api_key, glb_path)
        print(f"    Task: {task_id}")
    except Exception as exc:
        return RigResult(name=name, status="failed", error=f"submit: {exc}")

    # Poll
    try:
        result = poll_task(api_key, task_id)
    except Exception as exc:
        return RigResult(name=name, status="failed", task_id=task_id,
                         error=f"poll: {exc}")

    # Download rigged files — URLs are nested under result{}
    try:
        result_data = result.get("result", result)
        fbx_url = result_data.get("rigged_character_fbx_url", "")
        glb_url = result_data.get("rigged_character_glb_url", "")

        if fbx_url:
            download_file(fbx_url, rigged_fbx)
        if glb_url:
            download_file(glb_url, rigged_glb)

        if not fbx_url and not glb_url:
            return RigResult(name=name, status="failed", task_id=task_id,
                             error="No rigged file URLs in response")

        return RigResult(name=name, status="succeeded", task_id=task_id)
    except Exception as exc:
        return RigResult(name=name, status="failed", task_id=task_id,
                         error=f"download: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch rig GLB models via Meshy API")
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Directory with GLB subdirectories")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Output directory for rigged FBX files")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max models to process (0=all)")
    parser.add_argument("--height", type=float, default=1.7,
                        help="Approximate character height in meters")
    parser.add_argument("--log", type=Path, default=None,
                        help="JSON log file path")
    args = parser.parse_args()

    api_key = get_api_key()

    # Find all GLB files
    glb_files = sorted(args.input_dir.rglob("*.glb"))
    if args.limit > 0:
        glb_files = glb_files[:args.limit]

    print(f"Found {len(glb_files)} GLB files")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[RigResult] = []
    credits_used = 0

    for i, glb in enumerate(glb_files):
        name = glb.parent.name
        print(f"[{i+1}/{len(glb_files)}] {name}")

        result = process_one(api_key, glb, args.output_dir)
        results.append(result)

        if result.status == "succeeded":
            credits_used += 5
            print(f"    OK ({credits_used} credits used)")
        elif result.status == "skipped":
            print(f"    SKIP (already rigged)")
        else:
            print(f"    FAIL: {result.error}")

        # Brief pause between submissions to be polite to the API
        if result.status != "skipped":
            time.sleep(1)

    # Summary
    succeeded = sum(1 for r in results if r.status == "succeeded")
    failed = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status == "skipped")
    print(f"\nDone: {succeeded} rigged, {failed} failed, {skipped} skipped")
    print(f"Credits used: ~{credits_used}")

    # Save log
    log_path = args.log or args.output_dir / "rig_log.json"
    log_data = [
        {"name": r.name, "status": r.status, "task_id": r.task_id, "error": r.error}
        for r in results
    ]
    log_path.write_text(json.dumps(log_data, indent=2) + "\n")
    print(f"Log saved: {log_path}")

    # List failures for retry
    if failed > 0:
        print(f"\nFailed models ({failed}):")
        for r in results:
            if r.status == "failed":
                print(f"  {r.name}: {r.error}")


if __name__ == "__main__":
    main()

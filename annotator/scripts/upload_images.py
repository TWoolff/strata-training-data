#!/usr/bin/env python3
"""Upload images + pseudo-labels from preprocessed dataset to Hetzner bucket + Turso DB.

Usage:
    python upload_images.py \
        --dataset gemini_diverse \
        --source-dir /Volumes/TAMWoolff/data/preprocessed/gemini_diverse \
        --limit 500 \
        --priority-boost-classes neck,forearm_l,forearm_r,accessory
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

# Turso HTTP API endpoint derived from libsql URL
TURSO_URL = os.environ.get("TURSO_DATABASE_URL", "")
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "")

BUCKET_NAME = "strata-training-data"
BUCKET_ENDPOINT = "fsn1.your-objectstorage.com"
RCLONE_REMOTE = "hetzner"

# Strata 22-class region names (index = region ID)
REGION_NAMES = [
    "background",
    "head",
    "neck",
    "chest",
    "spine",
    "hips",
    "shoulder_l",
    "upper_arm_l",
    "forearm_l",
    "hand_l",
    "shoulder_r",
    "upper_arm_r",
    "forearm_r",
    "hand_r",
    "upper_leg_l",
    "lower_leg_l",
    "foot_l",
    "upper_leg_r",
    "lower_leg_r",
    "foot_r",
    "accessory",
    "hair_back",
]


def turso_http_url() -> str:
    """Convert libsql:// URL to HTTPS endpoint for HTTP API."""
    url = TURSO_URL
    if url.startswith("libsql://"):
        url = url.replace("libsql://", "https://")
    return url


def turso_execute(sql: str, args: list | None = None) -> dict:
    """Execute a SQL statement via Turso HTTP API."""
    url = f"{turso_http_url()}/v2/pipeline"
    headers = {"Authorization": f"Bearer {TURSO_TOKEN}", "Content-Type": "application/json"}
    stmt: dict = {"sql": sql}
    if args:
        stmt["args"] = [
            {"type": "text", "value": str(a)}
            if isinstance(a, str)
            else {"type": "integer", "value": str(a)}
            for a in args
        ]
    body = {"requests": [{"type": "execute", "stmt": stmt}, {"type": "close"}]}
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()


def turso_batch(statements: list[tuple[str, list]]) -> dict:
    """Execute multiple statements in a single pipeline request."""
    url = f"{turso_http_url()}/v2/pipeline"
    headers = {"Authorization": f"Bearer {TURSO_TOKEN}", "Content-Type": "application/json"}
    reqs = []
    for sql, args in statements:
        stmt: dict = {"sql": sql}
        if args:
            stmt["args"] = [
                {"type": "text", "value": str(a)}
                if isinstance(a, str)
                else {"type": "integer", "value": str(a)}
                for a in args
            ]
        reqs.append({"type": "execute", "stmt": stmt})
    reqs.append({"type": "close"})
    body = {"requests": reqs}
    resp = requests.post(url, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    return resp.json()


def get_existing_example_ids(dataset: str) -> set[str]:
    """Query Turso for example_ids already uploaded for this dataset."""
    result = turso_execute("SELECT example_id FROM images WHERE dataset = ?", [dataset])
    rows = result.get("results", [{}])[0].get("response", {}).get("result", {}).get("rows", [])
    return {row[0]["value"] for row in rows}


def compute_priority(seg_path: Path, boost_classes: list[str]) -> int:
    """Compute upload priority. Higher = shown to annotators first."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return 0

    seg = np.array(Image.open(seg_path))
    unique_ids = set(seg.flatten().tolist())

    priority = 0

    # Boost images containing weak/boosted classes
    for cls_name in boost_classes:
        if cls_name in REGION_NAMES:
            cls_id = REGION_NAMES.index(cls_name)
            if cls_id in unique_ids:
                priority += 10

    # Boost images with fewer distinct regions (likely harder / more ambiguous)
    foreground_ids = unique_ids - {0}
    if len(foreground_ids) < 6:
        priority += 5

    # If confidence.png exists, use mean confidence (lower = higher priority)
    conf_path = seg_path.parent / "confidence.png"
    if conf_path.exists():
        conf = np.array(Image.open(conf_path)).mean()
        # Scale: 0 confidence → +20 priority, 255 → +0
        priority += int(20 * (1.0 - conf / 255.0))

    return priority


def public_url(dataset: str, example_id: str, filename: str) -> str:
    """Construct the public URL for a file in the annotations bucket prefix."""
    return f"https://{BUCKET_ENDPOINT}/{BUCKET_NAME}/annotations/{dataset}/{example_id}/{filename}"


def rclone_upload(local_dir: Path, remote_prefix: str) -> bool:
    """Upload a local directory to the bucket via rclone."""
    remote = f"{RCLONE_REMOTE}:{BUCKET_NAME}/{remote_prefix}"
    cmd = [
        "rclone",
        "copy",
        str(local_dir),
        remote,
        "--transfers",
        "8",
        "--checkers",
        "16",
        "--fast-list",
        "--size-only",
        "-q",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  rclone error: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def scan_examples(source_dir: Path, limit: int | None) -> list[tuple[str, Path, Path]]:
    """Scan source dir for example_id dirs containing image.png + segmentation.png."""
    examples = []
    for entry in sorted(source_dir.iterdir()):
        if not entry.is_dir():
            continue
        img = entry / "image.png"
        seg = entry / "segmentation.png"
        if img.exists() and seg.exists():
            examples.append((entry.name, img, seg))
            if limit and len(examples) >= limit:
                break
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload images to Hetzner + Turso for annotation")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. gemini_diverse)")
    parser.add_argument(
        "--source-dir", required=True, type=Path, help="Path to preprocessed dataset"
    )
    parser.add_argument("--limit", type=int, default=None, help="Max examples to upload")
    parser.add_argument(
        "--priority-boost-classes",
        type=str,
        default="",
        help="Comma-separated region names to boost priority (e.g. neck,forearm_l,forearm_r)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be done without uploading"
    )
    parser.add_argument("--batch-size", type=int, default=50, help="DB insert batch size")
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        print(f"Error: source dir not found: {args.source_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        if not TURSO_URL or not TURSO_TOKEN:
            print("Error: TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be set", file=sys.stderr)
            sys.exit(1)
        # Verify rclone is available
        try:
            subprocess.run(["rclone", "version"], capture_output=True, check=True, timeout=10)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: rclone not found or not configured", file=sys.stderr)
            sys.exit(1)

    boost_classes = [c.strip() for c in args.priority_boost_classes.split(",") if c.strip()]

    print(f"Scanning {args.source_dir} ...")
    examples = scan_examples(args.source_dir, args.limit)
    print(f"Found {len(examples)} examples with image.png + segmentation.png")

    if not examples:
        print("Nothing to upload.")
        return

    # Check which are already in Turso
    existing = set()
    if not args.dry_run:
        print("Checking Turso for existing uploads ...")
        existing = get_existing_example_ids(args.dataset)
        print(f"  {len(existing)} already uploaded")

    new_examples = [(eid, img, seg) for eid, img, seg in examples if eid not in existing]
    print(f"{len(new_examples)} new examples to upload")

    if not new_examples:
        print("All examples already uploaded. Done.")
        return

    if args.dry_run:
        print("\n[DRY RUN] Would upload:")
        for eid, _img, seg in new_examples[:10]:
            pri = compute_priority(seg, boost_classes)
            print(f"  {eid}  priority={pri}")
        if len(new_examples) > 10:
            print(f"  ... and {len(new_examples) - 10} more")
        return

    # Upload in batches
    uploaded = 0
    failed = 0
    batch_stmts: list[tuple[str, list]] = []

    for eid, img_path, seg_path in new_examples:
        priority = compute_priority(seg_path, boost_classes)

        # Get image dimensions
        try:
            from PIL import Image

            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            w, h = 512, 512

        # Upload files to bucket
        remote_prefix = f"annotations/{args.dataset}/{eid}"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Copy files to temp dir for rclone
            import shutil

            shutil.copy2(img_path, tmp_path / "image.png")
            shutil.copy2(seg_path, tmp_path / "segmentation.png")

            if not rclone_upload(tmp_path, remote_prefix):
                print(f"  FAILED upload: {eid}")
                failed += 1
                continue

        # Queue DB insert
        image_url = public_url(args.dataset, eid, "image.png")
        seg_url = public_url(args.dataset, eid, "segmentation.png")

        batch_stmts.append(
            (
                "INSERT OR IGNORE INTO images (dataset, example_id, image_url, seg_url, width, height, priority) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [args.dataset, eid, image_url, seg_url, w, h, priority],
            )
        )

        uploaded += 1

        # Flush batch
        if len(batch_stmts) >= args.batch_size:
            turso_batch(batch_stmts)
            batch_stmts = []
            print(
                f"  [{uploaded + failed}/{len(new_examples)}] uploaded={uploaded} failed={failed}"
            )

    # Flush remaining
    if batch_stmts:
        turso_batch(batch_stmts)

    print(f"\nDone. uploaded={uploaded} failed={failed} skipped={len(existing)}")


if __name__ == "__main__":
    main()

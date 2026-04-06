#!/usr/bin/env python3
"""Export approved annotations from Turso DB to Strata training format.

Usage:
    python export_approved.py \
        --output-dir /Volumes/TAMWoolff/data/preprocessed/human_corrected \
        --format strata
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

TURSO_URL = os.environ.get("TURSO_DATABASE_URL", "")
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "")

BUCKET_NAME = "strata-training-data"
RCLONE_REMOTE = "hetzner"


def turso_http_url() -> str:
    """Convert libsql:// URL to HTTPS endpoint for HTTP API."""
    url = TURSO_URL
    if url.startswith("libsql://"):
        url = url.replace("libsql://", "https://")
    return url


def turso_execute(sql: str, args: list | None = None) -> list[dict]:
    """Execute SQL and return rows as list of dicts."""
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

    data = resp.json()
    result = data.get("results", [{}])[0].get("response", {}).get("result", {})
    cols = [c["name"] for c in result.get("cols", [])]
    rows = []
    for row in result.get("rows", []):
        rows.append({cols[i]: cell["value"] for i, cell in enumerate(row)})
    return rows


def decode_mask(mask_data: str, width: int, height: int) -> bytes:
    """Decode mask_data (base64 PNG) to raw PNG bytes.

    The annotation app stores mask_data as a base64-encoded grayscale PNG
    where pixel value = region ID (0-21).
    """
    from PIL import Image

    raw = base64.b64decode(mask_data)

    # Validate it's a valid PNG by opening it
    img = Image.open(io.BytesIO(raw))
    if img.size != (width, height):
        img = img.resize((width, height), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    return raw


def rclone_download(remote_path: str, local_path: Path) -> bool:
    """Download a single file from the bucket via rclone."""
    remote = f"{RCLONE_REMOTE}:{BUCKET_NAME}/{remote_path}"
    cmd = ["rclone", "copyto", remote, str(local_path), "-q"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Export approved annotations to Strata format")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for exported dataset",
    )
    parser.add_argument(
        "--format",
        choices=["strata"],
        default="strata",
        help="Output format (default: strata)",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset name")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without downloading")
    args = parser.parse_args()

    if not args.dry_run and (not TURSO_URL or not TURSO_TOKEN):
        print("Error: TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be set", file=sys.stderr)
        sys.exit(1)

    # Query all approved annotations
    sql = """
        SELECT
            i.dataset,
            i.example_id,
            i.image_url,
            i.width,
            i.height,
            a.mask_data,
            a.time_spent,
            a.created_at AS annotated_at,
            u.name AS annotator,
            r.created_at AS reviewed_at,
            r.notes AS review_notes
        FROM reviews r
        JOIN annotations a ON a.id = r.annotation_id
        JOIN images i ON i.id = a.image_id
        JOIN users u ON u.id = a.user_id
        WHERE r.approved = 1
    """
    filter_args: list = []
    if args.dataset:
        sql += " AND i.dataset = ?"
        filter_args.append(args.dataset)
    sql += " ORDER BY i.dataset, i.example_id"

    print("Querying Turso for approved annotations ...")
    rows = turso_execute(sql, filter_args if filter_args else None)
    print(f"Found {len(rows)} approved annotations")

    if not rows:
        print("Nothing to export.")
        return

    # Stats
    by_dataset: dict[str, int] = defaultdict(int)
    by_annotator: dict[str, int] = defaultdict(int)
    for row in rows:
        by_dataset[row["dataset"]] += 1
        by_annotator[row["annotator"]] += 1

    print("\nPer-dataset breakdown:")
    for ds, count in sorted(by_dataset.items()):
        print(f"  {ds}: {count}")
    print("\nPer-annotator breakdown:")
    for ann, count in sorted(by_annotator.items(), key=lambda x: -x[1]):
        print(f"  {ann}: {count}")

    if args.dry_run:
        print("\n[DRY RUN] Would export to:", args.output_dir)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    failed = 0

    for row in rows:
        example_id = row["example_id"]
        dataset = row["dataset"]
        width = int(row["width"])
        height = int(row["height"])
        example_dir = args.output_dir / example_id
        example_dir.mkdir(parents=True, exist_ok=True)

        # Download original image from bucket
        # Extract remote path from image_url
        image_url = row["image_url"]
        remote_path = (
            image_url.split(f"{BUCKET_NAME}/", 1)[-1] if BUCKET_NAME in image_url else None
        )
        img_path = example_dir / "image.png"

        if remote_path and not img_path.exists() and not rclone_download(remote_path, img_path):
            print(f"  FAILED download: {example_id}/image.png")
            failed += 1
            continue

        # Decode and save corrected segmentation mask
        seg_path = example_dir / "segmentation.png"
        try:
            mask_bytes = decode_mask(row["mask_data"], width, height)
            seg_path.write_bytes(mask_bytes)
        except Exception as e:
            print(f"  FAILED decode mask: {example_id}: {e}")
            failed += 1
            continue

        # Write metadata
        metadata = {
            "dataset": dataset,
            "example_id": example_id,
            "segmentation_source": "human_corrected",
            "review_status": "approved",
            "annotator": row["annotator"],
            "annotated_at": row["annotated_at"],
            "reviewed_at": row["reviewed_at"],
            "review_notes": row.get("review_notes"),
            "time_spent_seconds": int(row["time_spent"]) if row.get("time_spent") else None,
            "width": width,
            "height": height,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = example_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

        exported += 1
        if exported % 50 == 0:
            print(f"  [{exported}/{len(rows)}] exported")

    print(f"\nDone. exported={exported} failed={failed}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()

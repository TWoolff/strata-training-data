#!/usr/bin/env python3
"""Copy only reviewed (accepted) examples to a clean output directory.

Reads review_manifest.json and copies examples with status="reviewed"
to a new directory, ready for tarring and upload to the training bucket.

Usage::

    python scripts/filter_reviewed.py \
        --input-dir ./output/gemini_corrected \
        --output-dir ./output/gemini_corrected_clean

    # Then tar and upload:
    tar cf gemini_corrected.tar -C ./output gemini_corrected_clean
    rclone copy gemini_corrected.tar hetzner:strata-training-data/tars/ -P
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy reviewed examples to clean output directory."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status", type=str, default="reviewed",
                        help="Status to filter for (default: reviewed)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    manifest_path = args.input_dir / "review_manifest.json"
    if not manifest_path.exists():
        logger.error("No review_manifest.json found in %s", args.input_dir)
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = manifest.get("examples", {})

    matching = [name for name, info in examples.items() if info.get("status") == args.status]
    logger.info("Found %d examples with status=%s (of %d total)", len(matching), args.status, len(examples))

    if not matching:
        logger.warning("No examples to copy.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for name in sorted(matching):
        src = args.input_dir / name
        dst = args.output_dir / name
        if not src.is_dir():
            logger.warning("  SKIP: %s (directory not found)", name)
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        copied += 1

    print(f"\nFiltered {copied} examples with status={args.status}")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  tar cf gemini_corrected.tar -C {args.output_dir.parent} {args.output_dir.name}")
    print(f"  rclone copy gemini_corrected.tar hetzner:strata-training-data/tars/ -P")


if __name__ == "__main__":
    main()

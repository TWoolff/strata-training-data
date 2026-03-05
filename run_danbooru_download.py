"""Download Danbooru diverse images for all presets.

Usage:
    python run_danbooru_download.py \
        --raw_dir /Volumes/TAMWoolff/data/preprocessed/danbooru_diverse_raw \
        --output_dir /Volumes/TAMWoolff/data/preprocessed/danbooru_diverse \
        --max_images 10000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure repo root is importable.
repo_root = str(Path(__file__).resolve().parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from ingest.danbooru_diverse_adapter import (
    TAG_PRESETS,
    convert_directory,
    download_preset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Danbooru diverse images")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("/Volumes/TAMWoolff/data/preprocessed/danbooru_diverse_raw"),
        help="Directory for raw downloaded images",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/Volumes/TAMWoolff/data/preprocessed/danbooru_diverse"),
        help="Directory for Strata-formatted output",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=10000,
        help="Max images per preset",
    )
    parser.add_argument(
        "--presets",
        nargs="*",
        default=list(TAG_PRESETS.keys()),
        help="Presets to download (default: all)",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download, only run conversion",
    )
    parser.add_argument(
        "--skip_convert",
        action="store_true",
        help="Skip conversion, only download",
    )
    args = parser.parse_args()

    total_downloaded = 0
    start = time.time()

    if not args.skip_download:
        for preset in args.presets:
            logger.info("=== Downloading preset: %s ===", preset)
            t0 = time.time()
            count = download_preset(preset, args.raw_dir, max_images=args.max_images)
            elapsed = time.time() - t0
            total_downloaded += count
            logger.info(
                "Preset %s: %d images in %.1f min", preset, count, elapsed / 60
            )

        dl_elapsed = time.time() - start
        logger.info(
            "Download complete: %d total images in %.1f hours",
            total_downloaded,
            dl_elapsed / 3600,
        )

    if not args.skip_convert:
        logger.info("=== Converting to Strata format ===")
        t0 = time.time()
        result = convert_directory(args.raw_dir, args.output_dir)
        elapsed = time.time() - t0
        logger.info(
            "Conversion complete: %d processed, %d skipped in %.1f min",
            result.images_processed,
            result.images_skipped,
            elapsed / 60,
        )

    total_elapsed = time.time() - start
    logger.info("Total time: %.1f hours", total_elapsed / 3600)


if __name__ == "__main__":
    main()

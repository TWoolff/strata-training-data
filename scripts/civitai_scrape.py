"""Scrape non-anime full-body character illustrations from Civitai.

Two-phase approach:
1. Find LoRA/Checkpoint models tagged for character/concept art
2. Download their gallery images, filtering out anime/NSFW/photos

All images on Civitai are AI-generated (Stable Diffusion / Flux / etc.)
with no copyright holder — safe for AI training.

Usage::

    # Download ~1500 images:
    python scripts/civitai_scrape.py \
        --output-dir /Volumes/TAMWoolff/data/raw/civitai_characters \
        --max-images 1500

    # Quick test (20 images):
    python scripts/civitai_scrape.py \
        --output-dir ./output/civitai_test \
        --max-images 20

    # Resume after interruption (skips existing images):
    python scripts/civitai_scrape.py \
        --output-dir /Volumes/TAMWoolff/data/raw/civitai_characters \
        --max-images 1500
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Civitai API
# ---------------------------------------------------------------------------

BASE_URL = "https://civitai.com/api/v1"
HEADERS = {"Content-Type": "application/json"}

# ---------------------------------------------------------------------------
# Model discovery — search tags that produce character concept art
# ---------------------------------------------------------------------------

MODEL_SEARCH_TAGS = [
    "character art",
    "concept art",
    "character design",
    "fantasy art",
    "digital painting",
    "illustration",
    "character sheet",
    "rpg",
    "dnd",
    "d&d",
]

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

ANIME_KEYWORDS = {
    "anime", "manga", "waifu", "hentai", "ecchi", "chibi",
    "kawaii", "otaku", "loli", "shonen", "seinen", "isekai",
    "neko", "kemono", "ahegao", "oppai", "tsundere", "yandere",
    "naruto", "dragon ball", "one piece", "genshin",
    "honkai", "azur lane", "fate/", "touhou",
    "vtuber", "hololive", "anime style", "anime art",
    "counterfeit", "anything v", "aom3", "abyssorangemix",
    "meinamix", "pastelmix", "animagine",
}

ANIME_MODEL_TAGS = {"anime", "waifu", "manga", "hentai"}

NSFW_KEYWORDS = {
    "nsfw", "nude", "naked", "topless", "bottomless",
    "explicit", "xxx", "porn", "erotic", "lewd",
    "uncensored", "nipple", "genitalia",
}

# Images must have portrait-ish aspect ratio (taller than wide)
# to be likely full-body character art
MIN_ASPECT_RATIO = 1.1  # height/width


def _skip_image(item: dict) -> str | None:
    """Return skip reason or None if image should be downloaded."""
    meta = item.get("meta") or {}
    prompt = (meta.get("prompt") or "").lower()

    # If there's a prompt, check for anime/NSFW keywords
    if prompt:
        for kw in ANIME_KEYWORDS:
            if kw in prompt:
                return "anime"
        for kw in NSFW_KEYWORDS:
            if kw in prompt:
                return "nsfw"

    # Check aspect ratio — character art is usually portrait orientation
    w = item.get("width", 0)
    h = item.get("height", 0)
    if w > 0 and h > 0:
        ratio = h / w
        if ratio < MIN_ASPECT_RATIO:
            return "landscape"

    # Skip videos
    if item.get("type") != "image":
        return "not_image"

    return None


def _is_anime_model(model: dict) -> bool:
    """Check if a model is anime-focused by its tags."""
    tags = {t.lower() for t in model.get("tags", [])}
    return bool(tags & ANIME_MODEL_TAGS)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _api_get(endpoint: str, params: dict | None = None) -> dict:
    """Make a GET request to the Civitai API."""
    url = f"{BASE_URL}/{endpoint}"
    if params:
        url += "?" + urlencode(params)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_image(url: str, output_path: Path) -> bool:
    """Download a single image. Returns True on success."""
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        output_path.write_bytes(resp.content)
        return True
    except Exception as e:
        logger.warning("Failed to download %s: %s", url, e)
        return False


# ---------------------------------------------------------------------------
# Phase 1: Discover character art models
# ---------------------------------------------------------------------------


def discover_models(max_models: int = 200) -> list[int]:
    """Find model IDs that produce character concept art."""
    model_ids: set[int] = set()

    for tag in MODEL_SEARCH_TAGS:
        if len(model_ids) >= max_models:
            break

        logger.info("Searching models with tag: %s", tag)

        for model_type in ["LORA", "Checkpoint"]:
            try:
                data = _api_get("models", {
                    "tag": tag,
                    "types": model_type,
                    "sort": "Most Downloaded",
                    "nsfw": "false",
                    "limit": 20,
                })
            except Exception as e:
                logger.warning("Failed to search models for tag=%s: %s", tag, e)
                continue

            for model in data.get("items", []):
                if _is_anime_model(model):
                    continue
                model_ids.add(model["id"])

            time.sleep(0.3)

    logger.info("Discovered %d character art models", len(model_ids))
    return sorted(model_ids)


# ---------------------------------------------------------------------------
# Phase 2: Download images from model galleries
# ---------------------------------------------------------------------------


def scrape_civitai(
    output_dir: Path,
    max_images: int = 1500,
    delay: float = 0.3,
    max_models: int = 200,
) -> dict:
    """Scrape character images from Civitai model galleries.

    Args:
        output_dir: Directory to save downloaded images + manifest.
        max_images: Maximum number of images to download.
        delay: Seconds between API calls.
        max_models: Maximum models to search.

    Returns:
        Dict with download statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    # Load existing manifest for resume
    existing: dict[str, dict] = {}
    if manifest_path.exists():
        items = json.loads(manifest_path.read_text())
        existing = {item["filename"]: item for item in items}
        logger.info("Resuming — %d images already downloaded", len(existing))

    manifest_items: list[dict] = list(existing.values())
    seen_ids: set[int] = {item.get("civitai_id", 0) for item in existing.values()}

    stats = {
        "downloaded": 0,
        "skipped_anime": 0,
        "skipped_nsfw": 0,
        "skipped_no_prompt": 0,
        "skipped_landscape": 0,
        "skipped_existing": 0,
        "failed": 0,
        "models_searched": 0,
        "total_scanned": 0,
    }

    total_downloaded = len(existing)

    # Phase 1: Discover models
    model_ids = discover_models(max_models)

    # Phase 2: Download from each model's gallery
    for model_id in model_ids:
        if total_downloaded >= max_images:
            break

        stats["models_searched"] += 1
        cursor = None
        pages_for_model = 0
        max_pages_per_model = 5  # Don't go too deep per model

        while total_downloaded < max_images and pages_for_model < max_pages_per_model:
            try:
                params = {
                    "modelId": model_id,
                    "limit": 100,
                    "nsfw": "None",
                    "sort": "Most Reactions",
                }
                if cursor:
                    params["cursor"] = cursor

                data = _api_get("images", params)
            except Exception as e:
                logger.warning("API error for model %d: %s", model_id, e)
                if "429" in str(e):
                    logger.warning("Rate limited — waiting 30s...")
                    time.sleep(30)
                    continue
                break

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                stats["total_scanned"] += 1
                image_id = item.get("id")

                if image_id in seen_ids:
                    stats["skipped_existing"] += 1
                    continue
                seen_ids.add(image_id)

                # Apply filters
                skip_reason = _skip_image(item)
                if skip_reason:
                    stats[f"skipped_{skip_reason}"] = stats.get(
                        f"skipped_{skip_reason}", 0
                    ) + 1
                    continue

                # Download
                image_url = item.get("url", "")
                if not image_url:
                    continue

                ext = ".jpeg"
                if ".png" in image_url.lower():
                    ext = ".png"
                elif ".webp" in image_url.lower():
                    ext = ".webp"

                filename = f"civitai_{image_id}{ext}"
                image_path = output_dir / filename

                if image_path.exists():
                    stats["skipped_existing"] += 1
                    continue

                success = download_image(image_url, image_path)
                if success:
                    stats["downloaded"] += 1
                    total_downloaded += 1

                    meta = item.get("meta") or {}
                    entry = {
                        "filename": filename,
                        "civitai_id": image_id,
                        "prompt": meta.get("prompt", ""),
                        "width": item.get("width", 0),
                        "height": item.get("height", 0),
                        "base_model": item.get("baseModel", "unknown"),
                        "username": item.get("username", ""),
                        "model_id": model_id,
                    }
                    manifest_items.append(entry)

                    # Save manifest periodically
                    if stats["downloaded"] % 20 == 0:
                        manifest_path.write_text(
                            json.dumps(manifest_items, indent=2, ensure_ascii=False)
                        )

                    if total_downloaded % 50 == 0:
                        logger.info(
                            "Progress: %d/%d downloaded (scanned %d from %d models)",
                            total_downloaded,
                            max_images,
                            stats["total_scanned"],
                            stats["models_searched"],
                        )

                    if total_downloaded >= max_images:
                        break

                    time.sleep(0.05)
                else:
                    stats["failed"] += 1

            # Next page
            metadata = data.get("metadata", {})
            next_cursor = metadata.get("nextCursor")
            if not next_cursor:
                break
            cursor = next_cursor
            pages_for_model += 1
            time.sleep(delay)

    # Final manifest save
    manifest_path.write_text(
        json.dumps(manifest_items, indent=2, ensure_ascii=False)
    )

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape non-anime character illustrations from Civitai.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save downloaded images.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=1500,
        help="Maximum images to download (default: 1500).",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=200,
        help="Maximum models to search (default: 200).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Seconds between API calls (default: 0.3).",
    )
    args = parser.parse_args()

    logger.info("Starting Civitai character scrape...")
    logger.info("Output: %s", args.output_dir)
    logger.info("Target: %d images from up to %d models", args.max_images, args.max_models)

    stats = scrape_civitai(
        output_dir=args.output_dir,
        max_images=args.max_images,
        delay=args.delay,
        max_models=args.max_models,
    )

    print("\nScrape complete:")
    print(f"  Downloaded:          {stats['downloaded']}")
    print(f"  Models searched:     {stats['models_searched']}")
    print(f"  Skipped (anime):     {stats.get('skipped_anime', 0)}")
    print(f"  Skipped (no prompt): {stats.get('skipped_no_prompt', 0)}")
    print(f"  Skipped (landscape): {stats.get('skipped_landscape', 0)}")
    print(f"  Skipped (existing):  {stats.get('skipped_existing', 0)}")
    print(f"  Failed:              {stats['failed']}")
    print(f"  Total scanned:       {stats['total_scanned']}")


if __name__ == "__main__":
    main()

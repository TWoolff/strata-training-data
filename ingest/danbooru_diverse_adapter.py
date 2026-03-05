"""Convert tag-filtered Danbooru images to Strata training format.

Mirrors the FBAnimeHQ adapter but targets underrepresented character types:
male characters, diverse skin tones, muscular body types, and western
fantasy/comic aesthetics.

This adapter downloads images from Danbooru using tag-based queries, then
converts them to Strata's per-example directory format.  Images are resized
to 512×512 with aspect-ratio-preserving padding.

No annotations are provided by the source — joints should be added via
RTMPose enrichment as a separate post-processing step (same pipeline as
FBAnimeHQ).

Download uses the Danbooru API (no authentication required for safe-rated
images).  Rate limiting is applied to stay within API guidelines.

This module is pure Python (no Blender dependency).
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DANBOORU_DIVERSE_SOURCE = "danbooru_diverse"

STRATA_RESOLUTION = 512

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# Annotations that this dataset does NOT provide.
_MISSING_ANNOTATIONS = [
    "strata_segmentation",
    "joints",
    "draw_order",
    "fg_mask",
]

# Danbooru API base URL.
DANBOORU_API_BASE = "https://danbooru.donmai.us"

# Minimum Danbooru score to filter low-quality posts.
MIN_SCORE = 10

# Rate limit: seconds between API requests.
API_DELAY_SECONDS = 1.0

# Tag presets for underrepresented categories.
# Each preset is a dict with a label and a Danbooru tag query string.
TAG_PRESETS: dict[str, str] = {
    "male_full_body": "1boy full_body solo score:>10 rating:general",
    "dark_skin": "dark_skin full_body solo score:>10 rating:general",
    "muscular": "muscular full_body solo score:>10 rating:general",
    "western_fantasy": "1boy armor full_body score:>10 rating:general",
    "male_casual": "1boy full_body solo casual score:>10 rating:general",
}

# Maximum images per API page (Danbooru limit).
API_PAGE_LIMIT = 200


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AdapterResult:
    """Result of converting Danbooru images to Strata format."""

    images_processed: int = 0
    images_skipped: int = 0
    images_downloaded: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _fetch_posts(
    tags: str,
    page: int = 1,
    limit: int = API_PAGE_LIMIT,
) -> list[dict[str, Any]]:
    """Fetch posts from Danbooru API.

    Args:
        tags: Danbooru tag query string.
        page: Page number (1-indexed).
        limit: Posts per page (max 200).

    Returns:
        List of post dicts from the API.
    """
    import urllib.parse
    import urllib.request

    params = urllib.parse.urlencode(
        {
            "tags": tags,
            "page": page,
            "limit": limit,
        }
    )
    url = f"{DANBOORU_API_BASE}/posts.json?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "StrataTrainingPipeline/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("API request failed for tags=%r page=%d: %s", tags, page, exc)
        return []


def _download_image(url: str, output_path: Path) -> bool:
    """Download an image from a URL.

    Args:
        url: Image URL.
        output_path: Local path to save the image.

    Returns:
        True on success.
    """
    import urllib.request

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "StrataTrainingPipeline/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(resp.read())
        return True
    except Exception as exc:
        logger.warning("Download failed for %s: %s", url, exc)
        return False


def download_preset(
    preset: str,
    raw_dir: Path,
    *,
    max_images: int = 1000,
) -> int:
    """Download images for a tag preset from Danbooru.

    Args:
        preset: Key from TAG_PRESETS (e.g. "male_full_body").
        raw_dir: Directory to save raw downloaded images.
        max_images: Maximum images to download.

    Returns:
        Number of images successfully downloaded.
    """
    if preset not in TAG_PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from: {list(TAG_PRESETS)}")

    tags = TAG_PRESETS[preset]
    preset_dir = raw_dir / preset
    preset_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    page = 1

    while downloaded < max_images:
        posts = _fetch_posts(tags, page=page)
        if not posts:
            break

        for post in posts:
            if downloaded >= max_images:
                break

            file_url = post.get("file_url") or post.get("large_file_url")
            if not file_url:
                continue

            post_id = post.get("id", "unknown")
            ext = Path(file_url).suffix or ".jpg"
            out_path = preset_dir / f"{post_id}{ext}"

            if out_path.exists():
                downloaded += 1
                continue

            if _download_image(file_url, out_path):
                downloaded += 1
                logger.debug("Downloaded post %s (%d/%d)", post_id, downloaded, max_images)
            else:
                logger.debug("Failed to download post %s", post_id)

            time.sleep(API_DELAY_SECONDS)

        page += 1

    logger.info(
        "Downloaded %d images for preset %r to %s",
        downloaded,
        preset,
        preset_dir,
    )
    return downloaded


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _resize_to_strata(
    img: Image.Image,
    resolution: int = STRATA_RESOLUTION,
) -> Image.Image:
    """Resize an image to *resolution*x*resolution*, preserving aspect ratio.

    The longest edge is scaled to *resolution*, then the image is
    centered on a transparent RGBA canvas.

    Args:
        img: Input image (any mode).
        resolution: Target square resolution.

    Returns:
        *resolution*x*resolution* RGBA image.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    w, h = img.size
    if w == resolution and h == resolution:
        return img

    scale = resolution / max(w, h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    offset_x = (resolution - new_w) // 2
    offset_y = (resolution - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    return canvas


def _build_metadata(
    image_id: str,
    source_path: Path,
    resolution: int,
    *,
    original_size: tuple[int, int],
    preset: str,
) -> dict[str, Any]:
    """Build Strata metadata dict for a single image.

    Args:
        image_id: Strata-format image identifier.
        source_path: Original image file path.
        resolution: Output resolution.
        original_size: ``(width, height)`` of the source image.
        preset: Tag preset label used for download.

    Returns:
        Metadata dict ready for JSON serialization.
    """
    ow, oh = original_size
    return {
        "id": image_id,
        "source": DANBOORU_DIVERSE_SOURCE,
        "source_filename": source_path.name,
        "resolution": resolution,
        "original_width": ow,
        "original_height": oh,
        "padding_applied": ow != oh,
        "preset": preset,
        "has_segmentation_mask": False,
        "has_fg_mask": False,
        "has_joints": False,
        "has_draw_order": False,
        "missing_annotations": _MISSING_ANNOTATIONS,
    }


def discover_images(input_dir: Path) -> list[Path]:
    """Discover all image files under *input_dir* recursively.

    Args:
        input_dir: Root directory containing preset subdirectories with images.

    Returns:
        Sorted list of image file paths.
    """
    if not input_dir.is_dir():
        logger.warning("Not a directory: %s", input_dir)
        return []

    paths = sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )

    logger.info("Discovered %d images in %s", len(paths), input_dir)
    return paths


def convert_image(
    image_path: Path,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
) -> bool:
    """Convert a single downloaded Danbooru image to Strata training format.

    Args:
        image_path: Path to the source image file.
        output_dir: Root output directory.
        resolution: Target square resolution.
        only_new: Skip if output already exists.

    Returns:
        True if saved, False if skipped or errored.
    """
    # Use preset subdirectory name + stem for unique ID.
    preset = image_path.parent.name
    image_id = f"{DANBOORU_DIVERSE_SOURCE}_{preset}_{image_path.stem}"

    example_dir = output_dir / image_id
    if only_new and example_dir.exists():
        return False

    try:
        img = Image.open(image_path)
        img.load()
    except OSError as exc:
        logger.warning("Failed to load image %s: %s", image_path, exc)
        return False

    original_size = img.size
    resized = _resize_to_strata(img, resolution)

    example_dir.mkdir(parents=True, exist_ok=True)

    resized.save(example_dir / "image.png", format="PNG", compress_level=6)

    metadata = _build_metadata(
        image_id,
        image_path,
        resolution,
        original_size=original_size,
        preset=preset,
    )
    meta_path = example_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return True


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
    only_new: bool = False,
    max_images: int = 0,
    random_sample: bool = False,
    seed: int = 42,
) -> AdapterResult:
    """Convert downloaded Danbooru diverse images to Strata format.

    Args:
        input_dir: Root directory containing preset subdirectories with images.
        output_dir: Root output directory for Strata-formatted examples.
        resolution: Target image resolution (square).
        only_new: Skip existing output directories.
        max_images: Maximum images to process (0 = unlimited).
        random_sample: Randomly sample from discovered images.
        seed: Random seed for reproducible sampling.

    Returns:
        :class:`AdapterResult` summarizing the conversion.
    """
    result = AdapterResult()

    image_paths = discover_images(input_dir)
    if not image_paths:
        return result

    if random_sample and max_images > 0:
        rng = random.Random(seed)
        sample_size = min(max_images, len(image_paths))
        image_paths = rng.sample(image_paths, sample_size)
    elif max_images > 0:
        image_paths = image_paths[:max_images]

    total = len(image_paths)
    logger.info("Processing %d images from %s", total, input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(image_paths):
        saved = convert_image(
            image_path,
            output_dir,
            resolution=resolution,
            only_new=only_new,
        )

        if saved:
            result.images_processed += 1
        else:
            result.images_skipped += 1

        if (i + 1) % 100 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            logger.info("Processed %d/%d images (%.1f%%)", i + 1, total, pct)

    logger.info(
        "Danbooru diverse conversion complete: %d processed, %d skipped, %d errors",
        result.images_processed,
        result.images_skipped,
        len(result.errors),
    )

    return result

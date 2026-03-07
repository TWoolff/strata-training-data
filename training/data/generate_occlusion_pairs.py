"""Generate synthetic occlusion training pairs for inpainting model.

Takes complete character images and creates (masked_input, mask, target) triplets
by applying random occlusion masks. Three strategies:

1. **Rectangular cutouts** — random rectangles (10-30% of image area)
2. **Irregular brush masks** — random walk with varying brush width
3. **Elliptical blobs** — random ellipses simulating body-part overlaps

Usage::

    python -m training.data.generate_occlusion_pairs \\
        --source-dirs ./data_cloud/fbanimehq ./data_cloud/anime_seg \\
        --output-dir ./data_cloud/inpainting_pairs \\
        --masks-per-image 3 \\
        --resolution 512

Pure Python + NumPy/PIL (no Blender dependency).
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mask generation strategies
# ---------------------------------------------------------------------------

RESOLUTION = 512
MIN_COVERAGE = 0.05  # At least 5% of character pixels masked
MAX_COVERAGE = 0.50  # At most 50% masked


def _generate_rect_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Generate 1-3 random rectangular cutouts."""
    mask = np.zeros((h, w), dtype=np.uint8)
    n_rects = rng.randint(1, 3)
    for _ in range(n_rects):
        rh = rng.randint(h // 8, h // 3)
        rw = rng.randint(w // 8, w // 3)
        y = rng.randint(0, h - rh)
        x = rng.randint(0, w - rw)
        mask[y : y + rh, x : x + rw] = 255
    return mask


def _generate_irregular_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Generate irregular brush-stroke mask via random walk."""
    mask = np.zeros((h, w), dtype=np.uint8)
    n_strokes = rng.randint(3, 8)

    for _ in range(n_strokes):
        # Random starting point
        y, x = rng.randint(0, h - 1), rng.randint(0, w - 1)
        brush_w = rng.randint(8, 40)
        length = rng.randint(30, 150)

        for _ in range(length):
            # Draw circle at current position
            yy, xx = np.ogrid[-brush_w : brush_w + 1, -brush_w : brush_w + 1]
            circle = xx * xx + yy * yy <= brush_w * brush_w
            y0 = max(0, y - brush_w)
            y1 = min(h, y + brush_w + 1)
            x0 = max(0, x - brush_w)
            x1 = min(w, x + brush_w + 1)
            cy0 = y0 - (y - brush_w)
            cy1 = cy0 + (y1 - y0)
            cx0 = x0 - (x - brush_w)
            cx1 = cx0 + (x1 - x0)
            mask[y0:y1, x0:x1][circle[cy0:cy1, cx0:cx1]] = 255

            # Random walk step
            angle = rng.uniform(0, 2 * 3.14159)
            step = rng.randint(3, 15)
            y = int(np.clip(y + step * np.sin(angle), 0, h - 1))
            x = int(np.clip(x + step * np.cos(angle), 0, w - 1))

    return mask


def _generate_ellipse_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Generate 1-3 random elliptical blobs."""
    mask = np.zeros((h, w), dtype=np.uint8)
    n_ellipses = rng.randint(1, 3)

    yy, xx = np.mgrid[0:h, 0:w]

    for _ in range(n_ellipses):
        cy = rng.randint(h // 6, 5 * h // 6)
        cx = rng.randint(w // 6, 5 * w // 6)
        ry = rng.randint(h // 10, h // 3)
        rx = rng.randint(w // 10, w // 3)
        angle = rng.uniform(0, 3.14159)

        # Rotated ellipse equation
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dy = yy - cy
        dx = xx - cx
        a = (cos_a * dx + sin_a * dy) / rx
        b = (-sin_a * dx + cos_a * dy) / ry
        ellipse = a * a + b * b <= 1.0
        mask[ellipse] = 255

    return mask


MASK_STRATEGIES = {
    "rect": _generate_rect_mask,
    "irregular": _generate_irregular_mask,
    "ellipse": _generate_ellipse_mask,
}


def generate_mask(
    h: int,
    w: int,
    character_mask: np.ndarray | None = None,
    rng: random.Random | None = None,
) -> np.ndarray | None:
    """Generate a random occlusion mask, ensuring coverage constraints.

    Args:
        h: Image height.
        w: Image width.
        character_mask: Optional binary mask of character pixels (non-transparent).
        rng: Random number generator.

    Returns:
        Binary mask (255=occluded, 0=visible), or None if constraints can't be met.
    """
    if rng is None:
        rng = random.Random()

    strategy = rng.choice(list(MASK_STRATEGIES.keys()))
    mask = MASK_STRATEGIES[strategy](h, w, rng)

    # If we have a character mask, only count coverage within character area
    if character_mask is not None:
        # Only mask character pixels (not background)
        mask = mask & character_mask
        char_pixels = character_mask.sum() // 255
        if char_pixels == 0:
            return None
        coverage = (mask > 0).sum() / char_pixels
    else:
        coverage = (mask > 0).sum() / (h * w)

    if coverage < MIN_COVERAGE or coverage > MAX_COVERAGE:
        return None

    return mask


def process_image(
    image_path: Path,
    output_dir: Path,
    example_id: str,
    mask_index: int,
    resolution: int,
    rng: random.Random,
) -> bool:
    """Generate one occlusion pair from a source image.

    Args:
        image_path: Path to source character image.
        output_dir: Root output directory.
        example_id: Unique identifier for this example.
        mask_index: Index of mask variant for this image.
        resolution: Target resolution (square).
        rng: Random number generator.

    Returns:
        True if pair was generated successfully.
    """
    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception:
        return False

    # Resize to target resolution
    img = img.resize((resolution, resolution), Image.BILINEAR)
    img_arr = np.array(img)

    # Character mask from alpha channel
    alpha = img_arr[:, :, 3]
    character_mask = np.where(alpha > 10, 255, 0).astype(np.uint8)

    # Skip mostly-empty images
    char_ratio = (character_mask > 0).sum() / (resolution * resolution)
    if char_ratio < 0.05:
        return False

    # Generate mask (retry up to 5 times for coverage constraints)
    mask = None
    for _ in range(5):
        mask = generate_mask(resolution, resolution, character_mask, rng)
        if mask is not None:
            break
    if mask is None:
        return False

    # Create masked image: zero out occluded pixels (including alpha)
    masked_img = img_arr.copy()
    occluded = mask > 0
    masked_img[occluded] = 0

    # Save triplet
    pair_id = f"{example_id}_m{mask_index:02d}"
    pair_dir = output_dir / pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(masked_img).save(pair_dir / "masked.png")
    Image.fromarray(mask).save(pair_dir / "mask.png")
    Image.fromarray(img_arr).save(pair_dir / "target.png")

    return True


def discover_images(source_dir: Path) -> list[Path]:
    """Find character images in a Strata dataset directory.

    Prefers ``image.png`` inside per-example subdirectories.  Falls back to
    all PNG/JPG files only when no ``image.png`` files are found (flat layout).
    """
    # Prefer Strata-format: each subdir has image.png
    strata_images = sorted(source_dir.glob("*/image.png"))
    if strata_images:
        return strata_images

    # Fallback: flat directory of images
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(source_dir.rglob(ext))
    return sorted(images)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic occlusion pairs for inpainting training"
    )
    parser.add_argument(
        "--source-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Source directories containing character images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for occlusion pairs",
    )
    parser.add_argument(
        "--masks-per-image",
        type=int,
        default=3,
        help="Number of mask variants per source image (default: 3)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=RESOLUTION,
        help="Target resolution (default: 512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Max source images to process (0 = all)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all source images
    all_images: list[tuple[str, Path]] = []
    for source_dir in args.source_dirs:
        if not source_dir.is_dir():
            logger.warning("Source directory not found: %s", source_dir)
            continue
        images = discover_images(source_dir)
        logger.info("Found %d images in %s", len(images), source_dir)
        for img_path in images:
            # Use relative path (including parent dirs) as example_id to avoid
            # collisions when all files are named "image.png" (e.g., FBAnimeHQ)
            rel = img_path.relative_to(source_dir)
            parts = list(rel.parent.parts) + [rel.stem]
            example_id = f"{source_dir.name}_{'_'.join(parts)}"
            all_images.append((example_id, img_path))

    if args.max_images > 0:
        rng.shuffle(all_images)
        all_images = all_images[: args.max_images]

    logger.info("Processing %d source images, %d masks each", len(all_images), args.masks_per_image)

    generated = 0
    skipped = 0

    for i, (example_id, img_path) in enumerate(all_images):
        for m in range(args.masks_per_image):
            ok = process_image(img_path, args.output_dir, example_id, m, args.resolution, rng)
            if ok:
                generated += 1
            else:
                skipped += 1

        if (i + 1) % 1000 == 0:
            logger.info("Progress: %d/%d images, %d pairs generated", i + 1, len(all_images), generated)

    logger.info("Done. Generated: %d pairs, Skipped: %d", generated, skipped)


if __name__ == "__main__":
    main()

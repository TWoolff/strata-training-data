"""Convert NOVA-Human renders to Strata training format.

Reads the NOVA-Human dataset (10.2K anime characters rendered from VRoid Hub
models) and converts to Strata's per-example subdirectory format, compatible
with the enrichment pipeline (run_enrich.py, run_depth_enrich.py).

NOVA-Human directory structure (after extracting .exe archives)::

    data/nova_human/
    ├── ortho/{0-9}/{char_id}/{front,back,left,right,top}.png
    ├── ortho_mask/{0-9}/{char_id}/{front,back,left,right}.png
    ├── rgb/{0-9}/{char_id}/{0000-0015}.png
    └── rgb_mask/{0-9}/{char_id}/{0000-0015}.png

Output format (per-example subdirectories)::

    output/nova_human/
    └── nova_human_{char_id}_ortho_front/
        ├── image.png
        ├── segmentation.png   (fg/bg binary mask)
        └── metadata.json

After conversion, run enrichment to add joints and draw order::

    python run_enrich.py --input_dir output/nova_human \
        --det_model models/yolox_m_humanart.onnx \
        --pose_model models/rtmpose_m_body7.onnx --only_missing

    python run_depth_enrich.py --input_dir output/nova_human \
        --depth_model models/depth_anything_v2_vits.onnx --only_missing

This module is pure Python (no Blender dependency).

Reference: https://huggingface.co/datasets/ljsabc/NOVA-Human-Mirror-Purged
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOVA_HUMAN_SOURCE = "nova_human"
STRATA_RESOLUTION = 512

# Ortho views to process (skip "top" — not useful for character training)
ORTHO_VIEWS = ("front", "back", "left", "right")

# Max RGB views per character (16 available)
MAX_RGB_VIEWS = 16

_MISSING_ANNOTATIONS = ["strata_segmentation", "joints", "draw_order"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resize(img: Image.Image, resolution: int) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.LANCZOS)
    return img


def _resize_mask(mask: Image.Image, resolution: int) -> Image.Image:
    if mask.mode != "L":
        mask = mask.convert("L")
    if mask.size != (resolution, resolution):
        mask = mask.resize((resolution, resolution), Image.NEAREST)
    arr = np.array(mask, dtype=np.uint8)
    binary = ((arr > 127).astype(np.uint8)) * 255
    return Image.fromarray(binary, mode="L")


def _build_metadata(
    char_id: str,
    view_name: str,
    *,
    is_ortho: bool,
    has_mask: bool,
) -> dict[str, Any]:
    return {
        "id": f"{NOVA_HUMAN_SOURCE}_{char_id}",
        "source": NOVA_HUMAN_SOURCE,
        "source_character_id": char_id,
        "view_name": view_name,
        "camera_type": "orthographic" if is_ortho else "perspective",
        "resolution": STRATA_RESOLUTION,
        "has_segmentation_mask": False,
        "has_fg_mask": has_mask,
        "has_joints": False,
        "has_draw_order": False,
        "missing_annotations": list(_MISSING_ANNOTATIONS),
    }


def _save_example(
    output_dir: Path,
    example_name: str,
    image: Image.Image,
    mask: Image.Image | None,
    metadata: dict[str, Any],
    *,
    only_new: bool = True,
) -> bool:
    """Save one example as a subdirectory with image.png + metadata.json."""
    example_dir = output_dir / example_name

    if only_new and (example_dir / "image.png").exists():
        return False

    example_dir.mkdir(parents=True, exist_ok=True)

    image.save(example_dir / "image.png", format="PNG", compress_level=6)

    if mask is not None:
        mask.save(example_dir / "segmentation.png", format="PNG", compress_level=6)

    (example_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    return True


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_characters(nova_dir: Path) -> list[tuple[str, str]]:
    """Find all character IDs and their bucket subdirectory.

    Returns list of (bucket_num, char_id) tuples.
    """
    ortho_dir = nova_dir / "ortho"
    if not ortho_dir.is_dir():
        logger.error("ortho/ directory not found in %s", nova_dir)
        return []

    characters = []
    for bucket_dir in sorted(ortho_dir.iterdir()):
        if not bucket_dir.is_dir() or bucket_dir.name.startswith((".", "_")):
            continue
        for char_dir in sorted(bucket_dir.iterdir()):
            if char_dir.is_dir():
                characters.append((bucket_dir.name, char_dir.name))

    return characters


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def convert_character(
    nova_dir: Path,
    output_dir: Path,
    bucket_num: str,
    char_id: str,
    *,
    resolution: int = STRATA_RESOLUTION,
    include_rgb: bool = False,
    only_new: bool = True,
) -> int:
    """Convert a single character. Returns number of views saved."""
    saved = 0
    prefix = f"{NOVA_HUMAN_SOURCE}_{char_id}"

    # Ortho views
    ortho_base = nova_dir / "ortho" / bucket_num / char_id
    mask_base = nova_dir / "ortho_mask" / bucket_num / char_id

    for view in ORTHO_VIEWS:
        img_path = ortho_base / f"{view}.png"
        if not img_path.is_file():
            continue

        example_name = f"{prefix}_ortho_{view}"

        try:
            img = Image.open(img_path)
            img.load()
            img = _resize(img, resolution)

            mask = None
            has_mask = False
            mask_path = mask_base / f"{view}.png"
            if mask_path.is_file():
                mask = Image.open(mask_path)
                mask.load()
                mask = _resize_mask(mask, resolution)
                has_mask = True

            meta = _build_metadata(char_id, f"ortho_{view}", is_ortho=True, has_mask=has_mask)

            if _save_example(output_dir, example_name, img, mask, meta, only_new=only_new):
                saved += 1
        except OSError as exc:
            logger.warning("Failed to process %s/%s: %s", char_id, view, exc)

    # RGB views
    if include_rgb:
        rgb_base = nova_dir / "rgb" / bucket_num / char_id
        rgb_mask_base = nova_dir / "rgb_mask" / bucket_num / char_id

        if rgb_base.is_dir():
            for i in range(MAX_RGB_VIEWS):
                img_path = rgb_base / f"{i:04d}.png"
                if not img_path.is_file():
                    continue

                example_name = f"{prefix}_rgb_{i:02d}"

                try:
                    img = Image.open(img_path)
                    img.load()
                    img = _resize(img, resolution)

                    mask = None
                    has_mask = False
                    mask_path = rgb_mask_base / f"{i:04d}.png"
                    if mask_path.is_file():
                        mask = Image.open(mask_path)
                        mask.load()
                        mask = _resize_mask(mask, resolution)
                        has_mask = True

                    meta = _build_metadata(char_id, f"rgb_{i:02d}", is_ortho=False, has_mask=has_mask)

                    if _save_example(output_dir, example_name, img, mask, meta, only_new=only_new):
                        saved += 1
                except OSError as exc:
                    logger.warning("Failed to process %s/rgb_%04d: %s", char_id, i, exc)

    return saved


def convert_all(
    nova_dir: Path,
    output_dir: Path,
    *,
    resolution: int = STRATA_RESOLUTION,
    include_rgb: bool = False,
    only_new: bool = True,
    max_characters: int = 0,
) -> tuple[int, int]:
    """Convert all NOVA-Human characters.

    Returns (characters_processed, total_views_saved).
    """
    characters = discover_characters(nova_dir)
    if not characters:
        logger.error("No characters found in %s", nova_dir)
        return 0, 0

    if max_characters > 0:
        characters = characters[:max_characters]

    logger.info("Found %d characters to process", len(characters))

    total_saved = 0
    processed = 0

    for i, (bucket_num, char_id) in enumerate(characters):
        views = convert_character(
            nova_dir, output_dir, bucket_num, char_id,
            resolution=resolution,
            include_rgb=include_rgb,
            only_new=only_new,
        )
        total_saved += views
        processed += 1

        if (i + 1) % 500 == 0:
            logger.info(
                "Progress: %d/%d characters, %d views saved",
                i + 1, len(characters), total_saved,
            )

    logger.info(
        "Done: %d characters processed, %d views saved",
        processed, total_saved,
    )
    return processed, total_saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert NOVA-Human to Strata format")
    parser.add_argument("--input_dir", type=Path, default=Path("data/nova_human"))
    parser.add_argument("--output_dir", type=Path, default=Path("output/nova_human"))
    parser.add_argument("--resolution", type=int, default=STRATA_RESOLUTION)
    parser.add_argument("--include_rgb", action="store_true",
                        help="Include 16 perspective views per character")
    parser.add_argument("--max_characters", type=int, default=0,
                        help="Limit characters (0=all)")
    parser.add_argument("--only_new", action="store_true", default=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    chars, views = convert_all(
        args.input_dir,
        args.output_dir,
        resolution=args.resolution,
        include_rgb=args.include_rgb,
        only_new=args.only_new,
        max_characters=args.max_characters,
    )

    print(f"\nConverted {chars} characters, {views} views saved to {args.output_dir}")


if __name__ == "__main__":
    main()

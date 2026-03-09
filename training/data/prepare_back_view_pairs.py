"""Extract paired front + three-quarter + back views for back view generation training.

Scans multi-angle render datasets and creates paired training examples by
grouping views of the same character+pose together.

Supports two dataset layouts:

1. **Flat layout** (e.g. ``meshy_cc0_textured/images/``)::

       {char}_pose_{nn}_{angle}_{style}.png
       {char}_pose_{nn}_{style}.png          (front — no angle suffix)

2. **Per-example layout** (e.g. ``meshy_cc0_unrigged/``)::

       {char}_texture_{angle}/image.png

Output structure::

    back_view_pairs/
        pair_00000/
            front.png
            three_quarter.png
            back.png
        pair_00001/
            ...

Usage::

    python training/data/prepare_back_view_pairs.py \\
        --source-dir /path/to/meshy_cc0_textured \\
        --output-dir data/training/back_view_pairs/ \\
        --style textured

Pure Python + PIL (no Blender dependency).
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("prepare_back_view_pairs")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_RESOLUTION: int = 512

# Angles we need for each triplet
REQUIRED_ANGLES = ("front", "three_quarter", "back")

# Style suffixes to strip when parsing flat-layout filenames
STYLE_SUFFIXES = re.compile(r"_(flat|cel|pixel|painterly|sketch|unlit|textured)$")

# Flat layout: extract angle from filename like
#   rigged_xxx_pose_00_three_quarter_textured.png  → angle="three_quarter"
#   rigged_xxx_pose_00_textured.png                → angle="front" (no angle = front)
_FLAT_PATTERN = re.compile(
    r"^(?P<char>.+?)_pose_(?P<pose>\d+)"
    r"(?:_(?P<angle>front_22|front_high|front_low|three_quarter_67|three_quarter_high"
    r"|three_quarter_low|three_quarter_left|three_quarter_back_157"
    r"|three_quarter_back_left|three_quarter_back|three_quarter"
    r"|side_112|side_high|side_low|side_left|side"
    r"|iso_front_left|iso_front_right|iso_front"
    r"|iso_back_left|iso_back_right|iso_back"
    r"|iso_left|iso_right"
    r"|back_high|back"
    r"|topdown_front|topdown_back|topdown_left|topdown_right))?"
    r"_(?P<style>flat|cel|pixel|painterly|sketch|unlit|textured)$"
)

# Per-example layout: extract angle from directory name like
#   Meshy_AI_xxx_texture_back → angle="back"
_PER_EXAMPLE_PATTERN = re.compile(r"^(?P<char>.+?)_texture_(?P<angle>.+)$")


# ---------------------------------------------------------------------------
# Layout detection and scanning
# ---------------------------------------------------------------------------


def _detect_layout(source_dir: Path) -> str:
    """Detect whether source uses flat or per-example layout."""
    images_dir = source_dir / "images"
    if images_dir.is_dir() and any(images_dir.glob("*.png")):
        return "flat"
    # Check for per-example dirs with image.png
    for child in source_dir.iterdir():
        if child.is_dir() and (child / "image.png").exists():
            return "per_example"
    raise ValueError(f"Cannot detect layout in {source_dir}")


def _scan_flat_layout(
    source_dir: Path,
    style_filter: str | None = None,
) -> dict[str, dict[str, Path]]:
    """Scan flat-layout dataset, returning {char_pose: {angle: path}}.

    Args:
        source_dir: Dataset root containing ``images/`` subdirectory.
        style_filter: Only include images with this style suffix (e.g. "textured").
    """
    images_dir = source_dir / "images"
    groups: dict[str, dict[str, Path]] = defaultdict(dict)

    for img_path in sorted(images_dir.glob("*.png")):
        m = _FLAT_PATTERN.match(img_path.stem)
        if not m:
            continue

        style = m.group("style")
        if style_filter and style != style_filter:
            continue

        char = m.group("char")
        pose = m.group("pose")
        angle = m.group("angle") or "front"  # No angle suffix = front

        # Normalize angle names to our canonical set
        canonical = _normalize_angle(angle)
        if canonical is None:
            continue

        key = f"{char}_pose_{pose}"
        groups[key][canonical] = img_path

    return dict(groups)


def _scan_per_example_layout(source_dir: Path) -> dict[str, dict[str, Path]]:
    """Scan per-example-layout dataset, returning {char: {angle: path}}."""
    groups: dict[str, dict[str, Path]] = defaultdict(dict)

    for child in sorted(source_dir.iterdir()):
        if not child.is_dir():
            continue
        img_path = child / "image.png"
        if not img_path.exists():
            continue

        m = _PER_EXAMPLE_PATTERN.match(child.name)
        if not m:
            continue

        char = m.group("char")
        angle = m.group("angle")

        canonical = _normalize_angle(angle)
        if canonical is None:
            continue

        groups[char][canonical] = img_path

    return dict(groups)


def _normalize_angle(angle: str) -> str | None:
    """Map raw angle names to canonical names used in triplets.

    Returns:
        One of "front", "three_quarter", "back", or None if not needed.
    """
    if angle == "front":
        return "front"
    if angle == "three_quarter":
        return "three_quarter"
    if angle == "back":
        return "back"
    return None


# ---------------------------------------------------------------------------
# Pair extraction
# ---------------------------------------------------------------------------


def extract_pairs(
    source_dir: Path,
    output_dir: Path,
    *,
    style_filter: str | None = None,
    resolution: int = TARGET_RESOLUTION,
) -> int:
    """Extract front + three_quarter + back triplets from a multi-angle dataset.

    Args:
        source_dir: Dataset root directory.
        output_dir: Where to write paired examples.
        style_filter: For flat layout, only use this style (e.g. "textured").
        resolution: Target resolution (resizes with Lanczos if needed).

    Returns:
        Number of pairs extracted.
    """
    layout = _detect_layout(source_dir)
    logger.info("Detected %s layout in %s", layout, source_dir)

    if layout == "flat":
        groups = _scan_flat_layout(source_dir, style_filter=style_filter)
    else:
        groups = _scan_per_example_layout(source_dir)

    logger.info("Found %d character-pose groups", len(groups))

    # Filter to groups that have all 3 required angles
    complete = {k: v for k, v in groups.items() if all(a in v for a in REQUIRED_ANGLES)}
    missing = len(groups) - len(complete)
    if missing > 0:
        logger.info("Skipped %d groups missing required angles", missing)

    logger.info("Extracting %d complete triplets", len(complete))

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_idx = 0

    for key in sorted(complete):
        angles = complete[key]
        pair_dir = output_dir / f"pair_{pair_idx:05d}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        for angle_name in REQUIRED_ANGLES:
            src_path = angles[angle_name]
            dst_path = pair_dir / f"{angle_name}.png"

            img = Image.open(src_path)
            if img.size != (resolution, resolution):
                img = img.resize((resolution, resolution), Image.LANCZOS)
            # Ensure RGBA
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            img.save(dst_path)

        pair_idx += 1

    logger.info("Extracted %d pairs to %s", pair_idx, output_dir)
    return pair_idx


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract paired front/three_quarter/back views for back view training."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Multi-angle dataset directory to scan",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for paired examples",
    )
    parser.add_argument(
        "--style",
        type=str,
        default=None,
        help="Filter by style (flat layout only, e.g. 'textured')",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=TARGET_RESOLUTION,
        help=f"Target resolution (default {TARGET_RESOLUTION})",
    )

    args = parser.parse_args()

    count = extract_pairs(
        args.source_dir,
        args.output_dir,
        style_filter=args.style,
        resolution=args.resolution,
    )

    if count == 0:
        logger.warning("No pairs found! Check source directory and angle coverage.")
    else:
        logger.info("Done — %d pairs ready for training", count)


if __name__ == "__main__":
    main()

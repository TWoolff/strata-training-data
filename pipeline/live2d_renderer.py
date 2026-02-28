"""Render Live2D models into Strata dataset training data.

Loads Live2D models distributed as .model3.json + texture atlas PNGs (or
pre-extracted fragment PNGs), composites the character at the default pose,
builds segmentation masks from ArtMesh-to-region mapping, generates draw
order maps from explicit fragment render order, and applies augmentations.

Produces the same output format as the Blender and Spine pipelines:
- 512×512 RGBA composite image
- 512×512 8-bit grayscale segmentation mask (pixel value = region ID)
- 512×512 8-bit grayscale draw order map (0=back, 255=front)
- Joint position JSON (approximate, from region centroids)
- Per-character source metadata JSON

This module is pure Python (no Blender dependency) so it can be imported
outside Blender for testing and validation.

Live2D model3.json reference:
    https://docs.live2d.com/en/cubism-sdk-manual/cubism-spec/
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

from .config import (
    FLIP_REGION_SWAP,
    JOINT_BBOX_PADDING,
    LIVE2D_AUGMENTATION_COLOR_JITTER,
    LIVE2D_AUGMENTATION_ROTATIONS,
    LIVE2D_AUGMENTATION_SCALES,
    NUM_JOINT_REGIONS,
    REGION_NAMES,
    RENDER_RESOLUTION,
    RegionId,
)
from .live2d_mapper import map_fragment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Live2DFragment:
    """A single ArtMesh fragment from a Live2D model."""

    name: str
    texture_index: int = 0
    # UV coordinates in the texture atlas (normalized 0-1)
    vertex_uvs: list[tuple[float, float]] = field(default_factory=list)
    # Triangle indices referencing vertex_uvs
    triangle_indices: list[int] = field(default_factory=list)
    # Vertex positions in model space (pixels, origin at model center)
    vertex_positions: list[tuple[float, float]] = field(default_factory=list)
    # Draw order (higher = more in front)
    draw_order: int = 0
    # Opacity (0-1)
    opacity: float = 1.0


@dataclass
class Live2DModel:
    """Parsed Live2D model data from .model3.json and associated files."""

    name: str
    model_dir: Path
    version: int = 3  # Cubism SDK version (3 or 4)
    canvas_width: float = 0.0
    canvas_height: float = 0.0
    pixels_per_unit: float = 1.0
    texture_paths: list[str] = field(default_factory=list)
    fragments: list[Live2DFragment] = field(default_factory=list)


@dataclass
class Live2DRenderResult:
    """Result of rendering a single Live2D model."""

    char_id: str
    image: Image.Image  # 512×512 RGBA
    mask: np.ndarray  # 512×512 uint8 (region IDs)
    draw_order_map: np.ndarray  # 512×512 uint8 (0=back, 255=front)
    joint_data: dict[str, Any]  # same schema as joint_extractor
    fragment_count: int
    mapped_count: int
    unmapped_fragments: list[str]


# ---------------------------------------------------------------------------
# Model JSON parsing
# ---------------------------------------------------------------------------


def _find_model_json(model_dir: Path) -> Path | None:
    """Find the .model3.json file in a model directory.

    Args:
        model_dir: Directory containing the Live2D model files.

    Returns:
        Path to the .model3.json file, or None if not found.
    """
    candidates = sorted(model_dir.glob("*.model3.json"))
    if candidates:
        return candidates[0]
    # Fallback: look for any .json that contains "FileReferences"
    for json_path in sorted(model_dir.glob("*.json")):
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            if "FileReferences" in raw:
                return json_path
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return None


def _parse_model_json(model_json_path: Path) -> Live2DModel | None:
    """Parse a Live2D .model3.json file to extract model metadata.

    The .model3.json is the entry point file that references the .moc3,
    textures, and optional physics/expressions. We extract texture paths
    so we can load the atlas images.

    Args:
        model_json_path: Path to the .model3.json file.

    Returns:
        Partially populated Live2DModel, or None on parse failure.
    """
    try:
        raw = json.loads(model_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.error("Failed to parse model JSON %s: %s", model_json_path, exc)
        return None

    file_refs = raw.get("FileReferences", {})
    model_dir = model_json_path.parent

    # Extract texture paths
    textures = file_refs.get("Textures", [])
    texture_paths: list[str] = []
    for tex in textures:
        if isinstance(tex, str):
            texture_paths.append(tex)
        elif isinstance(tex, dict):
            texture_paths.append(tex.get("Name", tex.get("File", "")))

    model = Live2DModel(
        name=model_json_path.stem.replace(".model3", ""),
        model_dir=model_dir,
        texture_paths=texture_paths,
    )

    logger.info(
        "Parsed model JSON %s: %d textures referenced",
        model_json_path.name,
        len(texture_paths),
    )

    return model


def _load_cdi_json(model_dir: Path) -> dict[str, str]:
    """Load .cdi3.json (display info) if available for ArtMesh names.

    The CDI file maps internal IDs to human-readable display names,
    which are more likely to contain meaningful fragment names for mapping.

    Args:
        model_dir: Directory containing the model files.

    Returns:
        Dict of internal_id → display_name.
    """
    cdi_files = sorted(model_dir.glob("*.cdi3.json"))
    if not cdi_files:
        return {}

    try:
        raw = json.loads(cdi_files[0].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}

    name_map: dict[str, str] = {}
    for param in raw.get("Parameters", []):
        pid = param.get("Id", "")
        pname = param.get("GroupId", "") or param.get("Name", "")
        if pid and pname:
            name_map[pid] = pname

    for part in raw.get("Parts", []):
        pid = part.get("Id", "")
        pname = part.get("Name", "")
        if pid and pname:
            name_map[pid] = pname

    return name_map


# ---------------------------------------------------------------------------
# Fragment discovery from pre-extracted images
# ---------------------------------------------------------------------------


def _discover_fragment_images(model_dir: Path) -> list[tuple[str, Path]]:
    """Discover pre-extracted fragment PNG images in a model directory.

    Many community-distributed Live2D models include individual part images
    in a subdirectory (often named 'parts/', 'textures/', or the model name).

    Args:
        model_dir: Directory containing the model files.

    Returns:
        List of (fragment_name, image_path) tuples, sorted by name.
    """
    fragments: list[tuple[str, Path]] = []

    # Check common subdirectory names for part images
    candidate_dirs = [
        model_dir / "parts",
        model_dir / "textures",
        model_dir / "images",
        model_dir,
    ]

    seen: set[Path] = set()
    for search_dir in candidate_dirs:
        if not search_dir.is_dir():
            continue
        for png_path in sorted(search_dir.glob("*.png")):
            if png_path in seen:
                continue
            seen.add(png_path)
            # Skip files that look like full texture atlases (typically large)
            # Fragment images are usually smaller than 1024x1024
            try:
                with Image.open(png_path) as img:
                    w, h = img.size
                    if w > 2048 or h > 2048:
                        continue
            except Exception:
                continue
            fragment_name = png_path.stem
            fragments.append((fragment_name, png_path))

    return fragments


# ---------------------------------------------------------------------------
# Texture atlas loading
# ---------------------------------------------------------------------------


def _load_textures(model: Live2DModel) -> list[Image.Image]:
    """Load texture atlas images referenced by the model.

    Args:
        model: Parsed Live2DModel with texture_paths populated.

    Returns:
        List of PIL Images in RGBA mode, one per texture page.
    """
    textures: list[Image.Image] = []
    for tex_path_str in model.texture_paths:
        tex_path = model.model_dir / tex_path_str
        if not tex_path.exists():
            # Try without subdirectory
            tex_path = model.model_dir / Path(tex_path_str).name
        if not tex_path.exists():
            logger.warning("Texture not found: %s", tex_path_str)
            textures.append(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
            continue
        try:
            img = Image.open(tex_path).convert("RGBA")
            textures.append(img)
            logger.debug("Loaded texture %s (%dx%d)", tex_path.name, img.width, img.height)
        except Exception as exc:
            logger.warning("Failed to load texture %s: %s", tex_path_str, exc)
            textures.append(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))

    return textures


# ---------------------------------------------------------------------------
# Character compositing
# ---------------------------------------------------------------------------


def _composite_from_fragments(
    fragment_images: list[tuple[str, Image.Image, int]],
    fragment_to_region: dict[str, RegionId],
    resolution: int = RENDER_RESOLUTION,
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Composite a character from individual fragment images.

    Renders fragments back-to-front by draw order, building the composite
    image, segmentation mask, and draw order map simultaneously.

    Args:
        fragment_images: List of (fragment_name, image, draw_order) tuples.
        fragment_to_region: Fragment name → region ID mapping.
        resolution: Output resolution (square).

    Returns:
        Tuple of (RGBA image, segmentation mask, draw order map).
    """
    # Sort by draw order (lowest = backmost)
    sorted_fragments = sorted(fragment_images, key=lambda x: x[2])

    if not sorted_fragments:
        canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
        mask = np.zeros((resolution, resolution), dtype=np.uint8)
        draw_order = np.zeros((resolution, resolution), dtype=np.uint8)
        return canvas, mask, draw_order

    # Compute how many fragments and the draw order range for normalization
    num_fragments = len(sorted_fragments)
    if num_fragments > 1:
        min_order = sorted_fragments[0][2]
        max_order = sorted_fragments[-1][2]
        order_range = max_order - min_order if max_order != min_order else 1
    else:
        min_order = 0
        order_range = 1

    # Find the bounding box that encompasses all fragment images
    # Assume fragments are positioned relative to the character center
    # For pre-extracted fragments, we lay them out centered
    total_w = max(img.width for _, img, _ in sorted_fragments)
    total_h = max(img.height for _, img, _ in sorted_fragments)

    # Scale to fit in resolution with padding
    padding_frac = 0.05
    usable = resolution * (1 - 2 * padding_frac)
    scale = usable / max(total_w, total_h) if max(total_w, total_h) > 0 else 1.0
    scale = min(scale, 1.0)  # don't upscale

    canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    mask_arr = np.zeros((resolution, resolution), dtype=np.uint8)
    draw_order_arr = np.zeros((resolution, resolution), dtype=np.uint8)

    for frag_name, frag_img, frag_order in sorted_fragments:
        region_id = fragment_to_region.get(frag_name, 0)

        # Normalize draw order to [0, 255]
        if num_fragments > 1:
            normalized_depth = int((frag_order - min_order) / order_range * 255)
        else:
            normalized_depth = 127

        # Scale fragment
        new_w = max(1, round(frag_img.width * scale))
        new_h = max(1, round(frag_img.height * scale))
        try:
            scaled = frag_img.resize((new_w, new_h), Image.BILINEAR)
        except Exception:
            continue

        # Center in canvas
        paste_x = (resolution - new_w) // 2
        paste_y = (resolution - new_h) // 2

        # Composite onto canvas
        canvas.paste(scaled, (paste_x, paste_y), scaled)

        # Paint mask and draw order for opaque pixels
        if region_id > 0:
            arr = np.array(scaled)
            alpha = arr[:, :, 3]

            src_y_start = max(0, -paste_y)
            src_x_start = max(0, -paste_x)
            src_y_end = min(new_h, resolution - paste_y)
            src_x_end = min(new_w, resolution - paste_x)

            if src_y_end <= src_y_start or src_x_end <= src_x_start:
                continue

            dst_y_start = paste_y + src_y_start
            dst_x_start = paste_x + src_x_start
            dst_y_end = paste_y + src_y_end
            dst_x_end = paste_x + src_x_end

            alpha_slice = alpha[src_y_start:src_y_end, src_x_start:src_x_end]
            opaque = alpha_slice > 0
            mask_arr[dst_y_start:dst_y_end, dst_x_start:dst_x_end][opaque] = region_id
            draw_order_arr[dst_y_start:dst_y_end, dst_x_start:dst_x_end][opaque] = normalized_depth

    return canvas, mask_arr, draw_order_arr


# ---------------------------------------------------------------------------
# Joint extraction (from region centroids)
# ---------------------------------------------------------------------------


def _extract_joints_from_mask(
    mask: np.ndarray,
    resolution: int = RENDER_RESOLUTION,
) -> dict[str, Any]:
    """Extract approximate 2D joint positions from segmentation mask centroids.

    For 2D models without skeleton data, we approximate joint positions as
    the centroid of each region's pixels in the segmentation mask.

    Args:
        mask: Segmentation mask (uint8 array, pixel value = region ID).
        resolution: Image resolution.

    Returns:
        Joint data dict matching the 3D pipeline schema.
    """
    joints: dict[str, dict] = {}
    positions: dict[str, tuple[int, int]] = {}
    visibility: dict[str, bool] = {}

    for region_id in range(1, NUM_JOINT_REGIONS + 1):
        region_name = REGION_NAMES[region_id]
        region_pixels = np.where(mask == region_id)

        if len(region_pixels[0]) == 0:
            joints[region_name] = {
                "position": [-1, -1],
                "confidence": 0.0,
                "visible": False,
            }
            positions[region_name] = (-1, -1)
            visibility[region_name] = False
            continue

        # Centroid of region pixels (y, x from np.where)
        cy = int(np.mean(region_pixels[0]))
        cx = int(np.mean(region_pixels[1]))

        in_bounds = 0 <= cx < resolution and 0 <= cy < resolution
        clamped_x = max(0, min(cx, resolution - 1))
        clamped_y = max(0, min(cy, resolution - 1))

        joints[region_name] = {
            "position": [clamped_x, clamped_y],
            "confidence": 0.8 if in_bounds else 0.4,
            "visible": in_bounds,
        }
        positions[region_name] = (clamped_x, clamped_y)
        visibility[region_name] = in_bounds

    bbox = _compute_bbox(positions, visibility, resolution)

    visible_count = sum(1 for v in visibility.values() if v)
    logger.info(
        "Live2D joints: %d extracted (%d visible, %d missing)",
        len(joints),
        visible_count,
        len(joints) - visible_count,
    )

    return {
        "joints": joints,
        "bbox": bbox,
        "image_size": [resolution, resolution],
    }


def _compute_bbox(
    joint_positions: dict[str, tuple[int, int]],
    visible_flags: dict[str, bool],
    resolution: int,
) -> list[int]:
    """Compute 2D bounding box from visible joint positions."""
    visible_points = [
        pos
        for name, pos in joint_positions.items()
        if visible_flags.get(name, False) and pos != (-1, -1)
    ]

    if not visible_points:
        return [0, 0, resolution, resolution]

    xs = [p[0] for p in visible_points]
    ys = [p[1] for p in visible_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    width = x_max - x_min
    height = y_max - y_min
    pad_x = max(int(width * JOINT_BBOX_PADDING), 5)
    pad_y = max(int(height * JOINT_BBOX_PADDING), 5)

    return [
        max(0, x_min - pad_x),
        max(0, y_min - pad_y),
        min(resolution, x_max + pad_x),
        min(resolution, y_max + pad_y),
    ]


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def _apply_flip(
    image: Image.Image,
    mask: np.ndarray,
    draw_order_map: np.ndarray,
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Apply horizontal flip with left/right region swap.

    Args:
        image: RGBA composite image.
        mask: Segmentation mask.
        draw_order_map: Draw order map.

    Returns:
        Tuple of (flipped image, flipped mask, flipped draw order).
    """
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_mask = np.fliplr(mask).copy()
    flipped_draw_order = np.fliplr(draw_order_map).copy()

    # Swap left/right region IDs
    swapped_mask = flipped_mask.copy()
    for left_id, right_id in FLIP_REGION_SWAP.items():
        left_pixels = flipped_mask == left_id
        right_pixels = flipped_mask == right_id
        swapped_mask[left_pixels] = right_id
        swapped_mask[right_pixels] = left_id

    return flipped_image, swapped_mask, flipped_draw_order


def _apply_rotation(
    image: Image.Image,
    mask: np.ndarray,
    draw_order_map: np.ndarray,
    angle: float,
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Apply rotation augmentation.

    Args:
        image: RGBA composite image.
        mask: Segmentation mask.
        draw_order_map: Draw order map.
        angle: Rotation angle in degrees.

    Returns:
        Tuple of (rotated image, rotated mask, rotated draw order).
    """
    if abs(angle) < 0.01:
        return image, mask, draw_order_map

    rotated_image = image.rotate(
        angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0, 0)
    )

    # For mask, use nearest-neighbor to preserve discrete region IDs
    mask_img = Image.fromarray(mask, mode="L")
    rotated_mask_img = mask_img.rotate(angle, resample=Image.NEAREST, expand=False, fillcolor=0)
    rotated_mask = np.array(rotated_mask_img)

    do_img = Image.fromarray(draw_order_map, mode="L")
    rotated_do_img = do_img.rotate(angle, resample=Image.NEAREST, expand=False, fillcolor=0)
    rotated_do = np.array(rotated_do_img)

    return rotated_image, rotated_mask, rotated_do


def _apply_scale(
    image: Image.Image,
    mask: np.ndarray,
    draw_order_map: np.ndarray,
    scale_factor: float,
    resolution: int = RENDER_RESOLUTION,
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Apply uniform scale augmentation (scale then re-center).

    Args:
        image: RGBA composite image.
        mask: Segmentation mask.
        draw_order_map: Draw order map.
        scale_factor: Scale multiplier (e.g. 0.9 or 1.1).
        resolution: Output resolution.

    Returns:
        Tuple of (scaled image, scaled mask, scaled draw order).
    """
    if abs(scale_factor - 1.0) < 0.01:
        return image, mask, draw_order_map

    new_size = max(1, round(resolution * scale_factor))

    # Scale image
    scaled_img = image.resize((new_size, new_size), Image.BILINEAR)
    canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    offset = (resolution - new_size) // 2
    canvas.paste(scaled_img, (offset, offset), scaled_img)

    # Scale mask and draw order (nearest-neighbor to preserve discrete IDs)
    def _scale_grayscale(arr: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(arr, mode="L")
        scaled = np.array(pil.resize((new_size, new_size), Image.NEAREST))
        result = np.zeros((resolution, resolution), dtype=np.uint8)
        # Compute overlap region between scaled content and output canvas
        sy0, sx0 = max(0, -offset), max(0, -offset)
        sy1 = min(new_size, resolution - offset)
        sx1 = min(new_size, resolution - offset)
        if sy1 > sy0 and sx1 > sx0:
            result[offset + sy0 : offset + sy1, offset + sx0 : offset + sx1] = scaled[
                sy0:sy1, sx0:sx1
            ]
        return result

    return canvas, _scale_grayscale(mask), _scale_grayscale(draw_order_map)


def _apply_color_jitter(
    image: Image.Image,
    jitter_config: dict[str, tuple[float, float]] | None = None,
) -> Image.Image:
    """Apply random color jitter to the composite image.

    Only affects the color image, not the mask or draw order.

    Args:
        image: RGBA composite image.
        jitter_config: Dict with 'hue', 'saturation', 'brightness' ranges.

    Returns:
        Color-jittered image.
    """
    if jitter_config is None:
        jitter_config = LIVE2D_AUGMENTATION_COLOR_JITTER

    # Split alpha channel to preserve it
    r, g, b, a = image.split()
    rgb = Image.merge("RGB", (r, g, b))

    # Brightness
    bmin, bmax = jitter_config.get("brightness", (1.0, 1.0))
    brightness_factor = random.uniform(bmin, bmax)
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness_factor)

    # Saturation
    smin, smax = jitter_config.get("saturation", (1.0, 1.0))
    saturation_factor = random.uniform(smin, smax)
    rgb = ImageEnhance.Color(rgb).enhance(saturation_factor)

    # Hue shift via HSV
    hmin, hmax = jitter_config.get("hue", (0.0, 0.0))
    hue_shift = random.uniform(hmin, hmax)
    if abs(hue_shift) > 0.5:
        hsv = rgb.convert("HSV")
        h, s, v = hsv.split()
        h_arr = np.array(h, dtype=np.int16)
        # Hue in PIL HSV is 0-255 (mapped from 0-360 degrees)
        h_shift_scaled = int(hue_shift / 360 * 255)
        h_arr = (h_arr + h_shift_scaled) % 256
        h = Image.fromarray(h_arr.astype(np.uint8), mode="L")
        rgb = Image.merge("HSV", (h, s, v)).convert("RGB")

    # Recombine with original alpha
    r, g, b = rgb.split()
    return Image.merge("RGBA", (r, g, b, a))


def generate_augmentations(
    image: Image.Image,
    mask: np.ndarray,
    draw_order_map: np.ndarray,
    resolution: int = RENDER_RESOLUTION,
) -> list[tuple[str, Image.Image, np.ndarray, np.ndarray]]:
    """Generate augmented variants of a Live2D render.

    Produces: original + horizontal flip + rotation/scale/color combos.
    Returns at most 4 variants per model (matching issue target).

    Args:
        image: RGBA composite image.
        mask: Segmentation mask.
        draw_order_map: Draw order map.
        resolution: Output resolution.

    Returns:
        List of (augmentation_label, image, mask, draw_order_map) tuples.
    """
    results: list[tuple[str, Image.Image, np.ndarray, np.ndarray]] = []

    # 1. Original (identity)
    results.append(("identity", image.copy(), mask.copy(), draw_order_map.copy()))

    # 2. Horizontal flip
    flip_img, flip_mask, flip_do = _apply_flip(image, mask, draw_order_map)
    results.append(("flip", flip_img, flip_mask, flip_do))

    # 3. Random rotation + color jitter
    rot_angle = random.choice([a for a in LIVE2D_AUGMENTATION_ROTATIONS if a != 0.0])
    rot_img, rot_mask, rot_do = _apply_rotation(image, mask, draw_order_map, rot_angle)
    rot_img = _apply_color_jitter(rot_img)
    results.append((f"rot{rot_angle:+.0f}", rot_img, rot_mask, rot_do))

    # 4. Random scale + color jitter
    scale_factor = random.choice([s for s in LIVE2D_AUGMENTATION_SCALES if abs(s - 1.0) > 0.01])
    sc_img, sc_mask, sc_do = _apply_scale(image, mask, draw_order_map, scale_factor, resolution)
    sc_img = _apply_color_jitter(sc_img)
    results.append((f"scale{scale_factor:.1f}", sc_img, sc_mask, sc_do))

    return results


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def process_live2d_model(
    model_dir: Path,
    resolution: int = RENDER_RESOLUTION,
) -> Live2DRenderResult | None:
    """Process a single Live2D model directory into Strata dataset outputs.

    Supports two workflows:
    1. Models with .model3.json — parses metadata, loads textures
    2. Models with pre-extracted fragment PNGs — discovers and composites

    Args:
        model_dir: Directory containing the Live2D model files.
        resolution: Output image resolution (square).

    Returns:
        Live2DRenderResult with image, mask, draw order, and joint data,
        or None on error.
    """
    if not model_dir.is_dir():
        logger.error("Model directory not found: %s", model_dir)
        return None

    model_name = model_dir.name

    # Try to parse .model3.json for metadata
    model_json_path = _find_model_json(model_dir)
    model: Live2DModel | None = None
    if model_json_path:
        model = _parse_model_json(model_json_path)

    # Load CDI display names if available
    cdi_names = _load_cdi_json(model_dir)

    # Discover fragment images (pre-extracted PNGs)
    fragment_images_raw = _discover_fragment_images(model_dir)

    if not fragment_images_raw:
        # Try loading from texture atlas paths in model JSON
        if model and model.texture_paths:
            textures = _load_textures(model)
            # If we have textures but no fragments, we can't proceed
            # (would need .moc3 binary parsing for UV data)
            if textures:
                logger.warning(
                    "Model %s has texture atlases but no fragment images. "
                    "Binary .moc3 parsing not supported — skipping.",
                    model_name,
                )
        logger.warning("No fragment images found for model %s", model_name)
        return None

    logger.info(
        "Found %d fragment images for model %s",
        len(fragment_images_raw),
        model_name,
    )

    # Map fragments to regions and load images
    fragment_to_region: dict[str, RegionId] = {}
    unmapped: list[str] = []
    loaded_fragments: list[tuple[str, Image.Image, int]] = []

    for draw_order_idx, (frag_name, frag_path) in enumerate(fragment_images_raw):
        # Use CDI display name if available for better mapping
        display_name = cdi_names.get(frag_name, frag_name)
        _, region_id = map_fragment(display_name)
        if region_id < 0:
            # Try the raw fragment name too
            _, region_id = map_fragment(frag_name)

        if region_id >= 0:
            fragment_to_region[frag_name] = region_id
        else:
            fragment_to_region[frag_name] = 0  # unmapped → background
            unmapped.append(frag_name)

        try:
            img = Image.open(frag_path).convert("RGBA")
            loaded_fragments.append((frag_name, img, draw_order_idx))
        except Exception as exc:
            logger.warning("Failed to load fragment %s: %s", frag_path, exc)

    if not loaded_fragments:
        logger.error("No fragment images could be loaded for model %s", model_name)
        return None

    mapped_count = len(fragment_images_raw) - len(unmapped)
    total_count = len(fragment_images_raw)
    logger.info(
        "Live2D mapping for %s: %d/%d mapped (%.0f%%), %d unmapped",
        model_name,
        mapped_count,
        total_count,
        (mapped_count / total_count * 100) if total_count else 0,
        len(unmapped),
    )
    if unmapped:
        logger.warning("Unmapped fragments in %s: %s", model_name, unmapped)

    # Composite character
    image, mask, draw_order_map = _composite_from_fragments(
        loaded_fragments, fragment_to_region, resolution
    )

    # Extract joints from mask centroids
    joint_data = _extract_joints_from_mask(mask, resolution)

    char_id = f"live2d_{model_name}"

    return Live2DRenderResult(
        char_id=char_id,
        image=image,
        mask=mask,
        draw_order_map=draw_order_map,
        joint_data=joint_data,
        fragment_count=total_count,
        mapped_count=mapped_count,
        unmapped_fragments=unmapped,
    )


def process_live2d_directory(
    live2d_dir: Path,
    output_dir: Path,
    *,
    resolution: int = RENDER_RESOLUTION,
    styles: list[str] | None = None,
    enable_augmentation: bool = True,
    only_new: bool = False,
) -> list[Live2DRenderResult]:
    """Process all Live2D models in a directory.

    Each subdirectory of live2d_dir is treated as a separate model.
    Discovers models, processes each, applies augmentations, and saves
    outputs using the exporter module.

    Args:
        live2d_dir: Directory containing Live2D model subdirectories.
        output_dir: Root dataset output directory.
        resolution: Output image resolution.
        styles: Post-render art styles to generate (e.g. ["pixel", "sketch"]).
        enable_augmentation: Whether to generate augmented variants.
        only_new: Skip existing files.

    Returns:
        List of successful Live2DRenderResult objects.
    """
    from . import exporter

    # Lazy import: only load style_augmentor when non-flat styles are needed
    # (it depends on cv2 which may not be available in test environments)
    post_render_styles = {"pixel", "painterly", "sketch"}
    needs_style_augmentor = styles and any(s in post_render_styles for s in styles)
    apply_post_render_style = None
    if needs_style_augmentor:
        from .style_augmentor import apply_post_render_style

    # Discover model directories (each subdirectory = one model)
    model_dirs: list[Path] = sorted(p for p in live2d_dir.iterdir() if p.is_dir())

    if not model_dirs:
        logger.warning("No model directories found in %s", live2d_dir)
        return []

    logger.info("Found %d Live2D model directories in %s", len(model_dirs), live2d_dir)

    exporter.ensure_output_dirs(output_dir)
    results: list[Live2DRenderResult] = []

    for model_dir in model_dirs:
        result = process_live2d_model(model_dir, resolution=resolution)
        if result is None:
            continue

        # Generate augmentation variants
        if enable_augmentation:
            variants = generate_augmentations(
                result.image, result.mask, result.draw_order_map, resolution
            )
        else:
            variants = [("identity", result.image, result.mask, result.draw_order_map)]

        for pose_index, (_aug_label, aug_image, aug_mask, aug_draw_order) in enumerate(variants):
            # Save mask
            exporter.save_mask(aug_mask, output_dir, result.char_id, pose_index, only_new=only_new)

            # Save joints (recompute from augmented mask)
            joint_data = _extract_joints_from_mask(aug_mask, resolution)
            exporter.save_joints(
                joint_data, output_dir, result.char_id, pose_index, only_new=only_new
            )

            # Save draw order
            exporter.save_draw_order(
                aug_draw_order, output_dir, result.char_id, pose_index, only_new=only_new
            )

            # Save original image as "flat" style
            exporter.save_image(
                aug_image,
                output_dir,
                result.char_id,
                pose_index,
                "flat",
                only_new=only_new,
            )

            # Apply post-render styles if requested
            if styles and apply_post_render_style is not None:
                for style in styles:
                    if style == "flat":
                        continue
                    if style in post_render_styles:
                        styled = apply_post_render_style(aug_image, style)
                        exporter.save_image(
                            styled,
                            output_dir,
                            result.char_id,
                            pose_index,
                            style,
                            only_new=only_new,
                        )

        # Save source metadata (once per model)
        exporter.save_source_metadata(
            output_dir,
            result.char_id,
            source="live2d",
            name=result.char_id,
            license_="",
            attribution="",
            bone_mapping="auto",
            unmapped_bones=result.unmapped_fragments,
            character_type="humanoid",
            notes=(
                f"Live2D model, {result.fragment_count} fragments, "
                f"{result.mapped_count} mapped, "
                f"{len(result.unmapped_fragments)} unmapped"
            ),
            only_new=only_new,
        )

        results.append(result)
        logger.info(
            "Processed Live2D model %s: %d fragments (%d mapped), %d augmentation variants saved",
            result.char_id,
            result.fragment_count,
            result.mapped_count,
            len(variants),
        )

    logger.info(
        "Live2D processing complete: %d/%d models succeeded",
        len(results),
        len(model_dirs),
    )

    return results

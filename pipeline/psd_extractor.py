"""Extract layers from PSD files and generate segmentation masks.

Uses ``psd-tools`` to parse Photoshop files, map layer names to Strata
region IDs via regex matching, and produce composite images with
per-pixel segmentation masks.

Mapping strategy:
    1. Recursively walk the PSD layer tree (groups + leaf layers).
    2. Match each visible leaf layer name against PSD_LAYER_PATTERNS
       (case-insensitive regex, first match wins).
    3. Unmapped layers are flagged for manual review.
    4. Build the composite image from all visible layers.
    5. Build the segmentation mask by painting each mapped layer's
       alpha footprint with its region ID value.

This module is pure Python (no Blender dependency) so it can be imported
outside Blender for testing and validation.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import exporter
from .config import (
    PSD_LAYER_PATTERNS,
    REGION_NAME_TO_ID,
    RENDER_RESOLUTION,
    RegionId,
)

logger = logging.getLogger(__name__)

# Pre-compile all layer-name patterns for performance
_COMPILED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, re.IGNORECASE), region_name) for pattern, region_name in PSD_LAYER_PATTERNS
]

# Layer kinds that are not pixel content (skip for mapping)
_SKIP_LAYER_KINDS: frozenset[str] = frozenset(
    {
        "brightnesscontrast",
        "curves",
        "exposure",
        "gradient",
        "huesaturation",
        "invert",
        "levels",
        "photofilter",
        "posterize",
        "selectivecolor",
        "threshold",
        "vibrance",
        "colorbalance",
        "colorlookup",
        "channelmixer",
        "solidcolorfill",
        "patternfill",
        "gradientfill",
    }
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LayerMapping:
    """Mapping result for a single PSD layer."""

    layer_name: str
    strata_label: str  # Region name or "UNMAPPED"
    strata_region_id: RegionId  # Region ID or -1 if unmapped
    confirmed: str  # "auto" or "pending"
    group_path: str  # Parent group path (e.g. "body/arms")
    is_visible: bool
    is_group: bool


@dataclass
class PSDMapping:
    """Mapping results for an entire PSD file."""

    psd_id: str
    mappings: list[LayerMapping] = field(default_factory=list)

    @property
    def mapped_count(self) -> int:
        """Number of layers successfully mapped to a body region (not background)."""
        return sum(1 for m in self.mappings if m.strata_region_id > 0 and not m.is_group)

    @property
    def unmapped_count(self) -> int:
        """Number of leaf layers that could not be mapped."""
        return sum(1 for m in self.mappings if m.strata_region_id < 0 and not m.is_group)

    @property
    def total_count(self) -> int:
        """Total number of leaf layers."""
        return sum(1 for m in self.mappings if not m.is_group)

    @property
    def auto_rate(self) -> float:
        """Fraction of leaf layers that were auto-mapped (0.0-1.0)."""
        total = self.total_count
        if total == 0:
            return 0.0
        return self.mapped_count / total


@dataclass
class PSDResult:
    """Processing result for a single PSD file."""

    psd_id: str
    image: Image.Image  # Composite RGBA image
    mask: np.ndarray  # uint8 segmentation mask (H, W)
    mapping: PSDMapping
    width: int
    height: int


# ---------------------------------------------------------------------------
# Core mapping function
# ---------------------------------------------------------------------------


def map_layer(layer_name: str) -> tuple[str, RegionId]:
    """Map a single PSD layer name to a Strata region.

    Args:
        layer_name: Layer name from the PSD file.

    Returns:
        Tuple of (region_name, region_id). Returns ("UNMAPPED", -1) if
        no pattern matches.
    """
    for compiled_pattern, region_name in _COMPILED_PATTERNS:
        if compiled_pattern.search(layer_name):
            region_id = REGION_NAME_TO_ID[region_name]
            return region_name, region_id
    return "UNMAPPED", -1


def map_psd(
    psd_id: str,
    layer_names: list[str],
    *,
    group_paths: list[str] | None = None,
    visibilities: list[bool] | None = None,
    is_groups: list[bool] | None = None,
) -> PSDMapping:
    """Map all layers of a PSD to Strata regions.

    Args:
        psd_id: Unique identifier for the PSD file.
        layer_names: List of layer names.
        group_paths: Optional list of parent group paths per layer.
        visibilities: Optional list of visibility flags per layer.
        is_groups: Optional list of is-group flags per layer.

    Returns:
        PSDMapping with results for each layer.
    """
    n = len(layer_names)
    if group_paths is None:
        group_paths = [""] * n
    if visibilities is None:
        visibilities = [True] * n
    if is_groups is None:
        is_groups = [False] * n

    result = PSDMapping(psd_id=psd_id)

    for name, group_path, visible, is_group in zip(
        layer_names, group_paths, visibilities, is_groups, strict=True
    ):
        region_name, region_id = map_layer(name)
        confirmed = "auto" if region_id >= 0 else "pending"
        result.mappings.append(
            LayerMapping(
                layer_name=name,
                strata_label=region_name,
                strata_region_id=region_id,
                confirmed=confirmed,
                group_path=group_path,
                is_visible=visible,
                is_group=is_group,
            )
        )

    logger.info(
        "PSD mapping for %s: %d/%d auto-mapped (%.0f%%), %d unmapped",
        psd_id,
        result.mapped_count,
        result.total_count,
        result.auto_rate * 100,
        result.unmapped_count,
    )

    if result.unmapped_count > 0:
        unmapped = [
            m.layer_name for m in result.mappings if m.strata_region_id < 0 and not m.is_group
        ]
        logger.warning("Unmapped layers in %s: %s", psd_id, unmapped)

    return result


# ---------------------------------------------------------------------------
# PSD file processing
# ---------------------------------------------------------------------------


def _is_adjustment_layer(layer: Any) -> bool:
    """Check if a psd-tools layer is an adjustment or fill layer."""
    kind = getattr(layer, "kind", "")
    if isinstance(kind, str) and kind.lower() in _SKIP_LAYER_KINDS:
        return True
    layer_name = getattr(layer, "name", "")
    return isinstance(layer_name, str) and layer_name.lower().startswith("adjustment")


def _walk_layers(
    layers: Any,
    group_path: str = "",
) -> list[dict[str, Any]]:
    """Recursively walk PSD layer tree and collect leaf layer info.

    Args:
        layers: Iterable of psd-tools Layer objects.
        group_path: Dot-separated path of parent groups.

    Returns:
        List of dicts with keys: name, group_path, visible, is_group, layer.
    """
    results: list[dict[str, Any]] = []

    for layer in layers:
        is_group = bool(getattr(layer, "is_group", lambda: False)())

        if is_group:
            child_path = f"{group_path}/{layer.name}" if group_path else layer.name
            results.append(
                {
                    "name": layer.name,
                    "group_path": group_path,
                    "visible": layer.is_visible(),
                    "is_group": True,
                    "layer": layer,
                }
            )
            results.extend(_walk_layers(layer, child_path))
        else:
            if _is_adjustment_layer(layer):
                logger.debug("Skipping adjustment layer: %s", layer.name)
                continue

            results.append(
                {
                    "name": layer.name,
                    "group_path": group_path,
                    "visible": layer.is_visible(),
                    "is_group": False,
                    "layer": layer,
                }
            )

    return results


def _render_layer_alpha(layer: Any, canvas_size: tuple[int, int]) -> np.ndarray | None:
    """Extract the alpha channel of a layer composited onto a blank canvas.

    Args:
        layer: A psd-tools Layer object.
        canvas_size: (width, height) of the PSD canvas.

    Returns:
        Boolean numpy array (H, W) where True = layer has non-zero alpha,
        or None if the layer has no pixel data.
    """
    try:
        layer_image = layer.composite()
    except Exception:
        logger.debug("Could not composite layer '%s'", getattr(layer, "name", "?"))
        return None

    if layer_image is None:
        return None

    layer_image = layer_image.convert("RGBA")

    # Create full-canvas image and paste the layer at its offset
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    offset_x = getattr(layer, "left", 0)
    offset_y = getattr(layer, "top", 0)
    canvas.paste(layer_image, (offset_x, offset_y))

    alpha = np.array(canvas.split()[-1])
    return alpha > 0


def _resize_to_square(
    image: Image.Image,
    mask: np.ndarray,
    resolution: int,
) -> tuple[Image.Image, np.ndarray]:
    """Resize image and mask to a square canvas, preserving aspect ratio.

    The character is centered on the canvas with transparent padding.
    Mask uses nearest-neighbor interpolation to preserve region IDs.

    Args:
        image: RGBA composite image.
        mask: uint8 segmentation mask array.
        resolution: Target square resolution (e.g. 512).

    Returns:
        Tuple of (resized_image, resized_mask).
    """
    w, h = image.size
    scale = resolution / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    # Resize image (high quality) and mask (nearest neighbor)
    resized_img = image.resize((new_w, new_h), Image.LANCZOS)
    resized_mask = Image.fromarray(mask).resize((new_w, new_h), Image.NEAREST)

    # Center on square canvas
    canvas_img = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    canvas_mask = Image.new("L", (resolution, resolution), 0)

    offset_x = (resolution - new_w) // 2
    offset_y = (resolution - new_h) // 2

    canvas_img.paste(resized_img, (offset_x, offset_y))
    canvas_mask.paste(resized_mask, (offset_x, offset_y))

    return canvas_img, np.array(canvas_mask)


def process_psd_file(
    psd_path: Path,
    *,
    resolution: int = RENDER_RESOLUTION,
) -> PSDResult | None:
    """Process a single PSD file into a composite image and segmentation mask.

    Args:
        psd_path: Path to the .psd file.
        resolution: Target square resolution for output images.

    Returns:
        PSDResult with composite image and mask, or None if the PSD
        has no usable body-part layers.
    """
    try:
        from psd_tools import PSDImage
    except ImportError:
        logger.error("psd-tools is not installed. Run: pip install 'psd-tools>=2.0'")
        return None

    psd_id = f"psd_{psd_path.stem}"

    logger.info("Processing PSD: %s (id=%s)", psd_path, psd_id)

    psd = PSDImage.open(psd_path)
    canvas_size = (psd.width, psd.height)

    # Walk layer tree
    layer_infos = _walk_layers(psd)

    # Map layer names
    mapping = map_psd(
        psd_id,
        [info["name"] for info in layer_infos],
        group_paths=[info["group_path"] for info in layer_infos],
        visibilities=[info["visible"] for info in layer_infos],
        is_groups=[info["is_group"] for info in layer_infos],
    )

    # Check if any body-region layers were found (region > 0)
    body_region_count = sum(
        1
        for info, m in zip(layer_infos, mapping.mappings, strict=True)
        if m.strata_region_id > 0 and not info["is_group"] and info["visible"]
    )

    if body_region_count == 0:
        logger.warning(
            "No body-part layers found in %s — skipping. "
            "This PSD may use rendering-concern layer naming (lineart, color, etc.)",
            psd_path.name,
        )
        return None

    # Build segmentation mask
    mask = np.zeros((psd.height, psd.width), dtype=np.uint8)

    for info, m in zip(layer_infos, mapping.mappings, strict=True):
        if info["is_group"]:
            continue
        if not info["visible"]:
            continue
        if m.strata_region_id <= 0:
            continue

        alpha_mask = _render_layer_alpha(info["layer"], canvas_size)
        if alpha_mask is None:
            continue

        # Paint this layer's region ID where it has alpha.
        # Later (higher) layers overwrite earlier ones — matching PSD draw order.
        mask[alpha_mask] = m.strata_region_id

    # Composite image from psd-tools
    composite = psd.composite()
    if composite is None:
        logger.warning("Could not composite PSD %s", psd_path.name)
        return None
    composite = composite.convert("RGBA")

    # Resize to target resolution
    composite, mask = _resize_to_square(composite, mask, resolution)

    logger.info(
        "PSD %s: %dx%d, %d body-region layers, %d unmapped layers",
        psd_id,
        resolution,
        resolution,
        body_region_count,
        mapping.unmapped_count,
    )

    return PSDResult(
        psd_id=psd_id,
        image=composite,
        mask=mask,
        mapping=mapping,
        width=resolution,
        height=resolution,
    )


def process_psd_directory(
    psd_dir: Path,
    output_dir: Path,
    *,
    resolution: int = RENDER_RESOLUTION,
    styles: list[str] | None = None,
    only_new: bool = False,
) -> list[PSDResult]:
    """Batch-process all PSD files in a directory.

    Args:
        psd_dir: Directory containing .psd files.
        output_dir: Root dataset output directory.
        resolution: Target square resolution for output images.
        styles: Art style names to save (default: ["flat"]).
        only_new: If True, skip PSD files that already have output.

    Returns:
        List of successfully processed PSDResult objects.
    """
    if styles is None:
        styles = ["flat"]

    psd_files = sorted(psd_dir.glob("*.psd"))
    if not psd_files:
        logger.warning("No .psd files found in %s", psd_dir)
        return []

    logger.info("Found %d PSD files in %s", len(psd_files), psd_dir)

    exporter.ensure_output_dirs(output_dir)
    results: list[PSDResult] = []

    for psd_path in psd_files:
        result = process_psd_file(psd_path, resolution=resolution)
        if result is None:
            continue

        # Save outputs via exporter
        pose_index = 0  # Each PSD = one pose

        exporter.save_mask(result.mask, output_dir, result.psd_id, pose_index, only_new=only_new)

        for style in styles:
            exporter.save_image(
                result.image,
                output_dir,
                result.psd_id,
                pose_index,
                style,
                only_new=only_new,
            )

        unmapped_names = [
            m.layer_name
            for m in result.mapping.mappings
            if m.strata_region_id < 0 and not m.is_group
        ]

        exporter.save_source_metadata(
            output_dir,
            result.psd_id,
            source="psd",
            name=psd_path.stem,
            bone_mapping="auto",
            unmapped_bones=unmapped_names,
            character_type="humanoid",
            notes="PSD layer extraction",
            only_new=only_new,
        )

        results.append(result)

    logger.info(
        "PSD batch complete: %d/%d files produced output",
        len(results),
        len(psd_files),
    )

    return results


# ---------------------------------------------------------------------------
# CSV export / import
# ---------------------------------------------------------------------------

CSV_HEADER: list[str] = [
    "psd_id",
    "layer_name",
    "group_path",
    "strata_label",
    "strata_region_id",
    "confirmed",
    "is_visible",
]


def export_csv(
    mappings: list[PSDMapping],
    output_path: Path,
    *,
    append: bool = False,
) -> None:
    """Write PSD layer mappings to a CSV file for review.

    Args:
        mappings: List of PSDMapping results to export.
        output_path: Path to the output CSV file.
        append: If True, append to existing file without header.
    """
    mode = "a" if append else "w"
    write_header = not append or not output_path.exists()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        for psd in mappings:
            for m in psd.mappings:
                if m.is_group:
                    continue
                writer.writerow(
                    [
                        psd.psd_id,
                        m.layer_name,
                        m.group_path,
                        m.strata_label,
                        m.strata_region_id,
                        m.confirmed,
                        m.is_visible,
                    ]
                )

    total_rows = sum(sum(1 for m in psd.mappings if not m.is_group) for psd in mappings)
    logger.info("Exported %d layer mappings to %s", total_rows, output_path)


def load_csv(csv_path: Path) -> list[PSDMapping]:
    """Load PSD layer mappings from a previously exported CSV.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of PSDMapping objects reconstructed from the CSV.
    """
    models: dict[str, PSDMapping] = {}

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            psd_id = row["psd_id"]
            if psd_id not in models:
                models[psd_id] = PSDMapping(psd_id=psd_id)
            models[psd_id].mappings.append(
                LayerMapping(
                    layer_name=row["layer_name"],
                    strata_label=row["strata_label"],
                    strata_region_id=int(row["strata_region_id"]),
                    confirmed=row["confirmed"],
                    group_path=row["group_path"],
                    is_visible=row["is_visible"].lower() == "true",
                    is_group=False,
                )
            )

    return list(models.values())


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def region_summary(mapping: PSDMapping) -> dict[str, list[str]]:
    """Group layer names by their assigned Strata region.

    Args:
        mapping: A PSDMapping result.

    Returns:
        Dict of region_name -> list of layer names assigned to that region.
    """
    summary: dict[str, list[str]] = {}
    for m in mapping.mappings:
        if m.is_group:
            continue
        summary.setdefault(m.strata_label, []).append(m.layer_name)
    return summary

"""Parse Spine 2D animation projects and extract Strata dataset outputs.

Reads Spine JSON project files (.spine or exported .json), loads the
character's part images from the atlas directory, composites the assembled
character at the default (setup) pose, maps bones to Strata regions, and
produces:

- 512×512 RGBA character image
- 512×512 8-bit grayscale segmentation mask (pixel value = region ID)
- Joint position JSON (same schema as the 3D pipeline)

This module is pure Python (no Blender dependency) so it can be imported
outside Blender for testing and validation.

Spine JSON format reference: http://en.esotericsoftware.com/spine-json-format
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import (
    JOINT_BBOX_PADDING,
    NUM_JOINT_REGIONS,
    REGION_NAME_TO_ID,
    REGION_NAMES,
    RENDER_RESOLUTION,
    SPINE_BONE_PATTERNS,
    RegionId,
)

logger = logging.getLogger(__name__)

# Pre-compile Spine bone/slot patterns for performance
_COMPILED_SPINE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, re.IGNORECASE), region_name)
    for pattern, region_name in SPINE_BONE_PATTERNS
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SpineBone:
    """A bone from the Spine skeleton hierarchy."""

    name: str
    parent: str | None = None
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    length: float = 0.0
    # Computed world transform
    world_x: float = 0.0
    world_y: float = 0.0
    world_rotation: float = 0.0
    world_scale_x: float = 1.0
    world_scale_y: float = 1.0


@dataclass
class SpineSlot:
    """A slot from the Spine skeleton (draw order entry)."""

    name: str
    bone: str
    attachment: str | None = None
    color: str = "FFFFFFFF"


@dataclass
class SpineAttachment:
    """A region attachment from a Spine skin."""

    name: str
    slot_name: str
    type: str = "region"
    path: str | None = None  # image path override
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    width: int = 0
    height: int = 0


@dataclass
class SpineProject:
    """Parsed Spine project data."""

    name: str
    spine_version: str = ""
    skeleton_width: float = 0.0
    skeleton_height: float = 0.0
    bones: list[SpineBone] = field(default_factory=list)
    slots: list[SpineSlot] = field(default_factory=list)
    skins: dict[str, list[SpineAttachment]] = field(default_factory=dict)
    images_dir: str = "./images/"


@dataclass
class SpineParseResult:
    """Result of parsing and processing a single Spine character."""

    char_id: str
    skin_name: str
    image: Image.Image  # 512×512 RGBA
    mask: np.ndarray  # 512×512 uint8 (region IDs)
    joint_data: dict[str, Any]  # same schema as joint_extractor
    bone_to_region: dict[str, RegionId]
    unmapped_bones: list[str]
    slot_to_region: dict[str, RegionId]


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _parse_bones(raw_bones: list[dict[str, Any]]) -> list[SpineBone]:
    """Parse the bones array from Spine JSON."""
    return [
        SpineBone(
            name=b["name"],
            parent=b.get("parent"),
            x=b.get("x", 0.0),
            y=b.get("y", 0.0),
            rotation=b.get("rotation", 0.0),
            scale_x=b.get("scaleX", 1.0),
            scale_y=b.get("scaleY", 1.0),
            length=b.get("length", 0.0),
        )
        for b in raw_bones
    ]


def _parse_slots(raw_slots: list[dict[str, Any]]) -> list[SpineSlot]:
    """Parse the slots array from Spine JSON."""
    return [
        SpineSlot(
            name=s["name"],
            bone=s["bone"],
            attachment=s.get("attachment"),
            color=s.get("color", "FFFFFFFF"),
        )
        for s in raw_slots
    ]


def _parse_skins(
    raw_skins: list[dict[str, Any]] | dict[str, Any],
) -> dict[str, list[SpineAttachment]]:
    """Parse skins from Spine JSON.

    Handles both Spine 4.x format (array of objects with 'name' and
    'attachments' keys) and Spine 3.x format (dict of skin_name →
    slot_name → attachment_name → properties).
    """
    skins: dict[str, list[SpineAttachment]] = {}

    if isinstance(raw_skins, list):
        # Spine 4.x format: [{"name": "default", "attachments": {...}}, ...]
        for skin_obj in raw_skins:
            skin_name = skin_obj.get("name", "default")
            attachments = _parse_skin_attachments(
                skin_obj.get("attachments", {}),
            )
            skins[skin_name] = attachments
    elif isinstance(raw_skins, dict):
        # Spine 3.x format: {"default": {"slotName": {"attachName": {...}}}}
        for skin_name, slots_data in raw_skins.items():
            attachments = _parse_skin_attachments(slots_data)
            skins[skin_name] = attachments
    else:
        logger.warning("Unexpected skins format: %s", type(raw_skins))

    return skins


def _parse_skin_attachments(
    slots_data: dict[str, dict[str, Any]],
) -> list[SpineAttachment]:
    """Parse attachments from a single skin's slot data."""
    attachments: list[SpineAttachment] = []
    for slot_name, slot_attachments in slots_data.items():
        for attach_name, props in slot_attachments.items():
            attach_type = props.get("type", "region")
            if attach_type not in ("region", "regionsequence"):
                logger.debug(
                    "Skipping non-region attachment %s (type=%s) in slot %s",
                    attach_name,
                    attach_type,
                    slot_name,
                )
                continue
            attachments.append(SpineAttachment(
                name=attach_name,
                slot_name=slot_name,
                type=attach_type,
                path=props.get("path"),
                x=props.get("x", 0.0),
                y=props.get("y", 0.0),
                rotation=props.get("rotation", 0.0),
                scale_x=props.get("scaleX", 1.0),
                scale_y=props.get("scaleY", 1.0),
                width=int(props.get("width", 0)),
                height=int(props.get("height", 0)),
            ))
    return attachments


def parse_spine_json(json_path: Path) -> SpineProject:
    """Parse a Spine JSON project file.

    Args:
        json_path: Path to the .spine or .json file.

    Returns:
        Parsed SpineProject with bones, slots, and skins.

    Raises:
        FileNotFoundError: If the JSON file doesn't exist.
        json.JSONDecodeError: If the file isn't valid JSON.
    """
    raw = json.loads(json_path.read_text(encoding="utf-8"))

    skeleton = raw.get("skeleton", {})
    project = SpineProject(
        name=json_path.stem,
        spine_version=skeleton.get("spine", ""),
        skeleton_width=skeleton.get("width", 0.0),
        skeleton_height=skeleton.get("height", 0.0),
        images_dir=skeleton.get("images", "./images/"),
    )

    project.bones = _parse_bones(raw.get("bones", []))
    project.slots = _parse_slots(raw.get("slots", []))
    project.skins = _parse_skins(raw.get("skins", []))

    logger.info(
        "Parsed Spine project %s (v%s): %d bones, %d slots, %d skins",
        project.name,
        project.spine_version,
        len(project.bones),
        len(project.slots),
        len(project.skins),
    )

    return project


# ---------------------------------------------------------------------------
# Bone world transform computation
# ---------------------------------------------------------------------------


def _compute_world_transforms(bones: list[SpineBone]) -> dict[str, SpineBone]:
    """Compute world transforms for all bones by walking the hierarchy.

    Spine bones are stored parent-before-child, so a single pass suffices.

    Args:
        bones: Parsed bones in parent-before-child order.

    Returns:
        Dict of bone name → SpineBone with world_x/y/rotation/scale populated.
    """
    bone_map: dict[str, SpineBone] = {}

    for bone in bones:
        if bone.parent is None:
            # Root bone: local = world
            bone.world_x = bone.x
            bone.world_y = bone.y
            bone.world_rotation = bone.rotation
            bone.world_scale_x = bone.scale_x
            bone.world_scale_y = bone.scale_y
        else:
            parent = bone_map.get(bone.parent)
            if parent is None:
                logger.warning(
                    "Bone %s references missing parent %s, treating as root",
                    bone.name,
                    bone.parent,
                )
                bone.world_x = bone.x
                bone.world_y = bone.y
                bone.world_rotation = bone.rotation
                bone.world_scale_x = bone.scale_x
                bone.world_scale_y = bone.scale_y
            else:
                # Apply parent world transform to local offset
                rad = math.radians(parent.world_rotation)
                cos_r = math.cos(rad)
                sin_r = math.sin(rad)

                # Rotate and scale local position by parent's world transform
                local_x = bone.x * parent.world_scale_x
                local_y = bone.y * parent.world_scale_y
                bone.world_x = parent.world_x + local_x * cos_r - local_y * sin_r
                bone.world_y = parent.world_y + local_x * sin_r + local_y * cos_r
                bone.world_rotation = parent.world_rotation + bone.rotation
                bone.world_scale_x = parent.world_scale_x * bone.scale_x
                bone.world_scale_y = parent.world_scale_y * bone.scale_y

        bone_map[bone.name] = bone

    return bone_map


# ---------------------------------------------------------------------------
# Bone-to-region mapping
# ---------------------------------------------------------------------------


def map_spine_bone(bone_name: str) -> tuple[str, RegionId]:
    """Map a Spine bone name to a Strata region using pattern matching.

    Args:
        bone_name: Spine bone name.

    Returns:
        Tuple of (region_name, region_id). Returns ("UNMAPPED", -1) if
        no pattern matches.
    """
    for compiled_pattern, region_name in _COMPILED_SPINE_PATTERNS:
        if compiled_pattern.search(bone_name):
            region_id = REGION_NAME_TO_ID[region_name]
            return region_name, region_id
    return "UNMAPPED", -1


def _map_all_bones(
    bones: list[SpineBone],
) -> tuple[dict[str, RegionId], list[str]]:
    """Map all Spine bones to Strata regions.

    Returns:
        Tuple of (bone_to_region mapping, unmapped bone names).
    """
    bone_to_region: dict[str, RegionId] = {}
    unmapped: list[str] = []

    for bone in bones:
        _, region_id = map_spine_bone(bone.name)
        if region_id >= 0:
            bone_to_region[bone.name] = region_id
        else:
            unmapped.append(bone.name)

    mapped_count = len(bone_to_region)
    total = len(bones)
    logger.info(
        "Spine bone mapping: %d/%d mapped (%.0f%%), %d unmapped",
        mapped_count,
        total,
        (mapped_count / total * 100) if total else 0,
        len(unmapped),
    )
    if unmapped:
        logger.warning("Unmapped Spine bones: %s", unmapped)

    return bone_to_region, unmapped


def _map_slots_to_regions(
    slots: list[SpineSlot],
    bone_to_region: dict[str, RegionId],
) -> dict[str, RegionId]:
    """Map each slot to a region via its bone's region assignment.

    Args:
        slots: The Spine project's slot list.
        bone_to_region: Bone name → region ID mapping.

    Returns:
        Slot name → region ID mapping. Slots whose bones are unmapped
        are excluded.
    """
    slot_to_region: dict[str, RegionId] = {}
    for slot in slots:
        region_id = bone_to_region.get(slot.bone, -1)
        if region_id < 0:
            # Try mapping by slot name itself
            _, region_id = map_spine_bone(slot.name)
        if region_id >= 0:
            slot_to_region[slot.name] = region_id
    return slot_to_region


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _load_part_image(
    images_dir: Path,
    attachment: SpineAttachment,
) -> Image.Image | None:
    """Load a part image for a Spine attachment.

    Tries the attachment's path first, then the attachment name, looking
    for .png files in the images directory.

    Args:
        images_dir: Directory containing the part images.
        attachment: The SpineAttachment to load the image for.

    Returns:
        PIL Image in RGBA mode, or None if not found.
    """
    # Determine the image name to look for
    image_name = attachment.path or attachment.name

    # Try with and without .png extension
    candidates = [
        images_dir / f"{image_name}.png",
        images_dir / image_name,
    ]

    for candidate in candidates:
        if candidate.exists():
            try:
                img = Image.open(candidate).convert("RGBA")
                return img
            except Exception:
                logger.warning("Failed to load image %s", candidate)
                return None

    logger.debug("Part image not found for attachment %s", attachment.name)
    return None


# ---------------------------------------------------------------------------
# Viewport computation (shared by compositing and joint extraction)
# ---------------------------------------------------------------------------

_VIEWPORT_PADDING: float = 0.05  # fraction of span added as padding

_BBOX_CORNERS: list[tuple[int, int]] = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


@dataclass
class Viewport:
    """Maps Spine world coordinates to pixel coordinates."""

    min_x: float
    min_y: float
    pixel_scale: float
    offset_x: float
    offset_y: float
    resolution: int

    def spine_to_pixel(self, sx: float, sy: float) -> tuple[float, float]:
        """Convert Spine world coords to pixel coords (Y-flipped)."""
        px = (sx - self.min_x) * self.pixel_scale + self.offset_x
        py = self.resolution - (
            (sy - self.min_y) * self.pixel_scale + self.offset_y
        )
        return px, py


def _compute_viewport(
    project: SpineProject,
    bone_map: dict[str, SpineBone],
    attach_lookup: dict[tuple[str, str], SpineAttachment],
    resolution: int,
) -> Viewport | None:
    """Compute the viewport that frames the character in the output image.

    Collects bone positions and attachment bounding box corners, applies
    padding, and returns a Viewport that maps Spine world coords to pixels.

    Returns:
        Viewport, or None if no points could be computed.
    """
    all_points: list[tuple[float, float]] = []
    for bone in project.bones:
        wb = bone_map[bone.name]
        all_points.append((wb.world_x, wb.world_y))

    for slot in project.slots:
        if slot.attachment is None:
            continue
        attach = attach_lookup.get((slot.name, slot.attachment))
        if attach is None:
            continue
        bone = bone_map.get(slot.bone)
        if bone is None:
            continue

        rad = math.radians(bone.world_rotation + attach.rotation)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        hw = attach.width * abs(attach.scale_x) * abs(bone.world_scale_x) / 2
        hh = attach.height * abs(attach.scale_y) * abs(bone.world_scale_y) / 2

        local_x = attach.x * bone.world_scale_x
        local_y = attach.y * bone.world_scale_y
        bone_rad = math.radians(bone.world_rotation)
        bone_cos = math.cos(bone_rad)
        bone_sin = math.sin(bone_rad)
        cx = bone.world_x + local_x * bone_cos - local_y * bone_sin
        cy = bone.world_y + local_x * bone_sin + local_y * bone_cos

        for sx, sy in _BBOX_CORNERS:
            px = cx + (sx * hw) * cos_r - (sy * hh) * sin_r
            py = cy + (sx * hw) * sin_r + (sy * hh) * cos_r
            all_points.append((px, py))

    if not all_points:
        return None

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span_x = max_x - min_x or 1.0
    span_y = max_y - min_y or 1.0
    pad_x = span_x * _VIEWPORT_PADDING
    pad_y = span_y * _VIEWPORT_PADDING
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y
    span_x = max_x - min_x
    span_y = max_y - min_y

    pixel_scale = resolution / max(span_x, span_y)
    offset_x = (resolution - span_x * pixel_scale) / 2
    offset_y = (resolution - span_y * pixel_scale) / 2

    return Viewport(
        min_x=min_x,
        min_y=min_y,
        pixel_scale=pixel_scale,
        offset_x=offset_x,
        offset_y=offset_y,
        resolution=resolution,
    )


# ---------------------------------------------------------------------------
# Image compositing
# ---------------------------------------------------------------------------


def _composite_character(
    project: SpineProject,
    bone_map: dict[str, SpineBone],
    skin_attachments: list[SpineAttachment],
    images_dir: Path,
    resolution: int = RENDER_RESOLUTION,
) -> tuple[Image.Image, np.ndarray]:
    """Composite the character from part images at the setup pose.

    Renders each slot's attachment at its bone's world transform position,
    in slot draw order (back to front). Builds a segmentation mask using
    numpy slicing (each attachment's opaque pixels get its region ID).

    Args:
        project: Parsed Spine project.
        bone_map: Bone name → SpineBone with world transforms computed.
        skin_attachments: Attachments from the active skin.
        images_dir: Directory containing part images.
        resolution: Output image resolution (square).

    Returns:
        Tuple of (RGBA image, uint8 segmentation mask).
    """
    attach_lookup: dict[tuple[str, str], SpineAttachment] = {}
    for attach in skin_attachments:
        attach_lookup[(attach.slot_name, attach.name)] = attach

    viewport = _compute_viewport(project, bone_map, attach_lookup, resolution)
    if viewport is None:
        logger.warning("No points found for character compositing")
        canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
        mask_arr = np.zeros((resolution, resolution), dtype=np.uint8)
        return canvas, mask_arr

    canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    mask_arr = np.zeros((resolution, resolution), dtype=np.uint8)

    bone_to_region, _ = _map_all_bones(project.bones)
    slot_to_region = _map_slots_to_regions(project.slots, bone_to_region)

    for slot in project.slots:
        if slot.attachment is None:
            continue

        attach = attach_lookup.get((slot.name, slot.attachment))
        if attach is None:
            continue

        bone = bone_map.get(slot.bone)
        if bone is None:
            continue

        part_img = _load_part_image(images_dir, attach)
        if part_img is None:
            continue

        region_id = slot_to_region.get(slot.name, 0)

        bone_rad = math.radians(bone.world_rotation)
        bone_cos = math.cos(bone_rad)
        bone_sin = math.sin(bone_rad)
        local_x = attach.x * bone.world_scale_x
        local_y = attach.y * bone.world_scale_y
        world_cx = bone.world_x + local_x * bone_cos - local_y * bone_sin
        world_cy = bone.world_y + local_x * bone_sin + local_y * bone_cos

        total_rotation = bone.world_rotation + attach.rotation
        total_scale_x = bone.world_scale_x * attach.scale_x
        total_scale_y = bone.world_scale_y * attach.scale_y

        scaled_w = max(1, round(part_img.width * abs(total_scale_x) * viewport.pixel_scale))
        scaled_h = max(1, round(part_img.height * abs(total_scale_y) * viewport.pixel_scale))

        try:
            scaled_img = part_img.resize((scaled_w, scaled_h), Image.BILINEAR)
        except Exception:
            continue

        if total_scale_x < 0:
            scaled_img = scaled_img.transpose(Image.FLIP_LEFT_RIGHT)
        if total_scale_y < 0:
            scaled_img = scaled_img.transpose(Image.FLIP_TOP_BOTTOM)

        if abs(total_rotation) > 0.01:
            scaled_img = scaled_img.rotate(
                total_rotation,
                resample=Image.BILINEAR,
                expand=True,
            )

        center_px, center_py = viewport.spine_to_pixel(world_cx, world_cy)
        paste_x = round(center_px - scaled_img.width / 2)
        paste_y = round(center_py - scaled_img.height / 2)

        # Composite onto canvas (uses PIL alpha compositing)
        canvas.paste(scaled_img, (paste_x, paste_y), scaled_img)

        # Paint mask using numpy slicing (much faster than per-pixel loop)
        if region_id > 0:
            arr = np.array(scaled_img)
            alpha = arr[:, :, 3]

            # Compute the overlap region between the part and the canvas
            src_y_start = max(0, -paste_y)
            src_x_start = max(0, -paste_x)
            src_y_end = min(scaled_img.height, resolution - paste_y)
            src_x_end = min(scaled_img.width, resolution - paste_x)

            if src_y_end <= src_y_start or src_x_end <= src_x_start:
                continue

            dst_y_start = paste_y + src_y_start
            dst_x_start = paste_x + src_x_start
            dst_y_end = paste_y + src_y_end
            dst_x_end = paste_x + src_x_end

            alpha_slice = alpha[src_y_start:src_y_end, src_x_start:src_x_end]
            opaque = alpha_slice > 0
            mask_arr[dst_y_start:dst_y_end, dst_x_start:dst_x_end][opaque] = region_id

    return canvas, mask_arr


# ---------------------------------------------------------------------------
# Joint extraction
# ---------------------------------------------------------------------------


def _extract_joints(
    bone_map: dict[str, SpineBone],
    bone_to_region: dict[str, RegionId],
    spine_to_pixel_fn: Any,
    resolution: int = RENDER_RESOLUTION,
) -> dict[str, Any]:
    """Extract 2D joint positions from Spine bone world transforms.

    Args:
        bone_map: Bone name → SpineBone with world transforms.
        bone_to_region: Bone name → region ID mapping.
        spine_to_pixel_fn: Callable converting (spine_x, spine_y) to (px, py).
        resolution: Image resolution.

    Returns:
        Joint data dict matching the 3D pipeline schema.
    """
    # Invert: region_id → list of bone names
    region_to_bones: dict[RegionId, list[str]] = {}
    for bone_name, region_id in bone_to_region.items():
        if region_id == 0:
            continue
        region_to_bones.setdefault(region_id, []).append(bone_name)

    joints: dict[str, dict] = {}
    positions: dict[str, tuple[int, int]] = {}
    visibility: dict[str, bool] = {}

    for region_id in range(1, NUM_JOINT_REGIONS + 1):
        region_name = REGION_NAMES[region_id]
        bones_in_region = region_to_bones.get(region_id, [])

        if not bones_in_region:
            joints[region_name] = {
                "position": [-1, -1],
                "confidence": 0.0,
                "visible": False,
            }
            positions[region_name] = (-1, -1)
            visibility[region_name] = False
            continue

        # Use the first bone (usually the primary one for the region)
        primary_bone_name = bones_in_region[0]
        bone = bone_map[primary_bone_name]
        px, py = spine_to_pixel_fn(bone.world_x, bone.world_y)
        px_int = round(px)
        py_int = round(py)

        in_bounds = 0 <= px_int < resolution and 0 <= py_int < resolution
        clamped_x = max(0, min(px_int, resolution - 1))
        clamped_y = max(0, min(py_int, resolution - 1))

        joints[region_name] = {
            "position": [clamped_x, clamped_y],
            "confidence": 1.0 if in_bounds else 0.5,
            "visible": in_bounds,
        }
        positions[region_name] = (clamped_x, clamped_y)
        visibility[region_name] = in_bounds

    bbox = _compute_bbox(positions, visibility, resolution)

    visible_count = sum(1 for v in visibility.values() if v)
    logger.info(
        "Spine joints: %d extracted (%d visible, %d missing)",
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
# Main entry points
# ---------------------------------------------------------------------------


def process_spine_project(
    json_path: Path,
    images_dir: Path | None = None,
    resolution: int = RENDER_RESOLUTION,
    skin_name: str = "default",
) -> SpineParseResult | None:
    """Process a single Spine project file into Strata dataset outputs.

    Args:
        json_path: Path to the .spine or .json file.
        images_dir: Directory containing part images. If None, inferred
            from the Spine JSON's ``skeleton.images`` field, resolved
            relative to the JSON file's parent directory.
        resolution: Output image resolution (square).
        skin_name: Which skin to use for visual variant.

    Returns:
        SpineParseResult with image, mask, and joint data, or None on error.
    """
    try:
        project = parse_spine_json(json_path)
    except (json.JSONDecodeError, KeyError) as exc:
        logger.error("Failed to parse Spine JSON %s: %s", json_path, exc)
        return None

    # Resolve images directory
    if images_dir is None:
        # Spine JSON often has "images": "./images/" in skeleton section
        raw_images_dir = project.images_dir.rstrip("/")
        images_dir = (json_path.parent / raw_images_dir).resolve()

    if not images_dir.is_dir():
        logger.error("Images directory not found: %s", images_dir)
        return None

    # Get skin attachments
    skin_data = project.skins.get(skin_name)
    if skin_data is None:
        available = list(project.skins.keys())
        logger.error(
            "Skin '%s' not found in %s. Available: %s",
            skin_name,
            json_path.name,
            available,
        )
        return None

    # Compute world transforms
    bone_map = _compute_world_transforms(project.bones)

    # Map bones to regions
    bone_to_region, unmapped_bones = _map_all_bones(project.bones)
    slot_to_region = _map_slots_to_regions(project.slots, bone_to_region)

    # Composite character and build mask
    image, mask = _composite_character(
        project,
        bone_map,
        skin_data,
        images_dir,
        resolution,
    )

    # Reuse the same viewport for joint extraction
    attach_lookup: dict[tuple[str, str], SpineAttachment] = {
        (a.slot_name, a.name): a for a in skin_data
    }
    viewport = _compute_viewport(project, bone_map, attach_lookup, resolution)

    if viewport is not None:
        spine_to_pixel = viewport.spine_to_pixel
    else:
        half = float(resolution) / 2

        def spine_to_pixel(sx: float, sy: float) -> tuple[float, float]:
            return half, half

    joint_data = _extract_joints(
        bone_map,
        bone_to_region,
        spine_to_pixel,
        resolution,
    )

    char_id = f"spine_{project.name}"
    if skin_name != "default":
        char_id = f"spine_{project.name}_{skin_name}"

    return SpineParseResult(
        char_id=char_id,
        skin_name=skin_name,
        image=image,
        mask=mask,
        joint_data=joint_data,
        bone_to_region=bone_to_region,
        unmapped_bones=unmapped_bones,
        slot_to_region=slot_to_region,
    )


def process_spine_directory(
    spine_dir: Path,
    output_dir: Path,
    *,
    resolution: int = RENDER_RESOLUTION,
    styles: list[str] | None = None,
    only_new: bool = False,
) -> list[SpineParseResult]:
    """Process all Spine projects in a directory.

    Discovers .spine and .json files, processes each one, and saves
    outputs using the exporter module.

    Args:
        spine_dir: Directory containing Spine project files.
        output_dir: Root dataset output directory.
        resolution: Output image resolution.
        styles: Art styles to generate. If None, only saves the original.
        only_new: Skip existing files.

    Returns:
        List of successful SpineParseResult objects.
    """
    from . import exporter
    from .style_augmentor import apply_post_render_style

    # Discover Spine files
    spine_files: list[Path] = sorted(
        p
        for p in spine_dir.iterdir()
        if p.suffix in (".spine", ".json") and p.is_file()
    )

    if not spine_files:
        logger.warning("No Spine files found in %s", spine_dir)
        return []

    logger.info("Found %d Spine files in %s", len(spine_files), spine_dir)

    exporter.ensure_output_dirs(output_dir)
    results: list[SpineParseResult] = []

    for json_path in spine_files:
        # Detect available skins
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            raw_skins = raw.get("skins", [])
            if isinstance(raw_skins, list):
                skin_names = [s.get("name", "default") for s in raw_skins]
            elif isinstance(raw_skins, dict):
                skin_names = list(raw_skins.keys())
            else:
                skin_names = ["default"]
        except (json.JSONDecodeError, KeyError):
            skin_names = ["default"]

        for skin_name in skin_names:
            result = process_spine_project(
                json_path,
                resolution=resolution,
                skin_name=skin_name,
            )
            if result is None:
                continue

            pose_index = 0  # Default/setup pose

            # Save mask
            exporter.save_mask(
                result.mask,
                output_dir,
                result.char_id,
                pose_index,
                only_new=only_new,
            )

            # Save joints
            exporter.save_joints(
                result.joint_data,
                output_dir,
                result.char_id,
                pose_index,
                only_new=only_new,
            )

            # Save original image as "flat" style
            exporter.save_image(
                result.image,
                output_dir,
                result.char_id,
                pose_index,
                "flat",
                only_new=only_new,
            )

            # Apply post-render styles if requested
            if styles:
                for style in styles:
                    if style == "flat":
                        continue  # already saved
                    if style in ("pixel", "painterly", "sketch"):
                        styled = apply_post_render_style(result.image, style)
                        exporter.save_image(
                            styled,
                            output_dir,
                            result.char_id,
                            pose_index,
                            style,
                            only_new=only_new,
                        )

            # Save source metadata
            exporter.save_source_metadata(
                output_dir,
                result.char_id,
                source="spine",
                name=result.char_id,
                license_="",
                attribution="",
                bone_mapping="auto",
                unmapped_bones=result.unmapped_bones,
                character_type="humanoid",
                notes=f"Spine project, skin={result.skin_name}",
                only_new=only_new,
            )

            results.append(result)
            logger.info(
                "Processed Spine character %s (skin=%s): "
                "%d bones mapped, %d unmapped",
                result.char_id,
                result.skin_name,
                len(result.bone_to_region),
                len(result.unmapped_bones),
            )

    logger.info(
        "Spine processing complete: %d/%d characters succeeded",
        len(results),
        len(spine_files),
    )

    return results

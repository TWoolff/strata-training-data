"""Render additional camera angles for HumanRig samples using Blender.

This module is the Blender-side counterpart to ``humanrig_adapter.py``.
It renders ``image.png`` for camera angles that have no pre-rendered image
in the dataset (three_quarter, side, back) by importing each sample's
``rigged.glb`` and rendering orthographic views.

The ``humanrig_adapter.py`` pure-Python pass must run first so that
``joints.json`` and ``metadata.json`` already exist in each output
example directory.  This script fills in the missing ``image.png`` and
updates ``has_rendered_image`` in ``metadata.json``.

Camera convention (derived from HumanRig's extrinsic matrices):
- Character stands upright: X = lateral, Y = depth, Z = height.
- Front view: camera on the -Y side, looking toward +Y.
- Orbiting: camera sweeps around world Z while always looking at origin.

Render settings:
- Engine: Cycles CPU (64 samples) — reliable in headless mode.
- Orthographic camera, ortho_scale = 1.3 world units.
- Transparent background (RGBA PNG).
- Two directional lights for even coverage across all angles.

Requires Blender 4.0+ (bpy). Run via::

    blender --background --python run_humanrig_render.py -- \\
        --output_dir ./output/humanrig \\
        --input_dir /path/to/humanrig_opensource_final \\
        --angles three_quarter,side,back \\
        --only_new
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RENDER_RESOLUTION = 512
CYCLES_SAMPLES = 64
ORTHO_SCALE = 1.3   # world units across the rendered frame
CAMERA_DIST = 2.0   # camera distance from origin

# Camera angles: label → azimuth in degrees.
# azimuth=0: front (camera at -Y looking toward +Y)
# azimuth=90: right side (camera at +X looking toward -X)
# azimuth=180: back (camera at +Y looking toward -Y)
ANGLE_AZIMUTHS: dict[str, int] = {
    "front": 0,
    "three_quarter": 45,
    "side": 90,
    "back": 180,
}


# ---------------------------------------------------------------------------
# Blender scene setup (called once per GLB)
# ---------------------------------------------------------------------------


def _setup_scene() -> None:
    """Configure render settings on the current Blender scene."""
    import bpy

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = CYCLES_SAMPLES
    scene.cycles.device = "CPU"
    scene.render.resolution_x = RENDER_RESOLUTION
    scene.render.resolution_y = RENDER_RESOLUTION
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"


def _setup_lights() -> None:
    """Add two directional lights for even coverage from all angles."""
    import bpy

    scene = bpy.context.scene
    for energy, rx, rz in [
        (5.0, 45, 30),    # front-top-left
        (5.0, 45, 210),   # front-top-right (opposite side)
    ]:
        light = bpy.data.lights.new("Sun", "SUN")
        light.energy = energy
        light.angle = math.radians(5)
        obj = bpy.data.objects.new("Sun", light)
        scene.collection.objects.link(obj)
        obj.rotation_euler = (
            math.radians(rx),
            0,
            math.radians(rz),
        )


def _import_glb(glb_path: Path) -> bool:
    """Import a GLB file into the current scene.

    Removes the Icosphere background object included in every HumanRig GLB
    and sets all materials to OPAQUE (the GLB uses HASHED blend mode which
    renders incorrectly in headless Cycles).

    Returns:
        True if import succeeded, False on error.
    """
    import bpy

    try:
        bpy.ops.import_scene.gltf(filepath=str(glb_path))
    except Exception as exc:
        logger.warning("GLB import failed for %s: %s", glb_path, exc)
        return False

    # Remove the background sphere that ships with every HumanRig GLB.
    for obj in list(bpy.data.objects):
        if obj.name == "Icosphere":
            bpy.data.objects.remove(obj, do_unlink=True)

    # Fix material blend mode — HASHED causes transparent artefacts in Cycles.
    for mat in bpy.data.materials:
        mat.blend_method = "OPAQUE"

    return True


def _setup_world() -> None:
    """Set world background to pure black (transparent background)."""
    import bpy

    scene = bpy.context.scene
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.0


def _make_camera() -> Any:
    """Create an orthographic camera and set it as the scene camera."""
    import bpy

    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ORTHO_SCALE
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return cam_obj


def _set_camera_orbit(cam_obj: Any, azimuth_deg: float) -> None:
    """Position and orient the camera to orbit around world Z.

    Camera always points at the world origin with world Z as up.
    azimuth_deg=0 places the camera on the -Y side (front view).

    Args:
        cam_obj: Blender camera object.
        azimuth_deg: Horizontal rotation in degrees (0=front, 90=side, 180=back).
    """
    from mathutils import Matrix, Vector

    theta = math.radians(azimuth_deg)
    cam_x = math.sin(theta) * CAMERA_DIST
    cam_y = -math.cos(theta) * CAMERA_DIST
    cam_obj.location = Vector((cam_x, cam_y, 0.0))

    # Build look-at rotation: camera -Z points toward origin, world Z is up.
    forward = (-Vector((cam_x, cam_y, 0.0))).normalized()
    world_up = Vector((0.0, 0.0, 1.0))
    right = forward.cross(world_up).normalized()
    up = right.cross(forward).normalized()

    rot_mat = Matrix([
        [right.x,    right.y,    right.z,    0.0],
        [up.x,       up.y,       up.z,       0.0],
        [-forward.x, -forward.y, -forward.z, 0.0],
        [0.0,        0.0,        0.0,        1.0],
    ]).transposed()
    cam_obj.rotation_euler = rot_mat.to_3x3().to_euler()


# ---------------------------------------------------------------------------
# Per-sample rendering
# ---------------------------------------------------------------------------


def render_sample(
    glb_path: Path,
    output_example_dirs: dict[str, Path],
    *,
    only_new: bool = False,
) -> int:
    """Render additional camera angle images for one HumanRig sample.

    Imports the GLB once, then renders each requested angle whose
    ``image.png`` is missing from the corresponding output directory.

    Args:
        glb_path: Path to ``rigged.glb``.
        output_example_dirs: Mapping of angle_label → example directory.
            Only angles whose directory exists and lacks ``image.png``
            are rendered (unless ``only_new=False``).
        only_new: If True, skip angles that already have ``image.png``.

    Returns:
        Number of images rendered.
    """
    import bpy

    # Determine which angles actually need rendering.
    to_render: dict[str, tuple[Path, int]] = {}
    for label, example_dir in output_example_dirs.items():
        if not example_dir.is_dir():
            logger.debug("Example dir not found, skipping %s", example_dir)
            continue
        img_path = example_dir / "image.png"
        if only_new and img_path.exists():
            logger.debug("Skipping existing image %s", img_path)
            continue
        azimuth = ANGLE_AZIMUTHS.get(label)
        if azimuth is None:
            logger.warning("Unknown angle label %r", label)
            continue
        to_render[label] = (example_dir, azimuth)

    if not to_render:
        return 0

    # Reset scene and import GLB.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _setup_scene()
    _setup_world()
    _setup_lights()

    if not _import_glb(glb_path):
        return 0

    cam_obj = _make_camera()

    rendered = 0
    for label, (example_dir, azimuth) in to_render.items():
        _set_camera_orbit(cam_obj, azimuth)
        scene = bpy.context.scene
        scene.render.filepath = str(example_dir / "image.png")

        try:
            bpy.ops.render.render(write_still=True)
        except Exception as exc:
            logger.warning("Render failed for %s %s: %s", glb_path.parent.name, label, exc)
            continue

        # Update metadata to mark image as available.
        _patch_metadata(example_dir)
        rendered += 1

    return rendered


def _patch_metadata(example_dir: Path) -> None:
    """Set ``has_rendered_image=True`` and remove ``rendered_image`` from
    ``missing_annotations`` in the example's ``metadata.json``."""
    meta_path = example_dir / "metadata.json"
    if not meta_path.is_file():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["has_rendered_image"] = True
        missing = meta.get("missing_annotations", [])
        if "rendered_image" in missing:
            missing.remove("rendered_image")
            meta["missing_annotations"] = missing
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to patch metadata in %s: %s", example_dir, exc)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def render_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    angles: list[str] | None = None,
    only_new: bool = False,
    max_samples: int = 0,
) -> tuple[int, int, int]:
    """Render additional angle images for all HumanRig samples.

    Args:
        input_dir: ``humanrig_opensource_final/`` directory.
        output_dir: Root output directory (same as used by the Python adapter).
        angles: Angle labels to render. Defaults to
            ``["three_quarter", "side", "back"]`` (non-front angles only).
        only_new: Skip samples whose output image already exists.
        max_samples: Maximum samples to process (0 = all).

    Returns:
        Tuple of (rendered_count, skipped_count, error_count).
    """
    if angles is None:
        angles = ["three_quarter", "side", "back"]

    rendered_total = 0
    skipped_total = 0
    errors = 0

    # Discover sample directories.
    sample_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )

    if max_samples > 0:
        sample_dirs = sample_dirs[:max_samples]

    total = len(sample_dirs)
    logger.info(
        "Rendering %d samples × %d angles from %s",
        total, len(angles), input_dir,
    )

    for i, sample_dir in enumerate(sample_dirs):
        glb_path = sample_dir / "rigged.glb"
        if not glb_path.is_file():
            logger.warning("No rigged.glb in %s — skipping", sample_dir)
            errors += 1
            continue

        sample_id = int(sample_dir.name)

        # Build mapping of angle_label → output example directory.
        example_dirs: dict[str, Path] = {}
        for label in angles:
            example_id = f"humanrig_{sample_id:05d}_{label}"
            example_dirs[label] = output_dir / example_id

        try:
            n = render_sample(
                glb_path,
                example_dirs,
                only_new=only_new,
            )
        except Exception as exc:
            logger.warning("Error rendering sample %d: %s", sample_id, exc)
            errors += 1
            continue

        rendered_total += n
        skipped_total += len(angles) - n

        if (i + 1) % 50 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            logger.info(
                "Progress: %d/%d samples (%.1f%%) — %d rendered, %d skipped",
                i + 1, total, pct, rendered_total, skipped_total,
            )

    return rendered_total, skipped_total, errors

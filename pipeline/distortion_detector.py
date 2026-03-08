"""Detect mesh distortion in animated FBX files (withSkin baked animations).

Compares the mesh bounding box at each sampled animation frame against the
rest frame (frame 0 or 1) to flag characters whose rigs produce degenerate
deformations (exploding vertices, collapsing limbs, etc.).

This module requires Blender's ``bpy`` API but has no other pipeline
dependencies, so it can be imported and tested independently.

Usage inside Blender::

    from pipeline.distortion_detector import check_distortion
    ok, reason = check_distortion(Path("walking_withSkin.fbx"))

"""

from __future__ import annotations

import logging
from pathlib import Path

import bpy  # type: ignore[import-untyped]
from mathutils import Vector  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

DEFAULT_HEIGHT_THRESHOLD: float = 0.2   # 20 % height change from rest
DEFAULT_WIDTH_THRESHOLD: float = 1.0    # 100 % width change from rest
DEFAULT_VERTEX_TRAVEL_FACTOR: float = 3.0  # vertex moves > 3x char height
DEFAULT_FRAME_SAMPLE_STEP: int = 4       # sample every 4th frame


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_meshes() -> list[bpy.types.Object]:
    """Return all mesh objects in the current scene."""
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def _get_armature() -> bpy.types.Object | None:
    """Return the first armature in the current scene, or None."""
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            return obj
    return None


def _evaluated_bbox(
    meshes: list[bpy.types.Object],
    depsgraph: bpy.types.Depsgraph,
) -> tuple[Vector, Vector]:
    """Compute the world-space AABB of *evaluated* meshes at the current frame.

    Uses the dependency graph to get deformed (posed) geometry.

    Args:
        meshes: Scene mesh objects (un-evaluated).
        depsgraph: A current dependency graph.

    Returns:
        (bbox_min, bbox_max) as ``mathutils.Vector``.
    """
    all_corners: list[Vector] = []
    for mesh_obj in meshes:
        eval_obj = mesh_obj.evaluated_get(depsgraph)
        for corner in eval_obj.bound_box:
            all_corners.append(eval_obj.matrix_world @ Vector(corner))

    xs = [v.x for v in all_corners]
    ys = [v.y for v in all_corners]
    zs = [v.z for v in all_corners]

    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def _evaluated_vertex_positions(
    meshes: list[bpy.types.Object],
    depsgraph: bpy.types.Depsgraph,
) -> list[Vector]:
    """Collect all world-space vertex positions from evaluated meshes.

    Args:
        meshes: Scene mesh objects.
        depsgraph: A current dependency graph.

    Returns:
        Flat list of ``Vector`` positions.
    """
    positions: list[Vector] = []
    for mesh_obj in meshes:
        eval_obj = mesh_obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        for vert in eval_mesh.vertices:
            positions.append(eval_obj.matrix_world @ vert.co)
        eval_obj.to_mesh_clear()
    return positions


def _animation_range(scene: bpy.types.Scene) -> tuple[int, int]:
    """Return (frame_start, frame_end) from the scene's active action or frame range.

    Prefers the armature's action range if available, otherwise falls back
    to the scene's frame_start / frame_end.
    """
    armature = _get_armature()
    if armature and armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
        start = int(action.frame_range[0])
        end = int(action.frame_range[1])
        return start, end
    return scene.frame_start, scene.frame_end


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_distortion(
    fbx_path: Path,
    *,
    height_threshold: float = DEFAULT_HEIGHT_THRESHOLD,
    width_threshold: float = DEFAULT_WIDTH_THRESHOLD,
    vertex_travel_factor: float = DEFAULT_VERTEX_TRAVEL_FACTOR,
    frame_step: int = DEFAULT_FRAME_SAMPLE_STEP,
    clear_scene_before: bool = True,
) -> tuple[bool, str]:
    """Check whether a withSkin FBX produces distorted mesh at animated frames.

    Imports the FBX, samples frames across the animation range, and compares
    the mesh bounding box and vertex positions against the rest frame.

    Args:
        fbx_path: Path to a withSkin FBX file.
        height_threshold: Max allowed fractional height change (0.2 = 20 %).
        width_threshold: Max allowed fractional width change (1.0 = 100 %).
        vertex_travel_factor: Max vertex travel as a multiple of rest height.
        frame_step: Sample every N-th frame across the animation range.
        clear_scene_before: If True, clear the scene before importing.

    Returns:
        ``(is_ok, reason)`` — ``is_ok`` is True when the mesh looks fine.
        ``reason`` is an empty string when OK, otherwise a human-readable
        explanation of the distortion.
    """
    fbx_path = Path(fbx_path)
    if not fbx_path.is_file():
        return False, f"FBX file not found: {fbx_path}"

    # --- Import ---
    if clear_scene_before:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

    try:
        bpy.ops.import_scene.fbx(filepath=str(fbx_path))
    except Exception as exc:
        return False, f"FBX import failed: {exc}"

    scene = bpy.context.scene
    meshes = _get_meshes()
    if not meshes:
        return False, "No meshes found in FBX"

    # --- Rest frame measurements ---
    frame_start, frame_end = _animation_range(scene)
    rest_frame = frame_start  # typically 0 or 1

    scene.frame_set(rest_frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()

    rest_min, rest_max = _evaluated_bbox(meshes, depsgraph)
    rest_height = rest_max.z - rest_min.z
    rest_width = rest_max.x - rest_min.x

    if rest_height < 1e-6:
        return False, f"Near-zero rest height ({rest_height:.6f})"

    rest_verts = _evaluated_vertex_positions(meshes, depsgraph)
    max_travel_allowed = rest_height * vertex_travel_factor

    # --- Sample animated frames ---
    sample_frames = list(range(frame_start, frame_end + 1, frame_step))
    # Always include the last frame
    if frame_end not in sample_frames:
        sample_frames.append(frame_end)
    # Skip the rest frame itself
    sample_frames = [f for f in sample_frames if f != rest_frame]

    if not sample_frames:
        # Single-frame animation — nothing to check
        return True, ""

    for frame in sample_frames:
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()

        frame_min, frame_max = _evaluated_bbox(meshes, depsgraph)
        frame_height = frame_max.z - frame_min.z
        frame_width = frame_max.x - frame_min.x

        # Height check
        height_change = abs(frame_height - rest_height) / rest_height
        if height_change > height_threshold:
            reason = (
                f"Frame {frame}: height changed {height_change:.1%} "
                f"(rest={rest_height:.3f}, frame={frame_height:.3f}, "
                f"threshold={height_threshold:.0%})"
            )
            logger.warning("Distortion detected in %s: %s", fbx_path.name, reason)
            return False, reason

        # Width check
        if rest_width > 1e-6:
            width_change = abs(frame_width - rest_width) / rest_width
            if width_change > width_threshold:
                reason = (
                    f"Frame {frame}: width changed {width_change:.1%} "
                    f"(rest={rest_width:.3f}, frame={frame_width:.3f}, "
                    f"threshold={width_threshold:.0%})"
                )
                logger.warning("Distortion detected in %s: %s", fbx_path.name, reason)
                return False, reason

        # Vertex travel check
        frame_verts = _evaluated_vertex_positions(meshes, depsgraph)
        if len(frame_verts) == len(rest_verts):
            for i, (rv, fv) in enumerate(zip(rest_verts, frame_verts)):
                travel = (fv - rv).length
                if travel > max_travel_allowed:
                    reason = (
                        f"Frame {frame}: vertex {i} traveled {travel:.3f} "
                        f"(max allowed={max_travel_allowed:.3f}, "
                        f"rest_height={rest_height:.3f})"
                    )
                    logger.warning(
                        "Distortion detected in %s: %s", fbx_path.name, reason
                    )
                    return False, reason

    logger.info(
        "Distortion check passed for %s (%d frames sampled)",
        fbx_path.name,
        len(sample_frames),
    )
    return True, ""


def check_distortion_in_scene(
    *,
    height_threshold: float = DEFAULT_HEIGHT_THRESHOLD,
    width_threshold: float = DEFAULT_WIDTH_THRESHOLD,
    vertex_travel_factor: float = DEFAULT_VERTEX_TRAVEL_FACTOR,
    frame_step: int = DEFAULT_FRAME_SAMPLE_STEP,
) -> tuple[bool, str]:
    """Check distortion on an already-imported scene (no FBX import).

    Same logic as ``check_distortion`` but operates on the current scene
    contents. Useful when you have already imported the FBX and want to
    avoid re-importing.

    Args:
        height_threshold: Max allowed fractional height change.
        width_threshold: Max allowed fractional width change.
        vertex_travel_factor: Max vertex travel as a multiple of rest height.
        frame_step: Sample every N-th frame.

    Returns:
        ``(is_ok, reason)`` tuple.
    """
    scene = bpy.context.scene
    meshes = _get_meshes()
    if not meshes:
        return False, "No meshes in scene"

    frame_start, frame_end = _animation_range(scene)
    rest_frame = frame_start

    scene.frame_set(rest_frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()

    rest_min, rest_max = _evaluated_bbox(meshes, depsgraph)
    rest_height = rest_max.z - rest_min.z
    rest_width = rest_max.x - rest_min.x

    if rest_height < 1e-6:
        return False, f"Near-zero rest height ({rest_height:.6f})"

    rest_verts = _evaluated_vertex_positions(meshes, depsgraph)
    max_travel_allowed = rest_height * vertex_travel_factor

    sample_frames = list(range(frame_start, frame_end + 1, frame_step))
    if frame_end not in sample_frames:
        sample_frames.append(frame_end)
    sample_frames = [f for f in sample_frames if f != rest_frame]

    if not sample_frames:
        return True, ""

    for frame in sample_frames:
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()

        frame_min, frame_max = _evaluated_bbox(meshes, depsgraph)
        frame_height = frame_max.z - frame_min.z
        frame_width = frame_max.x - frame_min.x

        height_change = abs(frame_height - rest_height) / rest_height
        if height_change > height_threshold:
            return False, (
                f"Frame {frame}: height changed {height_change:.1%} "
                f"(threshold={height_threshold:.0%})"
            )

        if rest_width > 1e-6:
            width_change = abs(frame_width - rest_width) / rest_width
            if width_change > width_threshold:
                return False, (
                    f"Frame {frame}: width changed {width_change:.1%} "
                    f"(threshold={width_threshold:.0%})"
                )

        frame_verts = _evaluated_vertex_positions(meshes, depsgraph)
        if len(frame_verts) == len(rest_verts):
            for i, (rv, fv) in enumerate(zip(rest_verts, frame_verts)):
                travel = (fv - rv).length
                if travel > max_travel_allowed:
                    return False, (
                        f"Frame {frame}: vertex {i} traveled {travel:.3f} "
                        f"(max={max_travel_allowed:.3f})"
                    )

    return True, ""

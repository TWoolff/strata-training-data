"""Load animation FBX files, extract keyframes, and apply poses to armatures.

Supports three pose sources:
- **Animation clips**: Mixamo-style FBX files with baked animation data.
  Evenly-spaced keyframes are sampled and transferred to the character.
- **T-pose**: Built-in rest pose (all bone rotations set to identity).
- **A-pose**: Built-in pose with upper arms rotated ~45° downward.

Animation retargeting works by bone name matching. Mixamo-to-Mixamo is
automatic. For non-Mixamo rigs, bone name prefixes are stripped to improve
matching (reuses ``config.COMMON_PREFIXES``).

Usage::

    from pose_applicator import list_poses, apply_pose, reset_pose

    poses = list_poses(Path("./pose_library"))
    for pose in poses:
        apply_pose(armature, pose, Path("./pose_library"))
        # ... render ...
        reset_pose(armature)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import bpy  # type: ignore[import-untyped]
from mathutils import Euler  # type: ignore[import-untyped]

from config import A_POSE_SHOULDER_ANGLE, COMMON_PREFIXES, KEYFRAMES_PER_CLIP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class PoseInfo:
    """Metadata for a single pose to be applied."""

    name: str
    """Human-readable pose name (e.g. ``"walk_frame_07"``)."""

    source: str
    """Source animation filename (e.g. ``"Walking.fbx"``) or ``"built-in"``."""

    frame: int
    """Source frame number within the animation clip (0 for built-in poses)."""


# ---------------------------------------------------------------------------
# Built-in poses
# ---------------------------------------------------------------------------

TPOSE_INFO = PoseInfo(name="t_pose", source="built-in", frame=0)
APOSE_INFO = PoseInfo(name="a_pose", source="built-in", frame=0)

# Upper arm bone name substrings used to identify shoulders for A-pose
_LEFT_UPPER_ARM_KEYWORDS = ("leftarm", "upper_arm.l", "l_upperarm", "upperarm.l")
_RIGHT_UPPER_ARM_KEYWORDS = ("rightarm", "upper_arm.r", "r_upperarm", "upperarm.r")


# ---------------------------------------------------------------------------
# Bone name normalization (for retargeting)
# ---------------------------------------------------------------------------


def _normalize_bone_name(name: str) -> str:
    """Strip common prefixes and lowercase a bone name for matching.

    Args:
        name: Raw bone name from an armature.

    Returns:
        Lowercased name with known prefixes removed.
    """
    for prefix in COMMON_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.lower()


def _build_name_map(
    source_bones: list[str],
    target_bones: list[str],
) -> dict[str, str]:
    """Build a mapping from source bone names to target bone names.

    First tries exact name matches, then falls back to normalized
    (prefix-stripped, lowercased) comparison.

    Args:
        source_bones: Bone names from the animation armature.
        target_bones: Bone names from the character armature.

    Returns:
        Dict mapping source bone name → target bone name.
    """
    mapping: dict[str, str] = {}
    target_set = set(target_bones)
    target_normalized = {_normalize_bone_name(t): t for t in target_bones}

    for sname in source_bones:
        if sname in target_set:
            mapping[sname] = sname
        else:
            t_match = target_normalized.get(_normalize_bone_name(sname))
            if t_match is not None:
                mapping[sname] = t_match

    return mapping


# ---------------------------------------------------------------------------
# Keyframe sampling
# ---------------------------------------------------------------------------


def _compute_sample_frames(total_frames: int, num_keyframes: int) -> list[int]:
    """Compute evenly-spaced frame indices to sample from an animation.

    Args:
        total_frames: Total number of frames in the animation clip.
        num_keyframes: Desired number of keyframes to extract.

    Returns:
        Sorted list of frame indices (0-based).
    """
    if total_frames <= 0:
        return []
    if num_keyframes <= 0:
        return []
    if num_keyframes >= total_frames:
        return list(range(total_frames))
    if num_keyframes == 1:
        return [0]

    # Evenly space across the range [0, total_frames - 1]
    step = (total_frames - 1) / (num_keyframes - 1)
    return [round(i * step) for i in range(num_keyframes)]


# ---------------------------------------------------------------------------
# Animation FBX loading
# ---------------------------------------------------------------------------


def _import_animation_fbx(fbx_path: Path) -> bpy.types.Object | None:
    """Import an animation FBX and return its armature.

    Only imports the armature (no meshes, cameras, or lights). The imported
    armature is temporary and should be cleaned up after use.

    Args:
        fbx_path: Path to the animation FBX file.

    Returns:
        The imported armature object, or None if import failed.
    """
    # Track objects before import to identify new ones
    existing_objects = set(bpy.data.objects)

    try:
        bpy.ops.import_scene.fbx(
            filepath=str(fbx_path),
            use_anim=True,
            ignore_leaf_bones=True,
        )
    except Exception:
        logger.exception("Failed to import animation FBX: %s", fbx_path)
        return None

    # Find newly imported armature(s)
    new_armatures = [
        obj
        for obj in bpy.data.objects
        if obj not in existing_objects and obj.type == "ARMATURE"
    ]

    if not new_armatures:
        logger.error("No armature found in animation FBX: %s", fbx_path.name)
        return None

    if len(new_armatures) > 1:
        logger.warning(
            "Multiple armatures (%d) in %s — using the first",
            len(new_armatures),
            fbx_path.name,
        )

    return new_armatures[0]


def _cleanup_imported_armature(armature: bpy.types.Object) -> None:
    """Remove a temporarily imported animation armature and its data.

    Also removes any child objects (meshes that came along with the FBX)
    and purges orphaned data blocks.

    Args:
        armature: The armature object to remove.
    """
    # Collect objects to delete (armature + any children)
    objects_to_delete = [armature, *armature.children]

    for obj in objects_to_delete:
        bpy.data.objects.remove(obj, do_unlink=True)

    # Purge orphaned data blocks
    for collection in (bpy.data.armatures, bpy.data.actions, bpy.data.meshes):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _get_action_frame_range(armature: bpy.types.Object) -> tuple[int, int] | None:
    """Get the frame range of the animation action on an armature.

    Checks the armature's animation data for an active action, then falls
    back to checking all actions in the blend file.

    Args:
        armature: Armature with animation data.

    Returns:
        (start_frame, end_frame) as integers, or None if no action found.
    """
    action = None

    # Check direct animation data
    if armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action

    # Fallback: check NLA tracks
    if action is None and armature.animation_data:
        for track in armature.animation_data.nla_tracks:
            for strip in track.strips:
                if strip.action:
                    action = strip.action
                    break
            if action:
                break

    # Fallback: use first available action in blend data
    if action is None and bpy.data.actions:
        action = bpy.data.actions[0]

    if action is None:
        return None

    start = int(action.frame_range[0])
    end = int(action.frame_range[1])
    return start, end


# ---------------------------------------------------------------------------
# Pose application
# ---------------------------------------------------------------------------


def _apply_animation_pose(
    character_armature: bpy.types.Object,
    anim_armature: bpy.types.Object,
    target_frame: int,
) -> int:
    """Transfer an animation pose from one armature to another at a given frame.

    Sets the scene to the target frame, reads bone transforms from the
    animation armature, and copies them to matching bones on the character.

    Args:
        character_armature: The character's armature to pose.
        anim_armature: The animation source armature.
        target_frame: Frame number to sample.

    Returns:
        Number of bones successfully transferred.
    """
    scene = bpy.context.scene
    scene.frame_set(target_frame)

    # Build bone name mapping (anim → character)
    anim_bone_names = [b.name for b in anim_armature.pose.bones]
    char_bone_names = [b.name for b in character_armature.pose.bones]
    name_map = _build_name_map(anim_bone_names, char_bone_names)

    if not name_map:
        logger.warning(
            "No matching bones between animation (%s) and character (%s)",
            anim_armature.name,
            character_armature.name,
        )
        return 0

    # Transfer pose bone transforms
    transferred = 0
    for anim_bone_name, char_bone_name in name_map.items():
        anim_pbone = anim_armature.pose.bones.get(anim_bone_name)
        char_pbone = character_armature.pose.bones.get(char_bone_name)

        if anim_pbone is None or char_pbone is None:
            continue

        # Copy the local transform basis (rotation + location + scale)
        char_pbone.rotation_mode = anim_pbone.rotation_mode

        if anim_pbone.rotation_mode == "QUATERNION":
            char_pbone.rotation_quaternion = anim_pbone.rotation_quaternion.copy()
        elif anim_pbone.rotation_mode == "AXIS_ANGLE":
            char_pbone.rotation_axis_angle = tuple(anim_pbone.rotation_axis_angle)
        else:
            char_pbone.rotation_euler = anim_pbone.rotation_euler.copy()

        char_pbone.location = anim_pbone.location.copy()
        char_pbone.scale = anim_pbone.scale.copy()

        transferred += 1

    # Force scene update so positions are current for rendering
    bpy.context.view_layer.update()

    logger.debug(
        "Transferred %d/%d bones at frame %d",
        transferred,
        len(anim_bone_names),
        target_frame,
    )
    return transferred


def _apply_tpose(armature: bpy.types.Object) -> None:
    """Reset all pose bones to identity (T-pose).

    Args:
        armature: The armature to reset.
    """
    for pbone in armature.pose.bones:
        pbone.location = (0.0, 0.0, 0.0)
        pbone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pbone.rotation_euler = (0.0, 0.0, 0.0)
        pbone.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
        pbone.scale = (1.0, 1.0, 1.0)

    bpy.context.view_layer.update()


def _apply_apose(armature: bpy.types.Object) -> None:
    """Apply A-pose: T-pose with upper arms rotated downward.

    Args:
        armature: The armature to pose.
    """
    # Start from T-pose
    _apply_tpose(armature)

    angle_rad = math.radians(A_POSE_SHOULDER_ANGLE)

    for pbone in armature.pose.bones:
        name_lower = pbone.name.lower()

        is_left_arm = any(kw in name_lower for kw in _LEFT_UPPER_ARM_KEYWORDS)
        is_right_arm = any(kw in name_lower for kw in _RIGHT_UPPER_ARM_KEYWORDS)

        if is_left_arm:
            # Rotate left arm downward (positive Z rotation in bone-local space)
            pbone.rotation_mode = "XYZ"
            pbone.rotation_euler = Euler((0.0, 0.0, angle_rad), "XYZ")
        elif is_right_arm:
            # Rotate right arm downward (negative Z rotation in bone-local space)
            pbone.rotation_mode = "XYZ"
            pbone.rotation_euler = Euler((0.0, 0.0, -angle_rad), "XYZ")

    bpy.context.view_layer.update()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_poses(
    pose_dir: Path,
    keyframes_per_clip: int = KEYFRAMES_PER_CLIP,
    *,
    include_builtins: bool = True,
) -> list[PoseInfo]:
    """List all available poses from a pose library directory.

    Scans for ``.fbx`` files and computes keyframe indices for each clip.
    Optionally includes built-in T-pose and A-pose.

    Args:
        pose_dir: Directory containing animation FBX files.
        keyframes_per_clip: Number of keyframes to sample per animation.
        include_builtins: Whether to include T-pose and A-pose.

    Returns:
        List of PoseInfo objects representing all available poses.
    """
    poses: list[PoseInfo] = [TPOSE_INFO, APOSE_INFO] if include_builtins else []

    # Scan for animation FBX files
    if not pose_dir.is_dir():
        logger.warning("Pose directory does not exist: %s", pose_dir)
        return poses

    fbx_files = sorted(pose_dir.glob("*.fbx"))
    if not fbx_files:
        logger.warning("No .fbx files found in pose directory: %s", pose_dir)
        return poses

    for fbx_path in fbx_files:
        clip_name = fbx_path.stem.lower().replace(" ", "_")

        # Import temporarily to read frame range
        anim_armature = _import_animation_fbx(fbx_path)
        if anim_armature is None:
            continue

        frame_range = _get_action_frame_range(anim_armature)
        if frame_range is None:
            logger.warning("No animation data in %s — skipping", fbx_path.name)
            _cleanup_imported_armature(anim_armature)
            continue

        start, end = frame_range
        total_frames = end - start + 1
        sample_frames = _compute_sample_frames(total_frames, keyframes_per_clip)

        for frame_offset in sample_frames:
            absolute_frame = start + frame_offset
            poses.append(
                PoseInfo(
                    name=f"{clip_name}_frame_{absolute_frame:02d}",
                    source=fbx_path.name,
                    frame=absolute_frame,
                )
            )

        _cleanup_imported_armature(anim_armature)
        logger.info(
            "Indexed %s: %d frames, %d keyframes sampled",
            fbx_path.name,
            total_frames,
            len(sample_frames),
        )

    logger.info("Total poses available: %d", len(poses))
    return poses


def apply_pose(
    armature: bpy.types.Object,
    pose: PoseInfo,
    pose_dir: Path,
) -> bool:
    """Apply a pose to a character armature.

    For built-in poses (T-pose, A-pose), applies directly. For animation
    poses, temporarily imports the source FBX, transfers the pose at the
    specified frame, then cleans up.

    Args:
        armature: The character's armature object.
        pose: PoseInfo describing the target pose.
        pose_dir: Directory containing animation FBX files.

    Returns:
        True if the pose was applied successfully, False otherwise.
    """
    # Built-in poses
    if pose.source == "built-in":
        if pose.name == "t_pose":
            _apply_tpose(armature)
            logger.info("Applied T-pose")
            return True
        if pose.name == "a_pose":
            _apply_apose(armature)
            logger.info("Applied A-pose")
            return True
        logger.error("Unknown built-in pose: %s", pose.name)
        return False

    # Animation pose — load the source FBX
    fbx_path = pose_dir / pose.source
    if not fbx_path.is_file():
        logger.error("Animation FBX not found: %s", fbx_path)
        return False

    anim_armature = _import_animation_fbx(fbx_path)
    if anim_armature is None:
        return False

    transferred = _apply_animation_pose(armature, anim_armature, pose.frame)

    # Clean up the temporary animation armature
    _cleanup_imported_armature(anim_armature)

    if transferred == 0:
        logger.warning(
            "No bones transferred for pose %s from %s — bone names may not match",
            pose.name,
            pose.source,
        )
        return False

    logger.info("Applied pose %s (%d bones transferred)", pose.name, transferred)
    return True


def reset_pose(armature: bpy.types.Object) -> None:
    """Reset an armature to its rest pose (T-pose).

    Call this between pose applications to ensure a clean state.

    Args:
        armature: The armature to reset.
    """
    _apply_tpose(armature)
    logger.debug("Reset armature %s to rest pose", armature.name)

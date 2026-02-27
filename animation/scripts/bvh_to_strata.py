"""Retarget BVH motion capture data to Strata's 19-bone skeleton.

Maps CMU/SFU bone names to Strata region names, collapses multi-spine
hierarchies, and produces per-frame rotation data ready for proportion
normalization and blueprint export.

No Blender dependency — pure Python + numpy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from animation.scripts.bvh_parser import BVHFile, BVHSkeleton

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strata skeleton definition (canonical bone names, ordered by region ID)
# ---------------------------------------------------------------------------

STRATA_BONES: list[str] = [
    "hips",           # 5
    "spine",          # 4
    "chest",          # 3
    "neck",           # 2
    "head",           # 1
    "shoulder_l",     # 18
    "upper_arm_l",    # 6
    "lower_arm_l",    # 7
    "hand_l",         # 8
    "shoulder_r",     # 19
    "upper_arm_r",    # 9
    "lower_arm_r",    # 10
    "hand_r",         # 11
    "upper_leg_l",    # 12
    "lower_leg_l",    # 13
    "foot_l",         # 14
    "upper_leg_r",    # 15
    "lower_leg_r",    # 16
    "foot_r",         # 17
]

# ---------------------------------------------------------------------------
# CMU bone name → Strata bone name mapping
# ---------------------------------------------------------------------------
# Multi-spine rule: CMU Spine→ignored (subsumed by Spine1), Spine1→spine,
# Spine2→chest.  When only Spine exists (no Spine1/Spine2), it maps to spine.

CMU_TO_STRATA: dict[str, str] = {
    "Hips": "hips",
    "Spine": "spine",         # fallback if no Spine1
    "Spine1": "spine",
    "Spine2": "chest",
    "Neck": "neck",
    "Neck1": "neck",
    "Head": "head",
    "LeftShoulder": "shoulder_l",
    "LeftArm": "upper_arm_l",
    "LeftForeArm": "lower_arm_l",
    "LeftHand": "hand_l",
    "RightShoulder": "shoulder_r",
    "RightArm": "upper_arm_r",
    "RightForeArm": "lower_arm_r",
    "RightHand": "hand_r",
    "LeftUpLeg": "upper_leg_l",
    "LeftLeg": "lower_leg_l",
    "LeftFoot": "foot_l",
    "RightUpLeg": "upper_leg_r",
    "RightLeg": "lower_leg_r",
    "RightFoot": "foot_r",
}

# Bones that should never produce "unmapped" warnings (End Sites, fingers, toes)
_SILENTLY_IGNORED_SUFFIXES: tuple[str, ...] = (
    "_End",
    "Thumb", "Index", "Middle", "Ring", "Pinky",
    "ToeBase", "Toe_End", "Toe",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetargetedFrame:
    """A single frame of retargeted animation data.

    Attributes:
        rotations: Mapping of Strata bone name → (x, y, z) Euler rotation
            in degrees.  Rotation order is preserved from the source BVH.
        root_position: (x, y, z) world-space position of the root (hips).
    """

    rotations: dict[str, tuple[float, float, float]]
    root_position: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class RetargetedAnimation:
    """Complete retargeted animation in Strata skeleton space.

    Attributes:
        frames: Per-frame rotation + root position data.
        frame_count: Number of frames.
        frame_rate: Frames per second (derived from BVH frame_time).
        source_bones: Set of CMU bone names that were successfully mapped.
        unmapped_bones: CMU bones with no Strata equivalent (logged as warnings).
        rotation_order: Rotation axis order from the source BVH (e.g. "ZXY").
    """

    frames: list[RetargetedFrame] = field(default_factory=list)
    frame_count: int = 0
    frame_rate: float = 30.0
    source_bones: set[str] = field(default_factory=set)
    unmapped_bones: list[str] = field(default_factory=list)
    rotation_order: str = "ZXY"


# ---------------------------------------------------------------------------
# Multi-spine collapse logic
# ---------------------------------------------------------------------------

def _resolve_spine_mapping(skeleton: BVHSkeleton) -> dict[str, str]:
    """Determine the correct spine bone mapping for this skeleton.

    CMU skeletons may have Spine, Spine1, Spine2 — or just Spine.
    Rules:
        - If Spine1 + Spine2 exist: Spine1→spine, Spine2→chest, Spine→ignored
        - If only Spine1 exists (no Spine2): Spine1→spine
        - If only Spine exists (no Spine1/Spine2): Spine→spine

    Args:
        skeleton: Parsed BVH skeleton.

    Returns:
        Mapping of CMU spine bone names to Strata bone names.
    """
    joint_names = set(skeleton.joints.keys())
    mapping: dict[str, str] = {}

    has_spine = "Spine" in joint_names
    has_spine1 = "Spine1" in joint_names
    has_spine2 = "Spine2" in joint_names

    if has_spine1 and has_spine2:
        # Full CMU spine chain: ignore Spine, use Spine1→spine, Spine2→chest
        mapping["Spine1"] = "spine"
        mapping["Spine2"] = "chest"
        if has_spine:
            logger.debug(
                "Multi-spine collapse: Spine ignored, Spine1→spine, Spine2→chest"
            )
    elif has_spine1:
        mapping["Spine1"] = "spine"
    elif has_spine:
        mapping["Spine"] = "spine"

    return mapping


def _detect_rotation_order(skeleton: BVHSkeleton) -> str:
    """Detect the dominant rotation order from the skeleton's channels.

    Args:
        skeleton: Parsed BVH skeleton.

    Returns:
        Rotation order string (e.g. "ZXY", "YXZ").  Falls back to "ZXY".
    """
    for joint_name in skeleton.joint_order:
        joint = skeleton.joints[joint_name]
        rot_channels = [ch for ch in joint.channels if ch.endswith("rotation")]
        if len(rot_channels) >= 3:
            return "".join(ch[0] for ch in rot_channels[:3])
    return "ZXY"


# ---------------------------------------------------------------------------
# Core retargeting
# ---------------------------------------------------------------------------

def _build_bone_map(skeleton: BVHSkeleton) -> tuple[dict[str, str], list[str]]:
    """Build the CMU→Strata bone mapping for a specific skeleton.

    Args:
        skeleton: Parsed BVH skeleton.

    Returns:
        (mapped, unmapped) where mapped is {cmu_name: strata_name} and
        unmapped is a list of CMU bone names with no Strata equivalent.
    """
    spine_overrides = _resolve_spine_mapping(skeleton)
    mapped: dict[str, str] = {}
    unmapped: list[str] = []

    for joint_name in skeleton.joint_order:
        joint = skeleton.joints[joint_name]

        # Skip End Sites and finger/toe bones silently
        if any(joint_name.endswith(suffix) for suffix in _SILENTLY_IGNORED_SUFFIXES):
            continue
        if not joint.channels:
            continue

        # Check spine override first (handles multi-spine collapse)
        if joint_name in spine_overrides:
            mapped[joint_name] = spine_overrides[joint_name]
            continue

        # Skip Spine if it was collapsed (Spine1/Spine2 took over)
        if (
            joint_name == "Spine"
            and "Spine" not in spine_overrides
            and "Spine1" in skeleton.joints
        ):
            logger.debug("Spine ignored due to multi-spine collapse")
            continue

        # Standard lookup
        strata_name = CMU_TO_STRATA.get(joint_name)
        if strata_name and strata_name not in ("spine", "chest"):
            # Non-spine mapping from the standard table
            mapped[joint_name] = strata_name
        elif strata_name is None:
            unmapped.append(joint_name)

    return mapped, unmapped


def _extract_channels(
    channels: list[str],
    values: list[float],
    x_name: str,
    y_name: str,
    z_name: str,
) -> tuple[float, float, float]:
    """Extract an (x, y, z) triplet from named BVH channels.

    Args:
        channels: Channel names for this joint.
        values: Corresponding float values.
        x_name: Channel name for the X component.
        y_name: Channel name for the Y component.
        z_name: Channel name for the Z component.

    Returns:
        (x, y, z) values, defaulting to 0.0 for missing channels.
    """
    x = y = z = 0.0
    for ch, val in zip(channels, values, strict=False):
        if ch == x_name:
            x = val
        elif ch == y_name:
            y = val
        elif ch == z_name:
            z = val
    return (x, y, z)


def _extract_rotation(
    channels: list[str],
    values: list[float],
) -> tuple[float, float, float]:
    """Extract (rx, ry, rz) rotation in degrees from channel values."""
    return _extract_channels(channels, values, "Xrotation", "Yrotation", "Zrotation")


def _extract_position(
    channels: list[str],
    values: list[float],
) -> tuple[float, float, float]:
    """Extract (px, py, pz) position from channel values."""
    return _extract_channels(channels, values, "Xposition", "Yposition", "Zposition")


def retarget(bvh: BVHFile) -> RetargetedAnimation:
    """Retarget a parsed BVH file to Strata's 19-bone skeleton.

    Maps CMU bone names to Strata bone names, collapses multi-spine
    hierarchies, and extracts per-frame rotations and root position.
    Unmapped bones are logged as warnings and skipped.

    Args:
        bvh: Parsed BVH file from ``bvh_parser.parse_bvh()``.

    Returns:
        Retargeted animation with Strata bone names.
    """
    bone_map, unmapped = _build_bone_map(bvh.skeleton)

    for bone_name in unmapped:
        logger.warning("Unmapped CMU bone: %s — skipping", bone_name)

    rotation_order = _detect_rotation_order(bvh.skeleton)
    frame_rate = round(1.0 / bvh.motion.frame_time, 2) if bvh.motion.frame_time > 0 else 30.0

    # Find the root joint for position extraction
    root_joint_name = bvh.skeleton.root
    root_joint = bvh.skeleton.joints[root_joint_name]

    frames: list[RetargetedFrame] = []
    zero_rot: tuple[float, float, float] = (0.0, 0.0, 0.0)

    for frame_data in bvh.motion.frames:
        rotations: dict[str, tuple[float, float, float]] = {}

        # Initialize all Strata bones to zero rotation
        for bone in STRATA_BONES:
            rotations[bone] = zero_rot

        # Map source rotations to Strata bones
        for cmu_name, strata_name in bone_map.items():
            if cmu_name not in frame_data:
                continue
            joint = bvh.skeleton.joints[cmu_name]
            values = frame_data[cmu_name]
            rotations[strata_name] = _extract_rotation(joint.channels, values)

        # Extract root position
        root_position = zero_rot
        if root_joint_name in frame_data:
            root_position = _extract_position(
                root_joint.channels,
                frame_data[root_joint_name],
            )

        frames.append(RetargetedFrame(rotations=rotations, root_position=root_position))

    result = RetargetedAnimation(
        frames=frames,
        frame_count=len(frames),
        frame_rate=frame_rate,
        source_bones=set(bone_map.keys()),
        unmapped_bones=unmapped,
        rotation_order=rotation_order,
    )

    logger.info(
        "Retargeted %d frames: %d bones mapped, %d unmapped, rotation order %s",
        result.frame_count,
        len(bone_map),
        len(unmapped),
        rotation_order,
    )

    return result

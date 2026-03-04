"""Retarget BVH motion capture data to Strata's 19-bone skeleton.

Maps CMU/SFU bone names to Strata region names, collapses multi-spine
hierarchies, and produces per-frame rotation data ready for proportion
normalization and blueprint export.

Also provides Strata-compatibility checking: evaluates whether a BVH clip
uses only bones present in Strata's 19-bone skeleton or depends heavily on
unmapped bones (fingers, facial).

No Blender dependency — pure Python + numpy.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from animation.scripts.bvh_parser import BVHFile, BVHSkeleton, parse_bvh

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strata skeleton definition (canonical bone names, ordered by region ID)
# ---------------------------------------------------------------------------

STRATA_BONES: list[str] = [
    "hips",  # 5
    "spine",  # 4
    "chest",  # 3
    "neck",  # 2
    "head",  # 1
    "shoulder_l",  # 18
    "upper_arm_l",  # 6
    "forearm_l",  # 7
    "hand_l",  # 8
    "shoulder_r",  # 19
    "upper_arm_r",  # 9
    "forearm_r",  # 10
    "hand_r",  # 11
    "upper_leg_l",  # 12
    "lower_leg_l",  # 13
    "foot_l",  # 14
    "upper_leg_r",  # 15
    "lower_leg_r",  # 16
    "foot_r",  # 17
]

# ---------------------------------------------------------------------------
# CMU bone name → Strata bone name mapping
# ---------------------------------------------------------------------------
# Multi-spine rule: CMU Spine→ignored (subsumed by Spine1), Spine1→spine,
# Spine2→chest.  When only Spine exists (no Spine1/Spine2), it maps to spine.

CMU_TO_STRATA: dict[str, str] = {
    "Hips": "hips",
    "Spine": "spine",  # fallback if no Spine1
    "Spine1": "spine",
    "Spine2": "chest",
    "Neck": "neck",
    "Neck1": "neck",
    "Head": "head",
    "LeftShoulder": "shoulder_l",
    "LeftArm": "upper_arm_l",
    "LeftForeArm": "forearm_l",
    "LeftHand": "hand_l",
    "RightShoulder": "shoulder_r",
    "RightArm": "upper_arm_r",
    "RightForeArm": "forearm_r",
    "RightHand": "hand_r",
    "LeftUpLeg": "upper_leg_l",
    "LeftLeg": "lower_leg_l",
    "LeftFoot": "foot_l",
    "RightUpLeg": "upper_leg_r",
    "RightLeg": "lower_leg_r",
    "RightFoot": "foot_r",
}

# ---------------------------------------------------------------------------
# 100STYLE bone name → Strata bone name mapping
# ---------------------------------------------------------------------------
# 100STYLE uses: Hips → Chest → Chest2 → Chest3 → Chest4 → Neck/Head + arms/legs.
# 4-spine collapse: Chest→spine, Chest2+Chest3→ignored (mid-chain), Chest4→chest.

STYLE100_TO_STRATA: dict[str, str] = {
    "Hips": "hips",
    "Chest": "spine",
    # Chest2, Chest3 — mid-chain, collapsed (handled by _resolve_spine_mapping)
    "Chest4": "chest",
    "Neck": "neck",
    "Head": "head",
    "LeftCollar": "shoulder_l",
    "LeftShoulder": "upper_arm_l",
    "LeftElbow": "forearm_l",
    "LeftWrist": "hand_l",
    "RightCollar": "shoulder_r",
    "RightShoulder": "upper_arm_r",
    "RightElbow": "forearm_r",
    "RightWrist": "hand_r",
    "LeftHip": "upper_leg_l",
    "LeftKnee": "lower_leg_l",
    "LeftAnkle": "foot_l",
    "RightHip": "upper_leg_r",
    "RightKnee": "lower_leg_r",
    "RightAnkle": "foot_r",
}

# Bones that should never produce "unmapped" warnings (End Sites, fingers, toes,
# hip connectors).  Checked with endswith() so "LThumb" matches "Thumb".
_SILENTLY_IGNORED_SUFFIXES: tuple[str, ...] = (
    "_End",
    "Thumb",
    "Index",
    "Index1",
    "Middle",
    "Ring",
    "Pinky",
    "ToeBase",
    "Toe_End",
    "Toe",
    "FingerBase",
    "HipJoint",
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


def _is_100style_skeleton(skeleton: BVHSkeleton) -> bool:
    """Detect whether this BVH uses the 100STYLE skeleton layout.

    100STYLE skeletons have Chest/Chest2/Chest3/Chest4 (no Spine/Spine1/Spine2)
    and use LeftHip/RightHip instead of LeftUpLeg/RightUpLeg.
    """
    joint_names = set(skeleton.joints.keys())
    return "Chest4" in joint_names and "LeftHip" in joint_names


def _resolve_spine_mapping(skeleton: BVHSkeleton) -> dict[str, str]:
    """Determine the correct spine bone mapping for this skeleton.

    Handles three skeleton families:

    100STYLE layout:
        Hips → Chest → Chest2 → Chest3 → Chest4 → Neck
        Maps: Chest→spine, Chest4→chest, Chest2+Chest3→ignored (mid-chain)

    CMU Layout A (cgspeed BVH conversion):
        Hips → LowerBack → Spine → Spine1 → Neck
        Maps: LowerBack→spine, Spine1→chest, Spine→ignored

    CMU Layout B (generic/Mixamo-like):
        Hips → Spine → Spine1 → Spine2 → Neck
        Maps: Spine1→spine, Spine2→chest, Spine→ignored

    Fallbacks:
        - Only Spine1 (no Spine2, no LowerBack): Spine1→spine
        - Only Spine (no Spine1): Spine→spine

    Args:
        skeleton: Parsed BVH skeleton.

    Returns:
        Mapping of spine bone names to Strata bone names.
    """
    joint_names = set(skeleton.joints.keys())
    mapping: dict[str, str] = {}

    # 100STYLE: 4-segment spine (Chest → Chest2 → Chest3 → Chest4)
    if "Chest4" in joint_names and "Chest" in joint_names:
        mapping["Chest"] = "spine"
        mapping["Chest4"] = "chest"
        # Chest2, Chest3 are mid-chain — will be silently ignored
        logger.debug("100STYLE spine: Chest→spine, Chest4→chest, Chest2+Chest3 ignored")
        return mapping

    has_lower_back = "LowerBack" in joint_names
    has_spine = "Spine" in joint_names
    has_spine1 = "Spine1" in joint_names
    has_spine2 = "Spine2" in joint_names

    if has_lower_back and has_spine1:
        # Layout A: LowerBack → Spine → Spine1 → Neck
        mapping["LowerBack"] = "spine"
        mapping["Spine1"] = "chest"
        # Spine is mid-chain, ignored (subsumed by LowerBack + Spine1)
        logger.debug("CMU spine layout A: LowerBack→spine, Spine1→chest, Spine ignored")
    elif has_lower_back and has_spine:
        # LowerBack + Spine but no Spine1
        mapping["LowerBack"] = "spine"
        mapping["Spine"] = "chest"
        logger.debug("CMU spine: LowerBack→spine, Spine→chest")
    elif has_spine1 and has_spine2:
        # Layout B: Spine → Spine1 → Spine2 → Neck
        mapping["Spine1"] = "spine"
        mapping["Spine2"] = "chest"
        logger.debug("Multi-spine collapse: Spine ignored, Spine1→spine, Spine2→chest")
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
    """Build the source→Strata bone mapping for a specific skeleton.

    Auto-detects 100STYLE vs CMU skeleton layouts and uses the appropriate
    mapping table.

    Args:
        skeleton: Parsed BVH skeleton.

    Returns:
        (mapped, unmapped) where mapped is {source_name: strata_name} and
        unmapped is a list of source bone names with no Strata equivalent.
    """
    is_100style = _is_100style_skeleton(skeleton)
    spine_overrides = _resolve_spine_mapping(skeleton)
    base_table = STYLE100_TO_STRATA if is_100style else CMU_TO_STRATA
    mapped: dict[str, str] = {}
    unmapped: list[str] = []

    # Mid-chain spine bones to silently skip (100STYLE: Chest2, Chest3)
    mid_chain_spine: set[str] = set()
    if is_100style:
        mid_chain_spine = {"Chest2", "Chest3"}

    for joint_name in skeleton.joint_order:
        joint = skeleton.joints[joint_name]

        # Skip End Sites and finger/toe bones silently
        if any(joint_name.endswith(suffix) for suffix in _SILENTLY_IGNORED_SUFFIXES):
            continue
        if not joint.channels:
            continue

        # Skip mid-chain spine bones (100STYLE Chest2/Chest3)
        if joint_name in mid_chain_spine:
            logger.debug("Mid-chain spine bone %s ignored", joint_name)
            continue

        # Check spine override first (handles multi-spine collapse)
        if joint_name in spine_overrides:
            mapped[joint_name] = spine_overrides[joint_name]
            continue

        # Skip Spine if it was collapsed (LowerBack or Spine1/Spine2 took over)
        if (
            joint_name == "Spine"
            and "Spine" not in spine_overrides
            and ("Spine1" in skeleton.joints or "LowerBack" in skeleton.joints)
        ):
            logger.debug("Spine ignored due to multi-spine collapse")
            continue

        # Standard lookup from the appropriate table
        strata_name = base_table.get(joint_name)
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


# ---------------------------------------------------------------------------
# Strata compatibility checking
# ---------------------------------------------------------------------------

# Minimum rotation delta (degrees) for a bone to count as "active"
ROTATION_DELTA_THRESHOLD: float = 1.0


@dataclass
class CompatibilityResult:
    """Result of evaluating a BVH clip for Strata skeleton compatibility.

    Attributes:
        compatible: True if the clip maps cleanly to Strata's 19-bone skeleton.
        mapped_count: Number of source bones that mapped to Strata bones.
        unmapped_count: Number of source bones with no Strata equivalent.
        active_unmapped: Unmapped bones that have significant rotation data.
        reason: Human-readable explanation of the verdict.
    """

    compatible: bool
    mapped_count: int
    unmapped_count: int
    active_unmapped: list[str] = field(default_factory=list)
    reason: str = ""


def check_strata_compatibility(
    bvh: BVHFile,
    threshold: float = ROTATION_DELTA_THRESHOLD,
) -> CompatibilityResult:
    """Evaluate whether a BVH clip maps cleanly to Strata's 19-bone skeleton.

    A clip is compatible if all bones with significant rotation data map to
    Strata bones.  Bones with rotation deltas below *threshold* degrees
    (across all frames) are considered inactive and ignored.  This includes
    bones silently ignored by the retargeter (fingers, toes) — if they
    carry significant motion, the clip is flagged as incompatible.

    Args:
        bvh: Parsed BVH file from ``bvh_parser.parse_bvh()``.
        threshold: Minimum rotation delta (degrees) for a bone to be
            considered "active".  Defaults to ``ROTATION_DELTA_THRESHOLD``.

    Returns:
        CompatibilityResult with verdict and details.
    """
    bone_map, unmapped = _build_bone_map(bvh.skeleton)
    mapped_names = set(bone_map.keys())

    # Collect non-Strata bones, excluding silently-ignored ones (fingers, toes,
    # hip connectors) and mid-chain spine bones that were collapsed.
    non_strata: list[str] = list(unmapped)
    seen = set(non_strata)
    for joint_name in bvh.skeleton.joint_order:
        if joint_name in mapped_names or joint_name in seen:
            continue
        if not bvh.skeleton.joints[joint_name].channels:
            continue
        if any(joint_name.endswith(suffix) for suffix in _SILENTLY_IGNORED_SUFFIXES):
            continue
        # Skip mid-chain spine bones (CMU Spine, 100STYLE Chest2/Chest3)
        if joint_name == "Spine" and (
            "Spine1" in bvh.skeleton.joints or "LowerBack" in bvh.skeleton.joints
        ):
            continue
        if joint_name in ("Chest2", "Chest3") and _is_100style_skeleton(bvh.skeleton):
            continue
        non_strata.append(joint_name)
        seen.add(joint_name)

    # No motion data → compatible by default (skeleton-only file)
    if not bvh.motion.frames:
        return CompatibilityResult(
            compatible=True,
            mapped_count=len(bone_map),
            unmapped_count=len(non_strata),
            reason="No motion data — skeleton-only file",
        )

    # Find non-Strata bones that have significant rotation across frames
    active_unmapped: list[str] = []
    for bone_name in non_strata:
        if _bone_has_significant_rotation(bvh, bone_name, threshold):
            active_unmapped.append(bone_name)

    compatible = len(active_unmapped) == 0

    if compatible:
        if non_strata:
            reason = (
                f"Compatible: {len(bone_map)} bones mapped, {len(non_strata)} extra bones inactive"
            )
        else:
            reason = f"Compatible: all {len(bone_map)} bones mapped"
    else:
        reason = (
            f"Incompatible: {len(active_unmapped)} non-Strata bones have "
            f"significant motion ({', '.join(active_unmapped)})"
        )

    result = CompatibilityResult(
        compatible=compatible,
        mapped_count=len(bone_map),
        unmapped_count=len(non_strata),
        active_unmapped=active_unmapped,
        reason=reason,
    )

    logger.info("Compatibility check: %s", result.reason)
    return result


def _bone_has_significant_rotation(
    bvh: BVHFile,
    bone_name: str,
    threshold: float,
) -> bool:
    """Check if a bone has rotation deltas above the threshold across frames.

    Compares each frame's rotation to the first frame. If any axis delta
    exceeds the threshold on any frame, the bone is considered active.

    Args:
        bvh: Parsed BVH file.
        bone_name: Name of the bone to check.
        threshold: Minimum delta in degrees.

    Returns:
        True if the bone has significant rotation data.
    """
    joint = bvh.skeleton.joints.get(bone_name)
    if joint is None or not joint.channels:
        return False

    frames = bvh.motion.frames
    if len(frames) < 2:
        return False

    # Get first frame rotation as baseline
    if bone_name not in frames[0]:
        return False

    baseline = _extract_rotation(joint.channels, frames[0][bone_name])

    for frame_data in frames[1:]:
        if bone_name not in frame_data:
            continue
        rot = _extract_rotation(joint.channels, frame_data[bone_name])
        for b, r in zip(baseline, rot, strict=True):
            if abs(r - b) > threshold:
                return True

    return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli_check_compat(args: argparse.Namespace) -> None:
    """CLI handler for --check-compat."""
    for path_str in args.files:
        path = Path(path_str)
        try:
            bvh = parse_bvh(path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"{path.name}: ERROR — {exc}", file=sys.stderr)
            continue

        result = check_strata_compatibility(bvh, threshold=args.threshold)
        compat_str = "yes" if result.compatible else "no"
        print(f"{path.name}: {compat_str} — {result.reason}")


def main() -> None:
    """CLI entry point for BVH-to-Strata retargeting utilities."""
    parser = argparse.ArgumentParser(
        description="BVH-to-Strata retargeting utilities",
    )
    sub = parser.add_subparsers(dest="command")

    compat_parser = sub.add_parser(
        "check-compat",
        help="Evaluate BVH files for Strata skeleton compatibility",
    )
    compat_parser.add_argument(
        "files",
        nargs="+",
        help="BVH file paths to evaluate",
    )
    compat_parser.add_argument(
        "--threshold",
        type=float,
        default=ROTATION_DELTA_THRESHOLD,
        help=f"Rotation delta threshold in degrees (default: {ROTATION_DELTA_THRESHOLD})",
    )

    args = parser.parse_args()
    if args.command == "check-compat":
        _cli_check_compat(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

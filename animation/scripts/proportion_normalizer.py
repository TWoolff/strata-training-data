"""Normalize retargeted animation proportions to match a target character.

Extracts bone lengths from BVH skeleton offsets, computes scaling ratios
against target character bone lengths, and adjusts root translations
proportionally.  Rotations are preserved unchanged.

No Blender dependency — pure Python.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from animation.scripts.bvh_parser import BVHSkeleton
from animation.scripts.bvh_to_strata import (
    CMU_TO_STRATA,
    RetargetedAnimation,
    RetargetedFrame,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BoneLengths:
    """Bone lengths extracted from a skeleton, keyed by Strata bone name.

    Attributes:
        lengths: Mapping of Strata bone name → length in source units.
        total_height: Approximate total skeleton height (sum of vertical chain).
    """

    lengths: dict[str, float] = field(default_factory=dict)
    total_height: float = 0.0


# ---------------------------------------------------------------------------
# Bone length extraction
# ---------------------------------------------------------------------------

def _offset_length(offset: tuple[float, float, float]) -> float:
    """Compute the Euclidean length of a bone offset vector."""
    return math.hypot(*offset)


def extract_bone_lengths(skeleton: BVHSkeleton) -> BoneLengths:
    """Extract bone lengths from BVH skeleton offsets.

    Each bone's length is the Euclidean distance of its offset vector from
    its parent joint.  The total height is approximated as the sum of the
    vertical chain: hips → spine → chest → neck → head.

    Args:
        skeleton: Parsed BVH skeleton.

    Returns:
        Bone lengths keyed by Strata bone name.
    """
    lengths: dict[str, float] = {}

    for joint_name, joint in skeleton.joints.items():
        strata_name = CMU_TO_STRATA.get(joint_name)
        if strata_name is None:
            continue

        length = _offset_length(joint.offset)
        # If multiple CMU bones map to the same Strata bone (e.g. Spine→spine
        # and Spine1→spine), keep the longer one as it's more meaningful.
        if strata_name not in lengths or length > lengths[strata_name]:
            lengths[strata_name] = length

    # Approximate total height from the vertical chain
    height_chain = ["hips", "spine", "chest", "neck", "head"]
    total_height = sum(lengths.get(bone, 0.0) for bone in height_chain)

    result = BoneLengths(lengths=lengths, total_height=total_height)
    logger.info(
        "Extracted bone lengths: %d bones, total height %.2f",
        len(lengths),
        total_height,
    )

    return result


# ---------------------------------------------------------------------------
# Proportion normalization
# ---------------------------------------------------------------------------

def normalize_proportions(
    animation: RetargetedAnimation,
    source_lengths: BoneLengths,
    target_lengths: BoneLengths,
) -> RetargetedAnimation:
    """Scale root translations to match target character proportions.

    Rotations transfer directly regardless of bone length — only root
    (hips) position is scaled by the height ratio between source and
    target skeletons.

    Args:
        animation: Retargeted animation to normalize.
        source_lengths: Bone lengths from the source BVH skeleton.
        target_lengths: Bone lengths from the target character.

    Returns:
        New RetargetedAnimation with scaled root positions.
    """
    if source_lengths.total_height <= 0.0:
        logger.warning("Source skeleton has zero height — skipping normalization")
        return animation

    if target_lengths.total_height <= 0.0:
        logger.warning("Target skeleton has zero height — skipping normalization")
        return animation

    scale = target_lengths.total_height / source_lengths.total_height
    logger.info(
        "Proportion scale: %.4f (source height %.2f → target height %.2f)",
        scale,
        source_lengths.total_height,
        target_lengths.total_height,
    )

    normalized_frames: list[RetargetedFrame] = []
    for frame in animation.frames:
        px, py, pz = frame.root_position
        normalized_frames.append(
            RetargetedFrame(
                rotations=frame.rotations,
                root_position=(px * scale, py * scale, pz * scale),
            )
        )

    return RetargetedAnimation(
        frames=normalized_frames,
        frame_count=animation.frame_count,
        frame_rate=animation.frame_rate,
        source_bones=animation.source_bones,
        unmapped_bones=animation.unmapped_bones,
        rotation_order=animation.rotation_order,
    )

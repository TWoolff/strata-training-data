"""Synthetic degradation pipeline for animation training pairs.

Takes high-quality animations (retargeted BVH or Strata blueprint JSON) and
systematically strips animation principles to create (degraded input, full
output) training pairs for the in-betweening model.

7 degradation types:
    1. Strip to extremes — keep only frames with max/min joint angles
    2. Linearize arcs — replace curved paths with straight-line interpolation
    3. Remove easing — replace non-uniform timing with uniform spacing
    4. Remove secondary — lock secondary bones to parent rotation
    5. Reduce framerate — keep every Nth frame
    6. Simultaneous stop — make all bones stop on the same frame
    7. Remove anticipation — delete counter-movement frames

Each good animation generates up to 7 degraded versions = 7 training pairs.

No Blender dependency — pure Python + numpy.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from animation.scripts.bvh_parser import parse_bvh
from animation.scripts.bvh_to_strata import (
    STRATA_BONES,
    RetargetedAnimation,
    RetargetedFrame,
    retarget,
)

logger = logging.getLogger(__name__)

# Minimum frames required for degradation to be meaningful
MIN_FRAMES: int = 4

# ---------------------------------------------------------------------------
# Strata skeleton parent-child hierarchy (for secondary motion removal)
# ---------------------------------------------------------------------------

STRATA_PARENT: dict[str, str] = {
    "spine": "hips",
    "chest": "spine",
    "neck": "chest",
    "head": "neck",
    "shoulder_l": "chest",
    "upper_arm_l": "shoulder_l",
    "lower_arm_l": "upper_arm_l",
    "hand_l": "lower_arm_l",
    "shoulder_r": "chest",
    "upper_arm_r": "shoulder_r",
    "lower_arm_r": "upper_arm_r",
    "hand_r": "lower_arm_r",
    "upper_leg_l": "hips",
    "lower_leg_l": "upper_leg_l",
    "foot_l": "lower_leg_l",
    "upper_leg_r": "hips",
    "lower_leg_r": "upper_leg_r",
    "foot_r": "lower_leg_r",
}

# Bones considered "secondary" — distal limb ends that exhibit follow-through
SECONDARY_BONES: frozenset[str] = frozenset({
    "hand_l", "hand_r",
    "foot_l", "foot_r",
    "head",
})

# Velocity threshold (degrees/frame) below which a bone is considered stopped
STOP_VELOCITY_THRESHOLD: float = 0.5

# Velocity threshold for anticipation detection (degrees/frame)
ANTICIPATION_VELOCITY_THRESHOLD: float = 2.0

# ---------------------------------------------------------------------------
# Degradation parameter dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StripToExtremesParams:
    """Parameters for strip-to-extremes degradation.

    Attributes:
        min_keyframes: Minimum number of keyframes to keep.
    """

    min_keyframes: int = 3


@dataclass
class LinearizeArcsParams:
    """Parameters for linearize-arcs degradation.

    Attributes:
        keyframe_interval: Sample one keyframe every N frames for linear interp.
    """

    keyframe_interval: int = 8


@dataclass
class RemoveEasingParams:
    """Parameters for remove-easing degradation (no extra params needed)."""


@dataclass
class RemoveSecondaryParams:
    """Parameters for remove-secondary-motion degradation.

    Attributes:
        secondary_bones: Bones to lock to their parent's rotation.
    """

    secondary_bones: frozenset[str] = SECONDARY_BONES


@dataclass
class ReduceFramerateParams:
    """Parameters for framerate reduction degradation.

    Attributes:
        factor: Keep every Nth frame (2=half, 3=third, 4=quarter).
    """

    factor: int = 2


@dataclass
class SimultaneousStopParams:
    """Parameters for simultaneous-stop degradation.

    Attributes:
        velocity_threshold: Degrees/frame below which a bone is "stopped".
    """

    velocity_threshold: float = STOP_VELOCITY_THRESHOLD


@dataclass
class RemoveAnticipationParams:
    """Parameters for remove-anticipation degradation.

    Attributes:
        velocity_threshold: Minimum velocity (degrees/frame) for a bone to be
            considered "moving" (used to detect direction reversals).
        max_anticipation_frames: Maximum number of consecutive frames to
            classify as anticipation.
    """

    velocity_threshold: float = ANTICIPATION_VELOCITY_THRESHOLD
    max_anticipation_frames: int = 10


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _rot_to_array(frames: list[RetargetedFrame]) -> dict[str, np.ndarray]:
    """Convert per-frame rotations to numpy arrays keyed by bone name.

    Returns:
        {bone_name: ndarray of shape (num_frames, 3)} for each bone.
    """
    result: dict[str, np.ndarray] = {}
    for bone in STRATA_BONES:
        result[bone] = np.array(
            [f.rotations.get(bone, (0.0, 0.0, 0.0)) for f in frames],
            dtype=np.float64,
        )
    return result


def _root_positions_array(frames: list[RetargetedFrame]) -> np.ndarray:
    """Extract root positions as (num_frames, 3) array."""
    return np.array([f.root_position for f in frames], dtype=np.float64)


def _frames_from_arrays(
    bone_arrays: dict[str, np.ndarray],
    root_positions: np.ndarray,
) -> list[RetargetedFrame]:
    """Reconstruct RetargetedFrame list from numpy arrays."""
    num_frames = root_positions.shape[0]
    frames: list[RetargetedFrame] = []
    for i in range(num_frames):
        rotations: dict[str, tuple[float, float, float]] = {}
        for bone in STRATA_BONES:
            arr = bone_arrays[bone]
            rotations[bone] = (float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2]))
        frames.append(RetargetedFrame(
            rotations=rotations,
            root_position=(
                float(root_positions[i, 0]),
                float(root_positions[i, 1]),
                float(root_positions[i, 2]),
            ),
        ))
    return frames


def _build_result(
    original: RetargetedAnimation,
    new_frames: list[RetargetedFrame],
) -> RetargetedAnimation:
    """Build a new RetargetedAnimation preserving metadata from the original."""
    return RetargetedAnimation(
        frames=new_frames,
        frame_count=len(new_frames),
        frame_rate=original.frame_rate,
        source_bones=original.source_bones,
        unmapped_bones=original.unmapped_bones,
        rotation_order=original.rotation_order,
    )


def _lerp_frames(
    frame_a: RetargetedFrame,
    frame_b: RetargetedFrame,
    t: float,
) -> RetargetedFrame:
    """Linearly interpolate between two frames.

    Args:
        frame_a: Start frame.
        frame_b: End frame.
        t: Interpolation factor in [0, 1].
    """
    rotations: dict[str, tuple[float, float, float]] = {}
    for bone in STRATA_BONES:
        ra = frame_a.rotations.get(bone, (0.0, 0.0, 0.0))
        rb = frame_b.rotations.get(bone, (0.0, 0.0, 0.0))
        rotations[bone] = (
            ra[0] + (rb[0] - ra[0]) * t,
            ra[1] + (rb[1] - ra[1]) * t,
            ra[2] + (rb[2] - ra[2]) * t,
        )
    pa = frame_a.root_position
    pb = frame_b.root_position
    root_pos = (
        pa[0] + (pb[0] - pa[0]) * t,
        pa[1] + (pb[1] - pa[1]) * t,
        pa[2] + (pb[2] - pa[2]) * t,
    )
    return RetargetedFrame(rotations=rotations, root_position=root_pos)


# ---------------------------------------------------------------------------
# 1. Strip to extremes
# ---------------------------------------------------------------------------


def strip_to_extremes(
    animation: RetargetedAnimation,
    params: StripToExtremesParams | None = None,
) -> RetargetedAnimation:
    """Keep only frames at rotation extremes (max/min joint angles).

    Finds frames where any bone reaches a local maximum or minimum rotation
    on any axis.  Always includes the first and last frames.

    Args:
        animation: Source animation.
        params: Configuration parameters.

    Returns:
        Degraded animation containing only extreme-pose keyframes.
    """
    if animation.frame_count < MIN_FRAMES:
        logger.warning("Animation too short (%d frames) for strip_to_extremes", animation.frame_count)
        return animation

    params = params or StripToExtremesParams()
    bone_arrays = _rot_to_array(animation.frames)

    # Find frames that are local extremes for any bone on any axis
    extreme_indices: set[int] = {0, animation.frame_count - 1}

    for bone in STRATA_BONES:
        arr = bone_arrays[bone]  # (num_frames, 3)
        for axis in range(3):
            values = arr[:, axis]
            total_range = values.max() - values.min()
            if total_range <= 1.0:
                continue
            for i in range(1, len(values) - 1):
                is_max = values[i] >= values[i - 1] and values[i] >= values[i + 1]
                is_min = values[i] <= values[i - 1] and values[i] <= values[i + 1]
                if is_max or is_min:
                    extreme_indices.add(i)

    sorted_indices = sorted(extreme_indices)

    # Ensure minimum keyframe count by adding evenly spaced frames
    if len(sorted_indices) < params.min_keyframes:
        step = max(1, animation.frame_count // params.min_keyframes)
        for i in range(0, animation.frame_count, step):
            sorted_indices.append(i)
        sorted_indices = sorted(set(sorted_indices))

    new_frames = [animation.frames[i] for i in sorted_indices]

    logger.info(
        "strip_to_extremes: %d → %d frames",
        animation.frame_count,
        len(new_frames),
    )
    return _build_result(animation, new_frames)


# ---------------------------------------------------------------------------
# 2. Linearize arcs
# ---------------------------------------------------------------------------


def linearize_arcs(
    animation: RetargetedAnimation,
    params: LinearizeArcsParams | None = None,
) -> RetargetedAnimation:
    """Replace curved motion paths with straight-line interpolation.

    Samples keyframes at regular intervals, then linearly interpolates
    between them, destroying the curved arc motion of the original.

    Args:
        animation: Source animation.
        params: Configuration parameters.

    Returns:
        Degraded animation with linearized motion paths.
    """
    if animation.frame_count < MIN_FRAMES:
        logger.warning("Animation too short (%d frames) for linearize_arcs", animation.frame_count)
        return animation

    params = params or LinearizeArcsParams()
    interval = max(2, params.keyframe_interval)

    # Sample keyframe indices
    key_indices = list(range(0, animation.frame_count, interval))
    if key_indices[-1] != animation.frame_count - 1:
        key_indices.append(animation.frame_count - 1)

    # Rebuild all frames by linearly interpolating between keyframes
    new_frames: list[RetargetedFrame] = []
    for seg_idx in range(len(key_indices) - 1):
        start_i = key_indices[seg_idx]
        end_i = key_indices[seg_idx + 1]
        frame_a = animation.frames[start_i]
        frame_b = animation.frames[end_i]
        segment_len = end_i - start_i

        for j in range(segment_len):
            t = j / segment_len
            new_frames.append(_lerp_frames(frame_a, frame_b, t))

    # Add final frame
    new_frames.append(animation.frames[-1])

    logger.info(
        "linearize_arcs: %d frames linearized with interval %d (%d keyframes)",
        animation.frame_count,
        interval,
        len(key_indices),
    )
    return _build_result(animation, new_frames)


# ---------------------------------------------------------------------------
# 3. Remove easing
# ---------------------------------------------------------------------------


def remove_easing(
    animation: RetargetedAnimation,
    params: RemoveEasingParams | None = None,
) -> RetargetedAnimation:
    """Replace ease-in/out timing with uniform frame spacing.

    Re-samples the animation at uniformly spaced time points, destroying
    the acceleration/deceleration curves.  The original frame count is
    preserved but motion is redistributed to have constant velocity between
    detected keyframes.

    Args:
        animation: Source animation.
        params: Configuration parameters (currently unused).

    Returns:
        Degraded animation with uniform timing.
    """
    if animation.frame_count < MIN_FRAMES:
        logger.warning("Animation too short (%d frames) for remove_easing", animation.frame_count)
        return animation

    bone_arrays = _rot_to_array(animation.frames)
    root_positions = _root_positions_array(animation.frames)

    # Compute cumulative arc-length (sum of rotation deltas across all bones)
    # for a motion-based parameterization
    num_frames = animation.frame_count
    arc_lengths = np.zeros(num_frames, dtype=np.float64)

    for bone in STRATA_BONES:
        arr = bone_arrays[bone]
        deltas = np.diff(arr, axis=0)
        per_frame_delta = np.sqrt(np.sum(deltas ** 2, axis=1))
        arc_lengths[1:] += per_frame_delta

    cumulative = np.cumsum(arc_lengths)
    total_length = cumulative[-1]

    if total_length < 1e-6:
        # No motion — nothing to retime
        return animation

    # Uniform target parameterization
    uniform_params = np.linspace(0.0, total_length, num_frames)

    # Resample each bone and root position at uniform parameter values
    new_bone_arrays: dict[str, np.ndarray] = {}
    for bone in STRATA_BONES:
        arr = bone_arrays[bone]
        resampled = np.zeros_like(arr)
        for axis in range(3):
            resampled[:, axis] = np.interp(uniform_params, cumulative, arr[:, axis])
        new_bone_arrays[bone] = resampled

    new_root = np.zeros_like(root_positions)
    for axis in range(3):
        new_root[:, axis] = np.interp(uniform_params, cumulative, root_positions[:, axis])

    new_frames = _frames_from_arrays(new_bone_arrays, new_root)

    logger.info("remove_easing: %d frames retimed to uniform spacing", num_frames)
    return _build_result(animation, new_frames)


# ---------------------------------------------------------------------------
# 4. Remove secondary motion
# ---------------------------------------------------------------------------


def remove_secondary(
    animation: RetargetedAnimation,
    params: RemoveSecondaryParams | None = None,
) -> RetargetedAnimation:
    """Lock secondary bones to their parent's rotation.

    For each designated secondary bone, replaces its rotation with the
    parent bone's rotation, destroying follow-through and overlapping
    action on that bone.

    Args:
        animation: Source animation.
        params: Configuration parameters.

    Returns:
        Degraded animation with secondary motion removed.
    """
    if animation.frame_count < MIN_FRAMES:
        logger.warning("Animation too short (%d frames) for remove_secondary", animation.frame_count)
        return animation

    params = params or RemoveSecondaryParams()

    new_frames: list[RetargetedFrame] = []
    for frame in animation.frames:
        rotations = dict(frame.rotations)
        for bone in params.secondary_bones:
            parent = STRATA_PARENT.get(bone)
            if parent and parent in rotations:
                rotations[bone] = rotations[parent]
        new_frames.append(RetargetedFrame(
            rotations=rotations,
            root_position=frame.root_position,
        ))

    logger.info(
        "remove_secondary: locked %d bones to parent rotation",
        len(params.secondary_bones),
    )
    return _build_result(animation, new_frames)


# ---------------------------------------------------------------------------
# 5. Reduce framerate
# ---------------------------------------------------------------------------


def reduce_framerate(
    animation: RetargetedAnimation,
    params: ReduceFramerateParams | None = None,
) -> RetargetedAnimation:
    """Keep every Nth frame, discarding intermediate frames.

    Args:
        animation: Source animation.
        params: Configuration parameters.

    Returns:
        Degraded animation with reduced frame count.
    """
    if animation.frame_count < MIN_FRAMES:
        logger.warning("Animation too short (%d frames) for reduce_framerate", animation.frame_count)
        return animation

    params = params or ReduceFramerateParams()
    factor = max(2, params.factor)

    indices = list(range(0, animation.frame_count, factor))
    # Always include the last frame
    if indices[-1] != animation.frame_count - 1:
        indices.append(animation.frame_count - 1)

    new_frames = [animation.frames[i] for i in indices]

    logger.info(
        "reduce_framerate: %d → %d frames (factor %d)",
        animation.frame_count,
        len(new_frames),
        factor,
    )
    return _build_result(animation, new_frames)


# ---------------------------------------------------------------------------
# 6. Simultaneous stop
# ---------------------------------------------------------------------------


def simultaneous_stop(
    animation: RetargetedAnimation,
    params: SimultaneousStopParams | None = None,
) -> RetargetedAnimation:
    """Make all bones stop moving on the same frame.

    Finds the frame where each bone's velocity drops below threshold,
    then snaps all bones to stop on the earliest such frame.  From
    that frame onward, all bones hold their pose from that frame.

    Args:
        animation: Source animation.
        params: Configuration parameters.

    Returns:
        Degraded animation with simultaneous settling.
    """
    if animation.frame_count < MIN_FRAMES:
        logger.warning("Animation too short (%d frames) for simultaneous_stop", animation.frame_count)
        return animation

    params = params or SimultaneousStopParams()
    bone_arrays = _rot_to_array(animation.frames)

    # Find the last frame of significant motion for each bone
    # (scanning backwards from the end)
    last_motion_frames: list[int] = []
    for bone in STRATA_BONES:
        arr = bone_arrays[bone]
        total_range = arr.max(axis=0) - arr.min(axis=0)
        if total_range.max() < 1.0:
            # Bone is essentially static
            continue

        last_frame = 0
        for i in range(arr.shape[0] - 1, 0, -1):
            delta = np.sqrt(np.sum((arr[i] - arr[i - 1]) ** 2))
            if delta > params.velocity_threshold:
                last_frame = i
                break
        last_motion_frames.append(last_frame)

    if not last_motion_frames:
        return animation

    # All bones stop at the earliest "last motion" frame
    stop_frame = min(last_motion_frames)
    stop_frame = max(1, stop_frame)  # At least keep frame 0 and 1

    frozen_frame = animation.frames[stop_frame]
    new_frames = list(animation.frames[: stop_frame + 1])
    new_frames.extend(frozen_frame for _ in range(stop_frame + 1, animation.frame_count))

    logger.info(
        "simultaneous_stop: all bones frozen at frame %d/%d",
        stop_frame,
        animation.frame_count - 1,
    )
    return _build_result(animation, new_frames)


# ---------------------------------------------------------------------------
# 7. Remove anticipation
# ---------------------------------------------------------------------------


def remove_anticipation(
    animation: RetargetedAnimation,
    params: RemoveAnticipationParams | None = None,
) -> RetargetedAnimation:
    """Delete counter-movement frames before main motion.

    Detects frames where bones move in the opposite direction of their
    subsequent main motion (anticipation), and removes them.

    Args:
        animation: Source animation.
        params: Configuration parameters.

    Returns:
        Degraded animation with anticipation frames removed.
    """
    if animation.frame_count < MIN_FRAMES:
        logger.warning("Animation too short (%d frames) for remove_anticipation", animation.frame_count)
        return animation

    params = params or RemoveAnticipationParams()
    bone_arrays = _rot_to_array(animation.frames)

    # For each bone, compute per-frame velocity (signed, per axis)
    # Anticipation = short sequence of frames where velocity sign is opposite
    # to the dominant motion direction that follows
    anticipation_frames: set[int] = set()

    for bone in STRATA_BONES:
        arr = bone_arrays[bone]
        total_range = arr.max(axis=0) - arr.min(axis=0)
        if total_range.max() < params.velocity_threshold:
            continue

        velocity = np.diff(arr, axis=0)  # (num_frames-1, 3)

        for axis in range(3):
            if total_range[axis] < params.velocity_threshold:
                continue

            vel = velocity[:, axis]
            i = 0
            while i < len(vel):
                # Skip frames with negligible velocity
                if abs(vel[i]) < params.velocity_threshold:
                    i += 1
                    continue

                # Current direction
                direction = 1 if vel[i] > 0 else -1

                # Look ahead to find if direction reverses soon
                run_end = i + 1
                while run_end < len(vel) and vel[run_end] * direction > -params.velocity_threshold:
                    run_end += 1

                if run_end >= len(vel):
                    break

                # Found a reversal at run_end.  Check if the initial run
                # is short (anticipation) and the following run is longer (main motion)
                run_length = run_end - i
                if run_length > params.max_anticipation_frames:
                    i = run_end
                    continue

                # Check subsequent motion length
                reverse_end = run_end + 1
                while reverse_end < len(vel) and vel[reverse_end] * direction < params.velocity_threshold:
                    reverse_end += 1

                subsequent_length = reverse_end - run_end
                if subsequent_length > run_length:
                    # The initial run is anticipation — mark those frames
                    # Frame indices: velocity[i] corresponds to motion from frame i to i+1
                    for f in range(i + 1, run_end + 1):
                        anticipation_frames.add(f)

                i = run_end

    if not anticipation_frames:
        logger.info("remove_anticipation: no anticipation frames detected")
        return animation

    # Always keep first and last frames
    anticipation_frames.discard(0)
    anticipation_frames.discard(animation.frame_count - 1)

    kept_indices = [i for i in range(animation.frame_count) if i not in anticipation_frames]
    new_frames = [animation.frames[i] for i in kept_indices]

    logger.info(
        "remove_anticipation: removed %d anticipation frames (%d → %d)",
        len(anticipation_frames),
        animation.frame_count,
        len(new_frames),
    )
    return _build_result(animation, new_frames)


# ---------------------------------------------------------------------------
# Degradation registry
# ---------------------------------------------------------------------------

DEGRADATION_TYPES: dict[str, str] = {
    "strip_to_extremes": "Strip to extremes — keep only max/min joint angle frames",
    "linearize_arcs": "Linearize arcs — replace curved paths with straight lines",
    "remove_easing": "Remove easing — replace ease-in/out with linear timing",
    "remove_secondary": "Remove secondary — lock secondary bones to parent",
    "reduce_framerate": "Reduce framerate — keep every Nth frame",
    "simultaneous_stop": "Simultaneous stop — all bones stop on same frame",
    "remove_anticipation": "Remove anticipation — delete counter-movement frames",
}

_DEGRADATION_FUNCTIONS = {
    "strip_to_extremes": strip_to_extremes,
    "linearize_arcs": linearize_arcs,
    "remove_easing": remove_easing,
    "remove_secondary": remove_secondary,
    "reduce_framerate": reduce_framerate,
    "simultaneous_stop": simultaneous_stop,
    "remove_anticipation": remove_anticipation,
}

_DEFAULT_PARAMS: dict[str, Any] = {
    "strip_to_extremes": StripToExtremesParams(),
    "linearize_arcs": LinearizeArcsParams(),
    "remove_easing": RemoveEasingParams(),
    "remove_secondary": RemoveSecondaryParams(),
    "reduce_framerate": ReduceFramerateParams(),
    "simultaneous_stop": SimultaneousStopParams(),
    "remove_anticipation": RemoveAnticipationParams(),
}


def apply_degradation(
    animation: RetargetedAnimation,
    degradation_type: str,
    params: Any | None = None,
) -> RetargetedAnimation:
    """Apply a named degradation to an animation.

    Args:
        animation: Source animation.
        degradation_type: One of the keys in ``DEGRADATION_TYPES``.
        params: Type-specific parameter dataclass, or None for defaults.

    Returns:
        Degraded animation.

    Raises:
        ValueError: If degradation_type is not recognized.
    """
    func = _DEGRADATION_FUNCTIONS.get(degradation_type)
    if func is None:
        raise ValueError(
            f"Unknown degradation type: {degradation_type!r}. "
            f"Valid types: {', '.join(DEGRADATION_TYPES)}"
        )
    params = params or _DEFAULT_PARAMS[degradation_type]
    return func(animation, params)


# ---------------------------------------------------------------------------
# Blueprint JSON loading
# ---------------------------------------------------------------------------


def load_blueprint(path: Path | str) -> RetargetedAnimation:
    """Load a Strata blueprint JSON file as a RetargetedAnimation.

    Args:
        path: Path to a blueprint .json file.

    Returns:
        RetargetedAnimation reconstructed from the blueprint.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON structure is invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Blueprint file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    frame_rate = float(data.get("frame_rate", 30.0))
    rotation_order = data.get("rotation_order", "ZXY")
    raw_frames = data.get("frames", [])

    frames: list[RetargetedFrame] = []
    for raw_frame in raw_frames:
        rotations: dict[str, tuple[float, float, float]] = {}
        root_position = (0.0, 0.0, 0.0)

        for bone in STRATA_BONES:
            bone_data = raw_frame.get(bone, {})
            rot = bone_data.get("rotation", [0.0, 0.0, 0.0])
            rotations[bone] = (float(rot[0]), float(rot[1]), float(rot[2]))

            if bone == "hips":
                pos = bone_data.get("position", [0.0, 0.0, 0.0])
                root_position = (float(pos[0]), float(pos[1]), float(pos[2]))

        frames.append(RetargetedFrame(rotations=rotations, root_position=root_position))

    return RetargetedAnimation(
        frames=frames,
        frame_count=len(frames),
        frame_rate=frame_rate,
        rotation_order=rotation_order,
    )


# ---------------------------------------------------------------------------
# Training pair output
# ---------------------------------------------------------------------------


@dataclass
class TrainingPair:
    """A degraded/original training pair with metadata.

    Attributes:
        degraded: The degraded animation.
        original_path: Path to the original source file.
        degradation_type: Which degradation was applied.
        params_summary: Human-readable summary of parameters used.
    """

    degraded: RetargetedAnimation
    original_path: str
    degradation_type: str
    params_summary: str


def save_training_pair(
    pair: TrainingPair,
    output_dir: Path,
    base_name: str,
) -> Path:
    """Save a training pair as JSON.

    The JSON contains the degraded animation frames and metadata linking
    back to the original.

    Args:
        pair: Training pair to save.
        output_dir: Directory to write the output file.
        base_name: Base filename (without extension).

    Returns:
        Path to the saved JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{base_name}_{pair.degradation_type}.json"
    output_path = output_dir / filename

    # Build degraded frames as compact JSON
    frames_out: list[dict[str, Any]] = []
    for frame in pair.degraded.frames:
        frame_dict: dict[str, Any] = {}
        for bone in STRATA_BONES:
            rot = frame.rotations.get(bone, (0.0, 0.0, 0.0))
            bone_data: dict[str, Any] = {
                "rotation": [round(v, 4) for v in rot],
            }
            if bone == "hips":
                bone_data["position"] = [round(v, 4) for v in frame.root_position]
            frame_dict[bone] = bone_data
        frames_out.append(frame_dict)

    output_data = {
        "degradation_type": pair.degradation_type,
        "params": pair.params_summary,
        "original_path": pair.original_path,
        "frame_count": pair.degraded.frame_count,
        "frame_rate": pair.degraded.frame_rate,
        "rotation_order": pair.degraded.rotation_order,
        "frames": frames_out,
    }

    output_path.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info("Saved training pair: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def process_file(
    input_path: Path,
    output_dir: Path,
    degradation_types: list[str] | None = None,
    param_overrides: dict[str, Any] | None = None,
) -> list[Path]:
    """Process a single BVH or blueprint file through all degradation types.

    Args:
        input_path: Path to a .bvh or .json (blueprint) file.
        output_dir: Directory for output training pairs.
        degradation_types: Which degradations to apply (None = all).
        param_overrides: Optional per-type parameter overrides.

    Returns:
        List of output file paths.
    """
    suffix = input_path.suffix.lower()
    if suffix == ".bvh":
        bvh = parse_bvh(input_path)
        animation = retarget(bvh)
    elif suffix == ".json":
        animation = load_blueprint(input_path)
    else:
        logger.warning("Unsupported file type: %s — skipping", input_path)
        return []

    if animation.frame_count < MIN_FRAMES:
        logger.warning(
            "Skipping %s: too few frames (%d < %d)",
            input_path.name,
            animation.frame_count,
            MIN_FRAMES,
        )
        return []

    overrides = param_overrides or {}
    types_to_apply = degradation_types or list(DEGRADATION_TYPES)
    base_name = input_path.stem
    output_paths: list[Path] = []

    for deg_type in types_to_apply:
        params = overrides.get(deg_type)
        degraded = apply_degradation(animation, deg_type, params)
        effective_params = params or _DEFAULT_PARAMS[deg_type]
        pair = TrainingPair(
            degraded=degraded,
            original_path=str(input_path),
            degradation_type=deg_type,
            params_summary=str(effective_params),
        )
        path = save_training_pair(pair, output_dir, base_name)
        output_paths.append(path)

    return output_paths


def process_batch(
    input_dir: Path,
    output_dir: Path,
    degradation_types: list[str] | None = None,
    param_overrides: dict[str, Any] | None = None,
) -> list[Path]:
    """Process all BVH and blueprint files in a directory.

    Args:
        input_dir: Directory containing .bvh and/or .json files.
        output_dir: Directory for output training pairs.
        degradation_types: Which degradations to apply (None = all).
        param_overrides: Optional per-type parameter overrides.

    Returns:
        List of all output file paths.
    """
    input_files = sorted(
        list(input_dir.glob("*.bvh")) + list(input_dir.glob("*.json"))
    )

    if not input_files:
        logger.warning("No BVH or JSON files found in %s", input_dir)
        return []

    logger.info("Processing %d files from %s", len(input_files), input_dir)

    all_outputs: list[Path] = []
    for input_path in input_files:
        outputs = process_file(input_path, output_dir, degradation_types, param_overrides)
        all_outputs.extend(outputs)

    logger.info(
        "Batch complete: %d files → %d training pairs",
        len(input_files),
        len(all_outputs),
    )
    return all_outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for synthetic animation degradation."""
    parser = argparse.ArgumentParser(
        description="Synthetic animation degradation pipeline — create training pairs "
        "by systematically stripping animation principles.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input BVH/JSON file or directory for batch processing",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output/animation/degraded",
        help="Output directory for training pairs (default: output/animation/degraded)",
    )
    parser.add_argument(
        "-t", "--types",
        help="Comma-separated degradation types to apply (default: all). "
        f"Available: {', '.join(DEGRADATION_TYPES)}",
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="List all available degradation types and exit",
    )
    parser.add_argument(
        "--framerate-factor",
        type=int,
        default=ReduceFramerateParams.factor,
        help=f"Frame reduction factor for reduce_framerate (default: {ReduceFramerateParams.factor})",
    )
    parser.add_argument(
        "--keyframe-interval",
        type=int,
        default=LinearizeArcsParams.keyframe_interval,
        help=f"Keyframe interval for linearize_arcs (default: {LinearizeArcsParams.keyframe_interval})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.list_types:
        for name, desc in DEGRADATION_TYPES.items():
            print(f"  {name:25s} {desc}")
        sys.exit(0)

    # Parse degradation types
    degradation_types: list[str] | None = None
    if args.types:
        degradation_types = [t.strip() for t in args.types.split(",")]
        for t in degradation_types:
            if t not in DEGRADATION_TYPES:
                print(f"ERROR: Unknown degradation type: {t!r}", file=sys.stderr)
                print(f"Available: {', '.join(DEGRADATION_TYPES)}", file=sys.stderr)
                sys.exit(1)

    if args.input is None:
        parser.error("the following arguments are required: input")

    # Build param overrides from CLI flags
    param_overrides: dict[str, Any] = {}
    if args.framerate_factor != ReduceFramerateParams.factor:
        param_overrides["reduce_framerate"] = ReduceFramerateParams(factor=args.framerate_factor)
    if args.keyframe_interval != LinearizeArcsParams.keyframe_interval:
        param_overrides["linearize_arcs"] = LinearizeArcsParams(keyframe_interval=args.keyframe_interval)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if input_path.is_dir():
        outputs = process_batch(input_path, output_dir, degradation_types, param_overrides)
    elif input_path.is_file():
        outputs = process_file(input_path, output_dir, degradation_types, param_overrides)
    else:
        print(f"ERROR: Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Generated {len(outputs)} training pairs in {output_dir}")


if __name__ == "__main__":
    main()

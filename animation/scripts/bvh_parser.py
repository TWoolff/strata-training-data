"""Parse BVH (Biovision Hierarchy) motion capture files.

Extracts skeleton structure and per-frame motion data into Python data
structures.  Handles CMU Graphics Lab and SFU Motion Capture BVH variants.

No Blender dependency — pure Python + numpy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class BVHJoint:
    """A single joint in the BVH skeleton hierarchy."""

    name: str
    parent: str | None
    offset: tuple[float, float, float]
    channels: list[str]  # e.g. ["Xrotation", "Yrotation", "Zrotation"]
    children: list[str] = field(default_factory=list)


@dataclass
class BVHSkeleton:
    """Parsed BVH skeleton hierarchy."""

    joints: dict[str, BVHJoint]  # name -> joint data
    root: str  # root joint name
    joint_order: list[str] = field(default_factory=list)  # hierarchy traversal order


@dataclass
class BVHMotion:
    """Parsed BVH motion (frame) data."""

    frame_count: int
    frame_time: float  # seconds per frame
    frames: list[dict[str, list[float]]]  # per-frame, per-joint channel values


@dataclass
class BVHFile:
    """Complete parsed BVH file: skeleton + motion."""

    skeleton: BVHSkeleton
    motion: BVHMotion


# ---------------------------------------------------------------------------
# HIERARCHY parser
# ---------------------------------------------------------------------------

_CHANNEL_NAMES = frozenset({
    "Xposition", "Yposition", "Zposition",
    "Xrotation", "Yrotation", "Zrotation",
})


class BVHParseError(ValueError):
    """Raised when the BVH file is malformed."""


def _parse_hierarchy(lines: list[str], idx: int) -> tuple[BVHSkeleton, int]:
    """Parse the HIERARCHY section of a BVH file.

    Args:
        lines: All lines of the BVH file (stripped).
        idx: Current line index (should point to "HIERARCHY").

    Returns:
        (skeleton, next_line_index).

    Raises:
        BVHParseError: If the hierarchy is malformed.
    """
    if lines[idx].upper() != "HIERARCHY":
        raise BVHParseError(f"Expected HIERARCHY at line {idx + 1}, got: {lines[idx]!r}")
    idx += 1

    joints: dict[str, BVHJoint] = {}
    joint_order: list[str] = []
    root_name: str | None = None

    stack: list[str] = []
    end_site_counter = 0

    while idx < len(lines):
        line = lines[idx]
        tokens = line.split()

        if not tokens:
            idx += 1
            continue

        keyword = tokens[0].upper()

        if keyword == "MOTION":
            break

        if keyword in ("ROOT", "JOINT"):
            if len(tokens) < 2:
                raise BVHParseError(f"Missing joint name at line {idx + 1}")
            joint_name = tokens[1]

            parent_name = stack[-1] if stack else None
            joint = BVHJoint(
                name=joint_name,
                parent=parent_name,
                offset=(0.0, 0.0, 0.0),
                channels=[],
            )

            if parent_name and parent_name in joints:
                joints[parent_name].children.append(joint_name)

            joints[joint_name] = joint
            joint_order.append(joint_name)

            if keyword == "ROOT":
                if root_name is not None:
                    raise BVHParseError(f"Multiple ROOT joints at line {idx + 1}")
                root_name = joint_name

            # Next line should be '{'
            idx += 1
            idx = _skip_blank(lines, idx)
            if idx >= len(lines) or lines[idx].strip() != "{":
                raise BVHParseError(
                    f"Expected '{{' after {keyword} {joint_name} at line {idx + 1}"
                )
            stack.append(joint_name)
            idx += 1
            continue

        if keyword == "END" and len(tokens) >= 2 and tokens[1].upper() == "SITE":
            # End Site — synthetic leaf with offset only, no channels
            end_site_counter += 1
            parent_name = stack[-1] if stack else None
            end_name = f"{parent_name}_End" if parent_name else f"EndSite_{end_site_counter}"

            joint = BVHJoint(
                name=end_name,
                parent=parent_name,
                offset=(0.0, 0.0, 0.0),
                channels=[],
            )
            if parent_name and parent_name in joints:
                joints[parent_name].children.append(end_name)

            joints[end_name] = joint
            joint_order.append(end_name)

            # Read opening brace
            idx += 1
            idx = _skip_blank(lines, idx)
            if idx >= len(lines) or lines[idx].strip() != "{":
                raise BVHParseError(f"Expected '{{' after End Site at line {idx + 1}")

            stack.append(end_name)
            idx += 1
            continue

        if keyword == "OFFSET":
            if len(tokens) < 4:
                raise BVHParseError(f"OFFSET needs 3 values at line {idx + 1}")
            if not stack:
                raise BVHParseError(f"OFFSET outside of a joint block at line {idx + 1}")
            joint_name = stack[-1]
            joints[joint_name].offset = (
                float(tokens[1]),
                float(tokens[2]),
                float(tokens[3]),
            )
            idx += 1
            continue

        if keyword == "CHANNELS":
            if len(tokens) < 2:
                raise BVHParseError(f"CHANNELS needs count at line {idx + 1}")
            if not stack:
                raise BVHParseError(f"CHANNELS outside of a joint block at line {idx + 1}")

            num_channels = int(tokens[1])
            channel_names = tokens[2 : 2 + num_channels]
            if len(channel_names) != num_channels:
                raise BVHParseError(
                    f"Expected {num_channels} channel names, "
                    f"got {len(channel_names)} at line {idx + 1}"
                )
            for ch in channel_names:
                if ch not in _CHANNEL_NAMES:
                    logger.warning(
                        "Unrecognized channel %r at line %d — keeping as-is",
                        ch,
                        idx + 1,
                    )

            joint_name = stack[-1]
            joints[joint_name].channels = channel_names
            idx += 1
            continue

        if line.strip() == "}":
            if not stack:
                raise BVHParseError(f"Unexpected '}}' at line {idx + 1}")
            stack.pop()
            idx += 1
            continue

        # Skip unrecognized lines (some BVH files have comments or blank lines)
        logger.debug("Skipping unrecognized line %d: %r", idx + 1, line)
        idx += 1

    if root_name is None:
        raise BVHParseError("No ROOT joint found in HIERARCHY section")

    skeleton = BVHSkeleton(joints=joints, root=root_name, joint_order=joint_order)
    return skeleton, idx


def _skip_blank(lines: list[str], idx: int) -> int:
    """Advance past blank lines."""
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    return idx


# ---------------------------------------------------------------------------
# MOTION parser
# ---------------------------------------------------------------------------


def _parse_motion(
    lines: list[str],
    idx: int,
    skeleton: BVHSkeleton,
) -> BVHMotion:
    """Parse the MOTION section of a BVH file.

    Args:
        lines: All lines of the BVH file (stripped).
        idx: Current line index (should point to "MOTION").
        skeleton: Already-parsed skeleton (needed for channel layout).

    Returns:
        Parsed motion data.

    Raises:
        BVHParseError: If the motion section is malformed.
    """
    if idx >= len(lines) or lines[idx].upper() != "MOTION":
        raise BVHParseError(f"Expected MOTION at line {idx + 1}")
    idx += 1

    # Parse "Frames: N"
    idx = _skip_blank(lines, idx)
    if idx >= len(lines):
        raise BVHParseError("Unexpected end of file — missing Frames count")

    frame_count = _parse_key_value_int(lines[idx], "Frames", idx)
    idx += 1

    # Parse "Frame Time: F"
    idx = _skip_blank(lines, idx)
    if idx >= len(lines):
        raise BVHParseError("Unexpected end of file — missing Frame Time")

    frame_time = _parse_key_value_float(lines[idx], "Frame Time", idx)
    idx += 1

    # Build channel layout: ordered list of (joint_name, num_channels)
    channel_layout: list[tuple[str, int]] = []
    total_channels = 0
    for joint_name in skeleton.joint_order:
        joint = skeleton.joints[joint_name]
        if joint.channels:
            channel_layout.append((joint_name, len(joint.channels)))
            total_channels += len(joint.channels)

    # Parse frame data
    frames: list[dict[str, list[float]]] = []
    for frame_idx in range(frame_count):
        idx = _skip_blank(lines, idx)
        if idx >= len(lines):
            if frame_idx == 0 and frame_count > 0:
                raise BVHParseError(
                    f"Expected {frame_count} frames but file ended after header"
                )
            logger.warning(
                "BVH file has %d frames declared but only %d found",
                frame_count,
                frame_idx,
            )
            break

        values = lines[idx].split()
        if len(values) < total_channels:
            raise BVHParseError(
                f"Frame {frame_idx} has {len(values)} values, "
                f"expected {total_channels} at line {idx + 1}"
            )

        float_values = [float(v) for v in values[:total_channels]]

        # Distribute values to joints
        frame_data: dict[str, list[float]] = {}
        offset = 0
        for joint_name, num_ch in channel_layout:
            frame_data[joint_name] = float_values[offset : offset + num_ch]
            offset += num_ch

        frames.append(frame_data)
        idx += 1

    return BVHMotion(frame_count=len(frames), frame_time=frame_time, frames=frames)


def _parse_key_value_int(line: str, key: str, line_idx: int) -> int:
    """Parse a 'Key: value' line and return the integer value."""
    parts = line.split(":")
    if len(parts) < 2 or parts[0].strip().lower() != key.lower():
        raise BVHParseError(f"Expected '{key}: <int>' at line {line_idx + 1}, got: {line!r}")
    try:
        return int(parts[1].strip())
    except ValueError as exc:
        raise BVHParseError(
            f"Invalid integer for {key} at line {line_idx + 1}: {parts[1].strip()!r}"
        ) from exc


def _parse_key_value_float(line: str, key: str, line_idx: int) -> float:
    """Parse a 'Key: value' line and return the float value."""
    parts = line.split(":", 1)
    if len(parts) < 2 or parts[0].strip().lower() != key.lower():
        raise BVHParseError(f"Expected '{key}: <float>' at line {line_idx + 1}, got: {line!r}")
    try:
        return float(parts[1].strip())
    except ValueError as exc:
        raise BVHParseError(
            f"Invalid float for {key} at line {line_idx + 1}: {parts[1].strip()!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_bvh(path: Path | str) -> BVHFile:
    """Parse a BVH file and return skeleton + motion data.

    Args:
        path: Path to a .bvh file.

    Returns:
        BVHFile containing the parsed skeleton and motion data.

    Raises:
        FileNotFoundError: If the file does not exist.
        BVHParseError: If the file is malformed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"BVH file not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    stripped = [line.strip() for line in lines]

    # Skip leading blank lines
    idx = _skip_blank(stripped, 0)
    if idx >= len(stripped):
        raise BVHParseError(f"Empty BVH file: {path}")

    skeleton, idx = _parse_hierarchy(stripped, idx)
    idx = _skip_blank(stripped, idx)

    if idx >= len(stripped):
        # Skeleton-only file — return empty motion
        logger.warning("BVH file %s has no MOTION section", path)
        motion = BVHMotion(frame_count=0, frame_time=0.0, frames=[])
    else:
        motion = _parse_motion(stripped, idx, skeleton)

    joint_count = len([j for j in skeleton.joints.values() if j.channels])
    channel_count = sum(len(j.channels) for j in skeleton.joints.values())
    logger.info(
        "Parsed %s: %d joints (%d with channels), %d total channels, %d frames @ %.4fs",
        path.name,
        len(skeleton.joints),
        joint_count,
        channel_count,
        motion.frame_count,
        motion.frame_time,
    )

    return BVHFile(skeleton=skeleton, motion=motion)


def get_frame_array(motion: BVHMotion) -> NDArray[np.float64]:
    """Convert motion frames to a numpy array for batch processing.

    Args:
        motion: Parsed BVH motion data.

    Returns:
        Array of shape (frame_count, total_channels) with all channel values.
        Returns an empty (0, 0) array if there are no frames.
    """
    if not motion.frames:
        return np.empty((0, 0), dtype=np.float64)

    rows = []
    for frame_data in motion.frames:
        row: list[float] = []
        for values in frame_data.values():
            row.extend(values)
        rows.append(row)

    return np.array(rows, dtype=np.float64)


def get_joint_frame(
    motion: BVHMotion,
    joint_name: str,
    frame_index: int,
) -> list[float]:
    """Get channel values for a specific joint at a specific frame.

    Args:
        joint_name: Name of the joint.
        frame_index: Zero-based frame index.

    Returns:
        List of channel values for the joint at the given frame.

    Raises:
        KeyError: If joint_name is not in the frame data.
        IndexError: If frame_index is out of range.
    """
    if frame_index < 0 or frame_index >= motion.frame_count:
        raise IndexError(
            f"Frame index {frame_index} out of range (0-{motion.frame_count - 1})"
        )
    frame = motion.frames[frame_index]
    if joint_name not in frame:
        raise KeyError(f"Joint {joint_name!r} not found in frame data")
    return frame[joint_name]

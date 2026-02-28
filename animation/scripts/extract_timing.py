"""Extract frame spacing, velocity curves, and timing patterns from labeled BVH clips.

Reads labeled clips from ``animation/labels/cmu_action_labels.csv``, parses each
BVH file, and computes root velocity, joint angular velocity, and timing
statistics (acceleration, deceleration, hold phases).  Results are grouped by
action type and written as JSON files to ``animation/timing-norms/``.

No Blender dependency — pure Python + numpy.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from animation.scripts.bvh_parser import BVHFile, parse_bvh

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Root velocity below this threshold (units/sec) counts as a "hold" frame.
HOLD_VELOCITY_THRESHOLD: float = 0.1

# Default path constants (relative to project root)
DEFAULT_LABELS_CSV: str = "animation/labels/cmu_action_labels.csv"
DEFAULT_MOCAP_DIR: str = "data/mocap"
DEFAULT_OUTPUT_DIR: str = "animation/timing-norms"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ClipTimingStats:
    """Timing statistics for a single acceleration/deceleration analysis."""

    acceleration_frames: int = 0
    deceleration_frames: int = 0
    hold_frames: int = 0
    acceleration_ratio: float = 0.0
    deceleration_ratio: float = 0.0
    hold_ratio: float = 0.0


@dataclass
class VelocityStats:
    """Summary statistics for a velocity series."""

    mean: float = 0.0
    max: float = 0.0
    min: float = 0.0


@dataclass
class ClipTiming:
    """Extracted timing data for a single BVH clip."""

    filename: str = ""
    subcategory: str = ""
    duration_seconds: float = 0.0
    frame_count: int = 0
    frame_rate: float = 30.0
    root_velocity: VelocityStats = field(default_factory=VelocityStats)
    root_velocity_curve: list[float] = field(default_factory=list)
    joint_angular_velocities: dict[str, VelocityStats] = field(default_factory=dict)
    timing: ClipTimingStats = field(default_factory=ClipTimingStats)


@dataclass
class ActionNormStats:
    """Aggregated norm statistics for a single metric across clips."""

    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0


@dataclass
class ActionTimingNorms:
    """Timing norms for an action type, aggregated across all its clips."""

    action_type: str = ""
    clip_count: int = 0
    clips: list[ClipTiming] = field(default_factory=list)
    norms: dict[str, ActionNormStats] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


@dataclass
class LabelRow:
    """A single row from cmu_action_labels.csv."""

    filename: str
    action_type: str
    subcategory: str
    quality: str
    strata_compatible: str


def load_labels(csv_path: Path) -> list[LabelRow]:
    """Load clip labels from the CMU action labels CSV.

    Args:
        csv_path: Path to cmu_action_labels.csv.

    Returns:
        List of label rows.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

    rows: list[LabelRow] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                LabelRow(
                    filename=row["filename"],
                    action_type=row["action_type"],
                    subcategory=row["subcategory"],
                    quality=row["quality"],
                    strata_compatible=row["strata_compatible"],
                )
            )

    logger.info("Loaded %d label rows from %s", len(rows), csv_path)
    return rows


# ---------------------------------------------------------------------------
# Velocity extraction
# ---------------------------------------------------------------------------


def _extract_root_positions(bvh: BVHFile) -> np.ndarray:
    """Extract per-frame root (hip) positions as an (N, 3) array.

    Args:
        bvh: Parsed BVH file.

    Returns:
        Array of shape (frame_count, 3) with XYZ root positions.
        Returns empty (0, 3) array if no frames or no position channels.
    """
    root_name = bvh.skeleton.root
    root_joint = bvh.skeleton.joints[root_name]
    channels = root_joint.channels

    pos_indices: dict[str, int] = {}
    for i, ch in enumerate(channels):
        if ch in ("Xposition", "Yposition", "Zposition"):
            pos_indices[ch] = i

    if len(pos_indices) < 3:
        return np.empty((0, 3), dtype=np.float64)

    positions = np.empty((bvh.motion.frame_count, 3), dtype=np.float64)
    for frame_idx, frame_data in enumerate(bvh.motion.frames):
        values = frame_data.get(root_name, [])
        if len(values) <= max(pos_indices.values()):
            positions[frame_idx] = [0.0, 0.0, 0.0]
            continue
        positions[frame_idx] = [
            values[pos_indices["Xposition"]],
            values[pos_indices["Yposition"]],
            values[pos_indices["Zposition"]],
        ]

    return positions


def compute_root_velocity(bvh: BVHFile) -> np.ndarray:
    """Compute per-frame root velocity (speed) in units/second.

    Velocity at frame *i* is the Euclidean distance between frame *i* and
    frame *i-1*, divided by frame_time.  The first frame has velocity 0.

    Args:
        bvh: Parsed BVH file.

    Returns:
        1-D array of length frame_count with velocity values.
        Returns empty array if fewer than 2 frames.
    """
    positions = _extract_root_positions(bvh)
    if positions.shape[0] < 2:
        return np.array([], dtype=np.float64)

    frame_time = bvh.motion.frame_time if bvh.motion.frame_time > 0 else 1.0 / 30.0
    displacements = np.diff(positions, axis=0)
    speeds = np.linalg.norm(displacements, axis=1) / frame_time

    # Prepend 0 for the first frame
    return np.concatenate([[0.0], speeds])


def _extract_joint_rotations(bvh: BVHFile, joint_name: str) -> np.ndarray:
    """Extract per-frame rotation values for a joint as an (N, 3) array.

    Args:
        bvh: Parsed BVH file.
        joint_name: Name of the joint.

    Returns:
        Array of shape (frame_count, 3) with rotation values in degrees.
        Returns empty (0, 3) array if joint has no rotation channels.
    """
    joint = bvh.skeleton.joints.get(joint_name)
    if joint is None:
        return np.empty((0, 3), dtype=np.float64)

    channels = joint.channels
    rot_indices: dict[str, int] = {}
    for i, ch in enumerate(channels):
        if ch in ("Xrotation", "Yrotation", "Zrotation"):
            rot_indices[ch] = i

    if len(rot_indices) < 3:
        return np.empty((0, 3), dtype=np.float64)

    rotations = np.empty((bvh.motion.frame_count, 3), dtype=np.float64)
    for frame_idx, frame_data in enumerate(bvh.motion.frames):
        values = frame_data.get(joint_name, [])
        if len(values) <= max(rot_indices.values()):
            rotations[frame_idx] = [0.0, 0.0, 0.0]
            continue
        rotations[frame_idx] = [
            values[rot_indices["Xrotation"]],
            values[rot_indices["Yrotation"]],
            values[rot_indices["Zrotation"]],
        ]

    return rotations


def compute_joint_angular_velocity(bvh: BVHFile, joint_name: str) -> np.ndarray:
    """Compute per-frame angular velocity for a joint in degrees/second.

    Angular velocity at frame *i* is the Euclidean norm of the rotation delta
    between frame *i* and frame *i-1*, divided by frame_time.  First frame = 0.

    Args:
        bvh: Parsed BVH file.
        joint_name: Name of the joint.

    Returns:
        1-D array of length frame_count.
        Returns empty array if fewer than 2 frames.
    """
    rotations = _extract_joint_rotations(bvh, joint_name)
    if rotations.shape[0] < 2:
        return np.array([], dtype=np.float64)

    frame_time = bvh.motion.frame_time if bvh.motion.frame_time > 0 else 1.0 / 30.0
    deltas = np.diff(rotations, axis=0)
    angular_speeds = np.linalg.norm(deltas, axis=1) / frame_time

    return np.concatenate([[0.0], angular_speeds])


# ---------------------------------------------------------------------------
# Timing analysis
# ---------------------------------------------------------------------------


def compute_timing_stats(
    velocity_curve: np.ndarray,
    hold_threshold: float = HOLD_VELOCITY_THRESHOLD,
) -> ClipTimingStats:
    """Classify frames into acceleration, deceleration, and hold phases.

    Acceleration: velocity derivative > 0 (speed increasing).
    Deceleration: velocity derivative < 0 (speed decreasing).
    Hold: root velocity below *hold_threshold*.

    Hold frames are classified first (regardless of derivative), then
    the remaining frames are split by derivative sign.

    Args:
        velocity_curve: 1-D array of root velocity values.
        hold_threshold: Velocity threshold below which a frame is "hold".

    Returns:
        ClipTimingStats with frame counts and ratios.
    """
    n = len(velocity_curve)
    if n < 2:
        return ClipTimingStats()

    hold_mask = velocity_curve < hold_threshold
    hold_frames = int(np.sum(hold_mask))

    # Velocity derivative (acceleration)
    accel = np.diff(velocity_curve)
    # Pad to same length (first frame has no derivative, treat as 0)
    accel_padded = np.concatenate([[0.0], accel])

    # Non-hold frames classified by derivative sign
    accel_frames = int(np.sum((accel_padded > 0) & ~hold_mask))
    decel_frames = int(np.sum((accel_padded < 0) & ~hold_mask))

    return ClipTimingStats(
        acceleration_frames=accel_frames,
        deceleration_frames=decel_frames,
        hold_frames=hold_frames,
        acceleration_ratio=round(accel_frames / n, 4),
        deceleration_ratio=round(decel_frames / n, 4),
        hold_ratio=round(hold_frames / n, 4),
    )


# ---------------------------------------------------------------------------
# Per-clip extraction
# ---------------------------------------------------------------------------


def _joints_with_rotation(bvh: BVHFile) -> list[str]:
    """Return joint names that have rotation channels (excluding End Sites)."""
    result = []
    for name, joint in bvh.skeleton.joints.items():
        if name.endswith("_End"):
            continue
        rot_count = sum(1 for ch in joint.channels if ch.endswith("rotation"))
        if rot_count >= 3:
            result.append(name)
    return result


def extract_clip_timing(bvh: BVHFile, filename: str, subcategory: str) -> ClipTiming:
    """Extract all timing data from a single BVH clip.

    Args:
        bvh: Parsed BVH file.
        filename: Original BVH filename (for metadata).
        subcategory: Action subcategory from the labels CSV.

    Returns:
        ClipTiming with velocity curves, angular velocities, and timing stats.
    """
    frame_time = bvh.motion.frame_time if bvh.motion.frame_time > 0 else 1.0 / 30.0
    frame_rate = round(1.0 / frame_time, 2)
    duration = bvh.motion.frame_count * frame_time

    # Root velocity
    root_vel = compute_root_velocity(bvh)
    if root_vel.size > 0:
        root_vel_stats = VelocityStats(
            mean=round(float(np.mean(root_vel)), 4),
            max=round(float(np.max(root_vel)), 4),
            min=round(float(np.min(root_vel)), 4),
        )
        root_vel_curve = [round(float(v), 4) for v in root_vel]
    else:
        root_vel_stats = VelocityStats()
        root_vel_curve = []

    # Joint angular velocities
    joint_ang_vel: dict[str, VelocityStats] = {}
    for joint_name in _joints_with_rotation(bvh):
        ang_vel = compute_joint_angular_velocity(bvh, joint_name)
        if ang_vel.size > 0:
            joint_ang_vel[joint_name] = VelocityStats(
                mean=round(float(np.mean(ang_vel)), 4),
                max=round(float(np.max(ang_vel)), 4),
                min=round(float(np.min(ang_vel)), 4),
            )

    # Timing stats
    timing = compute_timing_stats(root_vel)

    return ClipTiming(
        filename=filename,
        subcategory=subcategory,
        duration_seconds=round(duration, 4),
        frame_count=bvh.motion.frame_count,
        frame_rate=frame_rate,
        root_velocity=root_vel_stats,
        root_velocity_curve=root_vel_curve,
        joint_angular_velocities=joint_ang_vel,
        timing=timing,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _compute_norm_stats(values: list[float]) -> ActionNormStats:
    """Compute aggregated norm stats from a list of numeric values."""
    if not values:
        return ActionNormStats()
    arr = np.array(values, dtype=np.float64)
    return ActionNormStats(
        mean=round(float(np.mean(arr)), 4),
        std=round(float(np.std(arr)), 4),
        min=round(float(np.min(arr)), 4),
        max=round(float(np.max(arr)), 4),
    )


def aggregate_by_action(clips: dict[str, list[ClipTiming]]) -> dict[str, ActionTimingNorms]:
    """Aggregate clip timing data by action type into timing norms.

    Args:
        clips: Mapping of action_type -> list of ClipTiming for that action.

    Returns:
        Mapping of action_type -> ActionTimingNorms.
    """
    result: dict[str, ActionTimingNorms] = {}

    for action_type, clip_list in sorted(clips.items()):
        norms: dict[str, ActionNormStats] = {}

        durations = [c.duration_seconds for c in clip_list]
        norms["duration_seconds"] = _compute_norm_stats(durations)

        vel_means = [c.root_velocity.mean for c in clip_list]
        norms["root_velocity_mean"] = _compute_norm_stats(vel_means)

        vel_maxes = [c.root_velocity.max for c in clip_list]
        norms["root_velocity_max"] = _compute_norm_stats(vel_maxes)

        accel_ratios = [c.timing.acceleration_ratio for c in clip_list]
        norms["acceleration_ratio"] = _compute_norm_stats(accel_ratios)

        decel_ratios = [c.timing.deceleration_ratio for c in clip_list]
        norms["deceleration_ratio"] = _compute_norm_stats(decel_ratios)

        hold_ratios = [c.timing.hold_ratio for c in clip_list]
        norms["hold_ratio"] = _compute_norm_stats(hold_ratios)

        result[action_type] = ActionTimingNorms(
            action_type=action_type,
            clip_count=len(clip_list),
            clips=clip_list,
            norms=norms,
        )

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_timing_norms(
    norms: dict[str, ActionTimingNorms],
    output_dir: Path,
) -> list[Path]:
    """Write timing norms as JSON files to the output directory.

    Creates one JSON file per action type plus a summary.json.

    Args:
        norms: Mapping of action_type -> ActionTimingNorms.
        output_dir: Directory to write JSON files to.

    Returns:
        List of written file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for action_type, norm_data in sorted(norms.items()):
        out_path = output_dir / f"{action_type}.json"
        data = asdict(norm_data)
        out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        written.append(out_path)
        logger.info("Wrote %s (%d clips)", out_path.name, norm_data.clip_count)

    # Write summary (norms only, no per-clip detail)
    summary: dict[str, dict] = {}
    for action_type, norm_data in sorted(norms.items()):
        summary[action_type] = {
            "clip_count": norm_data.clip_count,
            "norms": {k: asdict(v) for k, v in norm_data.norms.items()},
        }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    written.append(summary_path)
    logger.info("Wrote summary.json (%d action types)", len(summary))

    return written


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def extract_all(
    labels_csv: Path,
    mocap_dir: Path,
    output_dir: Path,
) -> dict[str, ActionTimingNorms]:
    """Run the full timing extraction pipeline.

    Args:
        labels_csv: Path to cmu_action_labels.csv.
        mocap_dir: Directory containing BVH files.
        output_dir: Directory for JSON output.

    Returns:
        Aggregated timing norms by action type.
    """
    labels = load_labels(labels_csv)

    clips_by_action: dict[str, list[ClipTiming]] = {}
    processed = 0
    skipped = 0

    for label in labels:
        bvh_path = mocap_dir / label.filename
        if not bvh_path.is_file():
            logger.warning("BVH file not found, skipping: %s", bvh_path)
            skipped += 1
            continue

        try:
            bvh = parse_bvh(bvh_path)
        except (ValueError, OSError) as exc:
            logger.warning("Failed to parse %s: %s", label.filename, exc)
            skipped += 1
            continue

        if bvh.motion.frame_count < 2:
            logger.warning("Clip %s has < 2 frames, skipping", label.filename)
            skipped += 1
            continue

        clip_timing = extract_clip_timing(bvh, label.filename, label.subcategory)
        clips_by_action.setdefault(label.action_type, []).append(clip_timing)
        processed += 1

    logger.info("Processed %d clips, skipped %d", processed, skipped)

    if not clips_by_action:
        logger.warning("No clips were processed — output will be empty")
        return {}

    norms = aggregate_by_action(clips_by_action)
    write_timing_norms(norms, output_dir)

    return norms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for timing extraction."""
    parser = argparse.ArgumentParser(
        description="Extract timing norms from labeled BVH mocap clips",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path(DEFAULT_LABELS_CSV),
        help=f"Path to labels CSV (default: {DEFAULT_LABELS_CSV})",
    )
    parser.add_argument(
        "--mocap-dir",
        type=Path,
        default=Path(DEFAULT_MOCAP_DIR),
        help=f"Directory containing BVH files (default: {DEFAULT_MOCAP_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for JSON files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    norms = extract_all(args.labels_csv, args.mocap_dir, args.output_dir)
    if norms:
        total_clips = sum(n.clip_count for n in norms.values())
        print(
            f"Extracted timing norms for {len(norms)} action types "
            f"({total_clips} clips) -> {args.output_dir}",
        )
    else:
        print("No timing data extracted. Check --mocap-dir and --labels-csv.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

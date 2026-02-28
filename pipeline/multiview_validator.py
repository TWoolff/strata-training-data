"""Multi-view segmentation consistency validation.

Validates that segmentation and measurement extraction produce consistent
results across camera angles for the same character × pose.  Flags
inconsistencies exceeding a configurable threshold.

Pure Python — no Blender dependency.

PRD reference: Section 13.4 (Multi-View Consistency Pairs).
See also: Issue #86.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DEVIATION_THRESHOLD: float = 0.10  # 10% deviation
MIN_PIXEL_COUNT: int = 50  # Skip regions with fewer pixels (noisy ratios)

# Midline regions expected to be visible from all angles
MIDLINE_REGIONS: set[str] = {"head", "neck", "chest", "spine", "hips"}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ConsistencyFailure:
    """A single multi-view consistency failure."""

    character_id: str
    pose: str
    region: str
    check: str
    angle_a: str
    angle_b: str
    detail: str


@dataclass
class ConsistencyCheckSummary:
    """Summary for a single consistency check type."""

    name: str
    passed: int = 0
    failed: int = 0
    failures: list[ConsistencyFailure] = field(default_factory=list)

    def record_pass(self) -> None:
        self.passed += 1

    def record_fail(self, failure: ConsistencyFailure) -> None:
        self.failed += 1
        self.failures.append(failure)


@dataclass
class ConsistencyReport:
    """Full multi-view consistency validation report."""

    checks: dict[str, ConsistencyCheckSummary] = field(default_factory=dict)
    characters_checked: int = 0
    pose_groups_checked: int = 0
    elapsed_seconds: float = 0.0

    @property
    def all_passed(self) -> bool:
        return all(s.failed == 0 for s in self.checks.values())

    @property
    def total_failures(self) -> int:
        return sum(s.failed for s in self.checks.values())

    def worst_regions(self, top_n: int = 5) -> list[tuple[str, int]]:
        """Return regions with the most failures across all checks."""
        counts: dict[str, int] = {}
        for summary in self.checks.values():
            for f in summary.failures:
                counts[f.region] = counts.get(f.region, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def worst_angle_pairs(self, top_n: int = 5) -> list[tuple[str, int]]:
        """Return angle pairs with the most failures across all checks."""
        counts: dict[str, int] = {}
        for summary in self.checks.values():
            for f in summary.failures:
                pair = f"{f.angle_a} vs {f.angle_b}"
                counts[pair] = counts.get(pair, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "passed": self.all_passed,
            "total_failures": self.total_failures,
            "characters_checked": self.characters_checked,
            "pose_groups_checked": self.pose_groups_checked,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "worst_regions": self.worst_regions(),
            "worst_angle_pairs": self.worst_angle_pairs(),
            "checks": {
                name: {
                    "passed": s.passed,
                    "failed": s.failed,
                    "failures": [
                        {
                            "character_id": f.character_id,
                            "pose": f.pose,
                            "region": f.region,
                            "angle_a": f.angle_a,
                            "angle_b": f.angle_b,
                            "detail": f.detail,
                        }
                        for f in s.failures
                    ],
                }
                for name, s in self.checks.items()
            },
        }


# ---------------------------------------------------------------------------
# Measurement file discovery
# ---------------------------------------------------------------------------


def _discover_measurement_groups(
    measurements_dir: Path,
    characters: list[str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Discover 2D measurement files and group by character+pose.

    Args:
        measurements_dir: Path to the ``measurements_2d/`` directory.
        characters: Optional character IDs to filter.

    Returns:
        Nested dict: ``{char_id}_{pose} -> {angle_name -> measurement_data}``.
    """
    if not measurements_dir.is_dir():
        return {}

    groups: dict[str, dict[str, Any]] = {}

    for path in sorted(measurements_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to read measurement file %s", path)
            continue

        char_id = data.get("character_id", "")
        pose = data.get("pose", "")
        angle = data.get("camera_angle", "")

        if not char_id or not pose or not angle:
            logger.debug("Skipping %s: missing character_id/pose/camera_angle", path)
            continue

        if characters and char_id not in characters:
            continue

        group_key = f"{char_id}_{pose}"
        groups.setdefault(group_key, {})
        groups[group_key][angle] = data

    return groups


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_region_presence(
    angle_measurements: dict[str, dict[str, Any]],
    character_id: str,
    pose: str,
    summary: ConsistencyCheckSummary,
) -> None:
    """Check that midline regions are visible across all available angles.

    Only checks regions in ``MIDLINE_REGIONS`` since limb visibility depends
    on pose and is expected to vary between angles.

    Args:
        angle_measurements: ``{angle_name -> measurement_data}`` for one pose.
        character_id: Character identifier.
        pose: Pose identifier.
        summary: Check summary to record pass/fail.
    """
    angles = sorted(angle_measurements.keys())
    if len(angles) < 2:
        return

    for region_name in MIDLINE_REGIONS:
        # Collect which angles see this region
        visible_angles = []
        invisible_angles = []
        for angle in angles:
            regions = angle_measurements[angle].get("regions", {})
            region_data = regions.get(region_name, {})
            if region_data.get("visible", False):
                visible_angles.append(angle)
            else:
                invisible_angles.append(angle)

        if visible_angles and invisible_angles:
            summary.record_fail(
                ConsistencyFailure(
                    character_id=character_id,
                    pose=pose,
                    region=region_name,
                    check="region_presence",
                    angle_a=visible_angles[0],
                    angle_b=invisible_angles[0],
                    detail=(
                        f"midline region '{region_name}' visible in "
                        f"{visible_angles} but missing in {invisible_angles}"
                    ),
                )
            )
        else:
            summary.record_pass()


def check_pixel_area_consistency(
    angle_measurements: dict[str, dict[str, Any]],
    character_id: str,
    pose: str,
    summary: ConsistencyCheckSummary,
    *,
    threshold: float = DEFAULT_DEVIATION_THRESHOLD,
) -> None:
    """Check that region pixel counts are consistent with foreshortening.

    For each region visible in both the front and side views, validates that
    the ratio of apparent widths is geometrically plausible.  The expected
    relationship is:

        ``width_at_angle ≈ width_front * |cos(θ)| + depth * |sin(θ)|``

    Since we don't know depth from 2D alone, we instead check that pixel
    area (total pixel count) doesn't change by more than the threshold
    between any two angles.  A fully visible region should have roughly
    consistent total pixel area regardless of viewing angle for an
    orthographic camera.

    Args:
        angle_measurements: ``{angle_name -> measurement_data}`` for one pose.
        character_id: Character identifier.
        pose: Pose identifier.
        summary: Check summary to record pass/fail.
        threshold: Maximum allowed relative deviation (default 10%).
    """
    angles = sorted(angle_measurements.keys())
    if len(angles) < 2:
        return

    # Build region -> angle -> pixel_count mapping
    region_pixels: dict[str, dict[str, int]] = {}
    for angle in angles:
        regions = angle_measurements[angle].get("regions", {})
        for region_name, data in regions.items():
            if not data.get("visible", False):
                continue
            pixel_count = data.get("pixel_count", 0)
            if pixel_count < MIN_PIXEL_COUNT:
                continue
            region_pixels.setdefault(region_name, {})[angle] = pixel_count

    # Compare each pair of angles for each region
    for region_name, angle_counts in region_pixels.items():
        angle_list = sorted(angle_counts.keys())
        if len(angle_list) < 2:
            continue

        for i in range(len(angle_list)):
            for j in range(i + 1, len(angle_list)):
                angle_a = angle_list[i]
                angle_b = angle_list[j]
                count_a = angle_counts[angle_a]
                count_b = angle_counts[angle_b]

                mean_count = (count_a + count_b) / 2.0
                deviation = abs(count_a - count_b) / mean_count

                if deviation > threshold:
                    summary.record_fail(
                        ConsistencyFailure(
                            character_id=character_id,
                            pose=pose,
                            region=region_name,
                            check="pixel_area_consistency",
                            angle_a=angle_a,
                            angle_b=angle_b,
                            detail=(
                                f"pixel count deviation {deviation:.1%} "
                                f"({angle_a}={count_a}, {angle_b}={count_b}) "
                                f"exceeds threshold {threshold:.0%}"
                            ),
                        )
                    )
                else:
                    summary.record_pass()


def check_measurement_ratio(
    angle_measurements: dict[str, dict[str, Any]],
    ground_truth: dict[str, Any],
    character_id: str,
    pose: str,
    summary: ConsistencyCheckSummary,
    *,
    threshold: float = DEFAULT_DEVIATION_THRESHOLD,
) -> None:
    """Check 2D apparent measurements against 3D ground truth.

    For regions visible from both front and side views, validates that::

        apparent_width_front / apparent_width_side ≈ true_width / true_depth

    Args:
        angle_measurements: ``{angle_name -> measurement_data}`` for one pose.
        ground_truth: 3D ground truth measurements with a ``"regions"`` key.
        character_id: Character identifier.
        pose: Pose identifier.
        summary: Check summary to record pass/fail.
        threshold: Maximum allowed relative deviation (default 10%).
    """
    front_data = angle_measurements.get("front")
    side_data = angle_measurements.get("side")
    if front_data is None or side_data is None:
        return

    gt_regions = ground_truth.get("regions", {})
    front_regions = front_data.get("regions", {})
    side_regions = side_data.get("regions", {})

    for region_name in front_regions:
        front_r = front_regions[region_name]
        side_r = side_regions.get(region_name, {})

        if not front_r.get("visible", False) or not side_r.get("visible", False):
            continue

        front_width = front_r.get("apparent_width", 0)
        side_width = side_r.get("apparent_width", 0)

        if front_width < 1 or side_width < 1:
            continue

        gt = gt_regions.get(region_name)
        if gt is None:
            continue

        true_width = gt.get("width", 0.0)
        true_depth = gt.get("depth", 0.0)

        if true_width <= 0 or true_depth <= 0:
            continue

        # Ratio of front-to-side apparent width vs. true width-to-depth
        apparent_ratio = front_width / side_width
        true_ratio = true_width / true_depth
        deviation = abs(apparent_ratio - true_ratio) / true_ratio

        if deviation > threshold:
            summary.record_fail(
                ConsistencyFailure(
                    character_id=character_id,
                    pose=pose,
                    region=region_name,
                    check="measurement_ratio",
                    angle_a="front",
                    angle_b="side",
                    detail=(
                        f"apparent ratio {apparent_ratio:.2f} vs true ratio "
                        f"{true_ratio:.2f} (deviation {deviation:.1%}, "
                        f"threshold {threshold:.0%})"
                    ),
                )
            )
        else:
            summary.record_pass()


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------


def validate_multiview_consistency(
    output_dir: Path,
    *,
    characters: list[str] | None = None,
    threshold: float = DEFAULT_DEVIATION_THRESHOLD,
) -> ConsistencyReport:
    """Run multi-view consistency checks on the dataset.

    Reads 2D measurement files from ``output_dir/measurements_2d/`` and
    optionally 3D ground truth from ``output_dir/measurements/``.

    Args:
        output_dir: Root dataset directory.
        characters: Optional character IDs to validate (None = all).
        threshold: Maximum allowed relative deviation (default 10%).

    Returns:
        ConsistencyReport with per-check summaries.
    """
    t_start = time.monotonic()

    report = ConsistencyReport()
    report.checks = {
        "region_presence": ConsistencyCheckSummary(name="region_presence"),
        "pixel_area_consistency": ConsistencyCheckSummary(name="pixel_area_consistency"),
        "measurement_ratio": ConsistencyCheckSummary(name="measurement_ratio"),
    }

    measurements_2d_dir = output_dir / "measurements_2d"
    measurements_dir = output_dir / "measurements"

    groups = _discover_measurement_groups(measurements_2d_dir, characters)

    if not groups:
        logger.warning("No multi-angle measurement files found in %s", measurements_2d_dir)
        report.elapsed_seconds = time.monotonic() - t_start
        return report

    # Filter to groups with multiple angles (single-angle groups can't be validated)
    multi_angle_groups = {k: v for k, v in groups.items() if len(v) >= 2}

    if not multi_angle_groups:
        logger.info("No multi-angle pose groups found; skipping consistency checks")
        report.elapsed_seconds = time.monotonic() - t_start
        return report

    character_ids = set()
    for _group_key, angle_data in multi_angle_groups.items():
        # Extract character_id from the first measurement in the group
        first_data = next(iter(angle_data.values()))
        char_id = first_data.get("character_id", "")
        pose = first_data.get("pose", "")
        character_ids.add(char_id)

        check_region_presence(angle_data, char_id, pose, report.checks["region_presence"])

        check_pixel_area_consistency(
            angle_data,
            char_id,
            pose,
            report.checks["pixel_area_consistency"],
            threshold=threshold,
        )

        # Load ground truth for measurement ratio check
        gt = _load_ground_truth(measurements_dir, char_id)
        if gt is not None:
            check_measurement_ratio(
                angle_data,
                gt,
                char_id,
                pose,
                report.checks["measurement_ratio"],
                threshold=threshold,
            )

    report.characters_checked = len(character_ids)
    report.pose_groups_checked = len(multi_angle_groups)
    report.elapsed_seconds = time.monotonic() - t_start

    logger.info(
        "Multi-view consistency: %d characters, %d pose groups, %d failures",
        report.characters_checked,
        report.pose_groups_checked,
        report.total_failures,
    )

    return report


def _load_ground_truth(measurements_dir: Path, char_id: str) -> dict[str, Any] | None:
    """Load 3D ground truth measurements for a character.

    Args:
        measurements_dir: Directory containing per-character measurement files.
        char_id: Character identifier.

    Returns:
        Parsed measurement dict, or None if unavailable.
    """
    path = measurements_dir / f"{char_id}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read ground truth %s", path)
        return None
    if not isinstance(data, dict) or "regions" not in data:
        return None
    return data


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def print_consistency_report(report: ConsistencyReport) -> None:
    """Print a human-readable multi-view consistency report to stdout.

    Args:
        report: The consistency report to print.
    """
    print()
    print("=" * 60)
    print("MULTI-VIEW CONSISTENCY REPORT")
    print("=" * 60)
    print()

    status = "PASSED" if report.all_passed else "FAILED"
    print(f"Overall:    {status} ({report.total_failures} failure(s))")
    print(f"Characters: {report.characters_checked}")
    print(f"Pose groups: {report.pose_groups_checked}")
    print(f"Time:       {report.elapsed_seconds:.2f}s")
    print()

    print(f"{'Check':<30} {'Pass':>6} {'Fail':>6}  Status")
    print("-" * 55)

    for name, summary in report.checks.items():
        check_status = "PASS" if summary.failed == 0 else "FAIL"
        print(f"{name:<30} {summary.passed:>6} {summary.failed:>6}  {check_status}")

    # Worst regions
    worst = report.worst_regions()
    if worst:
        print()
        print("Worst regions:")
        for region, count in worst:
            print(f"  {region}: {count} failure(s)")

    # Worst angle pairs
    worst_pairs = report.worst_angle_pairs()
    if worst_pairs:
        print()
        print("Worst angle pairs:")
        for pair, count in worst_pairs:
            print(f"  {pair}: {count} failure(s)")

    # Failure details
    has_failures = any(s.failed > 0 for s in report.checks.values())
    if has_failures:
        print()
        print("FAILURES:")
        print("-" * 55)
        for name, summary in report.checks.items():
            if summary.failed == 0:
                continue
            print(f"\n  [{name}] ({summary.failed} failure(s))")
            for f in summary.failures[:20]:
                print(
                    f"    - {f.character_id}/{f.pose} {f.region} "
                    f"({f.angle_a} vs {f.angle_b}): {f.detail}"
                )
            if len(summary.failures) > 20:
                print(f"    ... and {len(summary.failures) - 20} more")

    print()
    print("=" * 60)


def save_consistency_report(report: ConsistencyReport, output_path: Path) -> Path:
    """Save the consistency report as JSON.

    Args:
        report: The consistency report to save.
        output_path: Path to write the JSON file.

    Returns:
        The output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Consistency report saved to %s", output_path)
    return output_path

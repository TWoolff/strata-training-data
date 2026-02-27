"""Automated dataset validation checks for post-generation integrity.

Verifies mask completeness, mask uniqueness, joint bounds, joint count,
file pairing, resolution, and region distribution.  Pure Python — no Blender
dependency — runs standalone after batch generation.

See PRD §11.1 (Automated Checks) and Issue #23.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import NUM_JOINT_REGIONS, NUM_REGIONS, REGION_NAMES, RENDER_RESOLUTION

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_REGION_FRACTION: float = 0.60  # No single region >60% of foreground pixels

# Expected joint names (regions 1–19)
EXPECTED_JOINT_NAMES: set[str] = {
    REGION_NAMES[rid] for rid in range(1, NUM_JOINT_REGIONS + 1)
}

# Regex to parse image filename stems: {char_id}_pose_{nn}_{style}
_IMAGE_STEM_PATTERN = re.compile(r"^(.+)_pose_(\d{2}.*)_([^_]+)$")

# Regex to parse mask filename stems: {char_id}_pose_{nn}
_MASK_STEM_PATTERN = re.compile(r"^(.+)_pose_(\d{2}.*)$")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class CheckFailure:
    """A single validation failure."""

    file: str
    check: str
    detail: str


@dataclass
class CheckSummary:
    """Summary for a single check type."""

    name: str
    passed: int = 0
    failed: int = 0
    failures: list[CheckFailure] = field(default_factory=list)

    def record_pass(self) -> None:
        self.passed += 1

    def record_fail(self, file: str, detail: str) -> None:
        self.failed += 1
        self.failures.append(CheckFailure(file=file, check=self.name, detail=detail))


@dataclass
class ValidationReport:
    """Full validation report across all checks."""

    checks: dict[str, CheckSummary] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    @property
    def all_passed(self) -> bool:
        return all(s.failed == 0 for s in self.checks.values())

    @property
    def total_failures(self) -> int:
        return sum(s.failed for s in self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "passed": self.all_passed,
            "total_failures": self.total_failures,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "checks": {
                name: {
                    "passed": s.passed,
                    "failed": s.failed,
                    "failures": [
                        {"file": f.file, "detail": f.detail} for f in s.failures
                    ],
                }
                for name, s in self.checks.items()
            },
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_resolution(
    path: Path,
    expected: int,
) -> tuple[bool, str]:
    """Check that an image is exactly expected x expected pixels.

    Args:
        path: Path to a PNG file.
        expected: Expected width and height in pixels.

    Returns:
        (passed, detail) tuple.
    """
    img = Image.open(path)
    w, h = img.size
    if w != expected or h != expected:
        return False, f"expected {expected}x{expected}, got {w}x{h}"
    return True, ""


def check_mask_completeness(
    image_path: Path,
    mask_path: Path,
) -> tuple[bool, str]:
    """Check that every non-transparent image pixel has a non-zero mask region.

    Args:
        image_path: Path to an RGBA color image.
        mask_path: Path to the corresponding 8-bit grayscale mask.

    Returns:
        (passed, detail) tuple.
    """
    img = np.array(Image.open(image_path).convert("RGBA"))
    mask = np.array(Image.open(mask_path).convert("L"))

    # Pixels where alpha > 0 (non-transparent)
    opaque = img[:, :, 3] > 0

    # Of those, how many have mask == 0 (background)?
    unmapped = opaque & (mask == 0)
    count = int(np.sum(unmapped))

    if count > 0:
        total_opaque = int(np.sum(opaque))
        pct = (count / total_opaque * 100) if total_opaque > 0 else 0
        return False, f"{count} non-transparent pixels with mask=0 ({pct:.1f}%)"
    return True, ""


def check_mask_uniqueness(
    mask_path: Path,
) -> tuple[bool, str]:
    """Check that a mask contains more than one region (not all-one-region).

    A mask that is entirely one region (excluding background) indicates
    a failed bone mapping.

    Args:
        mask_path: Path to an 8-bit grayscale mask.

    Returns:
        (passed, detail) tuple.
    """
    mask = np.array(Image.open(mask_path).convert("L"))

    # Get unique non-background values
    unique = np.unique(mask)
    non_bg = unique[unique > 0]

    if len(non_bg) == 0:
        # All background — might be valid for fully transparent characters
        return True, ""

    if len(non_bg) < 2:
        region_name = REGION_NAMES.get(int(non_bg[0]), f"region_{non_bg[0]}")
        return False, f"mask contains only one region: {region_name} (ID {non_bg[0]})"
    return True, ""


def check_joint_count(
    joint_data: dict[str, Any],
) -> tuple[bool, str]:
    """Check that a joint file has exactly NUM_JOINT_REGIONS joints.

    Args:
        joint_data: Parsed joint JSON data.

    Returns:
        (passed, detail) tuple.
    """
    joints = joint_data.get("joints", {})
    count = len(joints)

    if count != NUM_JOINT_REGIONS:
        return False, f"expected {NUM_JOINT_REGIONS} joints, got {count}"

    # Check joint names match expected set
    actual_names = set(joints.keys())
    missing = EXPECTED_JOINT_NAMES - actual_names
    extra = actual_names - EXPECTED_JOINT_NAMES

    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing: {sorted(missing)}")
        if extra:
            parts.append(f"unexpected: {sorted(extra)}")
        return False, "; ".join(parts)

    return True, ""


def check_joint_bounds(
    joint_data: dict[str, Any],
) -> tuple[bool, str]:
    """Check that all visible joint positions are within image bounds.

    Args:
        joint_data: Parsed joint JSON data.

    Returns:
        (passed, detail) tuple.
    """
    joints = joint_data.get("joints", {})
    image_size = joint_data.get("image_size", [RENDER_RESOLUTION, RENDER_RESOLUTION])
    max_x, max_y = image_size[0], image_size[1]

    out_of_bounds: list[str] = []
    for name, info in joints.items():
        if not info.get("visible", False):
            continue

        pos = info.get("position", [-1, -1])
        x, y = pos[0], pos[1]

        if x < 0 or x >= max_x or y < 0 or y >= max_y:
            out_of_bounds.append(f"{name}=({x},{y})")

    if out_of_bounds:
        return False, f"visible joints out of bounds [{max_x}x{max_y}]: {', '.join(out_of_bounds)}"
    return True, ""


def check_region_distribution(
    mask_path: Path,
) -> tuple[bool, str]:
    """Check that no single region dominates >60% of foreground pixels.

    Args:
        mask_path: Path to an 8-bit grayscale mask.

    Returns:
        (passed, detail) tuple.
    """
    mask = np.array(Image.open(mask_path).convert("L"))

    counts = np.bincount(mask.ravel(), minlength=NUM_REGIONS)
    foreground_total = int(counts[1:].sum())

    if foreground_total == 0:
        return True, ""

    for region_id in range(1, NUM_REGIONS):
        fraction = counts[region_id] / foreground_total
        if fraction > MAX_REGION_FRACTION:
            region_name = REGION_NAMES.get(region_id, f"region_{region_id}")
            return (
                False,
                f"region {region_name} (ID {region_id}) = {fraction:.1%} of foreground (>{MAX_REGION_FRACTION:.0%})",
            )

    return True, ""


# ---------------------------------------------------------------------------
# File pairing discovery
# ---------------------------------------------------------------------------


def _extract_pose_key(image_stem: str) -> str | None:
    """Extract '{char_id}_pose_{nn}' from an image filename stem.

    E.g. 'mixamo_001_pose_00_flat' → 'mixamo_001_pose_00'
    """
    match = _IMAGE_STEM_PATTERN.match(image_stem)
    if match:
        return f"{match.group(1)}_pose_{match.group(2)}"
    return None


def _discover_files(
    output_dir: Path,
    characters: list[str] | None = None,
) -> tuple[list[Path], dict[str, list[Path]]]:
    """Discover image files and group by pose key.

    Args:
        output_dir: Root dataset directory.
        characters: Optional list of character IDs to filter.

    Returns:
        (all_image_paths, pose_key_to_image_paths) tuple.
    """
    images_dir = output_dir / "images"
    if not images_dir.is_dir():
        return [], {}

    image_paths = sorted(images_dir.glob("*.png"))

    if characters:
        char_set = set(characters)
        image_paths = [
            p for p in image_paths
            if (m := _IMAGE_STEM_PATTERN.match(p.stem)) and m.group(1) in char_set
        ]

    # Group by pose key
    pose_groups: dict[str, list[Path]] = {}
    for img_path in image_paths:
        key = _extract_pose_key(img_path.stem)
        if key:
            pose_groups.setdefault(key, []).append(img_path)

    return image_paths, pose_groups


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------


def validate_dataset(
    output_dir: Path,
    *,
    characters: list[str] | None = None,
    resolution: int = RENDER_RESOLUTION,
) -> ValidationReport:
    """Run all 7 validation checks on the dataset.

    Args:
        output_dir: Root dataset directory.
        characters: Optional list of character IDs to validate (None = all).
        resolution: Expected image resolution (square).

    Returns:
        ValidationReport with per-check summaries.
    """
    t_start = time.monotonic()

    report = ValidationReport()
    report.checks = {
        "resolution": CheckSummary(name="resolution"),
        "mask_completeness": CheckSummary(name="mask_completeness"),
        "mask_uniqueness": CheckSummary(name="mask_uniqueness"),
        "joint_count": CheckSummary(name="joint_count"),
        "joint_bounds": CheckSummary(name="joint_bounds"),
        "file_pairing": CheckSummary(name="file_pairing"),
        "region_distribution": CheckSummary(name="region_distribution"),
    }

    masks_dir = output_dir / "masks"
    joints_dir = output_dir / "joints"

    image_paths, pose_groups = _discover_files(output_dir, characters)

    if not image_paths:
        logger.warning("No image files found in %s/images/", output_dir)
        report.elapsed_seconds = time.monotonic() - t_start
        return report

    total_images = len(image_paths)
    logger.info("Validating %d images across %d poses...", total_images, len(pose_groups))

    # --- Check 1: Resolution (all images) ---
    for img_path in image_paths:
        try:
            passed, detail = check_resolution(img_path, resolution)
            if passed:
                report.checks["resolution"].record_pass()
            else:
                report.checks["resolution"].record_fail(img_path.name, detail)
        except Exception as exc:
            report.checks["resolution"].record_fail(img_path.name, f"error: {exc}")

    # --- Check 2: Resolution (all masks) ---
    mask_paths = sorted(masks_dir.glob("*.png")) if masks_dir.is_dir() else []
    if characters:
        char_set = set(characters)
        mask_paths = [
            p for p in mask_paths
            if (m := _MASK_STEM_PATTERN.match(p.stem)) and m.group(1) in char_set
        ]
    for mask_path in mask_paths:
        try:
            passed, detail = check_resolution(mask_path, resolution)
            if passed:
                report.checks["resolution"].record_pass()
            else:
                report.checks["resolution"].record_fail(mask_path.name, detail)
        except Exception as exc:
            report.checks["resolution"].record_fail(mask_path.name, f"error: {exc}")

    # --- Per-pose checks (file pairing, mask, joints) ---
    for pose_key, pose_images in pose_groups.items():
        mask_path = masks_dir / f"{pose_key}.png"
        joint_path = joints_dir / f"{pose_key}.json"

        # Check 3: File pairing — mask exists
        if not mask_path.exists():
            for img in pose_images:
                report.checks["file_pairing"].record_fail(
                    img.name, f"missing mask: {mask_path.name}"
                )
        else:
            for _img in pose_images:
                report.checks["file_pairing"].record_pass()

        # Check 4: File pairing — joints exist
        if not joint_path.exists():
            report.checks["file_pairing"].record_fail(
                pose_key, f"missing joints: {joint_path.name}"
            )
        else:
            report.checks["file_pairing"].record_pass()

        # Check 5: Mask uniqueness
        if mask_path.exists():
            try:
                passed, detail = check_mask_uniqueness(mask_path)
                if passed:
                    report.checks["mask_uniqueness"].record_pass()
                else:
                    report.checks["mask_uniqueness"].record_fail(mask_path.name, detail)
            except Exception as exc:
                report.checks["mask_uniqueness"].record_fail(mask_path.name, f"error: {exc}")

        # Check 6: Region distribution
        if mask_path.exists():
            try:
                passed, detail = check_region_distribution(mask_path)
                if passed:
                    report.checks["region_distribution"].record_pass()
                else:
                    report.checks["region_distribution"].record_fail(mask_path.name, detail)
            except Exception as exc:
                report.checks["region_distribution"].record_fail(mask_path.name, f"error: {exc}")

        # Check 7: Mask completeness (check first image per pose against the mask)
        if mask_path.exists() and pose_images:
            try:
                passed, detail = check_mask_completeness(pose_images[0], mask_path)
                if passed:
                    report.checks["mask_completeness"].record_pass()
                else:
                    report.checks["mask_completeness"].record_fail(
                        pose_images[0].name, detail
                    )
            except Exception as exc:
                report.checks["mask_completeness"].record_fail(
                    pose_images[0].name, f"error: {exc}"
                )

        # Check 8 & 9: Joint count + bounds
        if joint_path.exists():
            try:
                joint_data = json.loads(joint_path.read_text(encoding="utf-8"))

                passed, detail = check_joint_count(joint_data)
                if passed:
                    report.checks["joint_count"].record_pass()
                else:
                    report.checks["joint_count"].record_fail(joint_path.name, detail)

                passed, detail = check_joint_bounds(joint_data)
                if passed:
                    report.checks["joint_bounds"].record_pass()
                else:
                    report.checks["joint_bounds"].record_fail(joint_path.name, detail)
            except json.JSONDecodeError as exc:
                report.checks["joint_count"].record_fail(
                    joint_path.name, f"invalid JSON: {exc}"
                )
                report.checks["joint_bounds"].record_fail(
                    joint_path.name, f"invalid JSON: {exc}"
                )

    report.elapsed_seconds = time.monotonic() - t_start
    return report


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def print_report(report: ValidationReport) -> None:
    """Print a human-readable validation report to stdout.

    Args:
        report: The validation report to print.
    """
    print()
    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    print()

    all_passed = report.all_passed
    status = "PASSED" if all_passed else "FAILED"
    print(f"Overall: {status} ({report.total_failures} failure(s))")
    print(f"Time:    {report.elapsed_seconds:.2f}s")
    print()

    print(f"{'Check':<25} {'Pass':>6} {'Fail':>6}  Status")
    print("-" * 50)

    for name, summary in report.checks.items():
        check_status = "PASS" if summary.failed == 0 else "FAIL"
        print(f"{name:<25} {summary.passed:>6} {summary.failed:>6}  {check_status}")

    # Print failure details
    has_failures = any(s.failed > 0 for s in report.checks.values())
    if has_failures:
        print()
        print("FAILURES:")
        print("-" * 50)
        for name, summary in report.checks.items():
            if summary.failed == 0:
                continue
            print(f"\n  [{name}] ({summary.failed} failure(s))")
            for failure in summary.failures[:20]:
                print(f"    - {failure.file}: {failure.detail}")
            if len(summary.failures) > 20:
                print(f"    ... and {len(summary.failures) - 20} more")

    print()
    print("=" * 60)


def save_report(report: ValidationReport, output_path: Path) -> Path:
    """Save the validation report as JSON.

    Args:
        report: The validation report to save.
        output_path: Path to write the JSON file.

    Returns:
        The output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Validation report saved to %s", output_path)
    return output_path

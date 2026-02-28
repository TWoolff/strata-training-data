"""Verify integrity of downloaded pre-processed datasets.

Checks ``data/preprocessed/`` for expected directory structures, file counts,
and file formats.  Reports status per-dataset and exits with non-zero code
if any dataset fails verification.

Pure Python (no Blender dependency).

Usage::

    python -m scripts.verify_downloads [--data-dir ./data/preprocessed]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Expected structure for a single preprocessed dataset."""

    name: str
    required_patterns: list[str] = field(default_factory=list)
    optional_patterns: list[str] = field(default_factory=list)
    min_files: int = 0
    expected_formats: list[str] = field(default_factory=list)
    expected_resolution: tuple[int, int] | None = None


DATASET_SPECS: dict[str, DatasetSpec] = {
    s.name: s
    for s in [
        DatasetSpec(
            name="nova_human",
            required_patterns=["**/ortho/*.png"],
            optional_patterns=["**/rgb/*.png", "**/*_meta.json"],
            min_files=10,
            expected_formats=[".png"],
        ),
        DatasetSpec(name="stdgen", optional_patterns=["**/*.png", "**/*.json"]),
        DatasetSpec(name="animerun", optional_patterns=["**/contour/*.png", "**/anime/*.png"]),
        DatasetSpec(name="unirig", optional_patterns=["**/*.npz", "**/*.json", "**/*.glb"]),
        DatasetSpec(name="linkto_anime", optional_patterns=["**/*.png", "**/*.flo", "**/*.json"]),
        DatasetSpec(name="fbanimehq", optional_patterns=["**/*.png", "**/*.jpg"]),
        DatasetSpec(name="anime_segmentation", optional_patterns=["**/*.png", "**/*.jpg"]),
        DatasetSpec(name="anime_instance_seg", optional_patterns=["**/*.png", "**/*.jpg"]),
        DatasetSpec(name="charactergen", optional_patterns=["**/*.vrm", "**/*.png"]),
    ]
}


# ---------------------------------------------------------------------------
# Verification results
# ---------------------------------------------------------------------------


@dataclass
class DatasetCheckResult:
    """Result of verifying a single dataset."""

    name: str
    exists: bool = False
    file_count: int = 0
    missing_required: list[str] = field(default_factory=list)
    format_errors: list[str] = field(default_factory=list)
    resolution_errors: list[str] = field(default_factory=list)
    passed: bool = False

    @property
    def errors(self) -> list[str]:
        """All error messages for this dataset."""
        msgs: list[str] = []
        if not self.exists:
            msgs.append("directory does not exist or is empty")
        msgs.extend(self.missing_required)
        msgs.extend(self.format_errors)
        msgs.extend(self.resolution_errors)
        return msgs


@dataclass
class VerificationReport:
    """Summary of verification across all datasets."""

    results: list[DatasetCheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def total_datasets(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)


# ---------------------------------------------------------------------------
# Verification logic
# ---------------------------------------------------------------------------


def verify_dataset(data_dir: Path, spec: DatasetSpec) -> DatasetCheckResult:
    """Verify a single dataset against its spec.

    Args:
        data_dir: Root preprocessed data directory (e.g. ``data/preprocessed/``).
        spec: Expected structure for this dataset.

    Returns:
        Check result for the dataset.
    """
    result = DatasetCheckResult(name=spec.name)
    dataset_dir = data_dir / spec.name

    if not dataset_dir.is_dir():
        return result

    result.exists = True

    # Count all files (excluding hidden/dotfiles and READMEs)
    all_files = [
        f
        for f in dataset_dir.rglob("*")
        if f.is_file() and not f.name.startswith(".") and f.name.lower() != "readme.md"
    ]
    result.file_count = len(all_files)

    # Check required file patterns
    for pattern in spec.required_patterns:
        matches = list(dataset_dir.glob(pattern))
        if not matches:
            result.missing_required.append(f"no files matching '{pattern}'")

    # Check minimum file count
    if spec.min_files > 0 and result.file_count < spec.min_files:
        result.missing_required.append(
            f"expected at least {spec.min_files} files, found {result.file_count}"
        )

    # Validate file formats (sample up to 10 images)
    if spec.expected_formats:
        _check_formats(dataset_dir, spec, result)

    # Validate resolution if specified
    if spec.expected_resolution:
        _check_resolution(dataset_dir, spec.expected_resolution, result)

    result.passed = not result.errors
    return result


def _check_formats(dataset_dir: Path, spec: DatasetSpec, result: DatasetCheckResult) -> None:
    """Spot-check that image files are valid."""
    image_exts = {".png", ".jpg", ".jpeg"}
    expected = set(spec.expected_formats) & image_exts
    if not expected:
        return

    sample_count = 0
    for ext in expected:
        for img_path in dataset_dir.rglob(f"*{ext}"):
            if sample_count >= 10:
                return
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as exc:
                result.format_errors.append(f"corrupt image {img_path.name}: {exc}")
            sample_count += 1


def _check_resolution(
    dataset_dir: Path,
    expected: tuple[int, int],
    result: DatasetCheckResult,
) -> None:
    """Spot-check that images match expected resolution."""
    for i, img_path in enumerate(dataset_dir.rglob("*.png")):
        if i >= 5:
            return
        try:
            with Image.open(img_path) as img:
                if img.size != expected:
                    result.resolution_errors.append(f"{img_path.name}: {img.size} != {expected}")
        except Exception:
            pass  # format errors caught elsewhere


def verify_all(data_dir: Path) -> VerificationReport:
    """Verify all known datasets in the preprocessed data directory.

    Args:
        data_dir: Root preprocessed directory (e.g. ``data/preprocessed/``).

    Returns:
        Verification report covering all datasets.
    """
    report = VerificationReport()

    for spec in sorted(DATASET_SPECS.values(), key=lambda s: s.name):
        result = verify_dataset(data_dir, spec)
        report.results.append(result)
        level = "PASS" if result.passed else "FAIL"
        logger.info(
            "[%s] %s — %d files%s",
            level,
            spec.name,
            result.file_count,
            f" — {'; '.join(result.errors)}" if result.errors else "",
        )

    return report


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_report(report: VerificationReport) -> None:
    """Print a human-readable verification summary to stdout."""
    print(f"\n{'Dataset':<25} {'Status':<8} {'Files':>8}  Errors")
    print("-" * 70)

    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        errors = "; ".join(r.errors) if r.errors else ""
        print(f"{r.name:<25} {status:<8} {r.file_count:>8}  {errors}")

    print("-" * 70)
    print(
        f"Total: {report.total_datasets} datasets, "
        f"{report.passed_count} passed, {report.failed_count} failed"
    )


def save_report(report: VerificationReport, path: Path) -> Path:
    """Save verification report as JSON.

    Args:
        report: Verification report to save.
        path: Output file path.

    Returns:
        Path to the written file.
    """
    data = {
        "total_datasets": report.total_datasets,
        "passed": report.passed_count,
        "failed": report.failed_count,
        "datasets": [
            {
                "name": r.name,
                "exists": r.exists,
                "file_count": r.file_count,
                "passed": r.passed,
                "errors": r.errors,
            }
            for r in report.results
        ],
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 if any dataset fails."""
    parser = argparse.ArgumentParser(
        description="Verify integrity of downloaded pre-processed datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/preprocessed"),
        help="Root directory for preprocessed datasets (default: data/preprocessed)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Save report as JSON to this path",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report = verify_all(args.data_dir)
    print_report(report)

    if args.json:
        save_report(report, args.json)
        print(f"\nJSON report saved to {args.json}")

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

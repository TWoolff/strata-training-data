"""Batch-retarget 100STYLE BVH dataset to Strata 19-bone skeleton.

Processes all 100 locomotion styles × 8–10 content types (1,620 BVH files),
retargets to Strata's 19-bone skeleton, trims to Frame_Cuts.csv boundaries,
and exports blueprint JSON files.

Also generates ``animation/labels/100style_labels.csv`` with style + content
taxonomy per sequence.

No Blender dependency — pure Python.

Usage:
    python -m animation.scripts.retarget_100style \
        --input_dir /Volumes/TAMWoolff/data/preprocessed/100style/100STYLE \
        --output_dir output/animation/100style
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from animation.scripts.blueprint_exporter import export_blueprint
from animation.scripts.bvh_parser import parse_bvh
from animation.scripts.bvh_to_strata import RetargetedAnimation, retarget

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Content code → human-readable label
# ---------------------------------------------------------------------------

CONTENT_CODES: dict[str, str] = {
    "FW": "forward_walk",
    "BW": "backward_walk",
    "FR": "forward_run",
    "BR": "backward_run",
    "SW": "sideways_walk",
    "SR": "sideways_run",
    "ID": "idle",
    "TR1": "turn_1",
    "TR2": "turn_2",
    "TR3": "turn_3",
}

# Files to skip (not BVH motion data)
SKIP_FILES: set[str] = {"Dataset_List.csv", "Frame_Cuts.csv"}


# ---------------------------------------------------------------------------
# Frame cuts (trim padding from start/end of each clip)
# ---------------------------------------------------------------------------


@dataclass
class FrameCut:
    """Start/stop frame indices for a single content clip."""

    start: int
    stop: int


def load_frame_cuts(csv_path: Path) -> dict[str, dict[str, FrameCut]]:
    """Load Frame_Cuts.csv into {style_name: {content_code: FrameCut}}.

    Args:
        csv_path: Path to Frame_Cuts.csv.

    Returns:
        Nested dict mapping style name → content code → frame cut boundaries.
        Entries with N/A values are omitted.
    """
    cuts: dict[str, dict[str, FrameCut]] = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            style = row["STYLE_NAME"]
            style_cuts: dict[str, FrameCut] = {}
            for code in CONTENT_CODES:
                start_key = f"{code}_START"
                stop_key = f"{code}_STOP"
                start_val = row.get(start_key, "N/A")
                stop_val = row.get(stop_key, "N/A")
                if start_val != "N/A" and stop_val != "N/A":
                    style_cuts[code] = FrameCut(start=int(start_val), stop=int(stop_val))
            cuts[style] = style_cuts
    return cuts


def trim_animation(
    animation: RetargetedAnimation,
    cut: FrameCut,
) -> RetargetedAnimation:
    """Trim a retargeted animation to the given frame boundaries.

    Args:
        animation: Full retargeted animation.
        cut: Start/stop frame indices (inclusive start, exclusive stop).

    Returns:
        New RetargetedAnimation with only the trimmed frames.
    """
    start = max(0, cut.start)
    stop = min(cut.stop, animation.frame_count)
    if start >= stop:
        logger.warning(
            "Invalid frame cut [%d, %d) for %d frames", start, stop, animation.frame_count
        )
        return animation

    trimmed_frames = animation.frames[start:stop]
    return RetargetedAnimation(
        frames=trimmed_frames,
        frame_count=len(trimmed_frames),
        frame_rate=animation.frame_rate,
        source_bones=animation.source_bones,
        unmapped_bones=animation.unmapped_bones,
        rotation_order=animation.rotation_order,
    )


# ---------------------------------------------------------------------------
# Style metadata from Dataset_List.csv
# ---------------------------------------------------------------------------


@dataclass
class StyleMeta:
    """Metadata for a single locomotion style."""

    name: str
    description: str
    stochastic: bool
    symmetric: bool


def load_style_metadata(csv_path: Path) -> dict[str, StyleMeta]:
    """Load Dataset_List.csv into {style_name: StyleMeta}.

    Args:
        csv_path: Path to Dataset_List.csv.

    Returns:
        Dict mapping style name → metadata.
    """
    styles: dict[str, StyleMeta] = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Style Name"]
            styles[name] = StyleMeta(
                name=name,
                description=row.get("Description", ""),
                stochastic=row.get("Stochastic", "No") == "Yes",
                symmetric=row.get("Symmetric", "Yes") == "Yes",
            )
    return styles


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def _parse_filename(filename: str) -> tuple[str, str] | None:
    """Parse a 100STYLE BVH filename into (style_name, content_code).

    Expected format: ``{StyleName}_{ContentCode}.bvh``
    e.g. ``Angry_FW.bvh`` → ``("Angry", "FW")``

    Returns:
        (style_name, content_code) or None if parsing fails.
    """
    stem = Path(filename).stem
    # Split on last underscore (style names don't contain underscores,
    # but content codes are always the last segment)
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return None
    style_name, content_code = parts
    if content_code not in CONTENT_CODES:
        return None
    return style_name, content_code


@dataclass
class BatchResult:
    """Summary of batch retargeting results."""

    total_files: int
    success_count: int
    error_count: int
    skip_count: int
    total_frames_in: int
    total_frames_out: int
    errors: list[str]


def process_batch(
    input_dir: Path,
    output_dir: Path,
    labels_path: Path,
    *,
    trim: bool = True,
) -> BatchResult:
    """Batch-retarget all 100STYLE BVH files.

    Args:
        input_dir: Root 100STYLE directory containing style subdirectories.
        output_dir: Output directory for blueprint JSON files.
        labels_path: Output path for the labels CSV.
        trim: Whether to apply Frame_Cuts.csv trimming (default True).

    Returns:
        BatchResult with processing summary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load frame cuts for trimming
    frame_cuts: dict[str, dict[str, FrameCut]] = {}
    cuts_path = input_dir / "Frame_Cuts.csv"
    if trim and cuts_path.exists():
        frame_cuts = load_frame_cuts(cuts_path)
        logger.info("Loaded frame cuts for %d styles", len(frame_cuts))
    elif trim:
        logger.warning("Frame_Cuts.csv not found at %s — skipping trim", cuts_path)

    # Load style metadata
    style_meta: dict[str, StyleMeta] = {}
    meta_path = input_dir / "Dataset_List.csv"
    if meta_path.exists():
        style_meta = load_style_metadata(meta_path)
        logger.info("Loaded metadata for %d styles", len(style_meta))

    # Collect all BVH files
    bvh_files = sorted(input_dir.rglob("*.bvh"))
    logger.info("Found %d BVH files", len(bvh_files))

    result = BatchResult(
        total_files=len(bvh_files),
        success_count=0,
        error_count=0,
        skip_count=0,
        total_frames_in=0,
        total_frames_out=0,
        errors=[],
    )

    label_rows: list[dict[str, str]] = []
    t0 = time.monotonic()

    for i, bvh_path in enumerate(bvh_files, 1):
        # Skip macOS resource fork files
        if bvh_path.name.startswith("._"):
            result.skip_count += 1
            continue

        parsed = _parse_filename(bvh_path.name)
        if parsed is None:
            logger.debug("Skipping non-standard filename: %s", bvh_path.name)
            result.skip_count += 1
            continue

        style_name, content_code = parsed
        content_label = CONTENT_CODES[content_code]

        try:
            bvh = parse_bvh(bvh_path)
            animation = retarget(bvh)
            result.total_frames_in += animation.frame_count

            # Apply frame cut trimming
            if trim and style_name in frame_cuts and content_code in frame_cuts[style_name]:
                cut = frame_cuts[style_name][content_code]
                animation = trim_animation(animation, cut)

            result.total_frames_out += animation.frame_count

            # Export blueprint JSON
            out_subdir = output_dir / style_name
            out_file = out_subdir / f"{style_name}_{content_code}.json"
            export_blueprint(animation, out_file)

            # Build label row
            meta = style_meta.get(style_name)
            label_rows.append(
                {
                    "filename": f"{style_name}/{style_name}_{content_code}.json",
                    "style_name": style_name,
                    "content_code": content_code,
                    "content_type": content_label,
                    "description": meta.description if meta else "",
                    "stochastic": "yes" if meta and meta.stochastic else "no",
                    "symmetric": "yes" if meta and meta.symmetric else "no",
                    "frame_count": str(animation.frame_count),
                    "frame_rate": str(animation.frame_rate),
                    "strata_compatible": "yes",
                }
            )

            result.success_count += 1

            if i % 100 == 0 or i == len(bvh_files):
                elapsed = time.monotonic() - t0
                rate = i / elapsed if elapsed > 0 else 0
                logger.info(
                    "Progress: %d/%d files (%.0f files/sec)",
                    i,
                    len(bvh_files),
                    rate,
                )

        except Exception as exc:
            error_msg = f"{bvh_path.name}: {exc}"
            logger.error("Failed to process %s: %s", bvh_path.name, exc)
            result.errors.append(error_msg)
            result.error_count += 1

    # Write labels CSV
    if label_rows:
        _write_labels_csv(labels_path, label_rows)
        logger.info("Wrote %d label rows to %s", len(label_rows), labels_path)

    elapsed = time.monotonic() - t0
    logger.info(
        "Batch complete: %d success, %d errors, %d skipped in %.1fs",
        result.success_count,
        result.error_count,
        result.skip_count,
        elapsed,
    )
    logger.info(
        "Frames: %d in → %d out (%.1f%% retained after trimming)",
        result.total_frames_in,
        result.total_frames_out,
        (result.total_frames_out / result.total_frames_in * 100)
        if result.total_frames_in > 0
        else 0,
    )

    return result


def _write_labels_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the 100style_labels.csv file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "style_name",
        "content_code",
        "content_type",
        "description",
        "stochastic",
        "symmetric",
        "frame_count",
        "frame_rate",
        "strata_compatible",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["filename"]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for 100STYLE batch retargeting."""
    parser = argparse.ArgumentParser(
        description="Batch-retarget 100STYLE BVH dataset to Strata 19-bone skeleton",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Root 100STYLE directory containing style subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/animation/100style"),
        help="Output directory for blueprint JSON files (default: output/animation/100style)",
    )
    parser.add_argument(
        "--labels_path",
        type=Path,
        default=Path("animation/labels/100style_labels.csv"),
        help="Output path for labels CSV (default: animation/labels/100style_labels.csv)",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Skip Frame_Cuts.csv trimming (export full clips)",
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
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    result = process_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        labels_path=args.labels_path,
        trim=not args.no_trim,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("100STYLE Retargeting Complete")
    print(f"{'=' * 60}")
    print(f"  Success:  {result.success_count}")
    print(f"  Errors:   {result.error_count}")
    print(f"  Skipped:  {result.skip_count}")
    print(f"  Frames:   {result.total_frames_in:,} → {result.total_frames_out:,}")
    if result.errors:
        print("\nErrors:")
        for err in result.errors[:20]:
            print(f"  - {err}")
        if len(result.errors) > 20:
            print(f"  ... and {len(result.errors) - 20} more")

    sys.exit(1 if result.error_count > 0 else 0)


if __name__ == "__main__":
    main()

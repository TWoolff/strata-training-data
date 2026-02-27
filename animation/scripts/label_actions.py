"""Interactive CLI for reviewing and tagging BVH mocap clip action labels.

Loads animation/labels/cmu_action_labels.csv and presents clips for review.
Supports filtering by label status, action type, and batch skipping of
already-labeled clips. Saves progress incrementally after each update.

No Blender dependency — pure Python.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABELS_CSV = Path("animation/labels/cmu_action_labels.csv")
BVH_DIR = Path("data/mocap")

CSV_FIELDS = ["filename", "action_type", "subcategory", "quality", "strata_compatible"]

QUALITY_OPTIONS = ["high", "medium", "low"]
COMPAT_OPTIONS = ["yes", "no"]

# (CSV column, display label, constrained options or None for free-text)
EDITABLE_FIELDS: list[tuple[str, str, list[str] | None]] = [
    ("action_type", "Action type", None),
    ("subcategory", "Subcategory", None),
    ("quality", "Quality", QUALITY_OPTIONS),
    ("strata_compatible", "Strata compatible", COMPAT_OPTIONS),
]

# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


def load_csv(path: Path) -> list[dict[str, str]]:
    """Load the action labels CSV into a list of row dicts."""
    if not path.is_file():
        logger.error("CSV not found: %s", path)
        sys.exit(1)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write rows back to the CSV, preserving field order."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# BVH metadata (optional — files may not be present locally)
# ---------------------------------------------------------------------------


def get_bvh_metadata(filename: str) -> dict[str, str] | None:
    """Try to read BVH metadata for display. Returns None if file not found."""
    bvh_path = BVH_DIR / filename
    if not bvh_path.is_file():
        return None

    try:
        from animation.scripts.bvh_parser import parse_bvh

        bvh = parse_bvh(bvh_path)
        joint_count = len([j for j in bvh.skeleton.joints.values() if j.channels])
        duration = bvh.motion.frame_count * bvh.motion.frame_time
        return {
            "frames": str(bvh.motion.frame_count),
            "duration": f"{duration:.1f}s",
            "joints": str(joint_count),
            "fps": str(round(1.0 / bvh.motion.frame_time)) if bvh.motion.frame_time > 0 else "?",
        }
    except Exception as exc:
        logger.debug("Could not parse BVH %s: %s", filename, exc)
        return None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def display_clip(row: dict[str, str], index: int, total: int) -> None:
    """Print clip info for review."""
    print(f"\n{'=' * 60}")
    print(f"  Clip {index + 1} of {total}: {row['filename']}")
    print(f"{'=' * 60}")

    # Show BVH metadata if available
    meta = get_bvh_metadata(row["filename"])
    if meta:
        print(f"  BVH: {meta['frames']} frames, {meta['duration']}, "
              f"{meta['joints']} joints, {meta['fps']} fps")

    # Show current labels
    for col, label, _ in EDITABLE_FIELDS:
        value = row.get(col, "") or ("(unlabeled)" if col == "action_type" else "(none)")
        print(f"  {label + ':':<21}{value}")


def prompt_field(field_name: str, current: str, options: list[str] | None = None) -> str:
    """Prompt user for a field value. Enter to keep current."""
    hint = f" [{current}]" if current else ""
    if options:
        opts_str = ", ".join(options)
        raw = input(f"  {field_name}{hint} ({opts_str}): ").strip()
    else:
        raw = input(f"  {field_name}{hint}: ").strip()

    if not raw:
        return current
    return raw


# ---------------------------------------------------------------------------
# Main review loop
# ---------------------------------------------------------------------------


def review_clips(
    rows: list[dict[str, str]],
    csv_path: Path,
    *,
    unlabeled_only: bool = False,
    action_filter: str | None = None,
) -> int:
    """Interactive review loop. Returns number of clips updated."""
    # Filter clips to review
    to_review: list[tuple[int, dict[str, str]]] = []
    for i, row in enumerate(rows):
        if unlabeled_only and row.get("action_type", "").strip():
            continue
        if action_filter and row.get("action_type", "").strip() != action_filter:
            continue
        to_review.append((i, row))

    if not to_review:
        print("No clips match the filter criteria.")
        return 0

    print(f"\n{len(to_review)} clips to review.")
    print("Commands: Enter=keep current, 's'=skip, 'q'=quit\n")

    updated = 0

    for review_idx, (row_idx, row) in enumerate(to_review):
        display_clip(row, review_idx, len(to_review))
        print()

        # Check for skip/quit
        cmd = input("  Edit this clip? [y/s/q] (y): ").strip().lower()
        if cmd == "q":
            print(f"\nSaved. {updated} clips updated.")
            return updated
        if cmd == "s":
            continue

        # Prompt for each editable field
        changed = False
        for col, label, options in EDITABLE_FIELDS:
            current = row.get(col, "")
            new_val = prompt_field(label, current, options)
            if new_val != current:
                row[col] = new_val
                changed = True

        if changed:
            rows[row_idx] = row
            save_csv(csv_path, rows)
            updated += 1
            print("  -> Saved.")
        else:
            print("  -> No changes.")

    print(f"\nDone. {updated} clips updated out of {len(to_review)} reviewed.")
    return updated


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for label_actions."""
    parser = argparse.ArgumentParser(
        description="Interactive CLI for reviewing BVH mocap clip action labels.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=LABELS_CSV,
        help=f"Path to action labels CSV (default: {LABELS_CSV})",
    )
    parser.add_argument(
        "--unlabeled",
        action="store_true",
        help="Only show clips without an action_type label",
    )
    parser.add_argument(
        "--action-type",
        type=str,
        default=None,
        help="Only show clips with this action_type (e.g., 'walk', 'run')",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print label summary statistics and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    csv_path = args.csv
    rows = load_csv(csv_path)

    if args.summary:
        print_summary(rows)
        return

    try:
        review_clips(
            rows,
            csv_path,
            unlabeled_only=args.unlabeled,
            action_filter=args.action_type,
        )
    except KeyboardInterrupt:
        save_csv(csv_path, rows)
        print(f"\n\nInterrupted. Progress saved to {csv_path}.")
        sys.exit(0)


def print_summary(rows: list[dict[str, str]]) -> None:
    """Print summary statistics for the labels CSV."""
    total = len(rows)
    labeled = sum(1 for r in rows if r.get("action_type", "").strip())
    compat = sum(1 for r in rows if r.get("strata_compatible", "").strip() == "yes")

    action_counts = Counter(r.get("action_type", "").strip() or "(unlabeled)" for r in rows)
    quality_counts = Counter(r.get("quality", "").strip() or "(none)" for r in rows)

    print(f"\nTotal clips: {total}")
    print(f"Labeled: {labeled}, Unlabeled: {total - labeled}")
    print(f"Strata compatible: {compat}")
    print("\nBy action type:")
    for action, count in sorted(action_counts.items()):
        print(f"  {action}: {count}")
    print("\nBy quality:")
    for qual, count in sorted(quality_counts.items()):
        print(f"  {qual}: {count}")


if __name__ == "__main__":
    main()

"""Aggregate per-character measurement JSONs into a single profiles file.

Reads individual ``{char_id}.json`` files written by
``pipeline.exporter.save_measurements`` and combines them into
``mesh/measurements/measurement_profiles.json`` — the input format
expected by ``mesh/scripts/proportion_clusterer.py`` (Issue #81).

Supports incremental updates: if an output file already exists, new
characters are merged in (existing entries for the same character_id
are overwritten with the newer data).

No Blender dependency — pure Python.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROFILES_VERSION = "1.0"

KNOWN_SOURCE_PREFIXES = [
    "mixamo",
    "sketchfab",
    "quaternius",
    "kenney",
    "vroid",
    "stdgen",
    "nova_human",
    "live2d",
    "spine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def infer_source(character_id: str) -> str:
    """Infer the asset source from a character ID prefix.

    Args:
        character_id: Character identifier (e.g. ``"mixamo_ybot"``).

    Returns:
        Source name (e.g. ``"mixamo"``), or ``"unknown"`` if no known
        prefix matches.
    """
    for prefix in KNOWN_SOURCE_PREFIXES:
        if character_id.startswith(prefix + "_"):
            return prefix
    return "unknown"


def parse_measurement_file(path: Path) -> dict[str, Any] | None:
    """Read and validate a per-character measurement JSON file.

    Args:
        path: Path to a ``{char_id}.json`` measurement file.

    Returns:
        The parsed measurement dict, or ``None`` if the file is
        malformed or missing required keys.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Skipping %s: %s", path, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("Skipping %s: root is not a JSON object", path)
        return None

    if "regions" not in data:
        logger.warning("Skipping %s: missing 'regions' key", path)
        return None

    return data


def build_character_entry(
    character_id: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Build a single character entry for the profiles file.

    Extracts the dimension fields (width, depth, height) from each
    region, dropping internal fields like ``center`` and ``vertex_count``
    that aren't needed by downstream consumers.

    Args:
        character_id: Character identifier.
        data: Raw measurement data from ``parse_measurement_file``.

    Returns:
        Character entry dict matching the profiles schema.
    """
    measurements: dict[str, dict[str, float]] = {}
    for region_name, region_data in data.get("regions", {}).items():
        measurements[region_name] = {
            "width": region_data.get("width", 0.0),
            "depth": region_data.get("depth", 0.0),
            "height": region_data.get("height", 0.0),
        }

    return {
        "character_id": character_id,
        "source": infer_source(character_id),
        "measurements": measurements,
        "total_vertices": data.get("total_vertices", 0),
        "measured_regions": data.get("measured_regions", 0),
    }


def aggregate_measurements(
    measurements_dir: Path,
    output_path: Path,
    *,
    incremental: bool = True,
) -> dict[str, Any]:
    """Aggregate per-character measurement files into a single profiles JSON.

    Args:
        measurements_dir: Directory containing ``{char_id}.json`` files
            (typically ``output/segmentation/measurements/``).
        output_path: Where to write ``measurement_profiles.json``.
        incremental: If True and ``output_path`` exists, merge new
            characters into the existing profiles (overwriting duplicates).

    Returns:
        The complete profiles dict that was written to disk.
    """
    # Load existing profiles for incremental mode
    existing_characters: dict[str, dict[str, Any]] = {}
    if incremental and output_path.is_file():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            for entry in existing.get("characters", []):
                cid = entry.get("character_id")
                if cid:
                    existing_characters[cid] = entry
            logger.info(
                "Loaded %d existing profiles from %s",
                len(existing_characters),
                output_path,
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read existing profiles: %s", exc)

    # Scan measurement files
    json_files = sorted(measurements_dir.glob("*.json"))
    new_count = 0
    updated_count = 0

    for json_path in json_files:
        data = parse_measurement_file(json_path)
        if data is None:
            continue

        character_id = data.get("character_id", json_path.stem)
        entry = build_character_entry(character_id, data)

        if character_id in existing_characters:
            updated_count += 1
        else:
            new_count += 1
        existing_characters[character_id] = entry

    # Build output
    characters = list(existing_characters.values())
    characters.sort(key=lambda c: c["character_id"])

    profiles: dict[str, Any] = {
        "version": PROFILES_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "character_count": len(characters),
        "characters": characters,
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profiles, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Wrote %d profiles to %s (%d new, %d updated)",
        len(characters),
        output_path,
        new_count,
        updated_count,
    )

    return profiles


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_PATH = Path("mesh/measurements/measurement_profiles.json")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for measurement aggregation."""
    parser = argparse.ArgumentParser(
        description="Aggregate per-character measurements into measurement_profiles.json",
    )
    parser.add_argument(
        "measurements_dir",
        type=Path,
        help="Directory containing per-character measurement JSON files "
        "(e.g. output/segmentation/measurements/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for aggregated profiles (default: %(default)s)",
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Overwrite existing profiles instead of merging",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.measurements_dir.is_dir():
        logger.error("Not a directory: %s", args.measurements_dir)
        sys.exit(1)

    profiles = aggregate_measurements(
        args.measurements_dir,
        args.output,
        incremental=not args.no_incremental,
    )

    print(f"Aggregated {profiles['character_count']} character profiles.")


if __name__ == "__main__":
    main()

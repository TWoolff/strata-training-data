# Issue #98: Aggregate measurement_ground_truth.py output into measurement_profiles.json

## Understanding
- `pipeline/measurement_ground_truth.py` extracts per-character body measurements during pipeline runs
- Each character's measurements are saved as individual JSON files in `output/segmentation/measurements/{char_id}.json`
- The pipeline's `generate_dataset.py` already aggregates during a run via `save_measurement_profiles()`, but only for characters processed in that run
- Issue #81 (proportion_clusterer.py) needs `mesh/measurements/measurement_profiles.json` as input — this file needs to be buildable from the individual per-character JSONs independently of a pipeline run
- Need a standalone script that reads all per-character measurement JSONs and produces one aggregated file

## Approach
- Create `mesh/scripts/aggregate_measurements.py` — a standalone CLI script (no Blender dependency)
- Script scans a dataset directory's `measurements/` subdirectory for per-character JSON files
- Aggregates into the schema expected by future proportion_clusterer.py
- Supports incremental updates: reads existing profiles, merges new ones, writes back
- Output goes to `mesh/measurements/measurement_profiles.json`
- Use argparse for CLI, pathlib for paths, follow project conventions

## Files to Modify
1. **NEW: `mesh/scripts/aggregate_measurements.py`** — Main aggregation script
2. **NEW: `mesh/measurements/.gitkeep`** — Ensure directory is tracked (output JSON is gitignored)
3. **NEW: `tests/test_aggregate_measurements.py`** — Tests for aggregation logic

## Schema

Input per-character JSON (from exporter.py `save_measurements`):
```json
{
  "regions": {
    "head": {"width": 0.3, "depth": 0.25, "height": 0.3, "center": [0.0, 0.0, 1.8], "vertex_count": 500},
    ...
  },
  "total_vertices": 12345,
  "measured_regions": 17,
  "character_id": "mixamo_ybot"
}
```

Output `measurement_profiles.json` (matching issue spec):
```json
{
  "version": "1.0",
  "generated_at": "2026-02-28T12:00:00Z",
  "character_count": 2,
  "characters": [
    {
      "character_id": "mixamo_ybot",
      "source": "mixamo",
      "measurements": {
        "head": {"width": 0.3, "depth": 0.25, "height": 0.3},
        "chest": {"width": 0.35, "depth": 0.20, "height": 0.30},
        ...
      },
      "total_vertices": 12345,
      "measured_regions": 17
    }
  ]
}
```

Note: `source` is inferred from `character_id` prefix (e.g., "mixamo_ybot" → "mixamo").

## Risks & Edge Cases
- Malformed per-character JSON files (missing keys) — skip and log warning
- Empty measurements directory — produce empty profiles with character_count: 0
- Incremental mode: character already exists in profiles — overwrite with newer data
- Character ID may not have a recognizable source prefix — use "unknown"
- Very large number of character files — should be fine since we're just reading small JSONs

## Open Questions
- None — the issue spec and PRD §13.7 are clear on requirements

## Implementation Notes
- Created `mesh/scripts/aggregate_measurements.py` as a standalone CLI script (no Blender dependency)
- Created `mesh/__init__.py` and `mesh/scripts/__init__.py` for package importability
- Output schema includes `version`, `generated_at`, `character_count`, and sorted `characters` array
- Each character entry includes `character_id`, `source` (inferred from prefix), `measurements` (width/depth/height only — stripped center + vertex_count), `total_vertices`, `measured_regions`
- Incremental mode is the default — reads existing output and merges new characters
- Falls back to filename stem when `character_id` key is missing from measurement JSON
- 23 tests covering: source inference, file parsing, entry building, full aggregation (empty dir, malformed files, incremental merge, overwrite, non-incremental, nested output dirs, fallback ID)
- All 548 tests pass, ruff lint clean, ruff format clean

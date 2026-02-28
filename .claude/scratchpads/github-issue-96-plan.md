# Issue #96: Add top-level scripts/ for download verification and cross-source statistics

## Understanding
- Create a new `scripts/` directory with three utility scripts for cross-source dataset management
- `verify_downloads.py` — check integrity of `data/preprocessed/` datasets
- `compute_stats.py` — generate statistics across all data sources in `output/`
- `generate_splits.py` — unified train/val/test splits across all sources
- Type: new feature / infrastructure

## Approach
- Pure Python scripts (no Blender dependency), matching the pattern of `pipeline/splitter.py` and `pipeline/manifest.py`
- Each script is a module with a public API function + CLI via `if __name__ == "__main__"` with argparse
- Reuse existing logic: `generate_splits.py` extends `pipeline/splitter.py`'s logic for cross-source stratification
- `compute_stats.py` mirrors `pipeline/manifest.py`'s region distribution/counting but across all output sources
- `verify_downloads.py` defines per-dataset expected structures matching `download_datasets.sh`'s targets

### Design decisions
1. **verify_downloads.py**: Define a registry of dataset specs (expected dirs, file patterns, min counts). Walk `data/preprocessed/` and report status. Non-zero exit on failure.
2. **compute_stats.py**: Scan `output/` for all subdirectories (segmentation pipeline output + ingest adapter output). Count images/masks/joints per source. Compute region distribution from masks. Print table to stdout, optionally write JSON.
3. **generate_splits.py**: Discover characters from all output directories (both `output/segmentation/sources/*.json` and ingest adapter outputs). Group by source. Use `pipeline/splitter.py`'s proportional assignment logic. Output `splits.csv` mapping each example to train/val/test.

## Files to Create
- `scripts/__init__.py` — empty package init
- `scripts/verify_downloads.py` — download verification
- `scripts/compute_stats.py` — cross-source statistics
- `scripts/generate_splits.py` — unified cross-source splits
- `tests/test_verify_downloads.py` — tests for verify_downloads
- `tests/test_compute_stats.py` — tests for compute_stats
- `tests/test_generate_splits.py` — tests for generate_splits

## Files to Reference (not modify)
- `pipeline/splitter.py` — split logic to extend
- `pipeline/manifest.py` — stat computation patterns
- `pipeline/config.py` — SPLIT_RATIOS, REGION_NAMES, NUM_REGIONS, ART_STYLES
- `ingest/download_datasets.sh` — dataset names and directory targets

## Risks & Edge Cases
- Output directories may be empty or partially populated
- Ingest adapter output formats differ from Blender pipeline output format
- Some preprocessed datasets may be partially downloaded
- Region distribution computation on large datasets could be slow → use sampling like manifest.py
- A character appearing in multiple output sources could cause leakage if not deduplicated

## Open Questions
- None — the issue is well-specified

## Implementation Notes
- All three scripts implemented as planned with public API + CLI entry points
- `verify_downloads.py`: Registry of 9 `DatasetSpec` entries matching `download_datasets.sh`. Spot-checks image validity (up to 10 samples) and resolution. Non-zero exit on any failure.
- `compute_stats.py`: Discovers output sources by looking for `images/` or `masks/` subdirs. Region distribution uses sampling (200 masks max) like `manifest.py`. Coverage report flags missing (<0%) and under-represented (<1%) regions.
- `generate_splits.py`: Discovers characters from both `sources/*.json` metadata and image filename fallback. Disambiguates duplicate character IDs across sources. Outputs both CSV manifest and JSON format compatible with `pipeline/splitter.py`.
- `CAMERA_ANGLES` in config.py is a `dict[str, CameraAngle]` (not a list of dicts) — angle names are keys.
- 50 tests total across 3 test files, all passing. 12 pre-existing failures in `test_proportion_clusterer.py` unrelated.

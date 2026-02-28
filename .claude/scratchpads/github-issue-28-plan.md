# Issue #28: Merge 2D sources into unified dataset

## Understanding
- Build a merge tool that combines outputs from multiple source pipelines (3D synthetic, Spine 2D, Live2D, PSD, manual annotations, ingest adapters) into a single unified dataset
- Each source pipeline produces output in the standard directory layout (images/, masks/, joints/, weights/, draw_order/, sources/, measurements/, contours/)
- The merge tool copies/links files, validates inputs, resolves ID conflicts, regenerates splits.json and manifest.json
- Type: **new feature** — a new `pipeline/dataset_merger.py` module + `run_merge.py` CLI entry point

## Approach

### Core design: file-based merge
1. Accept a list of source directories (each containing the standard dataset layout)
2. Accept an output directory for the merged dataset
3. For each source directory:
   - Discover characters from `sources/*.json` metadata
   - Validate each character's files against resolution/format requirements
   - Copy or symlink files into the merged output directory
   - Track character IDs and detect collisions
4. After merging all sources:
   - Write `class_map.json` (canonical, from config)
   - Regenerate `splits.json` using `scripts/generate_splits.py` logic (stratified by source)
   - Generate a standalone manifest.json (scan-based, not results-based)
5. Support `--copy` (default) and `--link` modes

### Key decisions
- **Use `pipeline/dataset_merger.py`** — follows the module-per-responsibility pattern
- **CLI at `run_merge.py`** — follows the `run_validation.py` / `run_pipeline.py` pattern
- **Don't reuse `generate_manifest()` as-is** — it requires `CharacterResult` objects from the Blender pipeline. Instead, implement a `generate_merge_manifest()` that scans the output directory (reuse internal helpers like `_count_files`, `_count_images_by_style`, `_count_images_by_source`, `_compute_region_distribution`)
- **Reuse `scripts/generate_splits.py`** for cross-source splits (it already handles multi-directory discovery and source stratification)
- **Validate per-file before including** — use individual check functions from `pipeline/validator.py` (resolution, mask format) but don't run full `validate_dataset()` until after merge

### ID collision resolution
- Each source directory's characters have a `source` field in their metadata
- If the same `char_id` appears in two source directories with different sources, disambiguate by prefixing with source name (e.g., `nova_001` → `nova_human_nova_001`)
- Log warnings for all collisions

## Files to Modify
1. **NEW `pipeline/dataset_merger.py`** — Core merge logic:
   - `merge_datasets(source_dirs, output_dir, *, mode="copy", validate=True, resolution=512) -> MergeReport`
   - Internal helpers: `_discover_source_characters()`, `_validate_character_files()`, `_copy_character_files()`, `_link_character_files()`
   - `MergeReport` dataclass: counts of characters merged, skipped, collisions resolved
2. **NEW `run_merge.py`** — CLI entry point following `run_validation.py` pattern
3. **NEW `tests/test_dataset_merger.py`** — Unit tests

## Risks & Edge Cases
- Source directories with overlapping character IDs (different sources) — handled by disambiguation
- Source directories with overlapping character IDs (same source) — log error, skip duplicate
- Files that fail validation — skip with warning, don't halt the entire merge
- Empty source directories — skip gracefully
- Mixed file sets (some characters have weights/draw_order, others don't) — only copy what exists
- Very large datasets — use streaming/iterative approach, not load-all-into-memory

## Open Questions
- None — the issue and PRD are clear enough to proceed

## Implementation Notes

### What was implemented
- **`pipeline/dataset_merger.py`** — Core merge module with:
  - `merge_datasets()` — main public API accepting source dirs, output dir, mode, validation, resolution, seed
  - `MergeReport` dataclass — tracks sources processed, characters merged/skipped/renamed, files copied/linked, validation failures
  - `_discover_characters()` — reads `sources/*.json` metadata, falls back to image filename inference
  - `_collect_character_files()` — gathers all per-pose and per-character files, avoiding prefix false matches
  - `_validate_character()` — resolution checks on images and masks before merging
  - `_transfer_file()` — copy or symlink with parent dir creation
  - `_transfer_character_files()` — handles renaming for ID collisions, updates metadata
  - `_generate_merge_manifest()` — scan-based manifest generation reusing helpers from `pipeline/manifest.py`
  - `print_merge_report()` — human-readable report output
- **`run_merge.py`** — CLI entry point with `--source_dirs`, `--output_dir`, `--link`, `--no_validate`, `--resolution`, `--seed` flags
- **`tests/test_dataset_merger.py`** — 20 tests covering discovery, file collection, renaming, validation, full merge (basic, collisions, symlinks, validation skip, missing dirs, splits/manifest/class_map generation)

### Design decisions during implementation
- Used `pipeline/splitter.py`'s `generate_splits()` directly (it reads from `sources/` dir) rather than `scripts/generate_splits.py` (which scans subdirectories — not needed after merge flattens everything)
- Manifest reuses `_count_files`, `_count_images_by_style`, `_count_images_by_source`, `_compute_region_distribution` from `pipeline/manifest.py` — avoids duplicating scan logic
- ID collision when same source: skip (log warning). Different source: prefix with source name, update metadata with `original_id` field

### Test results
- 20/20 tests pass
- Ruff check + format clean
- Full test suite: 838 passed, 12 failed (pre-existing sklearn dep), 1 skipped — no regressions

# Issue #21: Generate dataset manifest with statistics and quality report

## Understanding
- Generate a `manifest.json` at the dataset root after a full batch run
- Records dataset metadata (version, date, pipeline version), file count statistics (by style, by source), region pixel distributions (sampled), and quality info (unmapped bones, failed characters, warnings)
- This is a post-processing step that scans the output directory and collects stats from the `CharacterResult` list

## Approach
- Create a new `pipeline/manifest.py` module (single responsibility, like exporter.py)
- Add `PIPELINE_VERSION` constant to `config.py`
- The manifest generator receives:
  1. `output_dir` — to scan for actual files on disk
  2. `results: list[CharacterResult]` — execution stats from the batch
  3. Pipeline config (resolution, styles, poses_per_character, etc.)
- Region distribution: sample up to 100 mask files, count pixels per region, average
- Source counts: parse `{source}_{id}` prefix from image filenames via the `sources/` metadata JSONs
- Call `generate_manifest()` at the end of `main()` in `generate_dataset.py`, after all processing

## Files to Modify
1. **`pipeline/config.py`** — Add `PIPELINE_VERSION = "0.1.0"` constant
2. **`pipeline/manifest.py`** (NEW) — Core manifest generation logic:
   - `generate_manifest(output_dir, config, results) -> Path`
   - `_count_files(output_dir) -> dict` — glob each subdir
   - `_count_by_style(output_dir) -> dict` — parse image filenames
   - `_count_by_source(output_dir) -> dict` — read source metadata JSONs
   - `_compute_region_distribution(output_dir, sample_size=100) -> dict` — sample masks, average pixel counts
   - `_collect_quality_info(results) -> dict` — unmapped bones, failures, warnings
3. **`pipeline/generate_dataset.py`** — Import and call `generate_manifest()` after `_print_summary()`

## Risks & Edge Cases
- Empty dataset (no characters processed) — should still produce a valid manifest with zero counts
- Mask files that are all-background (region 0) — valid edge case, include in distribution
- Characters with no source metadata JSON — count by filename prefix fallback
- Very large datasets — sample masks for distribution (cap at 100)
- `_infer_source()` already exists in generate_dataset.py — reuse for source counting

## Open Questions
- None — the issue spec is detailed and the schema is well-defined

## Implementation Notes
- Created `pipeline/manifest.py` as a new module (pure Python, no Blender dependency)
- Added `PIPELINE_VERSION = "0.1.0"` to `config.py` at the top, before type aliases
- Used `np.bincount` for vectorized region pixel counting (faster than per-region loop)
- Source counting uses a two-tier approach: reads `sources/*.json` metadata first, falls back to `_infer_source_from_id()` prefix heuristic
- Quality info is collected from both `CharacterResult` objects (failures/warnings) and source metadata files (unmapped bones)
- `generate_manifest()` uses keyword-only args for clarity, typed as `list[Any]` for results to avoid circular import with `CharacterResult`
- Manifest is generated after `_print_summary()` but before the exit code check in `main()`
- File counts verified against actual disk contents via `Path.iterdir()`, not from in-memory results

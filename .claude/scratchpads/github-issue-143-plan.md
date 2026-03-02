# Issue #143: Build vroid_lite ingest adapter (4,651 VRoid character renders)

## Understanding
- The `vroid_lite` dataset (7.3 GB, 4,651 RGBA images from 16 VRoid characters) is downloaded at `data/preprocessed/vroid_lite/` but has no ingest adapter
- Images are 1536×1024 RGBA PNGs (landscape) with rich per-image metadata in `metadata.jsonl`
- No annotations (no segmentation masks, joints, or draw order) — image-only like fbanimehq
- 16 unique characters (~290 images each), varying camera angles, poses, expressions, lighting
- This is a new feature — creating an adapter following the established pattern

## Approach
- Use `fbanimehq_adapter.py` as the closest template (simple image-only adapter)
- Key difference: JSONL-driven discovery instead of directory walk
  - Read `metadata.jsonl` first, resolve each entry to its image file
  - Images without metadata are ignored; missing images are flagged
- Example ID format: `vroid_lite_{vrm_name}_{uuid_first_segment}` (verified zero collisions in issue)
- Metadata includes standard Strata fields at top level + `"character"` field for per-character splits + all VRoid fields nested under `"vroid_lite_metadata"`
- Image resize: 1536×1024 → aspect-ratio-preserving fit into 512×512 with transparent padding

## Files to Modify
1. **`ingest/vroid_lite_adapter.py`** (NEW) — Main adapter module
   - `VroidLiteEntry` dataclass — pairs metadata row with resolved image path
   - `AdapterResult` dataclass — reuse pattern from fbanimehq
   - `discover_entries(input_dir)` — parse JSONL, resolve paths, skip missing
   - `_resize_to_strata(img, resolution)` — same logic as fbanimehq
   - `_make_example_id(entry)` — `vroid_lite_{vrm_name}_{uuid_first_segment}`
   - `_build_metadata(entry, resolution, original_size)` — Strata fields + character + nested vroid metadata
   - `_save_example()` — creates `{example_id}/image.png` + `{example_id}/metadata.json`
   - `convert_entry(entry, output_dir)` — single entry conversion
   - `convert_directory(input_dir, output_dir, ...)` — batch with max_images, random_sample, only_new

2. **`run_ingest.py`** — Register adapter
   - Add `"vroid_lite"` to `--adapter` choices
   - Add `_run_vroid_lite()` dispatch function
   - Add to `_ADAPTERS` dict

3. **`tests/test_vroid_lite_adapter.py`** (NEW) — Comprehensive tests
   - Test JSONL parsing with mock data
   - Test malformed JSONL line handling
   - Test missing image file handling
   - Test image resize and padding
   - Test metadata structure
   - Test example ID generation
   - Test batch processing with max_images, random_sample, only_new

## Risks & Edge Cases
- Malformed JSONL lines — must log and skip, not crash
- Missing image files — must log and skip
- UUID collision in example IDs — issue confirms zero collisions with first segment
- Images with unexpected dimensions or modes — handle gracefully like fbanimehq
- Empty metadata.jsonl or missing file — return empty result

## Open Questions
- None — the issue is very detailed and prescriptive

## Implementation Notes
- Implemented exactly as planned, no deviations from the approach
- JSONL-driven discovery works cleanly — `discover_entries()` parses line-by-line, skips malformed/missing gracefully
- Used `VroidLiteEntry` dataclass to pair metadata dict with resolved image Path
- `_make_example_id` uses UUID first segment (8 hex chars) — compact and collision-free
- `_build_metadata` nests all VRoid fields under `vroid_lite_metadata`, excludes `file_name` (redundant with `source_filename`)
- `character` field set to `vrm_name` for downstream per-character splits
- 38 tests covering discovery, ID generation, resize, metadata, single entry, and batch conversion
- Code simplifier found no issues — patterns match fbanimehq adapter exactly

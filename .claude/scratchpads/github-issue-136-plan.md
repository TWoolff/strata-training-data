# Issue #136: AnimeRun instance segmentation adapter

## Understanding
- AnimeRun provides per-object instance segmentation maps alongside anime frames
- Data is in `Segment/{scene}/` directories, parallel to `Frame_Anime/{scene}/original/`
- This is **instance segmentation** (whole objects), NOT Strata's 20-class body part segmentation
- Useful for character detection/tracking in animation, and as a pre-processing step
- Need a new adapter `animerun_segment_adapter.py` following the contour adapter pattern

## Approach
- Model closely after `animerun_contour_adapter.py` — same dataset structure, same scene-based discovery
- Reuse the same `_find_root()` and `discover_scenes()` logic adapted for Segment dirs
- Parse segment maps (PNG or NPY) and pair with anime frames
- Generate color overlay visualization for QA
- Output: `image.png`, `instance_mask.png`, `instance_overlay.png`, `metadata.json`
- Register as `animerun_segment` in `run_ingest.py`

## Files to Modify
1. **`ingest/animerun_segment_adapter.py`** (NEW) — Main adapter module
   - `SegmentFrame` dataclass — frame_id, scene_id, split, segment_path, anime_path
   - `AdapterResult` dataclass — scene_id, frames_saved, frames_skipped, errors
   - `discover_scenes()` — find scenes with both Segment/ and Frame_Anime/ dirs
   - `discover_frames()` — match segment maps to anime frames by stem
   - `convert_frame()` — load segment map + anime frame, resize, generate overlay, save
   - `convert_scene()` — process all frames in a scene
   - `convert_directory()` — entry point for full dataset

2. **`run_ingest.py`** — Register `animerun_segment` adapter
   - Add to `--adapter` choices
   - Add `_run_animerun_segment()` dispatch function
   - Add to `_ADAPTERS` dict

3. **`tests/test_animerun_segment_adapter.py`** (NEW) — Tests
   - Test discovery, conversion, metadata, overlay generation, edge cases

## Risks & Edge Cases
- Segment maps could be PNG or NPY — need to handle both formats
- Instance IDs in segment maps may be arbitrary integers (not contiguous 0-N)
- NPY files may have different dtypes (uint8, uint16, int32, etc.)
- Zero-instance frames (all background) should still be valid
- Segment map and anime frame resolution mismatch — need nearest-neighbor resize for masks

## Open Questions
- What colormap to use for overlay visualization? → Use a distinct color per instance ID
- Should we normalize instance IDs to contiguous range? → No, preserve original IDs

## Implementation Notes
- Adapter created at `ingest/animerun_segment_adapter.py` (~350 lines)
- Closely follows `animerun_contour_adapter.py` structure: same discovery pattern, scene-based processing
- Key functions: `load_segment_map()` handles both PNG and NPY, `generate_overlay()` blends colors at 50% alpha
- Mask resizing uses `Image.NEAREST` to preserve exact instance IDs (no interpolation artifacts)
- Metadata includes `instance_ids` list and `num_instances` count (excluding background=0)
- 20 distinct overlay colors with wrapping for scenes with many instances
- Registered as `--adapter animerun_segment` in `run_ingest.py`
- 32 tests passing, covering: load formats (PNG, NPY, RGB, 3D), resize, overlay generation, metadata, discovery, conversion, edge cases
- All ruff lint and format checks pass

# Issue #85: Build 2D-to-3D measurement extraction pipeline

## Understanding
- Create a module that extracts **apparent** body part dimensions from 2D segmentation masks at known camera angles and pairs them with **true** 3D dimensions from mesh data
- Produces training pairs: `(apparent_width, apparent_height, camera_angle) → (true_width, true_depth, true_height)`
- This is for the 3D mesh pipeline — a model will learn to predict true 3D body dimensions from 2D observations
- Type: **new feature** (pure Python module, no Blender dependency for extraction)

## Approach
1. **Per-image extraction**: Read a segmentation mask (8-bit grayscale, pixel=region_id), compute bounding box per visible region → apparent width/height in pixels
2. **Normalization**: Convert pixel measurements to a scale-independent ratio. Since the render uses orthographic projection with known `ortho_scale` and `RENDER_RESOLUTION`, we can convert pixels → world units. Alternatively, normalize by character bounding box height (always known). The issue asks for pixel-level measurements paired with world-unit ground truth, so we output both.
3. **Foreshortening handling**: At side view (90°), apparent width corresponds to true depth, not true width. The module records the raw `camera_angle` (azimuth) so the downstream model learns the mapping itself. We don't transform — we just provide the training signal.
4. **Partial occlusion**: Some regions have zero pixels at certain angles (e.g., left arm hidden at right-side view). Mark these with `"visible": false`.
5. **Aggregation**: Produce per-image JSON + an aggregated training pair dataset.

### Key design decisions
- Pure Python with NumPy/OpenCV — no Blender dependency
- Input: segmentation mask PNG + ground truth `measurements.json` (from issue #74) + metadata with camera angle
- Output: per-image `{char_id}_pose_{nn}_{angle}_measurements_2d.json` + aggregated `measurement_training_pairs.json`
- Functions follow the single-entry-point pattern: `extract_apparent_measurements()` for per-image, `build_training_pairs()` for aggregation

## Files to Modify
1. **CREATE** `pipeline/measurement_extractor.py` — Core module
2. **CREATE** `tests/test_measurement_extractor.py` — Test suite

No changes needed to existing modules for the core feature. Integration into `generate_dataset.py` is a separate concern (could be a follow-up or done inline if scope allows).

## Output Format

### Per-image measurement JSON
```json
{
  "character_id": "mixamo_ybot",
  "pose": "pose_01",
  "camera_angle": "front",
  "azimuth": 0,
  "regions": {
    "head": {
      "apparent_width": 48,
      "apparent_height": 52,
      "bbox": [232, 10, 280, 62],
      "pixel_count": 1850,
      "visible": true
    },
    ...
  }
}
```

### Training pair format (aggregated)
```json
{
  "version": "1.0",
  "pair_count": 15000,
  "pairs": [
    {
      "character_id": "mixamo_ybot",
      "region": "head",
      "camera_angle": "front",
      "azimuth": 0,
      "apparent_width": 48,
      "apparent_height": 52,
      "pixel_count": 1850,
      "true_width": 0.25,
      "true_depth": 0.22,
      "true_height": 0.28
    }
  ]
}
```

## Risks & Edge Cases
- **No pixels for a region**: At side/back views, some regions are fully occluded. Return `visible: false` with null measurements.
- **Very small regions**: Neck, shoulder regions may be only a few pixels. Still extract bbox — the model needs this signal.
- **Missing ground truth**: Some characters may not have `measurements.json`. Skip with warning.
- **Scale normalization**: Different characters have different ortho_scale per angle. Raw pixel measurements are relative to render. Downstream model handles this with camera angle as input.

## Open Questions
- None — the issue and PRD are clear on requirements.

## Implementation Notes

### What was implemented
- Created `pipeline/measurement_extractor.py` with 5 public functions:
  - `extract_apparent_measurements()` — extracts per-region bounding boxes from a segmentation mask
  - `build_training_pairs()` — combines apparent 2D measurements with ground truth 3D dimensions
  - `save_apparent_measurements()` — saves per-image measurement JSON to `measurements_2d/`
  - `save_training_pairs()` — saves aggregated training pairs JSON
  - `load_ground_truth()` — loads and validates per-character ground truth files
- Created `tests/test_measurement_extractor.py` with 26 tests across 5 test classes

### Design decisions made during implementation
- Invisible regions get `visible: false` with zeroed measurements and `bbox: null` (rather than omitting them entirely), so downstream consumers have a consistent schema
- `build_training_pairs()` silently skips regions without ground truth (logs at DEBUG level) — this is expected for characters with partial measurements
- No OpenCV dependency needed — NumPy alone handles all mask operations
- File I/O helpers create parent directories automatically (consistent with `exporter.py` pattern)

### Not yet implemented (follow-up work)
- Integration into `generate_dataset.py` render loop (calling extraction after mask render)
- CLI script for batch processing existing masks + ground truth into training pairs
- Ortho-scale-based pixel-to-world-unit conversion (deferred — downstream model can learn this)

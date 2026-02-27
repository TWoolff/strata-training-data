# Issue #42: Add draw order extractor to pipeline

## Understanding
- Add per-pixel draw order (Z-depth) extraction as a new output channel
- For each body region, compute average Z-depth in camera space from vertices
- Normalize to [0, 255] and paint each pixel of the segmentation mask with its region's depth value
- Output: 8-bit grayscale PNG (0=farthest, 255=nearest to camera)
- This is per-pose, NOT per-style (like masks and joints)
- Zero additional render cost — uses existing vertex data and camera parameters

## Approach
1. **New file `pipeline/draw_order_extractor.py`**:
   - Reuse `world_to_camera_view()` from `joint_extractor.py` pattern
   - For each region (1-19), gather all vertices belonging to bones mapped to that region
   - Project each vertex to camera space, take the Z component as depth
   - Compute mean Z-depth per region
   - Normalize all depths to [0, 255]: min depth → 0 (back), max depth → 255 (front)
   - Read the segmentation mask, replace each pixel's region ID with the region's normalized depth

2. **Modify `pipeline/exporter.py`**:
   - Add `"draw_order"` to `_SUBDIRS`
   - Add `draw_order_filename()` helper following the mask naming pattern
   - Add `save_draw_order()` function following `save_mask()` pattern

3. **Modify `pipeline/generate_dataset.py`**:
   - Import `extract_draw_order`
   - Call after segmentation mask is created (need the mask), before color render
   - Handle flip augmentation: horizontally flip the draw order map (no region ID swapping needed since depth values are per-pixel, not region IDs)

## Files to Modify
- `pipeline/draw_order_extractor.py` — **NEW** — core extraction logic
- `pipeline/exporter.py` — add draw_order subdir, filename helper, save function
- `pipeline/generate_dataset.py` — import and call draw order extraction in pose loop

## Risks & Edge Cases
- Regions with zero vertices: assign depth 0 (background/farthest)
- All regions at same depth: all get 127 or some midpoint (but unlikely in practice)
- Vertices not in any region (unmapped bones): excluded from computation
- Flip augmentation: simple horizontal flip of the depth map image (no ID swapping needed)
- The segmentation mask must exist before draw order can be computed (dependency ordering)

## Open Questions
- None — issue is well-specified with clear algorithm and function signature

## Implementation Notes

### What was implemented
Followed the plan exactly. Three files touched:

1. **`pipeline/draw_order_extractor.py`** (new, ~200 lines)
   - `_compute_region_depths()`: iterates all vertices across all meshes, finds dominant bone via highest vertex group weight, accumulates camera-space Z per region
   - `_normalize_depths()`: min-max normalizes to [0, 255], inverts so closer = higher value
   - `extract_draw_order()`: public entry point matching the issue's function signature. Reads the segmentation mask and paints each pixel with its region's normalized depth

2. **`pipeline/exporter.py`**
   - Added `"draw_order"` to `_SUBDIRS`
   - Added `draw_order_filename()` helper
   - Added `save_draw_order()` function (for standalone use; the pipeline uses inline saving for augmentation suffix support)
   - Updated module docstring directory layout

3. **`pipeline/generate_dataset.py`**
   - Draw order extraction runs after mask conversion and joint extraction, before color render
   - Uses inline `Image.fromarray().save()` (matching how masks bypass `save_mask()` to handle augmentation suffixes in filenames)
   - Flip augmentation: `Image.FLIP_LEFT_RIGHT` on the draw order map (no region ID swap needed)

### Design decisions
- **Inline saving vs `save_draw_order()`**: The pipeline constructs filenames with augmentation suffixes (`_flip`, `_s085`) using `pose_suffix`. The exporter's `save_draw_order()` uses `draw_order_filename()` which doesn't include augmentation suffixes. Kept both: inline for the pipeline (matches mask pattern), `save_draw_order()` in exporter for external tooling.
- **Depth direction**: `world_to_camera_view()` Z increases with distance from camera. We invert so 255 = nearest (front), 0 = farthest (back), matching the issue spec.
- **Edge case — all regions same depth**: assigns 127 (midpoint) to all regions.
- **Edge case — missing regions**: regions with no mapped vertices get depth 0 in the named output dict.

### No follow-up work needed
Implementation covers all acceptance criteria from the issue.

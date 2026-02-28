# Issue #82: Add Freestyle contour rendering with style augmentation

## Understanding
- Render paired images (with_contours, without_contours) using Blender Freestyle for contour-removal training data
- Compute binary contour_mask from the difference between the two renders
- Generate 5 hand-drawn contour style variants per pair via post-processing (PIL/OpenCV)
- New feature — two new pipeline modules: `contour_renderer.py` (Blender) + `contour_augmenter.py` (pure Python)
- Output matches the same format as `ingest/animerun_contour_adapter.py` (with_contours.png, without_contours.png, contour_mask.png)

## Approach
1. **contour_renderer.py** — Blender module that:
   - Enables Freestyle with configurable edge types (silhouette, crease, material boundary)
   - Renders the scene with original materials + Freestyle ON → `with_contours.png`
   - Renders same scene with Freestyle OFF → `without_contours.png`
   - Computes pixel difference → `contour_mask.png` (binary, 0/255)
   - Follows existing `apply_cel_style()` pattern for Freestyle setup/teardown
   - Uses `setup_color_render()` for EEVEE config, then adds Freestyle on top

2. **contour_augmenter.py** — Pure Python (PIL/OpenCV) that:
   - Takes `without_contours` image + `contour_mask`
   - Generates 5 style variants by dilating mask to target width, coloring, and compositing
   - Style 1: Thin black (1px, black, 100% opacity)
   - Style 2: Medium black (2px, black, 100% opacity)
   - Style 3: Thick brown (3px, dark brown, 90% opacity)
   - Style 4: Colored per-region (1px, color varies by body part, 80% opacity) — needs segmentation mask
   - Style 5: Hand-drawn wobbly (2px, jittered path, 100% opacity) — reuses sketch wobble pattern

3. **Integration** — Add `--contours` flag to CLI; in the camera-angle loop, after color renders,
   optionally render contour pairs and run augmentation. Output goes to `contours/` subdirectory.

## Files to Modify
- `pipeline/config.py` — Add CONTOUR_* constants (edge types, augmentation style definitions, diff threshold)
- `pipeline/contour_renderer.py` — NEW: Freestyle render pair + mask computation
- `pipeline/contour_augmenter.py` — NEW: 5 contour style variants
- `pipeline/exporter.py` — Add "contours" to _SUBDIRS, add save helpers
- `pipeline/generate_dataset.py` — Add --contours CLI flag, integrate into render loop
- `tests/` — Tests for both new modules

## Risks & Edge Cases
- Freestyle line rendering quality depends on mesh topology — low-poly meshes may produce sparse lines
- The difference-based mask computation needs a threshold (reuse CONTOUR_DIFF_THRESHOLD=30 from AnimeRun adapter)
- Style 4 (per-region coloring) requires the segmentation mask at contour pixels — need to sample region IDs from the seg mask
- Freestyle rendering adds another render pass per angle (perf impact ~2x for contour-enabled runs)
- Must ensure Freestyle state is fully cleaned up to not affect subsequent renders

## Open Questions
- None — PRD section 13.5 is clear, and the AnimeRun adapter provides the output format reference

## Implementation Notes
- **contour_renderer.py**: 180 lines. `enable_freestyle()` configures edge selection flags
  (silhouette, crease, material_boundary) from config constants. `render_contour_pair()` does two
  render passes (Freestyle on/off), writes to `contours/` subdir, computes diff mask.
  `compute_contour_mask()` is pure Python (testable without Blender).
- **contour_augmenter.py**: 200 lines. Pure Python (PIL/OpenCV). `_dilate_mask()` uses
  `cv2.getStructuringElement(MORPH_ELLIPSE)` for line width control. `_apply_wobble()` reuses the
  same `cv2.remap` displacement technique from `style_augmentor.apply_sketch()`.
  `_composite_contours()` does per-pixel alpha blending. `_composite_per_region()` colors contour
  pixels based on underlying segmentation region ID using `CONTOUR_REGION_COLORS` lookup.
- **config.py**: Added ~90 lines of CONTOUR_* constants: edge flags, Freestyle thickness,
  diff threshold, 5 style definitions (dicts), wobble range, per-region color map.
- **generate_dataset.py**: `--contours` flag, passed through `main()` → `process_character()` →
  `_process_single_pose()`. Contour rendering block inserted in the camera-angle loop after color
  renders and before seg material reassignment. Uses the seg mask from the same pose for per-region
  style coloring.
- **exporter.py**: Added "contours" to `_SUBDIRS`.
- **Tests**: 8 tests for `compute_contour_mask` (renderer), 19 tests for augmenter (skipped if
  cv2 unavailable). All pass. Pre-existing sklearn failures in proportion_clusterer unrelated.

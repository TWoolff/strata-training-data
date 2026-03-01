# Issue #53: Add occluded region rendering to pipeline (v2.0)

## Understanding
- Add per-region RGBA layer extraction to the Blender and Live2D pipelines
- Each training example gets 19 additional images (one per body region) showing that region in isolation against a transparent background
- Motivated by "See Through" paper (arXiv 2602.03749) which trains a model to decompose anime illustrations into semantic RGBA layers
- Type: new feature (opt-in via `--layers` CLI flag)

## Approach

### Blender pipeline (pipeline/layer_extractor.py)
For each pose, after segmentation + draw order extraction:
1. Identify which regions (1-19) have pixels in the segmentation mask (skip empty ones)
2. Restore original character materials, apply flat style (Diffuse BSDF with original colors)
3. For each active region R:
   - Replace all material slots except R with the transparent material
   - Render → RGBA image with only region R visible
   - Save to `layers/{char_id}_pose_{NN}_{RR}.png`
4. Restore segmentation materials so the rest of the pipeline continues

Key: We DON'T hide front regions — we render each region fully alone. Since it's 3D, the complete mesh surface exists even when normally occluded.

### Live2D pipeline (pipeline/live2d_renderer.py)
Fragments are already separate images. Group by region, composite each group independently onto its own transparent canvas.

### Performance
~2s per render × ~12 active regions × 20 poses × 61 chars ≈ 8 hours.
Opt-in only via `--layers` flag. Will run as a separate batch, not in overnight batch.

## Files to Modify
- **Create** `pipeline/layer_extractor.py` — Blender per-region isolation rendering (~120 lines)
- **Modify** `pipeline/exporter.py` — Add "layers" to _SUBDIRS, add `layer_filename()` helper
- **Modify** `pipeline/generate_dataset.py` — Add `--layers` flag, thread through, hook after draw order
- **Modify** `pipeline/live2d_renderer.py` — Add `_extract_region_layers()` function
- **Create** `tests/test_layer_extractor.py` — Unit tests for active region detection, naming

## Risks & Edge Cases
- Material slot count mismatch: Some meshes may have fewer than 20 slots if no faces map to certain regions. Need to handle gracefully.
- Transparent material setup: Must ensure the transparent material works for both Blender 4.x and 5.x (DITHERED vs CLIP blend modes)
- Render state: Must restore segmentation materials after layer extraction so the color render pass works correctly
- Performance: 19 renders per pose is expensive — active region skipping is critical

## Open Questions
- None — approach confirmed during planning phase

## Implementation Notes

### What was implemented

All planned files were created/modified as specified:

| File | Action | Lines changed |
|------|--------|--------------|
| `pipeline/layer_extractor.py` | Created | ~250 lines |
| `pipeline/exporter.py` | Modified | +20 lines (layers subdir + `layer_filename()`) |
| `pipeline/generate_dataset.py` | Modified | +40 lines (CLI flag + threading + hook point) |
| `pipeline/live2d_renderer.py` | Modified | +70 lines (`extract_region_layers()` + wiring) |
| `tests/test_layer_extractor.py` | Created | ~170 lines (15 tests) |
| `tests/test_animerun_contour_adapter.py` | Fixed | Updated for v2 AnimeRun directory layout |

### Design decisions during implementation

1. **Flat-style materials built inline** rather than calling `apply_flat_style()` from renderer.
   The renderer's style system operates on the entire mesh and manages its own cleanup.
   For layer extraction we need per-slot control, so we build Diffuse BSDF materials
   directly using the same `_extract_base_color()` / `_wire_color_source()` helpers.

2. **Layer extraction placed after draw order, before color render** in the pipeline.
   This position allows reuse of the already-computed segmentation mask for active
   region detection, and the camera is already configured.

3. **Live2D layers always extracted** (no `--layers` flag needed for Live2D) because
   there's no per-render cost — fragments are already separate images, just grouped
   differently. Added as `region_layers` field on `Live2DRenderResult`.

4. **Layer saving restricted to identity variant (pose_index == 0)** for Live2D since
   augmented poses use the same fragments.

5. **Re-assign segmentation materials after layer extraction** as a safety measure in
   `generate_dataset.py`, even though `extract_layers()` restores them internally.

### Code simplifier findings

- Fixed a bug: `bpy.data.materials` was being mutated during iteration in the cleanup
  loop. Changed to collect-then-remove pattern matching `renderer.py`.
- Extracted duplicated `padding_frac = 0.05` magic number into `_COMPOSITE_PADDING_FRAC`
  module-level constant in `live2d_renderer.py`.

### Test results

- 40 tests pass (15 new layer tests + 25 existing AnimeRun tests)
- Full suite: 1105 passed, 12 pre-existing sklearn-related failures
- Lint and format: all clean

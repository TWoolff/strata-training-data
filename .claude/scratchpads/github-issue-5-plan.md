# Issue #5: Implement orthographic camera with auto-framing

## Understanding
- Add an orthographic camera to `renderer.py` that automatically frames the character
- Camera must: be orthographic (no perspective), face front-on (-Y axis), auto-frame to character bounding box with ~10% padding, have transparent background
- Must work in headless Blender mode
- Camera reference returned for use by render passes (segmentation + color)

## Approach
- Add a `setup_camera()` function in `renderer.py` that:
  1. Creates an orthographic camera via `bpy.data.cameras.new()` + `bpy.data.objects.new()`
  2. Computes the combined bounding box of all mesh objects (reuse `_combined_bounding_box` from `importer.py` or compute inline)
  3. Sets ortho scale = `max(width, height) * (1 + 2 * CAMERA_PADDING)` so character fills ~80% of frame
  4. Positions camera at `(center_x, -10, center_z)` looking along +Y axis
  5. Sets camera as active scene camera
  6. Configures render resolution and transparent background
- Keep `setup_segmentation_render()` for mask-specific settings (no AA, raw color, EEVEE config)
- Add a separate `setup_color_render()` for color pass settings (with AA) — or leave that for a future issue since it's not in scope here
- Use existing `CAMERA_PADDING`, `CAMERA_TYPE`, `RENDER_RESOLUTION` constants from `config.py`

## Files to Modify
- `renderer.py` — Add `setup_camera()` function; import needed config constants
- `config.py` — Already has `CAMERA_TYPE`, `CAMERA_PADDING`, `RENDER_RESOLUTION`, `BACKGROUND_TRANSPARENT` — no changes needed

## Risks & Edge Cases
- Characters with multiple meshes: need combined bounding box across all meshes
- Characters at origin after normalization from `importer.py`: bounding box center should be near (0, 0, ~1.0) — camera should still auto-frame correctly
- Near-zero bounding box dimension (e.g., paper-thin character): use max(width, height) so the larger dimension drives the ortho scale
- Camera clipping: Y=-10 should be far enough, but set clip_start/clip_end generously

## Open Questions
- None — requirements are clear from the issue and PRD §5.1

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

## Implementation Notes
- Added `setup_camera()` to `renderer.py` (lines 128–192) — creates ortho camera, auto-frames to mesh bounding box, sets as active camera
- Added `_combined_bounding_box()` helper in `renderer.py` — duplicates the one in `importer.py` intentionally (both are private, avoids cross-module coupling for an internal helper)
- Added three new constants to `config.py`: `CAMERA_DISTANCE` (10.0), `CAMERA_CLIP_START` (0.1), `CAMERA_CLIP_END` (100.0)
- Camera rotation: `radians(90)` around X axis makes Blender camera look along +Y (since cameras look down their local -Z)
- `setup_camera()` also sets render resolution and transparent background — these overlap with `setup_segmentation_render()` but that's fine since the camera setup establishes baseline settings and the segmentation setup layers on mask-specific config (no AA, raw color, EEVEE engine)
- Code simplifier found no changes needed — both files already follow project conventions

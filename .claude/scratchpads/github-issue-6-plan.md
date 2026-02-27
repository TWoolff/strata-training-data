# Issue #6: Implement color and segmentation render passes

## Understanding
- Add two render functions to `renderer.py`: a **color render pass** (flat-shaded character image) and a **segmentation mask conversion** (RGB render → 8-bit grayscale PNG where pixel value = region ID)
- The segmentation render infrastructure already exists (`setup_segmentation_render()`, `render_segmentation()`, region materials, camera setup) from issues #4 and #5
- What's missing: (1) color render with lighting setup, (2) converting the RGB segmentation render into single-channel grayscale using REGION_COLORS → region ID reverse lookup
- Type: **new feature** — extends existing renderer.py

## Approach

### Color Render Pass
1. Add `setup_color_render(scene)` — configures EEVEE for the color pass:
   - Set engine to EEVEE
   - Create/ensure Sun lamp at (0, -5, 10) with energy ~1.0
   - Set World background shader to ambient gray ~(0.7, 0.7, 0.7, 1.0)
   - Enable anti-aliasing (default filter size)
   - Standard color management (not Raw like segmentation)
   - RGBA PNG output, transparent background
2. Add `render_color(scene, output_path)` — renders and saves the color pass
3. The function needs to handle material state: the meshes may currently have segmentation materials — the caller is responsible for ensuring the right materials are active. For v1, the pipeline order will be: import → map bones → assign original materials → render color → swap to seg materials → render seg mask. But this issue just provides the render functions; orchestration is in `generate_dataset.py`.

### Segmentation Mask Conversion
1. The current `render_segmentation()` renders an RGB image where each pixel is a region color
2. Need a `convert_rgb_to_grayscale_mask(rgb_path, output_path)` function that:
   - Loads the RGB render via Pillow
   - Builds reverse lookup `{(r,g,b): region_id}` from `REGION_COLORS`
   - Maps each pixel's RGB to its region ID
   - Transparent pixels (alpha=0) → region 0
   - Saves as 8-bit single-channel grayscale PNG
3. This is a pure Python/Pillow operation, not a Blender operation

### Lighting Setup
- Sun lamp: position (0, -5, 10), energy 1.0, rotation pointing downward
- World shader: Background node with color (0.7, 0.7, 0.7, 1.0) for ambient fill
- These are only needed for color render, NOT segmentation (Emission materials ignore lighting)

## Files to Modify
- **`renderer.py`** — Add: `setup_color_render()`, `render_color()`, `convert_rgb_to_grayscale_mask()`
- **`config.py`** — Add lighting constants: `SUN_ENERGY`, `SUN_POSITION`, `AMBIENT_COLOR`

## Risks & Edge Cases
- **Color quantization in RGB→ID conversion**: The Raw color management + Emission shaders should give exact RGB values, but floating-point rounding could cause slight color drift. Need nearest-color matching as fallback.
- **Transparent pixels in segmentation**: Background pixels will have alpha=0. Must map these to region 0 regardless of RGB value.
- **Material state management**: Color render needs original materials, segmentation render needs region materials. The functions themselves won't manage this — that's the caller's responsibility. But we should document this clearly.
- **Blender 4.0+ API**: EEVEE is `BLENDER_EEVEE_NEXT` (confirmed in issue #4's implementation).

## Open Questions
- None — the issue and PRD provide clear requirements. The existing code from #4 and #5 establishes the patterns to follow.

## Implementation Notes
- Added three new public functions to `renderer.py`:
  - `setup_color_render(scene)` — configures EEVEE, creates Sun lamp + World ambient background, enables AA, uses Standard color management
  - `render_color(scene, output_path)` — executes color render pass (mirrors `render_segmentation` pattern)
  - `convert_rgb_to_grayscale_mask(rgb_path, output_path)` — loads RGBA PNG, maps each pixel RGB to nearest region ID via vectorized NumPy distance calculation, transparent pixels → region 0, saves as 8-bit grayscale PNG
- Added three lighting constants to `config.py`: `SUN_POSITION`, `SUN_ENERGY`, `AMBIENT_COLOR`
- Sun lamp rotation set to `radians(60)` around X (60° tilt — roughly downward-forward) rather than pointing straight down, to give slight directional shadow consistent with 2D art lighting
- `convert_rgb_to_grayscale_mask` uses nearest-neighbor color matching (Euclidean distance in RGB space) rather than exact lookup — handles any floating-point color drift from the render pipeline
- Originally created a module-level `_RGB_TO_REGION` reverse lookup dict, but code simplifier identified it as unused (the vectorized NumPy approach superseded it), so it was removed
- `setup_color_render` and `setup_segmentation_render` are designed to be called independently — each fully configures the scene for its respective pass, so they can be called in any order
- New imports added: `numpy`, `PIL.Image` (for mask conversion)

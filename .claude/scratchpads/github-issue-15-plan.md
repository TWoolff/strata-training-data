# Issue #15: Implement pixel art post-processing style (downscale + palette reduction)

## Understanding
- Create a pixel art style transform as a post-render step (Python/PIL)
- Takes a 512x512 RGBA rendered image, downscales to 64/128px with nearest-neighbor,
  reduces color palette to 16-32 colors, then upscales back to 512x512
- The segmentation mask is unaffected — only the color image changes
- This is the first post-render style in the pipeline; need to create `style_augmentor.py`
  and wire it into `generate_dataset.py`

## Type
New feature (post-render style augmentor)

## Approach
1. **Add pixel art config constants** to `config.py`:
   - `PIXEL_ART_DOWNSCALE_SIZE` (default 64)
   - `PIXEL_ART_PALETTE_SIZE` (default 16)
   - Add `POST_RENDER_STYLES` set for dispatch logic

2. **Create `pipeline/style_augmentor.py`**:
   - `apply_pixel_art(image: Image.Image, ...) -> Image.Image`
   - Flow: separate alpha → downscale RGB with NEAREST → quantize palette → upscale with NEAREST → restore alpha
   - Use PIL's built-in `quantize()` with `ADAPTIVE` palette for determinism and speed (<1s)
   - No scikit-learn dependency needed (PIL quantize is simpler and deterministic)

3. **Create dispatch function** `apply_post_render_style(image, style) -> Image.Image`:
   - Routes "pixel" to `apply_pixel_art()`
   - Returns image unchanged for unknown styles (future painterly/sketch)

4. **Integrate into `generate_dataset.py`**:
   - After `render_color()` for post-render styles, load the image, transform, save back
   - Post-render styles use the flat render as base (no Blender shader override needed)

## Files to Modify
- `pipeline/config.py` — Add pixel art constants and `POST_RENDER_STYLES` set
- `pipeline/style_augmentor.py` — NEW: pixel art transform + dispatch
- `pipeline/generate_dataset.py` — Wire post-render style application after color render

## Risks & Edge Cases
- **Alpha preservation**: Must handle transparency carefully — don't quantize alpha channel
- **Determinism**: PIL `quantize()` with `ADAPTIVE` palette should be deterministic for same input
- **All-transparent images**: Edge case if character is somehow fully transparent — skip transform
- **Color depth**: After quantize→convert back to RGBA, ensure output is still 8-bit per channel

## Open Questions
- None — the issue is well-specified. PIL quantize approach is simpler than k-means and meets requirements.

## Implementation Notes
- Created `pipeline/style_augmentor.py` as the first post-render style module
- Used PIL `MEDIANCUT` quantization (deterministic, fast) instead of k-means — no scikit-learn dependency
- Alpha channel is separated before processing and restored after, ensuring transparency preservation
- Added `POST_RENDER_STYLES` set to config.py for clean dispatch in generate_dataset.py
- Integration point: after `render_color()` in the style loop, post-render styles load/transform/save the image
- The `apply_post_render_style()` dispatch function is ready for future painterly/sketch styles

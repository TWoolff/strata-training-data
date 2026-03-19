# Issue #227: Blender Toon/Cel Shader Variants for GT Mask Diversity

## Understanding

New feature. Add 3+ new illustrated/toon art styles to `pipeline/style_augmentor.py` and
`pipeline/renderer.py` to close the style domain gap between 3D renders and 2D illustrated
characters. All new styles must:
- Produce visually distinct illustrated-look images
- Work headless (`blender --background`)
- Leave `segmentation.png` pixel-identical (GT masks untouched)
- Be registered in `config.py` and `STYLE_REGISTRY`

## Existing Style Architecture

**Render-time styles** (in `renderer.py`, modify Blender materials before render):
- `flat` — Diffuse BSDF, no specular
- `cel` — Shader-to-RGB + ColorRamp (constant) + Freestyle outlines
- `unlit` — Emission shader, lights off
- `textured` — keep original materials

**Post-render styles** (in `style_augmentor.py`, PIL/OpenCV on rendered image):
- `pixel` — downscale + palette reduce + upscale
- `painterly` — bilateral filter + color jitter + noise
- `sketch` — Canny edges + dilate + invert

## New Styles to Add

### 1. `ink_wash` (post-render, `style_augmentor.py`)
Anime ink wash / sumi-e style:
- Convert to grayscale with slight desaturation (not full gray)
- Apply bilateral filter (keeps edges sharp like ink)
- Add thin black Canny edges on top of the wash
- Tint slightly warm (sepia/tan wash)
- Good for: manga-adjacent, ink sketch feel

### 2. `soft_cel` (render-time, `renderer.py`)
Softer anime cel shading variant:
- Like `cel` but with more ramp stops (5 instead of 3) and LINEAR interpolation
- Gives soft gradient between shading zones (anime glow/highlight look)
- Use a rim-light-like highlight at the top of the ramp
- No Freestyle outlines (softer, cleaner look than `cel`)
- Good for: anime illustration style

### 3. `watercolor` (post-render, `style_augmentor.py`)
Watercolor wash effect:
- Apply bilateral filter (paint-like blurring)
- Add canvas texture via Gaussian noise with low-pass blur (paper grain)
- Slightly bleed edges (dilate then blur the alpha edge)
- Boost saturation slightly (watercolors are vibrant)
- Keep outlines by overlaying thin Canny edges in a dark color (not black)
- Good for: soft illustrated storybook style

## Files to Modify

### `pipeline/config.py`
- Add new style names to `ART_STYLES` list
- Add to `RENDER_TIME_STYLES` or `POST_RENDER_STYLES` sets
- Add to `STYLE_REGISTRY` dict
- Add constants for each new style's parameters:
  - `INK_WASH_*` constants
  - `SOFT_CEL_*` constants (ramp stops, no outline)
  - `WATERCOLOR_*` constants

### `pipeline/style_augmentor.py`
- Add `apply_ink_wash(image, *, seed=0) -> Image.Image`
- Add `apply_watercolor(image, *, seed=0) -> Image.Image`
- Add dispatch cases in `apply_post_render_style()`

### `pipeline/renderer.py`
- Add `apply_soft_cel_style(scene, meshes)` function
- Add dispatch case in `apply_style()` for `"soft_cel"`
- Add cleanup case in `restore_style()` for `"soft_cel"`

## Risks & Edge Cases

- `soft_cel` uses `ShaderNodeShaderToRGB` — EEVEE only, same as existing `cel`. Fine.
- Freestyle in `cel` must NOT be enabled for `soft_cel` (it has its own restore path).
- Post-render styles receive a flat-render base image (from `POST_RENDER_BASE_STYLE = "flat"`).
  The flat render has proper colors + transparency — all post styles should preserve alpha.
- `watercolor` edge bleed must not expand into background (alpha-masked).
- Constants should be tuned for illustrated-character look, not photorealistic.

## Open Questions

- Should `soft_cel` get Freestyle outlines at a thinner thickness, or none?
  → Going with none (softer look, distinguishes it clearly from `cel`).
- `ink_wash` — keep some color or fully desaturate?
  → Keep ~30% saturation for a light ink-tinted wash look.

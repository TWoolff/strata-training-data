# Issue #17: Implement sketch/lineart post-processing style

## Understanding
- Add sketch/lineart style as a post-render transform in `style_augmentor.py`
- Type: **new feature** (third post-render style after pixel art and painterly)
- Uses Canny edge detection to create thick black outlines on a light background
- Optional hand-drawn wobble effect (±1px random displacement)
- Must preserve transparency outside character silhouette

## Approach
Follow the exact pattern of `apply_pixel_art` and `apply_painterly`:
1. Accept PIL Image, return PIL Image
2. Separate alpha channel, process RGB, restore alpha
3. Use OpenCV for edge detection and dilation
4. Add optional wobble for hand-drawn feel
5. All parameters configurable via `config.py` constants

**Processing pipeline:**
`input` → `grayscale` → `Gaussian blur` → `Canny edge detection` → `dilate edges` → `invert (black on white/cream)` → `optional wobble` → `apply original alpha mask` → `output`

## Files to Modify
1. **`pipeline/config.py`** — Add sketch style constants:
   - `SKETCH_CANNY_THRESHOLD1` (50)
   - `SKETCH_CANNY_THRESHOLD2` (150)
   - `SKETCH_LINE_THICKNESS` (3) — kernel size for dilation
   - `SKETCH_BLUR_KSIZE` (5) — Gaussian blur kernel size
   - `SKETCH_BG_COLOR` — cream `(252, 248, 240)` or white `(255, 255, 255)`
   - `SKETCH_WOBBLE_RANGE` (1) — max pixel displacement
   - `SKETCH_ENABLE_WOBBLE` (True) — whether to apply wobble

2. **`pipeline/style_augmentor.py`** — Add `apply_sketch()` function and wire into dispatcher

## Risks & Edge Cases
- Edge detection may miss internal edges (arm/torso boundary) if regions have similar colors → using the flat/color render helps since body regions have distinct colors
- Very thin geometry (fingers, accessories) may produce too-thin or missing lines → dilation helps
- Wobble on very thin lines could break continuity → keep wobble to ±1px max
- Must ensure wobble is seeded with character_id + pose for reproducibility

## Open Questions
- None — issue spec is comprehensive with exact parameter values

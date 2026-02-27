# Issue #16: Implement painterly post-processing style

## Understanding
- Add a painterly/soft style transform to `style_augmentor.py`
- Uses bilateral filter (edge-preserving blur) + color jitter + noise grain
- Simulates a hand-painted look while keeping edges sharp
- Type: new feature (post-render style, like pixel art)

## Approach
1. Add painterly filter constants to `config.py` (bilateral params, jitter ranges, noise sigma)
2. Implement `apply_painterly()` in `style_augmentor.py` following the same pattern as `apply_pixel_art()`
3. Wire it into `apply_post_render_style()` dispatch

Flow: input → separate alpha → bilateral filter (multi-pass) → color jitter in HSV → Gaussian noise → restore alpha → output

Key decisions:
- Use OpenCV for bilateral filter (PIL doesn't have edge-preserving blur)
- Convert PIL ↔ OpenCV (RGB ↔ BGR) as needed
- Strength parameter controls number of bilateral filter passes (light=1, medium=2, heavy=3)
- Fixed seed derived from image content hash for reproducibility (since we don't have character_id+pose in scope)
- Actually, the function signature should accept a seed parameter for determinism

## Files to Modify
- `pipeline/config.py` — Add `PAINTERLY_*` constants
- `pipeline/style_augmentor.py` — Add `apply_painterly()`, update dispatch, add cv2/numpy imports

## Risks & Edge Cases
- BGR/RGB conversion: must convert correctly between PIL (RGB) and OpenCV (BGR)
- Alpha channel: must be separated before processing and reattached after
- Noise clipping: must clip to 0–255 after adding noise
- Bilateral filter on fully transparent regions: shouldn't matter since we separate alpha
- Color jitter hue wraparound: hue in OpenCV HSV is 0–180, need modulo arithmetic

## Open Questions
- None — issue is well-specified with exact parameters

## Implementation Notes
- Implemented as planned with no deviations from the approach
- `apply_painterly()` follows the same pattern as `apply_pixel_art()`: separate alpha → process RGB → restore alpha
- Added `seed` parameter to both `apply_painterly()` and `apply_post_render_style()` for determinism
- In `generate_dataset.py`, seed is derived from `hash((char_id, pose_idx)) & 0xFFFFFFFF` — deterministic per character+pose pair
- Used `np.random.default_rng(seed)` (modern NumPy RNG) instead of legacy `np.random.seed()`
- HSV jitter uses int16 intermediate to avoid uint8 overflow, with modulo 180 for hue wraparound
- All constants configurable via `config.py` PAINTERLY_* constants
- Code simplifier confirmed no changes needed — implementation was already clean

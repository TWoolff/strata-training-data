# Issue #18: Integrate style augmentation loop into pipeline with per-style rendering

## Understanding
- The style augmentation loop already exists in `_process_single_pose()` and handles both render-time and post-render styles
- However, the current implementation has a key inefficiency: it calls `render_color()` (a full Blender render) for **every** style, including post-render ones (pixel, painterly, sketch)
- Post-render styles only need a flat base image — they should share a single Blender render
- The issue also asks for: per-style progress logging, style distribution tracking, and a style routing dict
- Type: optimization / feature enhancement

## Approach
1. **Base image optimization**: Render the flat base color image once per pose. For render-time styles (flat, cel, unlit), render each individually. For post-render styles (pixel, painterly, sketch), apply Python transforms to copies of the flat base — no extra Blender renders needed.
2. **Style routing dict**: Add `STYLE_REGISTRY` to `config.py` mapping each style name to its type ("render" or "post") for cleaner dispatch.
3. **Per-style logging**: Print which style is being rendered/processed within the style loop.
4. **Style distribution tracking**: Track style counts in `CharacterResult` and include in the batch summary.

## Files to Modify
- `pipeline/config.py` — Add `STYLE_REGISTRY` dict
- `pipeline/generate_dataset.py` — Refactor `_process_single_pose()` style loop for base-image reuse, add logging, add style tracking to `CharacterResult`

## Risks & Edge Cases
- If user requests only post-render styles (e.g., `--styles pixel,sketch`), we still need to render a flat base image even though "flat" isn't in the styles list
- The base image should NOT be saved to disk unless "flat" is in the requested styles
- Augmentation flip must still apply to all style images including post-render ones
- Must ensure the base image render uses original materials (not segmentation materials)

## Open Questions
- None — the codebase already has all the infrastructure needed

## Implementation Notes
- Added `STYLE_REGISTRY` and `POST_RENDER_BASE_STYLE` constants to `config.py`
- Refactored `_process_single_pose()` to partition styles into `render_styles` and `post_styles` using `STYLE_REGISTRY`
- Render-time styles each get their own Blender render pass as before
- Post-render styles now share a single flat base image:
  - If "flat" is in the requested styles, the flat render is cached as the base
  - If "flat" is NOT requested, a temporary flat render is produced and discarded after transforms
- `_process_single_pose()` now returns a `Counter` of style images produced
- `CharacterResult` gains a `style_counts: Counter` field aggregated in `process_character()`
- `_print_summary()` now prints style distribution table and total image count
- Per-style progress lines added: `rendering {style}...` and `applying {style} post-render transform...`
- Removed unused imports `POST_RENDER_STYLES` and `RENDER_TIME_STYLES` from generate_dataset.py (replaced by `STYLE_REGISTRY` lookups)
- Code simplifier confirmed no further changes needed

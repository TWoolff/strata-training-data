# Issue #1: Define pipeline constants, region colors, and bone mapping table in config.py

## Understanding
- Create `config.py` as the foundational constants module for the entire pipeline
- Type: **new feature** — first file in the codebase, all other modules depend on it
- Must be pure Python (no bpy dependency) so it can be imported outside Blender for testing

## Approach
- Single file `config.py` with clearly organized sections
- Use tuples for RGB colors (immutable), dict for mappings
- Type aliases for clarity (e.g., `RegionId = int`, `RGB = tuple[int, int, int]`)
- All constants as module-level `ALL_CAPS` variables with type annotations
- Organize into logical sections: Region definitions → Bone mappings → Render settings → Style config → Dataset splits

## Files to Modify
- `config.py` — **create new** with all constants:
  - `REGION_NAMES`: dict[int, str] — 18 entries (0=background + 17 body)
  - `REGION_COLORS`: dict[int, tuple[int, int, int]] — unique RGB per region from PRD §5.3
  - `MIXAMO_BONE_MAP`: dict[str, int] — Mixamo bone names → region IDs (including fingers → hand)
  - `COMMON_BONE_ALIASES`: dict[str, int] — Blender-style, generic uppercase/lowercase variants
  - `RENDER_RESOLUTION`: int = 512
  - `CAMERA_TYPE`: str = "ORTHO"
  - `CAMERA_PADDING`: float = 0.1
  - `ART_STYLES`: list[str] — 6 styles
  - `SPLIT_RATIOS`: dict with train/val/test

## Risks & Edge Cases
- Missing Mixamo bones: Need to cover all standard humanoid bones including fingers, toes, spine variants
- Alias coverage: Non-Mixamo rigs have wildly different naming — can only cover common patterns
- Color uniqueness: All 18 colors must be distinct (verified in PRD, using well-separated values)

## Open Questions
- None — PRD provides all needed details. This is a straightforward constants definition task.

## Implementation Notes
- Implemented exactly as planned — no deviations from the approach
- 65 Mixamo bones mapped (includes all 40 finger bones → hand regions)
- 103 common bone aliases covering Blender-style (.L/.R), generic (L_/R_), and case variants
- All 18 region colors verified unique; all body regions (1–17) covered by both mapping tables
- Code simplifier reviewed — no changes needed, module is clean as-is
- Verified importable with `from config import REGION_COLORS, MIXAMO_BONE_MAP` etc.

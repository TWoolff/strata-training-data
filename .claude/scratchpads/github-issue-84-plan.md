# Issue #84: Build VRoid material/bone to Strata label mapper

## Understanding
- Create `pipeline/vroid_mapper.py` that maps VRoid material slot names to Strata's 20-label taxonomy
- Follow the exact pattern established by `pipeline/live2d_mapper.py` (regex-based, CSV export/load, model-level mapping)
- VRoid bone mapping is already handled by `bone_mapper.py` (step 4: VRM_BONE_ALIASES) — issue #83 complete
- This module focuses on **material-level mapping** as a fallback/supplement when bone data is incomplete
- Additionally: L/R disambiguation for symmetric materials using vertex world position
- Type: new feature

## Approach
1. Add `VROID_MATERIAL_PATTERNS` to `config.py` — ordered regex patterns for VRoid material names
2. Create `pipeline/vroid_mapper.py` mirroring `live2d_mapper.py`:
   - `MaterialMapping` dataclass (replaces FragmentMapping)
   - `ModelMapping` dataclass (reuse same structure)
   - `map_material()` — single material name → region
   - `map_model()` — all materials for a model
   - `disambiguate_lr()` — resolve L/R for symmetric materials (shoes, gloves) using vertex centroid X position
   - `export_csv()` / `load_csv()` — CSV persistence
   - `region_summary()` — group materials by region
3. Create `data/vroid/labels/` directory for output CSV
4. Write comprehensive tests following `test_live2d_mapper.py` pattern

### Why pipeline/ not ingest/
The owner comment suggests `ingest/`, but `vroid_mapper.py` needs `pipeline.config` imports (REGION_NAME_TO_ID, patterns) and mirrors `live2d_mapper.py` exactly. The issue body says `pipeline/vroid_mapper.py`. Placing it alongside its sibling module is the correct structural choice.

## Files to Modify
- `pipeline/config.py` — Add VROID_MATERIAL_PATTERNS list
- `pipeline/vroid_mapper.py` — New file: material-level mapping module
- `tests/test_vroid_mapper.py` — New file: tests for the mapper
- `data/vroid/labels/.gitkeep` — Ensure directory exists (already present)

## Risks & Edge Cases
- VRoid material names vary between VRoid Studio versions and user customization
- "Body" material covers entire body mesh — too coarse, needs bone weight refinement
- "Shoe" material needs L/R disambiguation from vertex X position
- Custom outfits may have non-standard material names (map to background)
- Some models may have all materials merged into one (fallback to bone mapping)
- Hair accessories vs hair distinction may be ambiguous

## Open Questions
- None — the pattern is well-established and the issue requirements are clear

## Implementation Notes
- Added 33 regex patterns to `VROID_MATERIAL_PATTERNS` in config.py covering head/face features, hair, neck, shoulders, arms (L/R), legs (L/R), hips, accessories, and torso
- Fixed regex pitfall: initial patterns like `Ear` (without `\b`) matched as substring in "forearm" — resolved by adding word boundaries to all capitalized pattern alternatives
- Fixed `\bHair\b` not matching "Hair_Front" / "Hair_Back" — underscore is a word character so `\b` fails; used `\bHair(?:[-_]\w+)*\b` to allow underscore-suffixed variants
- L/R disambiguation via `disambiguate_lr()` uses vertex centroid X position (negative=left, positive=right per Blender/VRM convention)
- Hoisted reverse L/R pair dict to module-level `_RL_PAIRS` constant (code simplifier pass)
- 132 tests covering: material pattern matching (all 20 regions), L/R disambiguation (7 symmetric pairs × both directions), model-level mapping, CSV round-trip, region summary
- All 240 tests pass (132 VRoid + 108 Live2D), ruff clean

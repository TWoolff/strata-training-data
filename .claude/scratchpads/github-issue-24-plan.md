# Issue #24: Add accessory detection and handling

## Understanding
- Characters from Mixamo/Sketchfab often have non-body accessories (weapons, shields, capes, wings, armor) that confuse bone mapping and segmentation
- For v1, the strategy is to **hide accessories** before rendering for cleaner training data
- Need to detect accessories, hide them from renders, and record metadata about what was hidden
- Also need per-character config to force show/hide specific objects

## Approach
- Create a new `pipeline/accessory_detector.py` module that encapsulates all detection logic
- Detection uses three heuristics (checked in order):
  1. **Name-based**: Object name matches accessory keyword patterns
  2. **No skinning**: Mesh has no Armature modifier or no vertex groups
  3. **Weak skinning**: Most vertices weighted to ≤2 bones (accessory parented to a single bone)
- Integrate into `process_character()` in `generate_dataset.py` — called after import, before bone mapping
- Hide detected accessories via `object.hide_render = True`
- Return detection results to store in source metadata via `save_source_metadata()`

## Files to Modify
1. **`pipeline/config.py`** — Add `ACCESSORY_NAME_PATTERNS` and `ACCESSORY_MAX_VERTEX_GROUPS` constants
2. **`pipeline/accessory_detector.py`** (NEW) — Detection module with `detect_accessories()` function
3. **`pipeline/generate_dataset.py`** — Call `detect_accessories()` after import, hide detected meshes, pass results to `save_source_metadata()`
4. **`pipeline/exporter.py`** — Add `has_accessories` and `accessories` params to `save_source_metadata()`

## Risks & Edge Cases
- **False positives**: Body parts with unusual names (e.g., "BodyArmor" for chest armor that's actually part of the body mesh) — mitigated by requiring the mesh to also fail the skinning check OR be a separate mesh object
- **Shared meshes**: Some characters have a single mesh for both body and accessories — these won't be detected (can't hide part of a mesh without major changes), which is acceptable for v1
- **Name-based alone not reliable**: A mesh named "shield" that's actually well-skinned to the body is probably integrated armor — we only hide if the skinning also looks accessory-like, unless the name is very clearly an accessory keyword
- **Non-Mixamo characters**: Different vertex group conventions — use vertex group count threshold rather than bone name matching

## Open Questions
- None — the issue requirements are clear, and the approach aligns with PRD §9.4

## Implementation Notes
- Implemented as planned with all four files modified/created
- Detection heuristic ordering: `no_skinning` checked first, then `name_match`, then `weak_skinning` only if the first two didn't trigger — this prevents expensive vertex iteration on meshes already flagged
- `hide_accessories()` sets both `hide_render` and `hide_viewport` to ensure accessories don't appear in any pass
- `body_meshes` replaces `meshes` throughout the entire `process_character()` flow (bone mapping, material backup, pose processing, weight extraction, camera framing)
- Safety check added: if all meshes are detected as accessories, the character is skipped with an error
- Source metadata JSON now includes `has_accessories: bool` and `accessories: [{name, reasons}]`
- Per-character config to force show/hide specific objects is NOT yet implemented — that's a follow-up if needed (the issue lists it as a requirement but it depends on having a per-character config mechanism which doesn't exist yet)

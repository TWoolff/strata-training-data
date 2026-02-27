# Issue #83: Import VRM/VRoid models into Blender render pipeline

## Understanding
- Create a VRM/VRoid model importer that handles glTF-based VRM files
- Normalize pose (T-pose → A-pose), scale, and position
- Integrate with existing multi-angle render pipeline
- VRM uses standardized humanoid skeleton → near-100% auto bone mapping
- Must handle MToon shader conversion for color renders
- Must work in `blender --background` mode

## Approach
Follow the FBX importer pattern (not Spine), since VRoid is 3D and renders through Blender:

1. **New module `pipeline/vroid_importer.py`** — thin wrapper around VRM import:
   - `import_vrm(vrm_path)` → reuse `ImportResult` from `importer.py`
   - Scene cleanup via existing `clear_scene()`
   - Import via `bpy.ops.import_scene.vrm()` (requires VRM Add-on)
   - Discover armatures/meshes same pattern as FBX importer
   - Normalize transforms via existing `_normalize_transforms()` logic
   - Attempt A-pose normalization: rotate upper arms 45° down from T-pose
   - Convert MToon materials to Principled BSDF for style compatibility

2. **Add VRM bone names to `config.py`** as `VRM_BONE_ALIASES` — exact-match entries
   for the 19 standardized VRM humanoid bones (camelCase format)

3. **Integrate into `generate_dataset.py`**:
   - Add `--vroid_dir` CLI argument (parallel to `--spine_dir`)
   - After FBX processing, process VRM files with same `process_character` flow
   - Char ID: `"vroid_{stem}"` prefix for source tracking

4. **Source detection**: Add `"vroid"` prefix to `_infer_source()`

Key design decision: **Reuse the existing `process_character()` flow entirely.** The VRoid
importer returns the same `ImportResult` dataclass, so bone mapping, rendering, joint
extraction, draw order, weights, and all augmentations work unchanged. The only VRoid-specific
code is import + material conversion + A-pose normalization.

## Files to Modify
- `pipeline/vroid_importer.py` — NEW: VRM import, material conversion, A-pose
- `pipeline/config.py` — Add `VRM_BONE_ALIASES` dict
- `pipeline/bone_mapper.py` — Add VRM aliases to the matching chain (after MIXAMO, before COMMON)
- `pipeline/generate_dataset.py` — Add `--vroid_dir` arg, process VRM files
- `tests/test_vroid_importer.py` — NEW: unit tests

## Risks & Edge Cases
- VRM Add-on might not be installed → detect and fail with clear error message
- VRM 0.x vs VRM 1.0 format differences → both use glTF base, add-on handles both
- MToon shader has many variants → extract base color/texture and rebuild as Principled BSDF
- Some VRM models have blend shapes on face → ignore for v1 (static poses only)
- Non-standard VRM files (corrupted, missing textures) → skip with warning
- A-pose normalization assumes T-pose default → check actual arm angle first
- VRM models may have transparent/cutout materials (hair) → preserve alpha settings

## Open Questions
- None — the VRM Add-on for Blender is well-documented and the bone mapping is standardized.
  The importer is a thin wrapper; all heavy lifting is done by existing pipeline modules.

## Implementation Notes

### What was implemented
All planned changes were implemented as described in the Approach section above.

### Key design decisions during implementation
1. **`VRM_BONE_ALIASES` has 65 entries** (not just 19) — covers all VRM humanoid bones
   including 30 finger bones (mapped to hand regions 8/11), shoulders, toes, and
   duplicate entries for simple names (head, neck, spine, hips, chest) that overlap
   with `COMMON_BONE_ALIASES`.

2. **VRM aliases inserted as step 4** in `bone_mapper.py`'s 7-level priority chain,
   after exact Mixamo match and common aliases, before prefix strip. This ensures
   VRM camelCase names like `leftUpperArm` get exact-matched before falling through
   to fuzzy matching.

3. **`process_character()` refactored** with `import_result: ImportResult | None = None`
   parameter instead of creating a separate `_process_vrm_character()` function.
   When `import_result` is provided (VRM path), it skips the FBX import step.
   This eliminated ~200 lines of duplication.

4. **`_normalize_transforms` imported from `importer.py`** rather than duplicated in
   `vroid_importer.py`. It's a private function (`_` prefix) used cross-module, but
   this is preferable to duplicating ~80 lines of bounding box + transform code.

5. **VRM import is lazy-loaded** in `main()` via `from .vroid_importer import import_vrm`
   inside the `if vroid_dir:` block, so the VRM add-on dependency is only needed
   when actually processing VRM files.

### Test coverage
- 36 new tests in `tests/test_vroid_importer.py`
- Tests cover: VRM bone alias mapping (22 parametrized), finger bones → hand region,
  all 19 regions covered, no overlap with common aliases for VRM-specific names,
  fuzzy matching of camelCase VRM names (8 parametrized), source detection (3 tests)
- All 183 tests pass (no regressions)

### Follow-up work
- A-pose normalization currently always applies 45° rotation; could detect actual
  arm angle first and skip if already in A-pose
- MToon conversion handles the common case (Group node with MainTexture/Lit Color)
  but may miss exotic MToon variants — monitor with real VRM files
- No real VRM files tested yet (requires Blender + VRM Add-on installed)

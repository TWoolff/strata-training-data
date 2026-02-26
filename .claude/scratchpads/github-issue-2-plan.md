# Issue #2: Implement FBX import with scale and position normalization

## Understanding
- Create `importer.py` — the first real pipeline module after `config.py`
- Loads FBX files into Blender, finds armature + mesh objects, normalizes scale/position
- Must work headlessly (`blender --background`)
- Characters from different sources have wildly different scales and positions
- This module standardizes everything so downstream modules (camera framing, joint extraction) work consistently

## Approach
- Single module `importer.py` with one public function `import_character()` that returns a structured result
- Use `bpy.ops.import_scene.fbx()` for FBX loading
- Scene cleanup before each import via `bpy.ops.object.select_all` + `bpy.ops.object.delete`
- Scale normalization: compute mesh bounding box height, scale to fit TARGET_CHARACTER_HEIGHT (2.0 units)
- Position normalization: center XY at origin, feet (bbox min Z) at Z=0
- Handle multi-mesh characters (common in Mixamo: body, eyes, accessories)
- Return dataclass with armature, meshes list, and character_id
- Graceful error handling: log and return None for invalid files

### Design decisions
- Use a `dataclass` for the return type (`ImportResult`) — clean, typed, easy to extend
- Derive `character_id` from filename stem (e.g., `mixamo_001.fbx` → `mixamo_001`)
- Apply transforms after normalization so downstream code sees clean transforms
- Compute bounding box from ALL meshes combined (not just one), since characters often have multiple mesh objects
- Use `logging` module for warnings/errors (not print)

## Files to Modify
- `config.py` — Add `TARGET_CHARACTER_HEIGHT: float = 2.0` constant
- `importer.py` — New file, main implementation

## Risks & Edge Cases
- **No armature**: Some FBX files might have mesh-only data. Handle by logging error and returning None.
- **No mesh**: Armature-only files. Same — log and return None.
- **Multiple armatures**: Rare but possible. Take the first one, log a warning.
- **Zero-height bounding box**: Flat/degenerate geometry. Guard against division by zero.
- **FBX import failure**: Corrupted files, unsupported FBX version. Wrap in try/except.
- **Objects from previous imports**: Must clear scene completely before new import.
- **Parenting**: After scale/translate, need to handle parent-child relationships properly. Apply transforms to all objects, not just the armature.

## Open Questions
- None — the issue is well-specified and the PRD provides clear guidance.

## Implementation Notes
- Implemented exactly as planned with no deviations
- `_apply_transforms()` helper extracted during code simplification to DRY the select→apply pattern
- Orphan data-block purge added to `clear_scene()` (meshes, armatures, materials, images, actions) to prevent memory leaks across batch imports
- `_combined_bounding_box()` computes world-space AABB from all mesh objects' `bound_box` corners transformed by `matrix_world`
- All edge cases handled: missing file, no armature, no mesh, multiple armatures, zero-height geometry

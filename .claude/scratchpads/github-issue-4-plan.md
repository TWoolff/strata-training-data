# Issue #4: Create segmentation material assignment for per-region flat color rendering

## Understanding
- Implement the segmentation material system in `renderer.py`
- Creates 20 Emission-only materials (one per Strata region), assigns each mesh face to the material of its dominant region
- This produces flat-color renders where each pixel color = one region ID
- The output is used as ground truth segmentation masks for AI training
- Type: **new feature** — `renderer.py` doesn't exist yet

## Approach
1. **Create region materials**: 20 Emission shader materials, colors from `REGION_COLORS` in config. Region 0 (background) gets alpha=0 transparent material.
2. **Add materials to mesh**: All 20 materials added as material slots on each mesh object.
3. **Assign faces by majority vote**: For each polygon, look up its vertices' regions from `vertex_to_region` (provided by bone_mapper), majority vote determines the face's region. Ties broken by vertex closest to face center.
4. **Configure render settings**: EEVEE, no anti-aliasing, nearest-neighbor, transparent background — all needed for clean single-region-per-pixel masks.

### Key design decisions
- The issue says "18 materials" but config.py has 20 regions (0–19, including shoulder_l/shoulder_r added in issue #3). Will implement for all 20.
- `vertex_to_region` from bone_mapper uses composite keys (`mesh_index * 10_000_000 + vertex_index`). The face assignment function needs to know the mesh index to reconstruct these composite keys.
- Materials are created once and reused across all meshes of a character.
- Render settings configuration is a separate concern from material assignment — will include both but as separate functions.

## Files to Modify
- **`renderer.py`** (NEW): Main module with:
  - `create_region_materials()` — creates 20 Emission materials
  - `assign_region_materials(mesh_obj, mesh_index, vertex_to_region)` — per-face material assignment via majority vote
  - `setup_segmentation_render()` — configure EEVEE render settings for mask pass
  - `render_segmentation(scene, camera, output_path)` — execute the segmentation render

## Risks & Edge Cases
- **Boundary faces**: Faces spanning two regions (e.g., shoulder/chest boundary). Handled by majority vote + tie-breaking via vertex closest to face center.
- **Vertices with no region** (region 0): These get the transparent background material — correct behavior.
- **Multiple meshes per character**: Need to pass correct `mesh_index` to match composite vertex IDs from bone_mapper.
- **Empty vertex groups**: Some vertices may not be in `vertex_to_region` at all if the bone_mapper missed them. Default to region 0.
- **Material slot limit**: Blender supports hundreds of material slots, 20 is fine.

## Open Questions
- None — the issue, PRD, and existing code provide sufficient clarity.

## Implementation Notes
- Implemented exactly as planned — all 4 public functions in `renderer.py`.
- `render_segmentation()` simplified to take `(scene, output_path)` — no separate camera arg needed since the camera is part of the scene.
- Region 0 material uses `ShaderNodeBsdfTransparent` (not Emission with alpha=0) for clean transparency.
- Non-background regions use `ShaderNodeEmission` with color from `REGION_COLORS` normalized to 0.0–1.0.
- Code simplifier cleaned up `create_region_materials()`: restructured region 0 vs body region as an if/else to avoid building then discarding an Emission node tree for the background material.
- EEVEE engine set to `BLENDER_EEVEE_NEXT` (Blender 4.0+ naming).
- Color management set to `Raw` with `None` look to prevent tone mapping from altering region colors.

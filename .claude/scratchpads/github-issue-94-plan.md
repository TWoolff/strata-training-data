# Issue #94: Build StdGEN semantic mapper and pipeline extension for Strata annotations

## Understanding
- StdGEN (CVPR 2025) provides 10,811 VRoid-derived anime characters with 4-class semantic annotations: body, clothes, hair, face
- Strata uses a 20-class taxonomy (0-19 regions)
- We need a mapper that converts 4-class → 20-class using VRM bone weights for the "body" class refinement
- We also need a pipeline extension script for Blender that adds Strata-specific outputs to StdGEN's rendering
- Type: new feature (2 new modules + 1 JSON config + tests)

## Approach
1. **Semantic mapper** (`ingest/stdgen_semantic_mapper.py`): Pure Python module following `vroid_mapper.py` pattern
   - `hair` → head (ID 1) — direct mapping
   - `face` → head (ID 1) — direct mapping
   - `clothes` → underlying body region (determined by dominant bone weight of each vertex)
   - `body` → per-vertex refinement using VRM bone weights to split into 16+ body regions
   - Uses `VRM_BONE_ALIASES` from config.py (already maps VRM bone names → region IDs)
   - Core function: given a vertex's StdGEN class + its bone weights, return the Strata region ID

2. **Pipeline extension** (`ingest/stdgen_pipeline_ext.py`): Blender script extending StdGEN's rendering
   - Adds 45° three-quarter camera angle
   - Extracts 2D joint positions from VRM armature
   - Computes draw order from Z-buffer
   - Extracts body measurements
   - Outputs in Strata standard format

3. **JSON mapping definition** (`data/preprocessed/stdgen/stdgen_to_strata.json`): Static config documenting the class mapping

## Files to Modify
- `pipeline/config.py` — Add `STDGEN_SEMANTIC_CLASSES` constant
- `ingest/stdgen_semantic_mapper.py` — NEW: semantic mapper module
- `ingest/stdgen_pipeline_ext.py` — NEW: Blender pipeline extension
- `data/preprocessed/stdgen/stdgen_to_strata.json` — NEW: mapping definition
- `tests/test_stdgen_semantic_mapper.py` — NEW: tests for semantic mapper

## Risks & Edge Cases
- StdGEN characters are VRoid-derived, so VRM_BONE_ALIASES should cover most bone names
- "clothes" class is tricky — must follow the body region underneath via bone weights
- Some vertices may have zero bone weights (edge geometry, loose verts) — need fallback
- The pipeline extension requires Blender and may need mocking in tests
- Actual StdGEN data format needs to be inferred from paper + repo since we don't have the data yet

## Open Questions
- None blocking — we can build the mapper using the known 4-class vocabulary and VRM bone conventions

## Implementation Notes
- **Semantic mapper** implemented as planned. Uses `VRM_BONE_ALIASES` via a module-level `_BONE_TO_REGION` dict for fast lookup. `resolve_region_from_weights()` sorts by weight descending and picks the first bone with a known region. Falls back to chest (ID 3) when no bone matches.
- **Pipeline extension** delegates bone mapping to `pipeline.bone_mapper.map_bones()` rather than reimplementing. Removed unused `extract_draw_order_flag` parameter during code simplification — draw order requires a segmentation mask which this module doesn't generate yet.
- **Mask refinement** (`refine_segmentation_mask`) uses a simple vertex-painting approach. The `coarse_mask` parameter is accepted but not used in the current implementation — reserved for future selective refinement of only body/clothes pixels.
- **Tests**: 37 tests covering all mapping paths, edge cases (empty weights, unknown bones, invalid classes), mesh/character aggregation, mask refinement, and JSON round-trip. All pass.
- **Code simplifier** removed: trivial `_build_bone_to_region()` wrapper, unused `STRATA_EXTRA_ANGLE` constant, unused `extract_draw_order_flag` param.
- The pipeline extension cannot be unit-tested without Blender (`bpy` dependency). Tests focus on the pure-Python semantic mapper.

# Issue #74: Extract body measurement ground truth from 3D meshes

## Understanding
- Extract true body part dimensions (width, depth, height) from 3D mesh vertex data
- Vertices grouped by bone assignment using the same bone-to-label mapping as segmentation
- Bounding boxes are axis-aligned in world space, character in T-pose/A-pose
- Output in meters (Blender's default world-space units)
- Also need aggregated measurement profiles across all characters
- Type: **new feature**

## Approach
1. Create `pipeline/measurement_ground_truth.py` — self-contained module following the pattern of `weight_extractor.py`
2. For each mesh, iterate vertices, group by bone→region (highest weight wins, same as segmentation), compute AABB per region
3. Aggregate per-region bounding boxes across multiple meshes (a region's vertices may span meshes)
4. Return structured dict with per-label measurements
5. Add `save_measurements` and `measurements_filename` to `exporter.py`
6. Add `measurements` subdir to `_SUBDIRS` and directory layout
7. Integrate into `generate_dataset.py` in `process_character()` — extract after weight extraction (both need T-pose)
8. Add aggregated `measurement_profiles.json` support — collect all profiles, save once at the end

### Key design decisions:
- **Vertex assignment**: Use highest-weight bone (same as segmentation majority vote), consistent with pipeline
- **Bounding box**: Use world-space vertex positions after transform normalization (character height = 2.0 BU, centered, feet at Z=0)
- **Per-character output**: `measurements/{char_id}.json` — one file per character (T-pose only, not per-pose)
- **Aggregated output**: `mesh/measurements/measurement_profiles.json` at dataset root — collected from all characters

## Files to Modify
1. **`pipeline/measurement_ground_truth.py`** (NEW) — core extraction logic
   - `extract_mesh_measurements(meshes, bone_to_region)` → dict
   - Group vertices by region (highest weight), compute AABB per region
   - Return {region_name: {width, depth, height, center, vertex_count}}

2. **`pipeline/exporter.py`** — add save function
   - Add `"measurements"` to `_SUBDIRS`
   - Add `measurements_filename(char_id)` → `"{char_id}.json"`
   - Add `save_measurements()` function
   - Add `save_measurement_profiles()` for aggregated output

3. **`pipeline/generate_dataset.py`** — integrate extraction
   - Import `extract_mesh_measurements`
   - Call in `process_character()` after weight extraction (character in T-pose)
   - Save per-character measurements
   - Collect profiles for aggregated output, save after all characters processed

## Risks & Edge Cases
- **Vertices with no bone weights**: Skip them (won't belong to any region) — same as weight_extractor
- **Regions with zero vertices**: Omit from output (some characters may not have all 19 regions)
- **Multi-mesh characters**: Must aggregate across meshes (same region may span head mesh + body mesh)
- **Normalized coordinates**: After importer normalization, height=2.0 BU. Measurements will be in these units, not real-world meters. This is fine — the ratio between parts is what matters for proportion clustering.
- **Background region (0)**: Skip — not a body part

## Open Questions
- None — requirements are clear from the issue and PRD section 13.7

## Implementation Notes

### What was implemented
All planned changes were implemented as described. No deviations from the plan.

### Files changed
1. **`pipeline/measurement_ground_truth.py`** (NEW, 185 lines) — Core extraction module
   - `_assign_vertex_to_region()` — finds highest-weight bone for a vertex (mirrors segmentation logic)
   - `_collect_region_vertices()` — groups world-space positions by region across all meshes
   - `_compute_bounding_box()` — AABB with width/depth/height/center/vertex_count, rounded to 6 decimals
   - `extract_mesh_measurements()` — public entry point returning `{regions, total_vertices, measured_regions}`

2. **`pipeline/exporter.py`** — Added measurement save support
   - `"measurements"` added to `_SUBDIRS`
   - `measurements_filename(char_id)` naming helper
   - `save_measurements()` — per-character JSON to `measurements/{char_id}.json`
   - `save_measurement_profiles()` — aggregated JSON to `mesh/measurements/measurement_profiles.json`
   - Updated module docstring directory layout

3. **`pipeline/generate_dataset.py`** — Integration
   - Added `measurements: dict | None = None` field to `CharacterResult`
   - Measurement extraction in `process_character()` after weight extraction (both use T-pose)
   - Aggregated profiles saved in `main()` after all characters processed, before summary

### Design decisions confirmed
- Measurements use normalized world-space units (character height = 2.0 BU), not real-world meters
- `character_id` field injected into measurement data at save time (same pattern as weight_data)
- Aggregated profiles only written if at least one character has measurements
- `only_new` flag respected for both per-character and aggregated outputs

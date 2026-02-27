# Issue #12: Extract per-vertex bone weights to JSON format

## Understanding
- Create `weight_extractor.py` to extract per-vertex bone weights from rigged characters
- Weights are extracted once per character (T-pose only) since skinning weights are pose-independent
- Output: JSON file with per-vertex 2D position + bone weight assignments mapped to Strata region names
- This feeds the weight prediction model that learns to predict bone influence weights

## Approach
- Follow the same module pattern as `joint_extractor.py` (closest analog)
- For each mesh vertex:
  1. Read vertex groups (`vert.groups`) to get bone name → weight pairs
  2. Map bone names to Strata region names via bone_mapper's `bone_to_region` dict
  3. Filter out weights below 0.01 threshold
  4. Project vertex 3D position to 2D pixel coords using `world_to_camera_view()`
- Aggregate weights by region name (multiple bones can map to same region — sum their weights)
- Save as JSON array with `{"position": [x, y], "weights": {"region_name": weight_value}}`

### Key decisions:
- Use same camera projection as joint_extractor (`world_to_camera_view`)
- Need to handle mesh object transforms (apply `matrix_world` to vertex positions before projection)
- For vertices in evaluated/deformed meshes, use `mesh_obj.evaluated_get(depsgraph)` to get world-space positions
- Actually for T-pose, the rest position should be fine — just use `mesh_obj.matrix_world @ vert.co`
- Aggregate weights by region: if multiple bones map to the same region, sum their weights per vertex

## Files to Modify
1. **`weight_extractor.py`** (NEW) — main extraction logic
   - `extract_weights()` — main entry point, returns weight data dict
   - `save_weights()` — serialize to JSON (or use exporter.save_weights which already exists)
2. **`config.py`** — add `WEIGHT_THRESHOLD` constant (0.01)
3. **`generate_dataset.py`** — integrate weight extraction into the per-character pipeline

## Risks & Edge Cases
- Vertices with no bone weights → assign empty weights dict (issue spec says this explicitly)
- Vertex groups that don't correspond to any bone (shape keys, custom groups) → skip these
- Multiple bones mapping to same region → sum weights (and re-normalize? No, just sum — the issue says weights should sum to ~1.0)
- Very large meshes (10K+ vertices) → JSON could be 1-3MB per character (issue acknowledges this)
- Vertices behind the character from camera view → still include (the "only visible vertices" is marked as optional optimization in the issue)

## Open Questions
- None — the issue spec is very clear. The exporter already has `save_weights()` stub ready.

## Implementation Notes
- `weight_extractor.py` only exports `extract_weights()` — JSON serialization uses `exporter.save_weights()` (already existed as a stub)
- `character_id` is injected into the weight data dict in `generate_dataset.py` before saving (matches how `joint_data["augmentation"]` is added)
- Weight extraction runs before the augmentation loop using a temporary camera, which is cleaned up before the loop creates its own
- Vertex groups are independent of material assignments, so extraction works with segmentation materials active
- Weights are aggregated by region name (summed when multiple bones map to the same region) and rounded to 4 decimal places
- `WEIGHT_THRESHOLD` (0.01) added to `config.py` alongside other extraction constants

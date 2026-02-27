# Issue #7: Extract bone joint positions projected to 2D screen coordinates

## Understanding
- Create `joint_extractor.py` — a new pipeline module
- For each mapped bone in the armature, get its head position in 3D world space
- Project through orthographic camera to 2D pixel coordinates using `bpy_extras.object_utils.world_to_camera_view()`
- Output 17 joints (one per body region, excluding background) per pose as JSON
- Handle regions mapped to multiple bones by choosing a primary bone
- Detect occluded joints via raycasting and mark `visible: false`
- Compute 2D bounding box from visible joint positions
- Type: new feature

## Approach
1. **Primary bone selection**: Define a `PRIMARY_BONES` mapping in `config.py` that picks the most semantically central bone per region (e.g., for "spine" region with Spine1+Spine, use Spine1). This avoids runtime heuristics.
2. **3D→2D projection**: Use `bpy_extras.object_utils.world_to_camera_view(scene, camera, point)` → returns `(x, y, depth)` in normalized coords. Convert: `px_x = x * res_x`, `px_y = (1 - y) * res_y` (Blender Y is flipped).
3. **Posed bone positions**: `armature.matrix_world @ pose_bone.head` gives world-space position of the bone head in the current pose.
4. **Occlusion detection**: Cast a ray from the camera through each joint's 2D position using `scene.ray_cast()`. If the ray hits mesh geometry at a depth closer than the bone's depth, the joint is occluded.
5. **Bounding box**: min/max of all visible joint pixel positions, with some padding.
6. **Missing bones**: If a region has no bone mapped in this character, mark `visible: false` with position `[-1, -1]`.

## Files to Modify
- `config.py` — Add `PRIMARY_BONES` dict mapping region ID → preferred bone name patterns
- `joint_extractor.py` — New file: core extraction logic
  - `extract_joints()` — main entry point
  - `_get_primary_bone_for_region()` — select best bone per region
  - `_project_to_2d()` — 3D world → 2D pixel coords
  - `_check_occlusion()` — raycast-based visibility check
  - `_compute_bbox()` — bounding box from visible joints

## Risks & Edge Cases
- **Non-Mixamo rigs**: May have different bone naming; primary bone selection must use bone_mapper's bone_to_region mapping, not hard-coded names
- **Missing regions**: Some characters may not have all 19 regions mapped — these joints get `visible: false`, position `[-1, -1]`
- **Self-occlusion**: Raycasting from an orthographic camera needs correct setup (ray origin far away, direction along camera axis)
- **Degenerate poses**: All joints in a line could produce a zero-width bounding box; need to handle gracefully
- **Bone head vs tail**: Issue specifies "head" — that's the proximal end (closer to root). This is the standard joint position.

## Open Questions
- None — the issue spec and PRD are comprehensive enough to proceed.

## Implementation Notes

### What was implemented
- **`config.py`**: Added `PRIMARY_BONE_KEYWORDS` (region ID → preferred bone name substrings for Mixamo + Blender-style + generic naming), `NUM_JOINT_REGIONS = 19`, and `JOINT_BBOX_PADDING = 0.05`.
- **`joint_extractor.py`**: New module with 5 private helpers + 2 public functions:
  - `_select_primary_bone()` — keyword-based primary bone selection with fallback to first bone
  - `_project_bone_to_2d()` — uses `world_to_camera_view()` with Y-flip for pixel coords
  - `_check_occlusion()` — scene raycast from far behind camera along forward axis; compares hit distance vs bone distance with 0.02 tolerance
  - `_compute_bbox()` — min/max of visible joints + configurable padding, clamped to image bounds
  - `extract_joints()` — main entry: inverts bone_to_region mapping, iterates 19 regions, returns PRD-schema dict
  - `save_joints()` — writes JSON with full metadata (character_id, pose_name, source_animation, source_frame)

### Design decisions
- **19 joints, not 17**: The issue title says "17 joints" but the Strata skeleton has 19 body regions (1–19). We output one joint per region = 19 joints. The issue requirements also say "one per Strata region".
- **Confidence values**: 1.0 for visible joints, 0.5 for occluded, 0.0 for missing regions. The PRD schema includes `confidence` but doesn't specify values — these are reasonable defaults.
- **Occlusion ray origin**: Placed 100 Blender units behind the bone along the camera's reverse-forward axis. This ensures the ray starts well behind the camera for orthographic projection.
- **Evaluated depsgraph**: Used `bpy.context.evaluated_depsgraph_get()` for raycasting, which respects modifiers and pose deformation.

### Follow-up work
- Integration into `generate_dataset.py` pipeline (will need `generate_dataset.py` to exist first)
- Visual overlay validation (plot joints on rendered image to verify alignment)

# Issue #73: Add multi-camera angle rendering to Mixamo pipeline

## Understanding
- The current pipeline renders all characters from a single front-facing camera (looking along +Y)
- The segmentation model has zero training signal for non-frontal views (3/4, side, back)
- Body part boundaries shift dramatically with viewing angle — this is a critical blocker for the 3D mesh pipeline
- PRD §13.2 defines 5 camera angles: front (0°), three_quarter (45°), side (90°), three_quarter_back (135°), back (180°)
- Type: **new feature** extending existing pipeline

## Approach
**Strategy: Parameterize existing camera setup + add outer angle loop in pipeline**

1. Add `CAMERA_ANGLES` dict to `config.py` with the 5 standard angles
2. Modify `renderer.py:setup_camera()` to accept an `azimuth` parameter — orbit the camera around the character's Y-axis at the specified angle while keeping orthographic projection and auto-framing
3. Add outer loop in `generate_dataset.py:_process_single_pose()` that iterates over camera angles, repositioning camera for each and producing separate outputs
4. Update file naming to include angle: `{char_id}_pose_{nn}_{angle}_{style}.png` for images, `{char_id}_pose_{nn}_{angle}.png` for masks
5. Add `camera_angle` field to joint/draw_order metadata
6. Add `--angles` CLI arg (comma-separated or "all")

**Key design decisions:**
- Camera orbits around Y-axis at fixed distance from bounding box center — this is a simple rotation of the camera position
- Orthographic scale and auto-framing are recalculated per angle since the character's visible extent changes with viewing angle
- Segmentation materials are view-independent (vertex group coloring), so no material changes needed per angle
- Joint positions and draw order MUST be recomputed per angle (different 2D projections)
- Mask is per-pose-per-angle (different visible geometry per angle)
- Style rendering applies identically at each angle

## Files to Modify
1. **`pipeline/config.py`** — Add `CAMERA_ANGLES` constant dict, `DEFAULT_CAMERA_ANGLES` list
2. **`pipeline/renderer.py`** — Modify `setup_camera()` to accept `azimuth` param, compute camera position via orbital rotation around Y-axis
3. **`pipeline/generate_dataset.py`** — Add `--angles` CLI arg, add angle loop inside `_process_single_pose()`, update file naming, add `camera_angle` to metadata
4. **`pipeline/exporter.py`** — Update filename helpers to include angle parameter

## Risks & Edge Cases
- **Bounding box changes per angle:** The auto-framing bounding box needs to be computed per angle, since the character's silhouette width changes. Using the world-space AABB is fine — we just need the ortho_scale to encompass the full character regardless of viewing angle.
- **Self-occlusion at extreme angles:** Side and back views may occlude many joints — the existing occlusion detection in `joint_extractor.py` handles this correctly since it uses ray casting from the camera position.
- **Draw order reversal:** At 180° (back view), the draw order inverts — regions that were "front" become "back". The existing depth computation in `draw_order_extractor.py` already computes depth relative to the current camera, so this works automatically.
- **Volume multiplier:** 5x output volume. Need to ensure `--angles` can limit to a subset for testing.

## Open Questions
- None — the PRD is clear on requirements and the implementation approach is straightforward.

## Implementation Notes

### What was implemented
All requirements from the plan were implemented as designed:

1. **`pipeline/config.py`** — Added `CAMERA_ANGLES` dict (5 angles: front/three_quarter/side/three_quarter_back/back), `DEFAULT_CAMERA_ANGLES` (front only), `ALL_CAMERA_ANGLES`, and `CameraAngle` type alias.

2. **`pipeline/renderer.py`** — `setup_camera()` now accepts `azimuth` keyword argument (default 0.0). Camera orbits the bounding box center using `sin`/`cos` for position and rotation. Added `apparent_width` calculation that accounts for depth dimension at non-frontal angles so auto-framing remains correct.

3. **`pipeline/generate_dataset.py`** — Added `--angles` CLI arg (comma-separated names or "all") with validation. Camera angle loop added inside augmentation loop in `_process_single_pose()`. Each angle gets its own camera, segmentation mask, joint extraction, draw order, and color renders. Metadata includes `camera_angle`, `camera_azimuth`, `character_id`, and `pose_id` for cross-view linking.

4. **`pipeline/exporter.py`** — Added `_angle_infix()` helper and `angle` parameter to `image_filename()`, `mask_filename()`, `joints_filename()`, and `draw_order_filename()`. Front angle produces empty infix for backward compatibility.

### Design decisions
- **Backward compatibility**: When `--angles front` (default), output is identical to the pre-change pipeline — no angle infix in filenames, no extra metadata fields except the new camera angle fields.
- **Angle loop placement**: Inside augmentation loop, not outside. This means for each augmentation variant we render all angles before moving to the next variant. This minimizes scale/pose state changes.
- **Style seed**: Now includes `angle_name` in the hash `(char_id, pose_idx, angle_name)` so post-render transforms are deterministic but vary per angle.

### Follow-up work
- The `_is_already_processed()` check only looks at front-angle masks. Could be extended to check per-angle masks if needed.
- Weights are extracted once per character in T-pose (not per-angle), which is correct since weights are view-independent.

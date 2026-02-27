# Issue #11: Add pose augmentation: Y-axis flip and scale variation

## Understanding
- Add two pose augmentation transforms to `pose_applicator.py`:
  1. **Y-axis flip** (horizontal mirror): doubles dataset size, prevents left/right bias
  2. **Scale variation** (0.8x–1.2x): simulates different character proportions
- These are applied **on top of existing poses** before rendering
- Both are optional (config/CLI flag controlled)
- Must track augmentation metadata in output JSON: `{"flipped": true, "scale_factor": 1.15}`

## Approach

### Y-axis flip
- Use `armature.scale.x *= -1` then `bpy.ops.object.transform_apply(scale=True)`
- After flip, the mesh normals will be inverted — need to recalculate normals on meshes
- Camera auto-framing must be recalculated (handled by re-calling `setup_camera`)
- Joint extraction must swap left/right joint names
- Mask rendering inherently handles it — the flipped character will have the mesh faces
  pointing to swapped materials, but actually the material assignments are per-face based
  on vertex weights. Since we're physically mirroring the geometry, the left arm mesh now
  occupies the right side of screen. The region materials are still assigned to original
  bones. So we need to swap the region labels in post-processing:
  - Swap region IDs in the grayscale mask (e.g., 6↔9, 7↔10, 8↔11, etc.)
  - Swap joint names in the JSON output
- **Simpler approach**: Instead of physically flipping the armature (which has normals issues),
  we can flip the **rendered images** and masks in 2D (horizontal flip), then swap L/R labels.
  This is cleaner because:
  - No Blender normals issues
  - No transform_apply complications
  - Same visual result
  - Camera framing stays the same
  - Much simpler implementation

**Decision**: Use the 2D post-render flip approach. After rendering the original pose, generate
a flipped variant by:
1. Horizontally flip the color image
2. Horizontally flip the mask, then swap L/R region IDs
3. Horizontally flip joint X coordinates, then swap L/R joint names

### Scale variation
- Apply uniform scale to the armature object: `armature.scale = (factor, factor, factor)`
- Then `bpy.ops.object.transform_apply(scale=True)` — actually, don't apply transforms,
  just set the scale and let the camera reframe. The camera framing is recomputed anyway.
- Actually, the issue says to re-apply transforms and recompute camera framing.
- Approach: For each scale factor, set armature scale, update meshes, recalculate camera,
  render, then restore original scale.
- Scale factors: configurable list, default e.g. [0.85, 1.0, 1.15]

**Decision**: Scale the armature + meshes uniformly, recompute camera, render, restore.
Since we're using an orthographic camera that auto-frames, the visual difference is minimal
(ortho camera just changes the framing). The value is that joint coordinates change and
the character occupies different amounts of the frame, which helps the model generalize.

## Files to Modify
1. **`config.py`**: Add augmentation constants (flip enabled, scale range, scale samples)
2. **`pose_applicator.py`**: Add `AugmentationInfo` dataclass, `apply_flip_augmentation()`,
   `apply_scale_augmentation()`, `restore_scale()` functions
3. **`generate_dataset.py`**: Update `process_character()` to loop over augmentations,
   add CLI flags (`--enable_flip`, `--scale_factors`), pass augmentation metadata

## Risks & Edge Cases
- Normals inversion after X-axis scale flip — avoided by using 2D flip approach
- Scale factor 1.0 should not produce a separate augmented output (it IS the original)
- Flipped version naming: need a clear naming convention (e.g., `_flip` suffix)
- Camera reframing after scale — must delete and recreate camera
- Non-symmetric characters (e.g., one arm missing) — flip still works, just produces
  the mirror of whatever's there
- Scale too extreme could clip the camera — 0.8x to 1.2x is safe

## Open Questions
- None — the issue is well-specified and the 2D flip approach simplifies implementation.

## Implementation Notes

### What was implemented
- **config.py**: Added `ENABLE_FLIP`, `ENABLE_SCALE`, `SCALE_FACTORS` defaults,
  `FLIP_REGION_SWAP` (region ID↔ID), `FLIP_JOINT_SWAP` (joint name↔name)
- **pose_applicator.py**: Added `AugmentationInfo` dataclass with `to_dict()` and
  `suffix` property. Added `apply_scale()`, `restore_scale()`, `flip_image()`,
  `flip_mask()`, `flip_joints()` — all pure functions, no side effects on Blender state
  (except scale which sets object.scale).
- **generate_dataset.py**: Added `--enable_flip`, `--enable_scale`, `--scale_factors`
  CLI flags. Added `_build_augmentation_list()` to enumerate variants. Updated
  `process_character()` to iterate augmentation variants, recompute camera per scale,
  apply 2D flip as post-processing.

### Design decisions during implementation
- **2D flip over 3D flip**: Chose to flip rendered images/masks in 2D rather than
  mirroring the armature in Blender. Avoids normals issues, transform_apply complications,
  and is simpler to implement. The flipped variant renders from the same 3D scene — only
  the output files are flipped and labels swapped.
- **Scale sets object.scale without apply**: We set `armature.scale` and `mesh.scale`
  directly rather than using `bpy.ops.object.transform_apply`. The camera is recomputed
  per variant anyway, and restoring scale is trivial (set back to 1.0).
- **Augmentation suffix in filenames**: `AugmentationInfo.suffix` generates deterministic
  suffixes like `_flip`, `_s085`, `_flip_s115` to keep filenames unique and parseable.
- **Both augmentations disabled by default**: Config defaults and CLI flags default to
  `False`, so existing pipeline behavior is unchanged unless explicitly opted in.

### Follow-up work
- The random small bone rotation augmentation from PRD §6.3 is not implemented (not
  requested in this issue).
- When Phase 2 adds pose looping, the augmentation loop nests inside the pose loop.

# Issue #10: Implement pose library loading and keyframe extraction

## Understanding
- Create `pose_applicator.py` — a new module to load animation FBX files, extract evenly-spaced keyframes, and apply poses to character armatures for varied training data rendering.
- Type: **new feature** (Phase 2: Pose & Scale)
- Depends on: `config.py` (#1), `importer.py` (#2), `bone_mapper.py` (#3) — all implemented.

## Approach

The module has three core responsibilities:

1. **List available poses** — scan pose library directory for `.fbx` animation files, return metadata.
2. **Extract keyframes** — load an animation FBX, determine total frame count, sample N evenly-spaced keyframes.
3. **Apply poses** — transfer animation data from imported anim armature to character armature by bone name matching, then set the scene to the target frame. Also support built-in T-pose and A-pose without animation files.

### Design decisions:
- **Animation retargeting strategy**: Import the animation FBX into Blender as a temporary armature. Copy the `Action` from the imported armature to the character armature using Blender's NLA/action system, then `scene.frame_set()` to the target keyframe. Clean up the temp armature after.
- **Bone name matching for retargeting**: Both Mixamo animations and characters use `mixamorig:` prefixed names. For non-Mixamo rigs, use `bone_mapper.py`'s prefix stripping logic to normalize names and find correspondences.
- **T-pose**: Reset all pose bones to identity rotation (quaternion `(1,0,0,0)`, euler `(0,0,0)`, and zero location offset).
- **A-pose**: Same as T-pose but with ~45° downward rotation on upper arm bones. This is a common enough pose to hardcode.
- **Keyframe sampling**: For a clip with F total frames, sample at `[0, F//(N-1), 2*F//(N-1), ...]` for N keyframes. The issue specifies `[0, N//4, N//2, 3*N//4]` for 4 keyframes — generalize to any N.

### Public API:
```python
@dataclass
class PoseInfo:
    name: str          # e.g. "walk_frame_07"
    source: str        # e.g. "Walking.fbx" or "built-in"
    frame: int         # source frame number

def list_poses(pose_dir: Path, keyframes_per_clip: int = 4) -> list[PoseInfo]
def apply_pose(armature: bpy.types.Object, pose: PoseInfo, pose_dir: Path) -> None
def reset_pose(armature: bpy.types.Object) -> None
```

## Files to Modify
- **`pose_applicator.py`** (NEW) — main module
- **`config.py`** — add `KEYFRAMES_PER_CLIP = 4` and `A_POSE_SHOULDER_ANGLE` constants

## Risks & Edge Cases
- **Non-Mixamo animations on non-Mixamo characters**: Bone names won't match. Need to use prefix stripping and fallback to normalized name comparison.
- **Animations with no keyframes / very short clips**: Guard against division by zero when sampling. Clips with fewer frames than `keyframes_per_clip` should just use all available frames.
- **Multiple armatures in animation FBX**: Animation FBX files from Mixamo typically contain a single armature. Use the first one, warn on multiples (same pattern as `importer.py`).
- **Action data not found**: Some FBX files may have the animation baked into the armature's pose rather than as an Action. Handle both cases.
- **Scene state leaks**: Must clean up imported animation armature and its data blocks after extraction to avoid polluting the scene for the next pose.

## Open Questions
- None — the issue and PRD are sufficiently detailed. A-pose angle can be tuned later.

## Implementation Notes

### What was implemented
- **`pose_applicator.py`** (new, ~540 lines): Full module with `list_poses()`, `apply_pose()`, `reset_pose()` public API.
- **`config.py`**: Added `KEYFRAMES_PER_CLIP = 4` and `A_POSE_SHOULDER_ANGLE = 45.0` constants.

### Design decisions during implementation
- **Retargeting approach changed**: Instead of copying Actions via NLA, we directly read pose bone transforms at a given frame from the imported anim armature and copy them to the character armature. This is simpler and avoids NLA complexity — the anim armature is just used as a "pose reader" at a specific frame.
- **Rotation mode handling**: The transfer copies whichever rotation mode the source bone uses (quaternion, euler, or axis-angle), preserving fidelity.
- **Action discovery fallback chain**: `armature.animation_data.action` → NLA tracks → first action in `bpy.data.actions`. Covers Mixamo's typical baked-action export as well as NLA-based setups.
- **Bone name matching uses set + dict for efficiency**: Target bones are indexed into a set (for exact match) and a normalized-name dict (for prefix-stripped fallback), making lookups O(1).

### Follow-up work
- `generate_dataset.py` needs to be updated to use `list_poses()` / `apply_pose()` / `reset_pose()` in its per-character loop (currently hardcoded to T-pose only with `POSE_INDEX = 0`).
- A-pose shoulder rotation axis may need tuning per-rig — the Z-axis rotation works for Mixamo but may not be correct for all armature orientations.

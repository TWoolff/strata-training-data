# Issue #45: Build BVH-to-Strata skeleton retargeting pipeline

## Understanding
- Build a retargeting pipeline that maps CMU motion capture skeleton data to Strata's 19-bone skeleton
- Three new files: `bvh_to_strata.py` (core retargeting), `proportion_normalizer.py` (scale to character), `blueprint_exporter.py` (JSON export)
- Type: **New feature** in the animation intelligence pipeline
- Depends on: #44 (BVH parser) — already implemented

## Approach

### 1. `bvh_to_strata.py` — Core retargeting
- Define `CMU_TO_STRATA` bone name mapping dict as module constant
- Multi-spine collapse: CMU has Spine, Spine1, Spine2 → Strata uses `spine` (Spine1 rotation) and `chest` (Spine2 rotation)
- Issue mentions `forearm_l`/`forearm_r` in CMU_TO_STRATA but Strata uses `lower_arm_l`/`lower_arm_r` (IDs 7, 10). Use the Strata region names from config.py REGION_NAMES.
- Key data structure: `RetargetedFrame` dataclass holding per-bone rotations + root position
- `RetargetedAnimation` dataclass wrapping frames + metadata (frame_count, frame_rate, skeleton name)
- `retarget()` function takes BVHFile → RetargetedAnimation
- Unmapped bones (fingers, toes, End Sites) logged as warnings then skipped

### 2. `proportion_normalizer.py` — Scale mocap to character proportions
- Extract bone lengths from BVH T-pose (frame 0 or from skeleton offsets)
- Actually, BVH offsets in the HIERARCHY section already define bone lengths — use those directly rather than frame 0
- Accept target character bone lengths as a dict
- Scale root translations by ratio of target height to source height
- Keep rotations unchanged — angles transfer regardless of bone length
- Output: normalized RetargetedAnimation with adjusted root positions

### 3. `blueprint_exporter.py` — Export to JSON
- Takes RetargetedAnimation → JSON file matching the schema from the issue
- `{"skeleton": "strata_19", "frame_count": N, "frame_rate": fps, "frames": [...]}`
- Each frame: dict of bone name → {"rotation": [x,y,z], "position": [x,y,z]} (position only for root/hips)

## Files to Modify
- **New:** `animation/scripts/bvh_to_strata.py`
- **New:** `animation/scripts/proportion_normalizer.py`
- **New:** `animation/scripts/blueprint_exporter.py`
- **New:** `tests/test_bvh_to_strata.py`
- **New:** `tests/test_proportion_normalizer.py`
- **New:** `tests/test_blueprint_exporter.py`

## Risks & Edge Cases
- **Bone name mismatch:** Issue uses `forearm_l`/`forearm_r` in CMU_TO_STRATA but Strata config uses `lower_arm_l`/`lower_arm_r`. Must use canonical Strata names from REGION_NAMES.
- **Missing bones:** Some CMU skeletons may not have all bones (e.g., no Spine2). Need graceful handling.
- **Multi-spine collapse:** Must correctly pick Spine1→spine, Spine2→chest. If only Spine exists, assign to spine, leave chest as zero rotation.
- **Channel ordering:** CMU uses ZXY rotation order while some files use YXZ (SFU). Rotation values are passed through as-is — consumer must handle rotation order from the BVH channels metadata.
- **Proportion normalization edge cases:** Characters with zero bone lengths, mismatched bone sets.
- **Frame rate:** BVH frame_time → fps conversion (1/frame_time).

## Open Questions
- Should rotation order be preserved in the blueprint JSON? → Yes, include it as metadata since different BVH sources use different orders.
- Should the blueprint include only mapped bones or all 19 Strata bones (with zeros for unmissioned)? → All 19, with zero rotations for bones not present in source.

## Implementation Notes

### What was implemented
All three modules as planned, plus comprehensive test suites (43 tests total).

### Design decisions during implementation
- **`_extract_channels` consolidation:** `_extract_rotation` and `_extract_position` shared identical iteration logic. Consolidated into a shared `_extract_channels(channels, values, x_name, y_name, z_name)` helper with thin wrappers.
- **Bone name correction:** Issue #45 used `forearm_l`/`forearm_r` in the CMU_TO_STRATA mapping, but Strata's canonical names are `lower_arm_l`/`lower_arm_r`. Implementation uses the correct Strata names.
- **`_SILENTLY_IGNORED_SUFFIXES`:** End Sites, finger bones (Thumb, Index, Middle, Ring, Pinky), and toe bones are silently skipped rather than logged as unmapped warnings, since they're expected to have no Strata equivalent.
- **Rotation order detection:** Reads the first joint with 3+ rotation channels and extracts the axis order (e.g., "ZXY", "YXZ"). Stored in `RetargetedAnimation.rotation_order` and included in blueprint JSON as `rotation_order` metadata.
- **`math.hypot(*offset)`:** Used instead of manual `sqrt(x²+y²+z²)` for bone length calculation — cleaner and numerically equivalent.
- **Blueprint JSON rounding:** Values rounded to 4 decimal places for compact output while preserving sufficient precision.

### Test coverage
- `test_bvh_to_strata.py`: 19 tests — basic retargeting, multi-spine collapse, bone mapping, rotation order detection, empty motion edge case
- `test_proportion_normalizer.py`: 10 tests — bone length extraction, scaling, identity scale, zero-height guards
- `test_blueprint_exporter.py`: 14 tests — blueprint structure, file export, JSON roundtrip consistency

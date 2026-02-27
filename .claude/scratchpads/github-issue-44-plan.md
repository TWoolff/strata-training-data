# Issue #44: Build BVH motion capture file parser

## Understanding
- Build a BVH (Biovision Hierarchy) file parser as the foundation module for the animation retargeting pipeline
- New file: `animation/scripts/bvh_parser.py`
- Pure Python + numpy — NO Blender dependency
- Must handle both CMU Graphics Lab and SFU Motion Capture BVH variants
- Outputs structured Python dataclasses: BVHSkeleton, BVHJoint, BVHMotion
- This is a blocking dependency for R4 (retargeting), R5 (CMU labels), R9 (synthetic degradation)

## Approach
- Parse BVH in two phases: HIERARCHY then MOTION
- HIERARCHY: recursive descent parser that builds joint tree from indented JOINT/End Site blocks
- MOTION: read frame count + frame time, then parse flat float arrays per frame and distribute to joints based on channel order from HIERARCHY
- Use dataclasses matching the issue spec exactly
- Add a top-level `parse_bvh(path)` convenience function
- Handle CMU vs SFU differences:
  - CMU: Root is "Hips" with 6 channels (Xposition Yposition Zposition + 3 rotations)
  - SFU: Similar structure but may have different joint naming
  - Both: non-root joints typically have 3 rotation channels only
  - Varying whitespace and formatting between sources

## Files to Modify
- **NEW** `animation/scripts/bvh_parser.py` — main parser module
- **NEW** `animation/scripts/__init__.py` — make it a package (empty)
- **NEW** `tests/test_bvh_parser.py` — unit tests with synthetic BVH data

## Risks & Edge Cases
- BVH channel order varies (ZXY vs ZYX vs XYZ) — must preserve per-joint channel names
- Empty MOTION section (skeleton-only files) — return 0 frames
- Single-frame files — valid, just one frame of data
- "End Site" nodes have offsets but no channels — must handle without crashing
- Flat frame array must be sliced correctly by walking joints in HIERARCHY order
- Very large BVH files (thousands of frames) — use numpy for frame data storage
- Tab vs space indentation varies between CMU and SFU

## Open Questions
- None — issue spec is clear and complete

## Implementation Notes
- Created `animation/scripts/bvh_parser.py` with dataclasses matching issue spec exactly
- HIERARCHY parser uses iterative approach with explicit stack (not recursive) for robustness
- End Site nodes stored as `{parent}_End` with empty channels list
- `joint_order` field preserves hierarchy traversal order — critical for correct frame data distribution
- `_parse_key_value_int` and `_parse_key_value_float` kept as separate functions for type safety
- `get_frame_array()` provides numpy convenience for downstream batch processing
- 32 unit tests covering: CMU format, SFU format (extra whitespace, YXZ rotation order), skeleton-only files, single-frame files, malformed inputs, convenience API
- No Blender dependency — pure Python + numpy as required

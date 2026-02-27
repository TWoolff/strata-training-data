# Issue #50: Build synthetic animation degradation pipeline

## Understanding
- Build `animation/scripts/degrade_animation.py` — a script that takes high-quality animations and systematically strips animation principles to create (degraded input, full output) training pairs
- This is the primary training data strategy for the in-betweening model
- Input: Parsed BVH animation (from `bvh_parser.py`) or Strata blueprint JSON
- Output: Degraded animation + reference to original (training pair)
- Target: 500 clips × 7 degradation types = 3,500 training pairs
- Type: New feature

## Approach
- Create a single new module `animation/scripts/degrade_animation.py`
- Each degradation is a pure function: `degrade(animation, params) -> degraded_animation`
- Work with `RetargetedAnimation` from `bvh_to_strata.py` as the internal representation
- Support loading from both parsed BVH (via `bvh_parser.py` + `retarget()`) and Strata blueprint JSON (via `blueprint_exporter.py` format)
- Output training pairs as JSON: degraded animation + path to original
- Provide CLI for batch processing a directory of BVH files

### 7 Degradation Types
1. **Strip to extremes**: Find frames with max/min joint angles across all bones, keep only those keyframes
2. **Linearize arcs**: For each bone, replace curved motion paths with linear interpolation between keyframes (evenly spaced samples)
3. **Remove easing**: Replace non-uniform frame spacing with uniform spacing (linear time remap)
4. **Remove secondary**: For each frame, copy parent bone rotation to child bones on secondary chains (locks follow-through)
5. **Reduce framerate**: Keep every Nth frame (N configurable: 2, 3, 4)
6. **Simultaneous stop**: Find the last frame of motion for each bone, snap all to stop on the earliest stop frame
7. **Remove anticipation**: Detect direction reversals before main motion, delete those frames

### Design Decisions
- Each function takes `RetargetedAnimation` + type-specific params and returns a new `RetargetedAnimation` (immutable pattern, matching `proportion_normalizer.py`)
- Degradation params collected in a dataclass for each type, with sensible defaults
- CLI supports: single file, batch directory, specific degradation types, output directory
- Training pairs saved as JSON with `{degraded: ..., original_path: ..., degradation_type: ..., params: ...}` metadata

## Files to Modify
- **NEW**: `animation/scripts/degrade_animation.py` — all 7 degradation functions + CLI + training pair output
- No changes to existing files needed

## Risks & Edge Cases
- **Very short clips** (< 5 frames): Some degradations may produce empty or trivially short results — add minimum frame count guards
- **Static bones**: Bones with zero rotation across all frames shouldn't affect extreme detection
- **Anticipation detection**: Direction reversal heuristic may false-positive on subtle motions — use a velocity threshold
- **Root position**: Need to decide if root position also gets degraded (yes — it's part of the motion)
- **Blueprint input**: Need to parse the blueprint JSON format back into RetargetedAnimation

## Open Questions
- None — the issue is well-specified and maps directly to the PRD §5.4

## Implementation Notes
- All 7 degradation types implemented as pure functions returning new `RetargetedAnimation`
- Each degradation has a params dataclass with sensible defaults
- `STRATA_PARENT` dict defines the 18 parent-child bone relationships for secondary motion removal
- `SECONDARY_BONES` = {hand_l, hand_r, foot_l, foot_r, head} — distal limb ends
- `MIN_FRAMES = 4` guard on all degradations to avoid degenerate outputs
- `strip_to_extremes` uses local max/min detection with 1.0° range threshold to filter static bones
- `remove_easing` uses arc-length parameterization (cumulative rotation delta) then resamples at uniform parameter values via `np.interp`
- `remove_anticipation` scans velocity per-bone per-axis for short direction reversals followed by longer main motion
- CLI `--framerate-factor` and `--keyframe-interval` wired through `param_overrides` to `process_file`
- `load_blueprint()` parses Strata blueprint JSON back into `RetargetedAnimation` for round-trip support
- Code simplifier applied: hoisted `total_range` out of inner loop in `strip_to_extremes`, simplified `simultaneous_stop` frame assembly, wired dead CLI args

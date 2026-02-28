# Issue #80: Build extract_timing.py for frame spacing and velocity extraction

## Understanding
- Create `animation/scripts/extract_timing.py` to extract timing patterns from labeled BVH mocap clips
- Reads labeled clips from `animation/labels/cmu_action_labels.csv` (79 clips, 26 action types)
- For each clip: parse BVH, extract root velocity, joint angular velocity, timing statistics
- Group results by action type and output JSON files to `animation/timing-norms/`
- Pure Python + NumPy, no Blender dependency
- Type: new feature (animation intelligence)

## Approach
1. **CSV loading** — Read `cmu_action_labels.csv`, filter to clips where BVH file exists
2. **Per-clip extraction** — For each BVH:
   - Parse via `bvh_parser.parse_bvh()`
   - **Root velocity**: Compute frame-to-frame displacement of root (Hips) position channels; velocity = displacement / frame_time
   - **Joint angular velocity**: For each joint with rotation channels, compute frame-to-frame rotation delta / frame_time (degrees/sec)
   - **Timing statistics**: Identify acceleration phases (velocity increasing), deceleration phases (velocity decreasing), hold frames (near-zero velocity)
3. **Aggregation** — Group clip timing data by `action_type`, compute per-action-type norms (mean, std, min, max for key metrics)
4. **Output** — One JSON file per action type in `animation/timing-norms/{action_type}.json` plus a summary `animation/timing-norms/summary.json`

### JSON Schema (per action type):
```json
{
  "action_type": "walk",
  "clip_count": 10,
  "clips": [
    {
      "filename": "01_01.bvh",
      "subcategory": "forward",
      "duration_seconds": 3.5,
      "frame_count": 105,
      "frame_rate": 30.0,
      "root_velocity": {
        "mean": 1.2,
        "max": 2.1,
        "min": 0.0,
        "curve": [0.0, 0.3, 0.8, ...]
      },
      "joint_angular_velocities": {
        "hips": {"mean": 15.0, "max": 45.0},
        ...
      },
      "timing": {
        "acceleration_frames": 12,
        "deceleration_frames": 8,
        "hold_frames": 5,
        "acceleration_ratio": 0.11,
        "deceleration_ratio": 0.08,
        "hold_ratio": 0.05
      }
    }
  ],
  "norms": {
    "duration_seconds": {"mean": 3.2, "std": 0.8, "min": 2.1, "max": 4.5},
    "root_velocity_mean": {"mean": 1.1, "std": 0.3, "min": 0.5, "max": 1.8},
    "root_velocity_max": {"mean": 2.0, "std": 0.5, "min": 1.2, "max": 3.1}
  }
}
```

### Key design decisions:
- Root velocity is Euclidean norm of (dx, dy, dz) per frame — gives speed, not direction
- Angular velocity per joint is Euclidean norm of (drx, dry, drz) — simple but effective
- Acceleration/deceleration classified by sign of velocity derivative (positive = accel, negative = decel)
- Hold frames: root velocity below a threshold (e.g. 0.1 units/sec)
- Velocity curves stored as arrays for potential plotting; full curves only in per-clip data, not in norms

## Files to Modify
- **CREATE**: `animation/scripts/extract_timing.py` — Main script
- **CREATE**: `tests/test_extract_timing.py` — Tests

## Risks & Edge Cases
- BVH files referenced in CSV may not exist on disk (data/ is gitignored) — must handle gracefully with warning
- Clips with 0 or 1 frame — no velocity can be computed, skip gracefully
- NaN/Inf from division by zero if frame_time is 0 — guard against
- Very short clips may have nonsensical timing statistics — include but flag via frame count

## Open Questions
- None — requirements are clear from the issue. The BVH files won't be on disk for testing, so tests will use synthetic BVH content (same pattern as other test files).

## Implementation Notes
- Implemented exactly as planned. All 36 tests pass.
- Root velocity curve stored in `root_velocity_curve` field (separate from `root_velocity` stats) for JSON clarity.
- Joint angular velocities keyed by original BVH joint name (not Strata name) since this is a BVH analysis tool.
- `_dataclass_to_dict` initially written but removed during simplification — `dataclasses.asdict()` handles all serialization needs.
- Timing stats classify hold frames first (velocity < threshold), then split remaining by derivative sign.
- Summary JSON intentionally omits per-clip details to keep it scannable.
- Frame time 0 guarded against everywhere with fallback to 1/30.

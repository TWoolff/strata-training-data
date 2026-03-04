# Issue #158: Retarget 100STYLE dataset to Strata 19-bone skeleton

## Understanding
- Retarget 1,620 BVH motion clips (100 locomotion styles × 8-10 content types) to Strata's 19-bone skeleton
- Output Strata blueprint JSON files + taxonomy CSV
- Upload to Hetzner bucket under `animation/100style/`
- Type: new feature (new dataset adapter for animation data)

## 100STYLE Skeleton (different from CMU)
```
Hips (ROOT, 6 channels: pos + rot)
├── Chest (3ch rot)
│   └── Chest2 (3ch rot)
│       └── Chest3 (3ch rot)
│           └── Chest4 (3ch rot)
│               ├── Neck (3ch rot)
│               │   └── Head (3ch rot) → End Site
│               ├── RightCollar (3ch rot)
│               │   └── RightShoulder (3ch rot)
│               │       └── RightElbow (3ch rot)
│               │           └── RightWrist (3ch rot) → End Site
│               └── LeftCollar (3ch rot)
│                   └── LeftShoulder (3ch rot)
│                       └── LeftElbow (3ch rot)
│                           └── LeftWrist (3ch rot) → End Site
├── RightHip (3ch rot)
│   └── RightKnee (3ch rot)
│       └── RightAnkle (3ch rot)
│           └── RightToe (3ch rot) → End Site
└── LeftHip (3ch rot)
    └── LeftKnee (3ch rot)
        └── LeftAnkle (3ch rot)
            └── LeftToe (3ch rot) → End Site
```

All joints use **YXZ** rotation order. 60fps (Frame Time: 0.016667).

## 100STYLE → Strata Mapping
| 100STYLE Joint | Strata Bone |
|---|---|
| Hips | hips |
| Chest | spine (lowest spine joint) |
| Chest2, Chest3 | (collapsed/ignored — mid-chain) |
| Chest4 | chest (highest spine joint before neck) |
| Neck | neck |
| Head | head |
| LeftCollar | shoulder_l |
| LeftShoulder | upper_arm_l |
| LeftElbow | forearm_l |
| LeftWrist | hand_l |
| RightCollar | shoulder_r |
| RightShoulder | upper_arm_r |
| RightElbow | forearm_r |
| RightWrist | hand_r |
| LeftHip | upper_leg_l |
| LeftKnee | lower_leg_l |
| LeftAnkle | foot_l |
| RightHip | upper_leg_r |
| RightKnee | lower_leg_r |
| RightAnkle | foot_r |

Silently ignored: LeftToe, RightToe, End Sites, Chest2, Chest3 (mid-chain spine).

## Content Codes
| Code | Content |
|---|---|
| FW | forward_walk |
| BW | backward_walk |
| FR | forward_run |
| BR | backward_run |
| SW | sideways_walk |
| SR | sideways_run |
| ID | idle |
| TR1 | turn_1 |
| TR2 | turn_2 (16 styles only) |
| TR3 | turn_3 (4 styles only) |

## Approach
1. **Add 100STYLE mapping to `bvh_to_strata.py`** — add a `STYLE100_TO_STRATA` dict alongside the existing `CMU_TO_STRATA`. Modify `_build_bone_map()` to accept a mapping source parameter, or create a separate auto-detect function that identifies 100STYLE skeletons (presence of `Chest2/Chest3/Chest4` joints).
2. **Create `animation/scripts/retarget_100style.py`** — batch processing script that:
   - Iterates all style directories
   - Parses each BVH with existing `bvh_parser.parse_bvh()`
   - Retargets using updated `retarget()` with auto-detected mapping
   - Exports blueprint JSON via `blueprint_exporter.export_blueprint()`
   - Applies Frame_Cuts.csv trimming (remove padding frames at start/end)
   - Generates summary stats
3. **Create `animation/labels/100style_labels.csv`** — from `Dataset_List.csv` + content codes
4. **Upload to Hetzner bucket** under `animation/100style/` prefix

## Files to Modify
- `animation/scripts/bvh_to_strata.py` — add 100STYLE mapping + auto-detection logic
- `animation/scripts/retarget_100style.py` — NEW: batch retargeting script
- `animation/labels/100style_labels.csv` — NEW: taxonomy CSV
- `tests/test_bvh_to_strata.py` — add test for 100STYLE skeleton mapping

## Risks & Edge Cases
- 4-spine collapse (Chest→Chest2→Chest3→Chest4): need to pick the right ones for spine/chest
- Frame cuts: some sequences have N/A for TR2/TR3 — skip those
- 60fps data: much higher frame rate than typical 30fps CMU data — preserve as-is
- 1,620 files × ~5K frames avg = ~8M frames total — process sequentially per file, don't batch into memory
- BouncyLeft/BouncyRight have TR2 entries; 4 styles have TR3

## Open Questions
- None — all information is available from the dataset structure and existing pipeline

## Implementation Notes

### What was implemented
1. **`bvh_to_strata.py`** — Added `STYLE100_TO_STRATA` mapping dict, `_is_100style_skeleton()` auto-detection (checks for Chest4 + LeftHip joints), updated `_resolve_spine_mapping()` to handle 4-spine collapse (Chest→spine, Chest4→chest, Chest2/Chest3 ignored), updated `_build_bone_map()` to auto-select the correct mapping table, updated `check_strata_compatibility()` to skip mid-chain spine bones.

2. **`retarget_100style.py`** — New batch script with Frame_Cuts.csv trimming, Dataset_List.csv metadata loading, filename parsing (style + content code), label CSV generation. Skips macOS `._` resource fork files.

3. **`100style_labels.csv`** — 810 rows with columns: filename, style_name, content_code, content_type, description, stochastic, symmetric, frame_count, frame_rate, strata_compatible.

4. **Tests** — 11 new tests in `test_bvh_to_strata.py` covering 100STYLE detection, spine collapse, all-19-bones mapping, compatibility, YXZ rotation order, 60fps frame rate.

### Key results
- 810/810 sequences retargeted, 0 errors
- 4,779,750 frames in → 4,055,978 frames out (84.9% after trim)
- 19/19 Strata bones mapped per file, 0 unmapped
- 60fps, YXZ rotation order preserved
- Output: 8.7 GiB of blueprint JSON across 100 style subdirectories
- Uploaded to `s3://strata-training-data/animation/100style/`

### Design decisions
- Auto-detection approach (vs explicit parameter) — cleaner API, `retarget()` just works without caller needing to know which skeleton type
- Chest→spine, Chest4→chest mapping — chosen because Chest is the lowest spine joint (closest to hips = Strata "spine") and Chest4 is the highest (closest to neck = Strata "chest"). Chest2/Chest3 are mid-chain and silently ignored like CMU's mid-chain Spine.
- Frame trimming applied by default — the BVH files contain lead-in/lead-out padding that's not useful for training

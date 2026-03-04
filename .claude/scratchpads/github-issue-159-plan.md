# Issue #159: Download and ingest AnimeDrawingsDataset (2K manga joint annotations)

## Understanding
- Build an ingest adapter for the AnimeDrawingsDataset (dragonmeteor/AnimeDrawingsDataset)
- Dataset has 2,000 anime/manga images with **22 joint keypoints** (not 9 as the issue initially stated)
- Split: 1,400 train / 100 val / 500 test
- License: Not specified — needs verification before commercial use
- The dataset requires Ruby/rake to download images, or Docker
- This adapter converts images + joint annotations to Strata format

## Dataset Structure
The dataset JSON files (train.json, val.json, test.json, data.json) contain entries like:
```json
{
  "file_name": "data/images/1850571.png",
  "width": 371,
  "height": 600,
  "points": {
    "head": [x, y],
    "neck": [x, y],
    "nose_tip": [x, y],
    "nose_root": [x, y],
    "body_upper": [x, y],
    "arm_left": [x, y],      // shoulder
    "arm_right": [x, y],
    "elbow_left": [x, y],
    "elbow_right": [x, y],
    "wrist_left": [x, y],
    "wrist_right": [x, y],
    "thumb_left": [x, y],
    "thumb_right": [x, y],
    "leg_left": [x, y],      // hip
    "leg_right": [x, y],
    "knee_left": [x, y],
    "knee_right": [x, y],
    "ankle_left": [x, y],
    "ankle_right": [x, y],
    "tiptoe_left": [x, y],   // foot
    "tiptoe_right": [x, y]
  }
}
```

## Joint Mapping (22 dataset joints → Strata 19 regions)
| Dataset Joint    | Strata Region ID | Strata Name     | Notes                    |
|-----------------|------------------|-----------------|--------------------------|
| head            | 1                | head            | Direct                   |
| neck            | 2                | neck            | Direct                   |
| body_upper      | 3                | chest           | Best available torso pt  |
| nose_tip        | —                | (skip)          | No Strata equivalent     |
| nose_root       | —                | (skip)          | No Strata equivalent     |
| arm_left        | 7                | upper_arm_l     | Shoulder position        |
| arm_right       | 11               | upper_arm_r     | Shoulder position        |
| elbow_left      | 8                | forearm_l       | Elbow ≈ forearm          |
| elbow_right     | 12               | forearm_r       | Elbow ≈ forearm          |
| wrist_left      | 9                | hand_l          | Wrist ≈ hand             |
| wrist_right     | 13               | hand_r          | Wrist ≈ hand             |
| thumb_left      | —                | (skip)          | No Strata equivalent     |
| thumb_right     | —                | (skip)          | No Strata equivalent     |
| leg_left        | 14               | upper_leg_l     | Hip position             |
| leg_right       | 17               | upper_leg_r     | Hip position             |
| knee_left       | 15               | lower_leg_l     | Knee ≈ lower leg         |
| knee_right      | 18               | lower_leg_r     | Knee ≈ lower leg         |
| ankle_left      | 16               | foot_l          | Ankle ≈ foot             |
| ankle_right     | 19               | foot_r          | Ankle ≈ foot             |
| tiptoe_left     | —                | (skip)          | Already have ankle→foot  |
| tiptoe_right    | —                | (skip)          | Already have ankle→foot  |

**Mapped: 15 of 19** Strata joints. Missing: spine (4), hips (5), shoulder_l (6), shoulder_r (10).
These 4 are marked `visible: false` in output.

## Approach
- Follow the FBAnimeHQ adapter pattern (simplest single-image adapter) but add joint output like HumanRig
- Parse JSON annotation files, read images, resize to 512×512, map joints to Strata format
- Joint coordinates are in original image pixel space — rescale to match resized canvas
- Images vary in size, so pad to square canvas and adjust joint coords accordingly
- Output joints.json in the HumanRig format: list of 19 dicts with id/name/x/y/visible

## Files to Modify
1. **NEW**: `ingest/anime_drawings_adapter.py` — Main adapter module
2. **NEW**: `tests/test_anime_drawings_adapter.py` — Test suite (≥8 tests)
3. **EDIT**: `run_ingest.py` — Register `anime_drawings` adapter
4. **EDIT**: `.claude/prd/strata-training-data-checklist.md` — Mark PP-12 items done

## Risks & Edge Cases
- Images may fail to load (corrupted/missing) — log warning, skip
- Some joints may have coordinates outside image bounds — mark as `visible: false`
- The JSON file references `data/images/...` paths relative to the dataset root — need to resolve correctly
- Joint coords may be null/missing for some images — handle gracefully
- Dataset needs Ruby to download; adapter itself is pure Python and works on pre-downloaded data

## Open Questions
- None — approach is clear and follows established patterns

## Implementation Notes
- Dataset actually has **22 joints** (not 9 as issue stated): head, neck, nose_tip, nose_root, body_upper, arm_l/r, elbow_l/r, wrist_l/r, thumb_l/r, leg_l/r, knee_l/r, ankle_l/r, tiptoe_l/r
- 15 of 22 map to Strata; 7 discarded (nose_tip, nose_root, thumb_l/r, tiptoe_l/r)
- 4 Strata joints unmapped (spine, hips, shoulder_l, shoulder_r) — marked `visible: false`
- Adapter resolves image paths flexibly: tries direct, strips `data/` prefix, tries parent dir
- `_resize_to_strata` returns scale+offset tuple for joint coordinate transform
- Split fallback: loads `data.json` if present, else loads train/val/test.json individually with split tags
- 35 tests covering all adapter functions, all passing
- Registered as `--adapter anime_drawings` in `run_ingest.py`
- Checklist PP-12 updated to reflect adapter built status

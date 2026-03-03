# Issue #157: Build Meta Animated Drawings ingest adapter (178K illustrated figures, MIT)

## Understanding
- Convert Meta's Amateur Drawings dataset (178K hand-drawn figures) into Strata training format
- Dataset is COCO format: `amateur_drawings_annotations.json` + `amateur_drawings/{0-f}/*.png`
- Each image has exactly 1 annotation with 17 COCO keypoints (all visible, vis=2), polygon segmentation, bbox
- 17 keypoints: nose, left/right_eye, left/right_ear, left/right_shoulder, left/right_elbow, left/right_wrist, left/right_hip, left/right_knee, left/right_ankle
- Image sizes vary widely (91–9000 px width, 173–12000 px height)
- All polygon segmentation (no RLE), single polygon per annotation
- Type: new feature (ingest adapter)

## Dataset Structure
```
/Volumes/TAMWoolff/data/preprocessed/meta_animated_drawings/
├── amateur_drawings_annotations.json   # 288MB, COCO format
├── amateur_drawings/                   # 178K images in hex-sharded dirs (0-f)
│   ├── 0/{uuid}.png
│   ├── 1/{uuid}.png
│   └── ...f/{uuid}.png
└── amateur_drawings.tar                # 48GB original archive
```

## Joint Mapping (17 COCO → 19 Strata)

COCO keypoints → Strata regions:
- nose (0) → head (1)
- left_eye, right_eye, left_ear, right_ear → ignored (no Strata region; nose covers head)
- left_shoulder (5) → shoulder_l (6)
- right_shoulder (6) → shoulder_r (10)
- left_elbow (7) → forearm_l (8) — actually the upper_arm/forearm boundary
- right_elbow (8) → forearm_r (12)
- left_wrist (9) → hand_l (9)
- right_wrist (10) → hand_r (13)
- left_hip (11) → upper_leg_l (14) — use midpoint of L+R hips for hips(5)
- right_hip (12) → upper_leg_r (17)
- left_knee (13) → lower_leg_l (15)
- right_knee (14) → lower_leg_r (18)
- left_ankle (15) → foot_l (16)
- right_ankle (16) → foot_r (19)

Synthetic joints (interpolated):
- neck (2): midpoint of left_shoulder + right_shoulder
- chest (3): midpoint of neck and spine
- spine (4): midpoint of neck_pos and hips_pos
- hips (5): midpoint of left_hip + right_hip
- upper_arm_l (7): midpoint of left_shoulder + left_elbow
- upper_arm_r (11): midpoint of right_shoulder + right_elbow

Direct joints from COCO (13 unique):
- head(1), shoulder_l(6), forearm_l(8), hand_l(9), shoulder_r(10), forearm_r(12), hand_r(13), upper_leg_l(14), lower_leg_l(15), foot_l(16), upper_leg_r(17), lower_leg_r(18), foot_r(19)

## Approach
1. Load the COCO annotations JSON (build image_id → annotation + image info indexes)
2. For each annotation: load image, crop to bbox, resize to 512×512 centered
3. Rasterize COCO polygon segmentation → binary fg/bg mask (whole-body = no body-part decomposition)
4. Map 17 COCO keypoints → 19 Strata joints (with synthetic flag on interpolated ones)
5. Scale joint coordinates to match the resized 512×512 canvas
6. Save: image.png, segmentation.png, joints.json, metadata.json

Key design decisions:
- Crop to bbox first (drawings may have background), then resize to 512×512
- Use COCO polygon → binary mask (no body-part decomposition available)
- Mark interpolated joints with `"synthetic": true` in joints.json
- Use `has_fg_mask: True, has_segmentation_mask: False` (binary only)

## Files to Modify
- `ingest/animated_drawings_adapter.py` — NEW: main adapter module
- `run_ingest.py` — register `animated_drawings` adapter
- `tests/test_animated_drawings_adapter.py` — NEW: ≥10 tests

## Risks & Edge Cases
- 288MB JSON file — load once, index by image_id for O(1) lookup
- Very small images (91px wide) — may be poor quality; process anyway
- Very large images (9000+ px) — only interesting region is bbox area
- Polygon may have multiple sub-polygons (checked: first 5000 are single polygon, but handle list)
- Missing image file on disk — skip with warning
- All 17 keypoints are visible (vis=2) for every annotation — no visibility issues

## Open Questions
- None — dataset format is clear, mapping is specified in issue

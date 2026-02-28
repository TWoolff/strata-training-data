# Issue #119: Add 2D pose estimation enrichment for image-only datasets

## Understanding
- Image-only ingest datasets (FBAnimeHQ, anime-segmentation, NOVA-Human) lack joint annotations
- The existing `pipeline/joint_extractor.py` only works with Blender 3D armatures
- Need a pure-Python 2D pose estimation module that runs RTMPose via rtmlib/ONNX
- COCO 17-point keypoints must be mapped to Strata's 19-region skeleton
- This is a **new feature** — post-processing enrichment that runs after ingest adapters

## Approach
- Create `pipeline/pose_estimator.py` — pure Python module (no Blender dependency)
- Separate concerns: `coco_to_strata()` is a pure mapping function (easy to test without model)
- Model loading/inference via rtmlib ONNX interface
- `run_enrich.py` CLI walks ingest output directories, writes `joints.json` alongside `image.png`
- Output schema matches `joint_extractor.py`'s `save_joints()` format exactly
- Confidence inheritance: interpolated joints get `parent_confidence * 0.8`

### COCO → Strata mapping strategy
- 13 direct mappings (nose→head, shoulders, elbows→lower_arms, wrists→hands, hips midpoint→hips, knees→lower_legs, ankles→feet)
- 6 interpolated (neck, chest, spine, upper_arms, upper_legs)
- All 19 body regions covered; no background (region 0) in joints

## Files to Modify
| File | Action |
|---|---|
| `pipeline/pose_estimator.py` | **Create** — COCO→Strata mapping, model inference wrapper, joint data builder |
| `run_enrich.py` | **Create** — CLI entry point for batch enrichment |
| `tests/test_pose_estimator.py` | **Create** — unit tests for mapping logic (model-free) |
| `models/README.md` | **Create** — model download instructions |
| `.gitignore` | **Update** — add `models/*.onnx` |
| `requirements.txt` | **Update** — add `rtmlib`, `onnxruntime` |

## Risks & Edge Cases
- rtmlib API might differ from expected — design model interface to be swappable
- COCO keypoints with low confidence (< threshold) — mark those Strata joints as not visible
- Missing COCO keypoints (e.g., person partially off-screen) — interpolated joints from missing parents get confidence 0
- Image not containing a person — all joints invisible, confidence 0
- Multiple people in image — rtmlib returns multiple detections, we should pick the most prominent (largest bbox)
- Images already enriched — `--only_missing` flag for idempotency

## Open Questions
- None — the issue is well-specified with clear mapping tables and schema

## Implementation Notes
- Implemented exactly as planned — all files created/modified per the table above
- `coco_to_strata()` uses a local `joint()` helper to reduce repetition across 19 joint assignments
- `confidence_threshold` is threaded through the full call chain: CLI → `enrich_example` → `coco_to_strata` → `_make_joint`
- `estimate_pose()` picks the largest-bbox person when multiple are detected
- `enrich_example()` lazily imports `cv2` to avoid requiring OpenCV at module import time
- Tests mock cv2 via `patch.dict("sys.modules", ...)` since cv2 is lazily imported
- Output passes both `check_joint_count` and `check_joint_bounds` from `pipeline/validator.py`
- 37 unit tests, all passing; no regressions in full suite (1048 pass, 12 pre-existing sklearn failures)

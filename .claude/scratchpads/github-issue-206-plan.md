# Issue #206: Back view generation: model architecture + paired dataset preparation

## Understanding
- Strata needs a `back_view_generation.onnx` model that generates a back view from front + 3/4 view inputs
- Need: pair extraction script, U-Net model, dataset loader, config, tests
- Type: new feature (Model 6 in CLAUDE.md — Novel View Synthesis)

## Data Reality (differs from issue)
- **HumanRig**: Only has `front.png` per character — NO multi-angle renders. Issue's assumption of 11,434 multi-angle pairs is incorrect.
- **meshy_cc0_textured** (flat layout): 248 characters × multiple poses × ~31 angles each, with both `flat` and `textured` styles. Pattern: `{char}_pose_{nn}_{angle}_{style}.png`. Has front, three_quarter, back angles needed.
- **meshy_cc0_unrigged** (per-example layout): 142 characters × ~108 angles each. Pattern: `{char}_texture_{angle}/image.png`. Has front, three_quarter, back.
- **meshy_cc0** (flat layout): Same 248 chars, flat style only (no textured).
- Total paired examples: ~248 chars (textured, use `textured` style for realistic training) + 142 chars (unrigged). ~390 unique characters.

## Approach
- Build pair extraction to support BOTH flat layout (meshy_cc0_textured) and per-example layout (meshy_cc0_unrigged)
- For flat layout: parse filename to extract char_id, pose, angle, style — group by char+pose, pick front+three_quarter+back triplets
- For per-example layout: parse directory name to extract char_id, angle — group by char_id
- U-Net architecture: follow inpainting_model.py pattern exactly (same _DownBlock/_UpBlock building blocks, same skip connection pattern)
- 8-channel input (front RGBA + 3/4 RGBA concatenated), 4-channel RGBA output
- Use textured style images (not flat) for training — we want the model to work on real painted characters

## Files to Create
- `training/models/back_view_model.py` — U-Net (reuse _DownBlock/_UpBlock pattern from inpainting_model.py)
- `training/data/back_view_dataset.py` — Dataset loader with synchronized augmentation
- `training/data/prepare_back_view_pairs.py` — Pair extraction from meshy datasets
- `training/configs/back_view.yaml` — Training config
- `tests/test_back_view_model.py` — Unit tests

## Risks & Edge Cases
- Some characters may be missing angles (report and skip)
- meshy_cc0_textured has `_pose_00_flat.png` as the "front" default (no angle suffix) — need to handle this
- Horizontal flip augmentation must flip all 3 views together AND swap L/R semantics if applicable
- Alpha channel matters — transparent background must be preserved

## Open Questions
- None — the data exists, the architecture is well-specified in the issue

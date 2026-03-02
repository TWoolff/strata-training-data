# Issue #132: Build weight prediction dataset, model, and training script

## Understanding

The issue requests a **per-vertex weight prediction** pipeline — fundamentally different from the existing image-based models (segmentation, joints). Instead of operating on images, this model operates on per-vertex feature tensors and predicts per-vertex bone weights.

**Type:** New feature (dataset + model + training script)

**Key distinction from existing WeightModel:** The current `training/models/weight_model.py` is an **image-based** model (DeepLabV3+ backbone, takes `[B, 3, 512, 512]` images, outputs per-pixel predictions). Issue #132 asks for a **per-vertex MLP** model using 1x1 Conv2d that takes `[B, 31, N, 1]` feature tensors. These are two different model architectures for the same task. The per-vertex model needs new files.

## Approach

### Architecture Decision: Per-vertex MLP via 1x1 Conv2d

The issue specifies using 1x1 Conv2d layers instead of Linear layers. This makes the model naturally handle variable vertex counts (the N dimension is treated like a spatial dimension), and matches the `[B, C, N, 1]` tensor layout the Rust runtime expects.

### Dataset: per-vertex features from pipeline weight JSON

The pipeline's `weight_extractor.py` outputs per-vertex `{position, weights}` dicts. We need to:
1. Load these JSON files
2. Compute bone distances from vertex positions + skeleton joint positions (from joints.json)
3. Zero out heat diffusion features (slots 22-29) — these are runtime-only
4. Construct 31-dim feature vectors matching the Rust `build_feature_tensor()` layout
5. Zero-pad to MAX_VERTICES=2048

### Loss: KL divergence for soft labels

Ground truth weights are soft distributions (e.g., `{hips: 0.6, spine: 0.4}`), not one-hot. Use KL divergence between predicted softmax and GT weight distribution, plus BCE for confidence.

### ONNX Contract

The per-vertex model has a **different** ONNX contract from the existing image-based weight model:
- Input `"input"`: `[1, 31, 2048, 1]`
- Output `"weights"`: `[1, 20, 2048, 1]` raw logits
- Output `"confidence"`: `[1, 1, 2048, 1]` raw logits

This matches the Rust `build_feature_tensor()` / `postprocess()` functions in weights.rs.

## Files to Modify

### New files:
- `training/data/weight_dataset.py` — WeightDataset class
- `training/models/weight_prediction_model.py` — WeightPredictionModel (1x1 Conv2d MLP)
- `training/train_weights.py` — Training script
- `tests/test_weight_dataset.py` — Dataset tests
- `tests/test_weight_prediction_model.py` — Model tests
- `tests/test_train_weights.py` — Training script tests

### Modified files:
- `training/configs/weights.yaml` — Update config for per-vertex model
- `training/utils/metrics.py` — Add WeightMetrics class
- `training/export_onnx.py` — Add per-vertex model export config

## Risks & Edge Cases

1. **Variable vertex counts**: Meshes vary from hundreds to tens of thousands of vertices. Pad to 2048, mask padded vertices in loss.
2. **Missing joint data**: Some examples may not have joints.json — skip or handle gracefully.
3. **Empty weights**: Some vertices may have no weight assignments — confidence target should be 0.0 for these.
4. **Weight normalization**: GT weights from pipeline may not sum to 1.0 per vertex due to thresholding — normalize before training.
5. **Bone distance computation**: Need joint positions from joints.json, not the 3D bone positions. The 2D projected positions are what we have.

## Open Questions

- None — the issue is very detailed with clear specifications.

## Implementation Notes

### What was implemented

All acceptance criteria from the issue have been addressed:

1. **`training/data/weight_dataset.py`** — `WeightDataset` class that:
   - Loads pipeline weight JSON files (flat and per-example layouts)
   - Constructs 31-dim feature vectors matching `weights.rs` `build_feature_tensor()` layout
   - Computes bone distances from vertex positions + skeleton joint positions (joints.json)
   - Heat diffusion features (slots 22-29) zeroed — runtime-only data
   - Zero-pads to MAX_VERTICES=2048
   - GT weights normalized to sum to 1.0 per vertex
   - Confidence target = 1.0 for vertices with GT data, 0.0 otherwise

2. **`training/models/weight_prediction_model.py`** — `WeightPredictionModel` class:
   - Per-vertex MLP via 1x1 Conv2d (shared weights across vertex dimension)
   - Architecture: [31→128→256(+dropout)→128→20] for weights, [31→64→1] for confidence
   - Outputs raw logits (softmax/sigmoid applied at inference by Rust runtime)
   - Output shapes: weights `[B, 20, N, 1]`, confidence `[B, 1, N, 1]`

3. **`training/train_weights.py`** — Training script:
   - KL divergence loss for soft weight labels (GT weights are distributions, not one-hot)
   - BCE loss for per-vertex confidence
   - Masked loss computation: only real vertices with GT data contribute to weight loss
   - Full training loop with warmup, cosine LR, early stopping, TensorBoard, checkpoints

4. **`training/utils/metrics.py`** — Added `WeightMetrics` class:
   - Per-bone MAE, overall MAE, confidence accuracy
   - Respects num_vertices to exclude padded region

5. **`training/configs/weights.yaml`** — Updated for per-vertex model config

6. **`training/export_onnx.py`** — Added `weights_vertex` model config:
   - `WeightPredictionWrapper` for ONNX export
   - Custom input shape `[1, 31, 2048, 1]`
   - Validation function for per-vertex output shapes

### Design decisions

- **Separate model file**: Named `weight_prediction_model.py` (not overwriting `weight_model.py`) since the existing image-based model is still valid and used by ONNX export
- **ONNX model name**: `weights_vertex` in MODEL_CONFIGS to distinguish from the image-based `weights` model
- **Loss function**: KL divergence (not CrossEntropy) because GT labels are soft distributions, and KL is the natural divergence measure between probability distributions
- **No augmentation**: Per-vertex data doesn't benefit from image augmentations (no color jitter, flipping, etc.)

### Code simplifier pass

Applied simplifications (commit `db99b5a`):
- Removed unused `re` import and `_STYLE_SUFFIXES` regex constant from `weight_dataset.py`
- Hoisted `PIPELINE_TO_BONE` import to top-level (was deferred inside `_parse_joint_positions()`)
- Removed unused `_img_w, _img_h` unpacking in `build_features()`
- Simplified `bone_positions` construction from 5-line loop to 1-line list comprehension
- Added `weights_vertex` example to `export_onnx.py` usage docstring
- `weight_prediction_model.py`, `train_weights.py`, and `metrics.py` were already clean — no changes needed

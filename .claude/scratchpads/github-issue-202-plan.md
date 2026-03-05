# Issue #202: Diffusion weight prediction — model architecture + dataset loader

## Understanding
- Build Model 4 of Strata's 7 ONNX models: an enhanced weight prediction MLP
- Takes standard 31-dim vertex features (same as model 3) PLUS encoder features from segmentation model 1
- Encoder features provide visual/semantic context → better weights for unusual proportions (chibi, stylized)
- Need: model architecture, dataset loader, encoder feature precomputation script, configs

## Approach
- Follow exact patterns from `weight_prediction_model.py` and `weight_dataset.py`
- Model: concatenate vertex features [31] + encoder features [C] → wider MLP → same outputs
- Dataset: extend WeightDataset to also load precomputed .npy encoder feature files
- Precompute script: load trained segmentation model, extract backbone features, bilinear sample at vertex positions, save per-example .npy
- Keep the standard weight prediction loss (KL div + confidence BCE)

### Architecture Decision
- Use 1x1 Conv2d MLP pattern (matches weight_prediction_model.py exactly)
- Fusion by channel concatenation: [B, 31+C, N, 1] → shared MLP
- C = 960 (MobileNetV3-Large backbone output channels at /16 resolution)
- Issue suggests 31+C→256→128→64→heads. This is smaller than model 3 (31→128→256→128→20).
  I'll use: 31+C→256→256→128→heads to give more capacity for the larger input.

### Encoder Feature Extraction
- Segmentation model outputs backbone features at [B, 960, H/16, W/16] = [1, 960, 32, 32]
- For each vertex at normalized (x, y), bilinear sample from feature map
- Result: [N, 960] per example → save as .npy
- At training time: load .npy, reshape to [960, N, 1], pad to max_vertices

## Files to Create
- `training/models/diffusion_weight_model.py` — DiffusionWeightPredictionModel
- `training/data/diffusion_weight_dataset.py` — DiffusionWeightDataset
- `training/data/precompute_encoder_features.py` — CLI script
- `training/configs/diffusion_weights.yaml` — local config
- `training/configs/diffusion_weights_a100_lean.yaml` — lean A100 config
- `training/configs/diffusion_weights_a100.yaml` — full A100 config
- `training/train_diffusion_weights.py` — training script (issue #203 scope but natural to do together)

## Files to Modify
- `training/export_onnx.py` — add diffusion_weights export entry
- `training/train_all.sh` — add diffusion weights training step (optional, after model 1 completes)

## Risks & Edge Cases
- Encoder channel count (960) is hardcoded assumption for MobileNetV3-Large. Should detect dynamically.
- Precomputed features are tied to a specific segmentation checkpoint — if seg model is retrained, features must be recomputed.
- Variable vertex counts: must pad encoder features to max_vertices like vertex features.
- Some examples may not have corresponding images (weight-only data) — skip those during precomputation.

## Open Questions
- None — issue is well-specified, patterns are clear from existing code.

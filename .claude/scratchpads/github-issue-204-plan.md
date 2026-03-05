# Issue #204: Inpainting model — U-Net architecture + synthetic occlusion dataset

## Understanding
- Build Model 5 of Strata's 7 ONNX models: a U-Net that fills occluded body-part regions
- Need: model architecture, synthetic training data generator, dataset loader, training script, ONNX export
- Type: new feature (no existing code for this model)

## ONNX Contract (from Rust runtime — AUTHORITATIVE)
- **Dual-input** via `infer_dual_input()`:
  - Input `"image"`: `[1, 4, 512, 512]` float32 (RGBA, 0-1 normalized)
  - Input `"mask"`: `[1, 1, 512, 512]` float32 (1.0 = occluded/to-fill, 0.0 = visible)
- **Output** `"inpainted"`: `[1, 4, 512, 512]` float32 (completed RGBA)
- Note: Issue spec says 5-channel concatenated input, but Rust runtime sends separate tensors.
  The model will accept separate tensors and concatenate internally (wrapper handles ONNX export).

## Approach

### Model Architecture
- Lightweight U-Net encoder-decoder with skip connections
- Input: concat of image[4ch] + mask[1ch] = 5ch internally
- ONNX wrapper accepts two separate inputs, concatenates, runs model
- Encoder: 5 down-blocks (stride-2 conv), 64→128→256→512→512
- Decoder: 5 up-blocks (transposed conv + skip concat), mirrors encoder
- Output: 4ch RGBA via Sigmoid

### Training Data Generation
- Source: fbanimehq (~101K full-body anime images) — best source, diverse illustrated style
- 3 occlusion strategies: rectangular cutouts, irregular brush masks, elliptical blobs
- Region-based masking deferred (requires segmentation masks, not always available)
- Generate ~3 masks per image → ~300K pairs
- Pregenerate to disk as (masked.png, mask.png, target.png) triplets

### Loss Function
- L1 reconstruction loss (primary)
- Perceptual loss via VGG16 features (secondary, weight 0.1)
- No adversarial loss in v1 (simplicity)

## Files to Create
- `training/models/inpainting_model.py` — U-Net + ONNX wrapper
- `training/data/generate_occlusion_pairs.py` — CLI script
- `training/data/inpainting_dataset.py` — Dataset loader
- `training/train_inpainting.py` — Training loop
- `training/configs/inpainting.yaml` — Local config
- `training/configs/inpainting_a100.yaml` — Full A100 config
- `training/configs/inpainting_a100_lean.yaml` — Lean A100 config

## Files to Modify
- `training/export_onnx.py` — Add inpainting to MODEL_CONFIGS
- `training/train_all.sh` — Add inpainting step

## Risks & Edge Cases
- U-Net at 512×512 is memory-heavy — batch size 4-8 on A100
- fbanimehq images vary in size/aspect — need resize + center crop to 512
- Some images may have no character (blank/error) — skip those
- Mask shouldn't cover entire image — cap at 50% coverage
- Perceptual loss needs VGG16 — lazy-load to avoid import overhead

## Open Questions
- Should we also generate training data from segmentation/ and anime_seg/? → Yes, more diversity
- Do we need to handle alpha channel specially? → Masked pixels get alpha=0

## Implementation Notes
- **Critical ONNX contract difference**: Issue spec said 5-channel concatenated input, but Rust
  runtime uses `infer_dual_input()` with separate `"image"` [1,4,512,512] and `"mask"` [1,1,512,512].
  Model concatenates internally, ONNX wrapper exports as dual-input.
- U-Net has ~13.5M parameters (lightweight for 512x512)
- 3 mask strategies implemented: rect, irregular (random walk), ellipse
- Dataset splits by hash of directory name (no character-level split needed since pairs are synthetic)
- Perceptual loss uses VGG16 features[:16] with lazy loading
- train_all.sh now has 8 steps (was 6): added occlusion pair generation + inpainting training
- Occlusion pair generation runs from fbanimehq (~101K images) on the cloud instance
- Estimated training time: ~1h lean, ~2-3h full on A100

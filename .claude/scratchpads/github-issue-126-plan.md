# Issue #126: Implement multi-head DeepLabV3+ segmentation model

## Understanding
- New feature: PyTorch model definition for Strata's core multi-head segmentation network
- Must match ONNX contract from `strata/src-tauri/src/ai/segmentation.rs`
- Three output heads sharing a MobileNetV3-Large backbone:
  1. Segmentation: 22 classes (raw logits)
  2. Draw order: 1 channel (sigmoid)
  3. Confidence: 1 channel (sigmoid)
- Input: `[B, 3, 512, 512]` float32

## Approach
- Use `torchvision.models.segmentation.deeplabv3_mobilenet_v3_large` as the base
- Extract backbone and ASPP classifier from the pretrained model
- Add two lightweight auxiliary heads (draw_order, confidence) that branch from backbone features
- Each aux head: Conv2d(960→256, 3×3, pad=1) → BN → ReLU → Conv2d(256→1, 1×1)
- Safety check: verify backbone output channels at init, adapt if API changes
- All outputs bilinear-upsampled to input resolution

## Files to Modify
- **New**: `training/models/segmentation_model.py` — Model definition
- **Edit**: `training/models/__init__.py` — Export SegmentationModel
- **New**: `tests/test_segmentation_model.py` — Unit tests

## Risks & Edge Cases
- torchvision API may change the backbone output key or channel count — mitigated by runtime assertion
- The DeepLabV3 classifier internally upsamples; we need to verify it produces input-resolution output
- ONNX export compatibility — sigmoid should be part of the model forward for draw_order/confidence
- Memory: 512×512 input with full backbone may use significant GPU memory at batch_size=8

## Open Questions
- None — issue is well-specified with exact architecture and contract

## Implementation Notes

### What was implemented
- `training/models/segmentation_model.py`: `SegmentationModel(nn.Module)` with three output heads
- `training/models/__init__.py`: Re-exports `SegmentationModel`
- `tests/test_segmentation_model.py`: 9 tests covering instantiation, output shapes, value ranges, and backbone detection

### Design decisions
- **DeepLabV3 classifier does NOT auto-upsample**: The ASPP classifier outputs at backbone resolution (H/16 × W/16). We manually `F.interpolate` all three heads to input resolution. This was verified empirically.
- **Sigmoid in forward()**: Draw order and confidence heads apply `torch.sigmoid()` in the forward pass, matching the ONNX contract where the Rust runtime expects values in [0, 1].
- **`_detect_backbone_channels` via dry run**: Uses a 64×64 dummy tensor to detect channel count at init, making the model robust to torchvision API changes.
- **`_make_aux_head` factory**: Shared function builds both aux heads (Conv→BN→ReLU→Conv), reducing duplication.
- **`pretrained_backbone=False` in tests**: Avoids downloading ImageNet weights during CI.

### Verified
- All 9 tests pass with torch 2.10.0 / torchvision 0.25.0
- Ruff lint and format clean
- Backbone outputs 960 channels as expected for MobileNetV3-Large

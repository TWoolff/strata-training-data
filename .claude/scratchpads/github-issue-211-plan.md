# Issue #211: Texture inpainting training script + ONNX export

## Understanding
- Issue #210 (COMPLETE) built the model architecture, dataset loader, training script, configs, and ONNX export wrapper
- Issue #211 asks for integration improvements: mixed precision, SSIM metrics, lean A100 config, train_all.sh + cloud_setup.sh integration, TensorBoard logging, resume support
- Type: enhancement of existing feature

## Approach
- Upgrade `train_texture_inpainting.py` to match `train_back_view.py` quality (TensorBoard, resume, mixed precision, SSIM)
- Add missing `texture_inpainting_a100_lean.yaml` config
- Add texture inpainting steps to `train_all.sh` and `cloud_setup.sh`

## Files to Modify
1. `training/train_texture_inpainting.py` — add mixed precision (GradScaler), SSIM metric, TensorBoard logging, visual comparison, resume from checkpoint
2. `training/configs/texture_inpainting_a100_lean.yaml` — new lean A100 config
3. `training/train_all.sh` — add texture inpainting training step + ONNX export
4. `training/cloud_setup.sh` — add texture_pairs dataset download

## Risks & Edge Cases
- SSIM requires torchmetrics or manual implementation — use torch.nn.functional based SSIM to avoid new dependency
- Mixed precision + VGG perceptual loss needs autocast scoping to avoid issues
- train_all.sh step numbering needs updating

## Open Questions
- None — all patterns established by existing code

## Implementation Notes
- Issue #210 already created the core model/dataset/training script/ONNX export
- Enhanced training script with: mixed precision (fp16 via GradScaler), SSIM metric, TensorBoard + visual comparisons, checkpoint resume, MPS device support
- Added `texture_inpainting_a100_lean.yaml` (batch_size=8, epochs=60, mixed_precision=true)
- Added `mixed_precision: true` to existing `texture_inpainting_a100.yaml`
- Added texture inpainting as step [8/9] in `train_all.sh` + ONNX export
- Added `texture_pairs` dataset download to `cloud_setup.sh`
- SSIM implemented inline (no new dependency) using Gaussian-windowed conv2d on RGB channels
- All 14 existing tests pass, lint clean

# Issue #207: Back view generation — training script + ONNX export

## Understanding
- Training loop, ONNX export, and evaluation for the back view generation model (Model 6)
- Depends on #206 (architecture + dataset) which is COMPLETE
- Existing: `back_view_model.py`, `back_view_dataset.py`, `prepare_back_view_pairs.py`, `back_view.yaml`, tests
- Missing: `train_back_view.py`, ONNX wrapper/registry in `export_onnx.py`, evaluation in `evaluate.py`

## Approach
- Follow `train_inpainting.py` pattern closely — same L1 + perceptual loss structure
- Add palette consistency loss (novel: histogram matching between 3/4 view and predicted back)
- Add TensorBoard logging with visual comparisons (front | 3/4 | predicted | GT)
- ONNX: single input "input" [1,8,512,512] → single output "output" [1,4,512,512]
- Evaluation: L1, PSNR, SSIM, palette consistency, visual comparison grid

## Files to Modify
- `training/train_back_view.py` — NEW: training script (main deliverable)
- `training/export_onnx.py` — Add BackViewWrapper + registry entry + validation
- `training/evaluate.py` — Add evaluate_back_view() + CLI integration
- `tests/test_back_view.py` — Add tests for training, export, evaluation

## Risks & Edge Cases
- Palette loss: histogram computation must handle transparent pixels (only count non-zero alpha)
- Memory: U-Net at 512×512 batch_size=4 is heavy — gradient accumulation not needed per config
- VGG perceptual: reuse same PerceptualLoss class from inpainting (could extract but keep simple)
- TensorBoard image logging: clamp to [0,1] before writing

## Open Questions
- None — issue spec is very detailed and all dependencies are met

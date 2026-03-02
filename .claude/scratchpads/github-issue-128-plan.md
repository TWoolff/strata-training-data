# Issue #128: Build segmentation model training script with multi-head loss

## Understanding
- Create the main training script that orchestrates training the multi-head DeepLabV3+ model
- Two new files: `training/train_segmentation.py` (training loop) and `training/utils/checkpoint.py` (save/load/early stopping)
- Uses existing: `SegmentationModel`, `SegmentationDataset`, `SegmentationMetrics`, `segmentation.yaml`
- Type: new feature

## Approach
- **training/utils/checkpoint.py**: Simple checkpoint save/load + EarlyStopping class
- **training/train_segmentation.py**: CLI entry point with:
  - YAML config loading
  - Dataset/DataLoader creation (train + val)
  - Model, optimizer, scheduler (cosine + warmup) setup
  - Multi-head loss (CE + L1 + BCE with configurable weights)
  - Class weight computation via median frequency balancing
  - Training loop with val, TensorBoard logging, checkpointing, early stopping
  - `--resume` flag for continuing from a checkpoint

### Key design decisions:
1. **Linear warmup**: Implement as a manual LR schedule wrapper over CosineAnnealingLR — multiply LR by `epoch / warmup_epochs` during warmup phase
2. **Class weights**: Scan training set labels once before training, compute median frequency balancing
3. **has_draw_order handling**: Dataset returns bool per-example; collate into a boolean tensor per batch, mask draw order loss accordingly
4. **TensorBoard**: Use `torch.utils.tensorboard.SummaryWriter`; log per-step losses, per-epoch metrics, sample overlays every N epochs
5. **Device**: Auto-detect CUDA/MPS/CPU

## Files to Modify
- **NEW** `training/utils/checkpoint.py` — save_checkpoint, load_checkpoint, EarlyStopping
- **NEW** `training/train_segmentation.py` — main training script
- **MODIFY** `training/utils/__init__.py` — export EarlyStopping
- **NEW** `tests/test_training_checkpoint.py` — tests for checkpoint module
- **NEW** `tests/test_train_segmentation.py` — tests for training script functions

## Risks & Edge Cases
- Empty dataset (0 examples in train or val) — guard with early exit + warning
- All class weights zero if no training labels found — fallback to uniform weights
- `has_draw_order` is a Python bool, not a tensor — need custom collate or handle in loss
- MPS backend may not support all ops — fallback to CPU
- Large class imbalance (background dominates) — median frequency balancing mitigates

## Open Questions
- None — issue is very well-specified

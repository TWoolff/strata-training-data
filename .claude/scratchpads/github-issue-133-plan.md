# Issue #133: Build evaluation script and visualization utilities

## Understanding
- Build CLI tools to evaluate trained models on the test set and visualize predictions
- Three model types: segmentation, joints, weights
- Need per-class IoU tables, confusion matrix heatmaps, worst-performing examples, prediction overlay grids
- Joint evaluation needs per-joint MSE, presence accuracy, scatter plots
- Weight evaluation needs per-bone MAE, confidence accuracy
- All visualizations should use REGION_COLORS from pipeline/config.py for consistency
- Summary JSON for CI tracking, optional --min-miou threshold for exit code

## Approach
- **visualization.py**: Standalone utility functions using matplotlib + numpy + PIL
  - `overlay_segmentation` — color-codes mask using REGION_COLORS and blends with image
  - `save_prediction_grid` — triptych grid: image | GT mask | predicted mask
  - `plot_confusion_matrix` — matplotlib heatmap with region names
  - `plot_per_class_iou` — horizontal bar chart
  - `overlay_joints` — draw circles on image
  - `save_joint_comparison` — overlay GT (green) vs predicted (red) joints
- **evaluate.py**: CLI script with `--model {segmentation,joints,weights}` and `--all` mode
  - Loads checkpoint, builds model + test dataset, runs inference
  - Computes metrics using existing `SegmentationMetrics`, `JointMetrics`, `WeightMetrics`
  - For segmentation: tracks per-example IoU for worst-N identification
  - Outputs: console table, JSON summary, PNG visualizations
  - `--min-miou` flag for CI: exit code 1 if below threshold

### Key design decisions:
- Reuse existing metrics classes (no duplication)
- Reuse `select_device()` pattern from train_segmentation.py
- Reuse `collate_fn` from respective training scripts
- Weight evaluation uses `WeightPredictionModel` (per-vertex), not `WeightModel` (per-pixel)
- Keep matplotlib import lazy where possible (heavy import)
- For worst-N segmentation: compute per-example IoU during eval loop, sort at end

## Files to Modify
- **NEW** `training/utils/visualization.py` — All visualization functions
- **NEW** `training/evaluate.py` — CLI evaluation script
- **NEW** `tests/test_evaluate.py` — Tests for both modules

## Risks & Edge Cases
- Empty test set (0 examples) — should exit gracefully with warning
- Model checkpoint not found — clear error message
- No GPU available — must work on CPU
- Classes with 0 GT pixels in test set — per_class_iou returns 0.0 (handled by SegmentationMetrics)
- Joint model outputs offsets in [B, 2, 20] layout (needs transpose for JointMetrics)
- Weight model outputs [B, 20, N, 1] (needs squeeze)

## Open Questions
- None — the issue is well-specified and all dependencies exist

## Implementation Notes
- Implemented as planned with no significant deviations
- `visualization.py` (376 lines): 9 public functions covering segmentation overlays, confusion matrix, IoU charts, joint overlays/scatter/comparison, per-joint/per-bone error charts
- `evaluate.py` (708 lines): Three evaluation functions (segmentation, joints, weights) + CLI with --model, --all, --min-miou, --worst-n flags
- Uses cached color LUT (_COLOR_LUT) for efficient mask colorization
- matplotlib imports are lazy (inside functions) to keep module import fast
- cv2 imports also lazy for the same reason
- PIL used for save_joint_comparison (simple save, no matplotlib needed)
- Tests: 19 total (9 pass without deps, 6 skip without matplotlib, 4 skip without torch)
- Collate functions duplicated from training scripts rather than importing them (keeps evaluate.py self-contained and avoids circular dependency risk)
- Worst-N tracking uses a sorted list with O(N log N) insertion; fine for typical worst_n=16

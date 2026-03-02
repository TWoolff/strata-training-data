# Issue #129: Build ONNX export pipeline for all three models

## Understanding
- Need to export trained PyTorch models to ONNX format with exact tensor names matching Strata Rust runtime
- PyTorch models return dicts from `forward()`, but ONNX requires tuple outputs → need wrapper classes
- Three models: segmentation (exists), joint refinement (needs creation), weight prediction (needs creation)
- Post-export validation with onnxruntime to verify names/shapes/ranges
- CLI with `--model` single export and `--all` batch export modes
- Type: new feature

## Approach
1. **Create stub models for joints and weights** — The joint and weight models don't exist yet. I'll create minimal but architecturally correct models following the segmentation model's pattern (MobileNetV3-Large backbone + task-specific heads). These are needed so the export pipeline can be tested end-to-end.

2. **Create `training/export_onnx.py`** — Single CLI script that:
   - Loads a checkpoint via `load_checkpoint()`
   - Wraps the model in an ONNX-friendly wrapper that returns tuples instead of dicts
   - Calls `torch.onnx.export()` with correct names, dynamic axes, opset 17
   - Runs onnxruntime validation immediately after export
   - Supports `--model {segmentation,joints,weights}` and `--all` modes

3. **ONNX contracts** (from issue):
   - Segmentation: input `[1,3,512,512]` → `segmentation[1,22,512,512]`, `draw_order[1,1,512,512]`, `confidence[1,1,512,512]`
   - Joint refinement: input `[1,3,512,512]` → `offsets[40]`, `confidence[20]`, `present[20]`
   - Weight prediction: input `[1,3,512,512]` → `weights[1,20,2048,1]`, `confidence[1,1,2048,1]`

4. **Joint model design** — Based on JointMetrics (which expects `pred_offsets[B,J,2]` and `pred_present[B,J]`) and the ONNX contract (`offsets[40]`, `confidence[20]`, `present[20]`):
   - Backbone → FC layers → flat offsets (20 joints × 2 = 40), confidence (20), present (20)
   - The ONNX output is unbatched flat tensors

5. **Weight model design** — Based on ONNX contract (`weights[1,20,2048,1]`, `confidence[1,1,2048,1]`):
   - Backbone → per-point weight prediction head
   - N=2048 is the vertex count (dynamic axis)
   - Output: softmax weights over 20 bones per vertex, plus confidence

## Files to Modify
- `training/models/joint_model.py` — NEW: joint refinement model
- `training/models/weight_model.py` — NEW: weight prediction model
- `training/models/__init__.py` — Add JointModel, WeightModel exports
- `training/export_onnx.py` — NEW: ONNX export CLI
- `tests/test_export_onnx.py` — NEW: tests for export pipeline

## Risks & Edge Cases
- Joint model ONNX contract specifies flat `[40]`, `[20]`, `[20]` (no batch dim) — wrapper must squeeze batch
- Weight model has dynamic vertex count N — must set dynamic_axes correctly
- `torch.onnx.export` tracing may not handle dict returns — wrapper classes solve this
- Models without pretrained checkpoints should still export (random weights) for testing
- `onnxruntime` must be available for post-export validation

## Open Questions
- None — the issue spec is very detailed with exact tensor contracts

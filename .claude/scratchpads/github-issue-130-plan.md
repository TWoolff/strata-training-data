# Issue #130: Build ONNX validation script against Rust runtime contracts

## Understanding
- Standalone CLI script that validates exported ONNX models against the exact tensor contracts expected by the Strata Rust runtime
- Catches integration issues (wrong tensor names, shapes, value ranges) before models reach the app
- Three model types: segmentation, joint_refinement, weight_prediction
- Optional real-image cross-validation mode
- CI-friendly: exit code 0 on pass, 1 on fail

## Approach
- Create `training/validate_onnx.py` as a standalone script (separate from the inline `validate_onnx()` in `export_onnx.py`)
- The inline validation in `export_onnx.py` is tightly coupled to the export flow; this new script validates any ONNX file on disk, including models from other sources
- Use dataclass-based model specs for clean contract definitions
- Print clear pass/fail report with checkmarks
- Return sys.exit(1) on any failure for CI

## Rust Runtime Contracts (from src-tauri/src/ai/)

**Segmentation (segmentation.onnx):**
- Input: "input" [1, 3, 512, 512]
- Outputs: "segmentation" [1, 22, 512, 512], "draw_order" [1, 1, 512, 512], "confidence" [1, 1, 512, 512]
- Optional: "encoder_features" [1, C, H, W]
- Argmax of segmentation → values 0-21
- draw_order/confidence are sigmoid [0, 1]
- File size: 20-80 MB

**Joint Refinement (joint_refinement.onnx):**
- Input: "input" [1, 3, 512, 512]
- Outputs: "offsets" [40], "confidence" [20], "present" [20]
- Batch dim squeezed in ONNX wrapper
- confidence is sigmoid [0, 1]
- File size: 1-15 MB

**Weight Prediction (weight_prediction.onnx):**
- Input: "input" [1, 3, 512, 512]
- Outputs: "weights" [1, 20, N, 1], "confidence" [1, 1, N, 1]
- N = 512*512 = 262144 (default resolution)
- weights are softmax (>=0, sum to ~1 over bone dim)
- confidence is sigmoid [0, 1]
- File size: 3-30 MB

**ImageNet Normalization (Rust preprocess):**
- mean = [0.485, 0.456, 0.406]
- std = [0.229, 0.224, 0.225]

## Files to Modify
- **New**: `training/validate_onnx.py` — Main validation script
- **New**: `tests/test_validate_onnx.py` — Tests including mock ONNX with wrong names → fail

## Risks & Edge Cases
- ONNX files from different export pipelines may have slightly different shapes (dynamic axes)
- Weight model has dynamic vertex dimension — validate the known dimension (bone count) not the dynamic one
- Segmentation model may optionally include "encoder_features" output — should not fail if present
- File size bounds are estimates; should be configurable or have generous tolerances
- Real-image validation requires PIL/numpy for preprocessing

## Open Questions
- None — the issue spec is very detailed

## Implementation Notes
- Used dataclass-based `ModelSpec` and `OutputSpec` for clean, declarative contract definitions
- `OutputSpec` supports: exact shape, ndim + partial dim checks, value range, non-negative
- Weight model uses partial dim checks (ndim=4, specific dims 0,1,3) because dim 2 is dynamic (vertices)
- Added `check_file_size` parameter to `validate_model()` and `--no-check-file-size` CLI flag — needed because test mock ONNX models are tiny (few KB) vs real models (MB)
- Segmentation-specific checks: argmax range 0-21, draw_order finite values
- Cross-validation loads real images, applies ImageNet normalization matching Rust runtime, checks for degenerate output (>95% single region, <2 distinct regions)
- Test mock ONNX models use ConstantOfShape ops to produce correctly-shaped tensors without needing real model weights
- 15 tests covering: valid models pass, wrong names fail, missing files fail, CLI pass/fail, --all mode, cross-validation with images and empty dirs, report format
- All tests pass, lint clean

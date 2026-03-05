#!/usr/bin/env bash
# =============================================================================
# Strata Training — Train All Models + Export ONNX
# Run this after cloud_setup.sh completes. Trains all 4 models sequentially,
# exports to ONNX, and packages results for download.
#
# Usage:
#   chmod +x training/train_all.sh
#   ./training/train_all.sh          # Full A100 configs (~8-12h)
#   ./training/train_all.sh lean     # Lean A100 configs, core data only (~3-4h)
#   ./training/train_all.sh local    # Default configs (for 4070 Ti / local)
# =============================================================================
set -euo pipefail

MODE="${1:-a100}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/train_${TIMESTAMP}"
ONNX_DIR="./models/onnx"
mkdir -p "$LOG_DIR" "$ONNX_DIR"

# Select configs based on mode
if [ "$MODE" = "local" ]; then
    SEG_CONFIG="training/configs/segmentation.yaml"
    JOINT_CONFIG="training/configs/joints.yaml"
    WEIGHT_CONFIG="training/configs/weights.yaml"
    echo "Using LOCAL configs (default batch sizes)"
elif [ "$MODE" = "lean" ]; then
    SEG_CONFIG="training/configs/segmentation_a100_lean.yaml"
    JOINT_CONFIG="training/configs/joints_a100_lean.yaml"
    WEIGHT_CONFIG="training/configs/weights_a100_lean.yaml"
    echo "Using LEAN A100 configs (core data only, ~3-4h)"
else
    SEG_CONFIG="training/configs/segmentation_a100.yaml"
    JOINT_CONFIG="training/configs/joints_a100.yaml"
    WEIGHT_CONFIG="training/configs/weights_a100.yaml"
    echo "Using FULL A100 configs (all data, ~8-12h)"
fi

echo ""
echo "============================================"
echo "  Strata Training — All Models"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Model 1: Segmentation (DeepLabV3+ multi-head)
# ---------------------------------------------------------------------------
echo "[1/4] Training SEGMENTATION model..."
echo "  Config: $SEG_CONFIG"
echo "  Log: $LOG_DIR/segmentation.log"
echo ""

python -m training.train_segmentation \
    --config "$SEG_CONFIG" \
    2>&1 | tee "$LOG_DIR/segmentation.log"

echo ""
echo "  Segmentation training complete."
echo ""

# ---------------------------------------------------------------------------
# Model 2: Joint Refinement
# ---------------------------------------------------------------------------
echo "[2/4] Training JOINT REFINEMENT model..."
echo "  Config: $JOINT_CONFIG"
echo "  Log: $LOG_DIR/joints.log"
echo ""

python -m training.train_joints \
    --config "$JOINT_CONFIG" \
    2>&1 | tee "$LOG_DIR/joints.log"

echo ""
echo "  Joint training complete."
echo ""

# ---------------------------------------------------------------------------
# Model 3: Weight Prediction (per-vertex MLP)
# ---------------------------------------------------------------------------
echo "[3/4] Training WEIGHT PREDICTION model (per-vertex MLP)..."
echo "  Config: $WEIGHT_CONFIG"
echo "  Log: $LOG_DIR/weights.log"
echo ""

python -m training.train_weights \
    --config "$WEIGHT_CONFIG" \
    2>&1 | tee "$LOG_DIR/weights.log"

echo ""
echo "  Weight prediction training complete."
echo ""

# ---------------------------------------------------------------------------
# Model 4: ONNX Export
# ---------------------------------------------------------------------------
echo "[4/4] Exporting all models to ONNX..."
echo "  Output: $ONNX_DIR/"
echo ""

# Export segmentation
python -m training.export_onnx \
    --model segmentation \
    --checkpoint checkpoints/segmentation/best.pt \
    --output "$ONNX_DIR/segmentation.onnx" \
    2>&1 | tee -a "$LOG_DIR/export.log"

# Export joints
python -m training.export_onnx \
    --model joints \
    --checkpoint checkpoints/joints/best.pt \
    --output "$ONNX_DIR/joint_refinement.onnx" \
    2>&1 | tee -a "$LOG_DIR/export.log"

# Export weights (per-vertex MLP)
python -m training.export_onnx \
    --model weights_vertex \
    --checkpoint checkpoints/weights/best.pt \
    --output "$ONNX_DIR/weight_prediction_vertex.onnx" \
    2>&1 | tee -a "$LOG_DIR/export.log"

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Finished: $(date)"
echo ""
echo "  ONNX models:"
ls -lh "$ONNX_DIR/"*.onnx 2>/dev/null || echo "  (no ONNX files found)"
echo ""
echo "  Checkpoints:"
ls -lh checkpoints/*/best.pt 2>/dev/null || echo "  (no checkpoints found)"
echo ""
echo "  To download results to your Mac:"
echo "    scp -r <cloud-host>:$(pwd)/$ONNX_DIR/ ./models/"
echo "============================================"

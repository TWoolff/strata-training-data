#!/usr/bin/env bash
# =============================================================================
# Strata Training — Train All Models + Export ONNX
# Run this after cloud_setup.sh completes. Trains models 1-5 sequentially,
# exports to ONNX, and packages results for download.
#
# Models trained:
#   1. Segmentation (DeepLabV3+ multi-head: 22-class seg + draw order + confidence)
#   2. Joint Refinement (MobileNetV3 + regression heads, 20 joints)
#   3. Weight Prediction (per-vertex MLP, 20 bones)
#   4. Diffusion Weight Prediction (dual-input MLP: vertex features + seg encoder)
#   5. Inpainting (U-Net: RGBA + mask → completed RGBA)
#
# Usage:
#   chmod +x training/train_all.sh
#   ./training/train_all.sh          # Full A100 configs (~8-12h)
#   ./training/train_all.sh lean     # Lean A100 configs, core data only (~4-5h)
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
    DIFF_WEIGHT_CONFIG="training/configs/diffusion_weights.yaml"
    INPAINT_CONFIG="training/configs/inpainting.yaml"
    echo "Using LOCAL configs (default batch sizes)"
elif [ "$MODE" = "lean" ]; then
    SEG_CONFIG="training/configs/segmentation_a100_lean.yaml"
    JOINT_CONFIG="training/configs/joints_a100_lean.yaml"
    WEIGHT_CONFIG="training/configs/weights_a100_lean.yaml"
    DIFF_WEIGHT_CONFIG="training/configs/diffusion_weights_a100_lean.yaml"
    INPAINT_CONFIG="training/configs/inpainting_a100_lean.yaml"
    echo "Using LEAN A100 configs (core data only, ~6-8h)"
else
    SEG_CONFIG="training/configs/segmentation_a100.yaml"
    JOINT_CONFIG="training/configs/joints_a100.yaml"
    WEIGHT_CONFIG="training/configs/weights_a100.yaml"
    DIFF_WEIGHT_CONFIG="training/configs/diffusion_weights_a100.yaml"
    INPAINT_CONFIG="training/configs/inpainting_a100.yaml"
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
echo "[1/8] Training SEGMENTATION model..."
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
echo "[2/8] Training JOINT REFINEMENT model..."
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
echo "[3/8] Training WEIGHT PREDICTION model (per-vertex MLP)..."
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
# Step 4: Precompute encoder features (requires trained segmentation model)
# ---------------------------------------------------------------------------
echo "[4/8] Precomputing encoder features for diffusion weight training..."
echo "  Checkpoint: checkpoints/segmentation/best.pt"
echo "  Log: $LOG_DIR/precompute_encoder.log"
echo ""

python -m training.data.precompute_encoder_features \
    --segmentation-checkpoint checkpoints/segmentation/best.pt \
    --data-dirs ./data_cloud/humanrig ./data_cloud/segmentation \
    --output-dir ./data_cloud/encoder_features \
    --only-missing \
    2>&1 | tee "$LOG_DIR/precompute_encoder.log"

echo ""
echo "  Encoder feature precomputation complete."
echo ""

# ---------------------------------------------------------------------------
# Model 4: Diffusion Weight Prediction (dual-input MLP)
# ---------------------------------------------------------------------------
echo "[5/8] Training DIFFUSION WEIGHT PREDICTION model..."
echo "  Config: $DIFF_WEIGHT_CONFIG"
echo "  Log: $LOG_DIR/diffusion_weights.log"
echo ""

python -m training.train_diffusion_weights \
    --config "$DIFF_WEIGHT_CONFIG" \
    2>&1 | tee "$LOG_DIR/diffusion_weights.log"

echo ""
echo "  Diffusion weight prediction training complete."
echo ""

# ---------------------------------------------------------------------------
# Step 6: Generate occlusion pairs for inpainting (if not already done)
# ---------------------------------------------------------------------------
echo "[6/8] Generating occlusion pairs for inpainting training..."
echo "  Source: ./data_cloud/fbanimehq"
echo "  Log: $LOG_DIR/generate_occlusion.log"
echo ""

python -m training.data.generate_occlusion_pairs \
    --source-dirs ./data_cloud/fbanimehq \
    --output-dir ./data_cloud/inpainting_pairs \
    --masks-per-image 3 \
    --resolution 512 \
    2>&1 | tee "$LOG_DIR/generate_occlusion.log"

echo ""
echo "  Occlusion pair generation complete."
echo ""

# ---------------------------------------------------------------------------
# Model 5: Inpainting (U-Net)
# ---------------------------------------------------------------------------
echo "[7/8] Training INPAINTING model (U-Net)..."
echo "  Config: $INPAINT_CONFIG"
echo "  Log: $LOG_DIR/inpainting.log"
echo ""

python -m training.train_inpainting \
    --config "$INPAINT_CONFIG" \
    2>&1 | tee "$LOG_DIR/inpainting.log"

echo ""
echo "  Inpainting training complete."
echo ""

# ---------------------------------------------------------------------------
# Step 8: ONNX Export
# ---------------------------------------------------------------------------
echo "[8/8] Exporting all models to ONNX..."
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

# Export diffusion weights (dual-input MLP)
python -m training.export_onnx \
    --model diffusion_weights \
    --checkpoint checkpoints/diffusion_weights/best.pt \
    --output "$ONNX_DIR/diffusion_weight_prediction.onnx" \
    2>&1 | tee -a "$LOG_DIR/export.log"

# Export inpainting (U-Net)
python -m training.export_onnx \
    --model inpainting \
    --checkpoint checkpoints/inpainting/best.pt \
    --output "$ONNX_DIR/inpainting.onnx" \
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

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
#   6. Texture Inpainting (U-Net: partial UV + mask → completed UV texture)
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
    TEX_INPAINT_CONFIG="training/configs/texture_inpainting.yaml"
    echo "Using LOCAL configs (default batch sizes)"
elif [ "$MODE" = "lean" ]; then
    SEG_CONFIG="training/configs/segmentation_a100_lean.yaml"
    JOINT_CONFIG="training/configs/joints_a100_lean.yaml"
    WEIGHT_CONFIG="training/configs/weights_a100_lean.yaml"
    DIFF_WEIGHT_CONFIG="training/configs/diffusion_weights_a100_lean.yaml"
    INPAINT_CONFIG="training/configs/inpainting_a100_lean.yaml"
    TEX_INPAINT_CONFIG="training/configs/texture_inpainting_a100_lean.yaml"
    echo "Using LEAN A100 configs (core data only, ~6-8h)"
else
    SEG_CONFIG="training/configs/segmentation_a100.yaml"
    JOINT_CONFIG="training/configs/joints_a100.yaml"
    WEIGHT_CONFIG="training/configs/weights_a100.yaml"
    DIFF_WEIGHT_CONFIG="training/configs/diffusion_weights_a100.yaml"
    INPAINT_CONFIG="training/configs/inpainting_a100.yaml"
    TEX_INPAINT_CONFIG="training/configs/texture_inpainting_a100.yaml"
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
echo "[1/9] Training SEGMENTATION model..."
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
echo "[2/9] Training JOINT REFINEMENT model..."
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
echo "[3/9] Training WEIGHT PREDICTION model (per-vertex MLP)..."
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
echo "[4/9] Precomputing encoder features for diffusion weight training..."
echo "  Checkpoint: checkpoints/segmentation/best.pt"
echo "  Log: $LOG_DIR/precompute_encoder.log"
echo ""

python -m training.data.precompute_encoder_features \
    --segmentation-checkpoint checkpoints/segmentation/best.pt \
    --data-dirs ./data_cloud/humanrig ./data_cloud/unirig \
    --output-dir ./data_cloud/encoder_features \
    --only-missing \
    2>&1 | tee "$LOG_DIR/precompute_encoder.log"

echo ""
echo "  Encoder feature precomputation complete."
echo ""

# ---------------------------------------------------------------------------
# Model 4: Diffusion Weight Prediction (dual-input MLP)
# ---------------------------------------------------------------------------
echo "[5/9] Training DIFFUSION WEIGHT PREDICTION model..."
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
PAIRS_DIR="./data_cloud/inpainting_pairs"
PAIRS_COUNT=$(find "$PAIRS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -100 | wc -l)

if [ "$PAIRS_COUNT" -ge 100 ]; then
    echo "[6/9] Occlusion pairs already exist ($PAIRS_COUNT+ dirs in $PAIRS_DIR), skipping generation."
else
    echo "[6/9] Generating occlusion pairs for inpainting training..."
    echo "  Source: ./data_cloud/fbanimehq"
    echo "  Max: 15,000 source images × 3 masks = ~45K pairs"
    echo "  Log: $LOG_DIR/generate_occlusion.log"
    echo ""

    python -m training.data.generate_occlusion_pairs \
        --source-dirs ./data_cloud/fbanimehq \
        --output-dir "$PAIRS_DIR" \
        --masks-per-image 3 \
        --resolution 512 \
        --max-images 15000 \
        2>&1 | tee "$LOG_DIR/generate_occlusion.log"

    # Verify pairs were actually written to disk (disk space can cause silent failures)
    VERIFY_COUNT=$(find "$PAIRS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "  Verification: $VERIFY_COUNT pair directories on disk"
    if [ "$VERIFY_COUNT" -lt 1000 ]; then
        echo "  WARNING: Very few pairs generated — check disk space (df -h)"
        df -h .
    fi
fi

echo ""
echo "  Occlusion pair generation complete."
echo ""

# ---------------------------------------------------------------------------
# Model 5: Inpainting (U-Net)
# ---------------------------------------------------------------------------
echo "[7/9] Training INPAINTING model (U-Net)..."
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
# Model 6: Texture Inpainting (U-Net)
# ---------------------------------------------------------------------------
echo "[8/9] Training TEXTURE INPAINTING model (U-Net)..."
echo "  Config: $TEX_INPAINT_CONFIG"
echo "  Log: $LOG_DIR/texture_inpainting.log"
echo ""

python -m training.train_texture_inpainting \
    --config "$TEX_INPAINT_CONFIG" \
    2>&1 | tee "$LOG_DIR/texture_inpainting.log"

echo ""
echo "  Texture inpainting training complete."
echo ""

# ---------------------------------------------------------------------------
# Step 9: ONNX Export
# ---------------------------------------------------------------------------
echo "[9/9] Exporting all models to ONNX..."
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
    --output "$ONNX_DIR/weight_prediction.onnx" \
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

# Export texture inpainting (U-Net)
python -m training.export_onnx \
    --model texture_inpainting \
    --checkpoint checkpoints/texture_inpainting/best.pt \
    --output "$ONNX_DIR/texture_inpainting.onnx" \
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

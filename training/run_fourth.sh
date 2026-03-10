#!/usr/bin/env bash
# =============================================================================
# Strata Training — Fourth Run (A100)
#
# Key changes from run 3:
#   - Segmentation: resume from run 1 checkpoint (0.545 mIoU), not ImageNet
#   - Gemini diverse: ~191 pseudo-labeled 2D illustrations (domain gap bridge)
#   - UniRig weights: split_loader fix unlocks 14,950 weight examples
#   - Stronger augmentation + label smoothing to combat noisy labels
#   - Lower LR (5e-5) for fine-tuning
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#   rclone copy hetzner:strata-training-data/checkpoints/ ./checkpoints/ --transfers 32 --fast-list -P
#
# Usage:
#   chmod +x training/run_fourth.sh
#   ./training/run_fourth.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run4_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Fourth Run"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# 0. Install extra deps
# ---------------------------------------------------------------------------
echo "[0/8] Installing extra dependencies..."
pip install -q diffusers transformers accelerate
echo "  Done."
echo ""

# ---------------------------------------------------------------------------
# 1. Download Gemini diverse dataset from bucket
# ---------------------------------------------------------------------------
echo "[1/8] Downloading Gemini diverse dataset..."
echo ""

if [ -d "./data_cloud/gemini_diverse" ] && [ "$(ls -A ./data_cloud/gemini_diverse 2>/dev/null | head -1)" ]; then
    echo "  gemini_diverse already exists, skipping download."
else
    rclone copy hetzner:strata-training-data/gemini_diverse/ ./data_cloud/gemini_diverse/ \
        --transfers 32 --checkers 64 --fast-list --size-only -P
fi
echo ""

# ---------------------------------------------------------------------------
# 2. Normals + depth enrichment for new datasets
# ---------------------------------------------------------------------------
echo "[2/8] Enriching new datasets with surface normals + depth (Marigold)..."
echo ""

for ds in gemini_diverse live2d; do
    if [ -d "./data_cloud/$ds" ]; then
        echo "  Enriching $ds with normals + depth..."
        python run_normals_enrich.py \
            --input-dir "./data_cloud/$ds" \
            --only-missing \
            --batch-size 16 \
            2>&1 | tee "$LOG_DIR/enrich_normals_${ds}.log"
        echo ""
    fi
done

echo "  Normals enrichment complete."
echo ""

# ---------------------------------------------------------------------------
# 3. Generate inpainting pairs
# ---------------------------------------------------------------------------
echo "[3/8] Generating inpainting occlusion pairs..."
echo ""

PAIRS_DIR="./data_cloud/inpainting_pairs"
PAIRS_COUNT=$(find "$PAIRS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -100 | wc -l)

if [ "$PAIRS_COUNT" -ge 100 ]; then
    echo "  Occlusion pairs already exist ($PAIRS_COUNT+ dirs), skipping."
else
    python -m training.data.generate_occlusion_pairs \
        --source-dirs \
            ./data_cloud/meshy_cc0_textured \
            ./data_cloud/meshy_cc0_unrigged \
            ./data_cloud/humanrig \
            ./data_cloud/gemini_diverse \
        --output-dir "$PAIRS_DIR" \
        --max-images 15000 \
        --masks-per-image 3 \
        2>&1 | tee "$LOG_DIR/generate_inpainting_pairs.log"
fi
echo ""

# ---------------------------------------------------------------------------
# 4. Train segmentation (resume from run 1 checkpoint)
# ---------------------------------------------------------------------------
echo "[4/8] Training SEGMENTATION model (resume from run 1)..."
echo ""

# Use run 1 checkpoint as starting point
RUN1_CKPT="checkpoints/segmentation/run1_best.pt"
if [ ! -f "$RUN1_CKPT" ]; then
    # Try to find run 1's best checkpoint
    if [ -f "checkpoints/segmentation/best.pt" ]; then
        echo "  Using existing best.pt as resume checkpoint"
        RUN1_CKPT="checkpoints/segmentation/best.pt"
    else
        echo "  WARNING: No run 1 checkpoint found, training from ImageNet"
        RUN1_CKPT=""
    fi
fi

RESUME_FLAG=""
if [ -n "$RUN1_CKPT" ]; then
    RESUME_FLAG="--resume $RUN1_CKPT"
    echo "  Resuming from: $RUN1_CKPT"
fi

python -m training.train_segmentation \
    --config training/configs/segmentation_a100_run4.yaml \
    $RESUME_FLAG \
    2>&1 | tee "$LOG_DIR/segmentation.log"

echo ""
echo "  Segmentation training complete."
echo ""

# ---------------------------------------------------------------------------
# 5. Train joints (reuse run 3 config — already good)
# ---------------------------------------------------------------------------
echo "[5/8] Training JOINT REFINEMENT model..."
echo ""

python -m training.train_joints \
    --config training/configs/joints_a100_lean.yaml \
    2>&1 | tee "$LOG_DIR/joints.log"

echo ""
echo "  Joint training complete."
echo ""

# ---------------------------------------------------------------------------
# 6. Train weights (with UniRig fix)
# ---------------------------------------------------------------------------
echo "[6/8] Training WEIGHT PREDICTION model..."
echo ""

python -m training.train_weights \
    --config training/configs/weights_a100_run4.yaml \
    2>&1 | tee "$LOG_DIR/weights.log"

echo ""
echo "  Weight prediction training complete."
echo ""

# ---------------------------------------------------------------------------
# 7. Precompute encoder features + train diffusion weights + inpainting
# ---------------------------------------------------------------------------
echo "[7/8] Precomputing encoder features..."
echo ""

python -m training.data.precompute_encoder_features \
    --segmentation-checkpoint checkpoints/segmentation/best.pt \
    --data-dirs ./data_cloud/humanrig ./data_cloud/unirig \
    --output-dir ./data_cloud/encoder_features \
    --only-missing \
    2>&1 | tee "$LOG_DIR/precompute_encoder.log"

echo ""
echo "  Training DIFFUSION WEIGHT PREDICTION model..."
echo ""

python -m training.train_diffusion_weights \
    --config training/configs/diffusion_weights_a100_lean.yaml \
    2>&1 | tee "$LOG_DIR/diffusion_weights.log"

echo ""
echo "  Training INPAINTING model..."
echo ""

python -m training.train_inpainting \
    --config training/configs/inpainting_a100_lean.yaml \
    2>&1 | tee "$LOG_DIR/inpainting.log"

echo ""
echo "  Training TEXTURE INPAINTING model..."
echo ""

python -m training.train_texture_inpainting \
    --config training/configs/texture_inpainting_a100_lean.yaml \
    2>&1 | tee "$LOG_DIR/texture_inpainting.log"

echo ""

# ---------------------------------------------------------------------------
# 8. ONNX Export + Upload
# ---------------------------------------------------------------------------
echo "[8/8] Exporting to ONNX + uploading results..."
echo ""

ONNX_DIR="./models/onnx"
mkdir -p "$ONNX_DIR"

for model_export in \
    "segmentation checkpoints/segmentation/best.pt segmentation.onnx" \
    "joints checkpoints/joints/best.pt joint_refinement.onnx" \
    "weights_vertex checkpoints/weights/best.pt weight_prediction.onnx" \
    "diffusion_weights checkpoints/diffusion_weights/best.pt diffusion_weight_prediction.onnx" \
    "inpainting checkpoints/inpainting/best.pt inpainting.onnx" \
    "texture_inpainting checkpoints/texture_inpainting/best.pt texture_inpainting.onnx"
do
    set -- $model_export
    model_name=$1 ckpt=$2 onnx_file=$3
    if [ -f "$ckpt" ]; then
        echo "  Exporting $model_name → $onnx_file"
        python -m training.export_onnx \
            --model "$model_name" \
            --checkpoint "$ckpt" \
            --output "$ONNX_DIR/$onnx_file" \
            2>&1 | tee -a "$LOG_DIR/export.log"
    else
        echo "  SKIP $model_name — no checkpoint at $ckpt"
    fi
done

echo ""
echo "  Uploading to bucket..."
rclone copy ./checkpoints/ hetzner:strata-training-data/checkpoints/ \
    --transfers 32 --fast-list -P
rclone copy ./logs/ hetzner:strata-training-data/logs/ \
    --transfers 32 --fast-list -P
rclone copy ./models/onnx/ hetzner:strata-training-data/models/onnx/ \
    --transfers 32 --fast-list -P

echo ""
echo "============================================"
echo "  Fourth run complete!"
echo "  Finished: $(date)"
echo ""
echo "  ONNX models:"
ls -lh "$ONNX_DIR/"*.onnx 2>/dev/null || echo "  (no ONNX files found)"
echo ""
echo "  Checkpoints:"
ls -lh checkpoints/*/best.pt 2>/dev/null || echo "  (no checkpoints found)"
echo ""
echo "  To download results to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints/ ./checkpoints/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/onnx/ ./models/onnx/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/logs/ ./logs/ --transfers 32 --fast-list -P"
echo "============================================"

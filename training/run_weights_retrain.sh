#!/usr/bin/env bash
# =============================================================================
# Strata Training — Weights Retrain with Run 20 Seg Encoder (A100)
#
# Recomputes encoder features using the improved seg model (run 20, 0.6485 mIoU)
# then retrains the weight prediction model. Fixes limb deformation artifacts.
#
# Estimated: ~2-3 hrs (precompute ~30 min + train ~2 hrs)
#
# Usage:
#   ./training/run_weights_retrain.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/weights_retrain_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Weights Retrain (run 20 seg encoder)"
echo "  Started: $(date)"
echo "============================================"
echo ""

# Pre-flight
echo "[pre] Pre-flight checks..."
if ! rclone lsd hetzner:strata-training-data/ &>/dev/null; then echo "  FAIL: rclone"; exit 1; fi
echo "  OK: rclone"
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then echo "  FAIL: CUDA"; exit 1; fi
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
echo "  OK: CUDA — $GPU_NAME"
echo ""

# ---------------------------------------------------------------------------
# 0. Download checkpoints + data
# ---------------------------------------------------------------------------
echo "[0/4] Downloading checkpoints + data..."

# Seg checkpoint (for encoder feature extraction)
mkdir -p checkpoints/segmentation
SEG_CKPT="checkpoints/segmentation/run20_best.pt"
if [ ! -f "$SEG_CKPT" ]; then
    rclone copy hetzner:strata-training-data/checkpoints_run20_seg/segmentation/run20_best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
fi
echo "  Seg checkpoint: $SEG_CKPT"

# HumanRig data (has GT skinning weights)
mkdir -p data_cloud data/tars
if [ ! -d "data_cloud/humanrig" ] || [ -z "$(ls data_cloud/humanrig/ 2>/dev/null | head -1)" ]; then
    echo "  Downloading humanrig.tar..."
    rclone copy hetzner:strata-training-data/tars/humanrig.tar ./data/tars/ \
        --transfers 32 --fast-list -P
    tar xf ./data/tars/humanrig.tar -C ./data_cloud/
    rm -f ./data/tars/humanrig.tar
fi
HR_COUNT=$(ls -d ./data_cloud/humanrig/*/ 2>/dev/null | wc -l | tr -d ' ')
echo "  humanrig: $HR_COUNT examples"
echo ""

# ---------------------------------------------------------------------------
# 1. Precompute encoder features with run 20 seg model
# ---------------------------------------------------------------------------
echo "[1/4] Precomputing encoder features (run 20 seg model)..."
echo "  This extracts backbone features from the improved seg model"
echo "  for each training image + vertex position."
echo ""

python3 -m training.data.precompute_encoder_features \
    --segmentation-checkpoint "$SEG_CKPT" \
    --data-dirs ./data_cloud/humanrig \
    --output-dir ./data_cloud/encoder_features \
    --device cuda \
    2>&1 | tee "$LOG_DIR/precompute_encoder.log"

echo "  Encoder features precomputed."
echo ""

# ---------------------------------------------------------------------------
# 2. Train weights model
# ---------------------------------------------------------------------------
echo "[2/4] Training WEIGHTS model..."
echo "  Config: training/configs/weights_ship.yaml"
echo ""

mkdir -p checkpoints/weights
rm -f checkpoints/weights/latest.pt

python3 -m training.train_weights \
    --config training/configs/weights_ship.yaml \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""

# ---------------------------------------------------------------------------
# 3. Export ONNX
# ---------------------------------------------------------------------------
echo "[3/4] Exporting ONNX..."

mkdir -p ./models/onnx

if [ -f "checkpoints/weights/best.pt" ]; then
    cp checkpoints/weights/best.pt checkpoints/weights/retrain_run20_best.pt

    python3 -m training.export_onnx \
        --model weights \
        --checkpoint checkpoints/weights/retrain_run20_best.pt \
        --output ./models/onnx/weights_retrain.onnx \
        2>&1 | tee "$LOG_DIR/export.log"

    ONNX_SIZE=$(du -h ./models/onnx/weights_retrain.onnx 2>/dev/null | cut -f1)
    echo "  Exported weights_retrain.onnx ($ONNX_SIZE)"
fi
echo ""

# ---------------------------------------------------------------------------
# 4. Upload
# ---------------------------------------------------------------------------
echo "[4/4] Uploading..."

rclone copy ./checkpoints/weights/retrain_run20_best.pt \
    hetzner:strata-training-data/checkpoints_weights_retrain/ \
    --transfers 4 --fast-list --size-only -P

if [ -f "./models/onnx/weights_retrain.onnx" ]; then
    rclone copy ./models/onnx/weights_retrain.onnx \
        hetzner:strata-training-data/models/weights_retrain/ \
        --transfers 4 --fast-list --size-only -P
fi

rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/weights_retrain_${TIMESTAMP}/ \
    --transfers 4 --fast-list -P

echo ""
echo "============================================"
echo "  Weights Retrain complete!"
echo "  Finished: $(date)"
echo "  Results:"
grep -E "best val/mae|val/mae=" "$LOG_DIR/train.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  To download:"
echo "    rclone copy hetzner:strata-training-data/models/weights_retrain/ /Volumes/TAMWoolff/data/models/weights_retrain/ --transfers 32 --fast-list -P"
echo "============================================"

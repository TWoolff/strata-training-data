#!/usr/bin/env bash
# =============================================================================
# Strata Training — Back View Generation (Model 6) on A100
#
# Trains U-Net to generate back view RGBA from front + 3/4 view inputs.
# Data: 1,085 character triplets (Meshy FBX + GLB + VRoid CC0)
#
# Estimated: ~2-3 hrs on A100
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_back_view.sh
#   ./training/run_back_view.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run_back_view_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Back View (Model 6)"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "[pre] Pre-flight checks..."

PREFLIGHT_FAIL=0

if ! rclone lsd hetzner:strata-training-data/ &>/dev/null; then
    echo "  FAIL: rclone cannot connect to Hetzner bucket"
    PREFLIGHT_FAIL=1
else
    echo "  OK: rclone bucket connection"
fi

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  FAIL: CUDA not available"
    PREFLIGHT_FAIL=1
else
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEM=$(python3 -c "import torch; p=torch.cuda.get_device_properties(0); m=getattr(p,'total_memory',getattr(p,'total_mem',0)); print(f'{m/1e9:.0f}GB')")
    echo "  OK: CUDA — $GPU_NAME ($GPU_MEM)"
fi

if ! python3 -c "import torchvision" 2>/dev/null; then
    echo "  FAIL: torchvision not installed (needed for perceptual loss)"
    PREFLIGHT_FAIL=1
else
    echo "  OK: torchvision"
fi

if [ "$PREFLIGHT_FAIL" -ne 0 ]; then
    echo ""
    echo "Pre-flight failed. Fix issues above and retry."
    exit 1
fi

echo ""

# ---------------------------------------------------------------------------
# Step 1: Download data
# ---------------------------------------------------------------------------
echo "[1/4] Downloading back view pairs from bucket..."

mkdir -p ./data/training/back_view_pairs

rclone copy hetzner:strata-training-data/tars/back_view_pairs.tar ./data/tars/ \
    --transfers 16 --fast-list --size-only -P 2>&1 | tee "$LOG_DIR/download.log"

echo "  Extracting tar..."
tar xf ./data/tars/back_view_pairs.tar -C ./data/training/

# The tar contains back_view_pairs_merged/ — move contents to expected location
if [ -d "./data/training/back_view_pairs_merged" ]; then
    # Move pair dirs into back_view_pairs/
    mv ./data/training/back_view_pairs_merged/pair_* ./data/training/back_view_pairs/ 2>/dev/null || true
    rm -rf ./data/training/back_view_pairs_merged
fi

PAIR_COUNT=$(ls -d ./data/training/back_view_pairs/pair_* 2>/dev/null | wc -l | tr -d ' ')
echo "  Pairs: $PAIR_COUNT"

if [ "$PAIR_COUNT" -lt 100 ]; then
    echo "  FAIL: Expected 1000+ pairs, got $PAIR_COUNT"
    exit 1
fi

echo ""

# ---------------------------------------------------------------------------
# Step 2: Train
# ---------------------------------------------------------------------------
echo "[2/4] Training back view model..."

python3 -m training.train_back_view \
    --config training/configs/back_view_a100.yaml \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""

# ---------------------------------------------------------------------------
# Step 3: Export ONNX
# ---------------------------------------------------------------------------
echo "[3/4] Exporting ONNX..."

mkdir -p ./models/onnx_back_view

python3 -m training.export_onnx \
    --model back_view \
    --checkpoint ./checkpoints/back_view/best.pt \
    --output ./models/onnx_back_view/back_view.onnx \
    2>&1 | tee "$LOG_DIR/export.log"

ONNX_SIZE=$(du -h ./models/onnx_back_view/back_view.onnx | cut -f1)
echo "  ONNX size: $ONNX_SIZE"

echo ""

# ---------------------------------------------------------------------------
# Step 4: Upload results
# ---------------------------------------------------------------------------
echo "[4/4] Uploading checkpoints + ONNX to bucket..."

rclone copy ./checkpoints/back_view/ hetzner:strata-training-data/checkpoints_back_view/ \
    --transfers 16 --fast-list --size-only -P 2>&1 | tee "$LOG_DIR/upload_ckpt.log"

rclone copy ./models/onnx_back_view/ hetzner:strata-training-data/models/back_view/ \
    --transfers 16 --fast-list --size-only -P 2>&1 | tee "$LOG_DIR/upload_onnx.log"

rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run_back_view_${TIMESTAMP}/ \
    --transfers 16 --fast-list -P

echo ""
echo "============================================"
echo "  Done! Back View Model Training Complete"
echo "  Pairs: $PAIR_COUNT"
echo "  ONNX: ./models/onnx_back_view/back_view.onnx ($ONNX_SIZE)"
echo "  Logs: $LOG_DIR"
echo "  Finished: $(date)"
echo "============================================"

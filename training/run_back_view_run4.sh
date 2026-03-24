#!/usr/bin/env bash
# =============================================================================
# Strata Training — Back View Generation Run 4 (A100)
#
# +1,244 new Meshy FBX pairs. Combined: ~3,049 triplets.
# Resume from run 3 checkpoint (val/l1 = 0.2408).
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
#   chmod +x training/run_back_view_run4.sh
#   ./training/run_back_view_run4.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run4_back_view_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Back View Run 4"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
echo "[pre] Pre-flight checks..."

if ! rclone lsd hetzner:strata-training-data/ &>/dev/null; then
    echo "  FAIL: rclone cannot connect to Hetzner bucket"
    exit 1
fi
echo "  OK: rclone bucket connection"

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  FAIL: CUDA not available"
    exit 1
fi
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
GPU_MEM=$(python3 -c "import torch; p=torch.cuda.get_device_properties(0); m=getattr(p,'total_memory',getattr(p,'total_mem',0)); print(f'{m/1e9:.0f}GB')")
echo "  OK: CUDA — $GPU_NAME ($GPU_MEM)"

if ! python3 -c "import torchvision" 2>/dev/null; then
    echo "  FAIL: torchvision not installed (needed for perceptual loss)"
    exit 1
fi
echo "  OK: torchvision"
echo ""

# ---------------------------------------------------------------------------
# 0. Download run 3 checkpoint
# ---------------------------------------------------------------------------
echo "[0/4] Downloading run 3 checkpoint..."
mkdir -p checkpoints/back_view

RUN3_CKPT="checkpoints/back_view/run3_best.pt"
if [ -f "$RUN3_CKPT" ]; then
    echo "  run3_best.pt already exists."
else
    echo "  Downloading run 3 checkpoint..."
    rclone copy hetzner:strata-training-data/checkpoints_back_view/best.pt \
        ./checkpoints/back_view/ --transfers 32 --fast-list -P
    if [ -f "checkpoints/back_view/best.pt" ]; then
        cp checkpoints/back_view/best.pt "$RUN3_CKPT"
    else
        echo "  WARN: Could not download run 3 checkpoint — training from scratch"
    fi
fi
# Place as latest.pt so auto-resume picks it up
if [ -f "$RUN3_CKPT" ] && [ ! -f "checkpoints/back_view/latest.pt" ]; then
    cp "$RUN3_CKPT" checkpoints/back_view/latest.pt
fi
echo ""

# ---------------------------------------------------------------------------
# 1. Download data
# ---------------------------------------------------------------------------
echo "[1/4] Downloading back view pairs..."
echo ""

mkdir -p ./data/training ./data/tars

# Each tar extracts to its own subdirectory to avoid pair_NNNNN collisions.

download_bv_tar() {
    local tar_name="$1"
    local extract_dir="$2"

    if [ -d "$extract_dir" ] && [ "$(ls -d "$extract_dir"/pair_* 2>/dev/null | head -1)" ]; then
        local count=$(ls -d "$extract_dir"/pair_* 2>/dev/null | wc -l | tr -d ' ')
        echo "  $(basename "$extract_dir"): $count pairs (already exists)"
        return 0
    fi

    echo "  Downloading ${tar_name}..."
    rclone copy "hetzner:strata-training-data/tars/${tar_name}" ./data/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "./data/tars/${tar_name}" ]; then
        echo "  Extracting..."
        tar xf "./data/tars/${tar_name}" -C ./data/training/
        rm -f "./data/tars/${tar_name}"
        local count=$(ls -d "$extract_dir"/pair_* 2>/dev/null | wc -l | tr -d ' ')
        echo "  $(basename "$extract_dir"): $count pairs"
    else
        echo "  WARN: ${tar_name} not found in bucket"
    fi
}

# Rigged pairs (1,085 triplets) — tar contains back_view_pairs_merged/
download_bv_tar "back_view_pairs.tar" "./data/training/back_view_pairs_merged"

# Unrigged pairs (~720 triplets) — tar contains back_view_pairs_unrigged/
download_bv_tar "back_view_pairs_unrigged.tar" "./data/training/back_view_pairs_unrigged"

# New FBX pairs (1,244 triplets) — tar contains back_view_pairs_new/
download_bv_tar "back_view_pairs_new.tar" "./data/training/back_view_pairs_new"

PAIR_COUNT=0
for d in ./data/training/back_view_pairs_merged ./data/training/back_view_pairs_unrigged ./data/training/back_view_pairs_new; do
    if [ -d "$d" ]; then
        c=$(ls -d "$d"/pair_* 2>/dev/null | wc -l | tr -d ' ')
        PAIR_COUNT=$((PAIR_COUNT + c))
    fi
done
echo ""
echo "  Total pairs: $PAIR_COUNT"
if [ "$PAIR_COUNT" -lt 2500 ]; then
    echo "  WARN: Expected ~3,049 pairs, got $PAIR_COUNT"
fi
echo ""

# ---------------------------------------------------------------------------
# 2. Train
# ---------------------------------------------------------------------------
echo "[2/4] Training back view model..."
echo "  Config: training/configs/back_view_a100_run4.yaml"
echo ""

python3 -m training.train_back_view \
    --config training/configs/back_view_a100_run4.yaml \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""
echo "  Training complete."
echo ""

# ---------------------------------------------------------------------------
# 3. Export ONNX
# ---------------------------------------------------------------------------
echo "[3/4] Exporting ONNX..."

mkdir -p ./models/onnx

if [ -f "checkpoints/back_view/best.pt" ]; then
    cp checkpoints/back_view/best.pt checkpoints/back_view/run4_best.pt
    python3 -m training.export_onnx \
        --model back_view \
        --checkpoint checkpoints/back_view/run4_best.pt \
        --output ./models/onnx/back_view_run4.onnx \
        2>&1 | tee "$LOG_DIR/export.log"
    ONNX_SIZE=$(du -h ./models/onnx/back_view_run4.onnx 2>/dev/null | cut -f1)
    echo "  Exported back_view_run4.onnx ($ONNX_SIZE)"
fi
echo ""

# ---------------------------------------------------------------------------
# 4. Upload
# ---------------------------------------------------------------------------
echo "[4/4] Uploading results..."

rclone copy ./checkpoints/back_view/run4_best.pt \
    hetzner:strata-training-data/checkpoints_back_view_run4/ \
    --transfers 32 --fast-list -P
rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run4_back_view_${TIMESTAMP}/ \
    --transfers 32 --fast-list -P
if [ -f "./models/onnx/back_view_run4.onnx" ]; then
    rclone copy ./models/onnx/back_view_run4.onnx \
        hetzner:strata-training-data/models/back_view_run4/ \
        --transfers 32 --fast-list -P
fi

echo ""
echo "============================================"
echo "  Back View Run 4 complete!"
echo "  Finished: $(date)"
echo "  Pairs: $PAIR_COUNT"
echo "  Results:"
grep -E "best val/l1|val/l1=" "$LOG_DIR/train.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  To download to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_back_view_run4/ /Volumes/TAMWoolff/data/checkpoints_back_view_run4/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/back_view_run4/ /Volumes/TAMWoolff/data/models/back_view_run4/ --transfers 32 --fast-list -P"
echo "============================================"

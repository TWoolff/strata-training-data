#!/usr/bin/env bash
# =============================================================================
# Strata Training — Back View Run 6 (A100)
#
# Fine-tune on 5 demo character pairs at high weight (50.0).
# Resume from run 4 checkpoint (val/l1 = 0.2152).
#
# Estimated: ~1-2 hrs on A100
#
# Usage:
#   chmod +x training/run_back_view_run6.sh
#   ./training/run_back_view_run6.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run6_back_view_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Back View Run 6 (Demo)"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
echo "[pre] Pre-flight checks..."

if ! rclone lsd hetzner:strata-training-data/ &>/dev/null; then
    echo "  FAIL: rclone"; exit 1
fi
echo "  OK: rclone"

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  FAIL: CUDA"; exit 1
fi
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
echo "  OK: CUDA — $GPU_NAME"

if ! python3 -c "import torchvision" 2>/dev/null; then
    echo "  FAIL: torchvision"; exit 1
fi
echo "  OK: torchvision"
echo ""

# ---------------------------------------------------------------------------
# 0. Download run 4 checkpoint
# ---------------------------------------------------------------------------
echo "[0/4] Downloading run 4 checkpoint..."
mkdir -p checkpoints/back_view

RUN4_CKPT="checkpoints/back_view/run4_best.pt"
if [ ! -f "$RUN4_CKPT" ]; then
    rclone copy hetzner:strata-training-data/checkpoints_back_view_run4/run4_best.pt \
        ./checkpoints/back_view/ --transfers 32 --fast-list -P
fi
echo "  Checkpoint: $RUN4_CKPT"
echo ""

# ---------------------------------------------------------------------------
# 1. Download data
# ---------------------------------------------------------------------------
echo "[1/4] Downloading back view pairs..."
echo ""

mkdir -p ./data/training ./data/tars

download_bv_tar() {
    local tar_name="$1"
    local extract_dir="$2"

    if [ -d "$extract_dir" ] && [ "$(ls -d "$extract_dir"/pair_* 2>/dev/null | head -1)" ]; then
        local count=$(ls -d "$extract_dir"/pair_* 2>/dev/null | wc -l | tr -d ' ')
        echo "  $(basename "$extract_dir"): $count pairs (exists)"
        return 0
    fi

    echo "  Downloading ${tar_name}..."
    rclone copy "hetzner:strata-training-data/tars/${tar_name}" ./data/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "./data/tars/${tar_name}" ]; then
        echo "  Extracting..."
        mkdir -p "$extract_dir"
        tar xf "./data/tars/${tar_name}" -C ./data/training/
        rm -f "./data/tars/${tar_name}"
        local count=$(ls -d "$extract_dir"/pair_* 2>/dev/null | wc -l | tr -d ' ')
        echo "  $(basename "$extract_dir"): $count pairs"
    else
        echo "  WARN: ${tar_name} not found in bucket"
    fi
}

download_bv_tar "back_view_pairs.tar" "./data/training/back_view_pairs_merged"
download_bv_tar "back_view_pairs_unrigged.tar" "./data/training/back_view_pairs_unrigged"
download_bv_tar "back_view_pairs_new.tar" "./data/training/back_view_pairs_new"

# Demo pairs (5 characters)
echo "  Downloading demo pairs..."
rclone copy hetzner:strata-training-data/tars/demo_back_view_pairs.tar ./data/tars/ \
    --transfers 32 --fast-list -P
if [ -f "./data/tars/demo_back_view_pairs.tar" ]; then
    tar xf ./data/tars/demo_back_view_pairs.tar -C ./data/training/
    rm -f ./data/tars/demo_back_view_pairs.tar
fi
DEMO_COUNT=$(ls -d ./data/training/demo_pairs/pair_* 2>/dev/null | wc -l | tr -d ' ')
echo "  demo_pairs: $DEMO_COUNT pairs"

PAIR_COUNT=0
for d in ./data/training/back_view_pairs_merged ./data/training/back_view_pairs_unrigged \
         ./data/training/back_view_pairs_new ./data/training/demo_pairs; do
    if [ -d "$d" ]; then
        c=$(ls -d "$d"/pair_* 2>/dev/null | wc -l | tr -d ' ')
        PAIR_COUNT=$((PAIR_COUNT + c))
    fi
done
echo ""
echo "  Total pairs: $PAIR_COUNT (incl $DEMO_COUNT demo)"
echo ""

# ---------------------------------------------------------------------------
# 2. Train
# ---------------------------------------------------------------------------
echo "[2/4] Training back view model (demo fine-tune)..."
echo "  Config: training/configs/back_view_a100_run6.yaml"
echo ""

# Reset latest.pt with run 4 weights at epoch -1
rm -f checkpoints/back_view/latest.pt
python3 -c "
import torch
ckpt = torch.load('$RUN4_CKPT', map_location='cpu', weights_only=False)
ckpt['epoch'] = -1
torch.save(ckpt, 'checkpoints/back_view/latest.pt')
"

python3 -m training.train_back_view \
    --config training/configs/back_view_a100_run6.yaml \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""

# ---------------------------------------------------------------------------
# 3. Export ONNX
# ---------------------------------------------------------------------------
echo "[3/4] Exporting ONNX..."

mkdir -p ./models/onnx

if [ -f "checkpoints/back_view/best.pt" ]; then
    cp checkpoints/back_view/best.pt checkpoints/back_view/run6_best.pt
    python3 -m training.export_onnx \
        --model back_view \
        --checkpoint checkpoints/back_view/run6_best.pt \
        --output ./models/onnx/back_view_run6.onnx \
        2>&1 | tee "$LOG_DIR/export.log"
    ONNX_SIZE=$(du -h ./models/onnx/back_view_run6.onnx 2>/dev/null | cut -f1)
    echo "  Exported back_view_run6.onnx ($ONNX_SIZE)"
fi
echo ""

# ---------------------------------------------------------------------------
# 4. Upload
# ---------------------------------------------------------------------------
echo "[4/4] Uploading results..."

rclone copy ./checkpoints/back_view/run6_best.pt \
    hetzner:strata-training-data/checkpoints_back_view_run6/ \
    --transfers 4 --fast-list --size-only -P
rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run6_back_view_${TIMESTAMP}/ \
    --transfers 4 --fast-list -P
if [ -f "./models/onnx/back_view_run6.onnx" ]; then
    rclone copy ./models/onnx/back_view_run6.onnx \
        hetzner:strata-training-data/models/back_view_run6/ \
        --transfers 4 --fast-list --size-only -P
fi

echo ""
echo "============================================"
echo "  Back View Run 6 complete!"
echo "  Finished: $(date)"
echo "  Pairs: $PAIR_COUNT (incl $DEMO_COUNT demo at weight 50.0)"
echo "  Results:"
grep -E "best val/l1|val/l1=" "$LOG_DIR/train.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  To download:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_back_view_run6/ /Volumes/TAMWoolff/data/checkpoints_back_view_run6/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/back_view_run6/ /Volumes/TAMWoolff/data/models/back_view_run6/ --transfers 32 --fast-list -P"
echo "============================================"

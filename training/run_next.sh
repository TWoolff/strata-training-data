#!/usr/bin/env bash
# =============================================================================
# Strata Training — Next Run: View Synthesis Bear Chef Fine-tune
#
# Fine-tune view synthesis on bear chef A-pose for demo.
# Resume from run 2 (0.2100 val/l1). Short run (~2-3 hrs).
# Storage: ~25 GB
#
# Usage:
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#   ./training/run_next.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/next_run_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  View Synthesis Bear Chef Fine-tune"
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
# 1. Download data + checkpoint
# ---------------------------------------------------------------------------
echo "[1/3] Downloading data..."
mkdir -p checkpoints/view_synthesis data/training data/tars

# Run 2 checkpoint
RUN2_CKPT="checkpoints/view_synthesis/run2_best.pt"
if [ ! -f "$RUN2_CKPT" ]; then
    echo "  Downloading run 2 checkpoint..."
    rclone copy hetzner:strata-training-data/checkpoints_view_synthesis_run2/run2_best.pt \
        ./checkpoints/view_synthesis/ --transfers 32 --fast-list -P
fi
echo "  Checkpoint: $RUN2_CKPT"

# Demo pairs (includes bear chef A-pose)
echo "  Downloading demo pairs (3.4 GB)..."
rclone copy hetzner:strata-training-data/tars/demo_back_view_pairs.tar ./data/tars/ \
    --transfers 32 --fast-list -P
if [ -f "./data/tars/demo_back_view_pairs.tar" ]; then
    tar xf ./data/tars/demo_back_view_pairs.tar -C ./data/training/
    rm -f ./data/tars/demo_back_view_pairs.tar
fi
DEMO_COUNT=$(ls -d ./data/training/demo_pairs/pair_* 2>/dev/null | wc -l | tr -d ' ')
echo "  demo_pairs: $DEMO_COUNT pairs"

# 3D-rendered pairs
download_tar() {
    local tar_name="$1"
    local extract_dir="$2"
    if [ -d "$extract_dir" ] && [ "$(ls -d "$extract_dir"/pair_* 2>/dev/null | head -1)" ]; then
        local count=$(ls -d "$extract_dir"/pair_* 2>/dev/null | wc -l | tr -d ' ')
        echo "  $(basename "$extract_dir"): $count pairs (exists)"
        return 0
    fi
    echo "  Downloading ${tar_name}..."
    rclone copy "hetzner:strata-training-data/tars/${tar_name}" ./data/tars/ --transfers 32 --fast-list -P
    if [ -f "./data/tars/${tar_name}" ]; then
        mkdir -p "$extract_dir"
        tar xf "./data/tars/${tar_name}" -C ./data/training/
        rm -f "./data/tars/${tar_name}"
    fi
}

download_tar "back_view_pairs.tar" "./data/training/back_view_pairs_merged"
download_tar "back_view_pairs_unrigged.tar" "./data/training/back_view_pairs_unrigged"
download_tar "back_view_pairs_new.tar" "./data/training/back_view_pairs_new"

echo ""

# ---------------------------------------------------------------------------
# 2. Train
# ---------------------------------------------------------------------------
echo "[2/3] Training view synthesis (bear chef fine-tune)..."

rm -f checkpoints/view_synthesis/latest.pt
python3 -c "
import torch
ckpt = torch.load('$RUN2_CKPT', map_location='cpu', weights_only=False)
ckpt['epoch'] = -1
torch.save(ckpt, 'checkpoints/view_synthesis/latest.pt')
"

python3 -m training.train_view_synthesis \
    --config training/configs/view_synthesis_bear_chef.yaml \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""

# ---------------------------------------------------------------------------
# 3. Export + Upload
# ---------------------------------------------------------------------------
echo "[3/3] Exporting + uploading..."

mkdir -p ./models/onnx

if [ -f "checkpoints/view_synthesis/best.pt" ]; then
    cp checkpoints/view_synthesis/best.pt checkpoints/view_synthesis/bear_chef_best.pt

    python3 -m training.export_onnx \
        --model view_synthesis \
        --checkpoint checkpoints/view_synthesis/bear_chef_best.pt \
        --output ./models/onnx/view_synthesis_bear_chef.onnx \
        2>&1 | tee "$LOG_DIR/export.log"
fi

rclone copy ./checkpoints/view_synthesis/bear_chef_best.pt \
    hetzner:strata-training-data/checkpoints_view_synthesis_bear_chef/ \
    --transfers 4 --fast-list --size-only -P

if [ -f "./models/onnx/view_synthesis_bear_chef.onnx" ]; then
    rclone copy ./models/onnx/view_synthesis_bear_chef.onnx \
        hetzner:strata-training-data/models/view_synthesis_bear_chef/ \
        --transfers 4 --fast-list --size-only -P
fi

rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/next_run_${TIMESTAMP}/ \
    --transfers 4 --fast-list -P

echo ""
echo "============================================"
echo "  Done!"
echo "  Results:"
grep -E "best val/l1|New best" "$LOG_DIR/train.log" 2>/dev/null | tail -3 || true
echo ""
echo "  To download:"
echo "    rclone copy hetzner:strata-training-data/models/view_synthesis_bear_chef/ /Volumes/TAMWoolff/data/models/view_synthesis_bear_chef/ --transfers 32 --fast-list -P"
echo "============================================"

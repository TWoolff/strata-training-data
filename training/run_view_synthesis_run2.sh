#!/usr/bin/env bash
# =============================================================================
# Strata Training — View Synthesis Run 2 (A100)
#
# Resume from run 1 (0.2139 val/l1) with expanded turnaround sheet data.
# Lower LR for fine-tuning. ~3-4 hrs.
#
# Usage:
#   ./training/run_view_synthesis_run2.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/view_synthesis_run2_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  View Synthesis Run 2"
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
# 0. Download run 1 checkpoint
# ---------------------------------------------------------------------------
echo "[0/3] Downloading run 1 checkpoint..."
mkdir -p checkpoints/view_synthesis

RUN1_CKPT="checkpoints/view_synthesis/run1_best.pt"
if [ ! -f "$RUN1_CKPT" ]; then
    rclone copy hetzner:strata-training-data/checkpoints_view_synthesis_run1/run1_best.pt \
        ./checkpoints/view_synthesis/ --transfers 32 --fast-list -P
fi
echo "  Checkpoint: $RUN1_CKPT"
echo ""

# ---------------------------------------------------------------------------
# 1. Download data
# ---------------------------------------------------------------------------
echo "[1/3] Downloading data..."
mkdir -p ./data/training ./data/tars

# Demo pairs (Gemini turnaround sheets)
echo "  Downloading demo pairs..."
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

PAIR_COUNT=0
for d in ./data/training/demo_pairs ./data/training/back_view_pairs_merged \
         ./data/training/back_view_pairs_unrigged ./data/training/back_view_pairs_new; do
    if [ -d "$d" ]; then
        c=$(ls -d "$d"/pair_* 2>/dev/null | wc -l | tr -d ' ')
        PAIR_COUNT=$((PAIR_COUNT + c))
    fi
done
echo ""
echo "  Total pairs: $PAIR_COUNT (incl $DEMO_COUNT demo at weight 10.0)"
echo ""

# ---------------------------------------------------------------------------
# 2. Train (resume from run 1)
# ---------------------------------------------------------------------------
echo "[2/3] Training view synthesis model (resume from run 1)..."

# Set up latest.pt from run 1 checkpoint for auto-resume
rm -f checkpoints/view_synthesis/latest.pt
python3 -c "
import torch
ckpt = torch.load('$RUN1_CKPT', map_location='cpu', weights_only=False)
ckpt['epoch'] = -1
torch.save(ckpt, 'checkpoints/view_synthesis/latest.pt')
"

python3 -m training.train_view_synthesis \
    --config training/configs/view_synthesis_run2.yaml \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""

# ---------------------------------------------------------------------------
# 3. Export + Upload
# ---------------------------------------------------------------------------
echo "[3/3] Exporting ONNX + uploading..."

mkdir -p ./models/onnx

if [ -f "checkpoints/view_synthesis/best.pt" ]; then
    cp checkpoints/view_synthesis/best.pt checkpoints/view_synthesis/run2_best.pt

    python3 -m training.export_onnx \
        --model view_synthesis \
        --checkpoint checkpoints/view_synthesis/run2_best.pt \
        --output ./models/onnx/view_synthesis_run2.onnx \
        2>&1 | tee "$LOG_DIR/export.log"
fi

rclone copy ./checkpoints/view_synthesis/run2_best.pt \
    hetzner:strata-training-data/checkpoints_view_synthesis_run2/ \
    --transfers 4 --fast-list --size-only -P

if [ -f "./models/onnx/view_synthesis_run2.onnx" ]; then
    rclone copy ./models/onnx/view_synthesis_run2.onnx \
        hetzner:strata-training-data/models/view_synthesis_run2/ \
        --transfers 4 --fast-list --size-only -P
fi

rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/view_synthesis_run2_${TIMESTAMP}/ \
    --transfers 4 --fast-list -P

echo ""
echo "============================================"
echo "  View Synthesis Run 2 complete!"
echo "  Finished: $(date)"
echo "  Results:"
grep -E "best val/l1|val/l1=" "$LOG_DIR/train.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo "============================================"

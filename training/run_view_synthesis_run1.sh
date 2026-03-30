#!/usr/bin/env bash
# =============================================================================
# Strata Training — View Synthesis Run 1 (A100)
#
# Unified view synthesis: any 2 views + target angle → target view.
# Gemini turnaround sheets + 3D-rendered back view pairs.
# Trains from scratch (new model architecture).
#
# Estimated: ~3-4 hrs on A100
#
# Usage:
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#   ./training/run_view_synthesis_run1.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/view_synthesis_run1_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  View Synthesis Run 1"
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
# 1. Download data
# ---------------------------------------------------------------------------
echo "[1/3] Downloading data..."
echo ""

mkdir -p ./data/training ./data/tars

download_tar() {
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
        mkdir -p "$extract_dir"
        tar xf "./data/tars/${tar_name}" -C ./data/training/
        rm -f "./data/tars/${tar_name}"
        local count=$(ls -d "$extract_dir"/pair_* 2>/dev/null | wc -l | tr -d ' ')
        echo "  $(basename "$extract_dir"): $count pairs"
    else
        echo "  WARN: ${tar_name} not found"
    fi
}

# Demo pairs (Gemini turnaround sheets — multi-angle)
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
# 2. Train
# ---------------------------------------------------------------------------
echo "[2/3] Training view synthesis model..."
echo "  Config: training/configs/view_synthesis_run1.yaml"
echo ""

python3 -m training.train_view_synthesis \
    --config training/configs/view_synthesis_run1.yaml \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""

# ---------------------------------------------------------------------------
# 3. Export + Upload
# ---------------------------------------------------------------------------
echo "[3/3] Exporting ONNX + uploading..."

mkdir -p ./models/onnx

if [ -f "checkpoints/view_synthesis/best.pt" ]; then
    cp checkpoints/view_synthesis/best.pt checkpoints/view_synthesis/run1_best.pt

    python3 -m training.export_onnx \
        --model view_synthesis \
        --checkpoint checkpoints/view_synthesis/run1_best.pt \
        --output ./models/onnx/view_synthesis_run1.onnx \
        2>&1 | tee "$LOG_DIR/export.log"

    ONNX_SIZE=$(du -h ./models/onnx/view_synthesis_run1.onnx 2>/dev/null | cut -f1)
    echo "  Exported view_synthesis_run1.onnx ($ONNX_SIZE)"
fi

# Upload
rclone copy ./checkpoints/view_synthesis/run1_best.pt \
    hetzner:strata-training-data/checkpoints_view_synthesis_run1/ \
    --transfers 4 --fast-list --size-only -P

if [ -f "./models/onnx/view_synthesis_run1.onnx" ]; then
    rclone copy ./models/onnx/view_synthesis_run1.onnx \
        hetzner:strata-training-data/models/view_synthesis_run1/ \
        --transfers 4 --fast-list --size-only -P
fi

rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/view_synthesis_run1_${TIMESTAMP}/ \
    --transfers 4 --fast-list -P

echo ""
echo "============================================"
echo "  View Synthesis Run 1 complete!"
echo "  Finished: $(date)"
echo "  Pairs: $PAIR_COUNT"
echo "  Results:"
grep -E "best val/l1|val/l1=" "$LOG_DIR/train.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  To download:"
echo "    rclone copy hetzner:strata-training-data/models/view_synthesis_run1/ /Volumes/TAMWoolff/data/models/view_synthesis_run1/ --transfers 32 --fast-list -P"
echo "============================================"

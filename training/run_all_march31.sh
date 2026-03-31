#!/usr/bin/env bash
# =============================================================================
# Strata Training — March 31 Combined Run (A100)
#
# 1. View Synthesis Run 2: resume from run 1, +6,180 illustrated pairs
# 2. Weights Retrain: precompute encoder features with run 20 seg, retrain
#
# Estimated: ~6-7 hrs total on A100
# Storage: ~30 GB
#
# Usage:
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#   ./training/run_all_march31.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/march31_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — March 31 Combined Run"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
echo "[pre] Pre-flight checks..."

PREFLIGHT_FAIL=0
if ! rclone lsd hetzner:strata-training-data/ &>/dev/null; then
    echo "  FAIL: rclone"; PREFLIGHT_FAIL=1
else
    echo "  OK: rclone"
fi

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  FAIL: CUDA"; PREFLIGHT_FAIL=1
else
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEM=$(python3 -c "import torch; p=torch.cuda.get_device_properties(0); m=getattr(p,'total_memory',getattr(p,'total_mem',0)); print(f'{m/1e9:.0f}GB')")
    echo "  OK: CUDA — $GPU_NAME ($GPU_MEM)"
fi

if ! python3 -c "import torchvision" 2>/dev/null; then
    echo "  FAIL: torchvision"; PREFLIGHT_FAIL=1
else
    echo "  OK: torchvision"
fi

if [ "$PREFLIGHT_FAIL" -ne 0 ]; then echo "Pre-flight failed."; exit 1; fi
echo ""

# =============================================================================
# PART 1: VIEW SYNTHESIS RUN 2
# =============================================================================
echo ""
echo "########################################################"
echo "  PART 1: View Synthesis Run 2"
echo "  Resume from run 1 (0.2139), +6,180 illustrated pairs"
echo "########################################################"
echo ""

# ---------------------------------------------------------------------------
# 1.0 Download view synthesis checkpoint
# ---------------------------------------------------------------------------
echo "[1.0] Downloading view synthesis run 1 checkpoint..."
mkdir -p checkpoints/view_synthesis

RUN1_CKPT="checkpoints/view_synthesis/run1_best.pt"
if [ ! -f "$RUN1_CKPT" ]; then
    rclone copy hetzner:strata-training-data/checkpoints_view_synthesis_run1/run1_best.pt \
        ./checkpoints/view_synthesis/ --transfers 32 --fast-list -P
fi
echo "  Checkpoint: $RUN1_CKPT"
echo ""

# ---------------------------------------------------------------------------
# 1.1 Download view synthesis data
# ---------------------------------------------------------------------------
echo "[1.1] Downloading view synthesis data..."
mkdir -p ./data/training ./data/tars

# Demo pairs (6,180 illustrated pairs from turnaround sheets)
echo "  Downloading demo_back_view_pairs.tar (3.4 GB)..."
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
    rclone copy "hetzner:strata-training-data/tars/${tar_name}" ./data/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "./data/tars/${tar_name}" ]; then
        mkdir -p "$extract_dir"
        tar xf "./data/tars/${tar_name}" -C ./data/training/
        rm -f "./data/tars/${tar_name}"
    else
        echo "  WARN: ${tar_name} not found"
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
echo "  Total pairs: $PAIR_COUNT (incl $DEMO_COUNT illustrated at weight 10.0)"
echo ""

# ---------------------------------------------------------------------------
# 1.2 Train view synthesis (resume from run 1)
# ---------------------------------------------------------------------------
echo "[1.2] Training view synthesis model..."

rm -f checkpoints/view_synthesis/latest.pt
python3 -c "
import torch
ckpt = torch.load('$RUN1_CKPT', map_location='cpu', weights_only=False)
ckpt['epoch'] = -1
torch.save(ckpt, 'checkpoints/view_synthesis/latest.pt')
"

python3 -m training.train_view_synthesis \
    --config training/configs/view_synthesis_run2.yaml \
    2>&1 | tee "$LOG_DIR/view_synthesis_train.log"

echo ""

# ---------------------------------------------------------------------------
# 1.3 Export + Upload view synthesis
# ---------------------------------------------------------------------------
echo "[1.3] Exporting view synthesis ONNX..."

mkdir -p ./models/onnx

if [ -f "checkpoints/view_synthesis/best.pt" ]; then
    cp checkpoints/view_synthesis/best.pt checkpoints/view_synthesis/run2_best.pt

    python3 -m training.export_onnx \
        --model view_synthesis \
        --checkpoint checkpoints/view_synthesis/run2_best.pt \
        --output ./models/onnx/view_synthesis_run2.onnx \
        2>&1 | tee "$LOG_DIR/view_synthesis_export.log"

    ONNX_SIZE=$(du -h ./models/onnx/view_synthesis_run2.onnx 2>/dev/null | cut -f1)
    echo "  Exported view_synthesis_run2.onnx ($ONNX_SIZE)"
fi

rclone copy ./checkpoints/view_synthesis/run2_best.pt \
    hetzner:strata-training-data/checkpoints_view_synthesis_run2/ \
    --transfers 4 --fast-list --size-only -P

if [ -f "./models/onnx/view_synthesis_run2.onnx" ]; then
    rclone copy ./models/onnx/view_synthesis_run2.onnx \
        hetzner:strata-training-data/models/view_synthesis_run2/ \
        --transfers 4 --fast-list --size-only -P
fi

echo ""
echo "  View Synthesis Run 2 complete."
echo "  Results:"
grep -E "best val/l1|New best" "$LOG_DIR/view_synthesis_train.log" 2>/dev/null | tail -3 || true
echo ""

# =============================================================================
# PART 2: WEIGHTS RETRAIN
# =============================================================================
echo ""
echo "########################################################"
echo "  PART 2: Weights Retrain (run 20 seg encoder)"
echo "  Fix limb deformation artifacts"
echo "########################################################"
echo ""

# ---------------------------------------------------------------------------
# 2.0 Download seg checkpoint + humanrig data
# ---------------------------------------------------------------------------
echo "[2.0] Downloading seg checkpoint + humanrig data..."

mkdir -p checkpoints/segmentation data_cloud

SEG_CKPT="checkpoints/segmentation/run20_best.pt"
if [ ! -f "$SEG_CKPT" ]; then
    rclone copy hetzner:strata-training-data/checkpoints_run20_seg/segmentation/run20_best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
fi
echo "  Seg checkpoint: $SEG_CKPT"

if [ ! -d "data_cloud/humanrig" ] || [ -z "$(ls data_cloud/humanrig/ 2>/dev/null | head -1)" ]; then
    echo "  Downloading humanrig.tar (16.8 GB)..."
    rclone copy hetzner:strata-training-data/tars/humanrig.tar ./data/tars/ \
        --transfers 32 --fast-list -P
    tar xf ./data/tars/humanrig.tar -C ./data_cloud/
    rm -f ./data/tars/humanrig.tar
fi
HR_COUNT=$(ls -d ./data_cloud/humanrig/*/ 2>/dev/null | wc -l | tr -d ' ')
echo "  humanrig: $HR_COUNT examples"
echo ""

# ---------------------------------------------------------------------------
# 2.1 Precompute encoder features
# ---------------------------------------------------------------------------
echo "[2.1] Precomputing encoder features with run 20 seg model..."

python3 -m training.data.precompute_encoder_features \
    --segmentation-checkpoint "$SEG_CKPT" \
    --data-dirs ./data_cloud/humanrig \
    --output-dir ./data_cloud/encoder_features \
    --device cuda \
    2>&1 | tee "$LOG_DIR/precompute_encoder.log"

echo "  Encoder features precomputed."
echo ""

# ---------------------------------------------------------------------------
# 2.2 Train weights model
# ---------------------------------------------------------------------------
echo "[2.2] Training weights model..."

mkdir -p checkpoints/weights
rm -f checkpoints/weights/latest.pt

python3 -m training.train_weights \
    --config training/configs/weights_ship.yaml \
    2>&1 | tee "$LOG_DIR/weights_train.log"

echo ""

# ---------------------------------------------------------------------------
# 2.3 Export + Upload weights
# ---------------------------------------------------------------------------
echo "[2.3] Exporting weights ONNX..."

if [ -f "checkpoints/weights/best.pt" ]; then
    cp checkpoints/weights/best.pt checkpoints/weights/retrain_run20_best.pt

    python3 -m training.export_onnx \
        --model weights \
        --checkpoint checkpoints/weights/retrain_run20_best.pt \
        --output ./models/onnx/weights_retrain.onnx \
        2>&1 | tee "$LOG_DIR/weights_export.log"

    ONNX_SIZE=$(du -h ./models/onnx/weights_retrain.onnx 2>/dev/null | cut -f1)
    echo "  Exported weights_retrain.onnx ($ONNX_SIZE)"
fi

rclone copy ./checkpoints/weights/retrain_run20_best.pt \
    hetzner:strata-training-data/checkpoints_weights_retrain/ \
    --transfers 4 --fast-list --size-only -P

if [ -f "./models/onnx/weights_retrain.onnx" ]; then
    rclone copy ./models/onnx/weights_retrain.onnx \
        hetzner:strata-training-data/models/weights_retrain/ \
        --transfers 4 --fast-list --size-only -P
fi

# Upload all logs
rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/march31_${TIMESTAMP}/ \
    --transfers 4 --fast-list -P

echo ""
echo "============================================"
echo "  ALL DONE!"
echo "  Finished: $(date)"
echo ""
echo "  View Synthesis Run 2:"
grep -E "best val/l1|Training complete" "$LOG_DIR/view_synthesis_train.log" 2>/dev/null | tail -2 || true
echo ""
echo "  Weights Retrain:"
grep -E "best val/mae|Training complete" "$LOG_DIR/weights_train.log" 2>/dev/null | tail -2 || true
echo ""
echo "  To download models:"
echo "    rclone copy hetzner:strata-training-data/models/view_synthesis_run2/ /Volumes/TAMWoolff/data/models/view_synthesis_run2/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/weights_retrain/ /Volumes/TAMWoolff/data/models/weights_retrain/ --transfers 32 --fast-list -P"
echo "============================================"

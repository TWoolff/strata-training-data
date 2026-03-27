#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 21 Seg (A100)
#
# Re-pseudo-label with run 20 model + fix neck regression.
# +291 new illustrated images. Boundary softening with neck/accessory excluded.
# Resume from run 20 checkpoint (0.6171 val mIoU / 0.6485 test mIoU).
#
# Estimated: ~4 hrs total on A100 (pseudo-label + Marigold + train)
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_seg_run21.sh
#   ./training/run_seg_run21.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run21_seg_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 21 Seg"
echo "  Re-pseudo-label + fix neck regression"
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

if [ "$PREFLIGHT_FAIL" -ne 0 ]; then
    echo "  Pre-flight failed."
    exit 1
fi
echo ""

# ---------------------------------------------------------------------------
# 0. Download checkpoints + frozen splits
# ---------------------------------------------------------------------------
echo "[0/6] Downloading checkpoint + frozen splits..."

mkdir -p checkpoints/segmentation

RUN20_CKPT="checkpoints/segmentation/run20_best.pt"
if [ ! -f "$RUN20_CKPT" ]; then
    rclone copy hetzner:strata-training-data/checkpoints_run20_seg/segmentation/run20_best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
fi
echo "  Run 20 checkpoint: $RUN20_CKPT"

# Frozen splits
mkdir -p data_cloud
if [ ! -f "data_cloud/frozen_val_test.json" ]; then
    rclone copy hetzner:strata-training-data/data_cloud/frozen_val_test.json \
        ./data_cloud/ --transfers 4 --fast-list -P
fi
echo "  Frozen splits: data_cloud/frozen_val_test.json"
echo ""

# ---------------------------------------------------------------------------
# 1. Download datasets
# ---------------------------------------------------------------------------
echo "[1/6] Downloading datasets..."

mkdir -p data_cloud data/tars

download_tar() {
    local tar_name="$1"
    local extract_dir="$2"

    if [ -d "$extract_dir" ] && [ "$(ls "$extract_dir"/ 2>/dev/null | head -1)" ]; then
        local count=$(ls -d "$extract_dir"/*/ 2>/dev/null | wc -l | tr -d ' ')
        echo "  $(basename "$extract_dir"): $count examples (exists)"
        return 0
    fi

    echo "  Downloading $tar_name..."
    rclone copy "hetzner:strata-training-data/tars/$tar_name" ./data/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "./data/tars/$tar_name" ]; then
        tar xf "./data/tars/$tar_name" -C ./data_cloud/
        rm -f "./data/tars/$tar_name"
    else
        echo "  WARN: $tar_name not found in bucket"
    fi

    local count=$(ls -d "$extract_dir"/*/ 2>/dev/null | wc -l | tr -d ' ')
    echo "  $(basename "$extract_dir"): $count examples"
}

download_tar "humanrig.tar" "./data_cloud/humanrig"
download_tar "vroid_cc0.tar" "./data_cloud/vroid_cc0"
download_tar "meshy_cc0_restructured.tar" "./data_cloud/meshy_cc0_textured"
download_tar "gemini_li_converted.tar" "./data_cloud/gemini_li_converted"
download_tar "cvat_annotated.tar" "./data_cloud/cvat_annotated"
download_tar "sora_diverse.tar" "./data_cloud/sora_diverse"
download_tar "flux_diverse_clean.tar" "./data_cloud/flux_diverse_clean"

# Download new illustrated images (incremental tar)
echo "  Downloading sora_diverse_new.tar (291 new images)..."
if rclone ls hetzner:strata-training-data/tars/sora_diverse_new.tar &>/dev/null; then
    rclone copy hetzner:strata-training-data/tars/sora_diverse_new.tar ./data/tars/ \
        --transfers 32 --fast-list -P
    tar xf ./data/tars/sora_diverse_new.tar -C ./data_cloud/sora_diverse/ --strip-components=1 2>/dev/null || \
    tar xf ./data/tars/sora_diverse_new.tar -C ./data_cloud/ 2>/dev/null || true
    rm -f ./data/tars/sora_diverse_new.tar
fi

SORA_COUNT=$(ls -d ./data_cloud/sora_diverse/*/ 2>/dev/null | wc -l | tr -d ' ')
echo "  sora_diverse total: $SORA_COUNT examples"
echo ""

# ---------------------------------------------------------------------------
# 2. Re-pseudo-label sora_diverse with run 20 model
# ---------------------------------------------------------------------------
echo "[2/6] Re-pseudo-labeling sora_diverse with run 20 model..."

python3 scripts/batch_pseudo_label.py \
    --input-dir ./data_cloud/sora_diverse \
    --output-dir ./data_cloud/sora_diverse \
    --checkpoint "$RUN20_CKPT" \
    --device cuda \
    2>&1 | tee "$LOG_DIR/pseudo_label.log"

echo "  Pseudo-labeling complete."
echo ""

# ---------------------------------------------------------------------------
# 3. Marigold enrichment (new images only)
# ---------------------------------------------------------------------------
echo "[3/6] Marigold enrichment (new images only)..."

for ds in sora_diverse flux_diverse_clean gemini_li_converted; do
    ds_dir="./data_cloud/$ds"
    if [ -d "$ds_dir" ]; then
        python3 run_normals_enrich.py \
            --input-dir "$ds_dir" \
            --only-missing \
            --batch-size 16 \
            2>&1 | tee -a "$LOG_DIR/enrich.log"
        echo "  $ds: depth+normals OK."
    fi
done
echo ""

# ---------------------------------------------------------------------------
# 4. Quality filter (re-run for sora_diverse)
# ---------------------------------------------------------------------------
echo "[4/6] Quality filter..."

# Remove old quality filter for sora_diverse so it re-runs
rm -f ./data_cloud/sora_diverse/quality_filter.json

for ds_dir in ./data_cloud/humanrig ./data_cloud/vroid_cc0 ./data_cloud/meshy_cc0_restructured ./data_cloud/gemini_li_converted ./data_cloud/cvat_annotated ./data_cloud/flux_diverse_clean; do
    ds_name=$(basename "$ds_dir")
    if [ -f "$ds_dir/quality_filter.json" ]; then
        echo "  $ds_name: quality_filter.json exists, skipping."
    else
        echo "  $ds_name: running quality filter..."
        python3 scripts/filter_seg_quality.py \
            --data-dir "$ds_dir" \
            --min-regions 4 --max-single-region 0.70 --min-foreground 0.05 \
            2>&1 | tee -a "$LOG_DIR/quality_filter.log"
    fi
done

# sora_diverse: re-run with new pseudo-labels
echo "  sora_diverse: running quality filter (re-pseudo-labeled)..."
python3 scripts/filter_seg_quality.py \
    --data-dir ./data_cloud/sora_diverse \
    --min-regions 4 --max-single-region 0.70 --min-foreground 0.05 \
    2>&1 | tee -a "$LOG_DIR/quality_filter.log"

echo "  Quality filter complete."
echo ""

# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------
echo "[5/6] Training SEGMENTATION model..."
echo "  Resuming from: $RUN20_CKPT (run 20 best)"
echo "  Config: training/configs/segmentation_a100_run21.yaml"
echo "  Boundary softening: radius=2, neck/accessory excluded"
echo ""

python3 -m training.train_segmentation \
    --config training/configs/segmentation_a100_run21.yaml \
    --resume "$RUN20_CKPT" \
    --reset-epochs \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""

# ---------------------------------------------------------------------------
# 6. Export + Upload
# ---------------------------------------------------------------------------
echo "[6/6] Exporting ONNX + uploading..."

cp checkpoints/segmentation/best.pt checkpoints/segmentation/run21_best.pt

python3 -m training.export_onnx \
    --model segmentation \
    --checkpoint checkpoints/segmentation/run21_best.pt \
    --output ./models/onnx/segmentation_run21.onnx \
    2>&1 | tee "$LOG_DIR/export.log"

# Upload
rclone copy checkpoints/segmentation/run21_best.pt \
    hetzner:strata-training-data/checkpoints_run21_seg/segmentation/ \
    --transfers 4 --fast-list --size-only -P

rclone copy ./models/onnx/segmentation_run21.onnx \
    hetzner:strata-training-data/models/onnx_run21_seg/ \
    --transfers 4 --fast-list --size-only -P

rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run21_seg_${TIMESTAMP}/ \
    --transfers 4 --fast-list -P

echo ""
echo "============================================"
echo "  Run 21 complete!"
echo "  Finished: $(date)"
echo "  Results:"
grep -E "best mIoU|mIoU=" "$LOG_DIR/train.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  To download:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_run21_seg/ /Volumes/TAMWoolff/data/checkpoints_run21_seg/ --transfers 32 --fast-list -P"
echo "============================================"

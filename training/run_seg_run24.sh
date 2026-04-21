#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 24 Seg (A100)
#
# Expands training data with gemini_diverse (3,207 new illustrated chars,
# pseudo-labeled via run 20 checkpoint). Also re-pseudo-labels sora_diverse.
# Resume from run 20 checkpoint (0.6485 test mIoU).
#
# Estimated: ~4-5 hrs total (downloads ~20 min, pseudo-label ~20 min,
# Marigold ~15 min, train ~3-4 hrs)
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_seg_run24.sh
#   ./training/run_seg_run24.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run24_seg_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 24 Seg"
echo "  + gemini_diverse (3,207 new illustrated)"
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
# 0. Download checkpoint + frozen splits
# ---------------------------------------------------------------------------
echo "[0/6] Downloading checkpoint + frozen splits..."

mkdir -p checkpoints/segmentation data_cloud data/tars

RUN20_CKPT="checkpoints/segmentation/run20_best.pt"
if [ ! -f "$RUN20_CKPT" ]; then
    rclone copy hetzner:strata-training-data/checkpoints_run20_seg/segmentation/run20_best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
fi
echo "  Run 20 checkpoint: $RUN20_CKPT"

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
download_tar "gemini_diverse.tar" "./data_cloud/gemini_diverse"
echo ""

# ---------------------------------------------------------------------------
# 2. Pseudo-label new datasets with run 20 model
# ---------------------------------------------------------------------------
echo "[2/6] Pseudo-labeling gemini_diverse + re-pseudo-label sora_diverse..."

# gemini_diverse has no seg masks yet — generate them
python3 scripts/batch_pseudo_label.py \
    --input-dir ./data_cloud/gemini_diverse \
    --output-dir ./data_cloud/gemini_diverse \
    --checkpoint "$RUN20_CKPT" \
    --device cuda \
    2>&1 | tee "$LOG_DIR/pseudo_label_gemini_diverse.log"

# Re-pseudo-label sora_diverse (run 21 behavior)
python3 scripts/batch_pseudo_label.py \
    --input-dir ./data_cloud/sora_diverse \
    --output-dir ./data_cloud/sora_diverse \
    --checkpoint "$RUN20_CKPT" \
    --device cuda \
    2>&1 | tee "$LOG_DIR/pseudo_label_sora.log"

echo "  Pseudo-labeling complete."
echo ""

# ---------------------------------------------------------------------------
# 3. Marigold enrichment (depth + normals, only on images missing them)
# ---------------------------------------------------------------------------
echo "[3/6] Marigold enrichment..."

for ds in gemini_diverse sora_diverse flux_diverse_clean gemini_li_converted; do
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
# 4. Quality filter
# ---------------------------------------------------------------------------
echo "[4/6] Quality filter..."

# Force-rerun for datasets that got new pseudo-labels
rm -f ./data_cloud/gemini_diverse/quality_filter.json
rm -f ./data_cloud/sora_diverse/quality_filter.json

for ds_dir in ./data_cloud/humanrig ./data_cloud/vroid_cc0 ./data_cloud/meshy_cc0_restructured ./data_cloud/gemini_li_converted ./data_cloud/cvat_annotated ./data_cloud/flux_diverse_clean ./data_cloud/sora_diverse ./data_cloud/gemini_diverse; do
    ds_name=$(basename "$ds_dir")
    if [ -f "$ds_dir/quality_filter.json" ]; then
        echo "  $ds_name: quality_filter.json exists, skipping."
    else
        echo "  $ds_name: running quality filter..."
        python3 scripts/filter_seg_quality.py \
            --data-dirs "$ds_dir" \
            --min-regions 4 --max-single-region 0.70 --min-foreground 0.05 \
            2>&1 | tee -a "$LOG_DIR/quality_filter.log"
    fi
done

echo "  Quality filter complete."
echo ""

# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------
echo "[5/6] Training SEGMENTATION model..."
echo "  Resuming from: $RUN20_CKPT (run 20 best, 0.6485 test mIoU)"
echo "  Config: training/configs/segmentation_a100_run24.yaml"
echo ""

python3 -m training.train_segmentation \
    --config training/configs/segmentation_a100_run24.yaml \
    --resume "$RUN20_CKPT" \
    --reset-epochs \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""

# ---------------------------------------------------------------------------
# 6. Export + Upload
# ---------------------------------------------------------------------------
echo "[6/6] Exporting ONNX + uploading..."

cp checkpoints/segmentation/best.pt checkpoints/segmentation/run24_best.pt

python3 -m training.export_onnx \
    --model segmentation \
    --checkpoint checkpoints/segmentation/run24_best.pt \
    --output ./models/onnx/segmentation_run24.onnx \
    2>&1 | tee "$LOG_DIR/export.log"

rclone copy checkpoints/segmentation/run24_best.pt \
    hetzner:strata-training-data/checkpoints_run24_seg/segmentation/ \
    --transfers 4 --fast-list --size-only -P

rclone copy ./models/onnx/segmentation_run24.onnx \
    hetzner:strata-training-data/models/onnx_run24_seg/ \
    --transfers 4 --fast-list --size-only -P

rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run24_seg_${TIMESTAMP}/ \
    --transfers 4 --fast-list -P

echo ""
echo "============================================"
echo "  Run 24 complete!"
echo "  Finished: $(date)"
echo "  Results:"
grep -E "best mIoU|mIoU=" "$LOG_DIR/train.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  To download:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_run24_seg/ /Volumes/TAMWoolff/data/checkpoints_run24_seg/ --transfers 32 --fast-list -P"
echo "============================================"

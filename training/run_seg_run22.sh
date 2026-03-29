#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 22 Seg (A100)
#
# Re-pseudo-labeled data + boundary softening (neck/hair_back excluded).
# Resume from run 20 checkpoint (0.6171 val mIoU) — better starting point
# than run 21 since run 20 already has boundary softening baked in.
#
# Estimated: ~5 hrs on A100 (download + train + export + upload)
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_seg_run22.sh
#   ./training/run_seg_run22.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run22_seg_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 22 Seg"
echo "  Boundary softening + re-pseudo-labeled data"
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

if ! python3 -c "from scipy.ndimage import gaussian_filter" 2>/dev/null; then
    echo "  FAIL: scipy not installed (needed for boundary softening)"
    PREFLIGHT_FAIL=1
else
    echo "  OK: scipy"
fi

if [ "$PREFLIGHT_FAIL" -ne 0 ]; then
    echo "  Pre-flight failed."
    exit 1
fi
echo ""

# ---------------------------------------------------------------------------
# 0. Download checkpoint + frozen splits
# ---------------------------------------------------------------------------
echo "[0/5] Downloading checkpoint + frozen splits..."

mkdir -p checkpoints/segmentation data_cloud

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
echo "  Frozen splits: OK"
echo ""

# ---------------------------------------------------------------------------
# 1. Download datasets
# ---------------------------------------------------------------------------
echo "[1/5] Downloading datasets..."

mkdir -p data/tars

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
download_tar "meshy_cc0_textured_restructured.tar" "./data_cloud/meshy_cc0_restructured"
download_tar "gemini_li_converted.tar" "./data_cloud/gemini_li_converted"
download_tar "cvat_annotated.tar" "./data_cloud/cvat_annotated"
download_tar "sora_diverse.tar" "./data_cloud/sora_diverse"
download_tar "flux_diverse_clean.tar" "./data_cloud/flux_diverse_clean"

# New illustrated images
if rclone ls hetzner:strata-training-data/tars/sora_diverse_new.tar &>/dev/null 2>&1; then
    echo "  Downloading sora_diverse_new.tar..."
    rclone copy hetzner:strata-training-data/tars/sora_diverse_new.tar ./data/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "./data/tars/sora_diverse_new.tar" ]; then
        tar xf ./data/tars/sora_diverse_new.tar -C ./data_cloud/ 2>/dev/null || true
        rm -f ./data/tars/sora_diverse_new.tar
    fi
fi

SORA_COUNT=$(ls -d ./data_cloud/sora_diverse/*/ 2>/dev/null | wc -l | tr -d ' ')
echo "  sora_diverse total: $SORA_COUNT"
echo ""

# ---------------------------------------------------------------------------
# 2. Re-pseudo-label sora_diverse with run 21 checkpoint
# ---------------------------------------------------------------------------
echo "[2/5] Re-pseudo-labeling sora_diverse..."

# Use run 21 checkpoint (better than run 20)
python3 scripts/batch_pseudo_label.py \
    --input-dir ./data_cloud/sora_diverse \
    --output-dir ./data_cloud/sora_diverse \
    --checkpoint "$RUN20_CKPT" \
    --device cuda \
    2>&1 | tee "$LOG_DIR/pseudo_label.log"

echo ""

# ---------------------------------------------------------------------------
# 3. Marigold + quality filter
# ---------------------------------------------------------------------------
echo "[3/5] Marigold enrichment + quality filter..."

for ds in sora_diverse flux_diverse_clean gemini_li_converted; do
    ds_dir="./data_cloud/$ds"
    if [ -d "$ds_dir" ]; then
        python3 run_normals_enrich.py \
            --input-dir "$ds_dir" \
            --only-missing \
            --batch-size 16 \
            2>&1 | tee -a "$LOG_DIR/enrich.log"
        echo "  $ds: enriched."
    fi
done

# Quality filter — re-run sora_diverse with new pseudo-labels
rm -f ./data_cloud/sora_diverse/quality_filter.json

for ds_dir in ./data_cloud/humanrig ./data_cloud/vroid_cc0 ./data_cloud/meshy_cc0_restructured \
              ./data_cloud/gemini_li_converted ./data_cloud/cvat_annotated ./data_cloud/flux_diverse_clean; do
    ds_name=$(basename "$ds_dir")
    if [ -f "$ds_dir/quality_filter.json" ]; then
        echo "  $ds_name: quality_filter exists, skipping."
    elif [ -d "$ds_dir" ]; then
        python3 scripts/filter_seg_quality.py \
            --data-dir "$ds_dir" \
            --min-regions 4 --max-single-region 0.70 --min-foreground 0.05 \
            2>&1 | tee -a "$LOG_DIR/quality_filter.log"
    fi
done

echo "  sora_diverse: quality filter (re-pseudo-labeled)..."
python3 scripts/filter_seg_quality.py \
    --data-dir ./data_cloud/sora_diverse \
    --min-regions 4 --max-single-region 0.70 --min-foreground 0.05 \
    2>&1 | tee -a "$LOG_DIR/quality_filter.log"

echo "  Quality filter complete."
echo ""

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
echo "[4/5] Training SEGMENTATION model..."
echo "  Resuming from: $RUN20_CKPT"
echo "  Config: training/configs/segmentation_a100_run22.yaml"
echo "  Boundary softening: radius=2 (computed on-the-fly, neck/hair_back excluded)"
echo ""

rm -f checkpoints/segmentation/latest.pt

python3 -m training.train_segmentation \
    --config training/configs/segmentation_a100_run22.yaml \
    --resume "$RUN20_CKPT" \
    --reset-epochs \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""

# ---------------------------------------------------------------------------
# 5. Export + Upload
# ---------------------------------------------------------------------------
echo "[5/5] Exporting ONNX + uploading..."

cp checkpoints/segmentation/best.pt checkpoints/segmentation/run22_best.pt

python3 -m training.export_onnx \
    --model segmentation \
    --checkpoint checkpoints/segmentation/run22_best.pt \
    --output ./models/onnx/segmentation_run22.onnx \
    2>&1 | tee "$LOG_DIR/export.log"

# Evaluate
python3 -m training.evaluate \
    --model segmentation \
    --checkpoint checkpoints/segmentation/run22_best.pt \
    --dataset-dir ./data_cloud/humanrig \
    --dataset-dir ./data_cloud/vroid_cc0 \
    --dataset-dir ./data_cloud/meshy_cc0_restructured \
    --dataset-dir ./data_cloud/gemini_li_converted \
    --dataset-dir ./data_cloud/cvat_annotated \
    --dataset-dir ./data_cloud/sora_diverse \
    --dataset-dir ./data_cloud/flux_diverse_clean \
    --output-dir ./evaluation_run22 \
    2>&1 | tee "$LOG_DIR/evaluate.log"

# Upload everything
rclone copy checkpoints/segmentation/run22_best.pt \
    hetzner:strata-training-data/checkpoints_run22_seg/segmentation/ \
    --transfers 4 --fast-list --size-only -P

rclone copy ./models/onnx/segmentation_run22.onnx \
    hetzner:strata-training-data/models/onnx_run22_seg/ \
    --transfers 4 --fast-list --size-only -P

rclone copy ./evaluation_run22/ hetzner:strata-training-data/evaluation_run22/ \
    --transfers 4 --fast-list -P

rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run22_seg_${TIMESTAMP}/ \
    --transfers 4 --fast-list -P

echo ""
echo "============================================"
echo "  Run 22 complete!"
echo "  Finished: $(date)"
echo ""
grep -E "best mIoU|mIoU=" "$LOG_DIR/train.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "Per-class eval:"
grep -A22 "Per-Class IoU" "$LOG_DIR/evaluate.log" 2>/dev/null || echo "  (check evaluation_run22)"
echo ""
echo "  To download:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_run22_seg/ /Volumes/TAMWoolff/data/checkpoints_run22_seg/ --transfers 32 --fast-list -P"
echo "============================================"
echo ""
echo ""

# =============================================================================
# BACK VIEW RUN 6 — Fine-tune on 5 demo characters
# =============================================================================
echo "============================================"
echo "  Starting Back View Run 6 (Demo Fine-tune)"
echo "  $(date)"
echo "============================================"
echo ""

./training/run_back_view_run6.sh

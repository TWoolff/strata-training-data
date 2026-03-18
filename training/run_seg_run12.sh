#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 12 Seg: Cleaned Illustrated Data (A100)
#
# Fine-tunes from run 10 best (0.5038 mIoU) with cleaned illustrated datasets:
#   - sora_diverse: 1,056 Sora/Gemini chars (fixed aspect ratio, run 10 labels)
#   - flux_diverse_clean: 1,569 FLUX chars (anatomical errors removed, run 10 labels)
#   - gemini_diverse dropped: superseded by sora_diverse
#
# Estimated: ~4-5 hrs on A100
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_seg_run12.sh
#   ./training/run_seg_run12.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run12_seg_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 12 Seg"
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

pip install -q scipy diffusers transformers accelerate 2>/dev/null || true
echo "  Dependencies OK."
echo ""

# ---------------------------------------------------------------------------
# 0. Download run 10 checkpoint
# ---------------------------------------------------------------------------
echo "[0/5] Downloading checkpoint..."
mkdir -p checkpoints/segmentation

RUN10_CKPT="checkpoints/segmentation/run10_best.pt"
if [ -f "$RUN10_CKPT" ]; then
    echo "  run10_best.pt already exists."
else
    echo "  Downloading run 10 checkpoint (0.5038 mIoU)..."
    rclone copy hetzner:strata-training-data/checkpoints_run10_seg/segmentation/run10_best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
    if [ ! -f "$RUN10_CKPT" ]; then
        echo "  FATAL: Could not download run10_best.pt"
        exit 1
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# 1. Download datasets
# ---------------------------------------------------------------------------
echo "[1/5] Downloading datasets..."
echo ""

mkdir -p ./data_cloud/tars

download_dataset() {
    local name="$1"
    local tar_name="${2:-$name}"
    local dir="./data_cloud/$name"
    local tar="./data_cloud/tars/${tar_name}.tar"
    local check="${3:-}"

    if [ -n "$check" ]; then
        if [ -d "$dir" ] && [ "$(find "$dir" -maxdepth 2 -name "$check" | head -1)" ]; then
            echo "  $name already exists."
            return 0
        fi
    elif [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null | head -1)" ]; then
        echo "  $name already exists."
        return 0
    fi

    rm -rf "$dir"
    echo "  Downloading $name..."
    rclone copy "hetzner:strata-training-data/tars/${tar_name}.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "$tar" ]; then
        echo "  Extracting $name..."
        tar xf "$tar" -C ./data_cloud/
        rm -f "$tar"
        COUNT=$(ls "./data_cloud/$name/" 2>/dev/null | wc -l | tr -d ' ')
        echo "  $name: $COUNT examples"
    else
        echo "  FATAL: $name tar not found in bucket."
        exit 1
    fi
}

download_dataset humanrig humanrig "segmentation.png"
download_dataset vroid_cc0 vroid_cc0
download_dataset gemini_li_converted gemini_li_converted
download_dataset cvat_annotated cvat_annotated
download_dataset sora_diverse sora_diverse "segmentation.png"
download_dataset flux_diverse_clean flux_diverse_clean "segmentation.png"

# meshy_cc0_textured (restructured tar)
MESHY_DIR="./data_cloud/meshy_cc0_textured"
MESHY_TAR="./data_cloud/tars/meshy_cc0_textured_restructured.tar"
if [ -d "$MESHY_DIR" ] && [ "$(find "$MESHY_DIR" -maxdepth 2 -name 'segmentation.png' | head -1)" ]; then
    echo "  meshy_cc0_textured already exists."
else
    echo "  Downloading meshy_cc0_textured..."
    rm -rf "$MESHY_DIR"
    rclone copy "hetzner:strata-training-data/tars/meshy_cc0_textured_restructured.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "$MESHY_TAR" ]; then
        tar xf "$MESHY_TAR" -C ./data_cloud/
        rm -f "$MESHY_TAR"
        [ -d "./data_cloud/meshy_cc0_restructured" ] && mv ./data_cloud/meshy_cc0_restructured "$MESHY_DIR"
    else
        echo "  FATAL: meshy tar not found."
        exit 1
    fi
fi

echo ""
echo "  Dataset summary:"
for ds in humanrig vroid_cc0 meshy_cc0_textured gemini_li_converted cvat_annotated sora_diverse flux_diverse_clean; do
    count=$(ls "./data_cloud/$ds/" 2>/dev/null | wc -l | tr -d ' ')
    echo "    $ds: $count examples"
done
echo ""

# ---------------------------------------------------------------------------
# 2. Marigold enrichment
# ---------------------------------------------------------------------------
echo "[2/5] Marigold enrichment..."
echo ""

for ds_dir in "./data_cloud/sora_diverse" "./data_cloud/flux_diverse_clean" "./data_cloud/gemini_li_converted"; do
    ds_name=$(basename "$ds_dir")
    MISSING=$(find "$ds_dir" -name "image.png" -exec sh -c '
        dir=$(dirname "$1"); [ ! -f "$dir/depth.png" ] && echo m
    ' _ {} \; | wc -l | tr -d ' ')
    if [ "$MISSING" -gt 0 ]; then
        echo "  $ds_name: $MISSING missing depth — running Marigold..."
        python run_normals_enrich.py \
            --input-dir "$ds_dir" \
            --only-missing \
            --batch-size 16 \
            2>&1 | tee "$LOG_DIR/enrich_${ds_name}.log"
    else
        echo "  $ds_name: depth+normals OK."
    fi
done
echo ""

# ---------------------------------------------------------------------------
# 3. Quality filter
# ---------------------------------------------------------------------------
echo "[3/5] Quality filter..."
echo ""

for ds in humanrig vroid_cc0 meshy_cc0_textured gemini_li_converted cvat_annotated sora_diverse flux_diverse_clean; do
    ds_dir="./data_cloud/$ds"
    if [ -d "$ds_dir" ]; then
        rm -f "$ds_dir/quality_filter.json"
        echo "  Filtering $ds..."
        python scripts/filter_seg_quality.py \
            --data-dirs "$ds_dir" \
            --output-dir "$ds_dir" \
            --min-regions 4 \
            --max-single-region 0.70 \
            --min-foreground 0.05 \
            2>&1 | tee -a "$LOG_DIR/quality_filter.log"
    fi
done
echo "  Quality filter complete."
echo ""

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
echo "[4/5] Training SEGMENTATION model..."
echo "  Resuming from: $RUN10_CKPT (run 10, 0.5038 mIoU)"
echo "  Config: training/configs/segmentation_a100_run12.yaml"
echo ""

python -m training.train_segmentation \
    --config training/configs/segmentation_a100_run12.yaml \
    --reset-epochs \
    --resume "$RUN10_CKPT" \
    2>&1 | tee "$LOG_DIR/segmentation.log"

echo ""
echo "  Segmentation training complete."
echo ""

# ---------------------------------------------------------------------------
# 5. Export + Upload
# ---------------------------------------------------------------------------
echo "[5/5] Exporting + uploading..."
echo ""

mkdir -p ./models/onnx

if [ -f "checkpoints/segmentation/best.pt" ]; then
    cp checkpoints/segmentation/best.pt checkpoints/segmentation/run12_best.pt
    python -m training.export_onnx \
        --model segmentation \
        --checkpoint checkpoints/segmentation/run12_best.pt \
        --output ./models/onnx/segmentation_run12.onnx \
        2>&1 | tee "$LOG_DIR/export.log"
    echo "  Exported segmentation_run12.onnx"
fi

rclone copy ./checkpoints/segmentation/run12_best.pt \
    hetzner:strata-training-data/checkpoints_run12_seg/segmentation/ \
    --transfers 32 --fast-list -P
rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run12_seg_${TIMESTAMP}/ \
    --transfers 32 --fast-list -P
if [ -f "./models/onnx/segmentation_run12.onnx" ]; then
    rclone copy ./models/onnx/segmentation_run12.onnx \
        hetzner:strata-training-data/models/onnx_run12_seg/ \
        --transfers 32 --fast-list -P
fi

echo ""
echo "============================================"
echo "  Run 12 Seg complete!"
echo "  Finished: $(date)"
echo ""
echo "  Results:"
grep -E "New best mIoU|mIoU=" "$LOG_DIR/segmentation.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  To download to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_run12_seg/ ./checkpoints_run12_seg/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/onnx_run12_seg/ ./models/onnx_run12_seg/ --transfers 32 --fast-list -P"
echo "============================================"

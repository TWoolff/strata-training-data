#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 14 Seg (A100)
#
# Incremental from run 13a (0.5425 mIoU):
#   - Updated sora_diverse (~290 new illustrated chars)
#   - Re-pseudo-label humanrig_posed with run 13a checkpoint
#   - humanrig_posed at weight 0.3 (cautious)
#
# Uses enriched tars where available (skips Marigold).
# Estimated: ~3-4 hrs total on A100
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_seg_run14.sh
#   ./training/run_seg_run14.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run14_seg_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 14 Seg"
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
# 0. Download run 13a checkpoint
# ---------------------------------------------------------------------------
echo "[0/5] Downloading checkpoint..."
mkdir -p checkpoints/segmentation

RUN13_CKPT="checkpoints/segmentation/run13_best.pt"
if [ -f "$RUN13_CKPT" ]; then
    echo "  run13_best.pt already exists."
else
    echo "  Downloading run 13a checkpoint..."
    rclone copy hetzner:strata-training-data/checkpoints_run13_seg/segmentation/run13_best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
    if [ ! -f "$RUN13_CKPT" ]; then
        echo "  FATAL: Could not download run13_best.pt"
        exit 1
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# 1. Download datasets (enriched tars first, fallback to regular)
# ---------------------------------------------------------------------------
echo "[1/5] Downloading datasets..."
echo ""

mkdir -p ./data_cloud/tars

download_dataset() {
    local name="$1"
    local tar_name="${2:-$name}"
    local dir="./data_cloud/$name"
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

    # Try enriched tar first (has depth+normals, skips Marigold)
    local enriched_tar="./data_cloud/tars/${name}_enriched.tar"
    echo "  Checking for ${name}_enriched.tar..."
    rclone copy "hetzner:strata-training-data/tars/${name}_enriched.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P 2>/dev/null || true
    if [ -f "$enriched_tar" ]; then
        echo "  Found enriched tar — extracting $name..."
        tar xf "$enriched_tar" -C ./data_cloud/
        rm -f "$enriched_tar"
        COUNT=$(ls "./data_cloud/$name/" 2>/dev/null | wc -l | tr -d ' ')
        echo "  $name (enriched): $COUNT examples"
        return 0
    fi

    # Fall back to regular tar
    local tar="./data_cloud/tars/${tar_name}.tar"
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
download_dataset humanrig_posed humanrig_posed "image.png"
download_dataset vroid_cc0 vroid_cc0
download_dataset gemini_li_converted gemini_li_converted
download_dataset cvat_annotated cvat_annotated
download_dataset sora_diverse sora_diverse "image.png"
download_dataset flux_diverse_clean flux_diverse_clean "image.png"
download_dataset toon_pseudo toon_pseudo "image.png"

# meshy_cc0_textured (restructured tar, special naming)
MESHY_DIR="./data_cloud/meshy_cc0_textured"
if [ -d "$MESHY_DIR" ] && [ "$(find "$MESHY_DIR" -maxdepth 2 -name 'segmentation.png' | head -1)" ]; then
    echo "  meshy_cc0_textured already exists."
else
    echo "  Downloading meshy_cc0_textured..."
    rm -rf "$MESHY_DIR"
    rclone copy "hetzner:strata-training-data/tars/meshy_cc0_textured_restructured.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    MESHY_TAR="./data_cloud/tars/meshy_cc0_textured_restructured.tar"
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
for ds in humanrig humanrig_posed vroid_cc0 meshy_cc0_textured gemini_li_converted cvat_annotated sora_diverse flux_diverse_clean toon_pseudo; do
    count=$(ls "./data_cloud/$ds/" 2>/dev/null | wc -l | tr -d ' ')
    echo "    $ds: $count examples"
done
echo ""

# ---------------------------------------------------------------------------
# 1b. Re-pseudo-label humanrig_posed with run 13a checkpoint
# ---------------------------------------------------------------------------
echo "[1b/5] Re-pseudo-labeling humanrig_posed with run 13a checkpoint..."
echo ""

# Delete old pseudo-labels and quality filter (from run 12 model — bad labels)
echo "  Removing old pseudo-labels from humanrig_posed..."
find ./data_cloud/humanrig_posed -name "segmentation.png" -delete 2>/dev/null || true
rm -f ./data_cloud/humanrig_posed/quality_filter.json

echo "  Pseudo-labeling with run 13a checkpoint..."
python scripts/batch_pseudo_label.py \
    --input-dir "./data_cloud/humanrig_posed" \
    --output-dir "./data_cloud/humanrig_posed" \
    --checkpoint "$RUN13_CKPT" \
    --only-missing \
    2>&1 | tee "$LOG_DIR/pseudo_label_humanrig_posed.log"

# Also pseudo-label sora_diverse new images (if any missing)
for ds in sora_diverse toon_pseudo; do
    ds_dir="./data_cloud/$ds"
    if [ ! -d "$ds_dir" ]; then continue; fi
    TOTAL=$(find "$ds_dir" -maxdepth 2 -name "image.png" 2>/dev/null | wc -l | tr -d ' ')
    HAVE_SEG=$(find "$ds_dir" -maxdepth 2 -name "segmentation.png" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$TOTAL" -gt 0 ] && [ "$HAVE_SEG" -lt "$TOTAL" ]; then
        echo "  $ds: $TOTAL images, $HAVE_SEG labeled — pseudo-labeling remainder..."
        python scripts/batch_pseudo_label.py \
            --input-dir "$ds_dir" \
            --output-dir "$ds_dir" \
            --checkpoint "$RUN13_CKPT" \
            --only-missing \
            2>&1 | tee "$LOG_DIR/pseudo_label_${ds}.log"
    else
        echo "  $ds: all examples have segmentation masks."
    fi
done
echo ""

# ---------------------------------------------------------------------------
# 2. Marigold enrichment (only for new/missing — enriched tars skip this)
# ---------------------------------------------------------------------------
echo "[2/5] Marigold enrichment (new images only)..."
echo ""

for ds_dir in "./data_cloud/sora_diverse" "./data_cloud/flux_diverse_clean" "./data_cloud/gemini_li_converted" "./data_cloud/toon_pseudo"; do
    ds_name=$(basename "$ds_dir")
    if [ ! -d "$ds_dir" ]; then continue; fi
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
# 3. Quality filter (re-run for re-pseudo-labeled datasets)
# ---------------------------------------------------------------------------
echo "[3/5] Quality filter..."
echo ""

for ds in humanrig humanrig_posed vroid_cc0 meshy_cc0_textured gemini_li_converted cvat_annotated sora_diverse flux_diverse_clean toon_pseudo; do
    ds_dir="./data_cloud/$ds"
    if [ -d "$ds_dir" ]; then
        if [ -f "$ds_dir/quality_filter.json" ]; then
            echo "  $ds: quality_filter.json exists, skipping."
        else
            echo "  Filtering $ds..."
            python scripts/filter_seg_quality.py \
                --data-dirs "$ds_dir" \
                --output-dir "$ds_dir" \
                --min-regions 4 \
                --max-single-region 0.70 \
                --min-foreground 0.05 \
                2>&1 | tee -a "$LOG_DIR/quality_filter.log"
        fi
    fi
done
echo "  Quality filter complete."
echo ""

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
echo "[4/5] Training SEGMENTATION model..."
echo "  Resuming from: $RUN13_CKPT (run 13a best)"
echo "  Config: training/configs/segmentation_a100_run14.yaml"
echo ""

python -m training.train_segmentation \
    --config training/configs/segmentation_a100_run14.yaml \
    --reset-epochs \
    --resume "$RUN13_CKPT" \
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
    cp checkpoints/segmentation/best.pt checkpoints/segmentation/run14_best.pt
    python -m training.export_onnx \
        --model segmentation \
        --checkpoint checkpoints/segmentation/run14_best.pt \
        --output ./models/onnx/segmentation_run14.onnx \
        2>&1 | tee "$LOG_DIR/export.log"
    echo "  Exported segmentation_run14.onnx"
fi

rclone copy ./checkpoints/segmentation/run14_best.pt \
    hetzner:strata-training-data/checkpoints_run14_seg/segmentation/ \
    --transfers 32 --fast-list -P
rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run14_seg_${TIMESTAMP}/ \
    --transfers 32 --fast-list -P
if [ -f "./models/onnx/segmentation_run14.onnx" ]; then
    rclone copy ./models/onnx/segmentation_run14.onnx \
        hetzner:strata-training-data/models/onnx_run14_seg/ \
        --transfers 32 --fast-list -P
fi

echo ""
echo "============================================"
echo "  Run 14 Seg complete!"
echo "  Finished: $(date)"
echo "  Results:"
grep -E "New best mIoU|mIoU=" "$LOG_DIR/segmentation.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  To download to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_run14_seg/ /Volumes/TAMWoolff/data/checkpoints_run14_seg/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/onnx_run14_seg/ /Volumes/TAMWoolff/data/models/onnx_run14_seg/ --transfers 32 --fast-list -P"
echo "============================================"

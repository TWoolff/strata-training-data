#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 13 Seg: humanrig_posed GT + toon pseudo-labels (A100)
#
# Fine-tunes from run 12 best with massive new data:
#   - humanrig_posed: ~35K GT posed renders (13 poses × 3 angles)
#   - toon_pseudo: ~10K pseudo-labeled toon-style renders
#
# Estimated: ~6-8 hrs on A100 (seg) + ~3 hrs (back view) = ~10 hrs total
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_seg_run13.sh
#   ./training/run_seg_run13.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run13_seg_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 13 Seg"
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
# 0. Download run 12 checkpoint
# ---------------------------------------------------------------------------
echo "[0/5] Downloading checkpoint..."
mkdir -p checkpoints/segmentation

RUN12_CKPT="checkpoints/segmentation/run12_best.pt"
if [ -f "$RUN12_CKPT" ]; then
    echo "  run12_best.pt already exists."
else
    echo "  Downloading run 12 checkpoint..."
    rclone copy hetzner:strata-training-data/checkpoints_run12_seg/segmentation/run12_best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
    if [ ! -f "$RUN12_CKPT" ]; then
        echo "  FATAL: Could not download run12_best.pt"
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
download_dataset humanrig_posed humanrig_posed "segmentation.png"
download_dataset vroid_cc0 vroid_cc0
download_dataset gemini_li_converted gemini_li_converted
download_dataset cvat_annotated cvat_annotated
download_dataset sora_diverse sora_diverse "segmentation.png"
download_dataset flux_diverse_clean flux_diverse_clean "segmentation.png"
download_dataset toon_pseudo toon_pseudo "segmentation.png"

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
for ds in humanrig humanrig_posed vroid_cc0 meshy_cc0_textured gemini_li_converted cvat_annotated sora_diverse flux_diverse_clean toon_pseudo; do
    count=$(ls "./data_cloud/$ds/" 2>/dev/null | wc -l | tr -d ' ')
    echo "    $ds: $count examples"
done
echo ""

# ---------------------------------------------------------------------------
# 2. Marigold enrichment (only for illustrated datasets)
# ---------------------------------------------------------------------------
echo "[2/5] Marigold enrichment..."
echo ""

for ds_dir in "./data_cloud/sora_diverse" "./data_cloud/flux_diverse_clean" "./data_cloud/gemini_li_converted" "./data_cloud/toon_pseudo"; do
    ds_name=$(basename "$ds_dir")
    if [ ! -d "$ds_dir" ]; then
        continue
    fi
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

for ds in humanrig humanrig_posed vroid_cc0 meshy_cc0_textured gemini_li_converted cvat_annotated sora_diverse flux_diverse_clean toon_pseudo; do
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
echo "  Resuming from: $RUN12_CKPT (run 12 best)"
echo "  Config: training/configs/segmentation_a100_run13.yaml"
echo ""

python -m training.train_segmentation \
    --config training/configs/segmentation_a100_run13.yaml \
    --reset-epochs \
    --resume "$RUN12_CKPT" \
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
    cp checkpoints/segmentation/best.pt checkpoints/segmentation/run13_best.pt
    python -m training.export_onnx \
        --model segmentation \
        --checkpoint checkpoints/segmentation/run13_best.pt \
        --output ./models/onnx/segmentation_run13.onnx \
        2>&1 | tee "$LOG_DIR/export.log"
    echo "  Exported segmentation_run13.onnx"
fi

rclone copy ./checkpoints/segmentation/run13_best.pt \
    hetzner:strata-training-data/checkpoints_run13_seg/segmentation/ \
    --transfers 32 --fast-list -P
rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run13_seg_${TIMESTAMP}/ \
    --transfers 32 --fast-list -P
if [ -f "./models/onnx/segmentation_run13.onnx" ]; then
    rclone copy ./models/onnx/segmentation_run13.onnx \
        hetzner:strata-training-data/models/onnx_run13_seg/ \
        --transfers 32 --fast-list -P
fi

echo ""
echo "============================================"
echo "  Run 13 Seg complete!"
echo "  Finished: $(date)"
echo "  Results:"
grep -E "New best mIoU|mIoU=" "$LOG_DIR/segmentation.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Back View Run 3
# ---------------------------------------------------------------------------
echo "============================================"
echo "  Strata Training — Back View Run 3"
echo "  Started: $(date)"
echo "============================================"
echo ""

echo "[A/D] Downloading back view pairs..."
mkdir -p ./data/training/back_view_pairs ./data/tars

# Rigged pairs (1,085 triplets)
if [ -z "$(ls -A ./data/training/back_view_pairs 2>/dev/null)" ]; then
    rclone copy hetzner:strata-training-data/tars/back_view_pairs.tar ./data/tars/ \
        --transfers 16 --fast-list --size-only -P 2>&1 | tee "$LOG_DIR/bv_download.log"
    echo "  Extracting back_view_pairs.tar..."
    tar xf ./data/tars/back_view_pairs.tar -C ./data/training/
    if [ -d "./data/training/back_view_pairs_merged" ]; then
        mv ./data/training/back_view_pairs_merged/pair_* ./data/training/back_view_pairs/ 2>/dev/null || true
        rm -rf ./data/training/back_view_pairs_merged
    fi
    rm -f ./data/tars/back_view_pairs.tar
else
    echo "  back_view_pairs already exists, skipping."
fi

# Unrigged pairs (~960 additional triplets)
UNRIGGED_MARKER="./data/training/.back_view_pairs_unrigged_done"
if [ ! -f "$UNRIGGED_MARKER" ]; then
    rclone copy hetzner:strata-training-data/tars/back_view_pairs_unrigged.tar ./data/tars/ \
        --transfers 16 --fast-list --size-only -P 2>&1 | tee -a "$LOG_DIR/bv_download.log"
    echo "  Extracting back_view_pairs_unrigged.tar..."
    tar xf ./data/tars/back_view_pairs_unrigged.tar -C ./data/training/back_view_pairs/ --strip-components=1
    rm -f ./data/tars/back_view_pairs_unrigged.tar
    UNRIGGED_COUNT=$(ls -d ./data/training/back_view_pairs/pair_* 2>/dev/null | wc -l | tr -d ' ')
    echo "  After unrigged extraction: $UNRIGGED_COUNT total pairs"
    touch "$UNRIGGED_MARKER"
else
    echo "  back_view_pairs_unrigged already extracted, skipping."
fi

PAIR_COUNT=$(ls -d ./data/training/back_view_pairs/pair_* 2>/dev/null | wc -l | tr -d ' ')
echo "  Pairs: $PAIR_COUNT (target ~2045)"
if [ "$PAIR_COUNT" -lt 1500 ]; then
    echo "  WARN: Expected ~2000+ pairs, got $PAIR_COUNT"
fi
echo ""

echo "[B/D] Training back view model..."
python3 -m training.train_back_view \
    --config training/configs/back_view_a100.yaml \
    2>&1 | tee "$LOG_DIR/back_view_train.log"
echo ""

echo "[C/D] Exporting back view ONNX..."
mkdir -p ./models/onnx_back_view
python3 -m training.export_onnx \
    --model back_view \
    --checkpoint ./checkpoints/back_view/best.pt \
    --output ./models/onnx_back_view/back_view.onnx \
    2>&1 | tee "$LOG_DIR/bv_export.log"
BV_ONNX_SIZE=$(du -h ./models/onnx_back_view/back_view.onnx 2>/dev/null | cut -f1 || echo "unknown")
echo "  ONNX size: $BV_ONNX_SIZE"
echo ""

echo "[D/D] Uploading back view results..."
rclone copy ./checkpoints/back_view/ hetzner:strata-training-data/checkpoints_back_view_run3/ \
    --transfers 32 --fast-list -P
rclone copy ./models/onnx_back_view/ hetzner:strata-training-data/models/back_view_run3/ \
    --transfers 32 --fast-list -P
rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run13_${TIMESTAMP}/ \
    --transfers 32 --fast-list -P

echo ""
echo "============================================"
echo "  Run 13 complete!"
echo "  Finished: $(date)"
echo ""
echo "  Seg best mIoU:"
grep -E "New best mIoU" "$LOG_DIR/segmentation.log" 2>/dev/null | tail -3 || echo "  (check logs)"
echo "  Back view best val/l1:"
grep -E "New best val/l1" "$LOG_DIR/back_view_train.log" 2>/dev/null | tail -3 || echo "  (check logs)"
echo ""
echo "  To download to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_run13_seg/ ./checkpoints_run13_seg/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/onnx_run13_seg/ ./models/onnx_run13_seg/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/checkpoints_back_view_run3/ ./checkpoints_back_view_run3/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/back_view_run3/ ./models/back_view_run3/ --transfers 32 --fast-list -P"
echo "============================================"

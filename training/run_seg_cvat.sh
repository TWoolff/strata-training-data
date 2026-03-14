#!/usr/bin/env bash
# =============================================================================
# Strata Training — Seg + CVAT Manual Annotations (A100)
#
# Goal: Break past 0.37 mIoU with 50 hand-labeled diverse illustrated characters
#   - Backbone comparison proved data diversity is the bottleneck, not model capacity
#   - CVAT annotations at 10x weight — each diverse illustrated example is high-value
#   - Resume from run 5 seg-only checkpoint (0.3491 mIoU, CC0 baseline)
#
# Estimated: ~6 hrs on A100, ~$2
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_seg_cvat.sh
#   ./training/run_seg_cvat.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/seg_cvat_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Seg + CVAT Annotations"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "[pre] Pre-flight checks..."

if [ -z "${BUCKET_ACCESS_KEY:-}" ] || [ -z "${BUCKET_SECRET:-}" ]; then
    echo "  WARNING: BUCKET_ACCESS_KEY/BUCKET_SECRET not set."
    echo "  Upload step will fail. Set them if you want bucket upload."
fi

pip install -q scipy diffusers transformers accelerate 2>/dev/null || true
echo "  Dependencies OK."
echo ""

# ---------------------------------------------------------------------------
# 1. Download datasets
# ---------------------------------------------------------------------------
echo "[1/6] Downloading datasets..."
echo ""

# Helper: download + extract a tar, delete tar after
download_tar() {
    local name="$1"
    local ds_dir="./data_cloud/$name"

    if [ -d "$ds_dir" ] && [ "$(ls -A "$ds_dir" 2>/dev/null | head -1)" ]; then
        echo "  $name already exists."
        return 0
    fi

    mkdir -p ./data_cloud/tars
    local tar_file="./data_cloud/tars/${name}.tar"

    echo "  Downloading ${name}.tar..."
    rclone copy "hetzner:strata-training-data/tars/${name}.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P

    if [ -f "$tar_file" ]; then
        echo "  Extracting $name..."
        tar xf "$tar_file" -C ./data_cloud/
        rm -f "$tar_file"
        echo "  $name ready. (tar deleted)"
    else
        echo "  WARNING: Could not download $name."
        return 1
    fi
}

download_tar "humanrig"
download_tar "vroid_cc0"
download_tar "anime_seg"
download_tar "cvat_annotated"

# Meshy CC0 textured restructured
MESHY_DIR="./data_cloud/meshy_cc0_textured"
if [ -d "$MESHY_DIR" ] && [ "$(ls -A "$MESHY_DIR" 2>/dev/null | head -1)" ]; then
    echo "  meshy_cc0_textured already exists."
else
    mkdir -p ./data_cloud/tars
    echo "  Downloading meshy_cc0_textured_restructured.tar..."
    rclone copy "hetzner:strata-training-data/tars/meshy_cc0_textured_restructured.tar" \
        ./data_cloud/tars/ --transfers 32 --fast-list -P
    tar_file="./data_cloud/tars/meshy_cc0_textured_restructured.tar"
    if [ -f "$tar_file" ]; then
        echo "  Extracting..."
        tar xf "$tar_file" -C ./data_cloud/
        rm -f "$tar_file"
        for candidate in meshy_cc0_textured_restructured meshy_cc0_restructured; do
            if [ -d "./data_cloud/$candidate" ] && [ ! -d "$MESHY_DIR" ]; then
                mv "./data_cloud/$candidate" "$MESHY_DIR"
                echo "  Renamed $candidate → meshy_cc0_textured"
            fi
        done
    else
        echo "  WARNING: Could not download meshy_cc0_textured_restructured.tar"
    fi
fi

# Download resume checkpoint
echo "  Downloading run 5 seg checkpoint for resume..."
mkdir -p ./checkpoints/segmentation
rclone copy "hetzner:strata-training-data/checkpoints_run5_seg/best.pt" \
    ./checkpoints/segmentation/ --transfers 32 --fast-list -P
echo "  Checkpoint ready."

echo ""
echo "  Disk usage after downloads:"
du -sh ./data_cloud/* 2>/dev/null | head -20
df -h . | tail -1
echo ""

# Verify CVAT dataset exists
CVAT_DIR="./data_cloud/cvat_annotated"
if [ ! -d "$CVAT_DIR" ] || [ -z "$(ls -A "$CVAT_DIR" 2>/dev/null | head -1)" ]; then
    echo "  ERROR: cvat_annotated dataset not found!"
    echo "  Make sure tars/cvat_annotated.tar is in the bucket."
    exit 1
fi
CVAT_COUNT=$(find "$CVAT_DIR" -name "image.png" | wc -l)
echo "  CVAT annotated examples: $CVAT_COUNT"
echo ""

# ---------------------------------------------------------------------------
# 2. Quality filter + Marigold enrichment
# ---------------------------------------------------------------------------
echo "[2/6] Quality filter + Marigold normals..."
echo ""

for ds in humanrig vroid_cc0 meshy_cc0_textured anime_seg cvat_annotated; do
    ds_dir="./data_cloud/$ds"
    if [ -d "$ds_dir" ]; then
        rm -f "$ds_dir/quality_filter.json"
        echo "  Filtering $ds..."
        python scripts/filter_seg_quality.py \
            --data-dirs "$ds_dir" \
            --output-dir "$ds_dir" \
            --min-regions 3 \
            --max-single-region 0.70 \
            --min-foreground 0.05 \
            2>&1 | tee -a "$LOG_DIR/quality_filter.log"
    fi
done

# Enrich CVAT annotations with Marigold normals + depth
if [ -d "./data_cloud/cvat_annotated" ]; then
    echo "  Enriching cvat_annotated with Marigold normals..."
    python run_normals_enrich.py \
        --input-dir "./data_cloud/cvat_annotated" \
        --only-missing \
        --batch-size 16 \
        2>&1 | tee "$LOG_DIR/enrich_normals_cvat.log"
fi

echo "  Quality filter + enrichment complete."
echo ""

# ---------------------------------------------------------------------------
# 3. Train segmentation (MobileNetV3, resume from run 5)
# ---------------------------------------------------------------------------
echo "[3/6] Training SEGMENTATION with CVAT annotations..."
echo ""
echo "  Config: training/configs/segmentation_a100_cvat.yaml"
echo "  Resume: checkpoints/segmentation/best.pt (run 5, 0.3491 mIoU)"
echo ""

python -m training.train_segmentation \
    --config training/configs/segmentation_a100_cvat.yaml \
    --resume ./checkpoints/segmentation/best.pt \
    --reset-epochs \
    2>&1 | tee "$LOG_DIR/seg_cvat.log"

echo ""
echo "  Segmentation training complete."
echo ""

# ---------------------------------------------------------------------------
# 4. ONNX Export
# ---------------------------------------------------------------------------
echo "[4/6] Exporting segmentation to ONNX..."
echo ""

ONNX_DIR="./models/onnx_cvat"
mkdir -p "$ONNX_DIR"

if [ -f "./checkpoints/segmentation/best.pt" ]; then
    python -m training.export_onnx \
        --model segmentation \
        --checkpoint ./checkpoints/segmentation/best.pt \
        --output "$ONNX_DIR/segmentation.onnx" \
        2>&1 | tee "$LOG_DIR/export_seg.log"
    echo "  Exported segmentation.onnx"
fi

echo ""

# ---------------------------------------------------------------------------
# 5. Upload to bucket
# ---------------------------------------------------------------------------
echo "[5/6] Uploading checkpoints, logs, and ONNX..."
echo ""

rclone copy ./checkpoints/segmentation/ \
    hetzner:strata-training-data/checkpoints_cvat/segmentation/ \
    --transfers 32 --fast-list -P

rclone copy ./logs/ hetzner:strata-training-data/logs/ \
    --transfers 32 --fast-list -P

rclone copy "$ONNX_DIR/" hetzner:strata-training-data/models/onnx_cvat/ \
    --transfers 32 --fast-list -P

echo ""

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
echo "[6/6] Summary"
echo "============================================"
echo "  Seg + CVAT run complete!"
echo "  Finished: $(date)"
echo ""
echo "  Results:"
grep -E "Best mIoU|New best|miou" "$LOG_DIR/seg_cvat.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  To download results to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_cvat/ ./checkpoints_cvat/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/onnx_cvat/ ./models/onnx_cvat/ --transfers 32 --fast-list -P"
echo "============================================"

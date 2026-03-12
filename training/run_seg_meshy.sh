#!/usr/bin/env bash
# =============================================================================
# Strata Training — Seg+Meshy Short Run (A100)
#
# Goal: Add 15,281 meshy_cc0_textured examples to seg training
#   - Resume from seg-only checkpoint (CC0-only clean baseline, NOT run 1)
#   - meshy_cc0_textured: restructured from flat dirs to per-example subdirs
#   - gemini_diverse EXCLUDED (SAM2 pseudo-labels failed — only 13/698 usable)
#   - Datasets: humanrig + vroid_cc0 + meshy_cc0_textured + anime_seg
#
# Estimated: ~5-6 hrs on A100, ~$2
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_seg_meshy.sh
#   ./training/run_seg_meshy.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/seg_meshy_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Seg+Meshy Run"
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
# 0. Download checkpoints
# ---------------------------------------------------------------------------
echo "[0/5] Downloading checkpoints..."
echo ""

# Resume from seg-only run checkpoint (clean CC0-only baseline)
# NOT run 1 — run 1 used prohibited Mixamo/Live2D data
RESUME_CKPT="./checkpoints/segmentation/seg_only_best.pt"

echo "  Downloading seg-only run checkpoint..."
rclone copy hetzner:strata-training-data/checkpoints_run5_seg/segmentation/best.pt \
    ./checkpoints/segmentation/ --transfers 32 --fast-list -P

if [ -f "./checkpoints/segmentation/best.pt" ]; then
    cp ./checkpoints/segmentation/best.pt "$RESUME_CKPT"
    echo "  Resuming from seg-only run (CC0-only baseline)."
else
    echo "  FATAL: No seg-only checkpoint found in bucket."
    echo "  This run must resume from the seg-only checkpoint, not run 1 (prohibited data)."
    exit 1
fi
echo ""

# ---------------------------------------------------------------------------
# 1. Download datasets
# ---------------------------------------------------------------------------
echo "[1/5] Downloading datasets..."
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

# Download all datasets
download_tar "humanrig"
download_tar "vroid_cc0"
# Meshy CC0 textured restructured — may extract as various dir names
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
        # Rename to standard name regardless of what the tar contained
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
# gemini_diverse EXCLUDED — SAM2 pseudo-labels failed (13/698 passed quality filter)
# Joints model outputs all-center predictions on illustrated characters
download_tar "anime_seg"

# Show disk usage
echo ""
echo "  Disk usage after downloads:"
du -sh ./data_cloud/* 2>/dev/null | head -20
df -h . | tail -1
echo ""

# ---------------------------------------------------------------------------
# 2. Quality filter + Marigold enrichment
# ---------------------------------------------------------------------------
echo "[2/5] Quality filter + Marigold normals..."
echo ""

for ds in humanrig vroid_cc0 meshy_cc0_textured anime_seg; do
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

# Marigold normals on meshy (the new dataset)
for ds in meshy_cc0_textured vroid_cc0; do
    if [ -d "./data_cloud/$ds" ]; then
        echo "  Enriching $ds with Marigold normals..."
        python run_normals_enrich.py \
            --input-dir "./data_cloud/$ds" \
            --only-missing \
            --batch-size 16 \
            2>&1 | tee "$LOG_DIR/enrich_normals_${ds}.log"
    fi
done

echo "  Quality filter + enrichment complete."
echo ""

# ---------------------------------------------------------------------------
# 3. Train segmentation
# ---------------------------------------------------------------------------
echo "[3/5] Training SEGMENTATION model..."
echo ""
echo "  Resuming from: $RESUME_CKPT"
echo "  Config: training/configs/segmentation_a100_seg_meshy.yaml"
echo "  New data: meshy_cc0_textured (15,281 restructured examples)"
echo ""

python -m training.train_segmentation \
    --config training/configs/segmentation_a100_seg_meshy.yaml \
    --resume "$RESUME_CKPT" \
    --reset-epochs \
    2>&1 | tee "$LOG_DIR/segmentation.log"

echo ""
echo "  Segmentation training complete."
echo ""

# ---------------------------------------------------------------------------
# 4. ONNX Export
# ---------------------------------------------------------------------------
echo "[4/5] Exporting segmentation to ONNX..."
echo ""

ONNX_DIR="./models/onnx"
mkdir -p "$ONNX_DIR"

if [ -f "checkpoints/segmentation/best.pt" ]; then
    python -m training.export_onnx \
        --model segmentation \
        --checkpoint checkpoints/segmentation/best.pt \
        --output "$ONNX_DIR/segmentation.onnx" \
        2>&1 | tee "$LOG_DIR/export.log"
    echo "  Exported segmentation.onnx"
else
    echo "  WARNING: No seg checkpoint found for export."
fi
echo ""

# ---------------------------------------------------------------------------
# 5. Upload to bucket
# ---------------------------------------------------------------------------
echo "[5/5] Uploading checkpoints, logs, and ONNX..."
echo ""

rclone copy ./checkpoints/segmentation/ hetzner:strata-training-data/checkpoints_seg_meshy/segmentation/ \
    --transfers 32 --fast-list -P
rclone copy ./logs/ hetzner:strata-training-data/logs/ \
    --transfers 32 --fast-list -P
if [ -f "$ONNX_DIR/segmentation.onnx" ]; then
    rclone copy "$ONNX_DIR/segmentation.onnx" hetzner:strata-training-data/models/onnx_seg_meshy/ \
        --transfers 32 --fast-list -P
fi

echo ""
echo "============================================"
echo "  Seg+Meshy run complete!"
echo "  Finished: $(date)"
echo ""
echo "  Results:"
grep -E "Best mIoU|New best|miou" "$LOG_DIR/segmentation.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  To download results to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_seg_meshy/ ./checkpoints_seg_meshy/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/onnx_seg_meshy/ ./models/onnx_seg_meshy/ --transfers 32 --fast-list -P"
echo "============================================"

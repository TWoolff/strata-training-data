#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 7 Seg with Dr. Li's Expert Annotations (A100)
#
# Goal: Break past 0.37 mIoU plateau with diverse illustrated annotations
#   - gemini_li_converted: 694 expert-labeled illustrated chars (Dr. Li, weight 3.0)
#   - HumanRig T-pose: 11,434 GT 22-class
#   - VRoid CC0: 1,386 GT 22-class
#   - Meshy CC0 textured: 15,281 GT 22-class
#   - anime_seg: ~14K with existing masks
#   - Resume from run 5 seg-only checkpoint (0.3491 mIoU, CC0 baseline)
#   - Target: mIoU > 0.45
#
# Estimated: ~6-8 hrs on A100, ~$2-3
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_seg_li.sh
#   ./training/run_seg_li.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run7_seg_li_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 7 Seg (Dr. Li)"
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

# Download run 5 seg-only checkpoint (0.3491 mIoU — CC0 baseline)
RUN5_CKPT="checkpoints/segmentation/run5_best.pt"
if [ -f "$RUN5_CKPT" ]; then
    echo "  run5_best.pt already exists."
else
    echo "  Downloading run 5 seg-only checkpoint..."
    rclone copy hetzner:strata-training-data/checkpoints_run5_seg/best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
    if [ -f "checkpoints/segmentation/best.pt" ]; then
        cp checkpoints/segmentation/best.pt "$RUN5_CKPT"
        echo "  Saved as $RUN5_CKPT"
    else
        echo "  WARNING: No run 5 checkpoint found. Will train from scratch."
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# 1. Download datasets
# ---------------------------------------------------------------------------
echo "[1/5] Downloading datasets..."
echo ""

# gemini_li_converted — Dr. Li's expert annotations (NEW)
LI_DIR="./data_cloud/gemini_li_converted"
LI_TAR="./data_cloud/tars/gemini_li_converted.tar"
if [ -d "$LI_DIR" ] && [ "$(ls -A "$LI_DIR" 2>/dev/null | head -1)" ]; then
    echo "  gemini_li_converted already exists."
else
    mkdir -p ./data_cloud/tars
    echo "  Downloading gemini_li_converted tar..."
    rclone copy "hetzner:strata-training-data/tars/gemini_li_converted.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "$LI_TAR" ]; then
        echo "  Extracting gemini_li_converted..."
        tar xf "$LI_TAR" -C ./data_cloud/
        rm -f "$LI_TAR"
    else
        echo "  FATAL: gemini_li_converted tar not found."
        exit 1
    fi
fi

# Standard datasets (humanrig, vroid_cc0, anime_seg)
for ds in humanrig vroid_cc0 anime_seg; do
    ds_dir="./data_cloud/$ds"
    tar_file="./data_cloud/tars/${ds}.tar"
    if [ -d "$ds_dir" ] && [ "$(ls -A "$ds_dir" 2>/dev/null | head -1)" ]; then
        echo "  $ds already exists."
    elif [ -f "$tar_file" ]; then
        echo "  Extracting $ds from tar..."
        tar xf "$tar_file" -C ./data_cloud/
        rm -f "$tar_file"
    else
        echo "  Downloading $ds tar..."
        mkdir -p ./data_cloud/tars
        rclone copy "hetzner:strata-training-data/tars/${ds}.tar" ./data_cloud/tars/ \
            --transfers 32 --fast-list -P
        if [ -f "$tar_file" ]; then
            tar xf "$tar_file" -C ./data_cloud/
            rm -f "$tar_file"
        else
            echo "  WARNING: Could not download $ds."
        fi
    fi
done

# meshy_cc0_textured — use restructured tar (per-example subdirs with seg masks)
MESHY_RESTR_TAR="./data_cloud/tars/meshy_cc0_textured_restructured.tar"
MESHY_DIR="./data_cloud/meshy_cc0_textured"
if [ -d "$MESHY_DIR" ] && [ "$(find "$MESHY_DIR" -maxdepth 2 -name 'segmentation.png' | head -1)" ]; then
    echo "  meshy_cc0_textured already exists (restructured)."
else
    echo "  Downloading meshy_cc0_textured_restructured tar..."
    mkdir -p ./data_cloud/tars
    rclone copy "hetzner:strata-training-data/tars/meshy_cc0_textured_restructured.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "$MESHY_RESTR_TAR" ]; then
        echo "  Extracting meshy_cc0_textured_restructured..."
        rm -rf "$MESHY_DIR"  # Remove any flat-structure version
        tar xf "$MESHY_RESTR_TAR" -C ./data_cloud/
        rm -f "$MESHY_RESTR_TAR"
        # Restructured tar extracts to meshy_cc0_restructured — rename
        if [ -d "./data_cloud/meshy_cc0_restructured" ]; then
            mv ./data_cloud/meshy_cc0_restructured "$MESHY_DIR"
        fi
    else
        echo "  WARNING: Could not download meshy_cc0_textured_restructured."
    fi
fi

echo ""

# ---------------------------------------------------------------------------
# 2. Marigold enrichment (depth on gemini_li_converted — normals already done)
# ---------------------------------------------------------------------------
echo "[2/5] Marigold depth enrichment on gemini_li_converted..."
echo ""

# Count missing depth maps
MISSING_DEPTH=$(find "$LI_DIR" -name "image.png" -exec sh -c '
    dir=$(dirname "$1"); [ ! -f "$dir/depth.png" ] && echo missing
' _ {} \; | wc -l | tr -d ' ')

if [ "$MISSING_DEPTH" -gt 0 ]; then
    echo "  $MISSING_DEPTH examples missing depth.png — running Marigold..."
    python run_normals_enrich.py \
        --input-dir "$LI_DIR" \
        --only-missing \
        --batch-size 16 \
        --no-normals \
        2>&1 | tee "$LOG_DIR/enrich_depth_li.log"
else
    echo "  All gemini_li_converted examples have depth.png."
fi

# Also check normals (should be 694/694 but verify)
MISSING_NORMALS=$(find "$LI_DIR" -name "image.png" -exec sh -c '
    dir=$(dirname "$1"); [ ! -f "$dir/normals.png" ] && echo missing
' _ {} \; | wc -l | tr -d ' ')

if [ "$MISSING_NORMALS" -gt 0 ]; then
    echo "  $MISSING_NORMALS examples missing normals.png — running Marigold..."
    python run_normals_enrich.py \
        --input-dir "$LI_DIR" \
        --only-missing \
        --batch-size 16 \
        2>&1 | tee "$LOG_DIR/enrich_normals_li.log"
else
    echo "  All gemini_li_converted examples have normals.png."
fi

echo ""

# ---------------------------------------------------------------------------
# 3. Quality filter
# ---------------------------------------------------------------------------
echo "[3/5] Quality filter..."
echo ""

for ds in humanrig vroid_cc0 meshy_cc0_textured anime_seg gemini_li_converted; do
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
# 4. Train segmentation
# ---------------------------------------------------------------------------
echo "[4/5] Training SEGMENTATION model..."
echo ""

RESUME_CKPT="$RUN5_CKPT"
if [ ! -f "$RESUME_CKPT" ]; then
    RESUME_CKPT=""
    echo "  No resume checkpoint — training from scratch (pretrained backbone)."
fi

if [ -n "$RESUME_CKPT" ]; then
    echo "  Resuming from: $RESUME_CKPT (run 5, 0.3491 mIoU)"
fi
echo "  Config: training/configs/segmentation_a100_run7_li.yaml"
echo "  New data: gemini_li_converted (694 expert-labeled, weight 3.0)"
echo ""

TRAIN_CMD="python -m training.train_segmentation \
    --config training/configs/segmentation_a100_run7_li.yaml \
    --reset-epochs"

if [ -n "$RESUME_CKPT" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME_CKPT"
fi

eval "$TRAIN_CMD" 2>&1 | tee "$LOG_DIR/segmentation.log"

echo ""
echo "  Segmentation training complete."
echo ""

# ---------------------------------------------------------------------------
# 5. ONNX Export + Upload
# ---------------------------------------------------------------------------
echo "[5/5] Exporting + uploading..."
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
echo "  Uploading checkpoints, logs, ONNX..."

rclone copy ./checkpoints/segmentation/ hetzner:strata-training-data/checkpoints_run7_li/segmentation/ \
    --transfers 32 --fast-list -P
rclone copy ./logs/ hetzner:strata-training-data/logs/ \
    --transfers 32 --fast-list -P
if [ -f "$ONNX_DIR/segmentation.onnx" ]; then
    rclone copy "$ONNX_DIR/segmentation.onnx" hetzner:strata-training-data/models/onnx_run7_li/ \
        --transfers 32 --fast-list -P
fi

echo ""
echo "============================================"
echo "  Run 7 Seg (Dr. Li) complete!"
echo "  Finished: $(date)"
echo ""
echo "  Results:"
grep -E "Best mIoU|New best|miou" "$LOG_DIR/segmentation.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  To download results to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_run7_li/ ./checkpoints_run7_li/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/onnx_run7_li/ ./models/onnx_run7_li/ --transfers 32 --fast-list -P"
echo "============================================"

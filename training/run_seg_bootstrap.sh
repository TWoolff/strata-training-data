#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 9 Seg Bootstrapped Pseudo-Labels (A100)
#
# Goal: Break past 0.36 mIoU with bootstrapped diverse illustrated data
#   - NEW: gemini_diverse: 874 pseudo-labeled + auto-triaged illustrated chars (weight 4.0)
#   - DROPPED: anime_seg (~14K, 32% rejection rate, noisy labels)
#   - cvat_annotated: 49 hand-annotated illustrated chars (weight 10.0)
#   - gemini_li_converted: 694 expert-labeled illustrated chars (Dr. Li, weight 3.0)
#   - HumanRig T-pose: 11,434 GT 22-class
#   - VRoid CC0: 1,386 GT 22-class
#   - Meshy CC0 textured: 15,281 GT 22-class
#   - Resume from run 8 checkpoint (0.4721 mIoU)
#   - Target: mIoU > 0.52
#
# Estimated: ~5-7 hrs on A100, ~$2-3
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_seg_bootstrap.sh
#   ./training/run_seg_bootstrap.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run9_seg_bootstrap_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 9 Seg (Bootstrap)"
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

# Download run 8 checkpoint (0.4721 mIoU)
RUN8_CKPT="checkpoints/segmentation/run8_best.pt"
if [ -f "$RUN8_CKPT" ]; then
    echo "  run8_best.pt already exists."
else
    echo "  Downloading run 8 checkpoint..."
    rclone copy hetzner:strata-training-data/checkpoints_run8_bootstrap/segmentation/best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
    if [ -f "checkpoints/segmentation/best.pt" ]; then
        cp checkpoints/segmentation/best.pt "$RUN8_CKPT"
        echo "  Saved as $RUN8_CKPT"
    else
        echo "  WARNING: No run 8 checkpoint found. Will train from scratch."
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# 1. Download datasets
# ---------------------------------------------------------------------------
echo "[1/5] Downloading datasets..."
echo ""

# gemini_diverse — 874 pseudo-labeled + auto-triaged illustrated chars (NEW)
GD_DIR="./data_cloud/gemini_diverse"
GD_TAR="./data_cloud/tars/gemini_diverse.tar"
if [ -d "$GD_DIR" ] && [ "$(ls -A "$GD_DIR" 2>/dev/null | head -1)" ]; then
    echo "  gemini_diverse already exists."
else
    mkdir -p ./data_cloud/tars
    echo "  Downloading gemini_diverse tar..."
    rclone copy "hetzner:strata-training-data/tars/gemini_diverse.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "$GD_TAR" ]; then
        echo "  Extracting gemini_diverse..."
        tar xf "$GD_TAR" -C ./data_cloud/
        rm -f "$GD_TAR"
    else
        echo "  FATAL: gemini_diverse tar not found."
        exit 1
    fi
fi

# gemini_li_converted — Dr. Li's expert annotations
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

# CVAT hand-annotated (49 diverse illustrated chars)
CVAT_DIR="./data_cloud/cvat_annotated"
CVAT_TAR="./data_cloud/tars/cvat_annotated.tar"
if [ -d "$CVAT_DIR" ] && [ "$(ls -A "$CVAT_DIR" 2>/dev/null | head -1)" ]; then
    echo "  cvat_annotated already exists."
else
    mkdir -p ./data_cloud/tars
    echo "  Downloading cvat_annotated tar..."
    rclone copy "hetzner:strata-training-data/tars/cvat_annotated.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "$CVAT_TAR" ]; then
        echo "  Extracting cvat_annotated..."
        tar xf "$CVAT_TAR" -C ./data_cloud/
        rm -f "$CVAT_TAR"
    else
        echo "  WARNING: cvat_annotated tar not found."
    fi
fi

# Standard datasets (humanrig, vroid_cc0) — NO anime_seg
for ds in humanrig vroid_cc0; do
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
# 2. Marigold enrichment (depth + normals on gemini_diverse)
# ---------------------------------------------------------------------------
echo "[2/5] Marigold enrichment on gemini_diverse..."
echo ""

# Count missing depth maps on gemini_diverse
MISSING_DEPTH=$(find "$GD_DIR" -name "image.png" -exec sh -c '
    dir=$(dirname "$1"); [ ! -f "$dir/depth.png" ] && echo missing
' _ {} \; | wc -l | tr -d ' ')

if [ "$MISSING_DEPTH" -gt 0 ]; then
    echo "  $MISSING_DEPTH gemini_diverse examples missing depth.png — running Marigold..."
    python run_normals_enrich.py \
        --input-dir "$GD_DIR" \
        --only-missing \
        --batch-size 16 \
        --no-normals \
        2>&1 | tee "$LOG_DIR/enrich_depth_gd.log"
else
    echo "  All gemini_diverse examples have depth.png."
fi

# Check normals on gemini_diverse
MISSING_NORMALS=$(find "$GD_DIR" -name "image.png" -exec sh -c '
    dir=$(dirname "$1"); [ ! -f "$dir/normals.png" ] && echo missing
' _ {} \; | wc -l | tr -d ' ')

if [ "$MISSING_NORMALS" -gt 0 ]; then
    echo "  $MISSING_NORMALS gemini_diverse examples missing normals.png — running Marigold..."
    python run_normals_enrich.py \
        --input-dir "$GD_DIR" \
        --only-missing \
        --batch-size 16 \
        2>&1 | tee "$LOG_DIR/enrich_normals_gd.log"
else
    echo "  All gemini_diverse examples have normals.png."
fi

# Also check gemini_li_converted enrichment
MISSING_DEPTH_LI=$(find "$LI_DIR" -name "image.png" -exec sh -c '
    dir=$(dirname "$1"); [ ! -f "$dir/depth.png" ] && echo missing
' _ {} \; | wc -l | tr -d ' ')

if [ "$MISSING_DEPTH_LI" -gt 0 ]; then
    echo "  $MISSING_DEPTH_LI gemini_li_converted examples missing depth.png — running Marigold..."
    python run_normals_enrich.py \
        --input-dir "$LI_DIR" \
        --only-missing \
        --batch-size 16 \
        --no-normals \
        2>&1 | tee "$LOG_DIR/enrich_depth_li.log"
else
    echo "  All gemini_li_converted examples have depth.png."
fi

echo ""

# ---------------------------------------------------------------------------
# 3. Quality filter
# ---------------------------------------------------------------------------
echo "[3/5] Quality filter..."
echo ""

for ds in humanrig vroid_cc0 meshy_cc0_textured gemini_li_converted cvat_annotated gemini_diverse; do
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

RESUME_CKPT="$RUN8_CKPT"
if [ ! -f "$RESUME_CKPT" ]; then
    RESUME_CKPT=""
    echo "  No resume checkpoint — training from scratch (pretrained backbone)."
fi

if [ -n "$RESUME_CKPT" ]; then
    echo "  Resuming from: $RESUME_CKPT (run 8, 0.4721 mIoU)"
fi
echo "  Config: training/configs/segmentation_a100_run8_bootstrap.yaml"
echo "  New data: gemini_diverse (874, wt 4.0). Dropped: anime_seg."
echo ""

TRAIN_CMD="python -m training.train_segmentation \
    --config training/configs/segmentation_a100_run8_bootstrap.yaml \
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

rclone copy ./checkpoints/segmentation/ hetzner:strata-training-data/checkpoints_run9_bootstrap/segmentation/ \
    --transfers 32 --fast-list -P
rclone copy ./logs/ hetzner:strata-training-data/logs/ \
    --transfers 32 --fast-list -P
if [ -f "$ONNX_DIR/segmentation.onnx" ]; then
    rclone copy "$ONNX_DIR/segmentation.onnx" hetzner:strata-training-data/models/onnx_run9_bootstrap/ \
        --transfers 32 --fast-list -P
fi

echo ""
echo "============================================"
echo "  Run 9 Seg (Bootstrap) complete!"
echo "  Finished: $(date)"
echo ""
echo "  Results:"
grep -E "Best mIoU|New best|miou" "$LOG_DIR/segmentation.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  To download results to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_run9_bootstrap/ ./checkpoints_run9_bootstrap/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/onnx_run9_bootstrap/ ./models/onnx_run9_bootstrap/ --transfers 32 --fast-list -P"
echo "============================================"

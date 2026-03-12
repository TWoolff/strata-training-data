#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 4.5 (A100)
#
# Goal: Fix segmentation with SAM2 pseudo-labels + per-dataset loss weighting
#   - SAM2 pseudo-label anime_seg + gemini_diverse (sharp boundaries + joint assignment)
#   - Per-dataset loss weighting (upweight ground-truth, downweight noisy data)
#   - Fine-tune seg from run 4 checkpoint (0.4389 mIoU)
#   - Target: mIoU > 0.55 (recover and surpass run 1)
#
# This is a focused seg-only run (~3-4 hrs on A100).
# Joints, weights, inpainting are NOT retrained.
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_four_five.sh
#   ./training/run_four_five.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run4.5_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 4.5 (Seg Fix)"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "[pre] Pre-flight checks..."
echo ""

# Check for required env vars
if [ -z "${BUCKET_ACCESS_KEY:-}" ] || [ -z "${BUCKET_SECRET:-}" ]; then
    echo "  WARNING: BUCKET_ACCESS_KEY/BUCKET_SECRET not set."
    echo "  Upload step will fail. Set them if you want bucket upload."
fi

# Check for pip packages
pip install -q scipy diffusers transformers accelerate 2>/dev/null || true
# SAM2 must be installed from source (pip package doesn't provide 'sam2' module)
if ! python -c "import sam2" 2>/dev/null; then
    echo "  Installing SAM2 from source..."
    pip install -q git+https://github.com/facebookresearch/sam2.git
fi
echo "  Dependencies checked."
echo ""

# ---------------------------------------------------------------------------
# 0. Download checkpoints + SAM2 model
# ---------------------------------------------------------------------------
echo "[0/7] Downloading checkpoints + SAM2..."
echo ""

# Download run 4 seg checkpoint
RUN4_CKPT="checkpoints/segmentation/run4_best.pt"
if [ -f "$RUN4_CKPT" ]; then
    echo "  run4_best.pt already exists."
else
    echo "  Downloading run 4 seg checkpoint from bucket..."
    rclone copy hetzner:strata-training-data/checkpoints_run4/segmentation/best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
    if [ -f "checkpoints/segmentation/best.pt" ]; then
        cp checkpoints/segmentation/best.pt "$RUN4_CKPT"
        echo "  Saved as $RUN4_CKPT"
    else
        echo "  FATAL: No run 4 seg checkpoint found in bucket."
        echo "  Trying run 1 checkpoint as fallback..."
        rclone copy hetzner:strata-training-data/checkpoints_run1/segmentation/best.pt \
            ./checkpoints/segmentation/ --transfers 32 --fast-list -P
        if [ -f "checkpoints/segmentation/best.pt" ]; then
            cp checkpoints/segmentation/best.pt "$RUN4_CKPT"
            echo "  Using run 1 checkpoint as fallback."
        else
            echo "  FATAL: No segmentation checkpoint found. Cannot proceed."
            exit 1
        fi
    fi
fi

# Download SAM2 checkpoint
SAM2_CKPT="./models/sam2.1_hiera_large.pt"
SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_l.yaml"
if [ -f "$SAM2_CKPT" ]; then
    echo "  SAM2 checkpoint already exists."
else
    echo "  Downloading SAM2 checkpoint..."
    mkdir -p ./models
    # SAM2.1 large checkpoint from Meta
    wget -q -O "$SAM2_CKPT" \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
        || { echo "  FATAL: Failed to download SAM2 checkpoint."; exit 1; }
    echo "  SAM2 checkpoint downloaded."
fi
echo ""

# ---------------------------------------------------------------------------
# 1. Download datasets
# ---------------------------------------------------------------------------
echo "[1/7] Downloading datasets..."
echo ""

for ds in anime_seg; do
    ds_dir="./data_cloud/$ds"
    if [ -d "$ds_dir" ] && [ "$(ls -A "$ds_dir" 2>/dev/null | head -1)" ]; then
        echo "  $ds already exists, skipping."
    else
        echo "  Downloading $ds..."
        rclone copy "hetzner:strata-training-data/$ds/" "$ds_dir/" \
            --transfers 32 --checkers 64 --fast-list --size-only -P
    fi
done

# Download gemini_diverse from tar (all ~700 examples)
GEMINI_DIR="./data_cloud/gemini_diverse"
GEMINI_TAR="./data_cloud/tars/gemini_diverse.tar"
if [ -d "$GEMINI_DIR" ] && [ "$(ls -A "$GEMINI_DIR" 2>/dev/null | head -1)" ]; then
    echo "  gemini_diverse already exists."
else
    mkdir -p ./data_cloud/tars
    echo "  Downloading gemini_diverse tar..."
    rclone copy "hetzner:strata-training-data/tars/gemini_diverse.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "$GEMINI_TAR" ]; then
        echo "  Extracting gemini_diverse..."
        tar xf "$GEMINI_TAR" -C ./data_cloud/
    else
        echo "  No tar found, downloading loose files..."
        rclone copy "hetzner:strata-training-data/gemini_diverse/" "$GEMINI_DIR/" \
            --transfers 32 --checkers 64 --fast-list --size-only -P
    fi
fi

# Untar other datasets if not already present (fbanimehq EXCLUDED — binary only)
for ds in humanrig meshy_cc0 meshy_cc0_textured; do
    ds_dir="./data_cloud/$ds"
    tar_file="./data_cloud/tars/${ds}.tar"
    if [ -d "$ds_dir" ] && [ "$(ls -A "$ds_dir" 2>/dev/null | head -1)" ]; then
        echo "  $ds already exists."
    elif [ -f "$tar_file" ]; then
        echo "  Extracting $ds from tar..."
        tar xf "$tar_file" -C ./data_cloud/
    else
        echo "  $ds not found locally, downloading tar..."
        mkdir -p ./data_cloud/tars
        rclone copy "hetzner:strata-training-data/tars/${ds}.tar" ./data_cloud/tars/ \
            --transfers 32 --fast-list -P
        if [ -f "$tar_file" ]; then
            tar xf "$tar_file" -C ./data_cloud/
        else
            echo "  WARNING: Could not download $ds tar."
        fi
    fi
done
echo ""

# ---------------------------------------------------------------------------
# 2. SAM2 pseudo-labeling on anime_seg + gemini_diverse
# ---------------------------------------------------------------------------
echo "[2/7] Running SAM2 pseudo-labeling..."
echo ""

for ds in gemini_diverse anime_seg; do
    ds_dir="./data_cloud/$ds"
    stats_file="$ds_dir/sam2_pseudolabel_stats.json"
    if [ -f "$stats_file" ]; then
        echo "  $ds already has SAM2 labels, skipping."
        continue
    fi
    if [ ! -d "$ds_dir" ]; then
        echo "  $ds not found, skipping."
        continue
    fi

    echo "  Labeling $ds with SAM2..."
    python scripts/run_sam2_pseudolabel.py \
        --input-dir "$ds_dir" \
        --sam2-checkpoint "$SAM2_CKPT" \
        --sam2-config "$SAM2_CONFIG" \
        --device cuda \
        --points-per-side 32 \
        2>&1 | tee "$LOG_DIR/sam2_${ds}.log"
    echo ""
done

echo "  SAM2 pseudo-labeling complete."
echo ""

# ---------------------------------------------------------------------------
# 3. Quality filter (re-run with new SAM2 masks)
# ---------------------------------------------------------------------------
echo "[3/7] Running quality filter on all seg masks..."
echo ""

for ds in humanrig meshy_cc0 meshy_cc0_textured gemini_diverse anime_seg; do
    ds_dir="./data_cloud/$ds"
    if [ -d "$ds_dir" ]; then
        # Remove old filter to re-evaluate with SAM2 masks
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

echo ""
echo "  Quality filter complete."
echo ""

# ---------------------------------------------------------------------------
# 4. Marigold normals/depth enrichment on new data
# ---------------------------------------------------------------------------
echo "[4/7] Enriching datasets with Marigold normals + depth..."
echo ""

for ds in gemini_diverse; do
    if [ -d "./data_cloud/$ds" ]; then
        echo "  Enriching $ds..."
        python run_normals_enrich.py \
            --input-dir "./data_cloud/$ds" \
            --only-missing \
            --batch-size 16 \
            2>&1 | tee "$LOG_DIR/enrich_normals_${ds}.log"
        echo ""
    fi
done

echo "  Enrichment complete."
echo ""

# ---------------------------------------------------------------------------
# 5. Train segmentation (with per-dataset loss weighting)
# ---------------------------------------------------------------------------
echo "[5/7] Training SEGMENTATION model..."
echo ""
echo "  Resuming from: $RUN4_CKPT"
echo "  Config: training/configs/segmentation_a100_run4_5.yaml"
echo "  New: per-dataset loss weighting + SAM2 pseudo-labels"
echo ""

python -m training.train_segmentation \
    --config training/configs/segmentation_a100_run4_5.yaml \
    --resume "$RUN4_CKPT" \
    2>&1 | tee "$LOG_DIR/segmentation.log"

echo ""
echo "  Segmentation training complete."
echo ""

# ---------------------------------------------------------------------------
# 6. ONNX Export (seg only — other models keep run 4 checkpoints)
# ---------------------------------------------------------------------------
echo "[6/7] Exporting segmentation to ONNX..."
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
# 7. Upload to bucket
# ---------------------------------------------------------------------------
echo "[7/7] Uploading checkpoints, logs, and ONNX..."
echo ""

rclone copy ./checkpoints/segmentation/ hetzner:strata-training-data/checkpoints_run4.5/segmentation/ \
    --transfers 32 --fast-list -P
rclone copy ./logs/ hetzner:strata-training-data/logs/ \
    --transfers 32 --fast-list -P
if [ -f "$ONNX_DIR/segmentation.onnx" ]; then
    rclone copy "$ONNX_DIR/segmentation.onnx" hetzner:strata-training-data/models/onnx_run4.5/ \
        --transfers 32 --fast-list -P
fi

# Upload SAM2 pseudo-labeled datasets back to bucket
for ds in gemini_diverse anime_seg; do
    ds_dir="./data_cloud/$ds"
    if [ -f "$ds_dir/sam2_pseudolabel_stats.json" ]; then
        echo "  Uploading SAM2-labeled $ds..."
        rclone copy "$ds_dir/" "hetzner:strata-training-data/${ds}/" \
            --transfers 32 --checkers 64 --fast-list --size-only -P
    fi
done

echo ""
echo "============================================"
echo "  Run 4.5 complete!"
echo "  Finished: $(date)"
echo ""
echo "  Results:"
grep -E "Best mIoU|New best" "$LOG_DIR/segmentation.log" 2>/dev/null | tail -3 || echo "  (check logs)"
echo ""
echo "  To download results to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_run4.5/ ./checkpoints_run4.5/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/onnx_run4.5/ ./models/onnx_run4.5/ --transfers 32 --fast-list -P"
echo "============================================"

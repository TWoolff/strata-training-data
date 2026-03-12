#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 5 Seg-Only (A100)
#
# Goal: Fix segmentation with VRoid CC0 GT data + joint-conditioned SAM2
#   - VRoid CC0: 1,386 GT 22-class examples (11 VRoid Hub characters, posed)
#   - Gemini diverse: 698 examples, SAM2 pseudo-labeled with joints
#   - HumanRig T-pose: 11,434 GT 22-class (already in bucket)
#   - anime_seg: ~14K with existing masks
#   - Resume from run 1 checkpoint (0.545 mIoU — best seg so far)
#   - Target: mIoU > 0.55
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
#   chmod +x training/run_seg_only.sh
#   ./training/run_seg_only.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run5_seg_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 5 Seg-Only"
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

# SAM2 for Gemini pseudo-labeling
if ! python -c "import sam2" 2>/dev/null; then
    echo "  Installing SAM2 from source..."
    pip install -q git+https://github.com/facebookresearch/sam2.git
fi
echo "  Dependencies OK."
echo ""

# ---------------------------------------------------------------------------
# 0. Download checkpoints + models
# ---------------------------------------------------------------------------
echo "[0/7] Downloading checkpoints..."
echo ""

# Download run 1 seg checkpoint (0.545 mIoU — best seg result)
RUN1_CKPT="checkpoints/segmentation/run1_best.pt"
if [ -f "$RUN1_CKPT" ]; then
    echo "  run1_best.pt already exists."
else
    echo "  Downloading run 1 seg checkpoint..."
    rclone copy hetzner:strata-training-data/checkpoints_run1/segmentation/best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
    if [ -f "checkpoints/segmentation/best.pt" ]; then
        cp checkpoints/segmentation/best.pt "$RUN1_CKPT"
        echo "  Saved as $RUN1_CKPT"
    else
        echo "  FATAL: No run 1 seg checkpoint found."
        exit 1
    fi
fi

# Download run 3 joints checkpoint (for Gemini joints inference)
JOINTS_CKPT="checkpoints/joint_refinement/run3_best.pt"
if [ -f "$JOINTS_CKPT" ]; then
    echo "  Joints checkpoint already exists."
else
    echo "  Downloading run 3 joints checkpoint..."
    rclone copy hetzner:strata-training-data/checkpoints_run3/joints/best.pt \
        ./checkpoints/joint_refinement/ --transfers 32 --fast-list -P
    if [ -f "checkpoints/joint_refinement/best.pt" ]; then
        cp checkpoints/joint_refinement/best.pt "$JOINTS_CKPT"
        echo "  Saved as $JOINTS_CKPT"
    else
        echo "  WARNING: No joints checkpoint. SAM2 will use spatial fallback."
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
    wget -q -O "$SAM2_CKPT" \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
        || { echo "  FATAL: Failed to download SAM2 checkpoint."; exit 1; }
    echo "  SAM2 checkpoint downloaded."
fi
echo ""

# ---------------------------------------------------------------------------
# 1. Download datasets (only what we need for seg)
# ---------------------------------------------------------------------------
echo "[1/7] Downloading datasets..."
echo ""

# VRoid CC0 — new GT data
VROID_DIR="./data_cloud/vroid_cc0"
VROID_TAR="./data_cloud/tars/vroid_cc0.tar"
if [ -d "$VROID_DIR" ] && [ "$(ls -A "$VROID_DIR" 2>/dev/null | head -1)" ]; then
    echo "  vroid_cc0 already exists."
else
    mkdir -p ./data_cloud/tars
    echo "  Downloading vroid_cc0 tar..."
    rclone copy "hetzner:strata-training-data/tars/vroid_cc0.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "$VROID_TAR" ]; then
        echo "  Extracting vroid_cc0..."
        tar xf "$VROID_TAR" -C ./data_cloud/
        rm -f "$VROID_TAR"
    else
        echo "  No tar found, downloading loose files..."
        rclone copy "hetzner:strata-training-data/vroid_cc0/" "$VROID_DIR/" \
            --transfers 32 --checkers 64 --fast-list --size-only -P
    fi
fi

# HumanRig — GT 22-class
for ds in humanrig anime_seg; do
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

# Gemini diverse
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
        rm -f "$GEMINI_TAR"
    else
        echo "  No tar found, downloading loose files..."
        rclone copy "hetzner:strata-training-data/gemini_diverse/" "$GEMINI_DIR/" \
            --transfers 32 --checkers 64 --fast-list --size-only -P
    fi
fi

echo ""

# ---------------------------------------------------------------------------
# 2. Joints inference on Gemini (needed for SAM2 joint-conditioned mode)
# ---------------------------------------------------------------------------
echo "[2/7] Running joints inference on gemini_diverse..."
echo ""

if [ -f "$JOINTS_CKPT" ]; then
    # Count examples missing joints.json
    MISSING=$(find "$GEMINI_DIR" -name "image.png" -exec sh -c '
        dir=$(dirname "$1"); [ ! -f "$dir/joints.json" ] && echo missing
    ' _ {} \; | wc -l | tr -d ' ')

    if [ "$MISSING" -gt 0 ]; then
        echo "  $MISSING examples missing joints.json — running inference..."
        PYTHONPATH="$(pwd)" python scripts/run_joints_inference.py \
            --input-dir "$GEMINI_DIR" \
            --checkpoint "$JOINTS_CKPT" \
            --device cuda \
            2>&1 | tee "$LOG_DIR/joints_inference.log"
    else
        echo "  All Gemini examples have joints.json."
    fi
else
    echo "  No joints checkpoint — SAM2 will use spatial fallback."
fi
echo ""

# ---------------------------------------------------------------------------
# 3. SAM2 pseudo-labeling on Gemini (joint-conditioned)
# ---------------------------------------------------------------------------
echo "[3/7] Running SAM2 pseudo-labeling on gemini_diverse..."
echo ""

STATS_FILE="$GEMINI_DIR/sam2_pseudolabel_stats.json"
if [ -f "$STATS_FILE" ]; then
    echo "  gemini_diverse already has SAM2 labels, skipping."
else
    python scripts/run_sam2_pseudolabel.py \
        --input-dir "$GEMINI_DIR" \
        --sam2-checkpoint "$SAM2_CKPT" \
        --sam2-config "$SAM2_CONFIG" \
        --device cuda \
        --points-per-side 32 \
        2>&1 | tee "$LOG_DIR/sam2_gemini.log"
fi
echo ""

# ---------------------------------------------------------------------------
# 4. Quality filter + Marigold enrichment
# ---------------------------------------------------------------------------
echo "[4/7] Quality filter + Marigold normals..."
echo ""

for ds in humanrig vroid_cc0 gemini_diverse anime_seg; do
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

# Marigold normals on vroid_cc0 + gemini_diverse (small datasets, ~15 min)
for ds in vroid_cc0 gemini_diverse; do
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
# 5. Train segmentation
# ---------------------------------------------------------------------------
echo "[5/7] Training SEGMENTATION model..."
echo ""
echo "  Resuming from: $RUN1_CKPT (run 1, 0.545 mIoU)"
echo "  Config: training/configs/segmentation_a100_run5_seg.yaml"
echo "  New data: vroid_cc0 (1,386 GT) + expanded gemini (698)"
echo ""

python -m training.train_segmentation \
    --config training/configs/segmentation_a100_run5_seg.yaml \
    --resume "$RUN1_CKPT" \
    --reset-epochs \
    2>&1 | tee "$LOG_DIR/segmentation.log"

echo ""
echo "  Segmentation training complete."
echo ""

# ---------------------------------------------------------------------------
# 6. ONNX Export (seg only)
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
echo "[7/7] Uploading checkpoints, logs, ONNX, and enriched data..."
echo ""

rclone copy ./checkpoints/segmentation/ hetzner:strata-training-data/checkpoints_run5_seg/segmentation/ \
    --transfers 32 --fast-list -P
rclone copy ./logs/ hetzner:strata-training-data/logs/ \
    --transfers 32 --fast-list -P
if [ -f "$ONNX_DIR/segmentation.onnx" ]; then
    rclone copy "$ONNX_DIR/segmentation.onnx" hetzner:strata-training-data/models/onnx_run5_seg/ \
        --transfers 32 --fast-list -P
fi

# Upload enriched Gemini data (SAM2 labels + joints + normals)
if [ -d "$GEMINI_DIR" ]; then
    echo "  Uploading enriched gemini_diverse..."
    rclone copy "$GEMINI_DIR/" "hetzner:strata-training-data/gemini_diverse/" \
        --transfers 32 --checkers 64 --fast-list --size-only -P
fi

echo ""
echo "============================================"
echo "  Run 5 Seg-Only complete!"
echo "  Finished: $(date)"
echo ""
echo "  Results:"
grep -E "Best mIoU|New best|miou" "$LOG_DIR/segmentation.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  To download results to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_run5_seg/ ./checkpoints_run5_seg/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/onnx_run5_seg/ ./models/onnx_run5_seg/ --transfers 32 --fast-list -P"
echo "============================================"

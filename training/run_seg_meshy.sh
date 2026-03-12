#!/usr/bin/env bash
# =============================================================================
# Strata Training — Seg+Meshy Overnight Run (A100)
#
# Goal: Train seg with +15,281 meshy_cc0_textured, then resume inpainting
#   - Resume seg from seg-only checkpoint (CC0-only clean baseline, NOT run 1)
#   - Resume inpainting from run 4 checkpoint (was still improving at epoch 33)
#   - meshy_cc0_textured: restructured from flat dirs to per-example subdirs
#   - gemini_diverse EXCLUDED (SAM2 pseudo-labels failed — only 13/698 usable)
#   - Datasets: humanrig + vroid_cc0 + meshy_cc0_textured + anime_seg
#
# Estimated: ~10 hrs on A100, ~$3-4
#   - Seg: ~5-6 hrs (60 epochs)
#   - Inpainting pair generation: ~30 min
#   - Inpainting: ~3-4 hrs (50 epochs, resume from run 4)
#   - Quality filter + Marigold + ONNX + upload: ~30 min
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
echo "  Strata Training — Seg+Meshy+Inpainting"
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
echo "[0/7] Downloading checkpoints..."
echo ""

# Seg: resume from seg-only run checkpoint (clean CC0-only baseline)
# NOT run 1 — run 1 used prohibited Mixamo/Live2D data
RESUME_SEG="./checkpoints/segmentation/seg_only_best.pt"

echo "  Downloading seg-only run checkpoint..."
rclone copy hetzner:strata-training-data/checkpoints_run5_seg/segmentation/best.pt \
    ./checkpoints/segmentation/ --transfers 32 --fast-list -P

if [ -f "./checkpoints/segmentation/best.pt" ]; then
    cp ./checkpoints/segmentation/best.pt "$RESUME_SEG"
    echo "  Seg: resuming from seg-only run (CC0-only baseline)."
else
    echo "  FATAL: No seg-only checkpoint found in bucket."
    echo "  This run must resume from the seg-only checkpoint, not run 1 (prohibited data)."
    exit 1
fi

# Inpainting: resume from run 4 checkpoint (was still improving at epoch 33/50)
RESUME_INP="./checkpoints/inpainting/run4_best.pt"

echo "  Downloading run 4 inpainting checkpoint..."
mkdir -p ./checkpoints/inpainting
rclone copy hetzner:strata-training-data/checkpoints_run4/inpainting/best.pt \
    ./checkpoints/inpainting/ --transfers 32 --fast-list -P

if [ -f "./checkpoints/inpainting/best.pt" ]; then
    cp ./checkpoints/inpainting/best.pt "$RESUME_INP"
    echo "  Inpainting: resuming from run 4 (0.0028 val/l1, epoch 33)."
else
    echo "  WARNING: No run 4 inpainting checkpoint. Training from scratch."
    RESUME_INP=""
fi
echo ""

# ---------------------------------------------------------------------------
# 1. Download datasets
# ---------------------------------------------------------------------------
echo "[1/7] Downloading datasets..."
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

# Seg datasets
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
echo "[2/7] Quality filter + Marigold normals..."
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

# Marigold normals on new datasets that don't have them yet
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
echo "[3/7] Training SEGMENTATION model..."
echo ""
echo "  Resuming from: $RESUME_SEG"
echo "  Config: training/configs/segmentation_a100_seg_meshy.yaml"
echo "  New data: meshy_cc0_textured (15,281 restructured examples)"
echo ""

python -m training.train_segmentation \
    --config training/configs/segmentation_a100_seg_meshy.yaml \
    --resume "$RESUME_SEG" \
    --reset-epochs \
    2>&1 | tee "$LOG_DIR/segmentation.log"

echo ""
echo "  Segmentation training complete."
echo ""

# ---------------------------------------------------------------------------
# 4. Generate inpainting pairs + train inpainting
# ---------------------------------------------------------------------------
echo "[4/7] Generating inpainting pairs..."
echo ""

# Generate occlusion pairs from datasets already on disk
# (humanrig has seg masks, meshy_cc0_textured has seg masks, vroid_cc0 has seg masks)
PAIRS_DIR="./data_cloud/inpainting_pairs"
PAIRS_COUNT=$(find "$PAIRS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -100 | wc -l)

if [ "$PAIRS_COUNT" -ge 100 ]; then
    echo "  Occlusion pairs already exist ($PAIRS_COUNT+ dirs), skipping generation."
else
    python -m training.data.generate_occlusion_pairs \
        --source-dirs \
            ./data_cloud/humanrig \
            ./data_cloud/meshy_cc0_textured \
            ./data_cloud/vroid_cc0 \
        --output-dir "$PAIRS_DIR" \
        --max-images 15000 \
        --masks-per-image 3 \
        2>&1 | tee "$LOG_DIR/generate_inpainting_pairs.log"
fi

echo ""
echo "  Training INPAINTING model..."
echo ""

INPAINT_ARGS="--config training/configs/inpainting_a100_lean.yaml"
if [ -n "$RESUME_INP" ] && [ -f "$RESUME_INP" ]; then
    INPAINT_ARGS="$INPAINT_ARGS --resume $RESUME_INP --reset-epochs"
    echo "  Resuming from run 4 checkpoint."
fi

python -m training.train_inpainting \
    $INPAINT_ARGS \
    2>&1 | tee "$LOG_DIR/inpainting.log"

echo ""
echo "  Inpainting training complete."
echo ""

# ---------------------------------------------------------------------------
# 5. ONNX Export (seg + inpainting)
# ---------------------------------------------------------------------------
echo "[5/7] Exporting models to ONNX..."
echo ""

ONNX_DIR="./models/onnx"
mkdir -p "$ONNX_DIR"

if [ -f "checkpoints/segmentation/best.pt" ]; then
    python -m training.export_onnx \
        --model segmentation \
        --checkpoint checkpoints/segmentation/best.pt \
        --output "$ONNX_DIR/segmentation.onnx" \
        2>&1 | tee "$LOG_DIR/export_seg.log"
    echo "  Exported segmentation.onnx"
else
    echo "  WARNING: No seg checkpoint found for export."
fi

if [ -f "checkpoints/inpainting/best.pt" ]; then
    python -m training.export_onnx \
        --model inpainting \
        --checkpoint checkpoints/inpainting/best.pt \
        --output "$ONNX_DIR/inpainting.onnx" \
        2>&1 | tee "$LOG_DIR/export_inp.log"
    echo "  Exported inpainting.onnx"
else
    echo "  WARNING: No inpainting checkpoint found for export."
fi
echo ""

# ---------------------------------------------------------------------------
# 6. Upload to bucket
# ---------------------------------------------------------------------------
echo "[6/7] Uploading checkpoints, logs, and ONNX..."
echo ""

rclone copy ./checkpoints/segmentation/ hetzner:strata-training-data/checkpoints_seg_meshy/segmentation/ \
    --transfers 32 --fast-list -P
rclone copy ./checkpoints/inpainting/ hetzner:strata-training-data/checkpoints_seg_meshy/inpainting/ \
    --transfers 32 --fast-list -P
rclone copy ./logs/ hetzner:strata-training-data/logs/ \
    --transfers 32 --fast-list -P
for onnx_file in segmentation.onnx inpainting.onnx; do
    if [ -f "$ONNX_DIR/$onnx_file" ]; then
        rclone copy "$ONNX_DIR/$onnx_file" hetzner:strata-training-data/models/onnx_seg_meshy/ \
            --transfers 32 --fast-list -P
    fi
done

echo ""

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
echo "============================================"
echo "  Seg+Meshy+Inpainting run complete!"
echo "  Finished: $(date)"
echo ""
echo "  Segmentation results:"
grep -E "Best mIoU|New best|miou" "$LOG_DIR/segmentation.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  Inpainting results:"
grep -E "Best|val/l1|best" "$LOG_DIR/inpainting.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  To download results to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_seg_meshy/ ./checkpoints_seg_meshy/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/onnx_seg_meshy/ ./models/onnx_seg_meshy/ --transfers 32 --fast-list -P"
echo "============================================"

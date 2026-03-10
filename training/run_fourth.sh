#!/usr/bin/env bash
# =============================================================================
# Strata Training — Fourth Run (A100)
#
# Goal: Ship-ready models 1-4
#   - Model 1 (Seg): Fine-tune from run 1 (0.545 mIoU) + Gemini data + label smoothing
#   - Model 2 (Joints): Keep run 3 checkpoint (0.001206) — no retraining needed
#   - Model 3 (Weights): Keep run 3 checkpoint (0.023 MAE) — no retraining needed
#   - Model 4 (Inpainting): Train with fixed data loader + occlusion pairs
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_fourth.sh
#   ./training/run_fourth.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run4_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Fourth Run"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# 0. Download run 1 seg checkpoint (the good one — 0.545 mIoU)
# ---------------------------------------------------------------------------
echo "[0/6] Downloading run 1 segmentation checkpoint..."
echo ""

RUN1_CKPT="checkpoints/segmentation/run1_best.pt"
if [ -f "$RUN1_CKPT" ]; then
    echo "  run1_best.pt already exists."
else
    # Download from bucket — run 1 checkpoint was saved as checkpoints_run1/
    rclone copy hetzner:strata-training-data/checkpoints_run1/segmentation/best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
    if [ -f "checkpoints/segmentation/best.pt" ]; then
        cp checkpoints/segmentation/best.pt "$RUN1_CKPT"
        echo "  Saved as $RUN1_CKPT"
    else
        echo "  WARNING: Could not find run 1 checkpoint in bucket."
        echo "  Trying checkpoints/segmentation/best.pt from current bucket..."
        rclone copy hetzner:strata-training-data/checkpoints/segmentation/best.pt \
            ./checkpoints/segmentation/ --transfers 32 --fast-list -P
        if [ -f "checkpoints/segmentation/best.pt" ]; then
            cp checkpoints/segmentation/best.pt "$RUN1_CKPT"
            echo "  Saved as $RUN1_CKPT"
        else
            echo "  FATAL: No segmentation checkpoint found. Cannot resume."
            exit 1
        fi
    fi
fi

# Also download run 3 joints + weights checkpoints (keep those)
echo "  Downloading run 3 joints + weights checkpoints..."
rclone copy hetzner:strata-training-data/checkpoints_run3/joints/ \
    ./checkpoints/joints/ --transfers 32 --fast-list -P
rclone copy hetzner:strata-training-data/checkpoints_run3/weights/ \
    ./checkpoints/weights/ --transfers 32 --fast-list -P
echo ""

# ---------------------------------------------------------------------------
# 1. Download Gemini diverse dataset
# ---------------------------------------------------------------------------
echo "[1/6] Downloading Gemini diverse dataset..."
echo ""

if [ -d "./data_cloud/gemini_diverse" ] && [ "$(ls -A ./data_cloud/gemini_diverse 2>/dev/null | head -1)" ]; then
    echo "  gemini_diverse already exists, skipping download."
else
    rclone copy hetzner:strata-training-data/gemini_diverse/ ./data_cloud/gemini_diverse/ \
        --transfers 32 --checkers 64 --fast-list --size-only -P
fi
echo ""

# ---------------------------------------------------------------------------
# 2. Run quality filter on seg masks (reject bad labels)
# ---------------------------------------------------------------------------
echo "[2/8] Running segmentation quality filter..."
echo ""

for ds in humanrig unirig meshy_cc0 meshy_cc0_textured gemini_diverse; do
    ds_dir="./data_cloud/$ds"
    if [ -d "$ds_dir" ] && [ ! -f "$ds_dir/quality_filter.json" ]; then
        echo "  Filtering $ds..."
        python scripts/filter_seg_quality.py \
            --data-dirs "$ds_dir" \
            --output-dir "$ds_dir" \
            --min-regions 4 \
            --max-single-region 0.70 \
            --min-foreground 0.05 \
            2>&1 | tee -a "$LOG_DIR/quality_filter.log"
    else
        echo "  $ds: already filtered or not present, skipping."
    fi
done

echo ""
echo "  Quality filter complete."
echo ""

# ---------------------------------------------------------------------------
# 3. Normals + depth enrichment for datasets missing them
# ---------------------------------------------------------------------------
echo "[3/8] Enriching datasets with surface normals + depth (Marigold)..."
echo ""

pip install -q diffusers transformers accelerate 2>/dev/null

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
# 4. Train segmentation (resume from run 1 — 0.545 mIoU)
# ---------------------------------------------------------------------------
echo "[4/8] Training SEGMENTATION model (resume from run 1)..."
echo ""
echo "  Resuming from: $RUN1_CKPT"
echo "  Config: training/configs/segmentation_a100_run4.yaml"
echo "  Strategy: Fine-tune at 5e-5 LR, 50 epochs, label smoothing 0.05"
echo ""

python -m training.train_segmentation \
    --config training/configs/segmentation_a100_run4.yaml \
    --resume "$RUN1_CKPT" \
    2>&1 | tee "$LOG_DIR/segmentation.log"

echo ""
echo "  Segmentation training complete."
echo ""

# ---------------------------------------------------------------------------
# 5. Generate inpainting pairs + train inpainting
# ---------------------------------------------------------------------------
echo "[5/8] Generating inpainting pairs + training INPAINTING model..."
echo ""

PAIRS_DIR="./data_cloud/inpainting_pairs"
PAIRS_COUNT=$(find "$PAIRS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -100 | wc -l)

if [ "$PAIRS_COUNT" -ge 100 ]; then
    echo "  Occlusion pairs already exist ($PAIRS_COUNT+ dirs), skipping generation."
else
    python -m training.data.generate_occlusion_pairs \
        --source-dirs \
            ./data_cloud/humanrig \
            ./data_cloud/gemini_diverse \
        --output-dir "$PAIRS_DIR" \
        --max-images 15000 \
        --masks-per-image 3 \
        2>&1 | tee "$LOG_DIR/generate_inpainting_pairs.log"
fi

echo ""
echo "  Training INPAINTING model..."
echo ""

python -m training.train_inpainting \
    --config training/configs/inpainting_a100_lean.yaml \
    2>&1 | tee "$LOG_DIR/inpainting.log"

echo ""
echo "  Inpainting training complete."
echo ""

# ---------------------------------------------------------------------------
# 6. ONNX Export (all 4 models)
# ---------------------------------------------------------------------------
echo "[6/8] Exporting all models to ONNX..."
echo ""

ONNX_DIR="./models/onnx"
mkdir -p "$ONNX_DIR"

for model_export in \
    "segmentation checkpoints/segmentation/best.pt segmentation.onnx" \
    "joints checkpoints/joints/best.pt joint_refinement.onnx" \
    "weights_vertex checkpoints/weights/best.pt weight_prediction.onnx" \
    "inpainting checkpoints/inpainting/best.pt inpainting.onnx"
do
    set -- $model_export
    model_name=$1 ckpt=$2 onnx_file=$3
    if [ -f "$ckpt" ]; then
        echo "  Exporting $model_name -> $onnx_file"
        python -m training.export_onnx \
            --model "$model_name" \
            --checkpoint "$ckpt" \
            --output "$ONNX_DIR/$onnx_file" \
            2>&1 | tee -a "$LOG_DIR/export.log"
    else
        echo "  SKIP $model_name — no checkpoint at $ckpt"
    fi
done

echo ""

# ---------------------------------------------------------------------------
# 7. Seg-enrich new Gemini data with run 4 model (for run 5 bootstrap)
# ---------------------------------------------------------------------------
echo "[7/8] Re-enriching Gemini data with updated seg model..."
echo ""

if [ -d "./data_cloud/gemini_diverse" ]; then
    python run_seg_enrich.py \
        --checkpoint checkpoints/segmentation/best.pt \
        --input-dir ./data_cloud/gemini_diverse \
        --only-missing \
        2>&1 | tee "$LOG_DIR/seg_enrich_gemini.log"
    echo "  Re-enrichment complete."
else
    echo "  No gemini_diverse dir, skipping."
fi
echo ""

# ---------------------------------------------------------------------------
# 8. Upload everything to bucket
# ---------------------------------------------------------------------------
echo "[8/8] Uploading checkpoints, logs, and ONNX models..."
echo ""

rclone copy ./checkpoints/ hetzner:strata-training-data/checkpoints_run4/ \
    --transfers 32 --fast-list -P
rclone copy ./logs/ hetzner:strata-training-data/logs/ \
    --transfers 32 --fast-list -P
rclone copy ./models/onnx/ hetzner:strata-training-data/models/onnx_run4/ \
    --transfers 32 --fast-list -P

echo ""
echo "============================================"
echo "  Fourth run complete!"
echo "  Finished: $(date)"
echo ""
echo "  ONNX models:"
ls -lh "$ONNX_DIR/"*.onnx 2>/dev/null || echo "  (no ONNX files found)"
echo ""
echo "  Checkpoints:"
ls -lh checkpoints/*/best.pt 2>/dev/null || echo "  (no checkpoints found)"
echo ""
echo "  To download results to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_run4/ ./checkpoints/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/onnx_run4/ ./models/onnx/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/logs/ ./logs/ --transfers 32 --fast-list -P"
echo "============================================"

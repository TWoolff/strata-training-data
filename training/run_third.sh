#!/usr/bin/env bash
# =============================================================================
# Strata Training — Third Run (A100)
#
# Key changes from run 2:
#   - Meshy CC0 dataset: ~36K new CC0-licensed examples (flat + textured + unrigged)
#   - Dropped Mixamo (proprietary license) from seg + joints training
#   - Segmentation model now has depth + normals heads (Marigold-distilled)
#   - Weight data: 54 → ~27K examples (HumanRig + UniRig — kept for weights only)
#   - Inpainting pair generation now includes Meshy textured images
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#   rclone copy hetzner:strata-training-data/checkpoints/ ./checkpoints/ --transfers 32 --fast-list -P
#
# Usage:
#   chmod +x training/run_third.sh
#   ./training/run_third.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run3_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Third Run"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# 0. Install extra deps (Marigold normals + depth)
# ---------------------------------------------------------------------------
echo "[0/7] Installing extra dependencies..."
pip install -q diffusers transformers accelerate
echo "  Done."
echo ""

# ---------------------------------------------------------------------------
# 1. Normals + depth enrichment (Marigold LCM)
#    Meshy datasets already have Blender-rendered depth/normals.
#    Enrich remaining datasets that need Marigold depth.png + normals.png.
#    --only-missing skips examples already enriched in run 2.
# ---------------------------------------------------------------------------
echo "[1/7] Enriching datasets with surface normals + depth (Marigold)..."
echo ""

for ds in live2d curated_diverse humanrig unirig; do
    if [ -d "./data_cloud/$ds" ]; then
        echo "  Enriching $ds with normals + depth..."
        python run_normals_enrich.py \
            --input-dir "./data_cloud/$ds" \
            --only-missing \
            --batch-size 16 \
            2>&1 | tee "$LOG_DIR/enrich_normals_${ds}.log"
        echo ""
    fi
done

echo "  Normals enrichment complete."
echo ""

# ---------------------------------------------------------------------------
# 2. Generate inpainting pairs from Meshy textured + other datasets
# ---------------------------------------------------------------------------
echo "[2/7] Generating inpainting occlusion pairs..."
echo ""

python -m training.data.generate_occlusion_pairs \
    --source-dirs \
        ./data_cloud/meshy_cc0_textured \
        ./data_cloud/meshy_cc0_unrigged \
        ./data_cloud/humanrig \
        ./data_cloud/curated_diverse \
    --output-dir ./data_cloud/inpainting_pairs \
    --max-images 15000 \
    --masks-per-image 3 \
    2>&1 | tee "$LOG_DIR/generate_inpainting_pairs.log"

echo ""
echo "  Inpainting pairs generated."
echo ""

# ---------------------------------------------------------------------------
# 3. Train all models
# ---------------------------------------------------------------------------
echo "[3/7] Training all models (lean config)..."
echo ""

./training/train_all.sh lean 2>&1 | tee "$LOG_DIR/train_all.log"

echo ""
echo "  Training complete."
echo ""

# ---------------------------------------------------------------------------
# 4. Upload results to bucket
# ---------------------------------------------------------------------------
echo "[4/7] Uploading checkpoints, logs, and ONNX models to bucket..."
echo ""

rclone copy ./checkpoints/ hetzner:strata-training-data/checkpoints/ \
    --transfers 32 --fast-list -P
echo ""

rclone copy ./logs/ hetzner:strata-training-data/logs/ \
    --transfers 32 --fast-list -P
echo ""

rclone copy ./models/onnx/ hetzner:strata-training-data/models/onnx/ \
    --transfers 32 --fast-list -P
echo ""

echo "  Upload complete."
echo ""

# ---------------------------------------------------------------------------
# 5. Pack enriched datasets as tar archives
# ---------------------------------------------------------------------------
echo "[5/7] Packing enriched datasets as tar archives..."
echo ""

TAR_DIR="./data_cloud/_tars"
mkdir -p "$TAR_DIR"

# Re-tar datasets that were enriched (normals/depth added)
# Meshy datasets are uploaded separately from Mac before the run
for ds in live2d humanrig curated_diverse unirig; do
    if [ -d "./data_cloud/$ds" ]; then
        echo "  Packing $ds..."
        (cd ./data_cloud && tar cf - "$ds") > "$TAR_DIR/${ds}.tar"
        tar_size=$(du -sh "$TAR_DIR/${ds}.tar" 2>/dev/null | cut -f1)
        echo "    → ${ds}.tar ($tar_size)"
        echo "    → Uploading..."
        rclone copy "$TAR_DIR/${ds}.tar" hetzner:strata-training-data/tars/ \
            --transfers 8 --fast-list -P
        rm -f "$TAR_DIR/${ds}.tar"
        echo ""
    fi
done

echo "  Deleting loose files from bucket (only for datasets we just tarred)..."
for ds in live2d humanrig curated_diverse unirig; do
    if rclone lsf "hetzner:strata-training-data/tars/${ds}.tar" 2>/dev/null | grep -q "${ds}.tar"; then
        echo "    Deleting $ds/ (tar verified)..."
        rclone purge "hetzner:strata-training-data/$ds/" 2>/dev/null || true
    else
        echo "    SKIP $ds/ — tar not found, keeping loose files"
    fi
done
echo ""
echo "  Tar upload complete."
echo ""

# ---------------------------------------------------------------------------
# 6. Clean up tar staging dir
# ---------------------------------------------------------------------------
rmdir "$TAR_DIR" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
echo "============================================"
echo "  Third run complete!"
echo "  Finished: $(date)"
echo ""
echo "  ONNX models:"
ls -lh ./models/onnx/*.onnx 2>/dev/null || echo "  (no ONNX files found)"
echo ""
echo "  Checkpoints:"
ls -lh checkpoints/*/best.pt 2>/dev/null || echo "  (no checkpoints found)"
echo ""
echo "  Everything uploaded to bucket."
echo "  Safe to destroy this instance."
echo ""
echo "  To download results to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints/ ./checkpoints/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/onnx/ ./models/onnx/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/logs/ ./logs/ --transfers 32 --fast-list -P"
echo "============================================"

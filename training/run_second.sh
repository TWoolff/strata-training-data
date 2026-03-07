#!/usr/bin/env bash
# =============================================================================
# Strata Training — Second Run (A100)
#
# Complete pipeline: seg enrichment → normals enrichment → train all → upload
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#   rclone copy hetzner:strata-training-data/checkpoints/ ./checkpoints/ --transfers 32 --fast-list -P
#
# Usage:
#   chmod +x training/run_second.sh
#   ./training/run_second.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run2_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Second Run"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# 0. Install extra deps (normals)
# ---------------------------------------------------------------------------
echo "[0/6] Installing extra dependencies..."
pip install -q diffusers transformers accelerate
echo "  Done."
echo ""

# ---------------------------------------------------------------------------
# 1. Seg enrichment (pseudo-label ~168K images with trained Model 1)
# ---------------------------------------------------------------------------
echo "[1/6] Enriching datasets with 22-class segmentation..."
echo "  Using checkpoint: checkpoints/segmentation/best.pt"
echo ""

for ds in fbanimehq anime_seg; do
    if [ -d "./data_cloud/$ds" ]; then
        echo "  Enriching $ds..."
        python run_seg_enrich.py \
            --input-dir "./data_cloud/$ds" \
            --checkpoint checkpoints/segmentation/best.pt \
            --only-missing \
            2>&1 | tee "$LOG_DIR/enrich_seg_${ds}.log"
        echo ""
    fi
done

echo "  Seg enrichment complete."
echo ""

# ---------------------------------------------------------------------------
# 2. Normals enrichment (small datasets only, ~2h)
# ---------------------------------------------------------------------------
echo "[2/6] Enriching datasets with surface normals + depth (Marigold)..."
echo ""

for ds in segmentation live2d curated_diverse humanrig; do
    if [ -d "./data_cloud/$ds" ]; then
        echo "  Enriching $ds with normals..."
        python run_normals_enrich.py \
            --input-dir "./data_cloud/$ds" \
            --only-missing \
            2>&1 | tee "$LOG_DIR/enrich_normals_${ds}.log"
        echo ""
    fi
done

echo "  Normals enrichment complete."
echo ""

# ---------------------------------------------------------------------------
# 3. Train all models
# ---------------------------------------------------------------------------
echo "[3/6] Training all models (lean config)..."
echo ""

./training/train_all.sh lean 2>&1 | tee "$LOG_DIR/train_all.log"

echo ""
echo "  Training complete."
echo ""

# ---------------------------------------------------------------------------
# 4. Upload results to bucket
# ---------------------------------------------------------------------------
echo "[4/6] Uploading checkpoints, logs, and ONNX models to bucket..."
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
# 5. Pack datasets as tar archives (includes enriched normals + depth)
# ---------------------------------------------------------------------------
echo "[5/6] Packing datasets as tar archives (includes enriched data)..."
echo ""

TAR_DIR="./data_cloud/_tars"
mkdir -p "$TAR_DIR"

for ds in segmentation live2d humanrig anime_seg fbanimehq curated_diverse; do
    if [ -d "./data_cloud/$ds" ]; then
        echo "  Packing $ds..."
        (cd ./data_cloud && tar cf - "$ds") > "$TAR_DIR/${ds}.tar"
        tar_size=$(du -sh "$TAR_DIR/${ds}.tar" 2>/dev/null | cut -f1)
        echo "    → ${ds}.tar ($tar_size)"
    fi
done

echo ""
echo "  Uploading tar archives to bucket..."
rclone copy "$TAR_DIR/" hetzner:strata-training-data/tars/ \
    --transfers 8 --fast-list -P
echo ""

echo "  Deleting loose files from bucket (tars are now the source of truth)..."
for ds in segmentation live2d humanrig anime_seg fbanimehq curated_diverse; do
    echo "    Deleting $ds/ ..."
    rclone purge "hetzner:strata-training-data/$ds/" 2>/dev/null || true
done
echo ""
echo "  Tar upload complete. Future runs will download ~20 min instead of ~5 hours."
echo ""

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
echo "============================================"
echo "  Second run complete!"
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

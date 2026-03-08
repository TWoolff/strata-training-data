#!/usr/bin/env bash
# =============================================================================
# Strata Training — Third Run (A100)
#
# Key changes from run 2:
#   - Segmentation model now has depth + normals heads (Marigold-distilled)
#   - Weight data: 54 → ~26.5K examples (HumanRig + UniRig)
#   - Normals enrichment on unirig dataset
#   - No seg enrichment needed (done in run 2)
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
echo "[0/6] Installing extra dependencies..."
pip install -q diffusers transformers accelerate
echo "  Done."
echo ""

# ---------------------------------------------------------------------------
# 1. Normals + depth enrichment (Marigold LCM)
#    Enrich datasets that need depth.png + normals.png for the new seg heads.
#    --only-missing skips examples already enriched in run 2.
# ---------------------------------------------------------------------------
echo "[1/6] Enriching datasets with surface normals + depth (Marigold)..."
echo ""

for ds in segmentation live2d curated_diverse humanrig unirig; do
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
# 2. Train all models
# ---------------------------------------------------------------------------
echo "[2/6] Training all models (lean config)..."
echo ""

./training/train_all.sh lean 2>&1 | tee "$LOG_DIR/train_all.log"

echo ""
echo "  Training complete."
echo ""

# ---------------------------------------------------------------------------
# 3. Upload results to bucket
# ---------------------------------------------------------------------------
echo "[3/6] Uploading checkpoints, logs, and ONNX models to bucket..."
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
# 4. Pack datasets as tar archives (includes newly enriched normals + depth)
# ---------------------------------------------------------------------------
echo "[4/6] Packing datasets as tar archives (includes enriched data)..."
echo ""

TAR_DIR="./data_cloud/_tars"
mkdir -p "$TAR_DIR"

# Only re-tar datasets that were enriched (normals/depth added)
# anime_seg and fbanimehq are unchanged — skip them to save ~30GB upload
for ds in segmentation live2d humanrig curated_diverse unirig; do
    if [ -d "./data_cloud/$ds" ]; then
        echo "  Packing $ds..."
        (cd ./data_cloud && tar cf - "$ds") > "$TAR_DIR/${ds}.tar"
        tar_size=$(du -sh "$TAR_DIR/${ds}.tar" 2>/dev/null | cut -f1)
        echo "    → ${ds}.tar ($tar_size)"
        # Upload immediately and remove local tar to save disk space
        echo "    → Uploading..."
        rclone copy "$TAR_DIR/${ds}.tar" hetzner:strata-training-data/tars/ \
            --transfers 8 --fast-list -P
        rm -f "$TAR_DIR/${ds}.tar"
        echo ""
    fi
done

echo "  Deleting loose files from bucket (only for datasets we just tarred)..."
for ds in segmentation live2d humanrig curated_diverse unirig; do
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
# 5. Clean up tar staging dir
# ---------------------------------------------------------------------------
rmdir "$TAR_DIR" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 6. Summary
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

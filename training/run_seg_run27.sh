#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 27 Seg (A100) — See-Through SAM + Dr. Li's converter
#
# What Run 26 v1 did WRONG:
#   - Converted See-Through's 19 clothing classes to 22 anatomy via naive L/R
#     + vertical splits (50/50 for chest/spine, 40/35/25 for arm segments, etc.)
#   - Lost the hair_back class entirely (hair → head blanket mapping)
#   - Didn't use the topwear→arm extraction logic; handwear wasn't anchored to
#     actual joint/bbox reasoning.
#
# What Run 27 does RIGHT:
#   - Pseudo-labels with See-Through SAM (same model as Run 26) → 19-class output.
#   - Re-encodes that output in Dr. Li's PNG format (pixel values 0,10,…,180;255=bg).
#   - Delegates 19→22 conversion to convert_li_labels.convert_with_heuristics
#     (or convert_with_joints if joints.json exists) — the exact code that
#     produced gemini_li_converted, Run 20's highest-weight illustrated dataset.
#
# This means: anatomically-proper hair_back split, body-proportion-based knee
# position, face-centered midline, topwear-to-arm extraction, bbox-derived
# shoulder width. Same conversion quality as Dr. Li's 694 hand labels, applied
# to 3,207 See-Through pseudo-labels.
#
# Estimated: ~4-5 hrs total on A100
#
# Usage:
#   chmod +x training/run_seg_run27.sh
#   ./training/run_seg_run27.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run27_seg_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 27 Seg"
echo "  See-Through SAM + Dr. Li's converter"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
echo "[pre] Pre-flight checks..."

PREFLIGHT_FAIL=0

if ! rclone lsd hetzner:strata-training-data/ &>/dev/null; then
    echo "  FAIL: rclone"; PREFLIGHT_FAIL=1
else
    echo "  OK: rclone bucket connection"
fi

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  FAIL: CUDA"; PREFLIGHT_FAIL=1
else
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "  OK: CUDA — $GPU_NAME"
fi

RUN20_CKPT="checkpoints/segmentation/run20_best.pt"
if [ ! -f "$RUN20_CKPT" ]; then
    mkdir -p checkpoints/segmentation
    rclone copy hetzner:strata-training-data/checkpoints_run20_seg/segmentation/run20_best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
fi
echo "  OK: Run 20 checkpoint"

# Download missing datasets
download_if_missing() {
    local tar_name="$1"; local extract_dir="$2"
    if [ -d "$extract_dir" ] && [ -n "$(ls "$extract_dir"/ 2>/dev/null | head -1)" ]; then
        return 0
    fi
    echo "  $extract_dir missing — downloading $tar_name..."
    mkdir -p data/tars data_cloud
    rclone copy "hetzner:strata-training-data/tars/$tar_name" ./data/tars/ --transfers 32 --fast-list -P
    tar xf "./data/tars/$tar_name" -C ./data_cloud/
    rm -f "./data/tars/$tar_name"
}

download_if_missing "humanrig.tar" "./data_cloud/humanrig"
download_if_missing "vroid_cc0.tar" "./data_cloud/vroid_cc0"
download_if_missing "meshy_cc0_textured_restructured.tar" "./data_cloud/meshy_cc0_restructured"
download_if_missing "gemini_li_converted.tar" "./data_cloud/gemini_li_converted"
download_if_missing "cvat_annotated.tar" "./data_cloud/cvat_annotated"
download_if_missing "sora_diverse.tar" "./data_cloud/sora_diverse"
download_if_missing "flux_diverse_clean.tar" "./data_cloud/flux_diverse_clean"
download_if_missing "gemini_diverse.tar" "./data_cloud/gemini_diverse"

if [ ! -f "data_cloud/frozen_val_test.json" ]; then
    rclone copy hetzner:strata-training-data/data_cloud/frozen_val_test.json \
        ./data_cloud/ --transfers 4 --fast-list -P
fi
echo "  OK: frozen val/test splits"

if [ "$PREFLIGHT_FAIL" -ne 0 ]; then exit 1; fi
echo ""

# ---------------------------------------------------------------------------
# 1. See-Through setup
# ---------------------------------------------------------------------------
echo "[1/5] Setting up See-Through..."
if [ ! -d /workspace/see-through ]; then
    cd /workspace
    git clone https://github.com/shitagaki-lab/see-through.git
    cd - >/dev/null
fi

cd /workspace/see-through
pip install -q -r requirements.txt 2>&1 | tail -3
pip install -q huggingface_hub rembg 2>&1 | tail -1
cd - >/dev/null

export SEETHROUGH_ROOT=/workspace/see-through
echo "  OK: See-Through deps installed"
echo ""

# ---------------------------------------------------------------------------
# 2. Pseudo-label with See-Through SAM + Dr. Li converter
# ---------------------------------------------------------------------------
echo "[2/5] Pseudo-labeling gemini_diverse (See-Through SAM → Li format → 22-class via heuristics)..."
mkdir -p /workspace/weights

python3 scripts/seethrough_sam_to_seg.py \
    --input-dir ./data_cloud/gemini_diverse \
    --checkpoint /workspace/weights/li_sam_iter2.pt \
    --device cuda \
    2>&1 | tee "$LOG_DIR/seethrough_pseudo_label.log"

rm -f ./data_cloud/gemini_diverse/quality_filter.json
echo ""

# ---------------------------------------------------------------------------
# 3. Marigold enrichment (gemini_diverse only; others already done in Run 24)
# ---------------------------------------------------------------------------
echo "[3/5] Marigold enrichment (gemini_diverse, only-missing)..."
python3 run_normals_enrich.py \
    --input-dir ./data_cloud/gemini_diverse \
    --only-missing \
    --batch-size 16 \
    2>&1 | tee "$LOG_DIR/enrich.log" || true
echo ""

# ---------------------------------------------------------------------------
# 4. Quality filter
# ---------------------------------------------------------------------------
echo "[4/5] Quality filter..."

for ds_dir in ./data_cloud/humanrig ./data_cloud/vroid_cc0 ./data_cloud/meshy_cc0_restructured ./data_cloud/gemini_li_converted ./data_cloud/cvat_annotated ./data_cloud/sora_diverse ./data_cloud/flux_diverse_clean ./data_cloud/gemini_diverse; do
    ds_name=$(basename "$ds_dir")
    if [ -f "$ds_dir/quality_filter.json" ]; then
        echo "  $ds_name: exists, skip."
    else
        echo "  $ds_name: running quality filter..."
        python3 scripts/filter_seg_quality.py \
            --data-dirs "$ds_dir" \
            --min-regions 4 --max-single-region 0.70 --min-foreground 0.05 \
            2>&1 | tee -a "$LOG_DIR/quality_filter.log"
    fi
done
echo ""

# ---------------------------------------------------------------------------
# 5. Train — reuse Run 25 config (softening on, gemini_diverse wt 2.0)
# ---------------------------------------------------------------------------
echo "[5/5] Training SEGMENTATION model..."
echo "  Config: training/configs/segmentation_a100_run25.yaml (reused)"
echo "  Resume: $RUN20_CKPT — NO --reset-epochs"
echo ""

python3 -m training.train_segmentation \
    --config training/configs/segmentation_a100_run25.yaml \
    --resume "$RUN20_CKPT" \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""

# ---------------------------------------------------------------------------
# Export + Upload
# ---------------------------------------------------------------------------
echo "[final] Exporting ONNX + uploading..."

cp checkpoints/segmentation/best.pt checkpoints/segmentation/run27_best.pt

python3 -m training.export_onnx \
    --model segmentation \
    --checkpoint checkpoints/segmentation/run27_best.pt \
    --output ./models/onnx/segmentation_run27.onnx \
    2>&1 | tee "$LOG_DIR/export.log"

rclone copy checkpoints/segmentation/run27_best.pt \
    hetzner:strata-training-data/checkpoints_run27_seg/segmentation/ \
    --transfers 4 --fast-list --size-only -P

rclone copy ./models/onnx/segmentation_run27.onnx \
    hetzner:strata-training-data/models/onnx_run27_seg/ \
    --transfers 4 --fast-list --size-only -P

rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run27_seg_${TIMESTAMP}/ \
    --transfers 4 --fast-list -P

echo ""
echo "============================================"
echo "  Run 27 complete! Finished: $(date)"
grep -E "best mIoU|New best" "$LOG_DIR/train.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  Download: rclone copy hetzner:strata-training-data/checkpoints_run27_seg/ /Volumes/TAMWoolff/data/checkpoints_run27_seg/ --transfers 32 --fast-list -P"
echo "============================================"

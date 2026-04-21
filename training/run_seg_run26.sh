#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 26 Seg (A100) — See-Through SAM pseudo-labels
#
# What Run 25/24 did WRONG:
#   - Pseudo-labeled gemini_diverse with Run 20 seg model (self-distillation)
#   - Classic self-training ceiling: teacher's biases propagated to student
#   - Result: Run 25 plateau'd near/below Run 20, no data-expansion payoff
#
# What Run 26 does RIGHT:
#   - Pseudo-labels gemini_diverse with Dr. Li's See-Through SAM body parsing
#     model (9K illustrated chars of training, Apache-2.0, 19 clothing classes).
#   - Uses a custom 19→22 class converter (see scripts/seethrough_sam_to_seg.py)
#     that splits topwear/handwear/legwear/footwear into anatomy regions with
#     the same heuristics as convert_seethrough_to_seg.py.
#   - Different model = different biases = labels that push our Run 20 student
#     past its own ceiling.
#
# Estimated: ~4-5 hrs total on A100
#   - See-Through repo clone + deps (~5 min)
#   - SAM checkpoint download (~2 min)
#   - Pseudo-label 3,207 gemini_diverse imgs (~1 hr @ ~1 img/s)
#   - Quality filter re-run (~5 min)
#   - Train (~3 hrs, resume Run 20, 12 epochs)
#   - Export + upload (~5 min)
#
# Prerequisites:
#   - A100 already has data on disk from Run 24/25 (meshy, humanrig, etc.)
#   - export BUCKET_ACCESS_KEY BUCKET_SECRET
#
# Usage:
#   chmod +x training/run_seg_run26.sh
#   ./training/run_seg_run26.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run26_seg_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 26 Seg"
echo "  See-Through SAM pseudo-labels (breaking"
echo "  the self-distillation ceiling)"
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
    echo "  FAIL: rclone cannot connect to Hetzner"
    PREFLIGHT_FAIL=1
else
    echo "  OK: rclone bucket connection"
fi

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  FAIL: CUDA not available"
    PREFLIGHT_FAIL=1
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

# Ensure gemini_diverse is extracted (it should be from Run 24/25, but re-grab if missing)
if [ ! -d "./data_cloud/gemini_diverse" ] || [ -z "$(ls ./data_cloud/gemini_diverse/ 2>/dev/null | head -1)" ]; then
    echo "  gemini_diverse missing — downloading..."
    mkdir -p data/tars data_cloud
    rclone copy hetzner:strata-training-data/tars/gemini_diverse.tar ./data/tars/ \
        --transfers 32 --fast-list -P
    tar xf ./data/tars/gemini_diverse.tar -C ./data_cloud/
    rm -f ./data/tars/gemini_diverse.tar
fi
GEMINI_DIV_COUNT=$(ls -d ./data_cloud/gemini_diverse/*/ 2>/dev/null | wc -l | tr -d ' ')
echo "  OK: gemini_diverse ($GEMINI_DIV_COUNT examples)"

# Meshy — Run 24 script had the wrong tar name; download if missing
if [ ! -d "./data_cloud/meshy_cc0_restructured" ] || [ -z "$(ls ./data_cloud/meshy_cc0_restructured/ 2>/dev/null | head -1)" ]; then
    echo "  meshy_cc0_restructured missing — downloading..."
    mkdir -p data/tars
    rclone copy hetzner:strata-training-data/tars/meshy_cc0_textured_restructured.tar \
        ./data/tars/ --transfers 32 --fast-list -P
    tar xf ./data/tars/meshy_cc0_textured_restructured.tar -C ./data_cloud/
    rm -f ./data/tars/meshy_cc0_textured_restructured.tar
fi

# Frozen splits
if [ ! -f "data_cloud/frozen_val_test.json" ]; then
    rclone copy hetzner:strata-training-data/data_cloud/frozen_val_test.json \
        ./data_cloud/ --transfers 4 --fast-list -P
fi
echo "  OK: frozen val/test splits"

# Other datasets — re-download any that are missing (from Run 24 script misconfig)
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
download_if_missing "gemini_li_converted.tar" "./data_cloud/gemini_li_converted"
download_if_missing "cvat_annotated.tar" "./data_cloud/cvat_annotated"
download_if_missing "sora_diverse.tar" "./data_cloud/sora_diverse"
download_if_missing "flux_diverse_clean.tar" "./data_cloud/flux_diverse_clean"

if [ "$PREFLIGHT_FAIL" -ne 0 ]; then
    echo "  Pre-flight failed."
    exit 1
fi
echo ""

# ---------------------------------------------------------------------------
# 1. Clone See-Through + install deps
# ---------------------------------------------------------------------------
echo "[1/5] Setting up See-Through..."
if [ ! -d /workspace/see-through ]; then
    cd /workspace
    git clone https://github.com/shitagaki-lab/see-through.git
    cd - >/dev/null
fi

cd /workspace/see-through
pip install -q -r requirements.txt 2>&1 | tail -3
pip install -q huggingface_hub 2>&1 | tail -1
cd - >/dev/null

export SEETHROUGH_ROOT=/workspace/see-through
echo "  OK: See-Through cloned and deps installed"
echo ""

# ---------------------------------------------------------------------------
# 2. Pseudo-label gemini_diverse with See-Through SAM
# ---------------------------------------------------------------------------
echo "[2/5] Pseudo-labeling gemini_diverse with See-Through SAM..."
mkdir -p /workspace/weights

# The seethrough_sam_to_seg.py script auto-downloads the checkpoint from HF
# the first time. After that it loads from --checkpoint path.
python3 scripts/seethrough_sam_to_seg.py \
    --input-dir ./data_cloud/gemini_diverse \
    --checkpoint /workspace/weights/li_sam_iter2.pt \
    --device cuda \
    2>&1 | tee "$LOG_DIR/seethrough_pseudo_label.log"

# Force quality filter to re-run on gemini_diverse with the new labels
rm -f ./data_cloud/gemini_diverse/quality_filter.json
echo ""

# ---------------------------------------------------------------------------
# 3. Marigold enrichment for gemini_diverse (if not already done)
# ---------------------------------------------------------------------------
echo "[3/5] Marigold enrichment for gemini_diverse (only-missing)..."
python3 run_normals_enrich.py \
    --input-dir ./data_cloud/gemini_diverse \
    --only-missing \
    --batch-size 16 \
    2>&1 | tee "$LOG_DIR/enrich.log" || true
echo ""

# ---------------------------------------------------------------------------
# 4. Quality filter — re-run for gemini_diverse (existing filters for other datasets are kept)
# ---------------------------------------------------------------------------
echo "[4/5] Quality filter..."

for ds_dir in ./data_cloud/humanrig ./data_cloud/vroid_cc0 ./data_cloud/meshy_cc0_restructured ./data_cloud/gemini_li_converted ./data_cloud/cvat_annotated ./data_cloud/sora_diverse ./data_cloud/flux_diverse_clean ./data_cloud/gemini_diverse; do
    ds_name=$(basename "$ds_dir")
    if [ -f "$ds_dir/quality_filter.json" ]; then
        echo "  $ds_name: quality_filter.json exists, skipping."
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
# 5. Train using Run 25 config (softening on, gemini_diverse wt 2.0)
# ---------------------------------------------------------------------------
echo "[5/5] Training SEGMENTATION model..."
echo "  Resuming from: $RUN20_CKPT (run 20 best, 0.6485 test mIoU)"
echo "  Config: training/configs/segmentation_a100_run25.yaml (reused)"
echo "  NO --reset-epochs — LR schedule continues from low"
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

cp checkpoints/segmentation/best.pt checkpoints/segmentation/run26_best.pt

python3 -m training.export_onnx \
    --model segmentation \
    --checkpoint checkpoints/segmentation/run26_best.pt \
    --output ./models/onnx/segmentation_run26.onnx \
    2>&1 | tee "$LOG_DIR/export.log"

rclone copy checkpoints/segmentation/run26_best.pt \
    hetzner:strata-training-data/checkpoints_run26_seg/segmentation/ \
    --transfers 4 --fast-list --size-only -P

rclone copy ./models/onnx/segmentation_run26.onnx \
    hetzner:strata-training-data/models/onnx_run26_seg/ \
    --transfers 4 --fast-list --size-only -P

rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run26_seg_${TIMESTAMP}/ \
    --transfers 4 --fast-list -P

echo ""
echo "============================================"
echo "  Run 26 complete!"
echo "  Finished: $(date)"
echo "  Results:"
grep -E "best mIoU|New best" "$LOG_DIR/train.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  To download:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_run26_seg/ /Volumes/TAMWoolff/data/checkpoints_run26_seg/ --transfers 32 --fast-list -P"
echo "============================================"

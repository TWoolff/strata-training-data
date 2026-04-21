#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 28 Seg (A100) — Clean retest of gemini_diverse
#
# Tests: does gemini_diverse help under Run 20's training dynamics?
#
# Key differences from Runs 25/27 (which underperformed):
#   - learning_rate: 1.0e-5 (NOT 5e-6 — matches Run 20)
#   - epochs: 20 (NOT 12 — matches Run 20)
#   - Uses Run 20 self-distillation labels for gemini_diverse (same as Run 25,
#     so Run 28 vs Run 25 is purely a hyperparameter comparison)
#
# The pseudo-label source is re-generated at step [2/5] even if segmentation.png
# exists, because Run 27 overwrote those with See-Through+Li labels. For a clean
# comparison we want Run 20 self-distill labels (the baseline Run 25 used).
#
# Estimated: ~5 hrs total on A100
#   - Data check + Run 20 checkpoint download (~2 min, data mostly on disk)
#   - Re-pseudo-label gemini_diverse with Run 20 (~5 min)
#   - Marigold enrichment (--only-missing, ~0 min if already done)
#   - Quality filter on gemini_diverse (~3 min)
#   - Train 20 epochs at LR 1e-5, resume from Run 20 ckpt, --reset-epochs (~4 hrs)
#   - Export + upload (~10 min)
#
# Prerequisites:
#   - export BUCKET_ACCESS_KEY='...'  BUCKET_SECRET='...'
#   - A100 instance with data from prior runs still on disk, or fresh instance
#     (script will auto-download whatever's missing)
#
# Usage:
#   chmod +x training/run_seg_run28.sh
#   ./training/run_seg_run28.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run28_seg_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 28 Seg"
echo "  Clean retest: gemini_diverse + Run 20 hyperparams"
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
    GPU_MEM=$(python3 -c "import torch; p=torch.cuda.get_device_properties(0); m=getattr(p,'total_memory',getattr(p,'total_mem',0)); print(f'{m/1e9:.0f}GB')")
    echo "  OK: CUDA — $GPU_NAME ($GPU_MEM)"

    # Warn if GPU memory isn't clean — common cause of Run 28 OOM on shared A100
    FREE_MIB=$(python3 -c "import torch; t=torch.cuda.get_device_properties(0); a=torch.cuda.memory_allocated(0); print(int((t.total_memory-a)/1e6))")
    if [ "$FREE_MIB" -lt 30000 ]; then
        echo "  WARN: only ${FREE_MIB} MiB free — zombie CUDA contexts? check: fuser -v /dev/nvidia*"
    fi
fi

RUN20_CKPT="checkpoints/segmentation/run20_best.pt"
if [ ! -f "$RUN20_CKPT" ]; then
    mkdir -p checkpoints/segmentation
    rclone copy hetzner:strata-training-data/checkpoints_run20_seg/segmentation/run20_best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
fi
echo "  OK: Run 20 checkpoint"

# Datasets — download any missing
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
# 1. Re-pseudo-label gemini_diverse with Run 20 (overwrites Run 27's See-Through+Li labels)
# ---------------------------------------------------------------------------
echo "[1/4] Re-pseudo-labeling gemini_diverse with Run 20 checkpoint..."
echo "  (Overwrites any Run 27 See-Through+Li labels — we want Run 20 self-distill"
echo "   for a clean hyperparameter comparison to Run 25.)"
python3 scripts/batch_pseudo_label.py \
    --input-dir ./data_cloud/gemini_diverse \
    --output-dir ./data_cloud/gemini_diverse \
    --checkpoint "$RUN20_CKPT" \
    --device cuda \
    2>&1 | tee "$LOG_DIR/pseudo_label.log"

# Invalidate stale quality filter cache
rm -f ./data_cloud/gemini_diverse/quality_filter.json
echo ""

# ---------------------------------------------------------------------------
# 2. Marigold enrichment (only-missing)
# ---------------------------------------------------------------------------
echo "[2/4] Marigold enrichment (gemini_diverse, only-missing)..."
python3 run_normals_enrich.py \
    --input-dir ./data_cloud/gemini_diverse \
    --only-missing \
    --batch-size 16 \
    2>&1 | tee "$LOG_DIR/enrich.log" || true
echo ""

# ---------------------------------------------------------------------------
# 3. Quality filter
# ---------------------------------------------------------------------------
echo "[3/4] Quality filter..."

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
# 4. Train — Run 20 hyperparameters exactly + gemini_diverse
# ---------------------------------------------------------------------------
echo "[4/4] Training SEGMENTATION model..."
echo "  Config: training/configs/segmentation_a100_run28.yaml"
echo "  Resume: $RUN20_CKPT (Run 20 weights as init)"
echo "  --reset-epochs: fresh 20-epoch LR schedule (1e-5 cosine decay)"
echo ""

python3 -m training.train_segmentation \
    --config training/configs/segmentation_a100_run28.yaml \
    --resume "$RUN20_CKPT" \
    --reset-epochs \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""

# ---------------------------------------------------------------------------
# Export + Upload
# ---------------------------------------------------------------------------
echo "[final] Exporting ONNX + uploading..."

cp checkpoints/segmentation/best.pt checkpoints/segmentation/run28_best.pt

python3 -m training.export_onnx \
    --model segmentation \
    --checkpoint checkpoints/segmentation/run28_best.pt \
    --output ./models/onnx/segmentation_run28.onnx \
    2>&1 | tee "$LOG_DIR/export.log"

rclone copy checkpoints/segmentation/run28_best.pt \
    hetzner:strata-training-data/checkpoints_run28_seg/segmentation/ \
    --transfers 4 --fast-list --size-only -P

rclone copy ./models/onnx/segmentation_run28.onnx \
    hetzner:strata-training-data/models/onnx_run28_seg/ \
    --transfers 4 --fast-list --size-only -P

rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run28_seg_${TIMESTAMP}/ \
    --transfers 4 --fast-list -P

echo ""
echo "============================================"
echo "  Run 28 complete! Finished: $(date)"
echo "  Best mIoU (val):"
grep -E "best mIoU|New best" "$LOG_DIR/train.log" 2>/dev/null | tail -3 || echo "  (check logs)"
echo ""
echo "  Interpretation:"
echo "    >= 0.65  → gemini_diverse genuinely helps (new baseline)"
echo "    0.62-0.65 → matches Run 20 baseline; gemini_diverse net-neutral"
echo "    < 0.62   → data expansion hurts even under proper training"
echo ""
echo "  Download checkpoint: rclone copy hetzner:strata-training-data/checkpoints_run28_seg/ /Volumes/TAMWoolff/data/checkpoints_run28_seg/ --transfers 32 --fast-list -P"
echo "============================================"

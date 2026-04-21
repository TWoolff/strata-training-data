#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 25 Seg (A100) — Fix Run 24 regression
#
# Short continuation run on the SAME A100 that just did Run 24. Skips all
# downloads / pseudo-labeling / Marigold (already done on disk). Just trains
# with boundary softening re-enabled and gemini_diverse weight lowered.
#
# Estimated: ~2-3 hrs total (train only, 12 epochs warm-started from Run 20)
#
# Usage:
#   chmod +x training/run_seg_run25.sh
#   ./training/run_seg_run25.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run25_seg_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 25 Seg"
echo "  Fix: softening on, lower gemini_diverse wt"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight — verify Run 24 artifacts are still on disk
# ---------------------------------------------------------------------------
echo "[pre] Pre-flight checks..."

PREFLIGHT_FAIL=0

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  FAIL: CUDA not available"
    PREFLIGHT_FAIL=1
else
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "  OK: CUDA — $GPU_NAME"
fi

RUN20_CKPT="checkpoints/segmentation/run20_best.pt"
if [ ! -f "$RUN20_CKPT" ]; then
    echo "  Run 20 checkpoint missing — downloading..."
    mkdir -p checkpoints/segmentation
    rclone copy hetzner:strata-training-data/checkpoints_run20_seg/segmentation/run20_best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
fi
echo "  OK: Run 20 checkpoint"

# Bug fix: Run 24 script had wrong tar name (meshy_cc0_restructured.tar instead
# of meshy_cc0_textured_restructured.tar) — silently failed, so Run 24 trained
# WITHOUT 15K meshy examples. Download it now if missing.
if [ ! -d "./data_cloud/meshy_cc0_restructured" ] || [ -z "$(ls ./data_cloud/meshy_cc0_restructured/ 2>/dev/null | head -1)" ]; then
    echo "  meshy_cc0_restructured missing — downloading meshy_cc0_textured_restructured.tar..."
    mkdir -p data/tars
    rclone copy hetzner:strata-training-data/tars/meshy_cc0_textured_restructured.tar \
        ./data/tars/ --transfers 32 --fast-list -P
    tar xf ./data/tars/meshy_cc0_textured_restructured.tar -C ./data_cloud/
    rm -f ./data/tars/meshy_cc0_textured_restructured.tar
    MESHY_COUNT=$(ls -d ./data_cloud/meshy_cc0_restructured/*/ 2>/dev/null | wc -l | tr -d ' ')
    echo "  meshy_cc0_restructured: $MESHY_COUNT examples (restored — was missing from Run 24)"
fi

for ds in humanrig vroid_cc0 meshy_cc0_restructured gemini_li_converted cvat_annotated sora_diverse flux_diverse_clean gemini_diverse; do
    if [ ! -d "./data_cloud/$ds" ] || [ -z "$(ls ./data_cloud/$ds/ 2>/dev/null | head -1)" ]; then
        echo "  FAIL: ./data_cloud/$ds missing or empty"
        PREFLIGHT_FAIL=1
    fi
done
if [ "$PREFLIGHT_FAIL" -eq 0 ]; then
    echo "  OK: all 8 datasets present on disk"
fi

if [ ! -f "./data_cloud/frozen_val_test.json" ]; then
    echo "  FAIL: frozen_val_test.json missing"
    PREFLIGHT_FAIL=1
else
    echo "  OK: frozen val/test splits"
fi

if [ "$PREFLIGHT_FAIL" -ne 0 ]; then
    echo "  Pre-flight failed."
    exit 1
fi
echo ""

# ---------------------------------------------------------------------------
# Train — no download/pseudo-label/marigold steps (already done)
# ---------------------------------------------------------------------------
echo "[1/2] Training SEGMENTATION model..."
echo "  Resuming from: $RUN20_CKPT (run 20 best, 0.6485 test mIoU)"
echo "  Config: training/configs/segmentation_a100_run25.yaml"
echo "  Key changes vs Run 24:"
echo "    - boundary_softening_radius: 2 (was 0)"
echo "    - gemini_diverse weight: 2.0 (was 3.5)"
echo "    - LR: 5e-6 (was 1e-5), epochs: 12 (was 20)"
echo "    - NO --reset-epochs"
echo ""

python3 -m training.train_segmentation \
    --config training/configs/segmentation_a100_run25.yaml \
    --resume "$RUN20_CKPT" \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""

# ---------------------------------------------------------------------------
# Export + Upload
# ---------------------------------------------------------------------------
echo "[2/2] Exporting ONNX + uploading..."

cp checkpoints/segmentation/best.pt checkpoints/segmentation/run25_best.pt

python3 -m training.export_onnx \
    --model segmentation \
    --checkpoint checkpoints/segmentation/run25_best.pt \
    --output ./models/onnx/segmentation_run25.onnx \
    2>&1 | tee "$LOG_DIR/export.log"

rclone copy checkpoints/segmentation/run25_best.pt \
    hetzner:strata-training-data/checkpoints_run25_seg/segmentation/ \
    --transfers 4 --fast-list --size-only -P

rclone copy ./models/onnx/segmentation_run25.onnx \
    hetzner:strata-training-data/models/onnx_run25_seg/ \
    --transfers 4 --fast-list --size-only -P

rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run25_seg_${TIMESTAMP}/ \
    --transfers 4 --fast-list -P

echo ""
echo "============================================"
echo "  Run 25 complete!"
echo "  Finished: $(date)"
echo "  Results:"
grep -E "best mIoU|New best" "$LOG_DIR/train.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  To download:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_run25_seg/ /Volumes/TAMWoolff/data/checkpoints_run25_seg/ --transfers 32 --fast-list -P"
echo "============================================"

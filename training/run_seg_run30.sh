#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 30 Seg (A100) — Ensemble pseudo-labels
#
# Replaces Run 29's single-pass Run-20 pseudo-labels with a 3-stage ensemble:
#   1. Run 20 TTA (4 views averaged)
#   2. SAM 2.1 automatic mask generator → boundary refinement
#   3. Joints-consistency correction (convert_li_labels)
#
# Only variable changed vs Run 29: label source on illustrated datasets.
# Training hyperparameters identical. Expected lift: +0.02-0.04 test mIoU
# over Run 29 (~0.640) → target 0.66+ test.
#
# Estimated: ~7-8 hrs total
#   - SAM 2.1 install + checkpoint download (~5 min)
#   - Joints inference on illustrated datasets (~5 min)
#   - Ensemble pseudo-labeling: gemini (3.4K) + sora (2.9K) + flux (1.6K) at
#     ~1.5-2 sec/img → ~3-4 hrs
#   - Quality filter re-run (~2 min)
#   - Training (~4 hrs)
#   - Export + upload (~10 min)
#
# Prerequisites (assumes fresh A100 with the repo cloned):
#   export BUCKET_ACCESS_KEY='...'  BUCKET_SECRET='...'
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_seg_run30.sh
#   ./training/run_seg_run30.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run30_seg_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 30 Seg"
echo "  Ensemble pseudo-labels (Run 20 TTA + SAM 2.1 + joints)"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
echo "[pre] Pre-flight checks..."

if ! rclone lsd hetzner:strata-training-data/ &>/dev/null; then
    echo "  FAIL: rclone"; exit 1
fi
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  FAIL: CUDA"; exit 1
fi
echo "  OK: rclone + CUDA"

# Checkpoints
mkdir -p checkpoints/segmentation checkpoints/joints data_cloud data/tars

if [ ! -f "checkpoints/segmentation/run20_best.pt" ]; then
    rclone copy hetzner:strata-training-data/checkpoints_run20_seg/segmentation/run20_best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
fi
if [ ! -f "checkpoints/joints/best.pt" ]; then
    rclone copy hetzner:strata-training-data/checkpoints_joints/best.pt \
        ./checkpoints/joints/ --transfers 4 --fast-list -P
fi
echo "  OK: Run 20 seg + joints checkpoints"

# Datasets
download_if_missing() {
    local tar_name="$1"; local extract_dir="$2"
    if [ -d "$extract_dir" ] && [ -n "$(ls "$extract_dir"/ 2>/dev/null | head -1)" ]; then
        return 0
    fi
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

# Sora incremental tars (self-heal from Run 29 experience)
SORA_COUNT=$(ls -d ./data_cloud/sora_diverse/*/ 2>/dev/null | wc -l | tr -d ' ')
if [ "$SORA_COUNT" -lt 2000 ]; then
    echo "  sora_diverse only has $SORA_COUNT — adding sora_diverse_new tars..."
    for t in sora_diverse_new.tar sora_diverse_new2.tar; do
        rclone copy "hetzner:strata-training-data/tars/$t" ./data/tars/ --transfers 32 --fast-list -P 2>&1 | tail -2 || true
        if [ -f "./data/tars/$t" ]; then
            tar xf "./data/tars/$t" -C ./data_cloud/sora_diverse/ --strip-components=1 2>/dev/null || \
                tar xf "./data/tars/$t" -C ./data_cloud/ 2>/dev/null || true
            rm -f "./data/tars/$t"
        fi
    done
    SORA_COUNT=$(ls -d ./data_cloud/sora_diverse/*/ | wc -l | tr -d ' ')
    echo "  sora_diverse now: $SORA_COUNT examples"
fi

# Frozen splits
if [ ! -f "data_cloud/frozen_val_test.json" ]; then
    rclone copy hetzner:strata-training-data/data_cloud/frozen_val_test.json \
        ./data_cloud/ --transfers 4 --fast-list -P
fi
echo "  OK: all datasets"
echo ""

# ---------------------------------------------------------------------------
# 1. SAM 2.1 setup
# ---------------------------------------------------------------------------
echo "[1/6] Installing SAM 2.1 + downloading checkpoint..."

pip install -q sam2 2>&1 | tail -3 || {
    echo "  pip install sam2 failed — falling back to source install"
    cd /workspace
    [ ! -d sam2 ] && git clone https://github.com/facebookresearch/sam2.git
    cd sam2 && pip install -q -e . && cd - >/dev/null
}

mkdir -p /workspace/weights
SAM2_CKPT=/workspace/weights/sam2.1_hiera_large.pt
if [ ! -f "$SAM2_CKPT" ]; then
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(repo_id='facebook/sam2.1-hiera-large', filename='sam2.1_hiera_large.pt')
shutil.copy(path, '$SAM2_CKPT')
print(f'SAM 2.1 large weights ready at $SAM2_CKPT')
"
fi
export SAM2_CHECKPOINT="$SAM2_CKPT"
# Config must be a path relative to the sam2 package root, not just the
# filename. Otherwise Hydra's "pkg://sam2" provider can't resolve it.
export SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_l.yaml

# Verify SAM 2.1 actually loads before starting the ensemble — it's cheap
# and saves us from discovering the failure mid-run after hours of work.
if ! python3 -c "
from sam2.build_sam import build_sam2
build_sam2('$SAM2_CONFIG', '$SAM2_CKPT', device='cuda')
print('SAM 2.1 loaded OK')
" 2>&1 | tee -a "$LOG_DIR/sam_load_check.log" | grep -q "loaded OK"; then
    echo "  WARN: SAM 2.1 load test failed. Ensemble will fall back to TTA + joints only."
else
    echo "  OK: SAM 2.1 ready and loads correctly"
fi
echo ""

# ---------------------------------------------------------------------------
# 2. Joints inference on illustrated datasets (so convert_with_joints path fires)
# ---------------------------------------------------------------------------
echo "[2/6] Joints inference (only-missing) on illustrated datasets..."

for ds in gemini_diverse sora_diverse flux_diverse_clean; do
    python3 scripts/run_joints_inference.py \
        --input-dir "./data_cloud/$ds" \
        --checkpoint ./checkpoints/joints/best.pt \
        --device cuda --only-missing 2>&1 | tee -a "$LOG_DIR/joints.log"
done
echo ""

# ---------------------------------------------------------------------------
# 3. Ensemble pseudo-labeling — the new lever
# ---------------------------------------------------------------------------
echo "[3/6] Ensemble pseudo-labeling on illustrated datasets..."
echo "  Components: Run 20 TTA (4 views) + SAM 2.1 boundary refine + joints correction"
echo ""

for ds in gemini_diverse sora_diverse flux_diverse_clean; do
    echo "  --- $ds ---"
    python3 scripts/ensemble_pseudo_label.py \
        --input-dir "./data_cloud/$ds" \
        --seg-checkpoint ./checkpoints/segmentation/run20_best.pt \
        --device cuda \
        2>&1 | tee -a "$LOG_DIR/ensemble_label.log"
done

# Invalidate stale quality filter caches so the filter re-runs on the new labels
rm -f ./data_cloud/gemini_diverse/quality_filter.json
rm -f ./data_cloud/sora_diverse/quality_filter.json
rm -f ./data_cloud/flux_diverse_clean/quality_filter.json
echo ""

# ---------------------------------------------------------------------------
# 4. Marigold enrichment (only-missing, cheap)
# ---------------------------------------------------------------------------
echo "[4/6] Marigold enrichment (only-missing)..."
for ds in gemini_diverse sora_diverse flux_diverse_clean; do
    python3 run_normals_enrich.py \
        --input-dir "./data_cloud/$ds" \
        --only-missing --batch-size 16 \
        2>&1 | tee -a "$LOG_DIR/enrich.log" || true
done
echo ""

# ---------------------------------------------------------------------------
# 5. Quality filter — same validated combo as Run 29
# ---------------------------------------------------------------------------
echo "[5/6] Quality filter..."

for ds_dir in ./data_cloud/humanrig ./data_cloud/vroid_cc0 ./data_cloud/meshy_cc0_restructured; do
    ds_name=$(basename "$ds_dir")
    if [ -f "$ds_dir/quality_filter.json" ]; then
        echo "  $ds_name: exists, skip."
    else
        python3 scripts/filter_seg_quality.py \
            --data-dirs "$ds_dir" \
            --min-regions 4 --max-single-region 0.70 --min-foreground 0.05 \
            --skip-anatomy \
            2>&1 | tee -a "$LOG_DIR/quality_filter.log"
    fi
done

for ds_dir in ./data_cloud/gemini_li_converted ./data_cloud/cvat_annotated ./data_cloud/sora_diverse ./data_cloud/flux_diverse_clean ./data_cloud/gemini_diverse; do
    ds_name=$(basename "$ds_dir")
    if [ -f "$ds_dir/quality_filter.json" ]; then
        echo "  $ds_name: exists, skip."
    else
        python3 scripts/filter_seg_quality.py \
            --data-dirs "$ds_dir" \
            --min-regions 4 --max-single-region 0.70 --min-foreground 0.05 \
            --drop-head-below-torso \
            --max-bg-bleed 0.10 \
            --min-silhouette-coverage 0.50 \
            2>&1 | tee -a "$LOG_DIR/quality_filter.log"
    fi
done
echo ""

# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------
echo "[6/6] Training..."

python3 -m training.train_segmentation \
    --config training/configs/segmentation_a100_run30.yaml \
    --resume ./checkpoints/segmentation/run20_best.pt \
    --reset-epochs \
    2>&1 | tee "$LOG_DIR/train.log"
echo ""

# ---------------------------------------------------------------------------
# Export + Upload
# ---------------------------------------------------------------------------
echo "[final] Exporting ONNX + uploading..."

cp checkpoints/segmentation/best.pt checkpoints/segmentation/run30_best.pt

python3 -m training.export_onnx \
    --model segmentation \
    --checkpoint checkpoints/segmentation/run30_best.pt \
    --output ./models/onnx/segmentation_run30.onnx \
    2>&1 | tee "$LOG_DIR/export.log"

rclone copy checkpoints/segmentation/run30_best.pt \
    hetzner:strata-training-data/checkpoints_run30_seg/segmentation/ \
    --transfers 4 --fast-list --size-only -P
rclone copy ./models/onnx/segmentation_run30.onnx \
    hetzner:strata-training-data/models/onnx_run30_seg/ \
    --transfers 4 --fast-list --size-only -P
rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run30_seg_${TIMESTAMP}/ \
    --transfers 4 --fast-list -P

echo ""
echo "============================================"
echo "  Run 30 complete! Finished: $(date)"
grep -E "best mIoU|New best" "$LOG_DIR/train.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  Interpretation (vs Run 29 @ 0.6095 val):"
echo "    >= 0.625  → ensemble clearly worked"
echo "    0.610-0.625 → modest gain, still below Run 20"
echo "    <= 0.610  → ensemble didn't help; labels weren't the bottleneck"
echo ""
echo "  Download: rclone copy hetzner:strata-training-data/checkpoints_run30_seg/ /Volumes/TAMWoolff/data/checkpoints_run30_seg/ --transfers 32 --fast-list -P"
echo "============================================"

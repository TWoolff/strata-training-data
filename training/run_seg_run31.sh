#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 31 Seg (A100) — DINOv2-base backbone
#
# Architectural experiment: swap mobilenet_v3_large for DINOv2 ViT-B/14.
# Only variable changed vs Run 20 baseline. Same data, same weights, same
# augmentation, same loss, same boundary softening (radius=2).
#
# Hypothesis: DINOv2's self-supervised features (LVD-142M, no domain bias)
# transfer better to illustrated chars than ImageNet CNN backbones.
# Unlike Run 23 (Pascal-Person-Part) which had human-photo bias, DINOv2 was
# trained on diverse images.
#
# Expected lift: +0.03-0.05 test mIoU over Run 20 (0.6485) → 0.68+ test.
# Cost: ~$5-7 on A100 40GB at $0.6/hr.
#
# Estimated: ~5-6 hrs total
#   - Setup + data download (~30 min, mostly cached if reusing instance)
#   - Quality filter (~3 min)
#   - Training (~4-5 hrs, larger model than Run 20)
#   - Export + upload (~10 min)
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'  BUCKET_SECRET='...'
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_seg_run31.sh
#   ./training/run_seg_run31.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run31_seg_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 31 Seg"
echo "  DINOv2-base backbone (architectural swap)"
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
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
GPU_MEM=$(python3 -c "import torch; p=torch.cuda.get_device_properties(0); m=getattr(p,'total_memory',getattr(p,'total_mem',0)); print(f'{m/1e9:.0f}GB')")
echo "  OK: rclone + CUDA — $GPU_NAME ($GPU_MEM)"

# Verify transformers is available (DINOv2 dependency)
if ! python3 -c "from transformers import AutoModel" 2>/dev/null; then
    echo "  Installing transformers..."
    pip install -q transformers || { echo "  FAIL: transformers install"; exit 1; }
fi
echo "  OK: transformers available"

mkdir -p checkpoints/segmentation data_cloud data/tars

# ---------------------------------------------------------------------------
# 1. Download datasets (Run 20 mix — no gemini_diverse)
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Downloading datasets (Run 20 mix)..."

download_if_missing() {
    local tar_name="$1"; local extract_dir="$2"
    if [ -d "$extract_dir" ] && [ -n "$(ls "$extract_dir"/ 2>/dev/null | head -1)" ]; then
        echo "  $extract_dir: already extracted"
        return 0
    fi
    echo "  Downloading $tar_name..."
    rclone copy "hetzner:strata-training-data/tars/$tar_name" ./data/tars/ \
        --transfers 32 --fast-list -P 2>&1 | tail -2
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

# Sora incremental tars (in case base sora_diverse.tar is sparse)
SORA_COUNT=$(ls -d ./data_cloud/sora_diverse/*/ 2>/dev/null | wc -l | tr -d ' ')
if [ "$SORA_COUNT" -lt 2000 ]; then
    echo "  sora_diverse only has $SORA_COUNT — adding sora_diverse_new tars..."
    for t in sora_diverse_new.tar sora_diverse_new2.tar; do
        rclone copy "hetzner:strata-training-data/tars/$t" ./data/tars/ \
            --transfers 32 --fast-list -P 2>&1 | tail -2 || true
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
echo "  OK: all datasets ready"

# ---------------------------------------------------------------------------
# 2. Pre-cache DINOv2 weights (catches network errors early, not mid-training)
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Pre-caching DINOv2-base weights from HuggingFace..."

python3 -c "
from transformers import AutoModel
print('Downloading facebook/dinov2-base...')
model = AutoModel.from_pretrained('facebook/dinov2-base')
n_params = sum(p.numel() for p in model.parameters())
print(f'DINOv2-base loaded: {n_params/1e6:.1f}M params')
" 2>&1 | tee "$LOG_DIR/dinov2_download.log"

# Sanity-check our segmentation_model.py supports the new backbone
echo ""
echo "  Verifying SegmentationModel accepts dinov2_base backbone..."
python3 -c "
import torch
from training.models.segmentation_model import SegmentationModel
model = SegmentationModel(num_classes=22, backbone='dinov2_base', pretrained_backbone=True)
n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model built: {n_params/1e6:.1f}M total, {n_trainable/1e6:.1f}M trainable')
# Forward pass — use eval mode for bs=1 (BatchNorm in ASPP needs bs>1 in train mode)
model.eval()
x = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    out = model(x)
for k, v in out.items():
    print(f'  {k}: {tuple(v.shape)}')
" 2>&1 | tee -a "$LOG_DIR/dinov2_download.log"

# ---------------------------------------------------------------------------
# 3. Quality filter (Run 20 had this run already; cheap if cached)
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Quality filter..."

for ds_dir in ./data_cloud/humanrig ./data_cloud/vroid_cc0 ./data_cloud/meshy_cc0_restructured; do
    ds_name=$(basename "$ds_dir")
    if [ -f "$ds_dir/quality_filter.json" ]; then
        echo "  $ds_name: cached"
    else
        python3 scripts/filter_seg_quality.py \
            --data-dirs "$ds_dir" \
            --min-regions 4 --max-single-region 0.70 --min-foreground 0.05 \
            --skip-anatomy \
            2>&1 | tee -a "$LOG_DIR/quality_filter.log"
    fi
done

for ds_dir in ./data_cloud/gemini_li_converted ./data_cloud/cvat_annotated ./data_cloud/sora_diverse ./data_cloud/flux_diverse_clean; do
    ds_name=$(basename "$ds_dir")
    if [ -f "$ds_dir/quality_filter.json" ]; then
        echo "  $ds_name: cached"
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
# 4. Train (no --resume — DINOv2 is a different architecture, train from scratch)
# ---------------------------------------------------------------------------
echo "[4/4] Training (DINOv2-base, ~4-5 hrs)..."

python3 -m training.train_segmentation \
    --config training/configs/segmentation_a100_run31.yaml \
    2>&1 | tee "$LOG_DIR/train.log"
echo ""

# ---------------------------------------------------------------------------
# Export + Upload
# ---------------------------------------------------------------------------
echo "[final] Exporting ONNX + uploading..."

cp checkpoints/segmentation/best.pt checkpoints/segmentation/run31_best.pt

# Export with --backbone flag so the export script reconstructs the right architecture
python3 -m training.export_onnx \
    --model segmentation \
    --checkpoint checkpoints/segmentation/run31_best.pt \
    --backbone dinov2_base \
    --output ./models/onnx/segmentation_run31.onnx \
    2>&1 | tee "$LOG_DIR/export.log" || {
        echo "  WARN: ONNX export failed. Saving .pt only — can re-export later from local."
    }

# Upload checkpoint and (if successful) ONNX
rclone copy checkpoints/segmentation/run31_best.pt \
    hetzner:strata-training-data/checkpoints_run31_seg/segmentation/ \
    --transfers 4 --fast-list --size-only -P

if [ -f ./models/onnx/segmentation_run31.onnx ]; then
    rclone copy ./models/onnx/segmentation_run31.onnx \
        hetzner:strata-training-data/models/onnx_run31_seg/ \
        --transfers 4 --fast-list --size-only -P
fi

rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run31_seg_${TIMESTAMP}/ \
    --transfers 4 --fast-list -P

echo ""
echo "============================================"
echo "  Run 31 complete! Finished: $(date)"
grep -E "best mIoU|New best" "$LOG_DIR/train.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  Interpretation (vs Run 20 @ 0.6485 test):"
echo "    >= 0.69  → DINOv2 wins. Ship it. Funding-grade improvement."
echo "    0.66-0.69 → Modest gain. Probably ship."
echo "    0.62-0.66 → Same as Run 20. Ceiling is firm. Ship Run 20."
echo "    < 0.60  → DINOv2 hurts (like Run 23). Adjacent-domain priors strike again."
echo ""
echo "  Download: rclone copy hetzner:strata-training-data/checkpoints_run31_seg/ /Volumes/TAMWoolff/data/checkpoints_run31_seg/ --transfers 32 --fast-list -P"
echo "============================================"

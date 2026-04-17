#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 23 Combined (A100)
#
# Three sequential stages in a single A100 boot:
#   A) Pascal-Person-Part anatomy pretrain (DeepLabV3-ResNet50, ~1.5h)
#        → produces backbone.pth with real-human body-part priors
#   B) Strata seg fine-tune with anatomy init (~3-4h)
#        → Run 20 data mix, ResNet-50 backbone from Stage A, 22-class head fresh
#   C) Texture inpainting v4 with LPIPS perceptual loss (~2.5-3h)
#        → resume from v3 best (0.1282 val/l1), add x_0-decoded LPIPS training
#
# Estimated total: ~7-8h on A100 40GB (download + all three stages + upload)
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_a100_run23_combined.sh
#   ./training/run_a100_run23_combined.sh
#
# Re-entrant: each stage skips if its output is already on the bucket, so
# re-running after a mid-failure picks up where it left off.
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run23_combined_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 23 Combined"
echo "  Stage A: Pascal-Person-Part anatomy pretrain"
echo "  Stage B: Seg fine-tune (ResNet-50 + anatomy init)"
echo "  Stage C: Texture inpaint v4 (LPIPS perceptual)"
echo "  Started: $(date)"
echo "  Logs:    $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
echo "[pre] Pre-flight checks..."
PREFLIGHT_FAIL=0

if ! rclone lsd hetzner:strata-training-data/ &>/dev/null; then
    echo "  FAIL: rclone cannot connect to Hetzner bucket"
    PREFLIGHT_FAIL=1
else
    echo "  OK: rclone"
fi

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  FAIL: CUDA not available"
    PREFLIGHT_FAIL=1
else
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEM=$(python3 -c "import torch; p=torch.cuda.get_device_properties(0); m=getattr(p,'total_memory',getattr(p,'total_mem',0)); print(f'{m/1e9:.0f}GB')")
    echo "  OK: CUDA — $GPU_NAME ($GPU_MEM)"
fi

if ! python3 -c "from scipy.ndimage import gaussian_filter; from scipy.io import loadmat" 2>/dev/null; then
    echo "  FAIL: scipy missing (needed for boundary softening + Pascal-Part .mat parsing)"
    PREFLIGHT_FAIL=1
else
    echo "  OK: scipy"
fi

if ! python3 -c "import torchvision; from torchvision.models import vgg16" 2>/dev/null; then
    echo "  FAIL: torchvision missing (needed for VGG16 LPIPS + DeepLabV3-ResNet50)"
    PREFLIGHT_FAIL=1
else
    echo "  OK: torchvision"
fi

if [ "$PREFLIGHT_FAIL" -ne 0 ]; then
    echo "  Pre-flight failed."
    exit 1
fi
echo ""

# ---------------------------------------------------------------------------
# Step 0 — Download shared artifacts (frozen splits)
# ---------------------------------------------------------------------------
echo "[0/6] Downloading shared artifacts..."
mkdir -p data_cloud checkpoints/anatomy_init checkpoints_prev

if [ ! -f data_cloud/frozen_val_test.json ]; then
    rclone copy hetzner:strata-training-data/data_cloud/frozen_val_test.json \
        ./data_cloud/ --transfers 4 --fast-list -P 2>&1 | tail -2
fi
echo "  frozen_val_test.json: OK"
echo ""

# ---------------------------------------------------------------------------
# STAGE A — Pascal-Person-Part anatomy pretrain
# ---------------------------------------------------------------------------
echo "============================================"
echo "  STAGE A — Pascal-Person-Part pretrain"
echo "  $(date)"
echo "============================================"

ANATOMY_CKPT="./checkpoints/anatomy_init/backbone.pth"
BUCKET_ANATOMY_KEY="anatomy_init/backbone_resnet50_pascal_part.pth"

# Skip Stage A if the backbone is already on the bucket (reuse across runs)
if rclone lsf "hetzner:strata-training-data/${BUCKET_ANATOMY_KEY}" &>/dev/null \
   && [ -n "$(rclone lsf "hetzner:strata-training-data/${BUCKET_ANATOMY_KEY}" 2>/dev/null)" ]; then
    echo "[A] Anatomy backbone already on bucket — downloading instead of retraining"
    rclone copyto "hetzner:strata-training-data/${BUCKET_ANATOMY_KEY}" "$ANATOMY_CKPT" \
        --transfers 4 -P 2>&1 | tail -2
else
    echo "[A] Training DeepLabV3-ResNet50 on Pascal-Person-Part..."
    python3 -m training.pretrain_pascal_person_part \
        --output "$ANATOMY_CKPT" \
        --data-root ./data_cloud/pascal_person_part \
        --epochs 40 \
        --batch-size 16 \
        --num-workers 8 \
        2>&1 | tee "$LOG_DIR/stage_a_pretrain.log"

    # Sanity-check: backbone file exists and is non-trivial
    if [ ! -s "$ANATOMY_CKPT" ]; then
        echo "  FAIL: anatomy backbone not produced at $ANATOMY_CKPT"
        exit 1
    fi
    SIZE=$(du -m "$ANATOMY_CKPT" | awk '{print $1}')
    echo "  Anatomy backbone: ${SIZE}MB"

    # Upload for reuse across future runs
    echo "  Uploading anatomy backbone to bucket..."
    rclone copyto "$ANATOMY_CKPT" "hetzner:strata-training-data/${BUCKET_ANATOMY_KEY}" \
        --transfers 4 -P 2>&1 | tail -2
fi
echo ""

# ---------------------------------------------------------------------------
# STAGE B — Strata seg fine-tune with anatomy init
# ---------------------------------------------------------------------------
echo "============================================"
echo "  STAGE B — Strata seg fine-tune (Run 23)"
echo "  $(date)"
echo "============================================"

# Download seg datasets (reuses download_tar pattern from run_seg_run22.sh)
echo "[B/1] Downloading seg datasets..."
mkdir -p data/tars

download_tar() {
    local tar_name="$1"
    local extract_dir="$2"
    if [ -d "$extract_dir" ] && [ "$(ls "$extract_dir"/ 2>/dev/null | head -1)" ]; then
        local count=$(ls -d "$extract_dir"/*/ 2>/dev/null | wc -l | tr -d ' ')
        echo "  $(basename "$extract_dir"): $count examples (exists)"
        return 0
    fi
    echo "  Downloading $tar_name..."
    rclone copy "hetzner:strata-training-data/tars/$tar_name" ./data/tars/ \
        --transfers 32 --fast-list -P 2>&1 | tail -2
    if [ -f "./data/tars/$tar_name" ]; then
        tar xf "./data/tars/$tar_name" -C ./data_cloud/
        rm -f "./data/tars/$tar_name"
    else
        echo "  WARN: $tar_name not found in bucket"
    fi
    local count=$(ls -d "$extract_dir"/*/ 2>/dev/null | wc -l | tr -d ' ')
    echo "  $(basename "$extract_dir"): $count examples"
}

download_tar "humanrig.tar" "./data_cloud/humanrig"
download_tar "vroid_cc0.tar" "./data_cloud/vroid_cc0"
download_tar "meshy_cc0_textured_restructured.tar" "./data_cloud/meshy_cc0_restructured"
download_tar "gemini_li_converted.tar" "./data_cloud/gemini_li_converted"
download_tar "cvat_annotated.tar" "./data_cloud/cvat_annotated"
download_tar "sora_diverse.tar" "./data_cloud/sora_diverse"
download_tar "flux_diverse_clean.tar" "./data_cloud/flux_diverse_clean"

echo ""
echo "[B/2] Training seg (15 epochs, ResNet-50 + anatomy init)..."
echo "  Config: training/configs/segmentation_a100_run23.yaml"
echo "  Backbone init: $ANATOMY_CKPT"

# Note: no --resume — we're initializing head from scratch with anatomy backbone.
# The backbone_weights_path config field handles the backbone init.
python3 -m training.train_segmentation \
    --config training/configs/segmentation_a100_run23.yaml \
    2>&1 | tee "$LOG_DIR/stage_b_seg_train.log"

SEG_BEST="./checkpoints/segmentation_run23/best.pt"
if [ ! -s "$SEG_BEST" ]; then
    echo "  FAIL: seg best checkpoint not produced"
    exit 1
fi

# Save with a namespaced filename so it can coexist on the bucket
cp "$SEG_BEST" "./checkpoints/segmentation_run23/run23_best.pt"

echo ""
echo "[B/3] Exporting ONNX..."
python3 -m training.export_onnx \
    --model segmentation \
    --checkpoint "./checkpoints/segmentation_run23/run23_best.pt" \
    --output "./models/onnx/segmentation_run23.onnx" \
    2>&1 | tee "$LOG_DIR/stage_b_export.log" || echo "  WARN: ONNX export failed — continuing"

echo ""
echo "[B/4] Uploading seg checkpoint..."
rclone copy ./checkpoints/segmentation_run23/run23_best.pt \
    hetzner:strata-training-data/checkpoints_run23_seg/segmentation/ \
    --transfers 4 --fast-list --size-only -P 2>&1 | tail -2

if [ -f "./models/onnx/segmentation_run23.onnx" ]; then
    rclone copy ./models/onnx/segmentation_run23.onnx \
        hetzner:strata-training-data/models/onnx_run23_seg/ \
        --transfers 4 --fast-list --size-only -P 2>&1 | tail -2
fi
echo ""

# ---------------------------------------------------------------------------
# STAGE C — Texture inpainting v4 with LPIPS
# ---------------------------------------------------------------------------
echo "============================================"
echo "  STAGE C — Texture inpaint v4 (LPIPS)"
echo "  $(date)"
echo "============================================"

# Download texture pairs (front only, has real geometry maps)
echo "[C/1] Downloading texture_pairs_front.tar..."
if [ ! -d data_cloud/texture_pairs_front ] || [ -z "$(ls -d data_cloud/texture_pairs_front/*/ 2>/dev/null | head -1)" ]; then
    if [ ! -f data_cloud/texture_pairs_front.tar ]; then
        rclone copyto hetzner:strata-training-data/texture_pairs_front.tar \
            ./data_cloud/texture_pairs_front.tar --no-check-dest -P 2>&1 | tail -2
    fi
    rm -rf data_cloud/texture_pairs_front
    tar xf data_cloud/texture_pairs_front.tar -C data_cloud/
    if [ -d data_cloud/texture_pairs ] && [ ! -d data_cloud/texture_pairs_front ]; then
        mv data_cloud/texture_pairs data_cloud/texture_pairs_front
    fi
    rm -f data_cloud/texture_pairs_front.tar
fi
N_PAIRS=$(find data_cloud/texture_pairs_front -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
echo "  texture_pairs_front: $N_PAIRS pairs"

# Download v3 best checkpoint
echo "[C/2] Downloading v3 best checkpoint..."
rclone copy hetzner:strata-training-data/checkpoints_texture_inpaint_v3/best/ \
    ./checkpoints_prev/ --transfers 16 -P 2>&1 | tail -2

if [ -d checkpoints_prev/controlnet ]; then
    echo "  OK: controlnet subdir at checkpoints_prev/controlnet"
elif [ -f checkpoints_prev/diffusion_pytorch_model.safetensors ]; then
    echo "  Found flat checkpoint — moving into controlnet/"
    mkdir -p checkpoints_prev/controlnet
    mv checkpoints_prev/*.safetensors checkpoints_prev/*.json checkpoints_prev/controlnet/ 2>/dev/null || true
fi

# Pre-download SD Inpainting base model
echo "[C/3] Pre-downloading SD 1.5 Inpainting..."
python3 -c "
from transformers import CLIPTokenizer, CLIPTextModel
_ = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-inpainting', subfolder='tokenizer')
_ = CLIPTextModel.from_pretrained('runwayml/stable-diffusion-inpainting', subfolder='text_encoder')
print('  OK: SD 1.5 Inpainting cached')
" 2>&1

echo ""
echo "[C/4] Training ControlNet v4..."
python3 -m training.train_texture_inpainting \
    --config training/configs/texture_inpainting_a100_v4.yaml \
    --resume checkpoints_prev/controlnet \
    2>&1 | tee "$LOG_DIR/stage_c_inpaint_train.log"

echo ""
echo "[C/5] Uploading inpaint v4 checkpoint..."
INPAINT_V4_DIR="./checkpoints/texture_inpainting_controlnet_v4"
if [ -d "$INPAINT_V4_DIR/best" ]; then
    rclone copy "$INPAINT_V4_DIR/best" \
        hetzner:strata-training-data/checkpoints_texture_inpaint_v4/best/ \
        --transfers 16 -P 2>&1 | tail -2
    echo "  OK: inpaint v4 best uploaded"
else
    echo "  WARN: no best checkpoint found at $INPAINT_V4_DIR/best"
fi
echo ""

# ---------------------------------------------------------------------------
# Final — upload all logs and summary
# ---------------------------------------------------------------------------
echo "============================================"
echo "  Run 23 Combined — complete"
echo "  Finished: $(date)"
echo "============================================"

rclone copy "$LOG_DIR/" \
    "hetzner:strata-training-data/logs/run23_combined_${TIMESTAMP}/" \
    --transfers 4 --fast-list -P 2>&1 | tail -2

echo ""
echo "Summary:"
echo "---"
grep -E "Best val mIoU|best mIoU" "$LOG_DIR/stage_a_pretrain.log" 2>/dev/null | tail -1 || echo "  Stage A: (check logs)"
grep -E "best mIoU|mIoU=" "$LOG_DIR/stage_b_seg_train.log" 2>/dev/null | tail -2 || echo "  Stage B: (check logs)"
grep -E "Best val|val/l1=|val/lpips=" "$LOG_DIR/stage_c_inpaint_train.log" 2>/dev/null | tail -3 || echo "  Stage C: (check logs)"
echo "---"
echo ""
echo "Download to Mac:"
echo "  rclone copy hetzner:strata-training-data/checkpoints_run23_seg/ /Volumes/TAMWoolff/data/checkpoints_run23_seg/ --transfers 32 --fast-list -P"
echo "  rclone copy hetzner:strata-training-data/checkpoints_texture_inpaint_v4/ /Volumes/TAMWoolff/data/checkpoints_texture_inpaint_v4/ --transfers 32 --fast-list -P"
echo "============================================"

#!/usr/bin/env bash
# =============================================================================
# Strata — Texture Inpainting v3 (with real geometry maps)
#
# Retrains the v2 ControlNet with real position/normal maps (1,244 front pairs).
# Fine-tunes from v2 best checkpoint at lower LR.
# Est. time: ~2-3h on A100 40GB
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/texture_inpaint_v3_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata — Texture Inpainting v3"
echo "  Real geometry maps + fine-tune from v2"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
echo "[pre] Pre-flight checks..."
if ! rclone lsd hetzner:strata-training-data/ &>/dev/null; then
    echo "  FAIL: rclone cannot connect to Hetzner bucket"
    exit 1
fi
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  FAIL: CUDA not available"
    exit 1
fi
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
GPU_MEM=$(python3 -c "import torch; p=torch.cuda.get_device_properties(0); m=getattr(p,'total_memory',getattr(p,'total_mem',0)); print(f'{m/1e9:.0f}GB')")
echo "  OK: CUDA — $GPU_NAME ($GPU_MEM)"

# ---------------------------------------------------------------------------
# Step 1: Download texture pairs (with geometry maps)
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Downloading texture_pairs_front.tar (4.9G, has real geometry maps)..."
mkdir -p data_cloud
if [[ ! -f data_cloud/texture_pairs_front.tar ]]; then
    rclone copyto hetzner:strata-training-data/texture_pairs_front.tar \
        ./data_cloud/texture_pairs_front.tar --no-check-dest -P 2>&1 | tail -3
fi
echo "  Extracting..."
rm -rf data_cloud/texture_pairs_front
tar xf data_cloud/texture_pairs_front.tar -C data_cloud/
# The tar extracts into `texture_pairs/`, rename for clarity
if [[ -d data_cloud/texture_pairs && ! -d data_cloud/texture_pairs_front ]]; then
    mv data_cloud/texture_pairs data_cloud/texture_pairs_front
fi
N_PAIRS=$(find data_cloud/texture_pairs_front -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
echo "  OK: $N_PAIRS pairs"

# ---------------------------------------------------------------------------
# Step 2: Download v2 checkpoint to fine-tune from
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Downloading v2 best checkpoint..."
mkdir -p checkpoints_prev
rclone copy hetzner:strata-training-data/checkpoints_texture_inpaint_v2/best/ \
    ./checkpoints_prev/ --transfers 16 -P 2>&1 | tail -3

# The controlnet subdir is what train_texture_inpainting.py's --resume expects
if [[ -d checkpoints_prev/controlnet ]]; then
    echo "  OK: v2 checkpoint ready at checkpoints_prev/controlnet"
else
    echo "  WARN: no controlnet subdir found; checkpoint structure:"
    ls checkpoints_prev/ || true
    # Try to locate the model
    if [[ -f checkpoints_prev/diffusion_pytorch_model.safetensors ]]; then
        echo "  Found flat checkpoint — moving into controlnet/"
        mkdir -p checkpoints_prev/controlnet
        mv checkpoints_prev/*.safetensors checkpoints_prev/*.json checkpoints_prev/controlnet/ 2>/dev/null || true
    fi
fi

# ---------------------------------------------------------------------------
# Step 3: Pre-download SD Inpainting
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Pre-downloading SD 1.5 Inpainting..."
python3 -c "
from transformers import CLIPTokenizer, CLIPTextModel
_ = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-inpainting', subfolder='tokenizer')
_ = CLIPTextModel.from_pretrained('runwayml/stable-diffusion-inpainting', subfolder='text_encoder')
print('  OK: SD 1.5 Inpainting cached')
" 2>&1

# ---------------------------------------------------------------------------
# Step 4: Train (resume from v2)
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Training ControlNet v3..."
python3 -m training.train_texture_inpainting \
    --config training/configs/texture_inpainting_a100_v3.yaml \
    --resume checkpoints_prev/controlnet \
    2>&1 | tee "$LOG_DIR/train.log"

# ---------------------------------------------------------------------------
# Upload checkpoints
# ---------------------------------------------------------------------------
echo ""
echo "[final] Uploading checkpoints..."
CKPT_DIR="./checkpoints/texture_inpainting_controlnet_v3"
if [[ -d "$CKPT_DIR/best" ]]; then
    rclone copy "$CKPT_DIR/best" \
        "hetzner:strata-training-data/checkpoints_texture_inpaint_v3/best/" \
        --transfers 16 -P 2>&1 | tail -3
    echo "  OK: best checkpoint uploaded"
fi

rclone copy "$LOG_DIR" \
    "hetzner:strata-training-data/logs/texture_inpaint_v3_${TIMESTAMP}/" \
    --transfers 8 -P 2>&1 | tail -1

echo ""
echo "============================================"
echo "  Done! Finished: $(date)"
echo "============================================"

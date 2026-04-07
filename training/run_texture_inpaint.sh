#!/usr/bin/env bash
# =============================================================================
# Strata Training — ControlNet UV Texture Inpainting (A100)
#
# Fine-tunes a ControlNet conditioned on UV position + normal maps, using
# SD 1.5 Inpainting as frozen base.  Trains on texture pairs generated from
# Meshy CC0 + VRoid CC0 characters.
#
# Estimated: ~4-6 hrs on A100 40GB (full run), ~2h (lean)
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_texture_inpaint.sh
#   ./training/run_texture_inpaint.sh [--lean]
# =============================================================================
set -euo pipefail

LEAN=false
if [[ "${1:-}" == "--lean" ]]; then
    LEAN=true
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/texture_inpaint_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

if $LEAN; then
    CONFIG="training/configs/texture_inpainting_a100_lean.yaml"
    echo "============================================"
    echo "  Strata — Texture Inpainting (LEAN)"
else
    CONFIG="training/configs/texture_inpainting_a100.yaml"
    echo "============================================"
    echo "  Strata — Texture Inpainting (FULL)"
fi
echo "  Started: $(date)"
echo "  Config: $CONFIG"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
echo "[pre] Pre-flight checks..."

if ! rclone lsd hetzner:strata-training-data/ &>/dev/null; then
    echo "  FAIL: rclone cannot connect to Hetzner bucket"
    exit 1
fi
echo "  OK: rclone bucket connection"

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  FAIL: CUDA not available"
    exit 1
fi
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
GPU_MEM=$(python3 -c "import torch; p=torch.cuda.get_device_properties(0); m=getattr(p,'total_memory',getattr(p,'total_mem',0)); print(f'{m/1e9:.0f}GB')")
echo "  OK: CUDA — $GPU_NAME ($GPU_MEM)"

# Check diffusers
python3 -c "import diffusers; print(f'  OK: diffusers {diffusers.__version__}')"

# Check transformers (needed for CLIP text encoder)
python3 -c "import transformers; print(f'  OK: transformers {transformers.__version__}')"

echo ""

# ---------------------------------------------------------------------------
# Step 1: Download texture pairs
# ---------------------------------------------------------------------------
echo "[1/4] Downloading texture pair data..."

mkdir -p data_cloud
rclone copy hetzner:strata-training-data/texture_pairs.tar ./data_cloud/ \
    --transfers 16 --fast-list --size-only -P 2>&1 | tail -3

if [[ -f data_cloud/texture_pairs.tar ]]; then
    echo "  Extracting texture_pairs.tar..."
    tar xf data_cloud/texture_pairs.tar -C data_cloud/
    echo "  OK: $(find data_cloud/texture_pairs -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l) character pairs"
else
    echo "  WARN: texture_pairs.tar not found in bucket"
    echo "  Checking if texture_pairs/ directory exists in bucket..."
    rclone copy hetzner:strata-training-data/texture_pairs/ ./data_cloud/texture_pairs/ \
        --transfers 32 --checkers 64 --fast-list --size-only -P 2>&1 | tail -3
    N_PAIRS=$(find data_cloud/texture_pairs -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "  OK: $N_PAIRS character pairs"
fi

echo ""

# ---------------------------------------------------------------------------
# Step 2: Pre-download SD Inpainting model
# ---------------------------------------------------------------------------
echo "[2/4] Pre-downloading SD Inpainting model..."

python3 -c "
from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPTokenizer, CLIPTextModel

# This downloads and caches the model
print('  Downloading runwayml/stable-diffusion-inpainting...')
_ = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-inpainting', subfolder='tokenizer')
_ = CLIPTextModel.from_pretrained('runwayml/stable-diffusion-inpainting', subfolder='text_encoder')
print('  OK: SD 1.5 Inpainting cached')
" 2>&1

echo ""

# ---------------------------------------------------------------------------
# Step 3: Train
# ---------------------------------------------------------------------------
echo "[3/4] Training ControlNet..."
echo "  Config: $CONFIG"

python3 -m training.train_texture_inpainting \
    --config "$CONFIG" \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""

# ---------------------------------------------------------------------------
# Step 4: Upload checkpoints
# ---------------------------------------------------------------------------
echo "[4/4] Uploading checkpoints..."

CKPT_DIR="./checkpoints/texture_inpainting_controlnet"

if [[ -d "$CKPT_DIR/best" ]]; then
    rclone copy "$CKPT_DIR/best" \
        "hetzner:strata-training-data/checkpoints_texture_inpaint/best/" \
        --transfers 16 --fast-list -P 2>&1 | tail -3
    echo "  OK: best checkpoint uploaded"
fi

if [[ -d "$CKPT_DIR/latest" ]]; then
    rclone copy "$CKPT_DIR/latest" \
        "hetzner:strata-training-data/checkpoints_texture_inpaint/latest/" \
        --transfers 16 --fast-list -P 2>&1 | tail -3
    echo "  OK: latest checkpoint uploaded"
fi

# Upload logs
rclone copy "$LOG_DIR" \
    "hetzner:strata-training-data/logs/texture_inpaint_${TIMESTAMP}/" \
    --transfers 8 -P 2>&1 | tail -1

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Checkpoints: $CKPT_DIR"
echo "  Logs: $LOG_DIR"
echo "  Finished: $(date)"
echo "============================================"

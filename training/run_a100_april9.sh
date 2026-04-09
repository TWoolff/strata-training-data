#!/usr/bin/env bash
# =============================================================================
# Strata A100 Run — April 9, 2026
#
# Part 1: See-Through layer decomposition on 7 benchmark chars (~30 min)
# Part 2: Texture inpainting ControlNet v2 — 4,135 pairs, 100 epochs (~3-4 hrs)
#
# Storage: 80 GB recommended
# Time: ~4.5 hrs total
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_a100_april9.sh
#   ./training/run_a100_april9.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/a100_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata — A100 Run (April 9)"
echo "  Started: $(date)"
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
echo ""

# ============================================================================
# PART 1: See-Through Layer Decomposition
# ============================================================================
echo "======================================================="
echo "  PART 1: See-Through Layer Decomposition"
echo "======================================================="
echo ""

# Install See-Through dependencies
echo "[1.1] Installing See-Through dependencies..."
pip install -q einops kornia omegaconf psd-tools pycocotools natsort 2>&1 | tail -3

# Clone See-Through repo
if [[ ! -d /workspace/see-through ]]; then
    echo "[1.2] Cloning See-Through repo..."
    cd /workspace
    git clone --depth 1 https://github.com/shitagaki-lab/see-through.git
    cd /workspace/strata-training-data
else
    echo "[1.2] See-Through repo already exists"
fi

# Install See-Through as editable packages
echo "[1.3] Installing See-Through modules..."
pip install -q -e /workspace/see-through/common 2>&1 | tail -3
pip install -q -e /workspace/see-through/annotators 2>&1 | tail -5 || true

# Download benchmark images
echo "[1.4] Downloading benchmark test images..."
mkdir -p data_cloud/gemini_benchmark
rclone copy hetzner:strata-training-data/gemini_benchmark/ ./data_cloud/gemini_benchmark/ \
    --transfers 16 --fast-list --size-only -P 2>&1 | tail -3

N_BENCH=$(find data_cloud/gemini_benchmark -name "*.png" ! -name "._*" 2>/dev/null | wc -l)
if [[ "$N_BENCH" -lt 3 ]]; then
    echo "  WARN: Only $N_BENCH benchmark images. Skipping See-Through test."
    SKIP_SEETHROUGH=true
else
    echo "  OK: $N_BENCH benchmark images"
    SKIP_SEETHROUGH=false
fi

if [[ "${SKIP_SEETHROUGH}" == "false" ]]; then
    echo "[1.5] Running See-Through on benchmark characters..."

    python3 -c "
import sys, os, time
sys.path.insert(0, '/workspace/see-through/common')

from pathlib import Path

image_dir = Path('data_cloud/gemini_benchmark')
output_dir = Path('output/seethrough_test')
output_dir.mkdir(parents=True, exist_ok=True)

images = sorted([p for p in image_dir.glob('*.png') if not p.name.startswith('._')])
print(f'Found {len(images)} benchmark images')

# Import See-Through inference (must chdir for relative imports)
os.chdir('/workspace/see-through/common')
from utils.inference_utils import apply_layerdiff, apply_marigold, further_extr
os.chdir('/workspace/strata-training-data')

save_dir = str(output_dir)
total_start = time.time()

for i, img_path in enumerate(images):
    name = img_path.stem
    print(f'\n[{i+1}/{len(images)}] Processing {name}...')
    t0 = time.time()

    try:
        print('  Running LayerDiff3D...')
        apply_layerdiff(
            str(img_path),
            'layerdifforg/seethroughv0.0.2_layerdiff3d',
            save_dir=save_dir,
            seed=42,
            resolution=1280,
            disable_progressbar=False,
            num_inference_steps=30,
            group_offload=True,
        )

        print('  Running Marigold depth...')
        apply_marigold(
            str(img_path),
            '24yearsold/seethroughv0.0.1_marigold',
            save_dir=save_dir,
            seed=42,
            resolution=720,
            disable_progressbar=False,
            group_offload=True,
        )

        saved = os.path.join(save_dir, name)
        further_extr(saved, rotate=False, save_to_psd=True, tblr_split=True)

        elapsed = time.time() - t0
        print(f'  Done in {elapsed:.1f}s')

        layers = sorted(Path(saved).glob('*.png'))
        layer_names = [l.stem for l in layers if not l.stem.startswith('src')]
        print(f'  Layers ({len(layer_names)}): {layer_names}')

    except Exception as e:
        print(f'  ERROR: {e}')
        import traceback
        traceback.print_exc()

total_elapsed = time.time() - total_start
print(f'\nTotal: {total_elapsed:.1f}s for {len(images)} images ({total_elapsed/max(len(images),1):.1f}s/img)')
" 2>&1 | tee "$LOG_DIR/seethrough.log"

    echo ""
    echo "[1.6] Uploading See-Through results..."
    rclone copy output/seethrough_test/ \
        "hetzner:strata-training-data/evaluation_seethrough_${TIMESTAMP}/" \
        --transfers 32 --fast-list -P 2>&1 | tail -3
    echo "  OK: See-Through results uploaded"
fi

echo ""

# ============================================================================
# PART 2: Texture Inpainting ControlNet v2 (4,135 pairs)
# ============================================================================
echo "======================================================="
echo "  PART 2: Texture Inpainting ControlNet v2"
echo "======================================================="
echo ""

# Download all texture pair tars
echo "[2.1] Downloading texture pair data..."
mkdir -p data_cloud

TARS=(
    "texture_pairs_front.tar"
    "texture_pairs_side.tar"
    "texture_pairs_back.tar"
    "texture_pairs_100avatars.tar"
)

for TAR in "${TARS[@]}"; do
    if [[ ! -f "data_cloud/$TAR" ]]; then
        echo "  Downloading $TAR..."
        rclone copyto "hetzner:strata-training-data/$TAR" "./data_cloud/$TAR" \
            --no-check-dest -P 2>&1 | tail -3
    else
        echo "  $TAR already exists"
    fi
done

# Extract all tars into a unified directory
echo ""
echo "[2.2] Extracting texture pairs..."
mkdir -p data_cloud/texture_pairs_all

for TAR in "${TARS[@]}"; do
    if [[ -f "data_cloud/$TAR" ]]; then
        # Extract and flatten: each tar has a top-level dir, we want all pose dirs in one place
        TARBASE="${TAR%.tar}"
        echo "  Extracting $TAR..."
        tar xf "data_cloud/$TAR" -C data_cloud/
        # Move pose dirs into unified dir with suffix to avoid collisions
        SUFFIX="${TARBASE#texture_pairs_}"
        SRC_DIR="data_cloud/$TARBASE"
        if [[ -d "$SRC_DIR" ]]; then
            for d in "$SRC_DIR"/*_pose_*/; do
                [ -d "$d" ] || continue
                DNAME=$(basename "$d")
                mv "$d" "data_cloud/texture_pairs_all/${DNAME}_${SUFFIX}"
            done
            rm -rf "$SRC_DIR"
        fi
        # Clean up tar to save disk
        rm -f "data_cloud/$TAR"
    fi
done

N_PAIRS=$(find data_cloud/texture_pairs_all -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
echo "  OK: $N_PAIRS total texture pairs"

if [[ "$N_PAIRS" -lt 100 ]]; then
    echo "  ERROR: Only $N_PAIRS pairs — need at least 100"
    echo "  SKIPPING texture inpainting training."
else
    # Pre-download SD Inpainting model
    echo ""
    echo "[2.3] Pre-downloading SD 1.5 Inpainting model..."
    python3 -c "
from transformers import CLIPTokenizer, CLIPTextModel
print('  Downloading runwayml/stable-diffusion-inpainting...')
_ = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-inpainting', subfolder='tokenizer')
_ = CLIPTextModel.from_pretrained('runwayml/stable-diffusion-inpainting', subfolder='text_encoder')
print('  OK: SD 1.5 Inpainting cached')
" 2>&1

    # Train (full config — 100 epochs)
    echo ""
    echo "[2.4] Training ControlNet (100 epochs, 4K+ pairs)..."
    python3 -m training.train_texture_inpainting \
        --config training/configs/texture_inpainting_a100_v2.yaml \
        2>&1 | tee "$LOG_DIR/texture_inpaint_train.log"

    # Upload checkpoints
    echo ""
    echo "[2.5] Uploading checkpoints..."
    CKPT_DIR="./checkpoints/texture_inpainting_controlnet"

    if [[ -d "$CKPT_DIR/best" ]]; then
        rclone copy "$CKPT_DIR/best" \
            "hetzner:strata-training-data/checkpoints_texture_inpaint_v2/best/" \
            --transfers 16 --fast-list -P 2>&1 | tail -3
        echo "  OK: best checkpoint uploaded"
    fi

    if [[ -d "$CKPT_DIR/latest" ]]; then
        rclone copy "$CKPT_DIR/latest" \
            "hetzner:strata-training-data/checkpoints_texture_inpaint_v2/latest/" \
            --transfers 16 --fast-list -P 2>&1 | tail -3
        echo "  OK: latest checkpoint uploaded"
    fi
fi

# Upload logs
echo ""
echo "[final] Uploading logs..."
rclone copy "$LOG_DIR" \
    "hetzner:strata-training-data/logs/a100_${TIMESTAMP}/" \
    --transfers 8 -P 2>&1 | tail -1

echo ""
echo "============================================"
echo "  A100 Run Complete!"
echo "  Logs: $LOG_DIR"
echo "  Finished: $(date)"
echo "============================================"

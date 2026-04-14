#!/usr/bin/env bash
# =============================================================================
# StyleTex A100 test — apply style-consistent texture to a character mesh
#
# Takes a mesh + style reference illustration + text prompt, optimizes a UV
# texture field via SDS (~30-60 min on A100).
#
# Note: StyleTex generates texture from scratch — does NOT preserve pixel-
# accurate projection from the illustration. Stylistic match only.
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/styletex_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Character to texture (override via env)
CHAR_NAME="${CHAR_NAME:-lichtung}"
MESH_FILE="${MESH_FILE:-object_0.glb}"
STYLE_IMAGE="${STYLE_IMAGE:-lichtung-character.png}"
PROMPT="${PROMPT:-a dark blue cat with gold star constellations on its fur, watercolor painting, starry night sky pattern}"
REF_CONTENT_PROMPT="${REF_CONTENT_PROMPT:-a cat}"
MAX_STEPS="${MAX_STEPS:-2500}"

echo "============================================"
echo "  StyleTex — Style-consistent UV texture"
echo "  Character: $CHAR_NAME"
echo "  Started: $(date)"
echo "============================================"

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "FAIL: CUDA not available"
    exit 1
fi
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
echo "OK: CUDA — $GPU_NAME"

# ---------------------------------------------------------------------------
# Step 1: Clone StyleTex
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Cloning StyleTex..."
if [[ ! -d /workspace/StyleTex ]]; then
    cd /workspace
    git clone https://github.com/XZYW7/StyleTex.git
    cd /workspace/strata-training-data
else
    echo "  StyleTex already exists"
fi

# ---------------------------------------------------------------------------
# Step 2: Install dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Installing StyleTex dependencies..."
cd /workspace/StyleTex

# StyleTex uses threestudio under the hood — needs tiny-cuda-nn + nvdiffrast
# These compile from source — takes ~10 min
pip install -q ninja 2>&1 | tail -3
pip install -q -r requirements.txt 2>&1 | tail -10

# Set HF cache path
export HF_HOME=/workspace/.hf_cache
mkdir -p $HF_HOME

# ---------------------------------------------------------------------------
# Step 3: Download test files from bucket
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Downloading test inputs from bucket..."
mkdir -p /workspace/styletex_input

rclone copyto "hetzner:strata-training-data/${CHAR_NAME}_test/${MESH_FILE}" \
    "/workspace/styletex_input/${MESH_FILE}" --no-check-dest -P 2>&1 | tail -3
rclone copyto "hetzner:strata-training-data/${CHAR_NAME}_test/${STYLE_IMAGE}" \
    "/workspace/styletex_input/${STYLE_IMAGE}" --no-check-dest -P 2>&1 | tail -3

ls -lh /workspace/styletex_input/

# ---------------------------------------------------------------------------
# Step 4: Run StyleTex
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Running StyleTex (SDS optimization, ~30-60 min)..."
cd /workspace/StyleTex

# Override the HF cache path in the config on the fly
python3 launch.py --config configs/styletex.yaml --train --gpu 0 \
    system.prompt_processor.prompt="$PROMPT" \
    system.prompt_processor.pretrained_model_cache_dir="$HF_HOME" \
    system.guidance.cache_dir="$HF_HOME" \
    system.guidance.ref_img_path="/workspace/styletex_input/${STYLE_IMAGE}" \
    system.guidance.ref_content_prompt="$REF_CONTENT_PROMPT" \
    system.geometry.shape_init="mesh:/workspace/styletex_input/${MESH_FILE}" \
    system.geometry.shape_init_params=1.0 \
    trainer.max_steps=$MAX_STEPS \
    name="${CHAR_NAME}_${TIMESTAMP}" \
    2>&1 | tee "$LOG_DIR/train.log"

# ---------------------------------------------------------------------------
# Upload results
# ---------------------------------------------------------------------------
echo ""
echo "[final] Uploading results..."
OUT_DIR="/workspace/StyleTex/outputs/styletex/${CHAR_NAME}_${TIMESTAMP}"
if [[ -d "$OUT_DIR" ]]; then
    rclone copy "$OUT_DIR" \
        "hetzner:strata-training-data/styletex_${CHAR_NAME}_${TIMESTAMP}/" \
        --transfers 8 -P 2>&1 | tail -3
    echo "OK: uploaded to bucket"
fi

rclone copy "$LOG_DIR" \
    "hetzner:strata-training-data/logs/styletex_${TIMESTAMP}/" \
    --transfers 4 -P 2>&1 | tail -1

echo ""
echo "============================================"
echo "  Done! Finished: $(date)"
echo "============================================"

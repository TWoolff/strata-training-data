#!/usr/bin/env bash
# =============================================================================
# Strata Training — Cloud Setup Script
# Run this on a fresh cloud instance (Lambda Labs / Vast.ai / RunPod)
# with an A100 GPU to set up the environment and download training data.
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY="your-hetzner-access-key"
#   export BUCKET_SECRET="your-hetzner-secret-key"
#
# Usage:
#   chmod +x training/cloud_setup.sh
#   ./training/cloud_setup.sh
# =============================================================================
set -euo pipefail

echo "============================================"
echo "  Strata Training — Cloud Setup"
echo "============================================"

# ---------------------------------------------------------------------------
# 0. Validate credentials
# ---------------------------------------------------------------------------
if [[ -z "${BUCKET_ACCESS_KEY:-}" ]] || [[ -z "${BUCKET_SECRET:-}" ]]; then
    echo "ERROR: BUCKET_ACCESS_KEY and BUCKET_SECRET must be set."
    echo "  export BUCKET_ACCESS_KEY='your-key'"
    echo "  export BUCKET_SECRET='your-secret'"
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. System deps
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq rclone

# ---------------------------------------------------------------------------
# 2. Python deps
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Installing Python dependencies..."
pip install -q -r training/requirements.txt

# Verify CUDA is available
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'CUDA OK: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_mem / 1024**3:.0f} GB VRAM')"

# ---------------------------------------------------------------------------
# 3. Configure rclone for Hetzner bucket
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Configuring rclone for Hetzner Object Storage..."
mkdir -p ~/.config/rclone

cat > ~/.config/rclone/rclone.conf << RCLONE_EOF
[hetzner]
type = s3
provider = Other
env_auth = false
access_key_id = ${BUCKET_ACCESS_KEY}
secret_access_key = ${BUCKET_SECRET}
endpoint = fsn1.your-objectstorage.com
acl = private
RCLONE_EOF

echo "  rclone configured. Testing connection..."
rclone lsd hetzner:strata-training-data/ 2>/dev/null && echo "  Bucket connection OK" || { echo "ERROR: Cannot connect to bucket"; exit 1; }

# ---------------------------------------------------------------------------
# 4. Download training data from bucket
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Downloading training data from Hetzner bucket..."
echo "  Estimated total: ~25 GB"
echo ""

DATA_DIR="./data_cloud"
RCLONE_FLAGS="--transfers 32 --checkers 64 --fast-list --size-only -P"
mkdir -p "$DATA_DIR"

# --- Segmentation training data (image + 22-class mask + draw_order + joints) ---

# Mixamo 3D pipeline — flat layout: images/, masks/, joints/, draw_order/
echo "  [a] segmentation/ (Mixamo pipeline — ~600 MB, ~1,600 clean renders)..."
rclone copy hetzner:strata-training-data/segmentation/ "$DATA_DIR/segmentation/" $RCLONE_FLAGS

# Live2D — flat layout, full labels
echo ""
echo "  [b] live2d/ (~212 MB, 844 examples)..."
rclone copy hetzner:strata-training-data/live2d/ "$DATA_DIR/live2d/" $RCLONE_FLAGS

# --- Per-example datasets (image + binary/multi-class seg + joints) ---

# HumanRig — per-example: image + segmentation + joints + weights (3 angles)
echo ""
echo "  [c] humanrig/ (~5.6 GB, 11,434 characters)..."
rclone copy hetzner:strata-training-data/humanrig/ "$DATA_DIR/humanrig/" $RCLONE_FLAGS

# anime_seg — per-example: image + binary fg mask + RTMPose joints
echo ""
echo "  [d] anime_seg/ (~3.5 GB, 14,579 examples with joints)..."
rclone copy hetzner:strata-training-data/anime_seg/ "$DATA_DIR/anime_seg/" $RCLONE_FLAGS

# anime_instance_seg — per-example: image + instance mask + metadata
echo ""
echo "  [e] anime_instance_seg/ (~15 GB, ~45K uploaded so far)..."
rclone copy hetzner:strata-training-data/anime_instance_seg/ "$DATA_DIR/anime_instance_seg/" $RCLONE_FLAGS

# --- Joint-only datasets (image + joints.json, no seg masks) ---

# FBAnimeHQ — per-example: image + joints
echo ""
echo "  [f] fbanimehq/ (~11.4 GB, ~101K face/body crops)..."
rclone copy hetzner:strata-training-data/fbanimehq/ "$DATA_DIR/fbanimehq/" $RCLONE_FLAGS

# --- Draw order datasets ---

# InstaOrder — per-example: image + draw_order map + metadata
echo ""
echo "  [g] instaorder/ (~1.5 GB, 3,956 val examples)..."
rclone copy hetzner:strata-training-data/instaorder/ "$DATA_DIR/instaorder/" $RCLONE_FLAGS

# --- Weight prediction datasets ---

# UniRig — rigged meshes with skeleton + skinning weights
echo ""
echo "  [h] unirig/ (~42.6 GB, 66K files)..."
echo "       Skipping by default — very large. Uncomment to download."
# rclone copy hetzner:strata-training-data/unirig/ "$DATA_DIR/unirig/" $RCLONE_FLAGS

echo ""
echo "  Download complete."
echo ""

# ---------------------------------------------------------------------------
# 5. Verify data
# ---------------------------------------------------------------------------
echo "[5/5] Verifying downloaded data..."

for ds in segmentation live2d humanrig anime_seg anime_instance_seg fbanimehq instaorder; do
    if [[ -d "$DATA_DIR/$ds" ]]; then
        count=$(find "$DATA_DIR/$ds" -type f | wc -l)
        size=$(du -sh "$DATA_DIR/$ds" 2>/dev/null | cut -f1)
        echo "  $ds: $count files ($size)"
    else
        echo "  $ds: NOT DOWNLOADED"
    fi
done

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  To train all models:"
echo "    ./training/train_all.sh"
echo ""
echo "  To train a single model:"
echo "    python -m training.train_segmentation --config training/configs/segmentation_a100.yaml"
echo "    python -m training.train_joints --config training/configs/joints_a100.yaml"
echo "    python -m training.train_weights --config training/configs/weights_a100.yaml"
echo "============================================"

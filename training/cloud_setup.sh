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
#   ./training/cloud_setup.sh              # Download all data (~43 GB)
#   ./training/cloud_setup.sh lean         # Core data only (~10 GB, for lean training)
# =============================================================================
set -euo pipefail

MODE="${1:-full}"

echo "============================================"
echo "  Strata Training — Cloud Setup ($MODE)"
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
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; p=torch.cuda.get_device_properties(0); mem=getattr(p,'total_memory',getattr(p,'total_mem',0)); print(f'CUDA OK: {torch.cuda.get_device_name(0)}, {mem/1024**3:.0f} GB VRAM')"

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
DATA_DIR="./data_cloud"
RCLONE_FLAGS="--transfers 32 --checkers 64 --fast-list --size-only -P"
mkdir -p "$DATA_DIR"

# --- Core datasets (always downloaded) ---
echo "[4/5] Downloading training data from Hetzner bucket..."
if [ "$MODE" = "lean" ]; then
    echo "  LEAN mode: core data only (~21 GB)"
else
    echo "  FULL mode: all data (~43 GB)"
fi
echo ""

echo "  [a] segmentation/ (Mixamo — ~600 MB, 1,598 renders)..."
rclone copy hetzner:strata-training-data/segmentation/ "$DATA_DIR/segmentation/" $RCLONE_FLAGS

echo ""
echo "  [b] live2d/ (~212 MB, 844 examples)..."
rclone copy hetzner:strata-training-data/live2d/ "$DATA_DIR/live2d/" $RCLONE_FLAGS

echo ""
echo "  [c] humanrig/ (~5.6 GB, 11,434 examples)..."
rclone copy hetzner:strata-training-data/humanrig/ "$DATA_DIR/humanrig/" $RCLONE_FLAGS

echo ""
echo "  [d] anime_seg/ (~3.5 GB, 14,579 examples with joints)..."
rclone copy hetzner:strata-training-data/anime_seg/ "$DATA_DIR/anime_seg/" $RCLONE_FLAGS

echo ""
echo "  [e] fbanimehq/ (~11.4 GB, ~101K full-body anime with joints)..."
rclone copy hetzner:strata-training-data/fbanimehq/ "$DATA_DIR/fbanimehq/" $RCLONE_FLAGS

# --- Additional datasets (full mode only) ---
if [ "$MODE" != "lean" ]; then
    echo ""
    echo "  [f] anime_instance_seg/ (~15 GB, ~45K examples)..."
    rclone copy hetzner:strata-training-data/anime_instance_seg/ "$DATA_DIR/anime_instance_seg/" $RCLONE_FLAGS

    echo ""
    echo "  [g] instaorder/ (~7 GB, ~96K train+val examples)..."
    rclone copy hetzner:strata-training-data/instaorder/ "$DATA_DIR/instaorder/" $RCLONE_FLAGS

    # echo ""
    # echo "  [h] unirig/ (~42.6 GB, 66K files)..."
    # rclone copy hetzner:strata-training-data/unirig/ "$DATA_DIR/unirig/" $RCLONE_FLAGS
fi

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
        echo "  $ds: (not downloaded)"
    fi
done

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
if [ "$MODE" = "lean" ]; then
    echo "  To train (lean, ~3-4h):"
    echo "    ./training/train_all.sh lean"
else
    echo "  To train (full, ~8-12h):"
    echo "    ./training/train_all.sh"
fi
echo ""
echo "  Single model:"
echo "    python -m training.train_segmentation --config training/configs/segmentation_a100_lean.yaml"
echo "    python -m training.train_joints --config training/configs/joints_a100_lean.yaml"
echo "    python -m training.train_weights --config training/configs/weights_a100_lean.yaml"
echo "============================================"

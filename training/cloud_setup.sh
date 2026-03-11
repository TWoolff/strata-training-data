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
region = fsn1
acl = private
RCLONE_EOF

echo "  rclone configured. Testing connection..."
rclone lsd hetzner:strata-training-data/ 2>/dev/null && echo "  Bucket connection OK" || { echo "ERROR: Cannot connect to bucket"; exit 1; }

# ---------------------------------------------------------------------------
# 4. Download training data from bucket
# ---------------------------------------------------------------------------
echo ""
DATA_DIR="./data_cloud"
TAR_BUCKET="hetzner:strata-training-data/tars"
RCLONE_FLAGS="--transfers 32 --checkers 64 --fast-list --size-only -P"
mkdir -p "$DATA_DIR"

# Helper: download and extract tar archive
download_dataset() {
    local ds="$1"
    local desc="$2"

    echo "  $ds ($desc)..."
    echo "    → Downloading ${ds}.tar..."
    rclone copy "$TAR_BUCKET/${ds}.tar" "$DATA_DIR/_tars/" $RCLONE_FLAGS
    echo "    → Extracting..."
    tar xf "$DATA_DIR/_tars/${ds}.tar" -C "$DATA_DIR/"
    rm -f "$DATA_DIR/_tars/${ds}.tar"
    echo ""
}

# --- Core datasets (always downloaded) ---
echo "[4/5] Downloading training data from Hetzner bucket..."
if [ "$MODE" = "lean" ]; then
    echo "  LEAN mode: core data only (~45 GB)"
else
    echo "  FULL mode: all data (~55 GB)"
fi
echo ""

mkdir -p "$DATA_DIR/_tars"

# Meshy CC0 — primary CC0-licensed dataset (replaces Mixamo)
download_dataset "meshy_cc0"          "Meshy CC0 flat: ~7,700 examples, 22-class seg + joints + depth + normals"
download_dataset "meshy_cc0_textured" "Meshy CC0 textured: ~8,200 examples, 22-class seg + joints + depth + normals"
download_dataset "meshy_cc0_unrigged" "Meshy CC0 unrigged: ~20K textured multi-view (image + depth + normals only)"

# Other clean-licensed datasets
# live2d removed — Live2D ToS prohibits AI/ML training
download_dataset "humanrig"       "~5.6 GB, 11,434 examples (weights)"
download_dataset "anime_seg"      "~3.5 GB, 14,579 examples with joints"
download_dataset "fbanimehq"      "~11.4 GB, ~101K full-body anime with joints"
# curated_diverse removed — ArtStation artwork, no AI training permission
# texture_pairs — not yet generated, will be added in future runs

# UniRig: download only front views with weights (skip back views to save space)
echo "  unirig (~15 GB, ~15K front views with weights)..."
echo "    → Downloading front/ subdirs only..."
rclone copy "hetzner:strata-training-data/unirig/" "$DATA_DIR/unirig/" \
    --include "*/front/**" $RCLONE_FLAGS
echo ""

# --- Additional datasets (full mode only) ---
# anime_instance_seg and instaorder removed — prohibited licenses / unused
if [ "$MODE" != "lean" ]; then
    echo "  (No additional full-mode datasets currently configured)"
fi

# Clean up tar staging dir
rmdir "$DATA_DIR/_tars" 2>/dev/null || true

echo ""
echo "  Download complete."
echo ""

# ---------------------------------------------------------------------------
# 5. Verify data
# ---------------------------------------------------------------------------
echo "[5/5] Verifying downloaded data..."

for ds in meshy_cc0 meshy_cc0_textured meshy_cc0_unrigged humanrig unirig anime_seg fbanimehq; do
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

#!/usr/bin/env bash
# =============================================================================
# Strata Training — Cloud Setup Script
# Run this on a fresh cloud instance (Lambda Labs / Vast.ai / RunPod)
# with an A100 GPU to set up the environment and download training data.
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

cat > ~/.config/rclone/rclone.conf << 'RCLONE_EOF'
[hetzner]
type = s3
provider = Other
env_auth = false
access_key_id = 7TE7YDETDY571N0ANB31
secret_access_key = JKUKCNFk7lVanSsZOGVPFMdR6wzvuBFZXZZ5jl8C
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
echo "  This downloads all datasets with training labels."
echo "  Estimated download: ~20 GB"
echo ""

DATA_DIR="./data_cloud"
mkdir -p "$DATA_DIR"

# --- Full-label datasets (seg + joints + weights + draw_order) ---

# Segmentation (Mixamo 3D pipeline) — flat layout, full labels
echo "  Downloading segmentation/ (Mixamo pipeline — 599 MB)..."
rclone sync hetzner:strata-training-data/segmentation/ "$DATA_DIR/segmentation/" \
    --transfers 32 --checkers 64 --fast-list -P

# Live2D — flat layout, full labels
echo ""
echo "  Downloading live2d/ (212 MB)..."
rclone sync hetzner:strata-training-data/live2d/ "$DATA_DIR/live2d/" \
    --transfers 32 --checkers 64 --fast-list -P

# --- Joint-enriched datasets (image + joints.json via RTMPose) ---

# HumanRig — per-example: image + joints
echo ""
echo "  Downloading humanrig/ (5.6 GB)..."
rclone sync hetzner:strata-training-data/humanrig/ "$DATA_DIR/humanrig/" \
    --transfers 32 --checkers 64 --fast-list -P

# FBAnimeHQ — per-example: image + joints (~101K examples, enriched)
echo ""
echo "  Downloading fbanimehq/ (11.4 GB)..."
rclone sync hetzner:strata-training-data/fbanimehq/ "$DATA_DIR/fbanimehq/" \
    --transfers 32 --checkers 64 --fast-list -P

# anime_seg — per-example: image + binary seg + joints (~25K examples, enriched)
echo ""
echo "  Downloading anime_seg/ (2.5 GB)..."
rclone sync hetzner:strata-training-data/anime_seg/ "$DATA_DIR/anime_seg/" \
    --transfers 32 --checkers 64 --fast-list -P

# anime_instance_seg — per-example: image + binary seg + metadata (NO joints)
# Skipped for now — no joint annotations, binary-only seg masks
# Useful later for confidence training or pseudo-labeling
# echo ""
# echo "  Downloading anime_instance_seg/ (10.9 GB)..."
# rclone sync hetzner:strata-training-data/anime_instance_seg/ "$DATA_DIR/anime_instance_seg/" \
#     --transfers 32 --checkers 64 --fast-list -P

echo ""
echo "  Download complete."
echo ""

# ---------------------------------------------------------------------------
# 5. Verify data
# ---------------------------------------------------------------------------
echo "[5/5] Verifying downloaded data..."

# Count files per dataset
for ds in segmentation live2d humanrig fbanimehq anime_seg; do
    count=$(find "$DATA_DIR/$ds" -type f | wc -l)
    size=$(du -sh "$DATA_DIR/$ds" 2>/dev/null | cut -f1)
    echo "  $ds: $count files ($size)"
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

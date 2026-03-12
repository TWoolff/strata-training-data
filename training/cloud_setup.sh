#!/usr/bin/env bash
# =============================================================================
# Strata Training — Cloud Setup Script
# Run this on a fresh cloud instance (Lambda Labs / Vast.ai / RunPod)
# with an A100 GPU to set up the environment and configure bucket access.
#
# This script only installs dependencies and configures rclone.
# Dataset downloads are handled by each run script (run_seg_only.sh, etc.)
# so only the data needed for that specific run is downloaded.
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
echo "[1/3] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq rclone

# ---------------------------------------------------------------------------
# 2. Python deps
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] Installing Python dependencies..."
pip install -q -r training/requirements.txt

# Verify CUDA is available
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; p=torch.cuda.get_device_properties(0); mem=getattr(p,'total_memory',getattr(p,'total_mem',0)); print(f'CUDA OK: {torch.cuda.get_device_name(0)}, {mem/1024**3:.0f} GB VRAM')"

# ---------------------------------------------------------------------------
# 3. Configure rclone for Hetzner bucket
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Configuring rclone for Hetzner Object Storage..."
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

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Next: run your training script, e.g.:"
echo "    ./training/run_seg_only.sh"
echo "    ./training/run_fifth.sh"
echo ""
echo "  Each run script downloads only the data it needs."
echo "============================================"

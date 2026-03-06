#!/usr/bin/env bash
# =============================================================================
# Strata Training — One-shot Cloud Launch
# Paste this on a fresh Vast.ai / Lambda Labs instance to clone, setup, and train.
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY="your-hetzner-access-key"
#   export BUCKET_SECRET="your-hetzner-secret-key"
#
# Usage (copy-paste into SSH terminal):
#   curl -sL https://raw.githubusercontent.com/TWoolff/strata-training-data/main/training/launch.sh | bash -s lean
#
# Or after cloning:
#   ./training/launch.sh lean
# =============================================================================
set -euo pipefail

MODE="${1:-lean}"

echo "============================================"
echo "  Strata Training — Full Launch ($MODE)"
echo "  Started: $(date)"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Clone repo (skip if already in repo)
# ---------------------------------------------------------------------------
if [[ ! -f "training/cloud_setup.sh" ]]; then
    echo "[0/3] Cloning repository..."
    git clone https://github.com/TWoolff/strata-training-data.git
    cd strata-training-data
else
    echo "[0/3] Already in repo directory, skipping clone."
fi

# ---------------------------------------------------------------------------
# 2. Setup (install deps + download data)
# ---------------------------------------------------------------------------
echo ""
echo "[1/3] Running cloud setup ($MODE)..."
chmod +x training/cloud_setup.sh
./training/cloud_setup.sh "$MODE"

# ---------------------------------------------------------------------------
# 3. Train all models
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] Starting training ($MODE)..."
chmod +x training/train_all.sh
./training/train_all.sh "$MODE"

# ---------------------------------------------------------------------------
# 4. Done — show results
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] All done!"
echo ""
echo "  To download results:"
echo "    scp -r $(whoami)@$(hostname):$(pwd)/models/onnx/ ./models/onnx/"
echo "    scp -r $(whoami)@$(hostname):$(pwd)/checkpoints/ ./checkpoints/"
echo "    scp -r $(whoami)@$(hostname):$(pwd)/logs/ ./logs/"
echo ""
echo "  Finished: $(date)"
echo "============================================"

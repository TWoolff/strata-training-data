#!/usr/bin/env bash
# =============================================================================
# Fine-tune SHARP on turnaround sheet characters
#
# Requires: A100 with CUDA, ~20 GB VRAM
#
# Usage:
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#   ./training/run_sharp_finetune.sh
# =============================================================================
set -euo pipefail

echo "============================================"
echo "  SHARP Fine-tune: Illustrated Characters"
echo "  Started: $(date)"
echo "============================================"

# Pre-flight
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "FAIL: CUDA required"; exit 1
fi
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
echo "GPU: $GPU_NAME"

# 1. Install SHARP
echo ""
echo "[1] Setting up SHARP..."
if [ ! -d "../ml-sharp" ]; then
    git clone https://github.com/apple/ml-sharp.git ../ml-sharp
fi
pip install -q -e ../ml-sharp 2>&1 | tail -3

# 2. Download demo_pairs
echo ""
echo "[2] Downloading training data..."
mkdir -p data/training data/tars
if [ ! -d "data/training/demo_pairs" ] || [ -z "$(ls data/training/demo_pairs/ 2>/dev/null | head -1)" ]; then
    rclone copy hetzner:strata-training-data/tars/demo_back_view_pairs.tar ./data/tars/ --transfers 32 --fast-list -P
    tar xf ./data/tars/demo_back_view_pairs.tar -C ./data/training/
    rm -f ./data/tars/demo_back_view_pairs.tar
fi
PAIR_COUNT=$(ls -d data/training/demo_pairs/pair_* 2>/dev/null | wc -l | tr -d ' ')
echo "  demo_pairs: $PAIR_COUNT pairs"

# 3. Download SHARP checkpoint (or use cached)
echo ""
echo "[3] Downloading SHARP checkpoint..."
SHARP_CKPT="$HOME/.cache/torch/hub/checkpoints/sharp_2572gikvuh.pt"
if [ ! -f "$SHARP_CKPT" ]; then
    mkdir -p "$HOME/.cache/torch/hub/checkpoints"
    wget -q -O "$SHARP_CKPT" https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt
fi
echo "  Checkpoint: $SHARP_CKPT"

# 4. Train
echo ""
echo "[4] Training SHARP..."
mkdir -p checkpoints/sharp logs

# demo_pairs has ~6,200 pairs but many share the same character views.
# At internal_res=768, ~2-3s per step.
# Use --max-train-samples to cap epoch length.
# 500 samples/epoch × 20 epochs × ~2.5s = ~7 hrs
# 300 samples/epoch × 20 epochs × ~2.5s = ~4 hrs
python3 -m training.train_sharp \
    --data-dir ./data/training/demo_pairs \
    --checkpoint "$SHARP_CKPT" \
    --output-dir ./checkpoints/sharp \
    --epochs 20 \
    --lr 1e-5 \
    --batch-size 1 \
    --internal-resolution 768 \
    --freeze-encoder \
    --patience 10 \
    --max-train-samples 300 \
    2>&1 | tee logs/sharp_finetune.log

# 5. Upload
echo ""
echo "[5] Uploading results..."
rclone copy ./checkpoints/sharp/ \
    hetzner:strata-training-data/checkpoints_sharp/ --transfers 4 --fast-list --size-only -P

echo ""
echo "============================================"
echo "  DONE! $(date)"
echo "============================================"

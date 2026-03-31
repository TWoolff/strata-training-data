#!/usr/bin/env bash
# =============================================================================
# Strata Training — Next Run
#
# 1. Run Dr. Li's SAM Body Parsing on gemini_diverse (~30 min)
# 2. Fine-tune view synthesis on bear chef A-pose (~2-3 hrs)
#
# Total: ~3-4 hrs. Storage: 30 GB.
#
# Usage:
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#   ./training/run_next.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/next_run_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Next Run: SAM Seg + Bear Chef View Synth"
echo "  Started: $(date)"
echo "============================================"
echo ""

# Pre-flight
echo "[pre] Pre-flight checks..."
if ! rclone lsd hetzner:strata-training-data/ &>/dev/null; then echo "  FAIL: rclone"; exit 1; fi
echo "  OK: rclone"
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then echo "  FAIL: CUDA"; exit 1; fi
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
echo "  OK: CUDA — $GPU_NAME"
echo ""

# =============================================================================
# PART 1: Dr. Li's SAM Body Parsing on gemini_diverse
# =============================================================================
echo "########################################################"
echo "  PART 1: SAM Body Parsing (Dr. Li's See-Through)"
echo "########################################################"
echo ""

# 1.0 Install See-Through dependencies
echo "[1.0] Installing See-Through dependencies..."
if [ ! -d "see-through" ]; then
    git clone https://github.com/shitagaki-lab/see-through.git
fi
cd see-through
pip install -r requirements.txt 2>&1 | tail -5
pip install --no-build-isolation -r requirements-inference-sam2.txt 2>&1 | tail -5
ln -sf common/assets assets 2>/dev/null || true
cd ..
echo "  See-Through installed."
echo ""

# 1.1 Download sora_diverse (illustrated images to label)
echo "[1.1] Downloading illustrated images..."
mkdir -p data_cloud data/tars

if [ ! -d "data_cloud/sora_diverse" ] || [ -z "$(ls data_cloud/sora_diverse/ 2>/dev/null | head -1)" ]; then
    rclone copy hetzner:strata-training-data/tars/sora_diverse.tar ./data/tars/ --transfers 32 --fast-list -P
    tar xf ./data/tars/sora_diverse.tar -C ./data_cloud/
    rm -f ./data/tars/sora_diverse.tar
fi

# Also download new sora_diverse tars
for tar in sora_diverse_new.tar sora_diverse_new2.tar; do
    if rclone ls "hetzner:strata-training-data/tars/$tar" &>/dev/null 2>&1; then
        echo "  Downloading $tar..."
        rclone copy "hetzner:strata-training-data/tars/$tar" ./data/tars/ --transfers 32 --fast-list -P
        tar xf "./data/tars/$tar" -C ./data_cloud/ 2>/dev/null || true
        rm -f "./data/tars/$tar"
    fi
done

SORA_COUNT=$(ls -d ./data_cloud/sora_diverse/*/ 2>/dev/null | wc -l | tr -d ' ')
echo "  sora_diverse: $SORA_COUNT images"
echo ""

# 1.2 Run SAM Body Parsing on all images
echo "[1.2] Running SAM Body Parsing..."

python3 -c "
import sys, json, logging
from pathlib import Path
sys.path.insert(0, 'see-through')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Import SAM body parsing from see-through
try:
    from annotators.lang_sam import LangSAMAnnotator
    logger.info('LangSAM loaded')
except ImportError:
    logger.error('Could not import LangSAM — check see-through installation')
    sys.exit(1)

# Find all image directories
data_dir = Path('data_cloud/sora_diverse')
examples = sorted([d for d in data_dir.iterdir() if d.is_dir() and (d / 'image.png').exists()])
logger.info('Found %d images to process', len(examples))

# Initialize model
annotator = LangSAMAnnotator()

processed = 0
for i, ex_dir in enumerate(examples):
    img_path = ex_dir / 'image.png'
    out_path = ex_dir / 'sam_segmentation.png'

    # Skip if already processed
    if out_path.exists():
        continue

    try:
        # Run SAM body parsing
        result = annotator.predict(str(img_path))
        # Save result
        from PIL import Image
        import numpy as np
        if result is not None:
            mask = np.array(result, dtype=np.uint8)
            Image.fromarray(mask).save(out_path)
            processed += 1
    except Exception as e:
        logger.warning('Error on %s: %s', ex_dir.name, e)

    if (i + 1) % 100 == 0:
        logger.info('Progress: %d/%d (%d processed)', i + 1, len(examples), processed)

logger.info('Done: %d processed out of %d', processed, len(examples))
" 2>&1 | tee "$LOG_DIR/sam_parsing.log"

echo "  SAM Body Parsing complete."
echo ""

# =============================================================================
# PART 2: View Synthesis Bear Chef Fine-tune
# =============================================================================
echo "########################################################"
echo "  PART 2: View Synthesis Bear Chef Fine-tune"
echo "########################################################"
echo ""

# 2.0 Download checkpoint + data
echo "[2.0] Downloading data..."
mkdir -p checkpoints/view_synthesis data/training

RUN2_CKPT="checkpoints/view_synthesis/run2_best.pt"
if [ ! -f "$RUN2_CKPT" ]; then
    rclone copy hetzner:strata-training-data/checkpoints_view_synthesis_run2/run2_best.pt \
        ./checkpoints/view_synthesis/ --transfers 32 --fast-list -P
fi

# Demo pairs
if [ ! -d "data/training/demo_pairs" ] || [ -z "$(ls data/training/demo_pairs/ 2>/dev/null | head -1)" ]; then
    rclone copy hetzner:strata-training-data/tars/demo_back_view_pairs.tar ./data/tars/ --transfers 32 --fast-list -P
    tar xf ./data/tars/demo_back_view_pairs.tar -C ./data/training/
    rm -f ./data/tars/demo_back_view_pairs.tar
fi
DEMO_COUNT=$(ls -d ./data/training/demo_pairs/pair_* 2>/dev/null | wc -l | tr -d ' ')
echo "  demo_pairs: $DEMO_COUNT"

# 3D pairs
download_tar() {
    local tar_name="$1"; local extract_dir="$2"
    if [ -d "$extract_dir" ] && [ "$(ls -d "$extract_dir"/pair_* 2>/dev/null | head -1)" ]; then
        echo "  $(basename "$extract_dir"): exists"; return 0
    fi
    echo "  Downloading ${tar_name}..."
    rclone copy "hetzner:strata-training-data/tars/${tar_name}" ./data/tars/ --transfers 32 --fast-list -P
    if [ -f "./data/tars/${tar_name}" ]; then
        tar xf "./data/tars/${tar_name}" -C ./data/training/; rm -f "./data/tars/${tar_name}"
    fi
}
download_tar "back_view_pairs.tar" "./data/training/back_view_pairs_merged"
download_tar "back_view_pairs_unrigged.tar" "./data/training/back_view_pairs_unrigged"
download_tar "back_view_pairs_new.tar" "./data/training/back_view_pairs_new"
echo ""

# 2.1 Train
echo "[2.1] Training view synthesis (bear chef fine-tune)..."

rm -f checkpoints/view_synthesis/latest.pt
python3 -c "
import torch
ckpt = torch.load('$RUN2_CKPT', map_location='cpu', weights_only=False)
ckpt['epoch'] = -1
torch.save(ckpt, 'checkpoints/view_synthesis/latest.pt')
"

python3 -m training.train_view_synthesis \
    --config training/configs/view_synthesis_bear_chef.yaml \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""

# 2.2 Export + Upload
echo "[2.2] Exporting + uploading..."
mkdir -p ./models/onnx

if [ -f "checkpoints/view_synthesis/best.pt" ]; then
    cp checkpoints/view_synthesis/best.pt checkpoints/view_synthesis/bear_chef_best.pt
    python3 -m training.export_onnx --model view_synthesis \
        --checkpoint checkpoints/view_synthesis/bear_chef_best.pt \
        --output ./models/onnx/view_synthesis_bear_chef.onnx \
        2>&1 | tee "$LOG_DIR/export.log"
fi

rclone copy ./checkpoints/view_synthesis/bear_chef_best.pt \
    hetzner:strata-training-data/checkpoints_view_synthesis_bear_chef/ --transfers 4 --fast-list --size-only -P
if [ -f "./models/onnx/view_synthesis_bear_chef.onnx" ]; then
    rclone copy ./models/onnx/view_synthesis_bear_chef.onnx \
        hetzner:strata-training-data/models/view_synthesis_bear_chef/ --transfers 4 --fast-list --size-only -P
fi
rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/next_run_${TIMESTAMP}/ --transfers 4 --fast-list -P

echo ""
echo "============================================"
echo "  ALL DONE! $(date)"
echo ""
echo "  SAM Parsing:"
grep -c "processed" "$LOG_DIR/sam_parsing.log" 2>/dev/null || echo "  (check log)"
echo ""
echo "  View Synthesis:"
grep -E "best val/l1|New best" "$LOG_DIR/train.log" 2>/dev/null | tail -3 || true
echo "============================================"

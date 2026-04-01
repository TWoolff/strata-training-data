#!/usr/bin/env bash
# =============================================================================
# Strata Training — April 1 Run
#
# 1. Seg retrain with SAM labels (~3 hrs)
# 2. View synthesis bear chef fine-tune (~1 hr)
#
# Total: ~4-5 hrs. Storage: 40 GB.
#
# Usage:
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#   ./training/run_next.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/april1_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  April 1 Run: SAM Seg + Bear Chef"
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
# PART 1: Segmentation retrain with SAM labels
# =============================================================================
echo "########################################################"
echo "  PART 1: Seg Retrain with SAM Labels"
echo "########################################################"
echo ""

# 1.0 Download seg data + checkpoint
echo "[1.0] Downloading seg data..."
mkdir -p data_cloud data/tars checkpoints/segmentation

# Seg checkpoint (run 20)
SEG_CKPT="checkpoints/segmentation/run20_best.pt"
if [ ! -f "$SEG_CKPT" ]; then
    rclone copy hetzner:strata-training-data/checkpoints_run20_seg/segmentation/run20_best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
fi
echo "  Seg checkpoint: $SEG_CKPT"

# Frozen splits
if [ ! -f "data_cloud/frozen_val_test.json" ]; then
    rclone copy hetzner:strata-training-data/data_cloud/frozen_val_test.json \
        ./data_cloud/ --transfers 4 --fast-list -P
fi

# Download datasets
download_tar() {
    local tar_name="$1"; local extract_dir="$2"
    if [ -d "$extract_dir" ] && [ "$(ls "$extract_dir"/ 2>/dev/null | head -1)" ]; then
        local count=$(ls -d "$extract_dir"/*/ 2>/dev/null | wc -l | tr -d ' ')
        echo "  $(basename "$extract_dir"): $count examples (exists)"
        return 0
    fi
    echo "  Downloading $tar_name..."
    rclone copy "hetzner:strata-training-data/tars/$tar_name" ./data/tars/ --transfers 32 --fast-list -P
    if [ -f "./data/tars/$tar_name" ]; then
        tar xf "./data/tars/$tar_name" -C ./data_cloud/
        rm -f "./data/tars/$tar_name"
    fi
}

download_tar "humanrig.tar" "./data_cloud/humanrig"
download_tar "vroid_cc0.tar" "./data_cloud/vroid_cc0"
download_tar "meshy_cc0_textured_restructured.tar" "./data_cloud/meshy_cc0_restructured"
download_tar "gemini_li_converted.tar" "./data_cloud/gemini_li_converted"
download_tar "cvat_annotated.tar" "./data_cloud/cvat_annotated"
download_tar "sora_diverse.tar" "./data_cloud/sora_diverse"
download_tar "flux_diverse_clean.tar" "./data_cloud/flux_diverse_clean"

# Download new sora_diverse images
for tar in sora_diverse_new.tar sora_diverse_new2.tar; do
    if rclone ls "hetzner:strata-training-data/tars/$tar" &>/dev/null 2>&1; then
        echo "  Downloading $tar..."
        rclone copy "hetzner:strata-training-data/tars/$tar" ./data/tars/ --transfers 32 --fast-list -P
        if [ -f "./data/tars/$tar" ]; then
            tar xf "./data/tars/$tar" -C ./data_cloud/ 2>/dev/null || true
            rm -f "./data/tars/$tar"
        fi
    fi
done

# 1.1 Apply SAM-converted seg labels (overwrite old pseudo-labels)
echo ""
echo "[1.1] Applying SAM-converted seg labels to sora_diverse..."
rclone copy hetzner:strata-training-data/tars/sam_seg_converted.tar ./data/tars/ --transfers 32 --fast-list -P
if [ -f "./data/tars/sam_seg_converted.tar" ]; then
    tar xf ./data/tars/sam_seg_converted.tar -C ./data_cloud/
    rm -f ./data/tars/sam_seg_converted.tar
fi
SAM_COUNT=$(find ./data_cloud/sora_diverse -name "segmentation.png" | wc -l | tr -d ' ')
echo "  Applied SAM labels to $SAM_COUNT sora_diverse images"
echo ""

# 1.1b Extract unique view images from demo_pairs → seg-compatible layout
echo "[1.1b] Extracting demo_pairs views for SAM labeling..."
python3 -c "
from pathlib import Path
from PIL import Image
import hashlib, shutil

demo_dir = Path('data/training/demo_pairs')
out_dir = Path('data_cloud/demo_views')
out_dir.mkdir(parents=True, exist_ok=True)

# Deduplicate by image content hash
seen_hashes = set()
extracted = 0

for pair_dir in sorted(demo_dir.glob('pair_*')):
    for img_name in ['front.png', 'three_quarter.png', 'back.png']:
        img_path = pair_dir / img_name
        if not img_path.exists():
            continue
        # Hash to deduplicate (same source image used in many pairs)
        h = hashlib.md5(img_path.read_bytes()).hexdigest()[:12]
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # Create seg-compatible example dir: demo_views/char_HASH/image.png
        ex_dir = out_dir / f'{pair_dir.name}_{img_name.replace(\".png\", \"\")}_{h}'
        ex_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, ex_dir / 'image.png')
        extracted += 1

print(f'Extracted {extracted} unique images from demo_pairs')
"
DEMO_VIEW_COUNT=$(ls -d data_cloud/demo_views/*/ 2>/dev/null | wc -l | tr -d ' ')
echo "  demo_views: $DEMO_VIEW_COUNT unique images"
echo ""

# 1.1c Run SAM Body Parsing on demo_views + flux_diverse_clean
echo "[1.1c] Running SAM Body Parsing on demo_views + flux_diverse_clean..."
# Clone see-through repo for SAM inference (same as March 31 run)
if [ ! -d "../see-through" ]; then
    git clone https://github.com/shitagaki-lab/see-through.git ../see-through
    pip install -q -r ../see-through/requirements.txt 2>&1 | tail -5 || true
fi

python3 -c "
import sys, os
sys.path.insert(0, '../see-through')
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

from live2d.scrap_model import VALID_BODY_PARTS_V2
from utils.torch_utils import init_model_from_pretrained
from modules.semanticsam import SemanticSam

logger.info('Loading SAM model...')
model = init_model_from_pretrained(
    pretrained_model_name_or_path='24yearsold/l2d_sam_iter2',
    weights_name='checkpoint-18000.pt',
    module_cls=SemanticSam,
    download_from_hf=True,
    model_args=dict(class_num=19)
).to(device='cuda')
model.eval()
logger.info('Model loaded. 19 classes: %s', VALID_BODY_PARTS_V2)

# Process both demo_views and flux_diverse_clean
datasets = [
    Path('data_cloud/demo_views'),
    Path('data_cloud/flux_diverse_clean'),
]

for data_dir in datasets:
    if not data_dir.exists():
        logger.warning('Skipping %s (not found)', data_dir)
        continue
    examples = sorted([d for d in data_dir.iterdir() if d.is_dir() and (d / 'image.png').exists()])
    logger.info('Processing %s: %d images', data_dir.name, len(examples))

    t0 = time.time()
    processed = 0
    skipped = 0

    for i, ex_dir in enumerate(examples):
        out_path = ex_dir / 'sam_segmentation.npz'
        if out_path.exists():
            skipped += 1
            continue

        try:
            img = np.array(Image.open(ex_dir / 'image.png').convert('RGB'))
            with torch.inference_mode():
                preds = model.inference(img)[0]
                masks = (preds > 0).cpu().numpy().astype(np.uint8)  # [19, H, W]

            np.savez_compressed(ex_dir / 'sam_segmentation.npz', masks=masks, classes=VALID_BODY_PARTS_V2)
            processed += 1
        except Exception as e:
            logger.warning('Error on %s: %s', ex_dir.name, e)

        if (i + 1) % 100 == 0:
            rate = processed / max(time.time() - t0, 1)
            logger.info('  %s progress: %d/%d (%d processed, %d skipped, %.1f img/s)',
                        data_dir.name, i+1, len(examples), processed, skipped, rate)

    elapsed = time.time() - t0
    logger.info('%s done: %d processed, %d skipped, %.0fs (%.1f img/s)',
                data_dir.name, processed, skipped, elapsed, processed/max(elapsed,1))
" 2>&1 | tee "$LOG_DIR/sam_inference.log"

# 1.1d Convert SAM 19-class → Strata 22-class for demo_views + flux_diverse_clean
echo ""
echo "[1.1d] Converting SAM labels to 22-class..."
python3 scripts/convert_sam_labels.py --input-dir ./data_cloud/demo_views 2>&1 | tee -a "$LOG_DIR/sam_convert.log"
python3 scripts/convert_sam_labels.py --input-dir ./data_cloud/flux_diverse_clean 2>&1 | tee -a "$LOG_DIR/sam_convert.log"
echo ""

# 1.2 Marigold enrichment (new images only)
echo "[1.2] Marigold enrichment..."
for ds in sora_diverse flux_diverse_clean gemini_li_converted demo_views; do
    ds_dir="./data_cloud/$ds"
    if [ -d "$ds_dir" ]; then
        python3 run_normals_enrich.py --input-dir "$ds_dir" --only-missing --batch-size 1 \
            2>&1 | tee -a "$LOG_DIR/enrich.log"
        echo "  $ds: enriched."
    fi
done
echo ""

# 1.3 Quality filter
echo "[1.3] Quality filter..."
# Re-run quality filter on datasets with new SAM labels
rm -f ./data_cloud/sora_diverse/quality_filter.json
rm -f ./data_cloud/flux_diverse_clean/quality_filter.json
rm -f ./data_cloud/demo_views/quality_filter.json
for ds_dir in ./data_cloud/humanrig ./data_cloud/vroid_cc0 ./data_cloud/meshy_cc0_restructured \
              ./data_cloud/gemini_li_converted ./data_cloud/cvat_annotated; do
    ds_name=$(basename "$ds_dir")
    if [ -f "$ds_dir/quality_filter.json" ]; then
        echo "  $ds_name: exists, skipping."
    elif [ -d "$ds_dir" ]; then
        python3 scripts/filter_seg_quality.py --data-dir "$ds_dir" \
            --min-regions 4 --max-single-region 0.70 --min-foreground 0.05 \
            2>&1 | tee -a "$LOG_DIR/quality_filter.log"
    fi
done
for ds in sora_diverse flux_diverse_clean demo_views; do
    echo "  $ds: quality filter (SAM labels)..."
    python3 scripts/filter_seg_quality.py --data-dir "./data_cloud/$ds" \
        --min-regions 4 --max-single-region 0.70 --min-foreground 0.05 \
        2>&1 | tee -a "$LOG_DIR/quality_filter.log"
done
echo ""

# 1.4 Train seg
echo "[1.4] Training SEGMENTATION model..."
echo "  Config: training/configs/segmentation_a100_run22.yaml (boundary softening + SAM labels)"
echo "  Resuming from run 20 (0.6485 test mIoU)"
echo "  SAM labels on sora_diverse + demo_views + flux_diverse_clean"
echo ""

rm -f checkpoints/segmentation/latest.pt
python3 -m training.train_segmentation \
    --config training/configs/segmentation_a100_run22.yaml \
    --resume "$SEG_CKPT" --reset-epochs \
    2>&1 | tee "$LOG_DIR/seg_train.log"

echo ""

# 1.5 Export + evaluate + upload seg
echo "[1.5] Exporting seg..."
cp checkpoints/segmentation/best.pt checkpoints/segmentation/sam_best.pt

python3 -m training.export_onnx --model segmentation \
    --checkpoint checkpoints/segmentation/sam_best.pt \
    --output ./models/onnx/segmentation_sam.onnx \
    2>&1 | tee "$LOG_DIR/seg_export.log"

python3 -m training.evaluate --model segmentation \
    --checkpoint checkpoints/segmentation/sam_best.pt \
    --dataset-dir ./data_cloud/humanrig \
    --dataset-dir ./data_cloud/vroid_cc0 \
    --dataset-dir ./data_cloud/meshy_cc0_restructured \
    --dataset-dir ./data_cloud/gemini_li_converted \
    --dataset-dir ./data_cloud/cvat_annotated \
    --dataset-dir ./data_cloud/sora_diverse \
    --dataset-dir ./data_cloud/flux_diverse_clean \
    --output-dir ./evaluation_sam \
    2>&1 | tee "$LOG_DIR/seg_evaluate.log"

rclone copy checkpoints/segmentation/sam_best.pt \
    hetzner:strata-training-data/checkpoints_seg_sam/ --transfers 4 --fast-list --size-only -P
rclone copy ./models/onnx/segmentation_sam.onnx \
    hetzner:strata-training-data/models/seg_sam/ --transfers 4 --fast-list --size-only -P
rclone copy ./evaluation_sam/ hetzner:strata-training-data/evaluation_sam/ --transfers 4 --fast-list -P

echo ""
echo "  Seg results:"
grep -E "mIoU|Per-Class" "$LOG_DIR/seg_evaluate.log" 2>/dev/null | head -25 || true
echo ""

# =============================================================================
# PART 2: View Synthesis Bear Chef Fine-tune
# =============================================================================
echo "########################################################"
echo "  PART 2: View Synthesis Bear Chef"
echo "########################################################"
echo ""

# 2.0 Download view synthesis data
echo "[2.0] Downloading view synthesis data..."
mkdir -p checkpoints/view_synthesis data/training

RUN2_CKPT="checkpoints/view_synthesis/run2_best.pt"
if [ ! -f "$RUN2_CKPT" ]; then
    rclone copy hetzner:strata-training-data/checkpoints_view_synthesis_run2/run2_best.pt \
        ./checkpoints/view_synthesis/ --transfers 32 --fast-list -P
fi

# Demo pairs (includes bear chef A-pose)
if [ ! -d "data/training/demo_pairs" ] || [ -z "$(ls data/training/demo_pairs/ 2>/dev/null | head -1)" ]; then
    rclone copy hetzner:strata-training-data/tars/demo_back_view_pairs.tar ./data/tars/ --transfers 32 --fast-list -P
    tar xf ./data/tars/demo_back_view_pairs.tar -C ./data/training/
    rm -f ./data/tars/demo_back_view_pairs.tar
fi
DEMO_COUNT=$(ls -d ./data/training/demo_pairs/pair_* 2>/dev/null | wc -l | tr -d ' ')
echo "  demo_pairs: $DEMO_COUNT"

# Skip 3D pairs to save space — focus on demo pairs
echo ""

# 2.1 Create bear-chef-only dataset for fast fine-tune
echo "[2.1] Creating bear chef only dataset..."
mkdir -p data/training/bear_chef_only

# Find bear chef A-pose pairs (last 30 pairs in demo_pairs)
# Copy them to a separate directory
python3 -c "
from pathlib import Path
import shutil, json

demo = Path('data/training/demo_pairs')
out = Path('data/training/bear_chef_only')

# Find pairs that contain bear chef views (check image similarity or just use last 30)
all_pairs = sorted(demo.glob('pair_demo_*'))
# Bear chef A-pose was added last — find them
bear_count = 0
for pd in reversed(all_pairs):
    vi = pd / 'view_info.json'
    if vi.exists():
        # Check if images are 512x512 bear-like (just copy last 30)
        if bear_count < 30:
            dest = out / pd.name
            if not dest.exists():
                shutil.copytree(pd, dest)
            bear_count += 1
        else:
            break
print(f'Copied {bear_count} bear chef pairs')
"

BC_COUNT=$(ls -d data/training/bear_chef_only/pair_* 2>/dev/null | wc -l | tr -d ' ')
echo "  bear_chef_only: $BC_COUNT pairs"
echo ""

# 2.2 Train (bear chef only — very fast)
echo "[2.2] Training view synthesis (bear chef only)..."

rm -f checkpoints/view_synthesis/latest.pt
python3 -c "
import torch
ckpt = torch.load('$RUN2_CKPT', map_location='cpu', weights_only=False)
ckpt['epoch'] = -1
torch.save(ckpt, 'checkpoints/view_synthesis/latest.pt')
"

# Quick config: bear chef only, 100 epochs, should be very fast
python3 -c "
import yaml
cfg = {
    'model': {'type': 'view_synthesis', 'in_channels': 9, 'out_channels': 4},
    'data': {
        'dataset_dirs': ['./data/training/bear_chef_only'],
        'resolution': 512, 'split_seed': 42,
        'split_ratios': {'train': 0.85, 'val': 0.10, 'test': 0.05},
        'dataset_weights': {'bear_chef_only': 1.0},
    },
    'augmentation': {'horizontal_flip': False, 'color_jitter': {'brightness': 0.1, 'contrast': 0.1, 'saturation': 0.1, 'hue': 0.02}},
    'training': {'batch_size': 16, 'num_workers': 4, 'epochs': 100, 'optimizer': 'adam', 'learning_rate': 1e-4, 'weight_decay': 1e-5, 'scheduler': 'cosine', 'warmup_epochs': 3},
    'loss': {'l1_weight': 1.0, 'perceptual_weight': 0.1},
    'checkpointing': {'save_dir': './checkpoints/view_synthesis', 'early_stopping_patience': 20, 'early_stopping_metric': 'val/l1'},
}
with open('training/configs/view_synthesis_bear_chef_only.yaml', 'w') as f:
    yaml.dump(cfg, f)
print('Config written')
"

python3 -m training.train_view_synthesis \
    --config training/configs/view_synthesis_bear_chef_only.yaml \
    2>&1 | tee "$LOG_DIR/bear_chef_train.log"

echo ""

# 2.3 Export + upload
echo "[2.3] Exporting view synthesis..."
if [ -f "checkpoints/view_synthesis/best.pt" ]; then
    cp checkpoints/view_synthesis/best.pt checkpoints/view_synthesis/bear_chef_best.pt
    python3 -m training.export_onnx --model view_synthesis \
        --checkpoint checkpoints/view_synthesis/bear_chef_best.pt \
        --output ./models/onnx/view_synthesis_bear_chef.onnx \
        2>&1 | tee "$LOG_DIR/vs_export.log"
fi

rclone copy ./checkpoints/view_synthesis/bear_chef_best.pt \
    hetzner:strata-training-data/checkpoints_view_synthesis_bear_chef/ --transfers 4 --fast-list --size-only -P
if [ -f "./models/onnx/view_synthesis_bear_chef.onnx" ]; then
    rclone copy ./models/onnx/view_synthesis_bear_chef.onnx \
        hetzner:strata-training-data/models/view_synthesis_bear_chef/ --transfers 4 --fast-list --size-only -P
fi

rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/april1_${TIMESTAMP}/ --transfers 4 --fast-list -P

echo ""
echo "============================================"
echo "  ALL DONE! $(date)"
echo ""
echo "  Seg (SAM labels):"
grep -E "best mIoU|Training complete" "$LOG_DIR/seg_train.log" 2>/dev/null | tail -2 || true
echo ""
echo "  View Synthesis (bear chef):"
grep -E "best val/l1|Training complete" "$LOG_DIR/bear_chef_train.log" 2>/dev/null | tail -2 || true
echo "============================================"

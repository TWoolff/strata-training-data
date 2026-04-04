#!/usr/bin/env bash
# =============================================================================
# Step 2+3: SAM 3D Body anatomy labeling → retrain SAM 3 seg
#
# Run this AFTER SAM 3 seg epoch 3 finishes on the same A100.
# Assumes: SAM 3 checkpoint exists, GT data already downloaded.
#
# Usage:
#   # After epoch 3 finishes or is killed:
#   ./training/run_sam3d_labels_and_retrain.sh
# =============================================================================
set -euo pipefail

echo "============================================"
echo "  Step 2: SAM 3D Body → Anatomy Labels"
echo "  Step 3: Retrain SAM 3 with enriched data"
echo "  Started: $(date)"
echo "============================================"

# Pre-flight
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "FAIL: CUDA required"; exit 1
fi
echo "GPU: $(python3 -c "import torch; print(torch.cuda.get_device_name(0))")"
echo ""

# Check SAM 3 checkpoint exists
if [ ! -d "/workspace/strata-training-data/checkpoints/sam3_seg/checkpoints" ]; then
    echo "FAIL: No SAM 3 seg checkpoint found. Run epoch 3 first."
    exit 1
fi
echo "SAM 3 checkpoint: OK"

# =============================================================================
# STEP 2: SAM 3D Body → anatomy labels for illustrated characters
# =============================================================================
echo ""
echo "########################################################"
echo "  STEP 2: SAM 3D Body → Anatomy Labels"
echo "########################################################"
echo ""

# 2.1 Install SAM 3D Body
echo "[2.1] Installing SAM 3D Body..."
cd /workspace
if [ ! -d "sam-3d-body" ]; then
    git clone --depth 1 https://github.com/facebookresearch/sam-3d-body.git
fi
cd sam-3d-body
pip install -q pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill \
    pandas rich hydra-core pyrootutils roma joblib seaborn appdirs cython \
    pycocotools loguru optree fvcore huggingface_hub 2>&1 | tail -3
pip install -q 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps 2>&1 | tail -3 || true
cd /workspace/strata-training-data

# 2.2 Download SAM 3D Body checkpoint
echo ""
echo "[2.2] Downloading SAM 3D Body checkpoint..."
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('facebook/sam-3d-body-dinov3', local_dir='/workspace/sam-3d-body/checkpoints/dinov3')
print(f'Downloaded to: {path}')
" 2>&1 | tail -3

# 2.3 Download illustrated character datasets (if not already present)
echo ""
echo "[2.3] Downloading illustrated character data..."
for tar_name in sora_diverse.tar flux_diverse_clean.tar; do
    dir="data_cloud/$(echo $tar_name | sed 's/.tar//')"
    if [ -d "$dir" ] && [ "$(ls "$dir"/ 2>/dev/null | head -1)" ]; then
        echo "  $(basename "$dir"): exists"
    else
        echo "  Downloading $tar_name..."
        rclone copy "hetzner:strata-training-data/tars/$tar_name" ./data/tars/ --transfers 32 --fast-list -P
        if [ -f "./data/tars/$tar_name" ]; then
            tar xf "./data/tars/$tar_name" -C ./data_cloud/ 2>/dev/null
            rm -f "./data/tars/$tar_name"
        fi
    fi
done
# Also get sora_diverse_new
for tar in sora_diverse_new.tar sora_diverse_new2.tar; do
    if rclone ls "hetzner:strata-training-data/tars/$tar" &>/dev/null 2>&1; then
        rclone copy "hetzner:strata-training-data/tars/$tar" ./data/tars/ --transfers 32 --fast-list -P
        tar xf "./data/tars/$tar" -C ./data_cloud/ 2>/dev/null || true
        rm -f "./data/tars/$tar"
    fi
done

# 2.4 Run SAM 3D Body on illustrated characters
echo ""
echo "[2.4] Running SAM 3D Body on illustrated characters..."
echo "  This generates anatomy-perfect segmentation labels."

python3 scripts/batch_sam3d_body_labels.py \
    --input-dirs ./data_cloud/sora_diverse ./data_cloud/flux_diverse_clean \
    --output-dir ./data_cloud/sam3d_body_labels \
    --checkpoint-path /workspace/sam-3d-body/checkpoints/dinov3/model.ckpt \
    --mhr-path /workspace/sam-3d-body/checkpoints/dinov3/assets/mhr_model.pt \
    2>&1 | tee output/model_tests/sam3d_body_labeling.log

LABELED_COUNT=$(find data_cloud/sam3d_body_labels -name "segmentation.png" 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "  SAM 3D Body labels generated: $LABELED_COUNT examples"

# Upload labels to bucket
echo "  Uploading labels to bucket..."
cd data_cloud/sam3d_body_labels
tar cf /tmp/sam3d_body_labels.tar . 2>/dev/null
rclone copy /tmp/sam3d_body_labels.tar hetzner:strata-training-data/tars/ --transfers 4 --fast-list -P
rm -f /tmp/sam3d_body_labels.tar
cd /workspace/strata-training-data

echo ""

# =============================================================================
# STEP 3: Convert new labels to COCO + retrain SAM 3 seg
# =============================================================================
echo "########################################################"
echo "  STEP 3: Retrain SAM 3 with enriched data"
echo "########################################################"
echo ""

# 3.1 Convert SAM 3D Body labels to COCO format and merge with existing
echo "[3.1] Converting SAM 3D Body labels to COCO format..."

# Convert the new labels
python3 scripts/convert_gt_to_coco.py \
    --data-dirs ./data_cloud/sam3d_body_labels \
    --output-dir ./data_cloud/sam3d_body_coco \
    2>&1 | tail -5

# Merge with existing COCO data
echo ""
echo "[3.2] Merging with existing training data..."
python3 -c "
import json, os

# Load existing train data
with open('data_cloud/sam3_coco/train/_annotations.coco.json') as f:
    existing = json.load(f)

# Load new SAM 3D Body data
new_train_path = 'data_cloud/sam3d_body_coco/train/_annotations.coco.json'
if os.path.exists(new_train_path):
    with open(new_train_path) as f:
        new_data = json.load(f)

    # Offset IDs to avoid collision
    max_img_id = max(img['id'] for img in existing['images']) + 1
    max_ann_id = max(ann['id'] for ann in existing['annotations']) + 1

    for img in new_data['images']:
        old_id = img['id']
        img['id'] = old_id + max_img_id
        # Update annotations
        for ann in new_data['annotations']:
            if ann['image_id'] == old_id:
                ann['image_id'] = img['id']
                ann['id'] += max_ann_id

    # Copy images to existing images dir
    import shutil
    src_dir = 'data_cloud/sam3d_body_coco/train/images'
    dst_dir = 'data_cloud/sam3_coco/train/images'
    for img in new_data['images']:
        src = os.path.join(src_dir, img['file_name'])
        # Rename to avoid collisions
        new_name = f'sam3d_{img[\"file_name\"]}'
        dst = os.path.join(dst_dir, new_name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
        img['file_name'] = new_name

    # Merge
    existing['images'].extend(new_data['images'])
    existing['annotations'].extend(new_data['annotations'])

    with open('data_cloud/sam3_coco/train/_annotations.coco.json', 'w') as f:
        json.dump(existing, f)

    print(f'Merged: {len(existing[\"images\"])} total images, {len(existing[\"annotations\"])} annotations')
    print(f'  New SAM 3D Body examples: {len(new_data[\"images\"])}')
else:
    print('No new SAM 3D Body training data to merge')
"

# 3.3 Retrain SAM 3 seg with enriched data (resume from epoch 3 checkpoint)
echo ""
echo "[3.3] Retraining SAM 3 seg with enriched data..."
echo "  Resuming from epoch 3 checkpoint"

# Find the latest checkpoint
LATEST_CKPT=$(ls -t /workspace/strata-training-data/checkpoints/sam3_seg/checkpoints/checkpoint_*.pt 2>/dev/null | head -1)
if [ -z "$LATEST_CKPT" ]; then
    LATEST_CKPT=$(ls -t /workspace/strata-training-data/checkpoints/sam3_seg/checkpoints/checkpoint.pt 2>/dev/null | head -1)
fi
echo "  Checkpoint: $LATEST_CKPT"

cd /workspace/sam3
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 sam3/train/train.py \
    -c configs/sam3_seg_finetune.yaml \
    --use-cluster 0 --num-gpus 1 \
    2>&1 | tee /workspace/strata-training-data/output/model_tests/sam3_seg_retrain.log

# 3.4 Upload results
echo ""
echo "[3.4] Uploading checkpoints..."
cd /workspace/strata-training-data
rclone copy checkpoints/sam3_seg/checkpoints/ \
    hetzner:strata-training-data/checkpoints_sam3_seg_v2/ --transfers 4 --fast-list --size-only -P

echo ""
echo "============================================"
echo "  ALL DONE! $(date)"
echo "============================================"

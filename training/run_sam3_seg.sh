#!/usr/bin/env bash
# =============================================================================
# SAM 3 Segmentation Fine-tune (22-class anatomy)
#
# Requires: A100 with CUDA, 100 GB storage, HuggingFace token
#
# Usage:
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#   export HF_TOKEN=your_token_here
#   ./training/run_sam3_seg.sh
# =============================================================================
set -euo pipefail

echo "============================================"
echo "  SAM 3 Segmentation Fine-tune"
echo "  Started: $(date)"
echo "============================================"

# Pre-flight
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "FAIL: CUDA required"; exit 1
fi
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
echo "GPU: $GPU_NAME"

# Check storage
AVAIL=$(df /workspace --output=avail -B1G | tail -1 | tr -d ' ')
echo "Storage available: ${AVAIL}G"
if [ "$AVAIL" -lt 30 ]; then
    echo "WARNING: Less than 30 GB available. Checkpoint is ~8.7 GB. May fail to save."
fi

# HF login
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: Set HF_TOKEN environment variable"
    echo "  export HF_TOKEN=hf_your_token_here"
    exit 1
fi
python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" 2>/dev/null || {
    pip install -q huggingface_hub
    python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')"
}
echo "HuggingFace: logged in"
echo ""

# =============================================================================
# 1. Install SAM 3
# =============================================================================
echo "[1] Installing SAM 3..."
cd /workspace
if [ ! -d "sam3" ]; then
    git clone --depth 1 https://github.com/facebookresearch/sam3.git
fi
cd sam3
pip install -q -e ".[train]" 2>&1 | tail -3
pip install -q huggingface_hub submitit hydra-submitit-launcher hydra-colorlog fvcore scipy pycocotools einops 2>&1 | tail -3

# Patch fused.py for training (fused ops block gradients)
cat > sam3/perflib/fused.py << 'PYEOF'
import torch

def addmm_act(act_cls, linear, x):
    """Fused linear + activation. Falls back to standard ops when grad is enabled."""
    if torch.is_grad_enabled():
        out = torch.nn.functional.linear(x, linear.weight, linear.bias)
        return act_cls()(out)
    try:
        from sam3.perflib._C import addmm_gelu, addmm_relu
        if act_cls == torch.nn.GELU:
            return addmm_gelu(x, linear.weight, linear.bias)
        elif act_cls == torch.nn.ReLU:
            return addmm_relu(x, linear.weight, linear.bias)
    except ImportError:
        pass
    out = torch.nn.functional.linear(x, linear.weight, linear.bias)
    return act_cls()(out)
PYEOF
echo "  SAM 3 installed + patched"
echo ""

# =============================================================================
# 2. Download GT data
# =============================================================================
echo "[2] Downloading GT segmentation data..."
cd /workspace/strata-training-data
mkdir -p data_cloud data/tars

download_tar() {
    local tar_name="$1"; local extract_dir="$2"
    if [ -d "$extract_dir" ] && [ "$(ls "$extract_dir"/ 2>/dev/null | head -1)" ]; then
        echo "  $(basename "$extract_dir"): exists"
        return 0
    fi
    echo "  Downloading $tar_name..."
    rclone copy "hetzner:strata-training-data/tars/$tar_name" ./data/tars/ --transfers 32 --fast-list -P
    if [ -f "./data/tars/$tar_name" ]; then
        tar xf "./data/tars/$tar_name" -C ./data_cloud/ 2>/dev/null
        rm -f "./data/tars/$tar_name"
    fi
}

download_tar "humanrig.tar" "./data_cloud/humanrig"
download_tar "vroid_cc0.tar" "./data_cloud/vroid_cc0"
download_tar "meshy_cc0_textured_restructured.tar" "./data_cloud/meshy_cc0_restructured"
download_tar "gemini_li_converted.tar" "./data_cloud/gemini_li_converted"
download_tar "cvat_annotated.tar" "./data_cloud/cvat_annotated"

rclone copy hetzner:strata-training-data/data_cloud/frozen_val_test.json \
    ./data_cloud/ --transfers 4 --fast-list -P 2>&1 | tail -1
echo ""

# =============================================================================
# 3. Convert to COCO format
# =============================================================================
echo "[3] Converting GT data to COCO format..."
if [ -f "data_cloud/sam3_coco/train/_annotations.coco.json" ]; then
    echo "  COCO data exists, skipping conversion"
else
    python3 scripts/convert_gt_to_coco.py \
        --data-dirs ./data_cloud/humanrig ./data_cloud/vroid_cc0 \
                    ./data_cloud/meshy_cc0_restructured \
                    ./data_cloud/gemini_li_converted ./data_cloud/cvat_annotated \
        --output-dir ./data_cloud/sam3_coco \
        2>&1 | tail -5
fi

# Split train/val if val doesn't exist
if [ ! -f "data_cloud/sam3_coco/val/_annotations.coco.json" ]; then
    echo "  Splitting train/val (90/10)..."
    python3 -c "
import json, random, os

with open('data_cloud/sam3_coco/train/_annotations.coco.json') as f:
    data = json.load(f)

random.seed(42)
images = data['images']
random.shuffle(images)
n_val = len(images) // 10
val_images = images[:n_val]
train_images = images[n_val:]

val_ids = {img['id'] for img in val_images}
train_anns = [a for a in data['annotations'] if a['image_id'] not in val_ids]
val_anns = [a for a in data['annotations'] if a['image_id'] in val_ids]

with open('data_cloud/sam3_coco/train/_annotations.coco.json', 'w') as f:
    json.dump({'images': train_images, 'categories': data['categories'], 'annotations': train_anns}, f)

os.makedirs('data_cloud/sam3_coco/val/images', exist_ok=True)
for img in val_images:
    src = os.path.abspath(f'data_cloud/sam3_coco/train/images/{img[\"file_name\"]}')
    dst = f'data_cloud/sam3_coco/val/images/{img[\"file_name\"]}'
    if os.path.exists(src) and not os.path.exists(dst):
        os.symlink(src, dst)

with open('data_cloud/sam3_coco/val/_annotations.coco.json', 'w') as f:
    json.dump({'images': val_images, 'categories': data['categories'], 'annotations': val_anns}, f)

print(f'  Train: {len(train_images)} images, {len(train_anns)} annotations')
print(f'  Val: {len(val_images)} images, {len(val_anns)} annotations')
"
fi

TRAIN_COUNT=$(python3 -c "import json; d=json.load(open('data_cloud/sam3_coco/train/_annotations.coco.json')); print(len(d['images']))" 2>/dev/null || echo "0")
VAL_COUNT=$(python3 -c "import json; d=json.load(open('data_cloud/sam3_coco/val/_annotations.coco.json')); print(len(d['images']))" 2>/dev/null || echo "0")
echo "  COCO data: train=$TRAIN_COUNT, val=$VAL_COUNT"
echo ""

# =============================================================================
# 4. Copy config + fix save frequency
# =============================================================================
echo "[4] Setting up training config..."
cp /workspace/strata-training-data/training/configs/sam3_seg_finetune.yaml \
   /workspace/sam3/sam3/train/configs/sam3_seg_finetune.yaml
# Save checkpoint every epoch
sed -i 's/save_freq: 5/save_freq: 1/' /workspace/sam3/sam3/train/configs/sam3_seg_finetune.yaml
echo "  Config ready (save_freq: 1)"
echo ""

# Check storage before training
AVAIL=$(df /workspace --output=avail -B1G | tail -1 | tr -d ' ')
echo "  Storage before training: ${AVAIL}G available"
if [ "$AVAIL" -lt 20 ]; then
    echo "  WARNING: Low storage. Cleaning up source data to make room for checkpoint..."
    rm -rf /workspace/strata-training-data/data_cloud/humanrig
    rm -rf /workspace/strata-training-data/data_cloud/meshy_cc0_restructured
    rm -rf /workspace/strata-training-data/data_cloud/vroid_cc0
    rm -rf /workspace/strata-training-data/data_cloud/gemini_li_converted
    rm -rf /workspace/strata-training-data/data_cloud/cvat_annotated
    AVAIL=$(df /workspace --output=avail -B1G | tail -1 | tr -d ' ')
    echo "  After cleanup: ${AVAIL}G available"
fi

# =============================================================================
# 5. Train SAM 3 segmentation
# =============================================================================
echo "[5] Training SAM 3 segmentation (22-class anatomy)..."
echo "  Model: 840M params"
echo "  Data: $TRAIN_COUNT train, $VAL_COUNT val"
echo "  Epochs: 20 (expect 2-3 to be sufficient)"
echo "  Time per epoch: ~9 hrs"
echo "  Checkpoint size: ~8.7 GB"
echo ""

cd /workspace/sam3
mkdir -p /workspace/strata-training-data/output/model_tests

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 sam3/train/train.py \
    -c configs/sam3_seg_finetune.yaml \
    --use-cluster 0 --num-gpus 1 \
    2>&1 | tee /workspace/strata-training-data/output/model_tests/sam3_seg_train.log

# =============================================================================
# 6. Upload results
# =============================================================================
echo ""
echo "[6] Uploading checkpoints and logs..."
cd /workspace/strata-training-data

rclone copy /workspace/strata-training-data/checkpoints/sam3_seg/ \
    hetzner:strata-training-data/checkpoints_sam3_seg/ --transfers 4 --fast-list --size-only -P
rclone copy /workspace/strata-training-data/output/model_tests/ \
    hetzner:strata-training-data/model_tests_april3/ --transfers 4 --fast-list -P

echo ""
echo "============================================"
echo "  DONE! $(date)"
echo "============================================"

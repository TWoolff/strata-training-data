#!/usr/bin/env bash
# =============================================================================
# Strata — Segmentation Fine-tune using Dr. Li's SAM-HQ encoder (See-Through v3)
#
# Leverages the pretrained SAM-HQ encoder (trained on 9K illustrated characters)
# and fine-tunes OUR 22-class anatomy decoder on our existing seg training data.
#
# Strategy:
#   - Freeze Dr. Li's SAM-HQ encoder (the asset we want)
#   - Train fresh 22-class mask decoder from scratch
#   - Use our existing 22-class PNG masks (no format change needed)
#
# Est. time: 4-6 hrs on A100 40GB
# Expected: 0.65 → 0.72-0.78 mIoU (big jump if Li's encoder helps)
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/seethrough_seg_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata — See-Through SAM-HQ Seg Fine-tune"
echo "  22-class anatomy heads on Dr. Li's encoder"
echo "  Started: $(date)"
echo "============================================"

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "FAIL: CUDA not available"
    exit 1
fi
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
GPU_MEM=$(python3 -c "import torch; p=torch.cuda.get_device_properties(0); m=getattr(p,'total_memory',getattr(p,'total_mem',0)); print(f'{m/1e9:.0f}GB')")
echo "OK: CUDA — $GPU_NAME ($GPU_MEM)"

if ! rclone lsd hetzner:strata-training-data/ &>/dev/null; then
    echo "FAIL: rclone bucket not reachable"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: Clone See-Through and install
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Cloning See-Through repo + installing deps..."
if [[ ! -d /workspace/see-through ]]; then
    cd /workspace
    git clone https://github.com/shitagaki-lab/see-through.git
    cd /workspace/strata-training-data
fi

cd /workspace/see-through
# Install main deps
pip install -q -r requirements.txt 2>&1 | tail -5
pip install -q wandb 2>&1 | tail -3

# ---------------------------------------------------------------------------
# Step 2: Download training data from bucket
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Downloading seg training data..."
mkdir -p /workspace/seg_data

# Key tars (from CLAUDE.md bucket contents):
TARS=(
    "humanrig.tar"                           # 16.8G GT 22-class T-pose renders
    "gemini_li_converted.tar"                # 223M Dr Li's 694 expert-labeled
    "sora_diverse.tar"                       # 380M illustrated chars
    "flux_diverse_clean.tar"                 # 300M FLUX chars
    "vroid_cc0.tar"                          # 203M VRoid GT 22-class
    "meshy_cc0_textured_restructured.tar"    # 2.8G Meshy textured (15K examples)
    "cvat_annotated.tar"                     # 9M hand-annotated
)

for TAR in "${TARS[@]}"; do
    if [[ ! -f "/workspace/seg_data/$TAR" ]]; then
        echo "  Downloading $TAR..."
        rclone copyto "hetzner:strata-training-data/tars/$TAR" \
            "/workspace/seg_data/$TAR" --no-check-dest -P 2>&1 | tail -3
    fi
    echo "  Extracting $TAR..."
    tar xf "/workspace/seg_data/$TAR" -C /workspace/seg_data/
    rm "/workspace/seg_data/$TAR"
done

# Download frozen val/test splits
rclone copy hetzner:strata-training-data/data_cloud/frozen_val_test.json \
    /workspace/seg_data/ --no-check-dest 2>&1 | tail -1

ls /workspace/seg_data/ | head -20

# ---------------------------------------------------------------------------
# Step 3: Download Li's pretrained SAM body parsing checkpoint (for warm-start)
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Pre-downloading SAM-HQ encoder weights..."
mkdir -p /workspace/weights

# This will be auto-downloaded by sam_model_registry on first run,
# but let's pre-cache to avoid delays during training.
python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
# SAM-HQ ViT-B (~375MB) — best fit for A100 40GB with encoder frozen
path = hf_hub_download(repo_id='lkeab/hq-sam', filename='sam_hq_vit_b.pth')
target = '/workspace/weights/sam_hq_vit_b.pth'
if not os.path.exists(target):
    shutil.copy(path, target)
print(f'SAM-HQ ViT-B weights ready at {target}')
"

# Optionally: download Dr Li's body-parsing checkpoint for decoder warm-start
# (won't match our 22-class shape but strict=False loading accepts it)
python3 -c "
from huggingface_hub import hf_hub_download
try:
    path = hf_hub_download(repo_id='24yearsold/l2d_sam_iter2', filename='checkpoint-18000.pt')
    import shutil
    shutil.copy(path, '/workspace/weights/li_sam_iter2.pt')
    print('Dr Li body-parsing checkpoint ready for warm-start')
except Exception as e:
    print(f'Warning: could not fetch Li checkpoint ({e}) — will train from random decoder')
"

# ---------------------------------------------------------------------------
# Step 4: Build config with our 22-class schema + dataset list
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Building config + dataset list..."

# Generate a flat list of image paths pointing at our 22-class mask-having data
# Dataset loader expects: image path per line, plus `<stem>_faceseg.png` masks
# OR pass `label_dir` in config if masks have different naming.
#
# Our data has:
#   <char>/<pose>/image.png and <char>/<pose>/segmentation.png
# We need to create symlinks or a manifest.

python3 << 'PYEOF'
import os
from pathlib import Path
import json

data_root = Path("/workspace/seg_data")
train_list = []
val_list = []

# Load frozen val/test splits
splits_file = data_root / "frozen_val_test.json"
if splits_file.exists():
    splits = json.loads(splits_file.read_text())
    val_chars = set(splits.get("val", []))
    test_chars = set(splits.get("test", []))
else:
    val_chars = set()
    test_chars = set()

# Walk all example dirs that have image.png + segmentation.png
for ex_dir in data_root.rglob("*"):
    if not ex_dir.is_dir():
        continue
    img = ex_dir / "image.png"
    seg = ex_dir / "segmentation.png"
    if not (img.exists() and seg.exists()):
        continue

    # Create _faceseg.png symlink that the dataset loader expects
    faceseg = ex_dir / "image_faceseg.png"
    if not faceseg.exists():
        try:
            faceseg.symlink_to("segmentation.png")
        except Exception:
            pass

    char_id = ex_dir.parent.name
    if char_id in val_chars:
        val_list.append(str(img))
    elif char_id in test_chars:
        continue  # held-out
    else:
        train_list.append(str(img))

train_txt = data_root / "train.txt"
val_txt = data_root / "val.txt"
train_txt.write_text("\n".join(train_list) + "\n")
val_txt.write_text("\n".join(val_list) + "\n")
print(f"Train: {len(train_list)} images")
print(f"Val:   {len(val_list)} images")
PYEOF

# Generate config override
cat > /workspace/seethrough_strata_config.yaml << 'YAML_EOF'
# 22-class anatomy fine-tune on Dr. Li's SAM-HQ encoder
# Strata schema — replaces 19-class clothing heads

tag_list:
  - background
  - head
  - neck
  - chest
  - spine
  - hips
  - shoulder_l
  - upper_arm_l
  - forearm_l
  - hand_l
  - shoulder_r
  - upper_arm_r
  - forearm_r
  - hand_r
  - upper_leg_l
  - lower_leg_l
  - foot_l
  - upper_leg_r
  - lower_leg_r
  - foot_r
  - accessory
  - hair_back

model_args:
  model_type: b_hq           # ViT-B HQ — fits A100 40GB
  class_num: 22              # our anatomy schema
  fix_img_en: true           # freeze encoder (the asset we want)
  fix_prompt_en: true
  fix_mask_de: false         # train decoder from scratch
  sam_checkpoint: /workspace/weights/sam_hq_vit_b.pth

dataset:
  train_list: /workspace/seg_data/train.txt
  val_list: /workspace/seg_data/val.txt
  target_size: 1024
  label_suffix: _faceseg.png

training:
  learning_rate: 1.0e-4
  lr_scheduler: constant_with_warmup
  lr_warmup_steps: 500
  train_batch_size: 1
  gradient_accumulation_steps: 16   # effective batch 16
  mixed_precision: bf16
  loss_type: samhq                  # BCE + Dice
  max_train_steps: 8000
  save_steps: 1000
  validate_steps: 500

output_dir: /workspace/seethrough_checkpoints
YAML_EOF

# ---------------------------------------------------------------------------
# Step 5: Run training
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Training (max 8000 steps, ~4-6 hrs)..."

cd /workspace/see-through
python3 training/train/train_partseg.py \
    --config /workspace/seethrough_strata_config.yaml \
    2>&1 | tee "$LOG_DIR/train.log"

# ---------------------------------------------------------------------------
# Upload results
# ---------------------------------------------------------------------------
echo ""
echo "[final] Uploading checkpoints..."
CKPT_DIR="/workspace/seethrough_checkpoints"
if [[ -d "$CKPT_DIR" ]]; then
    rclone copy "$CKPT_DIR" \
        "hetzner:strata-training-data/checkpoints_seethrough_seg_${TIMESTAMP}/" \
        --transfers 8 -P 2>&1 | tail -3
    echo "OK: uploaded to bucket"
fi

rclone copy "$LOG_DIR" \
    "hetzner:strata-training-data/logs/seethrough_seg_${TIMESTAMP}/" \
    --transfers 4 -P 2>&1 | tail -1

echo ""
echo "============================================"
echo "  Done! Finished: $(date)"
echo "============================================"

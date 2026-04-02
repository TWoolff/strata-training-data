#!/usr/bin/env bash
# =============================================================================
# Test new models on illustrated characters
#
# 1. SAM 3D Body — single image → rigged 3D mesh (10-15 min)
# 2. TRELLIS.2 — single image → 3D mesh + PBR materials (10-15 min)
# 3. SAM 3 seg — text-prompted anatomy segmentation (10 min)
#
# Usage:
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#   ./training/run_model_tests.sh
# =============================================================================
set -euo pipefail

echo "============================================"
echo "  Model Tests: SAM 3D Body + TRELLIS.2 + SAM 3"
echo "  Started: $(date)"
echo "============================================"

# Pre-flight
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "FAIL: CUDA required"; exit 1
fi
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
echo "GPU: $GPU_NAME"
echo ""

# Download test images (bear chef + a few humanoid characters)
echo "[0] Preparing test images..."
mkdir -p test_images output/model_tests
rclone copy hetzner:strata-training-data/tars/demo_back_view_pairs.tar ./data/tars/ --transfers 32 --fast-list -P 2>&1 | tail -3
mkdir -p data/training
tar xf ./data/tars/demo_back_view_pairs.tar -C ./data/training/ 2>/dev/null
rm -f ./data/tars/demo_back_view_pairs.tar

# Extract a few test images (bear chef + 5 random humanoid characters)
python3 -c "
from pathlib import Path
import shutil
src = Path('data/training/demo_pairs')
dst = Path('test_images')
# Bear chef (last pairs)
pairs = sorted(src.glob('pair_demo_*'))
# Get 5 diverse characters from different parts of the dataset
test_pairs = [pairs[-1], pairs[-10], pairs[0], pairs[100], pairs[500], pairs[1000]]
for p in test_pairs:
    front = p / 'front.png'
    if front.exists():
        shutil.copy2(front, dst / f'{p.name}_front.png')
        print(f'Copied {p.name}')
"
echo "  Test images: $(ls test_images/*.png | wc -l)"
echo ""

# =============================================================================
# TEST 1: SAM 3D Objects (single image + mask → 3D Gaussian splat with texture)
# =============================================================================
echo "########################################################"
echo "  TEST 1: SAM 3D Objects (image → 3D textured model)"
echo "########################################################"
echo ""

echo "[1.1] Installing SAM 3D Objects..."
if [ ! -d "../sam-3d-objects" ]; then
    git clone --depth 1 https://github.com/facebookresearch/sam-3d-objects.git ../sam-3d-objects
fi

cd ../sam-3d-objects
pip install -q -r requirements.inference.txt 2>&1 | tail -5
pip install -q -e . 2>&1 | tail -3
cd /workspace/strata-training-data

echo ""
echo "[1.2] Downloading SAM 3D Objects checkpoint..."
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('facebook/sam-3d-objects', local_dir='../sam-3d-objects/checkpoints/hf')
print(f'Downloaded to: {path}')
" 2>&1 | tail -5

echo ""
echo "[1.3] Running SAM 3D Objects inference..."
python3 -c "
import sys, os
sys.path.insert(0, '../sam-3d-objects')
sys.path.insert(0, '../sam-3d-objects/notebook')
from inference import Inference, load_image
from PIL import Image
import numpy as np
import glob

# Load model
config_path = '../sam-3d-objects/checkpoints/hf/pipeline.yaml'
inference = Inference(config_path, compile=False)

images = sorted(glob.glob('test_images/*.png'))
out_dir = 'output/model_tests/sam3d_objects'
os.makedirs(out_dir, exist_ok=True)

for img_path in images:
    name = os.path.basename(img_path).replace('.png', '')
    print(f'Processing {name}...')
    try:
        # Load as RGBA — alpha channel IS the mask for illustrated characters
        img = Image.open(img_path).convert('RGBA')
        img_np = np.array(img)

        # Create mask from alpha channel (character = 1, background = 0)
        mask = (img_np[:, :, 3] > 128).astype(np.uint8)

        # Run 3D reconstruction
        output = inference(img_np, mask, seed=42)

        # Save Gaussian splat
        output['gs'].save_ply(f'{out_dir}/{name}.ply')
        print(f'  Saved {name}.ply')
    except Exception as e:
        print(f'  Error: {e}')
" 2>&1 | tee output/model_tests/sam3d_objects.log

echo ""
echo "  SAM 3D Objects results: $(ls output/model_tests/sam3d_objects/*.ply 2>/dev/null | wc -l) splats"
echo ""

# =============================================================================
# TEST 1b: SAM 3D Body (single image → rigged 3D mesh with skeleton)
# =============================================================================
echo "########################################################"
echo "  TEST 1b: SAM 3D Body (image → rigged skeleton mesh)"
echo "########################################################"
echo ""

echo "[1b.1] Installing SAM 3D Body..."
if [ ! -d "../sam-3d-body" ]; then
    git clone --depth 1 https://github.com/facebookresearch/sam-3d-body.git ../sam-3d-body
fi

cd ../sam-3d-body
pip install -q pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill \
    pandas rich hydra-core pyrootutils roma joblib seaborn appdirs cython \
    xtcocotools loguru optree fvcore pycocotools huggingface_hub 2>&1 | tail -3
pip install -q 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps 2>&1 | tail -3
cd /workspace/strata-training-data

echo ""
echo "[1b.2] Downloading SAM 3D Body checkpoint..."
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('facebook/sam-3d-body-dinov3', local_dir='../sam-3d-body/checkpoints/dinov3')
print(f'Downloaded to: {path}')
" 2>&1 | tail -5

echo ""
echo "[1b.3] Running SAM 3D Body inference..."
cd ../sam-3d-body
python3 demo.py \
    --image_folder /workspace/strata-training-data/test_images \
    --checkpoint_path ./checkpoints/dinov3 \
    --output_folder /workspace/strata-training-data/output/model_tests/sam3d_body \
    2>&1 | tee /workspace/strata-training-data/output/model_tests/sam3d_body.log
cd /workspace/strata-training-data

echo ""
echo "  SAM 3D Body results: $(ls output/model_tests/sam3d_body/*.jpg 2>/dev/null | wc -l) renders"
echo ""

# =============================================================================
# TEST 2: TRELLIS.2
# =============================================================================
echo "########################################################"
echo "  TEST 2: TRELLIS.2 (single image → 3D mesh + PBR)"
echo "########################################################"
echo ""

echo "[2.1] Installing TRELLIS.2..."
if [ ! -d "../TRELLIS.2" ]; then
    git clone --depth 1 --recursive https://github.com/microsoft/TRELLIS.2.git ../TRELLIS.2
fi

cd ../TRELLIS.2
# Install deps (this takes a while)
. ./setup.sh --basic --flash-attn --nvdiffrast --o-voxel 2>&1 | tail -10
cd /workspace/strata-training-data

echo ""
echo "[2.2] Running TRELLIS.2 inference..."
cd ../TRELLIS.2
python3 -c "
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
pipeline.cuda()

import glob
images = sorted(glob.glob('/workspace/strata-training-data/test_images/*.png'))
out_dir = '/workspace/strata-training-data/output/model_tests/trellis2'
os.makedirs(out_dir, exist_ok=True)

for img_path in images:
    name = os.path.basename(img_path).replace('.png', '')
    print(f'Processing {name}...')
    try:
        image = Image.open(img_path)
        mesh = pipeline.run(image)[0]
        mesh.simplify(16777216)
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces,
            attr_volume=mesh.attrs, coords=mesh.coords,
            attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=100000, texture_size=2048,
            remesh=True, remesh_band=1, remesh_project=0, verbose=False
        )
        glb.export(f'{out_dir}/{name}.glb', extension_webp=True)
        print(f'  Saved {name}.glb')
    except Exception as e:
        print(f'  Error: {e}')
" 2>&1 | tee /workspace/strata-training-data/output/model_tests/trellis2.log
cd /workspace/strata-training-data

echo ""
echo "  TRELLIS.2 results: $(ls output/model_tests/trellis2/*.glb 2>/dev/null | wc -l) meshes"
echo ""

# =============================================================================
# TEST 3: SAM 3 Segmentation
# =============================================================================
echo "########################################################"
echo "  TEST 3: SAM 3 (text-prompted anatomy segmentation)"
echo "########################################################"
echo ""

echo "[3.1] Installing SAM 3..."
if [ ! -d "../sam3" ]; then
    git clone --depth 1 https://github.com/facebookresearch/sam3.git ../sam3
fi
cd ../sam3
pip install -q -e . 2>&1 | tail -3
cd /workspace/strata-training-data

echo ""
echo "[3.2] Running SAM 3 anatomy segmentation..."
python3 -c "
import torch
import numpy as np
from PIL import Image
import os, glob

# Load SAM 3
from sam3 import build_sam3, SamPredictor
model = build_sam3(checkpoint=None)  # Will download from HF
model.cuda()
predictor = SamPredictor(model)

body_parts = [
    'head', 'neck', 'chest', 'spine', 'hips',
    'left shoulder', 'left upper arm', 'left forearm', 'left hand',
    'right shoulder', 'right upper arm', 'right forearm', 'right hand',
    'left upper leg', 'left lower leg', 'left foot',
    'right upper leg', 'right lower leg', 'right foot',
]

images = sorted(glob.glob('test_images/*.png'))
out_dir = 'output/model_tests/sam3_seg'
os.makedirs(out_dir, exist_ok=True)

for img_path in images:
    name = os.path.basename(img_path).replace('.png', '')
    print(f'Processing {name}...')
    try:
        image = np.array(Image.open(img_path).convert('RGB'))
        predictor.set_image(image)

        # Try text-prompted segmentation for each body part
        h, w = image.shape[:2]
        result = np.zeros((h, w), dtype=np.uint8)

        for i, part in enumerate(body_parts):
            masks = predictor.predict_with_text(part)
            if masks is not None and len(masks) > 0:
                mask = masks[0]  # best mask
                result[mask > 0] = i + 1  # 1-indexed (0 = background)

        Image.fromarray(result).save(f'{out_dir}/{name}_seg.png')
        print(f'  Saved {name}_seg.png ({np.unique(result).size} classes)')
    except Exception as e:
        print(f'  Error: {e}')
" 2>&1 | tee output/model_tests/sam3_seg.log

echo ""
echo "  SAM 3 seg results: $(ls output/model_tests/sam3_seg/*.png 2>/dev/null | wc -l) masks"
echo ""

# =============================================================================
# Upload results
# =============================================================================
echo "[4] Uploading results..."
rclone copy output/model_tests/ hetzner:strata-training-data/model_tests_april2/ --transfers 4 --fast-list -P

echo ""
echo "============================================"
echo "  ALL TESTS DONE! $(date)"
echo "============================================"

# Strata Training Pipeline

Bridges the data generation pipeline (`pipeline/`) and the Strata app's ONNX runtime (`src-tauri/src/ai/`). Trains three PyTorch models and exports them to ONNX for deployment.

## The Three Models

### 1. Segmentation Model
- **Input**: RGB image `[1, 3, 512, 512]` normalized with ImageNet mean/std
- **Output**: Per-pixel class logits `[1, 22, 512, 512]` (20 body regions + background + accessory)
- **Architecture**: MobileNetV3-Large backbone + DeepLabV3 head
- **ONNX contract**: `segmentation.onnx` loaded by `src-tauri/src/ai/segmentation.rs`
- **Preprocessing (Rust)**: Resize to 512x512, normalize with `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

### 2. Joint Prediction Model
- **Input**: RGB image `[1, 3, 512, 512]` normalized with ImageNet mean/std
- **Output**: Joint heatmaps `[1, 20, 128, 128]` (20 joints in BONE_ORDER)
- **Architecture**: MobileNetV3-Large backbone + heatmap regression head
- **ONNX contract**: `joints.onnx` loaded by `src-tauri/src/ai/joints.rs`
- **BONE_ORDER** (must match Rust's `BONE_NAMES`): `[hips, spine, chest, neck, head, shoulder_l, upper_arm_l, forearm_l, hand_l, shoulder_r, upper_arm_r, forearm_r, hand_r, upper_leg_l, lower_leg_l, foot_l, upper_leg_r, lower_leg_r, foot_r, hair_back]`

### 3. Weight Prediction Model
- **Input**: RGB image `[1, 3, 512, 512]` normalized with ImageNet mean/std
- **Output**: Per-pixel bone weights `[1, 20, 512, 512]` (weight per bone, sums to 1.0)
- **Architecture**: MobileNetV3-Large backbone + weight regression head
- **ONNX contract**: `weights.onnx` loaded by `src-tauri/src/ai/weights.rs`

## Setup

```bash
# Create virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r training/requirements.txt
```

### GPU Requirements

- **Development/debugging**: CPU is fine for small batches
- **Full training run**: NVIDIA GPU with 8+ GB VRAM (RTX 3070 or better)
- **Cloud cost**: ~$10-20 per model on Lambda Labs / RunPod (A10G instance, ~4-8 hours)
- **All three models**: ~$30-60 total cloud cost

## Training Commands

```bash
# Train segmentation model
python -m training.train --config training/configs/segmentation.yaml

# Train joint prediction model
python -m training.train --config training/configs/joints.yaml

# Train weight prediction model
python -m training.train --config training/configs/weights.yaml
```

## ONNX Export

```bash
# Export trained model to ONNX (after training completes)
python -m training.export --checkpoint checkpoints/segmentation/best.pt --output segmentation.onnx
python -m training.export --checkpoint checkpoints/joints/best.pt --output joints.onnx
python -m training.export --checkpoint checkpoints/weights/best.pt --output weights.onnx
```

## Deploying to Strata App

1. Train model and export to ONNX (see above)
2. Validate ONNX model: `python -c "import onnx; onnx.checker.check_model('segmentation.onnx')"`
3. Copy ONNX file to `strata/src-tauri/resources/models/`
4. The Rust runtime (`src-tauri/src/ai/runtime.rs`) loads models automatically on startup

## Directory Structure

```
training/
├── __init__.py
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── configs/
│   ├── segmentation.yaml         # Segmentation model config
│   ├── joints.yaml               # Joint prediction config
│   └── weights.yaml              # Weight prediction config
├── models/
│   └── __init__.py               # Model architectures (future)
├── data/
│   ├── __init__.py
│   ├── transforms.py             # Augmentation with L/R region swap
│   └── split_loader.py           # Character-level dataset splits
└── utils/
    └── __init__.py               # Training utilities (future)
```

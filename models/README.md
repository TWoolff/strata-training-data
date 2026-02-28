# ONNX Models for Pose Estimation

This directory stores ONNX model weights for the 2D pose estimation enrichment pipeline (`run_enrich.py`). Model files (`*.onnx`) are git-ignored — download them manually before running enrichment.

## Required Models

The enrichment pipeline uses RTMPose via [rtmlib](https://github.com/Tau-J/rtmlib) and requires two models:

1. **Person detector** (YOLOX) — detects bounding boxes around people in the image
2. **Pose estimator** (RTMPose) — predicts 17 COCO keypoints within each bounding box

## Recommended Models

| Model | Type | Input Size | AP | Download |
|-------|------|-----------|-----|----------|
| YOLOX-m | Detector | 640x640 | 59.1 | [yolox_m_8xb8-300e_humanart-c2c7a14a.zip](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip) |
| RTMPose-m | Pose | 256x192 | 74.9 | [rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip) |

Use the HumanArt-trained detector for best results on anime/cartoon characters.

## Download Instructions

```bash
cd models/

# Download and extract the detector
wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip
unzip yolox_m_8xb8-300e_humanart-c2c7a14a.zip

# Download and extract the pose model
wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip
unzip rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip
```

After extracting, you should have `.onnx` files in this directory.

## Usage

```bash
python run_enrich.py \
    --input_dir ./output/fbanimehq \
    --det_model ./models/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx \
    --pose_model ./models/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx \
    --device cpu
```

## Alternative Models

For faster processing (lower accuracy):

| Model | Type | Input Size | AP | Download |
|-------|------|-----------|-----|----------|
| YOLOX-nano | Detector | 416x416 | 38.9 | [yolox_nano...zip](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_nano_8xb8-300e_humanart-40f6f0d0.zip) |
| RTMPose-s | Pose | 256x192 | 69.7 | [rtmpose-s...zip](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip) |

For higher accuracy (slower):

| Model | Type | Input Size | AP | Download |
|-------|------|-----------|-----|----------|
| YOLOX-x | Detector | 640x640 | 61.3 | [yolox_x...zip](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip) |
| RTMPose-l | Pose | 384x288 | 78.3 | [rtmpose-l...zip](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.zip) |

When using non-default model sizes, pass `--det_input_size` and `--pose_input_size` to `run_enrich.py`.

## License

All models are Apache 2.0 licensed (from the MMPose project).

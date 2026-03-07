# Anime Character AI Training Data

Pipeline for generating labeled training data for 6 ONNX AI models that understand 2D illustrated characters — segmenting body regions, predicting skeleton joints, estimating skinning weights, inpainting occluded regions, and synthesizing novel views.

![Benchmark overview — 7 test characters across all model outputs](docs/training01_overview.png)
*Columns: Original | Segmentation (22 regions) | Joint Prediction (20 bones) | Draw Order | Surface Normals | Depth*

## Models

| # | Model | Architecture | Score | What it does |
|---|-------|-------------|-------|-------------|
| 1 | **Segmentation** | DeepLabV3+ MobileNetV3 (multi-head) | 0.5453 mIoU | 22-class body region map + depth + surface normals from a single 512x512 image |
| 2 | **Joint Refinement** | MobileNetV3 + regression | 0.001287 offset error | 20 skeleton joint positions + confidence from a 512x512 image |
| 3 | **Weight Prediction** | Per-vertex MLP | 0.0840 MAE | Skinning weights for 20 bones from vertex features, with optional visual encoder features |
| 4 | **Inpainting** | U-Net | In progress | Fills occluded body regions in 2D character paintings |
| 5 | **Texture Inpainting** | Diffusion | Planned | Fills unobserved texture regions when unwrapping 2D art onto 3D mesh |
| 6 | **Novel View Synthesis** | Multi-view diffusion | Planned | Generates unseen views (back, side, 3/4) from reference views |

Scores from first A100 training run (March 2026). Model 1's depth and normals heads will be added in the next run, distilled from Marigold LCM.

## Training Data

~870K+ files across 15 datasets, ~166 GB total. Mix of 3D rendered characters (Mixamo, HumanRig, UniRig, NOVA-Human, VRoid) and 2D illustrated characters (anime-segmentation, FBAnimeHQ, Live2D, CoNR).

| Annotation | Examples | Sources |
|-----------|---------|---------|
| 22-class body region masks | ~6K | Mixamo pipeline, Live2D composites |
| Skeleton joints (20 bones) | ~207K | Ground truth + RTMPose enriched |
| Skinning weights (20 bones) | ~26.5K | HumanRig + UniRig + Mixamo |
| Draw order depth | ~10K | Mixamo Z-depth, Live2D fragment stacking |
| Surface normals | ~14K | Marigold LCM enriched |
| Foreground/background masks | ~55K | anime-segmentation, curated diverse |

### 22 Body Region Classes

| ID | Region | ID | Region |
|----|--------|----|--------|
| 0 | background | 11 | upper_arm_r |
| 1 | head | 12 | forearm_r |
| 2 | neck | 13 | hand_r |
| 3 | chest | 14 | upper_leg_l |
| 4 | spine | 15 | lower_leg_l |
| 5 | hips | 16 | foot_l |
| 6 | shoulder_l | 17 | upper_leg_r |
| 7 | upper_arm_l | 18 | lower_leg_r |
| 8 | forearm_l | 19 | foot_r |
| 9 | hand_l | 20 | accessory |
| 10 | shoulder_r | 21 | hair_back |

## Pipeline

### Rendering (Blender)

Generates labeled training examples from rigged 3D characters:

```bash
blender --background --python run_pipeline.py -- \
  --input_dir ./data/fbx/ \
  --pose_dir ./data/poses/ \
  --output_dir ./output/segmentation/ \
  --styles flat,cel,pixel,painterly,sketch,unlit \
  --resolution 512 \
  --poses_per_character 20
```

- Orthographic camera, auto-framed with 10% padding
- 22 emission-only materials for pixel-perfect segmentation masks
- Multi-angle rendering: front (0°), three-quarter (45°), side (90°), three-quarter-back (135°), back (180°)
- Style augmentation: flat, cel/toon, unlit (render-time) + pixel art, painterly, sketch (post-render)

Requires Blender 4.0+.

### Enrichment

Adds pseudo-labels to datasets that lack certain annotations:

```bash
# Add 22-class segmentation masks using trained Model 1
python run_seg_enrich.py --input-dir ./data_cloud/anime_seg --checkpoint checkpoints/segmentation/best.pt

# Add surface normals + depth using Marigold LCM
python run_normals_enrich.py --input-dir ./data_cloud/segmentation --only-missing
```

### External Dataset Ingestion

14 adapter scripts in `ingest/` convert external datasets to a unified format:

```
{example_id}/
├── image.png           # Character render (512x512)
├── segmentation.png    # Per-pixel region IDs (grayscale, value = class ID)
├── draw_order.png      # Per-pixel depth (0=back, 255=front)
├── normals.png         # Surface normal map (RGB)
├── depth.png           # Depth map (grayscale)
├── joints.json         # 20 skeleton joint positions
└── metadata.json       # Source, style, pose, camera angle
```

### Training

```bash
# Set up A100 cloud instance
./training/cloud_setup.sh lean

# Run complete training pipeline (enrich → train all models → upload → pack)
./training/run_second.sh
```

Training configs for local GPU (4070 Ti), lean A100, and full A100 in `training/configs/`.

### Benchmarking

```bash
python3 run_benchmark.py
```

Runs all models on 7 curated test characters and produces a 6-column overview grid saved as `output/trainingNN_overview.png`.

## Output Format

Per training example:
```
example_001/
├── image.png           # 512x512 RGBA
├── segmentation.png    # 8-bit grayscale (pixel value = region ID 0-21)
├── draw_order.png      # 8-bit grayscale (normalized Z-depth)
├── joints.json         # {joint_name: [x, y], ...} for 20 joints
└── metadata.json       # Source type, style, pose, camera angle
```

Dataset splits are by **character** (not image) to prevent data leakage: 80% train / 10% val / 10% test.

## License

Training data sourced from CC0, CC-BY, and CC-BY-SA licensed assets only.

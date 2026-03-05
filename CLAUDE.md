# Strata Synthetic Data Pipeline

Blender-based pipeline that generates labeled training data for Strata's 7 ONNX AI models. Also houses animation intelligence scripts and curated metadata for animation training. Runs independently of the Strata codebase.

## Strata's 7 ONNX Models

Strata (Tauri/Rust/React desktop app at `../strata/`) uses 7 ONNX models defined in `src-tauri/src/ai/runtime.rs`:

| # | Model | ONNX File | Architecture | Training Pipeline | Status |
|---|-------|-----------|-------------|-------------------|--------|
| 1 | **Segmentation** | `segmentation.onnx` | DeepLabV3+ MobileNetV3 | `training/train_segmentation.py` | Has pipeline + data |
| 2 | **Joint Refinement** | `joint_refinement.onnx` | MobileNetV3 + regression heads | `training/train_joints.py` | Has pipeline + data |
| 3 | **Weight Prediction** | `weight_prediction.onnx` | Per-vertex MLP (31→128→256→128→20) | `training/train_weights.py` | Has pipeline + data |
| 4 | **Diffusion Weight Prediction** | `diffusion_weight_prediction.onnx` | Dual-input MLP (vertex features + seg encoder features) | **Not yet built** | Needs pipeline |
| 5 | **Inpainting** | `inpainting.onnx` | U-Net for occluded body regions | **Not yet built** | Needs pipeline + data |
| 6 | **Texture Inpainting** | `texture_inpainting.onnx` | Diffusion-based 3D texture fill | **Not yet built** | Needs pipeline + data |
| 7 | **Back View Generation** | `back_view_generation.onnx` | Multi-view conditioned diffusion | **Not yet built** | Needs pipeline + data |

### Model Details

**1. Segmentation** — Input: [1,3,512,512] image. Outputs: 22-class body region logits + draw order depth map (sigmoid) + confidence mask + encoder_features (passed to model 4). Fine-tunes from ImageNet MobileNetV3.

**2. Joint Refinement** — Input: [1,3,512,512] image. Outputs: [1,2,20] joint offsets (dx-first layout) + [1,20] confidence + [1,20] presence. Fine-tunes from ImageNet MobileNetV3. Falls back to geometric predictions if unavailable.

**3. Weight Prediction** — Input: [1,31,2048,1] vertex features (position, bone distances, heat diffusion, region label). Outputs: [1,20,2048,1] per-bone weights + [1,1,2048,1] confidence. MLP trained from scratch (no pretrained backbone).

**4. Diffusion Weight Prediction** — Dual-input variant of model 3. Takes vertex features + bilinearly sampled encoder features from model 1's segmentation backbone. Improves weight prediction for unusual proportions (chibi, elongated limbs, loose clothing). **Training pipeline needed.**

**5. Inpainting** — U-Net that fills occluded body regions in 2D paintings. Fallback: EdgeExtend (dilates visible edge pixels). **Training pipeline + paired data needed.**

**6. Texture Inpainting** — Diffusion model that fills unobserved texture regions when generating 3D mesh. **Training pipeline + data needed.**

**7. Back View Generation** — Multi-view conditioned diffusion model. Generates back view from front + 3/4 view. Fallback: PaletteFill (mirror + color adjustment). **Training pipeline + multi-view data needed.**

### Training Coverage

Models 1-3 have complete training pipelines with configs for local (4070 Ti), lean A100, and full A100 runs. Models 4-7 still need training pipelines built in this repo. All models are bundled in `../strata/src-tauri/models/` (~55MB total) and loaded lazily via the `ort` ONNX runtime crate.

## What This Project Does

Generates labeled training data (color images, segmentation masks, draw order maps, joint positions, weight maps) from rigged 3D characters across 6 art styles. Also includes: Spine 2D parser (`pipeline/spine_parser.py`), BVH mocap pipeline (`animation/`), Label Studio annotation (`annotation/`), and external dataset adapters (`ingest/`). Target: 300+ characters, 36K+ images. See `docs/preprocessed-datasets.md` for planned external data sources.

## Project Layout

Key dirs: `pipeline/` (Blender rendering + segmentation), `ingest/` (external dataset adapters), `annotation/` (Label Studio), `animation/` (BVH/mocap), `tests/`, `docs/`. Entry points: `run_pipeline.py`, `run_validation.py`. Config: `pipeline/config.py`.

**Tracked:** `pipeline/`, `ingest/`, `annotation/`, `animation/`, `tests/`, `docs/`, `data/*/labels/`. **Ignored:** `data/` (large binaries), `output/` (generated).

Label mappings live in `data/{source}/labels/` (e.g., `data/live2d/labels/live2d_mappings.csv`).

## Running the Pipeline

```bash
blender --background --python run_pipeline.py -- \
  --input_dir ./data/fbx/ \
  --pose_dir ./data/poses/ \
  --output_dir ./output/segmentation/ \
  --styles flat,cel,pixel,painterly,sketch,unlit \
  --resolution 512 \
  --poses_per_character 20
```

Requires Blender 4.0+ (uses bundled Python 3.10+). No GPU needed — EEVEE flat shading runs on CPU.

## Key Technical Details

### Strata Standard Skeleton (21 body regions + background = 22 classes)

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

### Bone Mapping Priority

1. Exact match → 2. Prefix match → 3. Substring match (case-insensitive) → 4. Manual override JSON

Mixamo characters should map 100% automatically. Non-Mixamo ~80% auto, ~20% manual via per-character override JSON.

### Segmentation Mask Rendering

- 22 Emission-only materials (one per region), no lighting
- Each mesh face assigned to region by majority vote of vertex bone weights
- No anti-aliasing, nearest-neighbor sampling — each pixel = exactly one region ID
- Output: 8-bit single-channel grayscale PNG (pixel value = region ID)

### Output Format

Per training example:
```
example_001/
├── image.png           ← Character render (512×512)
├── segmentation.png    ← Per-pixel label IDs (grayscale)
├── draw_order.png      ← Per-pixel depth (grayscale, 0=back 255=front)
├── joints.json         ← 2D joint positions
└── metadata.json       ← Source type, style, pose name, camera angle, draw order values
```

Draw order is computed from vertex Z-depth relative to camera (Mixamo) or from explicit render order indices (Live2D). Normalized to [0, 1] range per frame.

### Render Setup

- Orthographic camera (no perspective distortion)
- Auto-framed to character bounding box + 10% padding
- 512×512 output resolution
- Transparent background (alpha channel)
- Minimal lighting: single directional + high ambient (~0.7)
- **Multi-angle rendering**: 5 camera angles per character×pose — front (0°), three-quarter (45°), side (90°), three-quarter-back (135°), back (180°). Enabled via `--angles` CLI flag; defaults to front-only. Required for 3D mesh pipeline training data.

### Style Augmentation

Render-time (Blender shaders): flat, cel/toon, unlit
Post-render (Python/PIL/OpenCV): pixel art, painterly, sketch/lineart

Masks are per-pose, NOT per-style — the same mask applies to all style variants of a pose.

### Dataset Splits

Split by **character**, not by image (prevents data leakage). 80% train / 10% val / 10% test.

## Asset Sources

Only use CC0, CC-BY, or CC-BY-SA licenses. Never CC-NC. Log every asset's license in output metadata. Full source list in `docs/data-sources.md`, external datasets in `docs/preprocessed-datasets.md`.

## Validation Checks

Automated (run after every batch):
- Every non-transparent pixel has a non-zero mask region
- No mask is all-one-region
- All joint positions within image bounds
- 19 joints per pose
- Every image has a corresponding mask and joint file
- All images exactly 512×512
- No single region >60% of pixels

## Conventions

- Naming pattern: `{source}_{id}_pose_{nn}_{style}.png` for images, `{source}_{id}_pose_{nn}.png` for masks
- JSON metadata uses the schema defined in the PRD (sections 8.2–8.3)
- v1 is humanoid bipeds only. Non-humanoid (quadruped, bird, serpentine) is future work.
- Accessories: hide for v1 (cleaner training data). Flag with `has_accessories: true` in metadata.

## Storage Policy

- **Hetzner bucket** — upload only training-ready output (validated `output/` data: images, masks, joints, metadata). No raw source assets.
- **External hard drive** — store all raw/source data locally: FBX characters, BVH mocap, VRM files, sprite sheets, Live2D models, pre-processed external datasets. Download and process from here; do not upload raw data to the cloud bucket.
- **Local SSD** — avoid storing large output datasets locally. Process on the external HD when possible and upload from there.

### Bucket Upload/Download — Use rclone

**Always use `rclone copy` for bucket operations.** Do NOT use `aws s3 sync` — it is extremely slow with many small files.

```bash
# Upload (copy only adds files, never deletes remote)
rclone copy ./output/dataset/ hetzner:strata-training-data/dataset/ \
  --transfers 32 --checkers 64 --fast-list --size-only -P

# Download
rclone copy hetzner:strata-training-data/dataset/ ./output/dataset/ \
  --transfers 32 --checkers 64 --fast-list --size-only -P
```

- Config: `~/.config/rclone/rclone.conf`, remote name: `hetzner`
- `-P` shows real-time progress with transfer speed
- `--size-only` skips slow checksum comparison (safe for our use case)
- **Never use `rclone sync`** — it deletes remote files not present locally

### Bucket Contents (as of March 4 2026)

Bucket: `strata-training-data` at `fsn1.your-objectstorage.com`. ~790K+ files, ~160+ GiB total.

| Prefix | Files | Size | Source |
|--------|------:|-----:|--------|
| `animation/` | 18,628 | 66.7 GiB | 100STYLE retargeted mocap (100 styles × 8-10 contents) |
| `anime_instance_seg/` | ~135K | ~15 GiB | CartoonSegmentation instance masks (partially uploaded ~45K) |
| `anime_seg/` | ~65K | ~3.5 GiB | SkyTNT anime-segmentation v1+v2, RTMPose joints enriched |
| `animerun/` | 11,276 | 663 MiB | AnimeRun v2.2 segmentation frames |
| `animerun_correspondence/` | 19,493 | 930 MiB | AnimeRun cross-frame correspondence |
| `animerun_flow/` | 16,704 | 11.6 GiB | AnimeRun optical flow pairs |
| `animerun_linearea/` | 4,236 | 119 MiB | AnimeRun line area maps |
| `animerun_segment/` | 11,276 | 628 MiB | AnimeRun segment labels |
| `conr/` | ~7,269 | ~580 MiB | CoNR multi-view anime character sheets |
| `fbanimehq/` | 304,889 | 11.4 GiB | FBAnimeHQ face/body crops |
| `humanrig/` | 137,209 | 5.6 GiB | HumanRig rendered chars + joints + weights |
| `ingest/vroid_lite/` | 9,302 | 771 MiB | VRoid Lite CC0 characters |
| `instaorder/` | ~11,868 | ~1.5 GiB | InstaOrder draw order maps (val split) |
| `live2d/` | 3,587 | 212 MiB | Live2D .moc3 rendered models |
| `segmentation/` | 12,216 | 599 MiB | Mixamo pipeline segmentation output |
| `unirig/` | 66,030 | 42.6 GiB | UniRig rigged meshes |

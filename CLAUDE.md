# Strata Synthetic Data Pipeline

Blender-based pipeline that generates labeled training data for Strata's ONNX AI models. Runs independently of the Strata codebase.

## Strategic Direction

Per advisor at Erhvervhus Sjælland: **focus on shipping v1 (2D rigging + animation), defer v2 (3D)**. Investor signal is "users + LOIs" not model performance.

**v1 product = 2D character → rigged + animated 2D mesh, exported to GLB/Spine/JSON.** No 3D mesh generation, no UV texture inpainting in v1.

**Status (April 30, 2026): AI work for v1 is done.** Run 20 (0.6485 mIoU) is bundled in Strata. Architecture lever exhausted (Run 23 + Run 31 both failed). Data-expansion lever exhausted (Runs 24-30 all underperformed Run 20). See `docs/run-history.md` for the detailed record.

**Next focus areas (per `docs/v1-ship-checklist.md`):**
- Strata desktop UX polish (40% of time)
- Beta tester recruitment + user research (30%)
- Fundraising prep — demo videos, LOIs, pitch material (10%)
- AI work only if specific failure modes block the v1 demo (20%)

## v1 Active Models

| # | Model | Current | Ship Target | v1? |
|---|-------|---------|-------------|-----|
| 1 | **Segmentation** | 0.6485 mIoU | 0.70 (relaxed) | ✅ shipped |
| 2 | **Joint Refinement** | 0.00121 offset | 0.0005 | ✅ shipped |
| 3 | **Weight Prediction** | 0.0215 MAE | 0.01 | ✅ shipped |
| 4 | **2D Inpainting** (bg) | 0.0028 val/l1 | already ships | ✅ shipped |

Seg target relaxed 0.75 → 0.70 because Strata's post-processing improvements (confidence-gated cleanup, bilinear logit upscaling, small-component removal) compensate for a chunk of the gap. Run 20 + post-processing is the actual user experience.

**Deferred to v2 (post-funding):**
- Texture Inpainting (UV) — best 0.1282 val/l1 with geometry maps; fails on illustrated styles
- 3D Mesh — SAM 3D Objects validated, single-view projection pipeline built, needs better inpainting

## 22-Class Anatomy Schema

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

Maps directly to skeleton bones — each class = one bone's influence zone. Class 20 (accessory) is remapped to background in the dataset loader (unused by rigging pipeline). Dr. Li's 19-class clothing schema (topwear, legwear) cannot drive animation — that's why we have our own.

## Project Layout

Key dirs: `pipeline/` (Blender rendering), `ingest/` (external dataset adapters), `training/` (model training + configs + run scripts), `scripts/` (utilities), `animation/` (BVH/mocap), `tests/`, `docs/`.

Entry points: `run_pipeline.py`, `run_validation.py`. Config: `pipeline/config.py`.

## Output Format

```
example_001/
├── image.png           ← 512×512 RGBA
├── segmentation.png    ← 8-bit grayscale (pixel value = region ID 0-21)
├── depth.png           ← Depth map (uint8, Marigold LCM)
├── normals.png         ← Surface normals (RGB uint8, Marigold LCM)
├── joints.json         ← 2D joint positions
└── metadata.json       ← Source, style, enrichment flags
```

## Running the Pipeline

```bash
blender --background --python run_pipeline.py -- \
  --input_dir ./data/fbx/ --pose_dir ./data/poses/ \
  --output_dir ./output/segmentation/ \
  --styles flat,cel,pixel,painterly,sketch,unlit \
  --resolution 512 --poses_per_character 20
```

Requires Blender 4.0+/5.0+. Orthographic camera, 512×512, transparent bg, 22 Emission materials for seg masks.

## Storage & Bucket

- **Hetzner bucket** (`strata-training-data` at `fsn1.your-objectstorage.com`) — training-ready output only
- **External HD** (`/Volumes/TAMWoolff/data/`) — raw source data, never delete originals
- **Local SSD** — avoid large datasets

```bash
# Always use `rclone copy` (never `sync`, never `aws s3 sync`)
rclone copy ./output/ hetzner:strata-training-data/output/ --transfers 32 --checkers 64 --fast-list --size-only -P
```

Inventory of bucket contents and frozen val/test splits is in `docs/run-history.md`.

## Asset Sources & Licensing

Only CC0, CC-BY, CC-BY-SA. **Never CC-NC.**

**Prohibited:** Mixamo (Adobe ToS), Live2D (proprietary), ArtStation/curated_diverse (no AI permission), CartoonSegmentation (no license).

## A100 Training Workflow

If we ever return to training (post-funding, post-v1):

1. Prep locally → upload to bucket → push code
2. Spin up A100 → `cloud_setup.sh lean`
3. Run script (downloads data, trains, uploads checkpoints)
4. Destroy instance
5. Download checkpoints to Mac → benchmark

**Pipeline hygiene rules** (read before any new run): see `docs/run-history.md` § "April 21 Pipeline Hygiene Rules". Most important: one variable per experiment, reproduction check first, archive `.pt` with `.onnx`, never `pkill -f python3`.

## Quality Filter

`scripts/filter_seg_quality.py` flags: `--min-regions 4`, `--max-single-region 0.70`, `--min-foreground 0.05`. Plus the validated cheap-win flags from Run 29: `--drop-head-below-torso --max-bg-bleed 0.10 --min-silhouette-coverage 0.50`. Also ported into `ingest_gemini.py` so new ingestions are cleaned at entry.

## Bootstrapping Loop (caveat)

Model → pseudo-label new data → quality filter → retrain.

**Works only when teacher domain matches student domain** (e.g., GT-trained teacher bootstrapping more rendered data). Does **not** work for cross-domain transfer (Run 20 trained mostly on 3D-rendered chars pseudo-labeling illustrated chars → Runs 24-27 all underperformed). See `docs/run-history.md` for the full evidence.

## Strata Post-Processing (in `../strata/`)

Six improvements live in Strata's `segmentation.rs`, `joints.rs`, `weights.rs` that compensate for model imperfection without retraining:

1. Alpha-aware preprocessing (composite RGBA onto black)
2. Confidence-gated seg cleanup (<0.4 → 5×5 majority, remove <16px components)
3. Bilinear logit upscaling (interpolate logits then argmax)
4. Laplacian weight smoothing (2 iterations, confidence-adaptive)
5. Adaptive joint offset cap (15% of bbox, clamped [0.02, 0.10])
6. Confidence-weighted MLP/heat blending for weights

These are why we can ship Run 20 (0.6485 mIoU) instead of needing to push the model to 0.75.

## Later — Post Launch

- Interactive view correction paint tool
- Blueprint marketplace
- InnoFounder application (August 2026, 430K DKK grant)
- PreSeed Ventures / Accelerace / byFounders for investment

---

For the historical record of what's been tried and learned, see:
- `docs/run-history.md` — every training run, why it failed, what to revisit post-funding
- `docs/v1-ship-checklist.md` — the active v1 work plan, time allocation, decision log

# Strata Synthetic Data Pipeline

Blender-based pipeline that generates labeled training data for Strata's 6 ONNX AI models. Runs independently of the Strata codebase.

## Model Targets & Current Scores

**Goal:** Uploaded 2D character illustrations look natural from all generated angles when rigged and posed.

| # | Model | Current | Ship Target | Stretch | Key Issue If Bad |
|---|-------|---------|-------------|---------|------------------|
| 1 | **Segmentation** | 0.6485 mIoU | **0.75** | 0.80 | Background fringe, bg between legs, character parts deleted |
| 2 | **Joint Refinement** | 0.00121 offset | **0.0005** | 0.0003 | Joints misplaced → limbs bend wrong |
| 3 | **Weight Prediction** | 0.0215 MAE | **0.01** | 0.005 | Mesh deformation artifacts when posed |
| 4 | **Inpainting** (2D bg) | 0.0028 val/l1 | **0.002** | 0.001 | Visible seams where character removed |
| 5 | **Texture Inpainting** (UV) | 0.1282 val/l1 | **0.08** | 0.05 | Back/sides of 3D char look wrong |
| 6 | **3D Mesh** | Blurry (old U-Net) | **Clean geometry** | PBR | Character looks flat, geometry wrong |

### Priority Order (April 21, 2026)

1. **Segmentation** — Run 20 baseline (0.6485) still best. Four rounds of pseudo-label-based data expansion (Runs 24-27) all failed to beat it, including See-Through SAM with Dr. Li's joint-based converter. **Pseudo-labeling is exhausted as a lever for this task.** Currently running **Run 23** (ResNet-50 + Pascal-Person-Part anatomy-pretrained backbone) — architectural experiment, orthogonal to label quality. If flat, next lever is CVAT hand-labels on illustrated chars (slow but real GT).
2. **Texture Inpainting** — v3 ControlNet at 0.1282 val/l1 but fails on illustrated styles (lichtung cat test). Next: test StyleTex (SIGGRAPH 2024, Apache 2.0) or generate style-diverse training pairs.
3. **3D Mesh** — SAM 3D Objects validated + single-view projection pipeline built. Works end-to-end for PBR-style characters. Needs better texture inpainting for illustrated styles.
4. **Joint Refinement** — ViTPose++ fine-tune. Current model functional (0.00121 offset).
5. **Weight Prediction** — Study Puppeteer architecture. Current model functional (0.0215 MAE).
6. **Inpainting** (2D bg) — Low priority. User complaint is actually about segmentation quality (see #1), not inpainting quality.

### April 15 SAM-HQ Seg Run — What We Learned

Ran Dr. Li's V3 training pipeline with her SAM-HQ ViT-B encoder frozen, fresh 22-class decoder, 8000 steps. Trajectory:

| Step | val/mIoU |
|------|----------|
| 1000 | 0.198 |
| 2000 | 0.223 |
| 3000 | 0.239 |
| 4000 | 0.269 |
| 5000 | 0.288 |

Killed at step ~5500. Linear extrapolation: ~0.35 at step 8000 — still far below Run 20's 0.6485. Lessons:
- Her encoder was pretrained for clothing/style boundaries, not anatomy — may need unfreezing to adapt
- Decoder from scratch at 8K steps is cold-start; needs 20K+ for meaningful learning
- Multiple bugs in upstream V3 code required patches (gradient_checkpointing, checkpoint_total_limit, dataset_seg exceptions, resume_from_checkpoint unimplemented)

**Decision:** abandon this approach for now. Focus on expanding data + leveraging See-Through via *pseudo-labeling* (not encoder fine-tune).

### April 20-21 Pseudo-label Experiments — What We Learned (Runs 24-27)

Added 3,207 new illustrated chars to gemini_diverse, tried 4 different pseudo-label sources:

| Run | gemini_diverse labels | Val mIoU @ e19 | Notes |
|-----|------------------------|----------------|-------|
| 24 | Run 20 self-distill, wt 3.5, softening off, **meshy tar misnamed so missing** | broken | pipeline bug + softening off |
| 25 | Run 20 self-distill, wt 2.0, softening on, meshy restored | **0.5965** | ~= Run 20 baseline |
| 26 | See-Through SAM, naive L/R+vertical heuristic converter | 0.5471 | worse |
| 27 | See-Through SAM → Dr. Li PNG format → convert_with_joints | 0.5664 | worse than 25 |

**Conclusions:**
- **Data expansion via pseudo-labeling has not beaten Run 20 for this task.** Period.
- See-Through+Li labels pass quality filter at 4× the rate of Run 20 self-distill (5.2% rejected vs 23.2% for sora_diverse), but the model trained on them performs *worse*. Quality filter checks shape, not pixel correctness — labels can be well-formed yet mislocalized.
- **Noisy-but-spatially-correct beats well-formed-but-mislocalized.** Run 20's self-distill labels at least put regions in the right neighborhood.
- See-Through SAM's 19 clothing classes don't cleanly map to our 22 anatomy classes even with joint-based splits: the knee-at-joint-midpoint assumption breaks for dynamic poses, joints model trained on 3D-rendered chars mis-localizes on stylized art.
- **Boundary softening (radius=2) remains the single biggest lever** (+8.8% mIoU). Run 24 disabled it by accident and that alone cost ~0.05 mIoU.

### Path to Ship Quality (revised plan — April 21)

**Seg → 0.75 mIoU:**
1. **Run 23 (in progress)** — ResNet-50 + Pascal-Person-Part anatomy-pretrained backbone. Architectural experiment, orthogonal to label quality. Could break the mobilenet ceiling.
2. If Run 23 flat → **CVAT hand-labels** on 200-500 illustrated chars. Real GT. Slow (~2 days of human work) but the only path that consistently moves the needle when pseudo-labeling plateaus.
3. Per-class boundary softening (radius=1 for thin classes, radius=3 for large) — still untried lever on top of Run 20's radius=2.
4. Skip: more pseudo-labeling (Run 20 self-distill or See-Through). We've tested this with 4 label sources and all are ≤ Run 20 baseline.

**Texture Inpainting → 0.08 val/l1 + style-consistent:**
1. Option A: test StyleTex (queued, `training/run_styletex_test.sh`) — SDS-based, pretrained
2. Option B: generate style-diverse training data (watercolor, anime, hand-painted textures via NPR shaders on Meshy chars or Gemini img2img)
3. Option C: consider SDXL Inpainting or Flux Fill as base model (richer priors)
4. Add perceptual loss (LPIPS) on decoded output — current MSE-on-noise doesn't optimize for visual coherence

**3D Mesh:**
- Current pipeline works. Needs better inpainting to unlock illustrated-style workflows.
- Long-term: fine-tune TRELLIS.2 on our characters if SAM 3D quality plateaus

### Next A100 Runs Queued

**Run 23: ResNet-50 + Pascal-Person-Part anatomy pretrain (RUNNING April 21)**
- `training/run_a100_run23_combined.sh` on main
- 6-8 hrs on A100
- Hypothesis: architectural change (backbone + anatomy priors) breaks pseudo-label ceiling
- Result will determine whether next lever is more architecture or CVAT hand-labels

**Run A (ABANDONED, April 21): Expanded seg training via See-Through pseudo-labels**
- 4 rounds of pseudo-label experiments (Runs 24-27) all underperformed Run 20 baseline
- Don't retry this lever. See "April 20-21 Pseudo-label Experiments" above.

**Run B: StyleTex test on lichtung cat (~1.5 hrs)**
- `training/run_styletex_test.sh` ready
- 0-shot inference, no new training
- Tests whether style-aware texture transfer works for illustrated characters

## Model Strategy

Strata's unique value: combining 3D reconstruction with illustrated character understanding for animation-ready rigging.

| # | Task | New Model | License | Status |
|---|------|-------|---------|--------|
| 1 | **Segmentation** | Dr. Li's SAM-HQ encoder + our 22-class decoder | Apache 2.0 | Training code April 12 |
| 2 | **3D Mesh** | SAM 3D Objects | SAM License (commercial OK) | Validated |
| 3 | **Skeleton** | SAM 3D Body / ViTPose++ | SAM / Apache 2.0 | Tested — good on humanoid |
| 4 | **Texture** | Multi-view projection + ControlNet inpainting | Our code | Training now |
| 5 | **Weights** | Puppeteer / current MLP | TBD | Study architecture |
| 6 | **Backup 3D** | TRELLIS.2 (4B params, MIT) | MIT | Needs HF access |

**New Strata Pipeline (vision):**
1. User imports front view illustration
2. SAM 3D Objects → 3D mesh geometry (discard blurry texture)
3. Project artist's illustration onto mesh (pixel-perfect where visible)
4. Optional side/back views → more coverage
5. ControlNet inpaints remaining UV gaps (~10-30% of surface)
6. Segmentation → anatomy regions, SAM 3D Body → skeleton, weight prediction
7. Export rigged, textured, animatable 3D character

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

Maps directly to skeleton bones — each class = one bone's influence zone. Dr. Li's 19-class schema is clothing-based (topwear, legwear) and cannot drive animation.

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

## Asset Sources & Licensing

Only CC0, CC-BY, CC-BY-SA. **Never CC-NC.**

**Prohibited:** Mixamo (Adobe ToS), Live2D (proprietary), ArtStation/curated_diverse (no AI permission), CartoonSegmentation (no license).

## Storage & Bucket

- **Hetzner bucket** (`strata-training-data` at `fsn1.your-objectstorage.com`) — training-ready output only
- **External HD** (`/Volumes/TAMWoolff/data/`) — raw source data, never delete originals
- **Local SSD** — avoid large datasets
- **Always use `rclone copy`** (never `sync`, never `aws s3 sync`)

```bash
rclone copy ./output/ hetzner:strata-training-data/output/ --transfers 32 --checkers 64 --fast-list --size-only -P
```

### Key Bucket Contents

Tars in `tars/` prefix: humanrig (16.8G), humanrig_posed (12G), meshy_cc0_textured_restructured (2.8G), flux_diverse_clean (300M), sora_diverse (380M), gemini_li_converted (223M), vroid_cc0 (203M), cvat_annotated (9M), soft_targets_precomputed (1.9G), demo_back_view_pairs (3.4G).

Texture inpainting tars in root: texture_pairs_front (4.9G, with geometry maps), texture_pairs_side (2.8G), texture_pairs_back (2.8G), texture_pairs_100avatars (113M). Total ~4,135 pairs.

Frozen val/test splits: `data_cloud/frozen_val_test.json`. All runs must use this file.

## A100 Training Workflow

1. Prep locally → upload to bucket → push code
2. Spin up A100 → `cloud_setup.sh lean`
3. Run script (downloads data, trains, uploads checkpoints)
4. Destroy instance
5. Download checkpoints to Mac → benchmark

## Key Learnings

**Segmentation (best: run 20, 0.6485 mIoU):**
- Boundary label softening = biggest lever (+8.8% mIoU). Precompute as `.npz` files.
- Softening hurts thin regions (neck, accessory). Exclude or reduce radius for small regions (`SOFTENING_EXCLUDE_CLASSES = {2, 21}` auto-excludes).
- Clothing-based labels (SAM, Dr. Li's 19-class) don't help — anatomy ≠ clothing.
- SAM 3D Body labels don't help — body mesh ≠ clothing silhouette.
- **SAM-HQ encoder fine-tune (Apr 15) underperformed** — frozen encoder + fresh decoder plateaued ~0.29 at step 5500. Encoder was trained for clothing boundaries, not anatomy.
- **Pseudo-label data expansion is exhausted (Apr 20-21, Runs 24-27).** Tried Run 20 self-distill, naive 19→22 heuristic converter, and See-Through SAM + Dr. Li's joint-based converter. All ≤ Run 20 baseline. Quality filter checks shape (region count, area ratios), not spatial correctness — well-formed labels can still be mislocalized. **Don't retry this lever.** Next cheap architectural experiment: Run 23 (ResNet-50 + Pascal-Person-Part pretrain). If flat, move to CVAT hand-labels.
- Bootstrapping loop DOES work when teacher was trained on aligned data (GT-based teacher on 3D-rendered data expanding to more rendered data). Does NOT work for cross-domain transfer (3D-trained teacher on illustrated data).
- Class 20 remapped to background (unused by rigging pipeline).
- A100 is 40GB. Batch 16 for soft targets. Use frozen val/test splits.
- **`gemini_li_converted` was Dr. Li's 694 *hand-labeled* examples through `convert_li_labels.py` with joints.** That's real GT, not pseudo-labels — it had weight 3.0 in Run 20 for a reason. Pseudo-labels from See-Through SAM do not substitute for hand labels.

**Texture Inpainting (best: run 4 / v3, 0.1282 val/l1):**
- ControlNet on SD 1.5 Inpainting, 9-channel input (noisy latent + mask + partial).
- Run 1: 500 pairs → 0.1509. Run 2: 2,891 pairs → 0.1520. Run 3: 2,891 pairs, 100 epochs → 0.1497.
- **Run 4 (v3): 1,244 pairs WITH real geometry maps, fine-tuned from v3 → 0.1282 val/l1, 0.418 SSIM.** −14% L1, +6% SSIM. Geometry maps are the biggest single lever.
- Single-view "hero" pipeline confirmed: 1 illustration → SAM 3D mesh → auto-detect camera angle → project illustration → ControlNet inpaint gaps. Cleaner UX than asking for multi-view input.
- Manual denoising loop needed for validation (diffusers pipeline incompatible with 9-ch ControlNet).
- Use `torch.amp.autocast` for mixed fp16/fp32 validation.

**Pipeline experiment (lichtung cat, April 14):**
- SAM 3D Objects mesh + 1 watercolor cat illustration → silhouette-IoU search found best camera angle (72°, IoU 0.77).
- Hero view projects pixel-perfect for ~30% of UV. Rest needs inpainting.
- Multi-view (front/back) approach failed — Gemini-generated views don't match mesh from "true" front/back angles.
- TPS landmark-based warping helps but needs 20+ landmarks per view to be useful.
- Single-view + strong inpainting is the best UX.
- **v3 ControlNet failed on illustrated style**: Meshy-trained model produced dark solid fill, not watercolor. Empty-prompt training means text prompts + CFG have no effect. Need style-aware approach.
- **Next attempt: StyleTex** — takes illustration as style reference, explicitly decouples style from content via CLIP manipulation.

**3D Reconstruction:**
- U-Net view synthesis deprecated (blurry). SHARP abandoned (research-only license).
- SAM 3D Objects = primary 3D mesh source. TRELLIS.2 (MIT) = backup.
- Texture from artist's illustration projected onto mesh. AI only fills gaps.

## Segmentation Run History

| Run | mIoU | Key change |
|-----|------|-----------|
| 8 | 0.4721 | Bootstrapping loop validated |
| 13a | 0.5425 | +toon_pseudo |
| 16 | 0.5808 | Frozen val deployed |
| 18 | 0.5750 | True baseline with frozen val |
| 20 | **0.6485** | Boundary softening (radius=2) |
| 21 | 0.6361 | Re-pseudo-label, no softening |
| 22/22b | 0.56-0.59 | SAM labels — don't help |
| SAM-HQ (Apr 15) | ~0.29 (killed at step 5500) | Li's encoder frozen + fresh 22-class decoder — plateaued below baseline |
| 24 (Apr 21) | broken | +gemini_diverse (3207 new), Run 20 self-distill labels, softening OFF by mistake, meshy tar name wrong → missing |
| 25 (Apr 21) | ~0.60-0.61 (killed) | Same data, softening back on, meshy restored, lower LR on resume — ~= Run 20 |
| 26 (Apr 21) | 0.5471 @ e19 | See-Through SAM + naive L/R+vertical heuristic converter — worse |
| 27 (Apr 21) | 0.5664 @ e19 | See-Through SAM → Dr. Li PNG format → `convert_with_joints` — still worse than Run 20 |
| 23 (Apr 21, running) | TBD | ResNet-50 + Pascal-Person-Part anatomy pretrain — architectural experiment |

Best: **Run 20** (0.6485 test mIoU). Config: `training/configs/segmentation_a100_run20.yaml`.

## Quality Filter

`scripts/filter_seg_quality.py`: `--min-regions 4`, `--max-single-region 0.70`, `--min-foreground 0.05`. Checks `missing_head` and `missing_torso`.

## Bootstrapping Loop

Model → pseudo-label new data → quality filter → retrain. **Works only when teacher was trained on data aligned with the new data's domain** (e.g., GT-trained teacher bootstrapping more of the same rendered data). **Does NOT work for cross-domain** (Run 20 trained mostly on 3D-rendered chars pseudo-labeling illustrated chars → Runs 24-27 all underperformed, April 2026).

```bash
python scripts/ingest_gemini.py --input-dir /path/to/raw --output-dir /path/to/preprocessed --no-seg --only-new
# Then pseudo-label + quality filter + train on A100 (handled by run scripts)
```

## Pipeline Hygiene (Apr 21)

- **Archive `.pt` checkpoints to bucket, not just `.onnx`.** Joints `.pt` was missing from bucket for months; only `joint_refinement.onnx` was archived. Recovered from `/Volumes/TAMWoolff/data/checkpoints/joints/best.pt` on April 21 and uploaded to `checkpoints_joints/best.pt`. Fix: after every training run, `rclone copy best.pt` to `checkpoints_<run>/` before destroying the A100.
- **`ingest_gemini.py` should run joints inference by default** when given a joints checkpoint, so new gemini_diverse never ships without `joints.json`. Currently it only does seg pseudo-labels.
- **Tar names matter.** Run 24 script referenced `meshy_cc0_restructured.tar` which doesn't exist (actual name: `meshy_cc0_textured_restructured.tar`) and silently trained without 15K Meshy examples. Pre-flight should verify tar download succeeded before extracting.

## Strata Post-Processing (in `../strata/`)

6 improvements in segmentation.rs, joints.rs, weights.rs:
1. Alpha-aware preprocessing (composite RGBA onto black)
2. Confidence-gated seg cleanup (<0.4 → 5×5 majority, remove <16px components)
3. Bilinear logit upscaling (interpolate logits then argmax)
4. Laplacian weight smoothing (2 iterations, confidence-adaptive)
5. Adaptive joint offset cap (15% of bbox, clamped [0.02, 0.10])
6. Confidence-weighted MLP/heat blending for weights

## Later — Post Launch

- Interactive view correction paint tool
- Blueprint marketplace
- InnoFounder application (August 2026, 430K DKK)
- PreSeed Ventures / Accelerace / byFounders for investment

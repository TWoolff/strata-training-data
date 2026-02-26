# Strata — Synthetic Data Pipeline

## Training Data Factory for AI Models

**Version:** 1.0  
**Date:** February 26, 2026  
**Status:** Draft  
**Prerequisite:** Strata AI Integration PRD v1.1  
**Relationship:** This pipeline produces all training data for Strata's AI features (semantic segmentation, joint prediction, weight prediction, drawn pose estimation). It runs independently of the Strata codebase and can operate in parallel with core development.

---

## 1. What This Pipeline Does

One Blender script takes a rigged 3D character and outputs everything needed to train every AI model in Strata:

```
Input:  rigged_character.fbx (skeleton + weights + mesh)
        ↓
Output: character_0042_pose_017_style_flat.png      ← rendered image
        character_0042_pose_017_style_flat_seg.png   ← segmentation mask
        character_0042_pose_017_style_flat.json      ← joint positions + weight map + metadata
```

One character × 20 poses × 6 styles = 120 training examples with perfect labels. 100 source characters = 12,000 examples. That's enough to fine-tune DeepLabV3+ to production quality.

---

## 2. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│  ASSET COLLECTION                                           │
│  Mixamo / Sketchfab / Blender assets → standardized .fbx    │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  IMPORT & NORMALIZE                                         │
│  Load into Blender → retarget to standard skeleton →        │
│  normalize scale/position → validate bone mapping           │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  POSE                                                       │
│  Apply pose library (T-pose, idle, walk cycle, attack,      │
│  crouch, jump, etc.) → each pose = one render batch         │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  RENDER PASSES (per pose)                                   │
│  Pass 1: Color render (with style shader)                   │
│  Pass 2: Segmentation render (vertex group → flat color)    │
│  Pass 3: Joint overlay (bone head/tail → 2D projection)     │
│  Pass 4: Weight visualization (per-bone heat map)           │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  STYLE AUGMENTATION                                         │
│  Each color render → multiple art styles:                   │
│  flat shading, cel shading, pixel art, painterly,           │
│  sketch/outline, hand-painted                               │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  EXPORT                                                     │
│  Image + mask + JSON metadata → training-ready dataset      │
│  Split: 80% train / 10% validation / 10% test              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Asset Sources

### 3.1 Primary: Mixamo (Free)

Adobe's Mixamo provides free rigged humanoid characters and animations. This is the backbone of the dataset.

**What you get:**
- ~100 unique character models (varying body types, clothing, armor, fantasy, sci-fi, realistic)
- Standard skeleton with consistent bone naming
- Hundreds of animation clips (walk, run, attack, idle, dance, etc.)
- FBX export with skeleton + skin weights
- Free for commercial use (Adobe account required)

**What you need to do:**
- Create an Adobe account
- Download characters in FBX format, T-pose
- Download 15–20 animation clips separately (these become your pose library)

**Limitations:**
- All humanoid bipeds — no quadrupeds, no serpentine, no flying creatures
- Consistent art style (semi-realistic 3D) — style augmentation is critical
- Some characters have accessories (weapons, shields, wings) that need handling

### 3.2 Secondary: Sketchfab / TurboSquid (Free + Paid)

**CC0 / CC-BY models on Sketchfab:**
- Search for "rigged character" filtered by Creative Commons license
- More variety than Mixamo: animals, creatures, stylized characters, chibi
- Quality varies wildly — need manual curation
- ~200–500 usable rigged characters across various searches

**Important:** Only use models with licenses that permit derivative works and commercial use (CC0, CC-BY, CC-BY-SA). Avoid CC-NC (non-commercial). Log every asset's license and attribution in the dataset manifest.

### 3.3 Tertiary: Free Blender Rigs

**Sources:**
- Blender Cloud / Blender Studio (open movie characters — free, CC)
- BlendSwap (community models, check licenses individually)
- Quaternius (low-poly game assets, CC0)
- Kenney (game assets, CC0)

### 3.4 Asset Budget

| Source | Characters | License | Cost | Effort |
|--------|-----------|---------|------|--------|
| Mixamo | ~100 | Free commercial | $0 | Low (batch download) |
| Sketchfab CC | ~100–200 | CC0/CC-BY | $0 | Medium (curation) |
| Quaternius/Kenney | ~50 | CC0 | $0 | Low |
| Blender community | ~30–50 | Various CC | $0 | Medium (curation + validation) |
| Spine asset packs | ~50–100 (2D, not 3D) | Commercial license | $50–150 | Medium (format parsing) |
| **Total** | **~330–450 3D + 50–100 2D** | | **$50–150** | |

After pose × style multiplication: **40,000–100,000+ training images** from this asset base.

---

## 4. Skeleton Standardization

### 4.1 The Problem

Every asset source uses different bone names. Mixamo uses "mixamorig:LeftArm", Blender rigs use "upper_arm.L", Sketchfab models use anything the artist felt like typing. Strata's AI needs a single label set.

### 4.2 Strata Standard Skeleton (17 regions)

```
Region ID   Region Name        Mixamo Bone(s)                   Common Aliases
─────────   ───────────        ──────────────                   ──────────────
0           background         (none)                           -
1           head               Head, HeadTop_End                head, skull
2           neck               Neck                             neck
3           chest              Spine2                           upper_body, torso_upper
4           spine              Spine1, Spine                    torso, torso_lower, abdomen
5           hips               Hips                             pelvis, root
6           upper_arm_l        LeftArm                          upperarm.L, L_upperarm
7           lower_arm_l        LeftForeArm                      forearm.L, L_forearm
8           hand_l             LeftHand + fingers               hand.L, L_hand
9           upper_arm_r        RightArm                         upperarm.R, R_upperarm
10          lower_arm_r        RightForeArm                     forearm.R, R_forearm
11          hand_r             RightHand + fingers              hand.R, R_hand
12          upper_leg_l        LeftUpLeg                        thigh.L, L_thigh
13          lower_leg_l        LeftLeg                          shin.L, L_calf
14          foot_l             LeftFoot + LeftToeBase           foot.L, L_foot
15          upper_leg_r        RightUpLeg                       thigh.R, R_thigh
16          lower_leg_r        RightLeg                         shin.R, R_calf
17          foot_r             RightFoot + RightToeBase         foot.R, R_foot
```

### 4.3 Bone Mapping Script

A Python dictionary that maps known bone name patterns to Strata region IDs. The script:

1. Loads the armature from the FBX
2. Iterates all bones, tries to match each to a Strata region using the mapping table
3. Falls back to fuzzy matching (case-insensitive substring: if bone name contains "arm" and "left" and "up" → region 6)
4. Flags unmapped bones for manual review
5. Assigns each mesh vertex to a region based on its dominant bone weight

**Mapping priority:** exact match → prefix match → substring match → manual assignment.

**Expected coverage:** Mixamo characters should map 100% automatically. Sketchfab characters ~80% automatically, ~20% need manual bone mapping (stored in a per-character override JSON).

### 4.4 Non-Humanoid Handling (Future)

The v1 pipeline focuses on humanoid bipeds (Strata's primary use case). Future extensions:

- **Quadruped label set** (18 regions): head, neck, chest, spine, hips, 4× upper_leg, 4× lower_leg, 4× paw, tail
- **Bird label set** (14 regions): head, neck, body, 2× wing_inner, 2× wing_outer, 2× leg, 2× foot, tail
- **Serpentine label set** (variable): head + N body segments

Each needs its own bone mapping table and separate model training. Not in scope for v1 but the pipeline architecture supports it — just add a new mapping table and label set.

---

## 5. Render Setup

### 5.1 Camera

Strata characters are 2D — the camera setup should match how artists will use the tool:

- **Orthographic camera** (no perspective distortion — matches 2D game art)
- **Front-facing** (character faces camera directly)
- **Auto-framing:** Script calculates character bounding box and positions camera to fill the frame with ~10% padding on all sides
- **Output resolution:** 512 × 512 px (training resolution). Higher res renders downscaled — provides anti-aliased edges.
- **Transparent background** (alpha channel) — the segmentation mask uses region 0 (background) for transparent pixels

**Optional secondary angles** (for robustness):
- 3/4 view (±30° rotation) — some artists work in 3/4 perspective
- Side view (90°) — profile characters
- These multiply the dataset by 2–3× but are lower priority than style augmentation

### 5.2 Lighting

Minimal lighting variation — Strata works with flat art, not photorealistic renders:

- **Single directional light** (simulates ambient + key light)
- **Flat ambient** (high value, ~0.7 — reduces shadows that wouldn't exist in 2D art)
- **Optional:** 2–3 lighting presets (front-lit, top-lit, side-lit) for minor variation

Lighting is deliberately boring. We want the model to focus on shape and color, not lighting conditions.

### 5.3 Render Passes

Each pose generates multiple render passes from a single scene setup:

**Pass 1 — Color render:**
The character as it looks after style augmentation. This is the "input image" the AI will see at inference time.

**Pass 2 — Segmentation mask:**
Every mesh face is colored by its Strata region ID. The render uses an override material that reads vertex group assignment and outputs a flat, unshaded color per region:

```python
REGION_COLORS = {
    0:  (0, 0, 0),        # background (transparent in final)
    1:  (255, 0, 0),      # head
    2:  (0, 255, 0),      # neck
    3:  (0, 0, 255),      # chest
    4:  (255, 255, 0),    # spine
    5:  (255, 0, 255),    # hips
    6:  (128, 0, 0),      # upper_arm_l
    7:  (0, 128, 0),      # lower_arm_l
    8:  (0, 0, 128),      # hand_l
    9:  (128, 128, 0),    # upper_arm_r
    10: (128, 0, 128),    # lower_arm_r
    11: (0, 128, 128),    # hand_r
    12: (64, 0, 0),       # upper_leg_l
    13: (0, 64, 0),       # lower_leg_l
    14: (0, 0, 64),       # foot_l
    15: (64, 64, 0),      # upper_leg_r
    16: (64, 0, 64),      # lower_leg_r
    17: (0, 64, 64),      # foot_r
}
```

No anti-aliasing on segmentation render. Nearest-neighbor sampling. Each pixel maps to exactly one region ID.

**Pass 3 — Joint position data (not an image):**
For each bone in the armature, project the bone head (joint position) to 2D screen coordinates. Output as JSON, not an image. This is the ground truth for joint prediction training.

```json
{
  "joints": {
    "head": [256, 48],
    "neck": [256, 82],
    "chest": [256, 130],
    "spine": [256, 170],
    "hips": [256, 210],
    "shoulder_l": [210, 105],
    "elbow_l": [175, 155],
    "wrist_l": [155, 200],
    ...
  },
  "image_size": [512, 512],
  "character_bbox": [128, 20, 384, 480]
}
```

**Pass 4 — Weight map data (not an image, or optional visualization):**
For each vertex in the 2D-projected mesh, record its bone weights. This feeds the weight prediction model.

Format: per-vertex array of `{position: [x, y], weights: {bone_name: weight_value, ...}}`. Only store weights > 0.01 to keep file sizes sane.

Optional: render a visualization image per bone (heat map showing weight influence) for debugging and manual inspection.

---

## 6. Pose Library

### 6.1 Core Poses (Minimum Viable)

The model needs to see characters in varied poses to generalize. Minimum set:

| Category | Poses | Purpose |
|----------|-------|---------|
| **Neutral** | T-pose, A-pose, idle stand | Baseline, clear limb separation |
| **Locomotion** | Walk (4 keyframes), run (4 keyframes) | Arms/legs in motion, overlap |
| **Action** | Punch, kick, sword swing, crouch, jump apex | Extreme positions, foreshortening |
| **Rest** | Sitting, kneeling, lying down | Unusual configurations |
| **Emote** | Waving, pointing, arms crossed, hands on hips | Arm positions near body |

**Total: ~20 distinct poses.** From Mixamo, download these as animation clips and sample keyframes.

### 6.2 Pose Application

For each character:

1. Load T-pose (the default)
2. For each animation clip, sample N keyframes (evenly spaced through the clip)
3. Apply the pose to the armature
4. Run all render passes
5. Reset to T-pose, next clip

**Keyframe sampling:** For a 30-frame walk cycle, sample frames 0, 7, 15, 22 (4 keyframes). For a 60-frame attack, sample frames 0, 15, 30, 45 (4 keyframes). Target: ~4 keyframes per clip × ~5 clips per character = ~20 poses per character minimum.

### 6.3 Pose Augmentation

Additional variations applied on top of poses:

- **Random small rotations** on each bone (±5°) — simulates artist imprecision
- **Scale variation** on the full character (0.8× to 1.2×) — simulates different character proportions
- **Y-axis flip** (mirror the pose) — doubles the dataset for free, and ensures the model doesn't develop left/right bias

---

## 7. Style Augmentation

This is the most critical step. Without style augmentation, the model learns to segment 3D renders — useless for Strata, where the input is hand-painted 2D art.

### 7.1 Render-Time Styles (Blender Shaders)

These are applied as Blender material overrides before rendering:

**Style 1 — Flat shaded:**
Single color per face, no smooth shading, no textures. The closest to Triangulate's aesthetic. Use Blender's "Flat" shading mode with a diffuse-only shader.

**Style 2 — Cel shaded (toon):**
Quantized shading (2–3 tone steps), black outline. Common in indie 2D games. Blender shader nodes: ColorRamp with hard steps on the diffuse component, Freestyle for outlines.

**Style 3 — Unlit / color-only:**
Pure albedo color, no lighting at all. Flat colored shapes. Like a paper cutout.

### 7.2 Post-Render Styles (Image Processing)

Applied to the rendered image in Python (PIL/OpenCV) after Blender outputs it. The segmentation mask stays unchanged — only the color image is modified.

**Style 4 — Pixel art downscale:**
Downscale the 512px render to 64px or 128px using nearest-neighbor, then upscale back to 512px. Creates chunky pixel-art look. Apply palette reduction (16–32 colors) using k-means clustering or a fixed retro palette.

**Style 5 — Painterly / soft:**
Apply a bilateral filter (edge-preserving blur) + slight color jitter + noise grain. Simulates hand-painted look. Vary the filter strength for different levels of "painterliness."

**Style 6 — Sketch / lineart:**
Edge detection (Canny) on the render → thick outlines on white or cream background. Optional: add slight wobble to edges (displace edge pixels by ±1px random) to simulate hand-drawn lines.

### 7.3 Additional Augmentations (Applied to All Styles)

Standard data augmentation applied during training (not baked into the dataset):

- **Color jitter:** Brightness ±20%, saturation ±30%, hue ±10°
- **Random noise:** Gaussian noise, σ = 0.01–0.03
- **JPEG compression artifacts:** Quality 60–95 (simulates saved/re-saved web art)
- **Slight rotation:** ±5° (characters aren't always perfectly upright)
- **Crop/pad variation:** ±10% framing offset
- **Horizontal flip:** 50% chance (with joint label mirroring: left ↔ right)

These are applied at training time using Albumentations or torchvision transforms, not baked into the dataset files. This means the model sees slightly different versions of each image every epoch.

### 7.4 Style Distribution

Not all styles are equally important. Target distribution in the training set:

| Style | % of dataset | Rationale |
|-------|-------------|-----------|
| Flat shaded | 25% | Closest to typical indie game art |
| Cel / toon | 20% | Very common in 2D games |
| Unlit / color-only | 15% | Paper cutout / flat design style |
| Pixel art | 15% | Large Strata audience segment |
| Painterly | 15% | Strata's core differentiator (painters) |
| Sketch / lineart | 10% | Less common but important for coverage |

---

## 8. Output Format

### 8.1 Directory Structure

```
dataset/
├── manifest.json                    # dataset metadata, version, statistics
├── class_map.json                   # region ID → name mapping
├── splits.json                      # train/val/test file lists
├── images/
│   ├── mixamo_001_pose_00_flat.png
│   ├── mixamo_001_pose_00_cel.png
│   ├── mixamo_001_pose_00_pixel.png
│   ├── mixamo_001_pose_01_flat.png
│   └── ...
├── masks/
│   ├── mixamo_001_pose_00.png       # one mask per pose (shared across styles)
│   ├── mixamo_001_pose_01.png
│   └── ...
├── joints/
│   ├── mixamo_001_pose_00.json
│   ├── mixamo_001_pose_01.json
│   └── ...
├── weights/
│   ├── mixamo_001_pose_00.json
│   ├── mixamo_001_pose_01.json
│   └── ...
└── sources/
    ├── mixamo_001.json              # source character metadata + license
    ├── sketchfab_042.json
    └── ...
```

**Key insight:** Segmentation masks and joint/weight data are per-pose, not per-style. The same mask applies to `_flat.png`, `_cel.png`, `_pixel.png` etc. — the character shape doesn't change, only the rendering. This means the mask/joint/weight files are shared, and only the images directory multiplies by style count.

### 8.2 Mask Format

PNG, 8-bit single channel (grayscale). Pixel value = region ID (0–17). No anti-aliasing. Lossless compression (PNG level 9).

Why not RGB? Single-channel is simpler to load, and 256 possible values far exceeds the 18 regions we need. Standard format for segmentation training.

### 8.3 Metadata JSON

Per-character source file:

```json
{
  "id": "mixamo_001",
  "source": "mixamo",
  "name": "Y Bot",
  "license": "Mixamo free license",
  "attribution": "Adobe Mixamo",
  "download_url": "https://www.mixamo.com/...",
  "bone_mapping": "auto",
  "bone_mapping_overrides": {},
  "unmapped_bones": [],
  "character_type": "humanoid",
  "notes": ""
}
```

Per-pose joint file:

```json
{
  "character_id": "mixamo_001",
  "pose_name": "walk_frame_07",
  "source_animation": "Walking.fbx",
  "source_frame": 7,
  "image_size": [512, 512],
  "joints": {
    "head": {"position": [256, 48], "confidence": 1.0, "visible": true},
    "neck": {"position": [256, 82], "confidence": 1.0, "visible": true},
    "wrist_l": {"position": [155, 200], "confidence": 1.0, "visible": false}
  },
  "bbox": [128, 20, 384, 480]
}
```

The `visible` flag marks joints that are occluded in this pose (e.g., arm behind body). The AI model needs to learn which joints it can't see.

### 8.4 Dataset Splits

```json
{
  "train": ["mixamo_001", "mixamo_002", ..., "sketchfab_015", ...],
  "val": ["mixamo_090", "sketchfab_040", ...],
  "test": ["mixamo_095", "sketchfab_045", ...]
}
```

**Split by character, not by image.** All poses and styles of one character go into the same split. This prevents data leakage — the model shouldn't see different poses of the same character in both training and validation.

**Ratio:** 80% train / 10% val / 10% test. For 300 characters: 240 train, 30 val, 30 test.

---

## 9. The Blender Script

### 9.1 Architecture

One main Python script (`generate_dataset.py`) that runs as a Blender command-line job:

```bash
blender --background --python generate_dataset.py -- \
  --input_dir ./source_characters/ \
  --pose_dir ./pose_library/ \
  --output_dir ./dataset/ \
  --styles flat,cel,pixel,painterly,sketch,unlit \
  --resolution 512 \
  --poses_per_character 20
```

The script is modular:

```
generate_dataset.py          ← main entry, orchestrates everything
├── importer.py              ← load FBX, normalize, validate
├── bone_mapper.py           ← map bones to Strata regions
├── pose_applicator.py       ← apply animation keyframes
├── renderer.py              ← render color + segmentation passes
├── style_augmentor.py       ← post-render style transforms
├── joint_extractor.py       ← project bone positions to 2D
├── weight_extractor.py      ← extract per-vertex weights
├── exporter.py              ← save images, masks, JSON metadata
└── config.py                ← region colors, bone mappings, defaults
```

### 9.2 Per-Character Processing Flow

```python
# Pseudocode for main loop
for character_file in source_characters:
    # 1. Import
    scene = import_and_normalize(character_file)
    armature = scene.armature
    mesh = scene.mesh
    
    # 2. Map bones
    mapping = map_bones_to_regions(armature)
    if mapping.unmapped_bones:
        log_warning(f"{character_file}: unmapped bones {mapping.unmapped_bones}")
    
    # 3. Assign segmentation materials
    assign_region_materials(mesh, mapping)
    
    # 4. Setup camera
    camera = setup_orthographic_camera(mesh.bounding_box, padding=0.1)
    
    # 5. For each pose
    for pose in pose_library:
        apply_pose(armature, pose)
        
        # 6. Render segmentation mask (once per pose)
        mask = render_segmentation(scene, camera)
        save_mask(mask, character_id, pose.name)
        
        # 7. Extract joint positions (once per pose)
        joints = extract_joint_positions(armature, camera)
        save_joints(joints, character_id, pose.name)
        
        # 8. Extract weight data (once per pose, T-pose only for weights)
        if pose.is_tpose:
            weights = extract_vertex_weights(mesh, mapping, camera)
            save_weights(weights, character_id)
        
        # 9. Render color in each style
        for style in styles:
            color_image = render_color(scene, camera, style)
            augmented = apply_post_style(color_image, style)
            save_image(augmented, character_id, pose.name, style.name)
    
    # 10. Cleanup
    clear_scene()
```

### 9.3 Segmentation Material Setup

The key technical challenge: rendering a per-region color mask from a rigged mesh.

**Approach:** Use Blender's vertex group system. Each vertex is already assigned to bone weights (that's what skinning is). The region mapper assigns each vertex to the region of its dominant bone. Then:

1. Create 18 materials, one per region, each a flat Emission shader at the region's color
2. For each mesh face, determine which region owns the majority of its vertices
3. Assign that face to the corresponding material slot
4. Render with Emission-only materials and no lighting → perfect flat color per face

**Edge case:** Faces at region boundaries (e.g., shoulder area where upper_arm and chest blend). Rule: majority vote among the face's vertices. If tied, use the vertex closest to the face center.

### 9.4 Handling Accessories

Characters often have non-body geometry: weapons, shields, capes, hair ribbons, armor pieces.

**Strategy:**
- If the accessory is skinned to the skeleton (moves with a bone), assign it to the nearest body region. A shoulder pauldron becomes part of `upper_arm`. A sword in hand becomes part of `hand`.
- If the accessory is a separate object (not skinned), either: (a) hide it for clean body segmentation, or (b) assign it region 0 (background).
- Create a metadata flag `has_accessories: true` so the training pipeline can optionally filter.

For v1, hiding accessories is simpler and produces cleaner training data. Accessory handling is a refinement for v2 when the model needs to learn to segment "around" equipment.

---

## 10. Implementation Plan

### 10.1 Phase 1 — Skeleton (Week 1)

Build the minimal pipeline end-to-end with one character, one pose, one style:

- [ ] Blender script: import FBX, normalize scale/position
- [ ] Bone mapper: Mixamo → Strata region mapping (hardcoded for Mixamo naming)
- [ ] Segmentation material assignment
- [ ] Orthographic camera auto-framing
- [ ] Render: color (flat shaded) + segmentation mask
- [ ] Joint extraction: bone heads → 2D coordinates → JSON
- [ ] Export: save image + mask + JSON to output directory

**Deliverable:** Run the script on one Mixamo character. Get one image, one mask, one joint file. Visually verify the mask is correct by overlaying it on the image.

### 10.2 Phase 2 — Pose & Scale (Week 2)

- [ ] Pose library: download 5 Mixamo animations, extract keyframes
- [ ] Pose application loop: iterate poses per character
- [ ] Y-axis flip augmentation
- [ ] Scale variation (±20%)
- [ ] Batch processing: loop over multiple characters
- [ ] Weight extraction (T-pose only)
- [ ] Progress logging and error handling

**Deliverable:** Process 10 Mixamo characters × 20 poses = 200 image/mask/joint sets. Spot-check 10% for correctness.

### 10.3 Phase 3 — Style Augmentation (Week 3)

- [ ] Blender shaders: flat, cel/toon, unlit
- [ ] Post-render processing: pixel art, painterly, sketch
- [ ] Style application loop per rendered image
- [ ] Color jitter / noise augmentation (for training-time use)

**Deliverable:** 200 poses × 6 styles = 1,200 training images with masks. Visual review of each style to confirm the segmentation mask still aligns after style transformation.

### 10.4 Phase 4 — Scale Up (Week 4)

- [ ] Download full Mixamo character library (~100 characters)
- [ ] Curate Sketchfab/Quaternius characters (~100–200 more)
- [ ] Fuzzy bone mapper for non-Mixamo skeletons
- [ ] Per-character override JSON for manual bone mapping fixes
- [ ] Full batch run: all characters × all poses × all styles
- [ ] Dataset manifest generation
- [ ] Train/val/test split by character
- [ ] Dataset statistics and quality report

**Deliverable:** Complete v1 dataset. 300+ characters × 20 poses × 6 styles = 36,000+ training images. Ready for model training.

### 10.5 Phase 5 — 2D Source Integration (Week 5, parallel with training)

- [ ] Spine project parser: extract character image + part assignments → image + mask
- [ ] Spine joint extraction: bone positions from .spine JSON
- [ ] Manual annotation pipeline setup (Label Studio or CVAT)
- [ ] Annotate 50–100 hand-drawn characters for domain coverage
- [ ] Merge 2D sources into the same dataset format

**Deliverable:** Dataset v1.1 with mixed 3D-synthetic and 2D-real sources. The 2D data fills the domain gap that synthetic data can't cover.

---

## 11. Validation & Quality Control

### 11.1 Automated Checks

Run after every batch generation:

- **Mask completeness:** Every non-transparent pixel in the color image has a non-zero region in the mask
- **Mask uniqueness:** No mask is all-one-region (would mean the mapping failed)
- **Joint bounds:** All joint positions fall within the image bounds
- **Joint count:** Each pose has exactly 17 joints (one per region, minus background)
- **File pairing:** Every image has a corresponding mask and joint file
- **Resolution check:** All images are exactly 512 × 512
- **Region distribution:** No single region dominates >60% of pixels (would indicate bad mapping)

### 11.2 Visual Spot Checks

Manual review of randomly sampled outputs:

- **Overlay check:** Mask overlaid on color image with 50% opacity. Region boundaries should follow character anatomy.
- **Joint check:** Joint positions drawn as circles on the color image. Should land on the correct body part.
- **Style check:** Each style variant looks distinct from the others and plausibly resembles hand-drawn art.

Target: review 5% of outputs visually. Flag and fix any systematic errors before training.

### 11.3 Known Failure Cases

Issues to watch for:

- **Skinning bleed:** A vertex weighted 49% to arm, 51% to chest renders as chest but looks wrong. Fix: use weighted blending for boundary faces instead of hard majority vote.
- **Self-occlusion:** In some poses, the arm is behind the torso. The segmentation mask shows the front-most region, but the joint position is at the occluded arm. Fix: flag occluded joints with `visible: false`.
- **Accessories assigned to wrong region:** A cape skinned to the spine bone gets labeled as body. Fix: hide accessories or add an `accessory` region (18).
- **Thin geometry disappearing:** Fingers or thin weapons vanish at 512px. Fix: render at 1024px and downscale, or accept that the model learns to ignore sub-pixel features.

---

## 12. From Dataset to Training

Once the dataset is generated, training connects directly:

### 12.1 Segmentation Training

```bash
# Using MMSegmentation (Apache 2.0)
python tools/train.py configs/deeplabv3plus/deeplabv3plus_m-v3_512x512_strata.py
```

Key config:
- Model: DeepLabV3+ with MobileNetV3 backbone (pretrained on ImageNet)
- Input: 512 × 512 images
- Classes: 18 (background + 17 regions)
- Augmentation: horizontal flip, color jitter, random crop, random scale
- Training: ~200 epochs, batch size 8–16, Adam optimizer, lr 1e-4
- Hardware: single GPU, 4–8 hours

### 12.2 Joint Prediction Training

```bash
# Custom PyTorch training script
python train_joints.py --dataset ./dataset/ --model mlp --epochs 100
```

Input: segmentation mask (predicted or ground truth) → 17 joint positions. Tiny model. Trains in under an hour.

### 12.3 Weight Prediction Training

```bash
# Custom PyTorch training script
python train_weights.py --dataset ./dataset/ --model mlp --epochs 200
```

Input: mesh vertex positions + segmentation region → per-vertex bone weights. Also a small model. 1–2 hours training.

### 12.4 ONNX Export

After training, convert each model for Strata bundling:

```bash
python export_onnx.py --model segmentation --checkpoint best.pth --output strata_seg.onnx
python export_onnx.py --model joints --checkpoint best.pth --output strata_joints.onnx
python export_onnx.py --model weights --checkpoint best.pth --output strata_weights.onnx
```

These .onnx files ship with Strata. Total bundle size target: ~55MB.

---

## 13. Scaling & Iteration

### 13.1 Dataset Versioning

Each complete dataset generation is versioned:

- `dataset_v1.0` — Initial synthetic-only (300 characters, 36K images)
- `dataset_v1.1` — Add 2D sources (Spine + manual annotation)
- `dataset_v2.0` — Expanded characters (500+), non-humanoid support, more styles
- `dataset_v3.0` — User-contributed corrections integrated (post-launch)

Keep all versions. Models trained on v1.0 serve as baseline for comparison.

### 13.2 Continuous Generation

The pipeline is designed to add new characters incrementally:

1. Drop new FBX files into `source_characters/`
2. Run the script with `--only_new` flag (skip already-processed characters)
3. New outputs append to the dataset
4. Re-run train/val/test split (maintaining character-level isolation)

This means you can add characters one at a time as you find good ones, rather than batching everything.

### 13.3 When Is the Dataset Big Enough?

Track validation metrics as the dataset grows:

- **Segmentation mIoU:** Plot validation mIoU vs. dataset size. When the curve plateaus, you have enough data.
- **Expected plateau:** ~5,000–10,000 images for DeepLabV3+ on this task (relatively constrained domain).
- **Minimum viable:** ~2,000 images should get you to 80%+ mIoU. Enough to ship a useful v1.
- **Diminishing returns:** Past ~20,000 images, improvements will be marginal. Focus on domain diversity (more art styles) rather than volume.

---

## 14. Hardware & Dependencies

### 14.1 For Dataset Generation (Blender)

- **Blender 4.0+** (free, open source)
- **Python 3.10+** (bundled with Blender)
- **OpenCV** + **Pillow** (for post-render style augmentation)
- **NumPy** (array operations for weight extraction)
- Any CPU works. Rendering is lightweight (flat shading, no raytracing). A single character with 20 poses × 6 styles takes ~2–5 minutes. Full dataset of 300 characters: ~10–25 hours.
- **No GPU required** for dataset generation. Blender's EEVEE renderer handles flat shading fine on CPU.

### 14.2 For Model Training

- **PyTorch 2.0+**
- **MMSegmentation** (for DeepLabV3+ fine-tuning) or **torchvision** (for simpler setup)
- **Albumentations** (data augmentation)
- **ONNX** + **onnxruntime** (for export and validation)
- **GPU:** NVIDIA with 8GB+ VRAM for DeepLabV3+. 16GB+ for SAM 2 fine-tuning.
- **Cloud options:** Lambda Labs ($0.50–1.50/hr for T4/A10), Vast.ai (often cheaper), Google Colab Pro ($10/mo)

### 14.3 Total Cost Summary

| Item | Cost |
|------|------|
| Blender | Free |
| Python + libraries | Free |
| Mixamo assets | Free (Adobe account) |
| Sketchfab CC assets | Free |
| Quaternius / Kenney | Free |
| Spine asset packs (optional) | $50–150 |
| Label Studio (annotation) | Free |
| Cloud GPU training (~10 runs total) | $50–100 |
| **Total** | **$50–250** |

---

## 15. Quick Start Checklist

For getting the first training data out the door:

- [ ] Install Blender 4.0+
- [ ] Create Adobe account, download 10 Mixamo characters (FBX, T-pose)
- [ ] Download 5 Mixamo animations (walk, run, idle, attack, jump)
- [ ] Clone the pipeline repo (once you build it)
- [ ] Run: `blender --background --python generate_dataset.py -- --input_dir ./characters/ --output_dir ./dataset/`
- [ ] Visually verify 10 outputs (overlay mask on image)
- [ ] Scale up to full character library
- [ ] Train first model and evaluate

---

*The best training data is the kind that generates itself. Build the pipeline once, feed it characters forever. Every model in Strata's AI stack drinks from this well.*

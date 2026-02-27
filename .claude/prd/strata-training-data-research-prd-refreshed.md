# Strata Training Data — Research-Informed Improvements PRD

**Version:** 1.1
**Date:** February 27, 2026
**Status:** Planned
**Context:** v1.0 informed by "See-through" (arxiv, Feb 2026), Spiritus (ACM UIST 2025), Qwen-Image-Layered (Dec 2025), and ASMR (Mar 2025). v1.1 adds Section 13 (3D Mesh Pipeline requirements) informed by StdGEN (CVPR 2025), CharacterGen (SIGGRAPH 2024), UniRig (SIGGRAPH 2025), DrawingSpinUp (SIGGRAPH Asia 2024), MVPaint (2024), TripoSG (2025), and Hunyuan3D (2025).
**Repository:** strata-training-data

---

## 1. Problem Statement

The current training data pipeline generates segmentation masks and joint positions from Mixamo FBX characters rendered through Blender. This produces high-quality data but has two blind spots:

1. **Style homogeneity.** All training images come from 3D-rendered characters. The segmentation model will encounter hand-painted, pixel art, anime, and vector-style characters in production. The domain gap between 3D renders and 2D art is the same gap that caused SAM to fail for the Spiritus team.

2. **Missing output dimensions.** The pipeline generates segmentation labels and joint positions, but not draw order (which layer renders in front of which), occluded region data (what's hidden behind overlapping parts), or semantic features that could improve weight painting. The See-through paper proved these are learnable and valuable.

This PRD addresses both gaps by adding new data sources and new annotation types to the existing pipeline, without disrupting the Mixamo-based workflow that's already working.

---

## 2. New Data Source: Live2D Community Models

### 2.1 Rationale

The See-through paper built a dataset of 9,102 annotated characters by scraping Live2D models from community platforms. Live2D models are pre-decomposed into layered ArtMesh fragments — they provide near-free segmentation ground truth for 2D illustrated characters.

This directly addresses the style diversity problem. Live2D models span anime, chibi, realistic, stylized, and fantasy art styles. Adding them to training gives the segmentation model examples of hand-drawn anatomy alongside the Mixamo 3D renders.

### 2.2 What Live2D Models Provide

A Live2D model (.moc3 + textures) contains:

- **ArtMesh fragments:** Individual image regions (hair_front, eye_left, torso, arm_l, etc.) with pixel-precise boundaries
- **Fragment names:** Often semantically meaningful (though naming varies by artist)
- **Draw order:** Which fragment renders in front of which, defined as a numerical ordering
- **Deformation mesh:** Per-fragment triangulation (useful reference for Strata's mesh generation)
- **Parameter associations:** Which fragments move with which animation parameters

### 2.3 Acquisition Strategy

**Sources (free/CC-licensed models):**

| Source | Volume | License | Notes |
|--------|--------|---------|-------|
| Booth.pm (free section) | ~200–500 models | Varies (check per model) | Largest Live2D marketplace. Many free models with redistribution rights for derivative works. |
| DeviantArt | ~50–100 | Varies | Smaller volume, wider style variety |
| GitHub/open-source | ~30–50 | MIT/CC | Open-source VTuber models, sample projects |
| Live2D sample models | ~10–20 | Free for non-commercial | Official samples from Live2D Inc. |

**Target:** 300–500 models across art styles. The See-through team used ~850 for training + ~50 for validation. We don't need as many because we're supplementing Mixamo data, not replacing it.

**Legal check:** Only use models with licenses permitting derivative use for ML training. Document license per model in a CSV manifest.

### 2.4 Fragment-to-Strata Label Mapping

Live2D fragment names are artist-defined and inconsistent. Need a mapping tool.

**Automated mapping (first pass):**

```python
# Keyword-based mapping from Live2D fragment names to Strata 21-label taxonomy
FRAGMENT_MAP = {
    # Head
    r'head|face|kao': 'head',
    r'hair.*front|bangs|maegami': 'head',  # hair_front → head region
    r'hair.*back|ushirogami': 'head',
    r'eye.*[lL]|me_l': 'head',  # eyes map to head in base model
    r'eye.*[rR]|me_r': 'head',
    r'mouth|kuchi|lip': 'head',
    r'brow|mayu': 'head',
    r'ear|mimi': 'head',
    r'nose|hana': 'head',

    # Torso
    r'body|torso|karada|chest|mune': 'torso',
    r'neck|kubi': 'neck',

    # Arms
    r'arm.*upper.*[lL]|ude.*ue.*[lL]': 'upper_arm_l',
    r'arm.*upper.*[rR]|ude.*ue.*[rR]': 'upper_arm_r',
    r'arm.*lower.*[lL]|arm.*fore.*[lL]': 'forearm_l',
    r'arm.*lower.*[rR]|arm.*fore.*[rR]': 'forearm_r',
    r'hand.*[lL]|te_[lL]': 'hand_l',
    r'hand.*[rR]|te_[rR]': 'hand_r',

    # Legs
    r'leg.*upper.*[lL]|thigh.*[lL]|momo.*[lL]': 'upper_leg_l',
    r'leg.*upper.*[rR]|thigh.*[rR]|momo.*[rR]': 'upper_leg_r',
    r'leg.*lower.*[lL]|shin.*[lL]|sune.*[lL]': 'lower_leg_l',
    r'leg.*lower.*[rR]|shin.*[rR]|sune.*[rR]': 'lower_leg_r',
    r'foot.*[lL]|ashi.*[lL]': 'foot_l',
    r'foot.*[rR]|ashi.*[rR]': 'foot_r',

    # Hips
    r'hip|pelvis|koshi|waist': 'hips',

    # Accessories
    r'cloth|dress|skirt|hat|ribbon|accessory|cape|armor|weapon|shield': 'accessory',
}
```

**Manual review (second pass):** The automated mapper will misclassify ~20–30% of fragments (artists use idiosyncratic naming). Build a simple review UI:

1. Render the Live2D model composite image
2. Highlight each fragment one at a time
3. Show the auto-assigned Strata label
4. Let the human confirm or correct with a single keypress

At ~5 seconds per fragment, 20 fragments per model, 400 models = ~11 hours of annotation work. Spreadable across days.

### 2.5 Render Pipeline for Live2D Models

**Input:** .moc3 model + texture atlas
**Output:** Composite image + segmentation mask + draw order map

```
live2d_renderer.py:
    1. Load .moc3 model (use Live2D Cubism SDK for Native or cubism-rs)
    2. Set all parameters to default (neutral pose)
    3. Render composite image at 512×512 (PNG, transparent background)
    4. For each ArtMesh fragment:
       a. Render fragment in isolation → fragment mask (binary)
       b. Look up Strata label from mapping CSV
       c. Paint fragment pixels with label color on segmentation mask
       d. Record fragment draw order index
    5. Generate draw order map (per-pixel float: 0.0 = backmost layer, 1.0 = frontmost)
    6. Save: composite.png, segmentation.png, draw_order.png, metadata.json
```

**Augmentation:**
- Horizontal flip (doubles data, free)
- Slight rotation (±5°) and scale (±10%)
- Color jitter (hue, saturation, brightness)
- NO style transfer — the whole point is to keep natural art styles

**Volume:** 400 models × 4 augmentations = ~1,600 training images with pixel-perfect art-style segmentation.

### 2.6 Repository Structure Addition

```
strata-training-data/
├── data/
│   └── live2d/                     ← ⛔ .gitignore (large model files)
│       └── README.md               ← Sources, download instructions, license info
├── segmentation/
│   ├── scripts/
│   │   ├── live2d_renderer.py      ← ✅ Track — render pipeline
│   │   ├── live2d_mapper.py        ← ✅ Track — fragment name → Strata label
│   │   └── live2d_review_ui.py     ← ✅ Track — manual annotation review tool
│   └── labels/
│       └── live2d_mappings.csv     ← ✅ Track — model_id, fragment_name, strata_label, confirmed
```

---

## 3. New Annotation Type: Draw Order

### 3.1 Rationale

When a 2D character is decomposed into animated layers, the rendering engine needs to know which layer draws in front. The left arm might be in front of the torso, the torso in front of the right arm, hair_front in front of the face, etc.

The See-through paper treats this as a per-pixel pseudo-depth prediction problem. They normalize fragment draw order into a [0, 1] float and train a model to predict it alongside segmentation. This is directly applicable to Strata.

### 3.2 Draw Order for Mixamo Renders

The Blender pipeline already renders from a fixed camera. Body part depth relative to camera is deterministic from the pose:

```python
# For each labeled region, compute average Z-depth from camera
def compute_draw_order(character, camera, segmentation_labels):
    draw_order = {}
    for label_id, label_name in segmentation_labels.items():
        # Get all vertices belonging to bones mapped to this label
        vertices = get_region_vertices(character, label_name)
        # Project to camera space, get average Z
        avg_depth = mean([camera_project(v).z for v in vertices])
        draw_order[label_name] = avg_depth

    # Normalize to [0, 1] range (0 = furthest from camera, 1 = closest)
    min_d, max_d = min(draw_order.values()), max(draw_order.values())
    for name in draw_order:
        draw_order[name] = (draw_order[name] - min_d) / (max_d - min_d)

    return draw_order
```

**Output:** Per-pixel depth map as a grayscale PNG alongside the existing segmentation mask. Brighter = closer to camera = renders on top.

**Pipeline change:** Add `draw_order_extractor.py` to the render pipeline. Runs after segmentation mask generation. Zero additional render time — uses existing vertex data and camera parameters.

### 3.3 Draw Order for Live2D Models

Already provided by the model format. Each ArtMesh has an explicit render order index. Normalize to [0, 1] and paint as a grayscale map.

### 3.4 Training Data Format Extension

Current output per training example:
```
example_001/
├── image.png           ← Character render (512×512)
├── segmentation.png    ← Per-pixel label IDs (21 colors)
└── joints.json         ← 2D joint positions
```

Extended output:
```
example_001/
├── image.png           ← Character render (512×512)
├── segmentation.png    ← Per-pixel label IDs (21 colors)
├── draw_order.png      ← Per-pixel depth (grayscale, 0=back 255=front)  ← NEW
├── joints.json         ← 2D joint positions
└── metadata.json       ← Source type, style, pose name, draw order values  ← NEW
```

---

## 4. Taxonomy Validation: 19 vs 21 Labels

### 4.1 Rationale

The See-through paper uses a 19-class body part taxonomy tuned for anime characters. Strata uses 21 labels (19 body + background + accessory) tuned for game character rigging. These developed independently — comparing them may reveal blind spots.

### 4.2 Comparison Task

Create a document (`docs/taxonomy-comparison.md`) that maps each See-through class to each Strata class and identifies:

1. **Classes present in See-through but not Strata** — especially hair sub-regions. Anime characters often have hair that "sandwiches" the face (strands in front AND behind). If See-through separates hair_front / hair_back, Strata should consider whether this matters for painted characters.

2. **Classes present in Strata but not See-through** — Strata has `accessory` and `background`. See-through is character-only (no background, accessories absorbed into nearest body part).

3. **Boundary definitions that differ** — Where does "neck" end and "torso" begin? Where do "hips" end and "upper_leg" begin? These boundary decisions affect rigging quality. If See-through defines them differently, worth understanding why.

### 4.3 Potential Taxonomy Changes

**Hair layering consideration:**

Current Strata taxonomy treats all hair as part of `head` (label 1). If an artist paints a character with hair flowing behind the shoulders, that hair region is segmented as `head` and deforms with the head bone.

For animation, hair behind the shoulders should arguably follow the torso/shoulders, not the head. Consider adding:

| Option | Change | Impact |
|--------|--------|--------|
| A: No change | Hair stays in `head` | Simple. Works for most cases. Long hair behind shoulders looks stiff when head turns. |
| B: Add `hair_back` | New label 22 | Separate region for hair behind the neck/shoulder line. Deforms with spine/shoulder bones instead of head. Better animation for long-haired characters. |
| C: Detail-region approach | Add as optional detail under `head` | Same as option B but opt-in, like fingers/toes. Doesn't bloat base model. |

**Recommendation:** Option C. Add `hair_back` as an optional detail region in the Detail Regions PRD, not in the base 21-label model. Most characters don't need it. Characters with long flowing hair opt in.

### 4.4 Deliverable

```
docs/taxonomy-comparison.md       ← ✅ Track — class-by-class comparison with decisions
```

---

## 5. BVH Retargeting Pipeline

### 5.1 Rationale

The Animation Intelligence PRD plans for CMU mocap (BVH format) → Strata blueprint conversion. Spiritus has already solved BVH retargeting for 2D characters with different proportions. Their approach documents the joint remapping conventions.

The CMU dataset has 2,548 clips using one skeleton definition. Strata uses a 19-bone skeleton. The retargeting pipeline maps between them, handling proportion mismatches.

### 5.2 Scripts to Build

```
animation/scripts/
├── bvh_parser.py              ← Parse BVH files into bone hierarchy + frame data
├── bvh_to_strata.py           ← Retarget BVH skeleton → Strata 19-bone skeleton
├── proportion_normalizer.py   ← Normalize bone lengths to Strata proportions
└── blueprint_exporter.py      ← Export retargeted animation as Strata blueprint JSON
```

**bvh_to_strata.py core mapping:**

```python
# CMU BVH skeleton → Strata 19-bone skeleton
CMU_TO_STRATA = {
    'Hips':             'hips',
    'Spine':            'spine',       # CMU has Spine, Spine1, Spine2 — collapse to one
    'Spine1':           'spine',
    'Neck':             'neck',
    'Head':             'head',
    'LeftShoulder':     'shoulder_l',  # CMU shoulder → Strata shoulder
    'LeftArm':          'upper_arm_l',
    'LeftForeArm':      'forearm_l',
    'LeftHand':         'hand_l',
    'RightShoulder':    'shoulder_r',
    'RightArm':         'upper_arm_r',
    'RightForeArm':     'forearm_r',
    'RightHand':        'hand_r',
    'LeftUpLeg':        'upper_leg_l',
    'LeftLeg':          'lower_leg_l',
    'LeftFoot':         'foot_l',
    'RightUpLeg':       'upper_leg_r',
    'RightLeg':         'lower_leg_r',
    'RightFoot':        'foot_r',
}
# Bones not in Strata (LeftToeBase, finger bones, etc.) → ignored
# CMU multi-spine → Strata single spine: average rotations or use Spine1
```

**Proportion normalization:** CMU actors have realistic human proportions. Strata characters may be chibi (big head, small body), stylized (long legs), etc. The retargeter needs to:

1. Extract bone lengths from CMU BVH (T-pose frame)
2. Accept Strata character's bone lengths as target
3. Scale translations (hip movement, root motion) proportionally
4. Keep rotations unchanged (angles transfer regardless of bone length)

### 5.3 Action Labeling Integration

The `cmu_action_labels.csv` file already planned in the repo structure becomes the bridge:

```csv
filename,action_type,subcategory,quality,strata_compatible
01_01.bvh,walk,forward,high,yes
02_03.bvh,jump,standing,medium,yes
05_12.bvh,basketball,dribble,high,no
...
```

The `strata_compatible` column flags clips that map cleanly to the 19-bone skeleton. Full-body locomotion clips transfer well. Hand-specific or face-specific clips don't (no finger/face bones in base skeleton).

### 5.4 Synthetic Degradation for Animation Intelligence

The Animation Intelligence PRD identified synthetic degradation as the primary training data strategy for the in-betweening model. This pipeline creates it:

```
animation/scripts/
└── degrade_animation.py       ← Strip principled animation to sparse key poses
```

Takes a good animation (retargeted BVH or handmade blueprint) and removes frames to create (sparse input, full output) training pairs:

| Degradation Type | What it Removes | Training Signal |
|-----------------|----------------|-----------------|
| Strip to extremes | Remove all but extreme poses | Model learns to insert anticipation, follow-through |
| Linearize arcs | Replace curved paths with straight lines | Model learns to add arc corrections |
| Remove easing | Replace ease-in/out with linear timing | Model learns to add easing |
| Remove secondary | Lock secondary bones to parent | Model learns to add follow-through |
| Reduce framerate | Drop to every Nth frame | Model learns general in-betweening |
| Simultaneous stop | Make all bones stop on same frame | Model learns staggered settling |
| Remove anticipation | Delete counter-movement frames before actions | Model learns to insert anticipation |

Each good animation generates 7 degraded versions = 7 training pairs.

---

## 6. Occluded Region Data Collection (Future)

### 6.1 Rationale

The See-through paper's most novel contribution is inpainting occluded anatomy — generating plausible content for body parts hidden behind other parts. When a flat painting is decomposed into layers, the torso behind the arm is a gap. The See-through team trains a diffusion model to fill these gaps while preserving the artist's style.

This is a v2.0+ feature for Strata but the training data should be collected alongside current work because the opportunity is there now.

### 6.2 Data from Mixamo Renders

The Blender pipeline already renders characters in various poses. For occluded region data:

1. For each pose, render the **full character** (composite, as currently done)
2. For each body region, render the region **in isolation** with all occluding regions removed
3. The difference between the composite and the isolated render reveals what's hidden

```python
# For each region in [torso, upper_arm_l, upper_arm_r, ...]:
#   1. Hide all regions that render IN FRONT of this region
#   2. Render just this region → full_region.png (includes normally-hidden parts)
#   3. Render composite normally → visible_region.png (with occlusion)
#   4. Difference = occluded area that needs inpainting
```

**Output per training example (additional):**
```
example_001/
├── ...existing files...
├── occlusion/
│   ├── torso_full.png           ← Torso rendered without occluding arms
│   ├── torso_visible.png        ← Torso as seen in composite (with arm occlusion)
│   ├── torso_mask.png           ← Binary mask of occluded torso pixels
│   ├── upper_arm_l_full.png
│   └── ...
```

This data enables future training of an inpainting model: given visible region + occlusion mask → predict full region.

### 6.3 Data from Live2D Models

Live2D models provide this for free. Each ArtMesh fragment is a complete layer — the full extent of that body part, including parts normally hidden by overlapping fragments. Render each fragment independently and you have:

- Complete body part images (no occlusion)
- The composite (with natural occlusion from draw order)
- Occlusion masks (difference between the two)

### 6.4 Timeline

Don't build this now. Add it to the render pipeline **when the pipeline is stable** (after base segmentation data is validated). The incremental work is ~1 week to add isolated-region rendering to both the Mixamo and Live2D pipelines.

---

## 7. PSD File Collection (Opportunistic)

### 7.1 Rationale

Qwen-Image-Layered trained on real Photoshop documents, using PSD layer structure as ground truth for image decomposition. Many digital artists save layered PSDs with body parts on separate layers — these are natural segmentation annotations.

### 7.2 Collection Strategy

Not a priority pipeline — more of an opportunistic data collection:

- When artists share layered character PSDs (OpenGameArt, itch.io asset packs, Patreon-released art packs), save them
- Extract layer structure and map layer names to Strata taxonomy
- Flatten to composite + segmentation mask pairs

**Small script:**
```
segmentation/scripts/
└── psd_extractor.py    ← Extract layers from PSD, map to Strata labels, generate mask
```

Uses `psd-tools` Python library. Per-PSD processing is fast (~seconds). The bottleneck is finding PSDs with body-part-separated layers (most PSDs are organized by rendering concern — lineart, color, shading — not by body part).

### 7.3 Expected Volume

Low. Maybe 50–100 usable PSDs found opportunistically over months. But each one is a real artist's real work — incredibly valuable for style diversity even in small quantities.

---

## 8. Diff3F Semantic Feature Exploration (Research)

### 8.1 Rationale

The ASMR paper uses Diffusion 3D Features (Diff3F) — semantic descriptors extracted from diffusion model activations — to improve rigging quality. A pixel at an elbow carries features that say "this is a joint/bend region" regardless of art style.

This is relevant to Strata's weight painting prediction. Instead of inferring skin weights purely from geometric position + segmentation label, you could use diffusion features as additional input. This could improve weight painting on unusual body proportions.

### 8.2 Exploration Task

This is a research task, not a production pipeline change:

1. **Install Diff3F** (open source, available from the paper's repo)
2. **Extract features** from 20–30 Strata training images (Mixamo renders + Live2D composites)
3. **Visualize:** Do the features cluster by body part? By joint proximity? By deformation type?
4. **Evaluate:** Would these features improve weight prediction vs. current approach (geometric position + label)?

**Output:** A Jupyter notebook in `docs/research/diff3f_exploration.ipynb` with findings and recommendation on whether to integrate.

### 8.3 Timeline

2–3 days of exploration, non-blocking. If results are promising, file a follow-up task to integrate into the weight prediction pipeline.

---

## 9. Updated Repository Structure

Incorporating all additions, including pre-processed dataset ingestion (v2, Feb 27 2026):

```
strata-training-data/
│
├── pipeline/                               ← Blender/Python rendering pipeline
│   ├── bone_mapper.py                      ← Map skeleton bones → Strata 22-label taxonomy
│   ├── config.py                           ← Shared constants, label colors, camera angles
│   ├── exporter.py                         ← Export training examples to output/
│   ├── generate_dataset.py                 ← Main entry point — orchestrates full pipeline
│   ├── importer.py                         ← Import FBX models into Blender
│   ├── joint_extractor.py                  ← Extract 2D joint positions from 3D skeleton
│   ├── pose_applicator.py                  ← Apply animation poses to rigged models
│   ├── renderer.py                         ← Render composite images from Blender scenes
│   ├── weight_extractor.py                 ← Extract skin weights for segmentation masks
│   ├── draw_order_extractor.py             ← Compute per-pixel depth from camera Z-buffer
│   ├── multi_angle_renderer.py             ← Render from 5 camera angles (0°/45°/90°/135°/180°)
│   ├── measurement_extractor.py            ← Extract per-label 2D→3D body measurements
│   ├── measurement_ground_truth.py         ← True body dimensions from mesh bounding boxes
│   ├── contour_renderer.py                 ← Blender Freestyle contour on/off pair rendering
│   └── contour_augmenter.py               ← Hand-drawn contour style variations (5 styles)
│
├── ingest/                                 ← NEW — Pre-processed dataset ingestion adapters
│   ├── nova_human_adapter.py               ← Convert NOVA-Human renders → Strata training format
│   ├── stdgen_pipeline_ext.py              ← Extend StdGEN's distributed_uniform.py with Strata outputs
│   ├── stdgen_semantic_mapper.py           ← Map StdGEN 4-class (body/clothes/hair/face) → Strata 22
│   ├── animerun_contour_adapter.py         ← Convert AnimeRun contour pairs → contour training format
│   ├── unirig_skeleton_mapper.py           ← Map UniRig bone hierarchies → Strata 20-bone skeleton
│   ├── linkto_adapter.py                   ← Extract skeleton + flow data from LinkTo-Anime
│   ├── vroid_importer.py                   ← Import VRM/VRoid models into Blender (wraps StdGEN scripts)
│   ├── vroid_mapper.py                     ← VRoid material/bone → Strata label mapping
│   └── download_datasets.sh               ← Master download script for all pre-processed datasets
│
├── data/                                   ← ⛔ ALL subdirectories .gitignore'd (large files)
│   ├── fbx/                                ← Mixamo FBX character + animation files
│   ├── poses/                              ← Pose definition files
│   ├── mocap/                              ← CMU BVH motion capture clips
│   ├── sprites/                            ← Sprite sheet source files
│   ├── live2d/                             ← Live2D .moc3 + texture atlas files
│   │   └── README.md                       ← Sources: Booth.pm, DeviantArt, GitHub, Live2D Inc.
│   ├── psd/                                ← Layered Photoshop files (opportunistic)
│   │   └── README.md
│   ├── vroid/                              ← VRM source files from PAniC-3D downloader
│   │   └── README.md                       ← VRoid Hub sources, PAniC-3D methodology, cookie setup
│   │
│   └── preprocessed/                       ← NEW — Downloaded pre-processed datasets
│       ├── nova_human/                     ← NOVA-Human: 10.2K chars × 20 views (~50-80 GB)
│       │   └── README.md                   ← Download: github.com/NOVA-3D-Anime-Character-Synthesis/NOVA-3D
│       ├── stdgen/                          ← StdGEN: train/test lists, rendering scripts, weights
│       │   └── README.md                   ← Download: github.com/hyz317/StdGEN + HuggingFace
│       ├── unirig/                         ← UniRig Rig-XL: 14K rigged meshes (~20 GB)
│       │   └── README.md                   ← Download: github.com/VAST-AI-Research/UniRig
│       ├── animerun/                       ← AnimeRun: ~8K contour/color pairs (~5 GB)
│       │   └── README.md                   ← Download: lisiyao21.github.io/projects/AnimeRun
│       ├── linkto_anime/                   ← LinkTo-Anime: 29K frames + skeleton + flow (~10 GB)
│       │   └── README.md                   ← Download: check arXiv 2506.02733
│       ├── fbanimehq/                      ← FBAnimeHQ: 112K full-body images (~25 GB)
│       │   └── README.md                   ← Download: HuggingFace skytnt/fbanimehq
│       ├── anime_segmentation/             ← skytnt/anime-segmentation: fg/bg masks
│       │   └── README.md
│       ├── anime_instance_seg/             ← dreMaz/AnimeInstanceSegmentationDataset
│       │   └── README.md
│       └── charactergen/                   ← CharacterGen: alternative render scripts
│           └── README.md                   ← Download: github.com/zjp-shadow/CharacterGen
│
├── segmentation/
│   ├── scripts/
│   │   ├── live2d_renderer.py              ← Render Live2D models to training data
│   │   ├── live2d_mapper.py                ← Fragment name → Strata label (regex + manual)
│   │   ├── live2d_review_ui.py             ← Manual annotation review tool
│   │   ├── psd_extractor.py                ← PSD layer extraction → segmentation masks
│   │   └── occlusion_renderer.py           ← (future) Isolated region rendering for inpainting
│   └── labels/
│       ├── live2d_mappings.csv             ← model_id, fragment_name, strata_label, confirmed
│       ├── vroid_mappings.csv              ← VRoid-specific label overrides
│       └── stdgen_to_strata.json           ← NEW — StdGEN 4-class → Strata 22-class mapping rules
│
├── animation/
│   ├── scripts/
│   │   ├── bvh_parser.py                   ← Parse BVH files into bone hierarchy + frame data
│   │   ├── bvh_to_strata.py                ← Retarget BVH skeleton → Strata 19-bone skeleton
│   │   ├── proportion_normalizer.py        ← Scale mocap to arbitrary character proportions
│   │   ├── blueprint_exporter.py           ← Export retargeted animation as Strata blueprint JSON
│   │   ├── degrade_animation.py            ← Synthetic degradation (7 types) for AI training
│   │   ├── label_actions.py                ← Labeling helpers for CMU action categorization
│   │   └── extract_timing.py               ← Extract timing norms per action type
│   ├── labels/
│   │   └── cmu_action_labels.csv
│   ├── breakdowns/
│   └── timing-norms/
│
├── mesh/                                   ← 3D mesh pipeline training data
│   ├── scripts/
│   │   ├── proportion_clusterer.py         ← Cluster measurement profiles → template archetypes
│   │   └── texture_projection_trainer.py   ← (future) Partial→complete texture pairs
│   └── measurements/
│       └── measurement_profiles.json       ← Per-character ground truth dimensions
│
├── output/                                 ← ⛔ .gitignore — all generated training data
│   ├── segmentation/                       ← Training images + masks + joints + metadata
│   │   ├── mixamo/                         ← Renders from Mixamo FBX pipeline
│   │   ├── vroid/                          ← Renders from VRoid/StdGEN pipeline
│   │   ├── vroid_preprocessed/             ← Converted from NOVA-Human pre-renders
│   │   ├── live2d/                         ← Rendered from Live2D models
│   │   ├── psd/                            ← Extracted from PSD files
│   │   └── contour_pairs/                  ← Contour on/off pairs (generated + AnimeRun)
│   ├── animation/                          ← Retargeted BVH + degraded pairs + blueprints
│   └── mesh/                               ← Measurement profiles + texture projection pairs
│
├── docs/
│   ├── data-sources.md                     ← Overview of all data sources and their status
│   ├── labeling-guide.md                   ← Taxonomy definitions and labeling conventions
│   ├── taxonomy-comparison.md              ← Strata 22 vs See-through 19 class-by-class analysis
│   ├── preprocessed-datasets.md            ← NEW — Catalog of all pre-processed datasets with download/license info
│   └── research/
│       └── diff3f_exploration.ipynb        ← Diff3F feature evaluation notebook
│
├── scripts/                                ← NEW — Top-level utility scripts
│   ├── verify_downloads.py                 ← Verify all pre-processed datasets are complete
│   ├── compute_stats.py                    ← Count images, labels, coverage per source
│   └── generate_splits.py                  ← Generate train/val/test splits across all sources
│
└── .gitignore
```

**Key structural change (v2):** The new `ingest/` directory contains adapter scripts that convert pre-processed datasets from external research teams (NOVA-Human, StdGEN, UniRig, AnimeRun, etc.) into Strata's training format. This replaces the original plan of building everything from scratch. The `data/preprocessed/` directory holds the raw downloads, and the adapters convert them into `output/` format.

**What moved:** `vroid_importer.py` and `vroid_mapper.py` moved from `pipeline/` and `segmentation/scripts/` into `ingest/` — they now wrap StdGEN's proven Blender rendering scripts rather than being standalone implementations.

---

## 10. Implementation Priority

### Phase 1 — Immediate (week 1, parallel with base pipeline)

| Task | Effort | Impact | Why Now |
|------|--------|--------|---------|
| **Download pre-processed datasets** (NOVA-Human, StdGEN, UniRig, AnimeRun) | 1–2 days (bandwidth) | Critical | ~355K images available immediately. Highest ROI action in entire project. |
| **Write ingest adapters** (nova_human_adapter.py, stdgen_semantic_mapper.py) | 3–5 days | Critical | Converts downloaded data into Strata training format. |
| Add draw_order_extractor.py to Blender pipeline | 2 days | High | Free data from existing renders. Must be in place before training starts. |
| Taxonomy comparison document | 1 day | Medium | Informs whether to add hair_back before v1 model training. Includes StdGEN 4-class → Strata 22-class mapping design. |
| BVH parser + retargeting scripts | 1 week | High | Needed for CMU mocap labeling (already planned). |
| CMU action labeling kickoff | Ongoing | High | Already planned. Add `strata_compatible` column. |

### Phase 2 — Near-term (weeks 2–6)

| Task | Effort | Impact | Why |
|------|--------|--------|-----|
| **stdgen_pipeline_ext.py** — extend StdGEN Blender script with Strata outputs | 3–5 days | Critical | Adds 45° angle, joints, draw order, measurements, contours to proven VRoid rendering pipeline. |
| **Run extended pipeline on 10,811 VRoid characters** | 1–2 weeks (compute) | Critical | Core multi-view anime training data with full Strata annotations. |
| Live2D model acquisition (200+ models) | 1 week | High | Style diversity for segmentation model. |
| live2d_mapper.py + automated mapping | 3 days | High | Produces mappings for annotation review. |
| live2d_review_ui.py + manual review | 1 week | High | Confirms ~400 models × 20 fragments. |
| live2d_renderer.py + training data generation | 3 days | High | Generates 1,600+ art-style training images. |
| degrade_animation.py + synthetic pairs | 3 days | Medium | Training data for future in-betweening model. |

### Phase 3 — Research (non-blocking, fit around other work)

| Task | Effort | Impact | Why |
|------|--------|--------|-----|
| Diff3F exploration notebook | 2–3 days | Uncertain | May improve weight painting. Low risk to investigate. |
| PSD extractor script | 1 day | Low (for now) | Opportunistic. Script is simple, data is hard to find. |

### Phase 4 — Future (v2.0 timeframe)

| Task | Effort | Impact | Why |
|------|--------|--------|-----|
| Occluded region rendering pipeline | 1 week | High (future) | Enables single-painting decomposition magic. |
| Occlusion data from Live2D models | 2 days | High (future) | Free data — Live2D fragments are pre-separated. |

---

## 11. Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|---------------|
| Training data style diversity | 1 source (Mixamo 3D renders) | 5+ sources (Mixamo + VRoid/NOVA + Live2D + AnimeRun + PSD) | Count distinct art styles in training set |
| Pre-processed data ingested | 0 | ~355K images from 6+ external datasets | Count converted images in output/segmentation/vroid_preprocessed/ etc. |
| Segmentation accuracy on hand-drawn art | Unknown (model not yet trained) | IoU ≥ 0.85 on held-out illustrated characters | Test set with 100 hand-drawn character images |
| Segmentation accuracy on non-frontal views | N/A (no non-frontal data) | IoU ≥ 0.80 on 3/4 and side views | Test set with multi-angle renders held out per character |
| Draw order prediction | Not available | Accuracy ≥ 90% on layer ordering | Compare predicted vs. ground-truth draw order |
| BVH retargeting coverage | 0 clips | 500+ labeled, retargeted clips | Count in cmu_action_labels.csv |
| Synthetic degradation pairs | 0 | 3,500+ pairs (500 clips × 7 degradation types) | Count in animation/output/ |
| Measurement extraction accuracy | N/A | Mean error ≤ 5% of true dimension | Compare 2D-extracted vs. mesh ground truth (validated against UniRig) |
| Multi-view angle coverage | 1 angle (front) | 5 angles (0°, 45°, 90°, 135°, 180°) | Verify all angles present in training output |
| VRoid training data volume | 0 | 50,000+ multi-angle renders (10K from extended pipeline + NOVA-Human converted) | Count in output/segmentation/vroid/ + vroid_preprocessed/ |
| Contour pair volume | 0 | 58,000+ pairs (8K AnimeRun + 50K generated) | Count in output/segmentation/contour_pairs/ |

---

## 12. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Live2D SDK licensing restricts use for ML training | Medium | High | Use only the rendering output (images), not the SDK in the product. Check Cubism SDK license terms. Alternative: write a minimal .moc3 parser that reads ArtMesh geometry without the SDK. |
| Fragment naming too inconsistent for automated mapping | High | Medium | Accept 30% manual correction rate. The review UI handles this. Budget the annotation time. |
| Diff3F features don't help weight painting | Medium | Low | It's a research exploration. No production dependency. |
| Occluded region inpainting is too hard to ship | Medium | Medium | It's a v2.0+ feature. The data collection is cheap even if the model proves difficult. |
| VRoid Hub terms restrict ML training use | Medium | High | Check per-model license terms. Many models are CC-BY or allow derivative works. Follow PAniC-3D team's methodology — they navigated this for 14,500 models. Worst case: render only models with explicit derivative-use permission. |
| Segmentation model degrades on non-frontal views | Medium | Critical | Multi-angle training data is the mitigation. Validate early with held-out test set at each angle. If 3/4 view segmentation is poor, increase VRoid/Mixamo render volume at that angle. |
| Measurement extraction from 3/4 view is noisy | Medium | Medium | Ground truth from mesh data validates accuracy. If extraction error >5%, add side view as required input (fallback to front+side+3/4 configuration). |
| Contour line detection fails on some art styles | Low | Medium | Augmented contour training covers 5 style variants. Artists can also manually flag "this artwork has contour lines" for explicit handling. |
| Pre-processed dataset format drift | Medium | Medium | External datasets may change structure or go offline. Pin specific versions/commits. Mirror critical downloads locally. `verify_downloads.py` checks integrity on each run. |
| StdGEN 4-class → Strata 22-class mapping loses granularity | High | Medium | StdGEN only labels body/clothes/hair/face — can't distinguish upper_arm from forearm within "body." Must re-render from VRM source files using bone weights for full 22-class labels. StdGEN annotations still valuable for coarse training + as validation signal. |
| NOVA-Human renders lack Strata-specific annotations | Certain | Low | NOVA-Human provides images + masks + normals but NOT segmentation, joints, or draw order. These are inputs to the pipeline, not complete training examples. Must run ingest adapter to convert, and supplement with StdGEN-extended renders for full annotations. |
| Pre-processed data licensing incompatible with commercial use | Medium | High | Most datasets are released for research use only. If Strata is commercial, may need to: (a) train only on permissively-licensed subsets, (b) use pre-processed data for validation/testing only, not training, or (c) re-render from source VRM files under VRoid Hub's per-model terms. Track licensing status in `docs/preprocessed-datasets.md`. |

---

---

## 13. 3D Mesh Pipeline: Multi-View Training Data Requirements

**Context:** The 3D Mesh Research PRD (strata-3d-mesh-research-prd.md, Feb 27, 2026) establishes a hybrid template mesh + neural texture pipeline requiring front + 3/4 view input. This section documents the training data gaps that the 3D mesh pipeline introduces.

### 13.1 Problem: Segmentation Model Only Sees Front Views

The entire current pipeline — Mixamo renders and Live2D composites — produces front-facing images. When the segmentation model encounters a 3/4 view (45°), a side view (90°), or a back view (180°), it has zero training signal for how body parts look from those angles.

This matters because: body part boundaries shift dramatically with viewing angle. A front view clearly separates left arm from right arm. A 3/4 view partially occludes the far arm behind the torso. A side view collapses left/right into a single silhouette. The model must handle all these cases for the multi-view measurement extraction pipeline to work.

### 13.2 Multi-Angle Renders from Mixamo Pipeline

**Change to existing Blender pipeline:** Add camera angle as a render parameter.

```python
# renderer.py — extend camera configuration
CAMERA_ANGLES = {
    'front':       {'azimuth': 0,   'elevation': 0},
    'three_quarter': {'azimuth': 45,  'elevation': 0},
    'side':        {'azimuth': 90,  'elevation': 0},
    'three_quarter_back': {'azimuth': 135, 'elevation': 0},
    'back':        {'azimuth': 180, 'elevation': 0},
}

# For each character × pose × camera_angle:
#   1. Position camera at (azimuth, elevation, fixed_distance)
#   2. Render composite image
#   3. Render segmentation mask (same vertex-group coloring, new camera)
#   4. Extract 2D joint positions (project 3D joints to new camera)
#   5. Compute draw order (per-pixel depth from new camera)
```

**Volume impact:** Currently N characters × M poses = N×M images. With 5 camera angles this becomes N×M×5 images. For the base pipeline (say 200 characters × 10 poses = 2,000), this produces 10,000 images. Render time scales linearly — no model reload, just camera repositioning.

**Output per training example (multi-view extended):**
```
example_001_front/
├── image.png              ← Front view render
├── segmentation.png       ← Front view segmentation
├── draw_order.png         ← Front view depth
├── joints.json            ← Front view 2D joints
└── metadata.json          ← camera_angle: 0, character_id, pose_id

example_001_three_quarter/
├── image.png              ← 3/4 view render (same character, same pose)
├── segmentation.png       ← 3/4 view segmentation
├── draw_order.png
├── joints.json
└── metadata.json          ← camera_angle: 45, character_id, pose_id

example_001_back/
├── ...
```

**Critical:** The `character_id` and `pose_id` fields in metadata.json link all views of the same character in the same pose. This enables multi-view consistency training — the model can learn that the same character's segmentation should produce consistent measurements across views.

### 13.3 New Data Source: VRoid Hub Characters

Every major paper in anime character 3D reconstruction trains on VRoid models:
- **CharacterGen** (SIGGRAPH 2024): Anime3D dataset, 13,746 VRoid characters
- **StdGEN** (CVPR 2025): Anime3D++, extended with semantic part annotations
- **NOVA-3D** (2024): 10,200 VRoid characters, front + back renders
- **PAniC-3D** (CVPR 2023): Original VRoid dataset collection

VRoid characters are VRM format (glTF-based) with standardized skeleton, blend shapes, and material setup. They represent the full diversity of anime character design — every body type, outfit, hairstyle, and proportion variant that the community creates.

**Why this matters for Strata:** Mixamo characters are realistic/western-style 3D. Live2D models are 2D illustrated but front-facing only. VRoid models fill the gap: anime-style 3D characters that can be rendered from any angle. This is the largest available source of multi-view anime character training data.

**Acquisition:**

| Source | Volume | License | Notes |
|--------|--------|---------|-------|
| VRoid Hub (hub.vroid.com) | 100,000+ models | Per-model (check terms) | Largest VRM repository. Many models allow derivative use. |
| PAniC-3D collection instructions | ~14,500 curated | Research use | CharacterGen/StdGEN teams document their scraping methodology |
| VRoid Studio sample models | ~20 | Free | Official samples from pixiv |

**Target:** 2,000–5,000 VRoid models. Prioritize diversity of body type and outfit over raw volume. The PAniC-3D team's filtering removed non-humanoid models — follow their methodology.

**Render pipeline:**

```python
# vroid_renderer.py — render VRM models in Blender
# VRM is glTF-based, Blender imports natively with VRM addon

def render_vroid_character(vrm_path, output_dir):
    # 1. Import VRM into Blender
    # 2. Set A-pose (arms at 45° down) — VRoid default is often T-pose
    # 3. For each camera angle in CAMERA_ANGLES:
    #    a. Position camera
    #    b. Render composite image (512×512, transparent background)
    #    c. Render segmentation mask using material/vertex groups
    #    d. Extract 2D joint positions from armature
    #    e. Compute draw order from Z-buffer
    # 4. Extract ground truth body measurements from mesh bounding boxes
    #    (width per part from front camera, depth per part from side camera)
```

**Semantic part annotations:** VRoid models have standardized material slots (Body, Hair, Face, Outfit_Upper, Outfit_Lower, etc.) that map partially to Strata's 22-label taxonomy. StdGEN's Anime3D++ adds finer annotations. If we can access those annotations, they provide free ground truth.

**VRoid → Strata label mapping:**

```python
VROID_MATERIAL_TO_STRATA = {
    'Body':           'torso',       # Needs per-vertex refinement for limbs
    'Face':           'head',
    'Hair':           'head',        # Or 'hair_back' if Option C adopted
    'EyeWhite':       'head',
    'EyeIris':        'head',
    'Eyebrow':        'head',
    'Outfit_Upper':   'torso',       # Clothing follows body part underneath
    'Outfit_Lower':   'hips',
    'Shoe':           'foot_l',      # Need L/R disambiguation from vertex position
}
# VRoid standardized skeleton provides exact bone assignments for per-vertex
# label refinement — each vertex is weighted to bones, bones map to labels
```

The bone-to-label mapping is more reliable than material mapping. Since VRoid models have a standardized humanoid skeleton, the existing `bone_mapper.py` from the Mixamo pipeline should work with minimal adaptation.

### 13.4 Multi-View Consistency Pairs

The 3D mesh pipeline needs to validate that segmentation is consistent across views of the same character. Training data for this:

**Format:** Paired renders linked by character_id + pose_id (as in 13.2).

**Consistency ground truth:** For each character×pose, extract per-label measurements from each camera angle:

```python
# measurement_extractor.py
def extract_measurements(segmentation_mask, camera_angle, mesh_data):
    measurements = {}
    for label_id, label_name in STRATA_LABELS.items():
        pixels = (segmentation_mask == label_id)
        if pixels.any():
            bbox = bounding_box(pixels)
            measurements[label_name] = {
                'apparent_width': bbox.width,
                'apparent_height': bbox.height,
                'true_width': mesh_bounding_box(mesh_data, label_name).x_extent,
                'true_depth': mesh_bounding_box(mesh_data, label_name).y_extent,
                'true_height': mesh_bounding_box(mesh_data, label_name).z_extent,
                'camera_angle': camera_angle,
            }
    return measurements
```

This produces ground truth for the measurement extraction system described in the 3D Mesh PRD section 6. Given a segmentation mask at a known camera angle, what are the true 3D dimensions? The training data provides thousands of (apparent_measurement, camera_angle) → true_measurement pairs.

### 13.5 Contour Line Training Data

The 3D Mesh PRD specifies DrawingSpinUp-style contour removal before texture projection. Training this requires (with_contours, without_contours) image pairs.

**From Blender renders:**

```python
# contour_renderer.py
def render_contour_pairs(character, pose, camera_angle):
    # 1. Render normal composite (cel-shaded or toon-shaded)
    #    with Freestyle line rendering enabled → with_contours.png
    # 2. Render same scene with Freestyle disabled → without_contours.png
    # 3. Difference → contour_mask.png (binary mask of line pixels)
```

Blender's Freestyle renderer produces configurable edge lines — silhouette edges, crease edges, material boundaries. This is exactly the type of contour line that appears in hand-drawn character art.

**Augmentation for hand-drawn styles:**

```python
# contour_augmenter.py
def augment_contours(without_contours, contour_mask):
    # Vary contour appearance to simulate different art styles:
    styles = [
        {'width': 1, 'color': (0, 0, 0), 'opacity': 1.0},      # Thin black line
        {'width': 2, 'color': (0, 0, 0), 'opacity': 1.0},      # Medium black line
        {'width': 3, 'color': (0.2, 0.1, 0.05), 'opacity': 0.9}, # Thick brown line
        {'width': 1, 'color': 'per_region', 'opacity': 0.8},     # Colored line (varies by body part)
        {'width': 2, 'jitter': True, 'opacity': 1.0},            # Hand-drawn wobbly line
    ]
    # For each style, dilate contour_mask to target width,
    # composite contour over without_contours image
    # Result: same character with different line art styles
```

**Volume:** Each character×pose×angle produces 5 contour style variants = 5× multiplier on contour training data.

### 13.6 Texture Projection Ground Truth

For training the neural inpainting model that fills unobserved texture regions (3D Mesh PRD section 7.4), we need (partial_texture, complete_texture) training pairs.

**From multi-view renders:**

```python
# texture_projection_trainer.py
def generate_texture_pairs(character, pose):
    # 1. Render character from ALL angles (every 15°, 24 views)
    #    → complete_texture.png (UV map with full coverage)

    # 2. Render from ONLY front + 3/4 + back (3 views)
    #    → partial_texture.png (UV map with gaps at ~60-135°)

    # 3. The gap between partial and complete = what the inpainting
    #    model must learn to fill
    #    → inpainting_mask.png (binary: 1 where texture is missing)

    # Training pair: (partial_texture, inpainting_mask) → complete_texture
```

This is a future (Phase 4) addition — the neural inpainting model is P1 priority in the 3D Mesh PRD. But the multi-view renders from 13.2 already produce the raw materials. The texture projection step is incremental work on top.

### 13.7 Body Measurement Ground Truth

The 3D mesh pipeline extracts body part dimensions from segmented paintings to drive template deformation. Training and validating this measurement extraction requires ground truth.

**From 3D models (Mixamo + VRoid):**

For each character, compute true body part dimensions directly from the mesh:

```python
# measurement_ground_truth.py
def extract_mesh_measurements(mesh, skeleton):
    measurements = {}
    for bone_name, label_name in BONE_TO_LABEL.items():
        # Get all vertices weighted to this bone
        verts = get_bone_vertices(mesh, bone_name)
        if len(verts) > 0:
            bbox = axis_aligned_bounding_box(verts)
            measurements[label_name] = {
                'width': bbox.x_extent,   # Left-right
                'depth': bbox.y_extent,   # Front-back
                'height': bbox.z_extent,  # Up-down
                'center': bbox.center.tolist(),
            }
    return measurements
```

**Output:** `measurements.json` alongside each training example, providing ground truth body part dimensions in world units. Compare against measurements extracted from 2D segmentation masks to validate the extraction pipeline.

**Proportion archetypes:** Cluster the measurement profiles across all characters to identify the natural template archetypes. This directly informs the template mesh library design (3D Mesh PRD section 5). If the data shows 6 natural clusters, build 6 templates. If it shows 12, build 12.

### 13.8 Repository Structure (3D Mesh + Pre-Processed Dataset Additions)

All 3D mesh pipeline scripts and pre-processed dataset ingestion adapters are now reflected in the consolidated repository structure in **Section 9** (v2, updated Feb 27 2026). Key additions for this section:

- **`pipeline/`** gained: `multi_angle_renderer.py`, `measurement_extractor.py`, `measurement_ground_truth.py`, `contour_renderer.py`, `contour_augmenter.py`
- **`ingest/`** (new directory): `nova_human_adapter.py`, `stdgen_pipeline_ext.py`, `stdgen_semantic_mapper.py`, `animerun_contour_adapter.py`, `unirig_skeleton_mapper.py`, `linkto_adapter.py`, `vroid_importer.py`, `vroid_mapper.py`, `download_datasets.sh`
- **`data/vroid/`**: VRM source files from PAniC-3D downloader
- **`data/preprocessed/`** (new directory): NOVA-Human, StdGEN, UniRig, AnimeRun, LinkTo-Anime, FBAnimeHQ, CharacterGen downloads
- **`segmentation/labels/`** gained: `vroid_mappings.csv`, `stdgen_to_strata.json`
- **`mesh/`** (new directory): `proportion_clusterer.py`, `texture_projection_trainer.py`, `measurement_profiles.json`
- **`output/segmentation/`** gained: `vroid/`, `vroid_preprocessed/`, `contour_pairs/`

### 13.9 Implementation Priority (3D Mesh Data — Revised with Pre-Processed Datasets)

**Phase 1 — Downloads + Pipeline Setup (week 1)**

| Task | Effort | Impact | Why Now |
|------|--------|--------|---------|
| download_datasets.sh — download NOVA-Human, StdGEN, UniRig, AnimeRun | 1–2 days (bandwidth) | Critical | ~355K pre-rendered images available immediately. Highest ROI action. |
| nova_human_adapter.py — convert NOVA-Human renders to Strata format | 2 days | High | 163K multi-view anime images ready for training after conversion. |
| multi_angle_renderer.py — add camera angles to Mixamo pipeline | 2 days | Critical | Segmentation model WILL fail on 3/4 views without this. |
| measurement_ground_truth.py — extract true dimensions from meshes | 2 days | High | Informs template library design. Free data from existing meshes. |

**Phase 2 — StdGEN Integration + VRoid Pipeline (weeks 2–4)**

| Task | Effort | Impact | Why Now |
|------|--------|--------|---------|
| stdgen_semantic_mapper.py — map 4-class → Strata 22-class taxonomy | 2 days | High | Unlocks 10,811 characters' semantic annotations for training. |
| stdgen_pipeline_ext.py — extend StdGEN's Blender script with Strata outputs | 3–5 days | High | Adds joints, draw order, measurements, contours, 45° angle to proven pipeline. |
| Run extended StdGEN pipeline on 10,811 VRoid characters | 1–2 weeks (compute) | Critical | Core multi-view anime training data with full Strata annotations. |
| animerun_contour_adapter.py — convert AnimeRun pairs to training format | 1 day | Medium | 8K contour pairs ready immediately, supplements generated contours. |
| unirig_skeleton_mapper.py — map UniRig bones to Strata hierarchy | 2 days | Medium | Validation data for rigging pipeline. |
| proportion_clusterer.py — find natural template archetypes | 1 day | High | Determines template library design from 10K+ measurement profiles. |

**Phase 3 — Measurement Validation + Gap Filling (weeks 4–8)**

| Task | Effort | Impact | Why Now |
|------|--------|--------|---------|
| measurement_extractor.py — 2D→3D extraction pipeline | 3 days | High | Core of template deformation. Validate against mesh ground truth + UniRig. |
| Cross-view consistency validation | 2 days | Medium | Ensure segmentation + measurement extraction is robust across angles. |
| contour_renderer.py + augmenter | 3 days | Medium | Supplement AnimeRun with anime-specific Freestyle contours. |
| linkto_adapter.py — extract skeleton + flow from LinkTo-Anime | 2 days | Medium | 29K frames with Mixamo skeleton data for animation training. |

**Phase 4 — Future (v2.0+)**

| Task | Effort | Impact | Why Now |
|------|--------|--------|---------|
| texture_projection_trainer.py | 1 week | High (future) | Neural inpainting training data. Builds on multi-view renders. |

### 13.10 Volume Summary (Revised with Pre-Processed Datasets)

| Data Source | Views | Volume | Pre-processed? | Style |
|-------------|-------|--------|----------------|-------|
| NOVA-Human pre-renders | 4 ortho + 16 random | ~204,000 images | ✅ Download | Anime 3D |
| StdGEN semantic annotations | Multi-view | 10,811 characters annotated | ✅ Download (need re-render) | Anime 3D |
| AnimeRun contour pairs | Various | ~8,000 pairs | ✅ Download | 3D movie style |
| LinkTo-Anime frames | Multiple angles | ~29,270 frames | ✅ Download | Anime 3D |
| UniRig Rig-XL meshes | N/A (meshes) | 14,000 rigged models | ✅ Download | Diverse 3D |
| FBAnimeHQ | Front-ish | 112,806 images | ✅ Download | Anime 2D |
| Mixamo renders | 5 angles | ~10,000 images | ❌ Render | Realistic/western 3D |
| VRoid supplementary renders | 45° + poses | ~50,000 images | ❌ Render | Anime 3D |
| Live2D composites | Front only | ~1,600 images | ❌ Render | 2D illustration |
| PSD extraction | Front only | ~50–100 images | ❌ Manual | Mixed hand-drawn |
| Generated contour pairs | 5 angles × 5 styles | ~50,000 pairs | ❌ Generate | Augmented |
| **TOTAL** | | **~465,000+** | **~355K pre-processed** | **5+ style families** |

**Comparison to original plan:** The original v1 estimate was ~22,000–37,000 segmentation training images from self-rendered data only. With pre-processed datasets, the pipeline now has access to ~465,000+ images, of which ~355,000 are already rendered and downloadable. The self-rendering workload dropped from "everything" to "fill in the gaps" — primarily the 45° camera angle, Strata-specific annotations, and Mixamo realistic-style renders.

---

*The strongest training data pipeline isn't the one with the most images — it's the one with the most diverse images. A model that has only seen 3D renders will fail on the first watercolor painting it encounters. A model that has seen 3D renders, anime illustrations, pixel art, and painted characters will generalize to art styles it's never seen.*

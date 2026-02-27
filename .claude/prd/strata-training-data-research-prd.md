# Strata Training Data — Research-Informed Improvements PRD

**Version:** 1.0
**Date:** February 27, 2026
**Status:** Planned
**Context:** Competitive intelligence from "See-through" (arxiv, Feb 2026), Spiritus (ACM UIST 2025), Qwen-Image-Layered (Dec 2025), and ASMR (Mar 2025) papers revealed techniques that can strengthen Strata's training data pipeline.
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

Incorporating all additions:

```
strata-training-data/
├── pipeline/                           ← Existing Blender/Python pipeline
│   ├── bone_mapper.py
│   ├── config.py
│   ├── exporter.py
│   ├── generate_dataset.py
│   ├── importer.py
│   ├── joint_extractor.py
│   ├── pose_applicator.py
│   ├── renderer.py
│   ├── weight_extractor.py
│   └── draw_order_extractor.py         ← NEW — compute per-pixel depth from render
│
├── data/
│   ├── fbx/                            ← ⛔ .gitignore
│   ├── poses/                          ← ⛔ .gitignore
│   ├── mocap/                          ← ⛔ .gitignore
│   ├── sprites/                        ← ⛔ .gitignore
│   ├── live2d/                         ← ⛔ .gitignore — NEW
│   │   └── README.md                   ← Sources: Booth, DeviantArt, GitHub
│   └── psd/                            ← ⛔ .gitignore — NEW
│       └── README.md
│
├── segmentation/
│   ├── scripts/
│   │   ├── live2d_renderer.py          ← NEW — render Live2D models to training data
│   │   ├── live2d_mapper.py            ← NEW — fragment name → Strata label mapping
│   │   ├── live2d_review_ui.py         ← NEW — manual mapping review tool
│   │   ├── psd_extractor.py            ← NEW — PSD layer extraction
│   │   └── occlusion_renderer.py       ← NEW (future) — isolated region rendering
│   └── labels/
│       └── live2d_mappings.csv         ← NEW — model_id, fragment, label, confirmed
│
├── animation/
│   ├── scripts/
│   │   ├── bvh_parser.py               ← NEW — parse BVH into bone data
│   │   ├── bvh_to_strata.py            ← NEW — retarget BVH → Strata skeleton
│   │   ├── proportion_normalizer.py    ← NEW — scale mocap to character proportions
│   │   ├── blueprint_exporter.py       ← NEW — export as Strata blueprint JSON
│   │   ├── degrade_animation.py        ← NEW — synthetic degradation for AI training
│   │   ├── label_actions.py
│   │   └── extract_timing.py
│   ├── labels/
│   │   └── cmu_action_labels.csv
│   ├── breakdowns/
│   └── timing-norms/
│
├── output/                             ← ⛔ .gitignore
│   ├── segmentation/
│   └── animation/
│
├── docs/
│   ├── data-sources.md
│   ├── labeling-guide.md
│   ├── taxonomy-comparison.md          ← NEW — Strata 21 vs See-through 19 analysis
│   └── research/
│       └── diff3f_exploration.ipynb    ← NEW — Diff3F feature evaluation
│
└── .gitignore
```

---

## 10. Implementation Priority

### Phase 1 — Immediate (weeks 1–3, parallel with base pipeline)

| Task | Effort | Impact | Why Now |
|------|--------|--------|---------|
| Add draw_order_extractor.py to Blender pipeline | 2 days | High | Free data from existing renders. Must be in place before training starts. |
| Taxonomy comparison document | 1 day | Medium | Informs whether to add hair_back before v1 model training. |
| BVH parser + retargeting scripts | 1 week | High | Needed for CMU mocap labeling (already planned). |
| CMU action labeling kickoff | Ongoing | High | Already planned. Add `strata_compatible` column. |

### Phase 2 — Near-term (weeks 4–8)

| Task | Effort | Impact | Why |
|------|--------|--------|-----|
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
| Training data style diversity | 1 source (Mixamo 3D renders) | 3+ sources (Mixamo + Live2D + PSD) | Count distinct art styles in training set |
| Segmentation accuracy on hand-drawn art | Unknown (model not yet trained) | IoU ≥ 0.85 on held-out illustrated characters | Test set with 100 hand-drawn character images |
| Draw order prediction | Not available | Accuracy ≥ 90% on layer ordering | Compare predicted vs. ground-truth draw order |
| BVH retargeting coverage | 0 clips | 500+ labeled, retargeted clips | Count in cmu_action_labels.csv |
| Synthetic degradation pairs | 0 | 3,500+ pairs (500 clips × 7 degradation types) | Count in animation/output/ |

---

## 12. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Live2D SDK licensing restricts use for ML training | Medium | High | Use only the rendering output (images), not the SDK in the product. Check Cubism SDK license terms. Alternative: write a minimal .moc3 parser that reads ArtMesh geometry without the SDK. |
| Fragment naming too inconsistent for automated mapping | High | Medium | Accept 30% manual correction rate. The review UI handles this. Budget the annotation time. |
| Diff3F features don't help weight painting | Medium | Low | It's a research exploration. No production dependency. |
| Occluded region inpainting is too hard to ship | Medium | Medium | It's a v2.0+ feature. The data collection is cheap even if the model proves difficult. |

---

*The strongest training data pipeline isn't the one with the most images — it's the one with the most diverse images. A model that has only seen 3D renders will fail on the first watercolor painting it encounters. A model that has seen 3D renders, anime illustrations, pixel art, and painted characters will generalize to art styles it's never seen.*

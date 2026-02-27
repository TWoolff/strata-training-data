# VRoid Hub Characters

Anime-style 3D characters in VRM format for multi-angle segmentation rendering.

## Why VRoid

Mixamo characters are realistic/western-style 3D. Live2D models are 2D illustrated but front-facing only. VRoid models fill the gap: anime-style 3D characters renderable from any angle with a standardized humanoid skeleton. Every major paper in anime character 3D reconstruction trains on VRoid models.

## Sources

| Source | Volume | Notes |
|--------|--------|-------|
| **VRoid Hub** (hub.vroid.com) | 100,000+ models | Largest VRM repository. Per-model license terms. |
| **PAniC-3D methodology** | ~14,500 curated | Documented scraping + filtering pipeline |
| **VRoid Studio samples** | ~20 | Official samples from pixiv |

**Target volume:** 2,000–5,000 models. Prioritize body type and outfit diversity over raw count.

## PAniC-3D Collection Methodology

The PAniC-3D team (CVPR 2023) established the standard methodology for large-scale VRoid collection, later adopted by CharacterGen (SIGGRAPH 2024) and StdGEN (CVPR 2025):

1. **Crawl VRoid Hub** — Scrape model listings from hub.vroid.com using their public API or web scraping. Models are VRM format (glTF-based), downloadable when the author permits.
2. **Filter by license** — Only collect models that explicitly permit derivative use and redistribution for ML training. Reject models with restrictive terms.
3. **Remove non-humanoid** — Filter out mascots, animals, mecha, and abstract models. Keep bipedal humanoid characters only.
4. **Deduplicate** — Remove near-identical models (e.g., minor recolors of the same base). The PAniC-3D team reduced 14,500 → ~11,000 unique characters.
5. **Quality filter** — Remove broken models (missing textures, corrupted geometry, extreme polygon counts).

### Reference Papers

- **PAniC-3D** (Fang et al., CVPR 2023) — Original VRoid dataset collection, 14,500 models
- **CharacterGen** (Peng et al., SIGGRAPH 2024) — Anime3D dataset, 13,746 VRoid characters
- **StdGEN** (CVPR 2025) — Anime3D++ with semantic part annotations
- **NOVA-3D** (2024) — 10,200 VRoid characters, front + back renders

## License Requirements

- Only collect models whose VRoid Hub terms permit **derivative use** for ML training
- Check each model's individual license on its VRoid Hub page
- **Accept:** Models marked as allowing modification and redistribution
- **Reject:** Models with "personal use only", "no modification", or "no redistribution" restrictions
- Log the license status of every downloaded model in `labels/`

## Filtering Criteria

1. **Humanoid bipeds only** — No animals, mecha, fantasy creatures with non-standard anatomy
2. **Body type diversity** — Actively sample across body proportions (chibi, realistic, stylized)
3. **Outfit variety** — Mix of casual, fantasy, school uniform, armor, etc.
4. **Quality threshold** — Must have complete textures, reasonable poly count (<500k faces), intact skeleton
5. **No NSFW** — Skip explicit content models

## Download Instructions

### VRoid Hub API

VRoid Hub provides a REST API for browsing and downloading models:

```
GET https://hub.vroid.com/api/character_models
```

Models are VRM files (.vrm), which are glTF 2.0 containers with humanoid skeleton extensions.

### Blender Import

VRM files import into Blender via the [VRM Add-on for Blender](https://vrm-addon-for-blender.info/):

```bash
# Install the add-on in Blender preferences, then:
blender --background --python -c "
import bpy
bpy.ops.import_scene.vrm(filepath='model.vrm')
"
```

### Tooling Suggestions

- **UniVRM** (Unity) — Reference VRM importer, useful for batch validation
- **VRM Add-on for Blender** — Required for pipeline integration
- **glTF-Transform** (Node.js) — CLI tool for inspecting/modifying glTF/VRM files

## VRoid Skeleton Mapping

VRoid models use the VRM humanoid bone specification, which maps directly to Strata's 19-region skeleton via `bone_mapper.py`. The standardized naming means near-100% automatic mapping:

| VRM Bone | Strata Region |
|----------|--------------|
| head | head |
| neck | neck |
| chest / upperChest | chest |
| spine | spine |
| hips | hips |
| leftUpperArm | upper_arm_l |
| leftLowerArm | lower_arm_l |
| leftHand | hand_l |
| rightUpperArm | upper_arm_r |
| rightLowerArm | lower_arm_r |
| rightHand | hand_r |
| leftUpperLeg | upper_leg_l |
| leftLowerLeg | lower_leg_l |
| leftFoot | foot_l |
| rightUpperLeg | upper_leg_r |
| rightLowerLeg | lower_leg_r |
| rightFoot | foot_r |
| leftShoulder | shoulder_l |
| rightShoulder | shoulder_r |

## Expected Structure

Place .vrm files directly in this directory:

```
data/vroid/
├── README.md
├── labels/              ← License metadata, filtering logs (tracked in git)
│   └── .gitkeep
├── vroid_001.vrm
├── vroid_002.vrm
└── ...
```

## Render Pipeline

VRoid models follow the same pipeline as FBX characters:

1. Import VRM into Blender (via VRM add-on)
2. Map bones to Strata regions (automatic via standardized VRM skeleton)
3. Apply poses (T-pose → A-pose, plus animation clips)
4. Render from 5 camera angles: front (0°), three-quarter (45°), side (90°), three-quarter-back (135°), back (180°)
5. Generate segmentation masks, draw order maps, joint positions
6. Apply style augmentation (flat, cel, pixel art, painterly, sketch, unlit)

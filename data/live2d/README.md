# Live2D Community Models

Pre-decomposed 2D illustrated characters in .moc3 format for segmentation ground truth and style diversity.

## Why Live2D

Mixamo characters are realistic/western-style 3D renders. Live2D models are hand-drawn 2D illustrations spanning anime, chibi, stylized, and fantasy art styles. Each model is pre-decomposed into layered ArtMesh fragments (hair_front, eye_left, torso, arm_l, etc.) with pixel-precise boundaries — providing near-free segmentation ground truth without manual annotation. The See-through paper (Tsubota et al.) built a dataset of 9,102 annotated characters using this approach.

## Sources

| Source | Volume | Notes |
|--------|--------|-------|
| **Booth.pm** (free section) | ~200–500 models | Largest Live2D marketplace. Many free models with redistribution rights for derivative works. |
| **DeviantArt** | ~50–100 models | Smaller volume but wider style variety. Search "Live2D model free download". |
| **GitHub/open-source** | ~30–50 models | MIT/CC licensed VTuber models and sample projects. Search "live2d model" or "vtube studio model". |
| **Live2D sample models** | ~10–20 | Official samples from Live2D Inc. Free for non-commercial use. |

**Target volume:** 300–500 models. This supplements Mixamo 3D renders with illustrated art styles, ensuring the segmentation model generalizes beyond CG characters.

### Booth.pm

The primary source. Booth.pm is the largest marketplace for Live2D models, with many offered for free.

1. Browse https://booth.pm/ and search for "Live2D モデル" or "Live2D model"
2. Filter by price: free (0 JPY)
3. Check each model's terms — only download models that permit derivative use and redistribution
4. Models typically include a .moc3 file + texture atlas (.png) + optional .model3.json

### DeviantArt

Secondary source for style variety beyond the anime-heavy Booth.pm selection.

1. Search DeviantArt for "Live2D model download" or "Live2D free"
2. Look for downloads with explicit CC or permissive licenses in the description
3. Style range: anime, semi-realistic, chibi, fantasy, western illustration

### GitHub/Open-Source

Small but reliably licensed source.

1. Search GitHub for repositories containing `.moc3` files
2. Look for MIT, Apache 2.0, or CC-BY/CC-BY-SA licensed projects
3. Common sources: VTuber starter kits, Live2D tutorials, sample projects
4. Check repository LICENSE file — not just the README claims

### Live2D Official Samples

Baseline reference models from Live2D Inc.

1. Available from the Live2D Cubism Editor download page and documentation
2. Free for non-commercial use — verify terms permit ML training use
3. Small volume (~10–20) but high quality and well-structured

## License Requirements

- **Accept:** Models permitting derivative use and redistribution for ML training
- **Accept:** CC0, CC-BY, CC-BY-SA, MIT, Apache 2.0
- **Reject:** "Personal use only", "no modification", "no redistribution", CC-NC (non-commercial)
- **Verify:** Booth.pm models — check each model's individual terms page, not just the storefront listing
- Log the license status of every downloaded model in the CSV manifest (see below)

## CSV Manifest

Track per-model licenses in `labels/`. One row per model:

```
model_id,source,url,license,license_verified,fragment_count,notes
live2d_001,booth,https://booth.pm/items/XXXXX,CC-BY-SA,true,24,""
live2d_002,deviantart,https://deviantart.com/...,CC-BY,true,18,"chibi style"
live2d_003,github,https://github.com/...,MIT,true,31,"VTuber starter kit"
```

Fields:
- `model_id` — Sequential ID with `live2d_` prefix
- `source` — One of: booth, deviantart, github, live2d_official
- `url` — Original download URL
- `license` — SPDX identifier or short description
- `license_verified` — Whether license was manually confirmed
- `fragment_count` — Number of ArtMesh fragments in the model
- `notes` — Style tags, quality notes, etc.

## Download and Organization

Place .moc3 model files and their texture atlases in this directory:

```
data/live2d/
├── README.md
├── labels/              ← License metadata, fragment mappings (tracked in git)
│   └── .gitkeep
├── live2d_001/
│   ├── model.moc3
│   ├── model.model3.json
│   └── textures/
│       └── texture_00.png
├── live2d_002/
│   ├── model.moc3
│   └── ...
└── ...
```

Each model gets its own subdirectory named by model ID. The .moc3 file, texture atlas, and any .model3.json config go inside.

## Fragment-to-Label Mapping

Live2D fragment names are artist-defined and inconsistent (e.g., "arm_L", "left_arm", "腕左"). The `live2d_mapper.py` pipeline module handles automated keyword-based mapping from fragment names to Strata's 19-region labels, with ~20–30% requiring manual correction via review UI.

Confirmed mappings are stored in `labels/live2d_mappings.csv`.

## Render Pipeline

Live2D models follow a front-facing-only pipeline (no multi-angle rendering):

1. Load .moc3 model and composite the default pose
2. Map ArtMesh fragments to Strata regions via `live2d_mapper.py`
3. Generate segmentation mask from fragment boundaries
4. Extract draw order from ArtMesh render order indices (normalize to [0, 1])
5. Apply augmentation: horizontal flip, rotation (±5°), scale (±10°), color jitter
6. Output: composite.png + segmentation.png + draw_order.png + metadata.json

Expected output: ~1,600 training images from 400 models × 4 augmentations.

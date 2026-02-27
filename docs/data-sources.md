# Data Sources

Master list of all data sources used in the Strata training data pipeline.

## Segmentation Pipeline (Characters)

| Source | URL | License | Est. Count |
|--------|-----|---------|------------|
| Mixamo | https://www.mixamo.com/ | Free for use | ~100 |
| Sketchfab | https://sketchfab.com/ | CC0/CC-BY | ~100–200 |
| Quaternius | https://quaternius.com/ | CC0 | ~50 |
| Kenney | https://kenney.nl/ | CC0 | ~50 |
| Blender community | Various | CC | ~30–50 |

## Animation Intelligence (MoCap)

| Source | URL | License | Clips |
|--------|-----|---------|-------|
| CMU MoCap | https://github.com/una-dinosauria/cmu-mocap | Free | 2,548 |
| SFU MoCap | https://mocap.cs.sfu.ca/ | Academic | Varies |

## Live2D Community Models

Pre-decomposed 2D illustrated characters. ArtMesh fragments provide near-free segmentation ground truth. Reference: See-through paper (Tsubota et al.).

| Source | URL | License | Est. Count |
|--------|-----|---------|------------|
| Booth.pm (free) | https://booth.pm/ | Varies (check per model) | ~200–500 |
| DeviantArt | https://www.deviantart.com/ | Varies | ~50–100 |
| GitHub/open-source | Various | MIT/CC | ~30–50 |
| Live2D official samples | https://www.live2d.com/ | Free for non-commercial | ~10–20 |

**Target:** 300–500 models. Only models permitting derivative use for ML training.

## VRoid Hub Characters

Anime-style 3D characters in VRM format, renderable from any angle with a standardized humanoid skeleton. References: PAniC-3D (CVPR 2023), CharacterGen (SIGGRAPH 2024), StdGEN (CVPR 2025), NOVA-3D (2024).

| Source | URL | License | Est. Count |
|--------|-----|---------|------------|
| VRoid Hub | https://hub.vroid.com/ | Per-model terms | 2,000–5,000 |
| VRoid Studio samples | https://vroid.com/ | Free | ~20 |

**Target:** 2,000–5,000 models. Filter for humanoid bipeds, body type diversity, and derivative-use licenses.

## PSD Files (Opportunistic)

Layered Photoshop documents where body parts are on separate layers. Only body-part-separated PSDs are useful — most PSDs are organized by rendering concern (lineart/color/shading). Reference: Qwen-Image-Layered (Dec 2025).

| Source | URL | License | Est. Count |
|--------|-----|---------|------------|
| OpenGameArt | https://opengameart.org/ | CC0/CC-BY | ~20–50 |
| itch.io asset packs | https://itch.io/game-assets | Varies | ~20–40 |
| Patreon art packs | Various | Check per artist | ~10–20 |

**Target:** 50–100 usable PSDs, collected opportunistically over months.

## Sprite Sheets

| Source | URL | License |
|--------|-----|---------|
| OpenGameArt | https://opengameart.org/ | Various (CC0/CC-BY) |
| itch.io | https://itch.io/game-assets/free/tag-sprites | Various |

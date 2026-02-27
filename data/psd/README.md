# PSD Files (Layered Photoshop Documents)

Opportunistic collection of layered character PSDs where body parts are on separate layers — providing natural segmentation ground truth from real artist workflows.

## Why PSDs

Most training data comes from 3D renders (Mixamo, VRoid) or pre-decomposed 2D models (Live2D). Real artist-created PSDs add a third category: hand-drawn characters with layer structure that reflects how human artists think about body part separation. Qwen-Image-Layered demonstrated that PSD layer structure provides usable ground truth for image decomposition. Even a small collection of body-part-separated PSDs is valuable for style diversity.

## Important: Layer Organization Matters

Most PSDs are organized by **rendering concern** (lineart, color, shading, highlights) — NOT by body part. Only PSDs with **body-part-separated layers** are useful for segmentation training.

**Useful:**
- Layers named "head", "torso", "left_arm", "hair", "legs", etc.
- Character rigs with body parts on separate layers for animation
- Paper doll / dress-up style files with interchangeable parts

**Not useful:**
- Layers named "lineart", "flat color", "shadows", "highlights"
- Single-layer finished illustrations
- Layers organized by rendering pass rather than anatomy

## Sources

| Source | URL | License | Notes |
|--------|-----|---------|-------|
| **OpenGameArt** | https://opengameart.org/ | CC0/CC-BY | Search for layered character sprites. Game assets more likely to have body-part layers. |
| **itch.io asset packs** | https://itch.io/game-assets | Varies | Character sprite packs sometimes include PSD sources with body-part layers. |
| **Patreon art packs** | Various | Check per artist | Some character artists release layered PSDs. Verify license permits ML training use. |

**Target volume:** 50–100 usable PSDs, found opportunistically over months. Each one is a real artist's real work — valuable for style diversity even in small quantities.

## License Requirements

- **Accept:** CC0, CC-BY, CC-BY-SA, or explicit permission for derivative use
- **Reject:** "Personal use only", CC-NC, no redistribution
- **Verify:** Patreon releases — check the artist's terms carefully
- Log the license of every PSD in a manifest CSV

## File Organization

```
data/psd/
├── README.md
├── .gitkeep
├── psd_001.psd
├── psd_002.psd
└── ...
```

Name files with sequential IDs: `psd_001.psd`, `psd_002.psd`, etc.

## Processing Pipeline

Uses `psd-tools` Python library for layer extraction:

1. Open PSD and enumerate layers
2. Map layer names to Strata's 19-region labels (keyword-based, similar to Live2D fragment mapping)
3. Render each body-part layer independently to produce segmentation ground truth
4. Composite the full character image
5. Output: composite.png + segmentation.png + metadata.json

Per-PSD processing is fast (~seconds). The bottleneck is finding PSDs with body-part-separated layers.

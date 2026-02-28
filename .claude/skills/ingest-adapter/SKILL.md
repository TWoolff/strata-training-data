---
name: ingest-adapter
description: This skill should be used when the task involves the ingest/ directory, converting external datasets (NOVA-Human, StdGEN, AnimeRun, UniRig, LinkTo-Anime, FBAnimeHQ) to Strata format, or creating new dataset adapters.
user-invokable: false
---

# Ingest Adapter

Expert knowledge for working with the `ingest/` directory — adapter scripts that convert external pre-processed research datasets into the Strata training data format.

## Overview

The `ingest/` directory contains adapters for converting external datasets into Strata's standard output format. These datasets come from research teams who have already processed raw data (rendered images, extracted segmentation, annotated characters) — the adapters map their annotations and formats to Strata's 20-region taxonomy.

Reference: `docs/preprocessed-datasets.md` for dataset details and download instructions.

## Current Adapters

| Adapter | Source Dataset | What It Does |
|---------|---------------|--------------|
| `nova_human_adapter.py` | NOVA-Human (~10.2K anime characters from VRoid Hub) | Converts rendered multi-view images + segmentation to Strata format |
| `stdgen_semantic_mapper.py` | StdGEN (CVPR 2025, 10.8K annotated characters) | Maps StdGEN's 4-class taxonomy (hair, face, body, clothes) to Strata's 20-class taxonomy |
| `stdgen_pipeline_ext.py` | StdGEN | Blender extension for re-rendering StdGEN characters with Strata outputs |

## Adapter Patterns

When building or modifying adapters, follow these conventions:

### Pure Python Where Possible
- Mapper modules (semantic mapping, file conversion) should have **no Blender dependency**
- Only use Blender (`bpy`) for rendering extensions that need 3D scene access
- This allows adapters to be tested and run outside Blender

### Standard Imports and Constants
```python
from __future__ import annotations

import logging
from pathlib import Path

from pipeline.config import (
    REGION_NAMES,
    REGION_NAME_TO_ID,
    NUM_REGIONS,
    VRM_BONE_ALIASES,
)

logger = logging.getLogger(__name__)

SOURCE = "nova_human"  # Source identifier for metadata
```

### Dataclass Results
Use dataclasses for structured return types:
```python
from dataclasses import dataclass

@dataclass
class AdapterResult:
    """Result of converting one example from external dataset."""
    image_path: Path
    segmentation_path: Path
    joints: dict | None
    metadata: dict
    warnings: list[str]
```

### Strata Output Format
Each converted example must produce:
```
example_001/
├── image.png           <- Character render (512x512 RGBA PNG)
├── segmentation.png    <- Per-pixel region IDs (8-bit grayscale, values 0-19)
├── draw_order.png      <- Per-pixel depth (grayscale, 0=back 255=front) [if available]
├── joints.json         <- 2D joint positions (19 joints, regions 1-19) [if available]
└── metadata.json       <- Source type, style, pose name, camera angle
```

### Region Mapping
- Strata uses 20 regions: IDs 0-19 (background + 19 body parts)
- External datasets typically have coarser taxonomies that need refinement
- `STDGEN_SEMANTIC_CLASSES` in `config.py` maps StdGEN's 4 classes -> Strata regions
- Classes like "body" and "clothes" require per-vertex bone-weight refinement using `VRM_BONE_ALIASES`
- Direct mappings (e.g., "hair" -> head, "face" -> head) go into `config.py` as constants
- Ambiguous mappings (need bone weights) use `None` and are resolved at runtime

### Error Handling
```python
# Log warnings for unmapped items, don't crash
if source_label not in MAPPING:
    logger.warning("Unmapped label %r in %s, defaulting to background", source_label, filepath)
    return 0  # background
```

### Test Files
Every adapter needs a test in `tests/test_{adapter_name}.py`:
- Test with minimal fixture data (no large file downloads)
- Verify output structure matches Strata format
- Test edge cases: unknown labels, missing files, malformed data
- Test region ID ranges (all values 0-19)

## Planned Adapters

These datasets are documented in `docs/preprocessed-datasets.md` and await adapter implementation:

| Dataset | Size | What It Provides | Adapter Complexity |
|---------|------|-------------------|--------------------|
| AnimeRun | ~8K contour pairs | Anime contour lines + shading | Medium — contour-to-segmentation mapping |
| UniRig | 14K rigged meshes | Pre-rigged 3D models | Large — Blender rendering needed |
| LinkTo-Anime | ~29K frames | Anime video frames with labels | Medium — frame extraction + label mapping |
| FBAnimeHQ | ~113K images | High-quality anime faces | Small — face-only, maps to head region |

## Key Config Constants

From `pipeline/config.py`:
- `STDGEN_SEMANTIC_CLASSES` — StdGEN 4-class -> Strata mapping (hair/face -> head, body/clothes -> bone-weight refinement)
- `VRM_BONE_ALIASES` — VRM humanoid bone name -> region ID (used for bone-weight refinement)
- `REGION_NAMES` — ID -> name mapping (20 entries, 0-19)
- `REGION_NAME_TO_ID` — Reverse lookup: name -> ID
- `NUM_REGIONS` — 20 (including background)
- `NUM_JOINT_REGIONS` — 19 (body regions only)

# Strata Training Data — Repository Structure

**Date:** February 27, 2026  
**Purpose:** Reorganize the flat repo into a structure that separates pipeline code from data, scales across both segmentation and animation training, and keeps Git clean of large binaries.

---

## Current State

Everything lives in the repo root:

```
strata-training-data/
├── bone_mapper.py
├── config.py
├── exporter.py
├── generate_dataset.py
├── importer.py
├── joint_extractor.py
├── pose_applicator.py
├── renderer.py
├── weight_extractor.py
├── .gitignore
├── CLAUDE.md
└── (dotfiles)
```

This is fine for the segmentation pipeline alone, but the repo now serves two purposes — segmentation/rigging AI training data AND animation intelligence training data — with very different data types, sources, and workflows.

---

## Proposed Structure

```
strata-training-data/
│
├── README.md                          ← Repo overview, setup instructions
├── CLAUDE.md
├── requirements.txt
├── .gitignore
│
├── pipeline/                          ← Blender/Python segmentation pipeline (existing code)
│   ├── bone_mapper.py
│   ├── config.py
│   ├── exporter.py
│   ├── generate_dataset.py
│   ├── importer.py
│   ├── joint_extractor.py
│   ├── pose_applicator.py
│   ├── renderer.py
│   └── weight_extractor.py
│
├── data/                              ← ALL raw data lives here (mostly .gitignored)
│   │
│   ├── fbx/                           ← ⛔ .gitignore — Mixamo FBX characters
│   │   └── README.md                  ← Download instructions (Mixamo sources, expected filenames)
│   │
│   ├── poses/                         ← ⛔ .gitignore — FBX pose files from Mixamo
│   │   └── README.md
│   │
│   ├── mocap/                         ← ⛔ .gitignore — CMU BVH, SFU, other mocap
│   │   └── README.md                  ← "Clone github.com/una-dinosauria/cmu-mocap here"
│   │
│   └── sprites/                       ← ⛔ .gitignore — Downloaded sprite sheets
│       └── README.md                  ← Sources: OpenGameArt, itch.io, HuggingFace dataset
│
├── output/                            ← ⛔ .gitignore — Generated renders, masks, datasets
│   ├── segmentation/                  ← Rendered images + segmentation masks
│   └── animation/                     ← Processed mocap, extracted features
│
├── animation/                         ← Animation Intelligence scripts + tracked metadata
│   ├── scripts/
│   │   ├── bvh_parser.py              ← Parse BVH into Strata bone format
│   │   ├── label_actions.py           ← CLI tool for tagging mocap clips by action type
│   │   ├── extract_timing.py          ← Extract frame spacing/velocity from labeled clips
│   │   └── degrade_animation.py       ← Strip principles from good animations (synthetic pairs)
│   │
│   ├── labels/                        ← ✅ Track — action type CSVs, quality annotations
│   │   └── cmu_action_labels.csv      ← filename, action_type, subcategory, quality_notes
│   │
│   ├── breakdowns/                    ← ✅ Track — transcribed YouTube/book analyses
│   │   └── README.md                  ← Format guide for breakdown JSON files
│   │
│   └── timing-norms/                  ← ✅ Track — extracted from Williams/Thomas books
│       └── README.md                  ← "Numbers from The Animator's Survival Kit ch. X"
│
└── docs/                              ← Reference documentation
    ├── data-sources.md                ← Master list of all data sources with URLs + licenses
    └── labeling-guide.md              ← How to annotate action types, quality, principles
```

---

## What Gets Tracked vs. Ignored

| Directory | Tracked in Git | Why |
|-----------|---------------|-----|
| `pipeline/` | ✅ Yes | Your code — small, text, irreplaceable |
| `animation/scripts/` | ✅ Yes | Your code |
| `animation/labels/` | ✅ Yes | Hand-annotated metadata — small CSV/JSON, irreplaceable |
| `animation/breakdowns/` | ✅ Yes | Transcribed analyses — small JSON, irreplaceable |
| `animation/timing-norms/` | ✅ Yes | Extracted reference data — small, irreplaceable |
| `docs/` | ✅ Yes | Documentation |
| `data/fbx/` | ⛔ No | Large binaries, re-downloadable from Mixamo |
| `data/poses/` | ⛔ No | Large binaries, re-downloadable |
| `data/mocap/` | ⛔ No | Large binaries, re-downloadable from CMU/SFU |
| `data/sprites/` | ⛔ No | Large binaries, re-downloadable |
| `output/` | ⛔ No | Generated — reproducible by running pipeline |

---

## .gitignore

```gitignore
# Raw data (large, downloadable from source)
data/fbx/**
data/poses/**
data/mocap/**
data/sprites/**

# Generated output (reproducible)
output/

# Keep README files inside ignored directories
!data/fbx/README.md
!data/poses/README.md
!data/mocap/README.md
!data/sprites/README.md

# Python
__pycache__/
*.pyc
.ruff_cache/
```

---

## Migration Steps

```bash
# 1. Create directories
mkdir -p pipeline data/fbx data/poses data/mocap data/sprites
mkdir -p output/segmentation output/animation
mkdir -p animation/scripts animation/labels animation/breakdowns animation/timing-norms
mkdir -p docs

# 2. Move existing pipeline scripts
git mv bone_mapper.py pipeline/
git mv config.py pipeline/
git mv exporter.py pipeline/
git mv generate_dataset.py pipeline/
git mv importer.py pipeline/
git mv joint_extractor.py pipeline/
git mv renderer.py pipeline/
git mv pose_applicator.py pipeline/
git mv weight_extractor.py pipeline/
git mv joint_extractor.py pipeline/
git mv weight_extractor.py pipeline/

# 3. Move any existing FBX/data files into data/
# (adjust paths based on where you currently store them)

# 4. Update imports in pipeline scripts
# generate_dataset.py probably imports from the others — update to relative imports

# 5. Commit
git add -A
git commit -m "Restructure: separate pipeline code from data and animation training"
```

---

## Import Path Fix

After moving scripts into `pipeline/`, internal imports need updating. Add `pipeline/__init__.py` (empty file) and update any cross-imports:

```python
# Before (flat):
from bone_mapper import BoneMapper

# After (in pipeline/):
from pipeline.bone_mapper import BoneMapper
# or relative:
from .bone_mapper import BoneMapper
```

Alternatively, run scripts from the repo root with `python -m pipeline.generate_dataset` instead of `python pipeline/generate_dataset.py`.

---

## README.md Stubs

Each `data/*/README.md` should contain:

1. What goes in this directory
2. Where to download it (URLs)
3. Expected file structure after download
4. License information

Example for `data/mocap/README.md`:

```markdown
# Motion Capture Data

## CMU Graphics Lab (2,548 clips)
Clone into `cmu/`:
  git clone https://github.com/una-dinosauria/cmu-mocap cmu/
License: Free to use worldwide for any purpose.

## SFU Motion Capture
Download BVH files from https://mocap.cs.sfu.ca/ into `sfu/`

## Mixamo Animations
Downloaded via pipeline scripts into `mixamo/`
```

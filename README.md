# Strata Training Data

Blender-based pipeline that generates labeled training data for Strata's AI models (semantic segmentation, joint prediction, weight prediction, drawn pose estimation).

## Quick Start

```bash
blender --background --python run_pipeline.py -- \
  --input_dir ./data/fbx/ \
  --output_dir ./output/segmentation/ \
  --styles flat \
  --resolution 512
```

Requires Blender 4.0+ (uses bundled Python 3.10+).

## Repository Structure

```
strata-training-data/
├── pipeline/              Blender/Python segmentation pipeline
├── data/                  Raw data (gitignored, see READMEs for download instructions)
│   ├── fbx/               Mixamo/Sketchfab FBX characters
│   ├── poses/             Animation FBX clips
│   ├── mocap/             CMU/SFU BVH motion capture
│   └── sprites/           Sprite sheets for animation training
├── output/                Generated output (gitignored)
│   ├── segmentation/      Rendered images + segmentation masks
│   └── animation/         Processed mocap, extracted features
├── animation/             Animation intelligence scripts + metadata
│   ├── scripts/           BVH parsing, labeling, timing extraction
│   ├── labels/            Action type CSVs, quality annotations
│   ├── breakdowns/        Transcribed animation analyses
│   └── timing-norms/      Frame spacing reference data
└── docs/                  Reference documentation
```

See [CLAUDE.md](CLAUDE.md) for full technical documentation.

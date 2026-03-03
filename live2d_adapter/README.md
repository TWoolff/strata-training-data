# Live2D Adapter

Renders Live2D `.moc3` model directories into labeled training data:
composite images, segmentation masks (22 body region IDs), draw-order maps,
and 2D joint position JSON.

Completely self-contained — no external pipeline dependencies.

## Setup

**1. Python dependencies** (Python 3.10+):
```bash
pip install -r requirements.txt
```

**2. Node.js dependencies** (Node 18+):
```bash
cd tools/live2d_renderer
npm install
cd ../..
```

## Usage

```bash
python3 run_live2d.py \
    --input_dir /path/to/live2d/models \
    --output_dir ./output/live2d
```

Each model subdirectory must contain:
- A `.model3.json` entry point
- A `.moc3` binary file
- Texture atlas PNG(s) referenced by `.model3.json`

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input_dir` | *(required)* | Directory of Live2D model subdirs |
| `--output_dir` | `./output/live2d` | Output directory |
| `--resolution` | `512` | Output image resolution (px) |
| `--no_augmentation` | off | 1 variant per model instead of 4 |
| `--only_new` | off | Skip models already rendered |
| `--max_models` | `0` (all) | Limit models processed |

### Example (test with 5 models, no augmentation):
```bash
python3 run_live2d.py \
    --input_dir /path/to/live2d \
    --output_dir ./output/live2d \
    --max_models 5 \
    --no_augmentation
```

## Output

With augmentation (default), produces **4 variants per model**:
original + horizontal flip + ±5° rotation + ±10% scale.

```
output/live2d/
├── images/
│   └── live2d_{id}_pose_{nn}_flat.png   ← RGBA composite (transparent bg)
├── masks/
│   └── live2d_{id}_pose_{nn}.png        ← 8-bit grayscale, pixel = region ID
├── draw_order/
│   └── live2d_{id}_pose_{nn}.png        ← draw order map (0=back, 255=front)
├── joints/
│   └── live2d_{id}_pose_{nn}.json       ← 2D joint positions (from region centroids)
└── sources/
    └── live2d_{id}.json                 ← model metadata
```

## Region IDs

Segmentation masks use single-channel grayscale where pixel value = region ID:

| ID | Region | ID | Region |
|----|--------|----|--------|
| 0 | background | 11 | upper_arm_r |
| 1 | head | 12 | forearm_r |
| 2 | neck | 13 | hand_r |
| 3 | chest | 14 | upper_leg_l |
| 4 | spine | 15 | lower_leg_l |
| 5 | hips | 16 | foot_l |
| 6 | shoulder_l | 17 | upper_leg_r |
| 7 | upper_arm_l | 18 | lower_leg_r |
| 8 | forearm_l | 19 | foot_r |
| 9 | hand_l | 20 | accessory |
| 10 | shoulder_r | 21 | hair_back |

## How It Works

1. **Pass 1** — renders the model via Puppeteer + pixi-live2d-display (WebGL), exports
   the composite PNG and drawable metadata (mesh IDs, vertex positions, draw orders)
2. **Fragment mapping** — matches each ArtMesh drawable ID against regex patterns
   (English, Japanese romaji, Chinese) to assign a body region ID
3. **Pass 2** — re-renders the model with a custom WebGL shader that paints each mesh
   in its flat region ID colour, producing the segmentation mask
4. **Augmentation** — applies horizontal flip, rotation, and scale to produce 4 variants

Fragment names that don't match any pattern are logged as unmapped. Models with fewer
than 5% mapped fragments are skipped (usually non-humanoid or abstract models).

## Files

```
live2d_adapter/
├── README.md
├── requirements.txt
├── run_live2d.py                    ← CLI entry point
├── pipeline/
│   ├── __init__.py
│   ├── config.py                    ← region IDs, fragment patterns, augmentation settings
│   ├── live2d_renderer.py           ← two-pass renderer (composite + segmentation mask)
│   ├── live2d_mapper.py             ← fragment name → region ID mapping + CSV export
│   ├── live2d_review_ui.py          ← optional Tkinter UI for reviewing mappings
│   ├── moc3_parser.py               ← .moc3 binary parser + atlas fragment extractor
│   └── exporter.py                  ← saves images/masks/JSON in Strata training format
└── tools/live2d_renderer/
    ├── render_live2d.js             ← Puppeteer + pixi-live2d-display renderer
    ├── live2dcubismcore.min.js      ← Cubism Core WASM (redistributable per Live2D terms)
    ├── package.json
    └── package-lock.json            ← run: cd tools/live2d_renderer && npm install
```

## Notes

- `live2dcubismcore.min.js` is the Live2D Cubism Core SDK. It is redistributable for
  research and non-commercial use under the [Live2D Proprietary Software License](https://www.live2d.com/en/terms/live2d-open-software-license-agreement/).
  Check the license before commercial use.
- Live2D models are 2D — there are no true multi-angle views. Augmentation produces
  geometric variants (flip, rotate, scale) rather than camera angle changes.
- The review UI (`live2d_review_ui.py`) requires `tkinter` (usually bundled with Python)
  and `pillow`. Run it directly: `python3 -m pipeline.live2d_review_ui`.

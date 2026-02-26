---
name: render-debug
description: Debug rendering and segmentation issues by generating overlay visualizations. Creates mask-on-image overlays, joint position plots, and region distribution charts for a specific character or pose. Use when visual inspection is needed.
user-invokable: true
argument-hint: "<character_id or image path>"
---

# Render Debug

Generate debug visualizations for: $ARGUMENTS

If a character ID is given (e.g., `mixamo_001`), generate visualizations for all poses of that character. If a specific image path is given, generate visualization for that single image.

## Debug Visualizations

### 1. Mask Overlay
Overlay the segmentation mask on the color image at 50% opacity, with each region rendered in its assigned color from `config.py REGION_COLORS`.

```python
from PIL import Image
import numpy as np
from config import REGION_COLORS

def create_mask_overlay(image_path: str, mask_path: str, output_path: str):
    """Overlay segmentation mask on color image with 50% opacity."""
    image = Image.open(image_path).convert("RGBA")
    mask = np.array(Image.open(mask_path))

    # Create colored overlay from mask
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    for region_id, color in REGION_COLORS.items():
        region_mask = mask == region_id
        overlay[region_mask] = (*color, 128)  # 50% alpha

    overlay_img = Image.fromarray(overlay, "RGBA")
    composite = Image.alpha_composite(image, overlay_img)
    composite.save(output_path)
```

Save to: `dataset/debug/{char_id}_pose_{nn}_overlay.png`

### 2. Joint Position Plot
Draw joint positions as colored circles on the color image, with lines connecting parent-child joints (skeleton visualization).

```python
from PIL import Image, ImageDraw
import json

def create_joint_overlay(image_path: str, joint_path: str, output_path: str):
    """Draw joint positions and skeleton lines on the image."""
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)

    with open(joint_path) as f:
        data = json.load(f)

    for joint_name, joint_data in data["joints"].items():
        x, y = joint_data["position"]
        visible = joint_data["visible"]

        # Green for visible, red for occluded
        color = (0, 255, 0, 255) if visible else (255, 0, 0, 255)
        radius = 4
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
        draw.text((x + 6, y - 6), joint_name, fill=color)

    image.save(output_path)
```

Save to: `dataset/debug/{char_id}_pose_{nn}_joints.png`

### 3. Region Distribution Chart
For each mask, show a bar chart of pixel counts per region. Helps identify mapping errors (e.g., entire arm mapped to chest).

```python
import numpy as np
from PIL import Image

def region_distribution(mask_path: str) -> dict:
    """Count pixels per region in a segmentation mask."""
    mask = np.array(Image.open(mask_path))
    unique, counts = np.unique(mask, return_counts=True)
    total_nonzero = counts[unique != 0].sum()

    distribution = {}
    for region_id, count in zip(unique, counts):
        if region_id != 0:  # Skip background
            distribution[int(region_id)] = {
                "pixels": int(count),
                "percent": round(100 * count / total_nonzero, 1)
            }

    return distribution
```

Print distribution to console and flag any region >60% or any expected region missing.

### 4. Style Comparison Grid
For a given character + pose, create a side-by-side grid of all style variants with the mask overlay, confirming all styles align with the same mask.

Save to: `dataset/debug/{char_id}_pose_{nn}_style_grid.png`

### 5. Bone Mapping Visualization
For a character in T-pose, color each mesh face by its assigned region and label with the bone name that determined the assignment. Helps debug bone mapping issues.

Save to: `dataset/debug/{char_id}_bone_map.png`

## Usage

```bash
# Debug a specific character (all poses)
# /render-debug mixamo_001

# Debug a specific image
# /render-debug dataset/images/mixamo_001_pose_00_flat.png

# Debug all characters (spot check — random 5%)
# /render-debug --spot-check
```

## Output

All debug images are saved to `dataset/debug/`. Report:
- Number of visualizations generated
- Any anomalies found (misaligned masks, out-of-bounds joints, skewed region distributions)
- Recommendations for fixes if issues are detected

# Annotation Guide: 2D Character Segmentation

Step-by-step instructions for annotating hand-drawn 2D characters using Label Studio.

## Prerequisites

- Python 3.10+
- Label Studio installed: `pip install label-studio`
- Project dependencies: `pip install -r requirements.txt`

## 1. Start Label Studio

```bash
label-studio start
```

Open http://localhost:8080 in your browser and create an account.

## 2. Create a Project

1. Click **Create Project**
2. Name it: `Strata Segmentation`
3. Go to **Settings → Labeling Interface → Code**
4. Paste the contents of `annotation/label_studio_config.xml`
5. Save

## 3. Prepare and Import Images

Resize source images to 512×512 and optionally upload them:

```bash
# Resize only (then import manually via Label Studio UI)
python -m annotation.import_images \
    --image_dir ./data/sprites/ \
    --output_dir ./data/sprites_resized/

# Resize + upload via API
python -m annotation.import_images \
    --image_dir ./data/sprites/ \
    --output_dir ./data/sprites_resized/ \
    --ls_url http://localhost:8080 \
    --ls_token YOUR_API_TOKEN \
    --project_id 1

# Generate import JSON for local file serving
python -m annotation.import_images \
    --image_dir ./data/sprites/ \
    --output_dir ./data/sprites_resized/ \
    --generate_tasks ./data/sprites_resized/tasks.json
```

To find your API token: Label Studio → Account & Settings → Access Token.

To import manually: project → Import → Upload Files → select the PNGs from `data/sprites_resized/`.

## 4. Annotating a Character

Each task has two steps:

### Step 1: Polygon Segmentation

Draw polygon outlines around each visible body region. The 19 body regions are:

| ID | Region | ID | Region |
|----|--------|----|--------|
| 1  | head | 10 | forearm_r |
| 2  | neck | 11 | hand_r |
| 3  | chest | 12 | upper_leg_l |
| 4  | spine | 13 | lower_leg_l |
| 5  | hips | 14 | foot_l |
| 6  | upper_arm_l | 15 | upper_leg_r |
| 7  | forearm_l | 16 | lower_leg_r |
| 8  | hand_l | 17 | foot_r |
| 9  | upper_arm_r | 18 | shoulder_l |
|    |      | 19 | shoulder_r |

**Tips:**
- Select a label from the left panel, then click points around the body part to draw a polygon
- Close the polygon by clicking the first point again
- **Overlap is OK** — later polygons overwrite earlier ones. Draw large regions (chest) first, then smaller regions (hands) on top
- Any unlabeled area becomes background (region 0)
- Left/right is from the **character's perspective** (mirror of what you see)
- Zoom in (`Ctrl + scroll`) for precision on small regions like hands and feet

### Step 2: Keypoint Placement

Place one keypoint at the center of each visible joint:

- Select a joint label, then click once to place the keypoint
- Place keypoints at the **center of the joint** (e.g., middle of the elbow for lower_arm)
- Skip joints that are not visible (occluded or out of frame) — they will be marked as `visible: false`
- You don't need all 19 keypoints — annotate what you can see

### Quality Checklist

Before submitting each annotation:

- [ ] All visible body regions have polygon outlines
- [ ] No region is accidentally labeled as a different region
- [ ] Left/right sides are correct (character's perspective)
- [ ] Keypoints are placed at joint centers, not edges
- [ ] Polygons follow the body contour closely (±5px tolerance)

## 5. Export Annotations

After annotating, export from Label Studio:

1. Go to your project → Export
2. Choose **JSON** format
3. Download the file

Then run the export script:

```bash
python -m annotation.export_annotations \
    --ls_export ./annotation_export.json \
    --image_dir ./data/sprites_resized/ \
    --output_dir ./output/segmentation/ \
    --start_id 1
```

This produces:
- `masks/{char_id}_pose_00.png` — 8-bit grayscale segmentation mask
- `joints/{char_id}_pose_00.json` — Joint position JSON
- `images/{char_id}_pose_00_flat.png` — Source image with correct naming
- `sources/{char_id}.json` — Source metadata

## 6. Validate Output

Run the dataset validator to ensure exported annotations pass all quality checks:

```bash
python run_validation.py --dataset_dir ./output/segmentation/ --save_report
```

All 7 checks must pass:
- Resolution: 512×512
- Mask completeness: every non-transparent pixel has a region
- Mask uniqueness: at least 2 distinct body regions per mask
- Joint count: 19 joints present
- Joint bounds: visible joints within image bounds
- File pairing: every image has a mask and joint file
- Region distribution: no single region > 60% of foreground

## 7. Common Issues

| Issue | Solution |
|-------|----------|
| Mask validation fails: all-one-region | You probably forgot to annotate multiple body parts. Go back and add polygons for chest, arms, legs, etc. |
| Joints out of bounds | Keypoint was placed outside the image. Zoom in and re-place it within the 512×512 canvas. |
| Missing mask for image | The export script didn't find polygon annotations. Check that the task has at least one polygon. |
| Wrong left/right | Remember: left/right is from the **character's** perspective. The character's left arm is on YOUR right side of the screen. |

## Target

Annotate 50–100 characters at approximately 1–2 minutes per character for an experienced annotator.

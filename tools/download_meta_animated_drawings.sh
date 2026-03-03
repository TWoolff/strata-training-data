#!/usr/bin/env bash
# Download the Meta Animated Drawings dataset to external HD.
# License: MIT
# Source: https://github.com/facebookresearch/AnimatedDrawings
#
# What this downloads:
#   - amateur_drawings_annotations.json (~275 MB) — bounding boxes, segmentation
#     masks, and 15-joint keypoint annotations for 178K+ hand-drawn figures
#   - amateur_drawings.tar (~50 GB) — the JPEG images

set -euo pipefail

DEST="/Volumes/TAMWoolff/data/preprocessed/meta_animated_drawings"

# --- preflight ---
if [ ! -d "/Volumes/TAMWoolff" ]; then
    echo "ERROR: External HD not mounted at /Volumes/TAMWoolff"
    exit 1
fi

mkdir -p "$DEST"
echo "Saving to: $DEST"

# --- annotations (~275 MB) ---
ANNOTATIONS_URL="https://dl.fbaipublicfiles.com/amateur_drawings/amateur_drawings_annotations.json"
ANNOTATIONS_DEST="$DEST/amateur_drawings_annotations.json"

if [ -f "$ANNOTATIONS_DEST" ]; then
    echo "Annotations already downloaded, skipping."
else
    echo ""
    echo "Downloading annotations (~275 MB)..."
    curl -L --continue-at - --progress-bar \
         -o "$ANNOTATIONS_DEST" \
         "$ANNOTATIONS_URL"
    echo "Annotations done."
fi

# --- images (~50 GB) ---
IMAGES_URL="https://dl.fbaipublicfiles.com/amateur_drawings/amateur_drawings.tar"
IMAGES_DEST="$DEST/amateur_drawings.tar"

if [ -f "$IMAGES_DEST" ]; then
    echo "Images tar already downloaded, skipping."
else
    echo ""
    echo "Downloading images (~50 GB) — this will take a while..."
    curl -L --continue-at - --progress-bar \
         -o "$IMAGES_DEST" \
         "$IMAGES_URL"
    echo "Images tar done."
fi

# --- extract ---
IMAGES_DIR="$DEST/images"
if [ -d "$IMAGES_DIR" ]; then
    echo "Images already extracted, skipping."
else
    echo ""
    echo "Extracting images..."
    tar -xf "$IMAGES_DEST" -C "$DEST"
    echo "Extraction done."
fi

echo ""
echo "=== Meta Animated Drawings download complete ==="
echo "  Annotations: $ANNOTATIONS_DEST"
echo "  Images dir:  $IMAGES_DIR"
du -sh "$DEST"

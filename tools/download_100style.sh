#!/usr/bin/env bash
# Download the 100STYLE mocap dataset to external HD.
# License: CC BY 4.0
# Source: https://zenodo.org/records/8127870
#
# What this downloads:
#   - 100STYLE.zip (~1.5 GB) — BVH motion capture files for 100 locomotion styles
#   - 100Style-Labelled-Data.zip (~14.8 GB) — labelled data with style annotations

set -euo pipefail

DEST="/Volumes/TAMWoolff/data/preprocessed/100style"

# --- preflight ---
if [ ! -d "/Volumes/TAMWoolff" ]; then
    echo "ERROR: External HD not mounted at /Volumes/TAMWoolff"
    exit 1
fi

mkdir -p "$DEST"
echo "Saving to: $DEST"

# --- BVH zip (~1.5 GB) ---
BVH_URL="https://zenodo.org/records/8127870/files/100STYLE.zip?download=1"
BVH_DEST="$DEST/100STYLE.zip"

if [ -f "$BVH_DEST" ]; then
    echo "BVH zip already downloaded, skipping."
else
    echo ""
    echo "Downloading BVH data (~1.5 GB)..."
    curl -L --continue-at - --progress-bar \
         -o "$BVH_DEST" \
         "$BVH_URL"
    echo "BVH zip done."
fi

# --- labelled data (~14.8 GB) ---
LABELLED_URL="https://zenodo.org/records/8127870/files/100Style-Labelled-Data.zip?download=1"
LABELLED_DEST="$DEST/100Style-Labelled-Data.zip"

if [ -f "$LABELLED_DEST" ]; then
    echo "Labelled data zip already downloaded, skipping."
else
    echo ""
    echo "Downloading labelled data (~14.8 GB) — this will take a while..."
    curl -L --continue-at - --progress-bar \
         -o "$LABELLED_DEST" \
         "$LABELLED_URL"
    echo "Labelled data zip done."
fi

# --- extract BVH ---
BVH_DIR="$DEST/100STYLE"
if [ -d "$BVH_DIR" ]; then
    echo "BVH data already extracted, skipping."
else
    echo ""
    echo "Extracting BVH data..."
    unzip -q "$BVH_DEST" -d "$DEST"
    echo "BVH extraction done."
fi

# --- extract labelled ---
LABELLED_DIR="$DEST/100Style-Labelled-Data"
if [ -d "$LABELLED_DIR" ]; then
    echo "Labelled data already extracted, skipping."
else
    echo ""
    echo "Extracting labelled data..."
    unzip -q "$LABELLED_DEST" -d "$DEST"
    echo "Labelled data extraction done."
fi

echo ""
echo "=== 100STYLE download complete ==="
echo "  BVH data:      $BVH_DIR"
echo "  Labelled data: $LABELLED_DIR"
du -sh "$DEST"

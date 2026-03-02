#!/usr/bin/env bash
# Re-render segmentation data with corrected 22-class region IDs.
#
# The previous render used the old 20-class scheme (shoulders at IDs 18-19,
# lower_arm naming). This script re-generates all masks, joints, weights,
# draw order, layers, and measurements with the corrected taxonomy that
# matches Strata's skeleton.ts RegionId enum.
#
# Previous render config (from manifest.json):
#   - 61 characters, 5 poses each, flat style only, front angle, 512px
#
# This script reproduces those settings, then uploads to the bucket
# (replacing the old data).
#
# Usage:
#   caffeinate -dims bash rerender_segmentation.sh 2>&1 | tee rerender_log.txt

set -uo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Configuration — match the original render settings
# ---------------------------------------------------------------------------
BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
INPUT_DIR="./data/fbx"
POSE_DIR="./data/poses"
OUTPUT_DIR="./output/segmentation"
RESOLUTION=512
STYLES="flat"
POSES_PER_CHARACTER=50
ANGLES="front"

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
if [ ! -f .env ]; then
    echo "Error: .env file not found. Add BUCKET_ACCESS_KEY and BUCKET_SECRET."
    exit 1
fi
source .env
export AWS_ACCESS_KEY_ID="$BUCKET_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$BUCKET_SECRET"
ENDPOINT="https://fsn1.your-objectstorage.com"
BUCKET="s3://strata-training-data"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "=== Segmentation Re-render (22-class region IDs) ==="
echo "    Started:  $(date)"
echo "    Blender:  $BLENDER"
echo "    Input:    $INPUT_DIR"
echo "    Poses:    $POSE_DIR"
echo "    Output:   $OUTPUT_DIR"
echo "    Config:   ${RESOLUTION}px, styles=${STYLES}, poses=${POSES_PER_CHARACTER}, angles=${ANGLES}"
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

if [ ! -x "$BLENDER" ]; then
    echo "Error: Blender not found at $BLENDER"
    echo "  Set BLENDER= to the correct path."
    exit 1
fi

FBX_COUNT=$(find "$INPUT_DIR" -name "*.fbx" -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')
POSE_COUNT=$(find "$POSE_DIR" -name "*.fbx" -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')
echo "    Characters: $FBX_COUNT FBX files"
echo "    Poses available: $POSE_COUNT animation clips"
echo ""

if [ "$FBX_COUNT" -eq 0 ]; then
    echo "Error: No FBX files found in $INPUT_DIR"
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 1: Clear old local output
# ---------------------------------------------------------------------------
echo "=== Phase 1: Clearing old local output ==="
if [ -d "$OUTPUT_DIR" ] && [ "$(find "$OUTPUT_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')" -gt 0 ]; then
    echo "  Removing old output..."
    rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"
echo "  Done."
echo ""

# ---------------------------------------------------------------------------
# Phase 2: Run Blender pipeline
# ---------------------------------------------------------------------------
START_RENDER=$(date +%s)
echo "=== Phase 2: Rendering ($FBX_COUNT characters × $POSES_PER_CHARACTER poses) ==="

"$BLENDER" --background --python run_pipeline.py -- \
    --input_dir "$INPUT_DIR" \
    --pose_dir "$POSE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --styles "$STYLES" \
    --resolution "$RESOLUTION" \
    --poses_per_character "$POSES_PER_CHARACTER" \
    --angles "$ANGLES"

RENDER_STATUS=$?
END_RENDER=$(date +%s)
RENDER_MIN=$(( (END_RENDER - START_RENDER) / 60 ))

if [ $RENDER_STATUS -ne 0 ]; then
    echo ""
    echo "WARNING: Blender exited with status $RENDER_STATUS"
    echo "  Check output above for errors. Continuing with upload anyway..."
fi

OUTPUT_COUNT=$(find "$OUTPUT_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
OUTPUT_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
echo ""
echo "  Render complete in ${RENDER_MIN} minutes."
echo "  Output: $OUTPUT_COUNT files ($OUTPUT_SIZE)"
echo ""

# ---------------------------------------------------------------------------
# Phase 3: Delete old bucket data and upload new
# ---------------------------------------------------------------------------
echo "=== Phase 3: Uploading to bucket ==="
echo "  Deleting old segmentation data from bucket..."
aws s3 rm "${BUCKET}/segmentation/" \
    --endpoint-url "$ENDPOINT" \
    --recursive \
    --only-show-errors
echo "  Old data deleted."

echo "  Uploading new segmentation data..."
aws s3 sync "$OUTPUT_DIR" "${BUCKET}/segmentation/" \
    --endpoint-url "$ENDPOINT" \
    --no-progress \
    --only-show-errors
echo "  Upload complete."
echo ""

# ---------------------------------------------------------------------------
# Phase 4: Verify
# ---------------------------------------------------------------------------
echo "=== Phase 4: Verification ==="
echo "  Bucket contents:"
aws s3 ls "${BUCKET}/segmentation/" --endpoint-url "$ENDPOINT" 2>&1
echo ""

echo "  Downloading new class_map.json to verify..."
aws s3 cp "${BUCKET}/segmentation/class_map.json" - \
    --endpoint-url "$ENDPOINT" 2>/dev/null | python3 -m json.tool
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
END_TIME=$(date +%s)
TOTAL_MIN=$(( (END_TIME - START_RENDER) / 60 ))
echo "=== Re-render complete ==="
echo "    Render time:  ${RENDER_MIN} minutes"
echo "    Total time:   ${TOTAL_MIN} minutes ($(( TOTAL_MIN / 60 ))h $(( TOTAL_MIN % 60 ))m)"
echo "    Output files: $OUTPUT_COUNT ($OUTPUT_SIZE)"
echo "    Free disk:    $(df -h . | tail -1 | awk '{print $4}')"
echo "    Finished:     $(date)"

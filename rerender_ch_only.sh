#!/usr/bin/env bash
# Re-render only Ch##_nonPBR characters with the fixed mixamorig#: prefix support.
# Outputs to output/segmentation_ch/ then uploads to bucket under segmentation/
# (merging with existing data — does NOT delete the full dataset first).
#
# Usage:
#   caffeinate -dims bash rerender_ch_only.sh 2>&1 | tee rerender_ch_log.txt

set -uo pipefail
cd "$(dirname "$0")"

BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
INPUT_DIR="./data/fbx_ch_only"
POSE_DIR="./data/poses"
OUTPUT_DIR="./output/segmentation_ch"
RESOLUTION=512
STYLES="flat"
POSES_PER_CHARACTER=50
ANGLES="front"

if [ ! -f .env ]; then
    echo "Error: .env file not found."
    exit 1
fi
source .env
export AWS_ACCESS_KEY_ID="$BUCKET_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$BUCKET_SECRET"
ENDPOINT="https://fsn1.your-objectstorage.com"
BUCKET="s3://strata-training-data"

FBX_COUNT=$(find "$INPUT_DIR" -name "*.fbx" -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')
echo "=== Ch##_nonPBR Re-render ==="
echo "  Input:   $INPUT_DIR ($FBX_COUNT FBX files)"
echo "  Output:  $OUTPUT_DIR"
echo "  Poses:   $POSES_PER_CHARACTER per character"
echo ""

# ---------------------------------------------------------------------------
# Phase 1: Clear output dir
# ---------------------------------------------------------------------------
echo "=== Phase 1: Clearing output ==="
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
echo "  Cleared."
echo ""

# ---------------------------------------------------------------------------
# Phase 2: Render
# ---------------------------------------------------------------------------
echo "=== Phase 2: Rendering ==="
START_RENDER=$(date +%s)

"$BLENDER" --background --python run_pipeline.py -- \
    --input_dir "$INPUT_DIR" \
    --pose_dir "$POSE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --resolution "$RESOLUTION" \
    --styles "$STYLES" \
    --poses_per_character "$POSES_PER_CHARACTER" \
    --angles "$ANGLES"

RENDER_EXIT=$?
END_RENDER=$(date +%s)
RENDER_MINS=$(( (END_RENDER - START_RENDER) / 60 ))

if [ $RENDER_EXIT -ne 0 ]; then
    echo "ERROR: Blender exited with code $RENDER_EXIT"
    exit $RENDER_EXIT
fi

FILE_COUNT=$(find "$OUTPUT_DIR" -type f | wc -l | tr -d ' ')
DIR_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
echo ""
echo "  Render complete in $RENDER_MINS minutes."
echo "  Output: $FILE_COUNT files ($DIR_SIZE)"
echo ""

# ---------------------------------------------------------------------------
# Phase 3: Upload (merge — do NOT delete existing bucket data)
# ---------------------------------------------------------------------------
echo "=== Phase 3: Uploading to bucket (merge) ==="
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
aws s3 ls "${BUCKET}/segmentation/images/" --endpoint-url "$ENDPOINT" --summarize 2>&1 | tail -3
echo ""

END_TOTAL=$(date +%s)
TOTAL_MINS=$(( (END_TOTAL - START_RENDER) / 60 ))
FREE_DISK=$(df -h / | awk 'NR==2{print $4}')

echo "=== Done ==="
echo "    Render time:  $RENDER_MINS minutes"
echo "    Total time:   $TOTAL_MINS minutes"
echo "    Output files: $FILE_COUNT ($DIR_SIZE)"
echo "    Free disk:    $FREE_DISK"
echo "    Finished:     $(date)"

#!/usr/bin/env bash
# Render additional camera angles for HumanRig and upload to Hetzner bucket.
#
# Run this after run_ingest.py (Phase 1) has already completed.
# It renders three_quarter, side, and back views via Blender Cycles
# for all samples that don't already have image.png, then syncs to S3.
#
# Usage:
#   caffeinate -dims bash run_humanrig_blender.sh 2>&1 | tee humanrig_render_log.txt

set -uo pipefail
cd "$(dirname "$0")"

source .env
export AWS_ACCESS_KEY_ID="$BUCKET_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$BUCKET_SECRET"
ENDPOINT="https://fsn1.your-objectstorage.com"
BUCKET="s3://strata-training-data"

HUMANRIG_INPUT="/Volumes/TAWoolff EXT/data/preprocessed/humanrig/data/54T/chuzedong/autorig/preprocess/humanrig_opensource_final"
OUTPUT_DIR="./output/humanrig"
BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"

START_TIME=$(date +%s)
echo "=== HumanRig Blender render started at $(date) ==="
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if [ ! -d "$HUMANRIG_INPUT" ]; then
    echo "ERROR: HumanRig source not found at:"
    echo "       $HUMANRIG_INPUT"
    echo "       Is the external HD mounted?"
    exit 1
fi

if [ ! -f "$BLENDER" ]; then
    echo "ERROR: Blender not found at $BLENDER"
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory not found at $OUTPUT_DIR"
    echo "       Run Phase 1 first: python3 run_ingest.py --adapter humanrig ..."
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 1: Blender render (three_quarter, side, back)
# ---------------------------------------------------------------------------
echo "=== Phase 1: Blender render (three_quarter + side + back) ==="
"$BLENDER" --background --python run_humanrig_render.py -- \
    --input_dir "$HUMANRIG_INPUT" \
    --output_dir "$OUTPUT_DIR" \
    --angles three_quarter,side,back \
    --only_new \
    || { echo "ERROR: Blender render failed"; exit 1; }

RENDER_END=$(date +%s)
echo "    Render complete. Elapsed: $(( (RENDER_END - START_TIME) / 60 )) min"
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

# ---------------------------------------------------------------------------
# Phase 2: Upload to Hetzner bucket
# ---------------------------------------------------------------------------
echo "=== Phase 2: Upload to bucket ==="
file_count=$(find "$OUTPUT_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
echo "    Uploading humanrig ($file_count files, $size)..."

aws s3 sync "$OUTPUT_DIR" "${BUCKET}/humanrig/" \
    --endpoint-url "$ENDPOINT" \
    --no-progress \
    --only-show-errors

echo "    Upload complete."

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
END_TIME=$(date +%s)
TOTAL_MIN=$(( (END_TIME - START_TIME) / 60 ))
echo ""
echo "=== Done at $(date) ==="
echo "    Total runtime: ${TOTAL_MIN} minutes"
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""
echo "=== Bucket contents ==="
aws s3 ls "${BUCKET}/" --endpoint-url "$ENDPOINT" 2>&1

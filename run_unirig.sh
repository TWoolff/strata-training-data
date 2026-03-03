#!/usr/bin/env bash
# Ingest UniRig / Rig-XL dataset (16,641 meshes) and upload to Hetzner bucket.
#
# Pipeline:
#   Phase 1 (~6-12 hours) — Blender: load each raw_data.npz, filter to humanoids,
#                            render front + back orthographic views with segmentation masks.
#   Phase 2 (~2 min)      — Upload ./output/unirig/ to S3 bucket.
#
# Usage:
#   caffeinate -dims bash run_unirig.sh 2>&1 | tee unirig_log.txt

set -uo pipefail
cd "$(dirname "$0")"

source .env
export AWS_ACCESS_KEY_ID="$BUCKET_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$BUCKET_SECRET"
ENDPOINT="https://fsn1.your-objectstorage.com"
BUCKET="s3://strata-training-data"

UNIRIG_INPUT="/Volumes/TAWoolff EXT/data/preprocessed/unirig/rigxl"
OUTPUT_DIR="./output/unirig"
BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"

START_TIME=$(date +%s)
echo "=== UniRig pipeline started at $(date) ==="
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if [ ! -d "$UNIRIG_INPUT" ]; then
    echo "ERROR: UniRig source not found at:"
    echo "       $UNIRIG_INPUT"
    echo "       Is the external HD mounted?"
    exit 1
fi

if [ ! -f "$BLENDER" ]; then
    echo "ERROR: Blender not found at $BLENDER"
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 1: Blender ingest
#   - Loads each raw_data.npz, maps bones → Strata regions
#   - Filters non-humanoids (requires ≥60% coverage + bilateral symmetry)
#   - Renders front (0°) + back (180°) orthographic views
#   - Writes image.png, segmentation.png, metadata.json per view
# ---------------------------------------------------------------------------
echo "=== Phase 1: Blender ingest (front + back renders) ==="
"$BLENDER" --background --python run_unirig.py -- \
    --input_dir "$UNIRIG_INPUT" \
    --output_dir "$OUTPUT_DIR" \
    --only_new \
    || { echo "ERROR: Blender ingest failed"; exit 1; }

PHASE1_END=$(date +%s)
echo "    Phase 1 complete. Elapsed: $(( (PHASE1_END - START_TIME) / 60 )) min"
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

# ---------------------------------------------------------------------------
# Phase 2: Upload to Hetzner bucket
# ---------------------------------------------------------------------------
echo "=== Phase 2: Upload to bucket ==="
file_count=$(find "$OUTPUT_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
echo "    Uploading unirig ($file_count files, $size)..."

aws s3 sync "$OUTPUT_DIR" "${BUCKET}/unirig/" \
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

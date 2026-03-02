#!/usr/bin/env bash
# Ingest HumanRig dataset (11,434 samples × 4 angles) and upload to Hetzner bucket.
# Estimated runtime: ~15 minutes (13 min ingest + 2 min upload).
# Output: ~130 MB (front images + joint projections for all angles).
#
# Usage:
#   caffeinate -dims bash run_humanrig.sh 2>&1 | tee humanrig_log.txt

set -uo pipefail
cd "$(dirname "$0")"

source .env
export AWS_ACCESS_KEY_ID="$BUCKET_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$BUCKET_SECRET"
ENDPOINT="https://fsn1.your-objectstorage.com"
BUCKET="s3://strata-training-data"

HUMANRIG_INPUT="/Volumes/TAWoolff EXT/data/preprocessed/humanrig/data/54T/chuzedong/autorig/preprocess/humanrig_opensource_final"
OUTPUT_DIR="./output/humanrig"

START_TIME=$(date +%s)
echo "=== HumanRig ingest started at $(date) ==="
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

# ---------------------------------------------------------------------------
# Phase 1: Ingest (front image + joint projections for all 4 angles)
# ---------------------------------------------------------------------------
echo "=== Phase 1: Ingest ==="

if [ ! -d "$HUMANRIG_INPUT" ]; then
    echo "ERROR: HumanRig source not found at: $HUMANRIG_INPUT"
    echo "       Is the external HD mounted?"
    exit 1
fi

python3 run_ingest.py \
    --adapter humanrig \
    --input_dir "$HUMANRIG_INPUT" \
    --output_dir "$OUTPUT_DIR" \
    --angles front,three_quarter,side,back \
    --only_new \
    || { echo "ERROR: ingest failed"; exit 1; }

echo ""
echo "    Free disk after ingest: $(df -h . | tail -1 | awk '{print $4}')"

# ---------------------------------------------------------------------------
# Phase 2: Upload to Hetzner bucket
# ---------------------------------------------------------------------------
echo ""
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

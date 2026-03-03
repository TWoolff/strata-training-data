#!/usr/bin/env bash
# Upload completed UniRig output to Hetzner bucket.
set -uo pipefail
cd "$(dirname "$0")"

source .env
export AWS_ACCESS_KEY_ID="$BUCKET_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$BUCKET_SECRET"
ENDPOINT="https://fsn1.your-objectstorage.com"
BUCKET="s3://strata-training-data"
OUTPUT_DIR="./output/unirig"

file_count=$(find "$OUTPUT_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
echo "Uploading unirig ($file_count files, $size) at $(date)..."

aws s3 sync "$OUTPUT_DIR" "${BUCKET}/unirig/" \
    --endpoint-url "$ENDPOINT" \
    --no-progress \
    --only-show-errors

echo "Done at $(date)"

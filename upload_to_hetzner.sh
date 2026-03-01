#!/usr/bin/env bash
# Upload training data output to Hetzner Object Storage (S3-compatible).
#
# Syncs each output subdirectory to the bucket. Only uploads new/changed files.
# Safe to run multiple times — aws s3 sync is incremental.
#
# Usage:
#   bash upload_to_hetzner.sh                    # Upload all output dirs
#   bash upload_to_hetzner.sh segmentation       # Upload specific dataset
#   bash upload_to_hetzner.sh --delete-local      # Upload all, then delete local copies

set -euo pipefail
cd "$(dirname "$0")"

# Load credentials from .env
if [ ! -f .env ]; then
    echo "Error: .env file not found. Add BUCKET_ACCESS_KEY and BUCKET_SECRET."
    exit 1
fi
source .env

ENDPOINT="https://fsn1.your-objectstorage.com"
BUCKET="s3://strata-training-data"
OUTPUT_DIR="./output"
DELETE_LOCAL=false
SPECIFIC_DATASET=""

# Parse args
for arg in "$@"; do
    if [ "$arg" = "--delete-local" ]; then
        DELETE_LOCAL=true
    else
        SPECIFIC_DATASET="$arg"
    fi
done

export AWS_ACCESS_KEY_ID="$BUCKET_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$BUCKET_SECRET"

sync_dataset() {
    local name="$1"
    local src="${OUTPUT_DIR}/${name}"

    if [ ! -d "$src" ]; then
        echo "  Skipping $name — directory not found"
        return
    fi

    local file_count
    file_count=$(find "$src" -type f 2>/dev/null | wc -l | tr -d ' ')
    if [ "$file_count" -eq 0 ]; then
        echo "  Skipping $name — empty"
        return
    fi

    local size
    size=$(du -sh "$src" 2>/dev/null | cut -f1)
    echo "  Uploading $name ($file_count files, $size)..."

    aws s3 sync "$src" "${BUCKET}/${name}/" \
        --endpoint-url "$ENDPOINT" \
        --no-progress \
        --only-show-errors

    echo "  $name uploaded."

    if [ "$DELETE_LOCAL" = true ]; then
        echo "  Deleting local copy of $name..."
        rm -rf "$src"
        mkdir -p "$src"
        echo "  Local copy deleted."
    fi
}

echo "=== Uploading to Hetzner Object Storage ==="
echo "    Endpoint: $ENDPOINT"
echo "    Bucket:   $BUCKET"
echo "    Started:  $(date)"
echo ""

if [ -n "$SPECIFIC_DATASET" ] && [ "$SPECIFIC_DATASET" != "--delete-local" ]; then
    sync_dataset "$SPECIFIC_DATASET"
else
    # Upload all non-empty output directories
    for dir in "$OUTPUT_DIR"/*/; do
        name=$(basename "$dir")
        [ "$name" = "." ] && continue
        [ "$name" = ".." ] && continue
        sync_dataset "$name"
    done
fi

echo ""
echo "=== Upload complete at $(date) ==="

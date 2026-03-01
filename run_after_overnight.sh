#!/usr/bin/env bash
# Runs after run_overnight.sh finishes (waits for PID 51444).
# Uploads each dataset to Hetzner, deletes local copy, then processes
# FBAnimeHQ shards with upload+cleanup per shard.
#
# Usage: nohup bash run_after_overnight.sh >> overnight_log.txt 2>&1 &

set -euo pipefail
cd "$(dirname "$0")"

# Load credentials
source .env
export AWS_ACCESS_KEY_ID="$BUCKET_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$BUCKET_SECRET"
ENDPOINT="https://fsn1.your-objectstorage.com"
BUCKET="s3://strata-training-data"

upload_and_cleanup() {
    local name="$1"
    local src="./output/${name}"

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

    echo "  Upload complete. Deleting local copy..."
    rm -rf "$src"
    mkdir -p "$src"
    echo "  $name: uploaded and cleaned. Free disk: $(df -h . | tail -1 | awk '{print $4}')"
}

# ---------------------------------------------------------------------------
# Wait for overnight script to finish
# ---------------------------------------------------------------------------
echo "=== Post-overnight pipeline started at $(date) ==="
echo "    Waiting for overnight script (PID 51444) to finish..."

while kill -0 51444 2>/dev/null; do
    sleep 60
done
echo "    Overnight script finished at $(date)"
echo ""

# ---------------------------------------------------------------------------
# Phase 3: Upload Blender output → Hetzner → delete local
# ---------------------------------------------------------------------------
echo "=== Phase 3: Upload Blender segmentation output ==="
upload_and_cleanup "segmentation"
echo ""

# ---------------------------------------------------------------------------
# Phase 4: Upload anime_seg output → Hetzner → delete local
# ---------------------------------------------------------------------------
echo "=== Phase 4: Upload anime_seg output ==="
upload_and_cleanup "anime_seg"
echo ""

# ---------------------------------------------------------------------------
# Phase 5: Upload existing fbanimehq output → Hetzner → delete local
# ---------------------------------------------------------------------------
echo "=== Phase 5: Upload existing FBAnimeHQ output ==="
upload_and_cleanup "fbanimehq"
echo ""

# ---------------------------------------------------------------------------
# Phase 6: FBAnimeHQ shards 08-11 (extract → ingest → upload → cleanup each)
# ---------------------------------------------------------------------------
echo "=== Phase 6: Process remaining FBAnimeHQ shards ==="
SHARD_DIR="./data/preprocessed/fbanimehq/data"

for SHARD_NUM in 08 09 10 11; do
    SHARD_NAME="fbanimehq-${SHARD_NUM}"
    ZIP_PATH="${SHARD_DIR}/${SHARD_NAME}.zip"
    EXTRACT_DIR="${SHARD_DIR}/${SHARD_NAME}"

    if [ ! -f "$ZIP_PATH" ]; then
        echo "  Skipping ${SHARD_NAME} — zip not found"
        continue
    fi

    echo "  --- ${SHARD_NAME} ---"
    echo "  Free disk: $(df -h . | tail -1 | awk '{print $4}')"

    # Extract
    echo "  Extracting..."
    unzip -q -o "$ZIP_PATH" -d "$SHARD_DIR"

    # Ingest
    echo "  Ingesting..."
    python3 run_ingest.py \
        --adapter fbanimehq \
        --input_dir "$EXTRACT_DIR" \
        --output_dir "./output/fbanimehq"

    # Delete extracted source to reclaim ~2.8GB
    echo "  Cleaning up extracted source..."
    rm -rf "$EXTRACT_DIR"

    # Upload to Hetzner and delete local output
    echo "  Uploading to Hetzner..."
    upload_and_cleanup "fbanimehq"

    echo "  ${SHARD_NAME} done at $(date)"
    echo ""
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=== All post-overnight phases complete at $(date) ==="
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

# List what's in the bucket
echo "=== Bucket contents ==="
aws s3 ls "${BUCKET}/" --endpoint-url "$ENDPOINT" --summarize --human-readable 2>&1 | tail -5
echo ""
echo "=== Done ==="

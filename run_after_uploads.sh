#!/usr/bin/env bash
# Runs after run_after_overnight.sh finishes (waits for PID 62585).
# Extracts AnimeRun zip, ingests, uploads to Hetzner, cleans up.
#
# Usage: nohup bash run_after_uploads.sh >> overnight_log.txt 2>&1 &

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
# Wait for upload script to finish
# ---------------------------------------------------------------------------
echo "=== AnimeRun pipeline started at $(date) ==="
echo "    Waiting for upload script (PID 62585) to finish..."

while kill -0 62585 2>/dev/null; do
    sleep 60
done
echo "    Upload script finished at $(date)"
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

# ---------------------------------------------------------------------------
# Phase 7: AnimeRun — extract zip → ingest → upload → cleanup
# ---------------------------------------------------------------------------
ANIMERUN_DIR="./data/preprocessed/animerun"
ANIMERUN_ZIP="${ANIMERUN_DIR}/AnimeRun.zip"

if [ ! -f "$ANIMERUN_ZIP" ]; then
    echo "  AnimeRun zip not found at $ANIMERUN_ZIP — skipping"
    exit 0
fi

echo "=== Phase 7: AnimeRun ==="
echo "    Free disk before extract: $(df -h . | tail -1 | awk '{print $4}')"

# Extract
echo "    Extracting AnimeRun.zip (~22GB)..."
unzip -q -o "$ANIMERUN_ZIP" -d "$ANIMERUN_DIR"
echo "    Extracted. Free disk: $(df -h . | tail -1 | awk '{print $4}')"

# Ingest
echo "    Ingesting AnimeRun contour pairs..."
python3 run_ingest.py \
    --adapter animerun \
    --input_dir "$ANIMERUN_DIR" \
    --output_dir "./output/animerun"

# Delete extracted source to reclaim space (keep the zip as backup)
echo "    Cleaning up extracted data..."
find "$ANIMERUN_DIR" -mindepth 1 -not -name 'AnimeRun.zip' -not -name 'README.md' -not -name '.DS_Store' -exec rm -rf {} + 2>/dev/null || true
echo "    Extracted data cleaned. Free disk: $(df -h . | tail -1 | awk '{print $4}')"

# Upload and delete local output
upload_and_cleanup "animerun"

echo ""
echo "=== AnimeRun complete at $(date) ==="
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

# Final bucket summary
echo "=== Bucket contents ==="
aws s3 ls "${BUCKET}/" --endpoint-url "$ENDPOINT" 2>&1
echo ""
echo "=== All done ==="

#!/usr/bin/env bash
# Overnight batch run #2 — fixes from first run
#
# 1. Anime segmentation ingest (~22K images) — didn't run last time
# 2. FBAnimeHQ shards 08-11 — fix: pass parent dir to adapter, not shard subdir
# 3. AnimeRun — extract only contour + Frame_Anime dirs to save space
#
# Each phase: process → upload to Hetzner → delete local
#
# Usage:
#   caffeinate -dims bash run_overnight_2.sh 2>&1 | tee overnight_log_2.txt

set -uo pipefail  # No -e: don't abort on non-zero exit (some adapters return warnings)
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

START_TIME=$(date +%s)
echo "=== Overnight batch #2 started at $(date) ==="
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

# ---------------------------------------------------------------------------
# Phase 1: Anime segmentation ingest (~22K images)
# ---------------------------------------------------------------------------
echo "=== Phase 1: Anime segmentation ingest ==="
echo "    Started at $(date)"

python3 run_ingest.py \
  --adapter anime_seg \
  --input_dir ./data/preprocessed/anime_segmentation \
  --output_dir ./output/anime_seg || echo "  WARNING: anime_seg adapter exited with errors"

upload_and_cleanup "anime_seg"

PHASE1_END=$(date +%s)
echo "=== Phase 1 complete at $(date) ($(( (PHASE1_END - START_TIME) / 60 )) min) ==="
echo ""

# ---------------------------------------------------------------------------
# Phase 2: FBAnimeHQ shards 08-11
# The zips extract to numbered dirs (0080/, 0081/, etc.) not fbanimehq-08/.
# Fix: extract to temp location, point adapter at parent dir containing
# the numbered subdirs.
# ---------------------------------------------------------------------------
echo "=== Phase 2: FBAnimeHQ shards 08-11 ==="
SHARD_DIR="./data/preprocessed/fbanimehq/data"

for SHARD_NUM in 08 09 10 11; do
    SHARD_NAME="fbanimehq-${SHARD_NUM}"
    ZIP_PATH="${SHARD_DIR}/${SHARD_NAME}.zip"
    EXTRACT_DIR="${SHARD_DIR}/${SHARD_NAME}_tmp"

    if [ ! -f "$ZIP_PATH" ]; then
        echo "  Skipping ${SHARD_NAME} — zip not found"
        continue
    fi

    echo "  --- ${SHARD_NAME} ---"
    echo "  Free disk: $(df -h . | tail -1 | awk '{print $4}')"

    # Extract into a temp dir so we know the path
    mkdir -p "$EXTRACT_DIR"
    echo "  Extracting to ${EXTRACT_DIR}..."
    unzip -q -o "$ZIP_PATH" -d "$EXTRACT_DIR"

    # Ingest — point at the temp dir which now contains numbered subdirs
    echo "  Ingesting..."
    python3 run_ingest.py \
        --adapter fbanimehq \
        --input_dir "$EXTRACT_DIR" \
        --output_dir "./output/fbanimehq" || echo "  WARNING: fbanimehq adapter exited with errors"

    # Delete extracted source
    echo "  Cleaning up extracted source..."
    rm -rf "$EXTRACT_DIR"

    # Upload and delete local output
    upload_and_cleanup "fbanimehq"

    echo "  ${SHARD_NAME} done at $(date)"
    echo ""
done

PHASE2_END=$(date +%s)
echo "=== Phase 2 complete at $(date) ($(( (PHASE2_END - PHASE1_END) / 60 )) min) ==="
echo ""

# ---------------------------------------------------------------------------
# Phase 3: AnimeRun — extract only contour + Frame_Anime to save space
# Full extraction is 85GB; we only need contour + Frame_Anime (~35GB).
# ---------------------------------------------------------------------------
echo "=== Phase 3: AnimeRun ==="
ANIMERUN_DIR="./data/preprocessed/animerun"
ANIMERUN_ZIP="${ANIMERUN_DIR}/AnimeRun.zip"

if [ ! -f "$ANIMERUN_ZIP" ]; then
    echo "  AnimeRun zip not found — skipping"
else
    echo "  Free disk: $(df -h . | tail -1 | awk '{print $4}')"
    echo "  Extracting only contour + Frame_Anime dirs..."

    # Extract only the dirs the adapter needs
    unzip -q -o "$ANIMERUN_ZIP" "AnimeRun_v2/*/contour/*" "AnimeRun_v2/*/Frame_Anime/*" -d "$ANIMERUN_DIR" \
        || echo "  WARNING: some files may have failed to extract"

    echo "  Extracted. Free disk: $(df -h . | tail -1 | awk '{print $4}')"

    # Ingest
    echo "  Ingesting..."
    python3 run_ingest.py \
        --adapter animerun \
        --input_dir "${ANIMERUN_DIR}/AnimeRun_v2" \
        --output_dir "./output/animerun" || echo "  WARNING: animerun adapter exited with errors"

    # Delete extracted data (keep the zip)
    echo "  Cleaning up extracted data..."
    rm -rf "${ANIMERUN_DIR}/AnimeRun_v2"
    echo "  Cleaned. Free disk: $(df -h . | tail -1 | awk '{print $4}')"

    # Upload and delete local output
    upload_and_cleanup "animerun"
fi

PHASE3_END=$(date +%s)
echo "=== Phase 3 complete at $(date) ($(( (PHASE3_END - PHASE2_END) / 60 )) min) ==="
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL_MIN=$(( (PHASE3_END - START_TIME) / 60 ))
echo "=== All phases complete ==="
echo "    Total runtime: ${TOTAL_MIN} minutes ($(( TOTAL_MIN / 60 ))h $(( TOTAL_MIN % 60 ))m)"
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

echo "=== Bucket contents ==="
aws s3 ls "${BUCKET}/" --endpoint-url "$ENDPOINT" 2>&1
echo ""
echo "=== Done at $(date) ==="

#!/usr/bin/env bash
# Overnight batch #3:
# 1. Shrink AnimeRun zip (remove already-ingested contour + Frame_Anime)
# 2. Extract remaining AnimeRun data in stages → upload raw → delete each
# 3. FBAnimeHQ shards 08-11 (extract → ingest → upload → cleanup each)
#
# Usage:
#   caffeinate -dims bash run_overnight_3.sh 2>&1 | tee overnight_log_3.txt

set -uo pipefail
cd "$(dirname "$0")"

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

# Upload a raw extracted directory to the bucket (not via output/)
upload_raw_and_cleanup() {
    local bucket_path="$1"
    local local_path="$2"

    if [ ! -d "$local_path" ]; then
        echo "  Skipping $bucket_path — directory not found"
        return
    fi

    local file_count
    file_count=$(find "$local_path" -type f 2>/dev/null | wc -l | tr -d ' ')
    local size
    size=$(du -sh "$local_path" 2>/dev/null | cut -f1)
    echo "  Uploading $bucket_path ($file_count files, $size)..."

    aws s3 sync "$local_path" "${BUCKET}/${bucket_path}/" \
        --endpoint-url "$ENDPOINT" \
        --no-progress \
        --only-show-errors

    echo "  Upload complete. Deleting local copy..."
    rm -rf "$local_path"
    echo "  $bucket_path: uploaded and cleaned. Free disk: $(df -h . | tail -1 | awk '{print $4}')"
}

START_TIME=$(date +%s)
echo "=== Overnight batch #3 started at $(date) ==="
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

# ---------------------------------------------------------------------------
# Phase 1: Shrink AnimeRun zip (remove already-ingested contour + Frame_Anime)
# ---------------------------------------------------------------------------
ANIMERUN_DIR="./data/preprocessed/animerun"
ANIMERUN_ZIP="${ANIMERUN_DIR}/AnimeRun.zip"

echo "=== Phase 1: Shrink AnimeRun zip ==="
echo "    Current zip size: $(du -sh "$ANIMERUN_ZIP" 2>/dev/null | cut -f1)"
echo "    Removing contour + Frame_Anime (already ingested)..."

zip -q -d "$ANIMERUN_ZIP" "AnimeRun_v2/*/contour/*" "AnimeRun_v2/*/Frame_Anime/*" \
    || echo "  WARNING: zip delete had errors (may already be removed)"

echo "    New zip size: $(du -sh "$ANIMERUN_ZIP" 2>/dev/null | cut -f1)"
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

# ---------------------------------------------------------------------------
# Phase 2: Extract remaining AnimeRun data one type at a time
# Each type: extract from zip → upload raw to bucket → delete local
#
# Remaining dirs (after removing contour + Frame_Anime):
#   Flow (20.1GB), Segment (10.4GB), UnmatchedForward (5.1GB),
#   UnmatchedBackward (5.1GB), LineArea (4.0GB), SegMatching (<1GB)
# ---------------------------------------------------------------------------
echo "=== Phase 2: AnimeRun remaining data (raw upload) ==="

for DATA_TYPE in SegMatching LineArea UnmatchedBackward UnmatchedForward Segment Flow; do
    echo "  --- AnimeRun/$DATA_TYPE ---"
    echo "  Free disk: $(df -h . | tail -1 | awk '{print $4}')"

    echo "  Extracting $DATA_TYPE..."
    unzip -q -o "$ANIMERUN_ZIP" "AnimeRun_v2/*/${DATA_TYPE}/*" -d "$ANIMERUN_DIR" \
        || echo "  WARNING: extraction had errors"

    # Upload raw to bucket under animerun_raw/{DATA_TYPE}/
    upload_raw_and_cleanup "animerun_raw/${DATA_TYPE}" "${ANIMERUN_DIR}/AnimeRun_v2"

    # Remove from zip to free space and avoid re-extracting
    echo "  Removing $DATA_TYPE from zip..."
    zip -q -d "$ANIMERUN_ZIP" "AnimeRun_v2/*/${DATA_TYPE}/*" \
        || echo "  WARNING: zip delete had errors"

    echo "  Zip size: $(du -sh "$ANIMERUN_ZIP" 2>/dev/null | cut -f1)"
    echo ""
done

# Delete the now-empty zip
echo "  Deleting emptied AnimeRun zip..."
rm -f "$ANIMERUN_ZIP"
echo "  Free disk: $(df -h . | tail -1 | awk '{print $4}')"

PHASE2_END=$(date +%s)
echo "=== Phase 2 complete at $(date) ($(( (PHASE2_END - START_TIME) / 60 )) min) ==="
echo ""

# ---------------------------------------------------------------------------
# Phase 3: FBAnimeHQ shards 08-11
# Zips extract to numbered dirs (0080/, etc.) not fbanimehq-08/.
# Fix: extract into a temp dir, point adapter at it.
# ---------------------------------------------------------------------------
echo "=== Phase 3: FBAnimeHQ shards 08-11 ==="
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

    mkdir -p "$EXTRACT_DIR"
    echo "  Extracting..."
    unzip -q -o "$ZIP_PATH" -d "$EXTRACT_DIR"

    echo "  Ingesting..."
    python3 run_ingest.py \
        --adapter fbanimehq \
        --input_dir "$EXTRACT_DIR" \
        --output_dir "./output/fbanimehq" || echo "  WARNING: adapter exited with errors"

    echo "  Cleaning up extracted source..."
    rm -rf "$EXTRACT_DIR"

    upload_and_cleanup "fbanimehq"

    echo "  ${SHARD_NAME} done at $(date)"
    echo ""
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
END_TIME=$(date +%s)
TOTAL_MIN=$(( (END_TIME - START_TIME) / 60 ))
echo ""
echo "=== All phases complete ==="
echo "    Total runtime: ${TOTAL_MIN} minutes ($(( TOTAL_MIN / 60 ))h $(( TOTAL_MIN % 60 ))m)"
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

echo "=== Bucket contents ==="
aws s3 ls "${BUCKET}/" --endpoint-url "$ENDPOINT" 2>&1
echo ""
echo "=== Done at $(date) ==="

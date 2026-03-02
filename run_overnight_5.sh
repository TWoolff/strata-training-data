#!/usr/bin/env bash
# Overnight batch #5:
# Re-run ONLY the AnimeRun correspondence adapter (fixed SegMatching path bug).
# Flow and Segment already succeeded in batch 4 — skip them.
#
# 1. Extract SegMatching + Unmatched + Frame_Anime from zip
# 2. Ingest via fixed animerun_correspondence adapter
# 3. Upload to bucket
# 4. Delete zip
#
# Usage:
#   caffeinate -dims bash run_overnight_5.sh 2>&1 | tee overnight_log_5.txt

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

START_TIME=$(date +%s)
echo "=== Overnight batch #5 started at $(date) ==="
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

ANIMERUN_DIR="./data/preprocessed/animerun"
ANIMERUN_ZIP="${ANIMERUN_DIR}/AnimeRun.zip"

if [ ! -f "$ANIMERUN_ZIP" ]; then
    echo "ERROR: AnimeRun.zip not found at $ANIMERUN_ZIP"
    exit 1
fi

echo "    Zip size: $(du -sh "$ANIMERUN_ZIP" 2>/dev/null | cut -f1)"
echo ""

# ---------------------------------------------------------------------------
# Phase 1: AnimeRun — Correspondence (SegMatching + Unmatched + Frame_Anime)
# ---------------------------------------------------------------------------
echo "=== Phase 1: AnimeRun Correspondence (SegMatching + occlusion) ==="
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"

echo "  Extracting SegMatching + UnmatchedForward + UnmatchedBackward + Frame_Anime..."
unzip -q -o "$ANIMERUN_ZIP" \
    "AnimeRun_v2/*/SegMatching/*" \
    "AnimeRun_v2/*/UnmatchedForward/*" \
    "AnimeRun_v2/*/UnmatchedBackward/*" \
    "AnimeRun_v2/*/Frame_Anime/*" \
    -d "$ANIMERUN_DIR" \
    || echo "  WARNING: extraction had errors"
echo "  Extracted. Free disk: $(df -h . | tail -1 | awk '{print $4}')"

echo "  Ingesting via animerun_correspondence adapter..."
python3 run_ingest.py \
    --adapter animerun_correspondence \
    --input_dir "${ANIMERUN_DIR}/AnimeRun_v2" \
    --output_dir "./output/animerun_correspondence" || echo "  WARNING: adapter exited with errors"

echo "  Cleaning up extracted data..."
rm -rf "${ANIMERUN_DIR}/AnimeRun_v2"
echo "  Cleaned. Free disk: $(df -h . | tail -1 | awk '{print $4}')"

upload_and_cleanup "animerun_correspondence"

# Delete the zip — all data types have now been ingested
echo "  Deleting AnimeRun zip..."
rm -f "$ANIMERUN_ZIP"
echo "  Free disk: $(df -h . | tail -1 | awk '{print $4}')"

PHASE1_END=$(date +%s)
echo "=== Phase 1 complete at $(date) ($(( (PHASE1_END - START_TIME) / 60 )) min) ==="
echo ""

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

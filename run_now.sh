#!/usr/bin/env bash
# Quick run: anime_seg + AnimeRun (no FBAnimeHQ)
# Estimated: ~1 hour
#
# Usage:
#   caffeinate -dims bash run_now.sh 2>&1 | tee overnight_log_2.txt

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
echo "=== Run started at $(date) ==="
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

# ---------------------------------------------------------------------------
# Phase 1: Anime segmentation ingest (~22K images)
# ---------------------------------------------------------------------------
echo "=== Phase 1: Anime segmentation ingest ==="
python3 run_ingest.py \
  --adapter anime_seg \
  --input_dir ./data/preprocessed/anime_segmentation \
  --output_dir ./output/anime_seg || echo "  WARNING: anime_seg adapter exited with errors"

upload_and_cleanup "anime_seg"
echo ""

# ---------------------------------------------------------------------------
# Phase 2: AnimeRun — extract only contour + Frame_Anime
# ---------------------------------------------------------------------------
echo "=== Phase 2: AnimeRun ==="
ANIMERUN_DIR="./data/preprocessed/animerun"
ANIMERUN_ZIP="${ANIMERUN_DIR}/AnimeRun.zip"

if [ ! -f "$ANIMERUN_ZIP" ]; then
    echo "  AnimeRun zip not found — skipping"
else
    echo "  Extracting only contour + Frame_Anime dirs..."
    unzip -q -o "$ANIMERUN_ZIP" "AnimeRun_v2/*/contour/*" "AnimeRun_v2/*/Frame_Anime/*" -d "$ANIMERUN_DIR" \
        || echo "  WARNING: some files may have failed to extract"
    echo "  Extracted. Free disk: $(df -h . | tail -1 | awk '{print $4}')"

    echo "  Ingesting..."
    python3 run_ingest.py \
        --adapter animerun \
        --input_dir "${ANIMERUN_DIR}/AnimeRun_v2" \
        --output_dir "./output/animerun" || echo "  WARNING: animerun adapter exited with errors"

    echo "  Cleaning up extracted data..."
    rm -rf "${ANIMERUN_DIR}/AnimeRun_v2"
    echo "  Cleaned. Free disk: $(df -h . | tail -1 | awk '{print $4}')"

    upload_and_cleanup "animerun"

    # Remove ingested dirs from zip to shrink it (~22GB → ~13GB).
    # Keeps Flow, Segment, LineArea, etc. for future use.
    echo "  Removing contour + Frame_Anime from zip (keeping remaining data)..."
    zip -q -d "$ANIMERUN_ZIP" "AnimeRun_v2/*/contour/*" "AnimeRun_v2/*/Frame_Anime/*" \
        || echo "  WARNING: zip delete had errors"
    echo "  Zip shrunk. New size: $(du -sh "$ANIMERUN_ZIP" 2>/dev/null | cut -f1)"
    echo "  Free disk: $(df -h . | tail -1 | awk '{print $4}')"
fi

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

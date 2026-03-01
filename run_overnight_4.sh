#!/usr/bin/env bash
# Overnight batch #4:
# Re-run the 3 AnimeRun adapters that failed in batch #3 because
# Frame_Anime had been stripped from the zip.
#
# 1. Flow + Frame_Anime → animerun_flow adapter → upload → delete Flow from zip
# 2. Segment + Frame_Anime → animerun_segment adapter → upload → delete Segment from zip
# 3. SegMatching + Unmatched + Frame_Anime → animerun_correspondence adapter → upload → delete from zip
# 4. Delete v1 anime_segmentation local copy (~17 GB, already in bucket)
#
# Key fix: Frame_Anime is extracted in every phase but NOT removed from the
# zip until all adapters have finished (Phase 4).
#
# Usage:
#   caffeinate -dims bash run_overnight_4.sh 2>&1 | tee overnight_log_4.txt

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
echo "=== Overnight batch #4 started at $(date) ==="
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
# Phase 1: AnimeRun — Flow (optical flow adapter)
# ---------------------------------------------------------------------------
echo "=== Phase 1: AnimeRun Flow (optical flow) ==="
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"

echo "  Extracting Flow + Frame_Anime..."
unzip -q -o "$ANIMERUN_ZIP" "AnimeRun_v2/*/Flow/*" "AnimeRun_v2/*/Frame_Anime/*" -d "$ANIMERUN_DIR" \
    || echo "  WARNING: extraction had errors"
echo "  Extracted. Free disk: $(df -h . | tail -1 | awk '{print $4}')"

echo "  Ingesting via animerun_flow adapter..."
python3 run_ingest.py \
    --adapter animerun_flow \
    --input_dir "${ANIMERUN_DIR}/AnimeRun_v2" \
    --output_dir "./output/animerun_flow" || echo "  WARNING: adapter exited with errors"

echo "  Cleaning up extracted data..."
rm -rf "${ANIMERUN_DIR}/AnimeRun_v2"
echo "  Cleaned. Free disk: $(df -h . | tail -1 | awk '{print $4}')"

upload_and_cleanup "animerun_flow"

echo "  Removing Flow from zip..."
zip -q -d "$ANIMERUN_ZIP" "AnimeRun_v2/*/Flow/*" \
    || echo "  WARNING: zip delete had errors"
echo "  Zip size: $(du -sh "$ANIMERUN_ZIP" 2>/dev/null | cut -f1)"

PHASE1_END=$(date +%s)
echo "=== Phase 1 complete at $(date) ($(( (PHASE1_END - START_TIME) / 60 )) min) ==="
echo ""

# ---------------------------------------------------------------------------
# Phase 2: AnimeRun — Segment (instance segmentation adapter)
# ---------------------------------------------------------------------------
echo "=== Phase 2: AnimeRun Segment (instance segmentation) ==="
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"

echo "  Extracting Segment + Frame_Anime..."
unzip -q -o "$ANIMERUN_ZIP" "AnimeRun_v2/*/Segment/*" "AnimeRun_v2/*/Frame_Anime/*" -d "$ANIMERUN_DIR" \
    || echo "  WARNING: extraction had errors"
echo "  Extracted. Free disk: $(df -h . | tail -1 | awk '{print $4}')"

echo "  Ingesting via animerun_segment adapter..."
python3 run_ingest.py \
    --adapter animerun_segment \
    --input_dir "${ANIMERUN_DIR}/AnimeRun_v2" \
    --output_dir "./output/animerun_segment" || echo "  WARNING: adapter exited with errors"

echo "  Cleaning up extracted data..."
rm -rf "${ANIMERUN_DIR}/AnimeRun_v2"
echo "  Cleaned. Free disk: $(df -h . | tail -1 | awk '{print $4}')"

upload_and_cleanup "animerun_segment"

echo "  Removing Segment from zip..."
zip -q -d "$ANIMERUN_ZIP" "AnimeRun_v2/*/Segment/*" \
    || echo "  WARNING: zip delete had errors"
echo "  Zip size: $(du -sh "$ANIMERUN_ZIP" 2>/dev/null | cut -f1)"

PHASE2_END=$(date +%s)
echo "=== Phase 2 complete at $(date) ($(( (PHASE2_END - PHASE1_END) / 60 )) min) ==="
echo ""

# ---------------------------------------------------------------------------
# Phase 3: AnimeRun — Correspondence (SegMatching + Unmatched + Frame_Anime)
# ---------------------------------------------------------------------------
echo "=== Phase 3: AnimeRun Correspondence (SegMatching + occlusion) ==="
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

echo "  Removing SegMatching + Unmatched + Frame_Anime from zip..."
zip -q -d "$ANIMERUN_ZIP" \
    "AnimeRun_v2/*/SegMatching/*" \
    "AnimeRun_v2/*/UnmatchedForward/*" \
    "AnimeRun_v2/*/UnmatchedBackward/*" \
    "AnimeRun_v2/*/Frame_Anime/*" \
    || echo "  WARNING: zip delete had errors"
echo "  Zip size: $(du -sh "$ANIMERUN_ZIP" 2>/dev/null | cut -f1)"

# Delete the now-empty (or near-empty) zip
echo "  Deleting emptied AnimeRun zip..."
rm -f "$ANIMERUN_ZIP"
echo "  Free disk: $(df -h . | tail -1 | awk '{print $4}')"

PHASE3_END=$(date +%s)
echo "=== Phase 3 complete at $(date) ($(( (PHASE3_END - PHASE2_END) / 60 )) min) ==="
echo ""

# ---------------------------------------------------------------------------
# Phase 4: Clean up v1 anime_segmentation (already in bucket, reclaim ~17 GB)
# ---------------------------------------------------------------------------
echo "=== Phase 4: Clean up v1 anime_segmentation ==="
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"

ANIMESEG_V1="./data/preprocessed/anime_segmentation"
if [ -d "$ANIMESEG_V1" ]; then
    local_size=$(du -sh "$ANIMESEG_V1" 2>/dev/null | cut -f1)
    echo "  Deleting $ANIMESEG_V1 ($local_size) — already uploaded to bucket..."
    rm -rf "$ANIMESEG_V1"
    echo "  Deleted. Free disk: $(df -h . | tail -1 | awk '{print $4}')"
else
    echo "  Skipping — $ANIMESEG_V1 not found"
fi

PHASE4_END=$(date +%s)
echo "=== Phase 4 complete at $(date) ($(( (PHASE4_END - PHASE3_END) / 60 )) min) ==="
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

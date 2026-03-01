#!/usr/bin/env bash
# Overnight batch #3:
# 1. Skip Phase 1 (zip already trimmed of contour + Frame_Anime)
# 2. Extract remaining AnimeRun data one type at a time:
#    - Flow → ingest via animerun_flow adapter → upload → delete from zip
#    - Segment → ingest via animerun_segment adapter → upload → delete from zip
#    - SegMatching + Unmatched → ingest via animerun_correspondence adapter → upload → delete from zip
#    - LineArea → upload raw (no adapter) → delete from zip
# 3. FBAnimeHQ shards 08-11 (extract → ingest → upload → cleanup each)
# 4. anime_seg_v2 ingest
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

ANIMERUN_DIR="./data/preprocessed/animerun"
ANIMERUN_ZIP="${ANIMERUN_DIR}/AnimeRun.zip"

# ---------------------------------------------------------------------------
# Phase 1: AnimeRun — Flow (optical flow adapter)
# Extract Flow + Frame_Anime → ingest → upload → delete from zip
# Flow is ~20GB extracted but we also need Frame_Anime for the adapter
# ---------------------------------------------------------------------------
echo "=== Phase 1: AnimeRun Flow (optical flow) ==="
echo "    Zip size: $(du -sh "$ANIMERUN_ZIP" 2>/dev/null | cut -f1)"
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
# Extract Segment + Frame_Anime → ingest → upload → delete from zip
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
# Extract all three + Frame_Anime → ingest → upload → delete from zip
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

echo "  Removing SegMatching + Unmatched from zip..."
zip -q -d "$ANIMERUN_ZIP" \
    "AnimeRun_v2/*/SegMatching/*" \
    "AnimeRun_v2/*/UnmatchedForward/*" \
    "AnimeRun_v2/*/UnmatchedBackward/*" \
    || echo "  WARNING: zip delete had errors"
echo "  Zip size: $(du -sh "$ANIMERUN_ZIP" 2>/dev/null | cut -f1)"

PHASE3_END=$(date +%s)
echo "=== Phase 3 complete at $(date) ($(( (PHASE3_END - PHASE2_END) / 60 )) min) ==="
echo ""

# ---------------------------------------------------------------------------
# Phase 4: AnimeRun — LineArea (line art adapter)
# ---------------------------------------------------------------------------
echo "=== Phase 4: AnimeRun LineArea (line art) ==="
echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"

echo "  Extracting LineArea..."
unzip -q -o "$ANIMERUN_ZIP" "AnimeRun_v2/*/LineArea/*" -d "$ANIMERUN_DIR" \
    || echo "  WARNING: extraction had errors"
echo "  Extracted. Free disk: $(df -h . | tail -1 | awk '{print $4}')"

echo "  Ingesting via animerun_linearea adapter..."
python3 run_ingest.py \
    --adapter animerun_linearea \
    --input_dir "${ANIMERUN_DIR}/AnimeRun_v2" \
    --output_dir "./output/animerun_linearea" || echo "  WARNING: adapter exited with errors"

echo "  Cleaning up extracted data..."
rm -rf "${ANIMERUN_DIR}/AnimeRun_v2"
echo "  Cleaned. Free disk: $(df -h . | tail -1 | awk '{print $4}')"

upload_and_cleanup "animerun_linearea"

echo "  Removing LineArea from zip..."
zip -q -d "$ANIMERUN_ZIP" "AnimeRun_v2/*/LineArea/*" \
    || echo "  WARNING: zip delete had errors"

# Also remove Frame_Anime from zip (extracted multiple times above, now done)
echo "  Removing Frame_Anime from zip..."
zip -q -d "$ANIMERUN_ZIP" "AnimeRun_v2/*/Frame_Anime/*" \
    || echo "  WARNING: zip delete had errors (may already be removed)"

echo "  Zip size: $(du -sh "$ANIMERUN_ZIP" 2>/dev/null | cut -f1)"

# Delete the now-empty (or near-empty) zip
echo "  Deleting emptied AnimeRun zip..."
rm -f "$ANIMERUN_ZIP"
echo "  Free disk: $(df -h . | tail -1 | awk '{print $4}')"

PHASE4_END=$(date +%s)
echo "=== Phase 4 complete at $(date) ($(( (PHASE4_END - PHASE3_END) / 60 )) min) ==="
echo ""

# ---------------------------------------------------------------------------
# Phase 5: FBAnimeHQ shards 08-11
# Zips extract to numbered dirs (0080/, etc.) not fbanimehq-08/.
# Fix: extract into a temp dir, point adapter at it.
# ---------------------------------------------------------------------------
echo "=== Phase 5: FBAnimeHQ shards 08-11 ==="
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

PHASE5_END=$(date +%s)
echo "=== Phase 5 complete at $(date) ($(( (PHASE5_END - PHASE4_END) / 60 )) min) ==="
echo ""

# ---------------------------------------------------------------------------
# Phase 6: anime_seg_v2 ingest
# ---------------------------------------------------------------------------
ANIME_SEG_V2_DIR="./data/preprocessed/anime_seg_v2"

if [ -d "$ANIME_SEG_V2_DIR" ]; then
    echo "=== Phase 6: anime_seg_v2 ingest ==="
    echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"

    python3 run_ingest.py \
        --adapter anime_seg \
        --input_dir "$ANIME_SEG_V2_DIR" \
        --output_dir "./output/anime_seg" || echo "  WARNING: adapter exited with errors"

    upload_and_cleanup "anime_seg"

    PHASE6_END=$(date +%s)
    echo "=== Phase 6 complete at $(date) ($(( (PHASE6_END - PHASE5_END) / 60 )) min) ==="
    echo ""
else
    echo "=== Phase 6: Skipping anime_seg_v2 — directory not found ==="
    PHASE6_END=$PHASE5_END
    echo ""
fi

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

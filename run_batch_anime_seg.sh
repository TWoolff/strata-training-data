#!/bin/bash
# ============================================================================
# Batch process anime-segmentation datasets (v1 + v2)
#
# Processes foreground character images from both dataset variants.
# v1 is already extracted; v2 fg zips are unzipped one at a time.
# Background images are skipped entirely.
#
# Usage:
#   ./run_batch_anime_seg.sh           # process both v1 and v2
#   ./run_batch_anime_seg.sh v1        # process only v1
#   ./run_batch_anime_seg.sh v2        # process only v2
# ============================================================================

set -euo pipefail

V1_DATA_DIR="data/preprocessed/anime_segmentation/data"
V2_DATA_DIR="data/preprocessed/anime_seg_v2"
OUTPUT_DIR="output/anime_seg"
MODE="${1:-all}"

echo "========================================"
echo "Anime-segmentation batch processing"
echo "Mode: $MODE"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""

# ---------------------------------------------------------------------------
# v1 — already extracted, multiple fg directories
# ---------------------------------------------------------------------------
if [ "$MODE" = "all" ] || [ "$MODE" = "v1" ]; then
    echo "----------------------------------------"
    echo "v1 — $(date '+%H:%M:%S')"
    echo "----------------------------------------"

    # Check if already processed
    V1_EXISTING=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "animeseg_v1_*" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$V1_EXISTING" -ge 10000 ]; then
        echo "  SKIP: ~$V1_EXISTING v1 examples already exist"
    elif [ -d "$V1_DATA_DIR" ]; then
        echo "  Running ingest + enrichment on v1 ..."
        python3 run_ingest.py \
            --adapter anime_seg \
            --input_dir "$V1_DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --enrich \
            --only_new
        echo "  Done with v1."
    else
        echo "  SKIP: $V1_DATA_DIR not found"
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# v2 — unzip fg shards one at a time, skip bg
# ---------------------------------------------------------------------------
if [ "$MODE" = "all" ] || [ "$MODE" = "v2" ]; then
    for SHARD_NUM in 01 02 03; do
        ZIP_FILE="$V2_DATA_DIR/fg-${SHARD_NUM}.zip"
        SHARD_DIR="$V2_DATA_DIR/fg-${SHARD_NUM}"

        echo "----------------------------------------"
        echo "v2 fg-$SHARD_NUM — $(date '+%H:%M:%S')"
        echo "----------------------------------------"

        if [ ! -f "$ZIP_FILE" ]; then
            echo "  SKIP: $ZIP_FILE not found"
            echo ""
            continue
        fi

        # Unzip if not already extracted
        if [ ! -d "$SHARD_DIR" ]; then
            echo "  Unzipping $ZIP_FILE ..."
            mkdir -p "$SHARD_DIR"
            unzip -q "$ZIP_FILE" -d "$SHARD_DIR"
            echo "  Unzipped."
        else
            echo "  Already extracted: $SHARD_DIR"
        fi

        # Run ingest + enrichment
        echo "  Running ingest + enrichment ..."
        python3 run_ingest.py \
            --adapter anime_seg \
            --input_dir "$SHARD_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --enrich \
            --only_new

        # Clean up
        echo "  Cleaning up extracted files ..."
        rm -rf "$SHARD_DIR"
        echo "  Deleting zip to free disk space ..."
        rm -f "$ZIP_FILE"
        echo "  Done with v2 fg-$SHARD_NUM."
        echo ""
    done
fi

echo "========================================"
echo "Batch complete — $(date '+%H:%M:%S')"
TOTAL=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "animeseg_*" 2>/dev/null | wc -l | tr -d ' ')
echo "Total examples in $OUTPUT_DIR: $TOTAL"
echo "========================================"

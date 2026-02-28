#!/bin/bash
# ============================================================================
# Batch process all FBAnimeHQ shards (00–11)
#
# For each shard: unzip → ingest + enrich → delete extracted folder.
# Skips shards that are already fully processed in the output directory.
# Safe to interrupt and re-run — picks up where it left off.
#
# Usage:
#   ./run_batch_fbanimehq.sh              # process all shards 00–11
#   ./run_batch_fbanimehq.sh 03 07        # process shards 03 through 07
# ============================================================================

set -euo pipefail

DATA_DIR="data/preprocessed/fbanimehq/data"
OUTPUT_DIR="output/fbanimehq"
START_SHARD="${1:-00}"
END_SHARD="${2:-11}"

# Pad single digits
START_SHARD=$(printf "%02d" "$((10#$START_SHARD))")
END_SHARD=$(printf "%02d" "$((10#$END_SHARD))")

echo "========================================"
echo "FBAnimeHQ batch processing"
echo "Shards: $START_SHARD → $END_SHARD"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""

for i in $(seq "$((10#$START_SHARD))" "$((10#$END_SHARD))"); do
    SHARD=$(printf "%02d" "$i")
    ZIP_FILE="$DATA_DIR/fbanimehq-${SHARD}.zip"
    SHARD_DIR="$DATA_DIR/fbanimehq-${SHARD}"

    echo "----------------------------------------"
    echo "Shard $SHARD — $(date '+%H:%M:%S')"
    echo "----------------------------------------"

    # Check zip exists
    if [ ! -f "$ZIP_FILE" ]; then
        echo "  SKIP: $ZIP_FILE not found"
        echo ""
        continue
    fi

    # Check if shard is already processed by counting examples across all
    # buckets for this shard (e.g. shard 01 = buckets 0010–0019).
    BUCKET_START=$((10#$SHARD * 10))
    EXISTING=0
    for b in $(seq "$BUCKET_START" "$((BUCKET_START + 9))"); do
        BP=$(printf "%04d" "$b")
        COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "fbanimehq_${BP}_*" 2>/dev/null | wc -l | tr -d ' ')
        EXISTING=$((EXISTING + COUNT))
    done

    if [ "$EXISTING" -ge 9000 ]; then
        echo "  SKIP: ~$EXISTING examples already exist for shard $SHARD"
        echo ""
        continue
    fi

    # Unzip into a shard-named directory (zips contain bare bucket folders
    # like 0010/, 0011/ — not nested under a parent).
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
        --adapter fbanimehq \
        --input_dir "$SHARD_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --enrich \
        --only_new

    # Clean up extracted shard and zip to save disk space
    echo "  Cleaning up extracted files ..."
    rm -rf "$SHARD_DIR"
    echo "  Deleting zip to free disk space ..."
    rm -f "$ZIP_FILE"
    echo "  Done with shard $SHARD."
    echo ""
done

echo "========================================"
echo "Batch complete — $(date '+%H:%M:%S')"
TOTAL=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "fbanimehq_*" | wc -l | tr -d ' ')
echo "Total examples in $OUTPUT_DIR: $TOTAL"
echo "========================================"

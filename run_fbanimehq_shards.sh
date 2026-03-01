#!/usr/bin/env bash
# Process FBAnimeHQ shards 08-09 (extract → ingest → cleanup extracted data)
#
# Each shard: ~2.8GB extract, ~8.6GB output. Processes one at a time
# to minimize disk usage.
#
# Usage:
#   bash run_fbanimehq_shards.sh 2>&1 | tee -a overnight_log.txt

set -euo pipefail
cd "$(dirname "$0")"

SHARD_DIR="./data/preprocessed/fbanimehq/data"
OUTPUT_DIR="./output/fbanimehq"

START_TIME=$(date +%s)
echo "=== FBAnimeHQ shard processing started at $(date) ==="
echo ""

for SHARD_NUM in 08 09; do
    SHARD_NAME="fbanimehq-${SHARD_NUM}"
    ZIP_PATH="${SHARD_DIR}/${SHARD_NAME}.zip"
    EXTRACT_DIR="${SHARD_DIR}/${SHARD_NAME}"

    if [ ! -f "$ZIP_PATH" ]; then
        echo "  Skipping ${SHARD_NAME} — zip not found"
        continue
    fi

    echo "=== Processing ${SHARD_NAME} ==="
    echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"

    # Extract
    echo "    Extracting ${ZIP_PATH}..."
    unzip -q -o "$ZIP_PATH" -d "$SHARD_DIR"

    # Ingest
    echo "    Ingesting ${SHARD_NAME}..."
    python3 run_ingest.py \
        --adapter fbanimehq \
        --input_dir "$EXTRACT_DIR" \
        --output_dir "$OUTPUT_DIR"

    # Cleanup extracted data to reclaim ~2.8GB
    echo "    Cleaning up extracted data..."
    rm -rf "$EXTRACT_DIR"

    PHASE_END=$(date +%s)
    echo "    ${SHARD_NAME} done at $(date) ($(( (PHASE_END - START_TIME) / 60 )) min elapsed)"
    echo "    Free disk: $(df -h . | tail -1 | awk '{print $4}')"
    echo ""
done

END_TIME=$(date +%s)
TOTAL_MIN=$(( (END_TIME - START_TIME) / 60 ))
echo "=== FBAnimeHQ processing complete ==="
echo "    Total runtime: ${TOTAL_MIN} minutes"
echo "    Output examples: $(ls -d ${OUTPUT_DIR}/*/ 2>/dev/null | wc -l)"
echo "    Output size: $(du -sh ${OUTPUT_DIR} 2>/dev/null | cut -f1)"
echo "=== Done at $(date) ==="

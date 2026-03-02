#!/usr/bin/env bash
# Batch process all CMU BVH clips through the degradation pipeline.
#
# For each of the 2,548 clips:
#   1. Retargets to Strata's 19-bone skeleton
#   2. Generates 7 degradation variants (training pairs)
#   3. Exports as JSON to output/animation/cmu_degraded/
#
# Expected output: ~17,836 training pair JSON files
#
# Usage:
#   ./run_cmu_batch.sh              # Process all clips
#   ./run_cmu_batch.sh --dry-run    # Count files only

set -euo pipefail

INPUT_BASE="data/mocap/cmu/data"
OUTPUT_DIR="output/animation/cmu_degraded"
LOG_FILE="output/animation/cmu_batch.log"

mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")"

if [[ "${1:-}" == "--dry-run" ]]; then
    total=$(find "$INPUT_BASE" -name "*.bvh" | wc -l | tr -d ' ')
    echo "Dry run: found $total BVH files across $(ls "$INPUT_BASE" | wc -l | tr -d ' ') subject dirs"
    echo "Expected output: ~$((total * 7)) training pair JSON files"
    exit 0
fi

echo "=== CMU BVH Batch Processing ==="
echo "Input:  $INPUT_BASE"
echo "Output: $OUTPUT_DIR"
echo "Log:    $LOG_FILE"
echo ""

processed=0
skipped=0
failed=0
total_pairs=0
start_time=$(date +%s)

# Process each subject directory
for subject_dir in "$INPUT_BASE"/*/; do
    subject=$(basename "$subject_dir")
    file_count=$(find "$subject_dir" -name "*.bvh" | wc -l | tr -d ' ')

    if [[ "$file_count" -eq 0 ]]; then
        continue
    fi

    echo -n "Subject $subject ($file_count clips)... "

    # Process all BVH files in this subject directory
    output=$(python3 -m animation.scripts.degrade_animation \
        "$subject_dir" \
        -o "$OUTPUT_DIR" \
        2>&1) || {
        echo "FAILED"
        echo "[$subject] FAILED: $output" >> "$LOG_FILE"
        failed=$((failed + file_count))
        continue
    }

    # Extract pair count from output
    pairs=$(echo "$output" | grep -o '[0-9]* training pairs' | head -1 | grep -o '[0-9]*' || echo "0")
    total_pairs=$((total_pairs + pairs))
    processed=$((processed + file_count))

    echo "OK ($pairs pairs)"
    echo "[$subject] $file_count clips → $pairs pairs" >> "$LOG_FILE"
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
elapsed_min=$((elapsed / 60))
elapsed_sec=$((elapsed % 60))

echo ""
echo "=== SUMMARY ==="
echo "Processed: $processed clips"
echo "Skipped:   $skipped clips"
echo "Failed:    $failed clips"
echo "Output:    $total_pairs training pairs in $OUTPUT_DIR"
echo "Time:      ${elapsed_min}m ${elapsed_sec}s"
echo "Log:       $LOG_FILE"

# Count actual output files
actual=$(find "$OUTPUT_DIR" -name "*.json" | wc -l | tr -d ' ')
echo "Files on disk: $actual JSON files"

echo "" >> "$LOG_FILE"
echo "=== BATCH COMPLETE ===" >> "$LOG_FILE"
echo "Processed: $processed | Skipped: $skipped | Failed: $failed | Pairs: $total_pairs" >> "$LOG_FILE"
echo "Time: ${elapsed_min}m ${elapsed_sec}s" >> "$LOG_FILE"

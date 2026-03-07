#!/usr/bin/env bash
# =============================================================================
# Pack datasets into tar archives for fast bucket transfer.
#
# Creates one .tar per dataset, uploads to hetzner:strata-training-data/tars/,
# and deletes the loose individual files from the bucket.
#
# The A100 cloud_setup.sh downloads these tars instead of 300K+ individual files,
# cutting download time from ~5 hours to ~20-30 minutes.
#
# Usage:
#   ./training/pack_datasets.sh                    # Pack + upload all
#   ./training/pack_datasets.sh segmentation live2d # Pack + upload specific datasets
#   KEEP_LOOSE=1 ./training/pack_datasets.sh       # Keep individual files in bucket
# =============================================================================
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data_cloud}"
TAR_DIR="${TAR_DIR:-./data_cloud/_tars}"
BUCKET_BASE="hetzner:strata-training-data"
BUCKET_TARS="$BUCKET_BASE/tars"
KEEP_LOOSE="${KEEP_LOOSE:-0}"

# Default datasets to pack (all lean-mode datasets)
ALL_DATASETS=(segmentation live2d humanrig anime_seg fbanimehq curated_diverse)

if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

mkdir -p "$TAR_DIR"

echo "============================================"
echo "  Pack Datasets → Tar Archives"
echo "============================================"
echo ""

PACKED_DATASETS=()

for ds in "${DATASETS[@]}"; do
    src="$DATA_DIR/$ds"
    tar_file="$TAR_DIR/${ds}.tar"

    if [ ! -d "$src" ]; then
        echo "  SKIP $ds (not found at $src)"
        continue
    fi

    count=$(find "$src" -type f | wc -l | tr -d ' ')
    size=$(du -sh "$src" 2>/dev/null | cut -f1)
    echo "  Packing $ds ($count files, $size)..."

    # Use tar without compression — PNGs and JSONs don't compress much,
    # and uncompressed tar is much faster to create and extract
    (cd "$DATA_DIR" && tar cf - "$ds") > "$tar_file"

    tar_size=$(du -sh "$tar_file" 2>/dev/null | cut -f1)
    echo "    → $tar_file ($tar_size)"
    echo ""
    PACKED_DATASETS+=("$ds")
done

echo "Uploading tar archives to bucket..."
echo ""

rclone copy "$TAR_DIR/" "$BUCKET_TARS/" \
    --transfers 8 --fast-list --size-only -P

echo ""

# Delete loose individual files from bucket (tars are the source of truth now)
if [ "$KEEP_LOOSE" = "0" ] && [ ${#PACKED_DATASETS[@]} -gt 0 ]; then
    echo "Cleaning up loose files from bucket (only for datasets we just tarred)..."
    echo ""
    for ds in "${PACKED_DATASETS[@]}"; do
        # Safety: only delete if the tar actually exists in the bucket
        if rclone lsf "$BUCKET_TARS/${ds}.tar" 2>/dev/null | grep -q "${ds}.tar"; then
            echo "  Deleting $BUCKET_BASE/$ds/ (tar verified in bucket)..."
            rclone purge "$BUCKET_BASE/$ds/" 2>/dev/null || true
        else
            echo "  SKIP deleting $ds/ — tar not found in bucket, keeping loose files"
        fi
    done
    echo ""
    echo "  Cleanup complete."
else
    echo "  KEEP_LOOSE=1 — keeping individual files in bucket."
fi

echo ""
echo "============================================"
echo "  Pack + upload complete!"
echo ""
echo "  Tar archives in bucket:"
rclone ls "$BUCKET_TARS/" 2>/dev/null | while read -r size name; do
    echo "    $name ($(echo "$size" | awk '{printf "%.1f GB", $1/1024/1024/1024}'))"
done
echo "============================================"

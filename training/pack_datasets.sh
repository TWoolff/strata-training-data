#!/usr/bin/env bash
# =============================================================================
# Pack datasets into tar archives for fast bucket transfer.
#
# Creates one .tar per dataset and uploads to hetzner:strata-training-data/tars/
# The A100 cloud_setup.sh downloads these tars instead of 300K+ individual files,
# cutting download time from ~5 hours to ~20-30 minutes.
#
# Usage:
#   ./training/pack_datasets.sh                    # Pack + upload all
#   ./training/pack_datasets.sh segmentation live2d # Pack + upload specific datasets
# =============================================================================
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data_cloud}"
TAR_DIR="${TAR_DIR:-./data_cloud/_tars}"
BUCKET="hetzner:strata-training-data/tars"

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
done

echo "Uploading tar archives to bucket..."
echo ""

rclone copy "$TAR_DIR/" "$BUCKET/" \
    --transfers 8 --fast-list --size-only -P

echo ""
echo "============================================"
echo "  Pack + upload complete!"
echo ""
echo "  Tar archives in bucket:"
rclone ls "$BUCKET/" 2>/dev/null | while read -r size name; do
    echo "    $name ($(echo "$size" | awk '{printf "%.1f GB", $1/1024/1024/1024}'))"
done
echo ""
echo "  To update cloud_setup.sh to use tars,"
echo "  the download step fetches tars/ and extracts."
echo "============================================"

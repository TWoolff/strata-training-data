#!/usr/bin/env bash
# =============================================================================
# Pack Meshy CC0 datasets as tar archives and upload to Hetzner bucket.
#
# Run this from the Mac after all 3 Meshy renders are complete:
#   ./scripts/upload_meshy_tars.sh
#
# Prerequisites:
#   - rclone configured with 'hetzner' remote (~/.config/rclone/rclone.conf)
#   - Meshy renders complete in /Volumes/TAMWoolff/data/preprocessed/
# =============================================================================
set -euo pipefail

DATA_DIR="/Volumes/TAMWoolff/data/preprocessed"
TAR_DIR="/Volumes/TAMWoolff/data/_tars"
TAR_BUCKET="hetzner:strata-training-data/tars"

echo "============================================"
echo "  Meshy CC0 — Tar Pack & Upload"
echo "  Started: $(date)"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# 0. Pre-flight checks
# ---------------------------------------------------------------------------
echo "[0/4] Pre-flight checks..."

for ds in meshy_cc0 meshy_cc0_textured meshy_cc0_unrigged; do
    if [ ! -d "$DATA_DIR/$ds" ]; then
        echo "  WARNING: $DATA_DIR/$ds not found — skipping"
    else
        count=$(find "$DATA_DIR/$ds" -name "image.png" -o -name "*_flat.png" | wc -l | tr -d ' ')
        size=$(du -sh "$DATA_DIR/$ds" 2>/dev/null | cut -f1)
        echo "  $ds: $count images ($size)"
    fi
done

rclone lsd hetzner:strata-training-data/ >/dev/null 2>&1 && echo "  Bucket connection: OK" || { echo "ERROR: Cannot connect to bucket"; exit 1; }
echo ""

# ---------------------------------------------------------------------------
# 1. Clean up withSkin files from textured run
# ---------------------------------------------------------------------------
echo "[1/4] Cleaning withSkin files from meshy_cc0_textured..."

if [ -d "$DATA_DIR/meshy_cc0_textured" ]; then
    withskin_count=$(find "$DATA_DIR/meshy_cc0_textured" -name "*withSkin*" | wc -l | tr -d ' ')
    if [ "$withskin_count" -gt 0 ]; then
        echo "  Found $withskin_count withSkin files — deleting..."
        find "$DATA_DIR/meshy_cc0_textured" -name "*withSkin*" -exec rm -rf {} + 2>/dev/null || true
        echo "  Cleaned."
    else
        echo "  No withSkin files found — clean."
    fi
fi

# Also clean macOS resource forks
find "$DATA_DIR/meshy_cc0" "$DATA_DIR/meshy_cc0_textured" "$DATA_DIR/meshy_cc0_unrigged" \
    -name "._*" -delete 2>/dev/null || true
echo "  Cleaned macOS resource forks."
echo ""

# ---------------------------------------------------------------------------
# 2. Delete dirs not needed for training (before tarring)
#    Training loaders only read: images/, masks/, depth/, normals/, joints/,
#    splits.json, sources/, and metadata.json (in per-example dirs).
#    Everything else is pipeline output we can regenerate.
# ---------------------------------------------------------------------------
echo "[2/5] Stripping unnecessary dirs to reduce tar size..."

STRIP_DIRS="draw_order contours layers measurements mesh weights"
for ds in meshy_cc0 meshy_cc0_textured; do
    if [ ! -d "$DATA_DIR/$ds" ]; then continue; fi
    for subdir in $STRIP_DIRS; do
        if [ -d "$DATA_DIR/$ds/$subdir" ]; then
            size=$(du -sh "$DATA_DIR/$ds/$subdir" 2>/dev/null | cut -f1)
            echo "  Deleting $ds/$subdir/ ($size)..."
            rm -rf "$DATA_DIR/$ds/$subdir"
        fi
    done
done

echo "  Done stripping."
echo ""

# ---------------------------------------------------------------------------
# 3. Tar pack each dataset
# ---------------------------------------------------------------------------
echo "[3/5] Packing tar archives..."
mkdir -p "$TAR_DIR"

for ds in meshy_cc0 meshy_cc0_textured meshy_cc0_unrigged; do
    if [ ! -d "$DATA_DIR/$ds" ]; then
        echo "  SKIP $ds (not found)"
        continue
    fi

    echo "  Packing $ds..."
    (cd "$DATA_DIR" && tar cf - "$ds") > "$TAR_DIR/${ds}.tar"
    tar_size=$(du -sh "$TAR_DIR/${ds}.tar" 2>/dev/null | cut -f1)
    echo "    ${ds}.tar ($tar_size)"
done
echo ""

# ---------------------------------------------------------------------------
# 3. Upload tars to bucket
# ---------------------------------------------------------------------------
echo "[4/5] Uploading tar archives to bucket..."

for ds in meshy_cc0 meshy_cc0_textured meshy_cc0_unrigged; do
    tar_file="$TAR_DIR/${ds}.tar"
    if [ ! -f "$tar_file" ]; then
        continue
    fi

    tar_size=$(du -sh "$tar_file" 2>/dev/null | cut -f1)
    echo "  Uploading ${ds}.tar ($tar_size)..."
    rclone copy "$tar_file" "$TAR_BUCKET/" \
        --transfers 8 --fast-list --size-only -P
    echo ""
done
echo ""

# ---------------------------------------------------------------------------
# 4. Verify uploads
# ---------------------------------------------------------------------------
echo "[5/5] Verifying uploads..."

for ds in meshy_cc0 meshy_cc0_textured meshy_cc0_unrigged; do
    if rclone lsf "$TAR_BUCKET/${ds}.tar" 2>/dev/null | grep -q "${ds}.tar"; then
        remote_size=$(rclone size "$TAR_BUCKET/${ds}.tar" 2>/dev/null | grep "Total size" | awk '{print $3, $4}')
        echo "  ${ds}.tar: OK ($remote_size)"
    else
        echo "  ${ds}.tar: MISSING — upload may have failed!"
    fi
done

echo ""
echo "============================================"
echo "  Upload complete!"
echo "  Finished: $(date)"
echo ""
echo "  Tar archives are at: $TAR_BUCKET/"
echo "  Local tars at: $TAR_DIR/ (safe to delete after verification)"
echo ""
echo "  Next: push code to GitHub, then spin up A100"
echo "============================================"

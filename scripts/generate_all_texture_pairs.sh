#!/usr/bin/env bash
# Generate texture pairs with multiple partial angle configurations.
# Each angle config goes to a separate output dir, then merged.
#
# Usage: ./scripts/generate_all_texture_pairs.sh

set -euo pipefail

BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
INPUT_DIR="/Volumes/TAMWoolff/data/fbx/"
BASE_OUTPUT="./output/texture_pairs"
FILTER="Meshy_AI"

echo "============================================"
echo "  Texture Pair Generation — All Angles"
echo "============================================"

# Angle configurations: (name, angles)
# Front-only already done in base output dir
CONFIGS=(
    "side:90"
    "back:180"
    "front_side:0,90"
)

# 1. Check base front-only pairs exist
N_BASE=$(ls "$BASE_OUTPUT"/ 2>/dev/null | grep -c pose || echo 0)
echo "Base (front-only) pairs: $N_BASE"

if [[ "$N_BASE" -lt 500 ]]; then
    echo "Base pairs still generating. Wait for them to finish first."
    echo "Check: ls $BASE_OUTPUT | grep -c pose"
    exit 1
fi

# 2. Generate each angle variant
for config in "${CONFIGS[@]}"; do
    NAME="${config%%:*}"
    ANGLES="${config##*:}"
    OUT_DIR="${BASE_OUTPUT}_${NAME}"

    echo ""
    echo "--- Generating $NAME (angles: $ANGLES) ---"

    "$BLENDER" --background --python scripts/batch_texture_pairs.py -- \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUT_DIR" \
        --max_chars 0 \
        --partial_angles "$ANGLES" \
        --name_filter "$FILTER" 2>&1 | \
        grep -E "INFO (Processing|Pair saved|Done|Skipping)" | tail -5

    N=$(ls "$OUT_DIR"/ 2>/dev/null | grep -c pose || echo 0)
    echo "  $NAME: $N pairs"
done

# 3. Merge all into one directory
echo ""
echo "--- Merging all variants ---"
MERGED="./output/texture_pairs_merged"
rm -rf "$MERGED"
mkdir -p "$MERGED"

# Copy base (front-only)
for d in "$BASE_OUTPUT"/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    [[ "$name" == *_pose_* ]] || continue
    cp -r "$d" "$MERGED/${name}_front"
done

# Copy each variant
for config in "${CONFIGS[@]}"; do
    NAME="${config%%:*}"
    SRC="${BASE_OUTPUT}_${NAME}"
    for d in "$SRC"/*/; do
        [ -d "$d" ] || continue
        dname=$(basename "$d")
        [[ "$dname" == *_pose_* ]] || continue
        cp -r "$d" "$MERGED/${dname}_${NAME}"
    done
done

TOTAL=$(ls "$MERGED"/ 2>/dev/null | wc -l)
echo "Total merged pairs: $TOTAL"

# 4. Tar for upload
echo ""
echo "--- Creating tar ---"
tar cf output/texture_pairs_merged.tar -C output texture_pairs_merged/
ls -lh output/texture_pairs_merged.tar

echo ""
echo "Done! Upload with:"
echo "  rclone copyto output/texture_pairs_merged.tar hetzner:strata-training-data/texture_pairs.tar --no-check-dest -P"

#!/bin/bash
# ============================================================================
# Overnight batch: run all dataset pipelines sequentially (~6 hours total)
#
# 1. FBAnimeHQ shards 01–11    (~110K images, ~3.5 hrs)
# 2. anime-segmentation v1+v2  (~25K images,  ~50 min)
# 3. Blender FBX pipeline      (61 chars × 20 poses × 6 styles, ~1.2 hrs)
#
# Usage:
#   ./run_batch_all.sh
# ============================================================================

set -euo pipefail

BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
START_TIME=$(date '+%s')

echo "========================================"
echo "Starting overnight batch — $(date '+%H:%M:%S')"
echo "Budget: ~7 hours"
echo "========================================"
echo ""

# --- 1. FBAnimeHQ (~3.5 hrs) ---
echo ">>> [1/3] FBAnimeHQ shards 01–11"
./run_batch_fbanimehq.sh 01 11
echo ""

# --- 2. anime-segmentation (~50 min) ---
echo ">>> [2/3] anime-segmentation (v1 + v2)"
./run_batch_anime_seg.sh
echo ""

# --- 3. Blender FBX pipeline (~1.2 hrs) ---
echo ">>> [3/3] Blender FBX pipeline — 61 characters × 20 poses × 6 styles"
echo "----------------------------------------"
echo "Starting Blender pipeline — $(date '+%H:%M:%S')"
echo "----------------------------------------"

if [ ! -x "$BLENDER" ] && ! command -v blender &>/dev/null; then
    echo "  SKIP: Blender not found at $BLENDER"
else
    "$BLENDER" --background --python run_pipeline.py -- \
        --input_dir ./data/fbx/ \
        --pose_dir ./data/poses/ \
        --output_dir ./output/segmentation/ \
        --styles flat,cel,pixel,painterly,sketch,unlit \
        --resolution 512 \
        --poses_per_character 20 \
        --only_new
fi

echo ""

# --- Summary ---
END_TIME=$(date '+%s')
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo "========================================"
echo "All batches complete — $(date '+%H:%M:%S')"
echo "Total elapsed: ${ELAPSED} minutes"
echo ""
echo "Output summary:"
FBANIMEHQ=$(find output/fbanimehq -maxdepth 1 -type d -name "fbanimehq_*" 2>/dev/null | wc -l | tr -d ' ')
ANIMESEG=$(find output/anime_seg -maxdepth 1 -type d -name "animeseg_*" 2>/dev/null | wc -l | tr -d ' ')
BLENDER_IMGS=$(find output/segmentation/images -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
echo "  FBAnimeHQ:       $FBANIMEHQ examples"
echo "  anime-seg:       $ANIMESEG examples"
echo "  Blender renders: $BLENDER_IMGS images"
echo "========================================"

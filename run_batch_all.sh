#!/bin/bash
# ============================================================================
# Overnight batch: run all dataset pipelines sequentially (~6 hours total)
#
# 1. FBAnimeHQ shards 07–11    (~40K remaining, ~1.5 hrs)
# 2. anime-segmentation v1+v2  (~25K images,    ~50 min)
#    → cleanup: delete anime_seg input data
# 3. Blender FBX pipeline      (61 chars × 20 poses × 6 styles, ~1.2 hrs)
# 4. AnimeRun contour pairs    (~8K frames,     ~20 min)
#    → cleanup: delete AnimeRun extracted data
#
# Deletes processed input data between stages to keep disk free.
# All inputs should be backed up externally before running.
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

# --- 1. FBAnimeHQ ---
echo ">>> [1/4] FBAnimeHQ shards 01–11"
./run_batch_fbanimehq.sh 01 11
echo ""

# --- 2. anime-segmentation ---
echo ">>> [2/4] anime-segmentation (v1 + v2)"
./run_batch_anime_seg.sh
echo ""

# Cleanup: delete anime_seg input data to free ~29GB
echo ">>> Cleanup: removing processed anime_seg input data..."
if [ -d "data/preprocessed/anime_segmentation/data" ]; then
    rm -rf data/preprocessed/anime_segmentation/data
    echo "  Deleted anime_segmentation/data (~17GB freed)"
fi
for z in data/preprocessed/anime_seg_v2/*.zip; do
    [ -f "$z" ] && rm -f "$z" && echo "  Deleted $(basename "$z")"
done
echo ""

# --- 3. Blender FBX pipeline ---
echo ">>> [3/4] Blender FBX pipeline — 61 characters × 20 poses × 6 styles"
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

# --- 4. AnimeRun contour pairs ---
ANIMERUN_ZIP="data/preprocessed/animerun/AnimeRun.zip"
ANIMERUN_EXTRACTED="data/preprocessed/animerun/extracted"

echo ">>> [4/4] AnimeRun contour pairs"
echo "----------------------------------------"
echo "Starting AnimeRun — $(date '+%H:%M:%S')"
echo "----------------------------------------"

# Extract if zip exists and extracted directory doesn't
if [ -f "$ANIMERUN_ZIP" ] && [ ! -d "$ANIMERUN_EXTRACTED" ]; then
    echo "  Unzipping AnimeRun (~22GB)..."
    mkdir -p "$ANIMERUN_EXTRACTED"
    unzip -q "$ANIMERUN_ZIP" -d "$ANIMERUN_EXTRACTED"
    echo "  Unzipped."
fi

if [ -d "$ANIMERUN_EXTRACTED" ]; then
    python3 run_ingest.py \
        --adapter animerun \
        --input_dir "$ANIMERUN_EXTRACTED" \
        --output_dir output/animerun \
        --only_new

    # Cleanup: delete extracted AnimeRun + zip
    echo "  Cleaning up AnimeRun input data..."
    rm -rf "$ANIMERUN_EXTRACTED"
    [ -f "$ANIMERUN_ZIP" ] && rm -f "$ANIMERUN_ZIP"
    echo "  Deleted AnimeRun input data (~22GB freed)"
else
    echo "  SKIP: No AnimeRun data found at $ANIMERUN_EXTRACTED or $ANIMERUN_ZIP"
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
ANIMERUN_COUNT=$(find output/animerun -maxdepth 1 -type d -name "animerun_*" 2>/dev/null | wc -l | tr -d ' ')
echo "  FBAnimeHQ:       $FBANIMEHQ examples"
echo "  anime-seg:       $ANIMESEG examples"
echo "  Blender renders: $BLENDER_IMGS images"
echo "  AnimeRun:        $ANIMERUN_COUNT contour pairs"
echo ""
echo "Disk space:"
df -h / | tail -1
echo "========================================"

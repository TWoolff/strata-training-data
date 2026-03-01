#!/usr/bin/env bash
# Overnight batch run — started 2026-03-01
#
# 1. Blender pipeline: 62 FBX chars, 5 poses, 5 angles, flat style, layers
#    Estimated: ~12 hours
# 2. Anime segmentation ingest: ~22K images
#    Estimated: ~30-45 min
#
# Disk budget: ~7 GB (Blender ~2.5 GB + anime_seg ~4.5 GB)
# Free disk at start: ~20 GB
#
# Usage:
#   caffeinate -dims bash run_overnight.sh 2>&1 | tee overnight_log.txt

set -euo pipefail
cd "$(dirname "$0")"

START_TIME=$(date +%s)
echo "=== Overnight batch started at $(date) ==="
echo ""

# ---------------------------------------------------------------------------
# Phase 1: Blender pipeline (62 chars × 5 poses × 5 angles × flat + layers)
# ---------------------------------------------------------------------------
echo "=== Phase 1: Blender pipeline ==="
echo "    62 characters, 5 poses each, 5 angles, flat style, with layers"
echo "    Started at $(date)"
echo ""

/Applications/Blender.app/Contents/MacOS/Blender --background --python run_pipeline.py -- \
  --input_dir ./data/fbx/ \
  --pose_dir ./data/poses/ \
  --output_dir ./output/segmentation/ \
  --styles flat \
  --resolution 512 \
  --angles front,three_quarter,side,three_quarter_back,back \
  --layers \
  --poses_per_character 5

PHASE1_END=$(date +%s)
echo ""
echo "=== Phase 1 complete at $(date) ($(( (PHASE1_END - START_TIME) / 60 )) min) ==="
echo ""

# ---------------------------------------------------------------------------
# Phase 2: Anime segmentation ingest (~22K images)
# ---------------------------------------------------------------------------
echo "=== Phase 2: Anime segmentation ingest ==="
echo "    Started at $(date)"
echo ""

python3 run_ingest.py \
  --adapter anime_seg \
  --input_dir ./data/preprocessed/anime_segmentation \
  --output_dir ./output/anime_seg

PHASE2_END=$(date +%s)
echo ""
echo "=== Phase 2 complete at $(date) ($(( (PHASE2_END - PHASE1_END) / 60 )) min) ==="
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL_MIN=$(( (PHASE2_END - START_TIME) / 60 ))
echo "=== All phases complete ==="
echo "    Total runtime: ${TOTAL_MIN} minutes ($(( TOTAL_MIN / 60 ))h $(( TOTAL_MIN % 60 ))m)"
echo "    Blender output: $(find ./output/segmentation/images -name '*.png' 2>/dev/null | wc -l) images"
echo "    Layers output:  $(find ./output/segmentation/layers -name '*.png' 2>/dev/null | wc -l) layers"
echo "    Anime seg:      $(find ./output/anime_seg -maxdepth 1 -type d 2>/dev/null | wc -l) examples"
echo "    Disk used:      $(du -sh ./output/segmentation/ 2>/dev/null | cut -f1) (blender) + $(du -sh ./output/anime_seg/ 2>/dev/null | cut -f1) (anime_seg)"
echo ""
echo "=== Done at $(date) ==="

#!/bin/bash
# ============================================================================
# Overnight batch: run all dataset pipelines sequentially
#
# 1. FBAnimeHQ shards 01–11  (~110K images, ~3.5 hours)
# 2. anime-segmentation v1 + v2  (~25K images, ~55 min)
#
# Usage:
#   ./run_batch_all.sh
# ============================================================================

set -euo pipefail

echo "========================================"
echo "Starting full overnight batch — $(date '+%H:%M:%S')"
echo "========================================"
echo ""

# --- FBAnimeHQ ---
echo ">>> FBAnimeHQ shards 01–11"
./run_batch_fbanimehq.sh 01 11
echo ""

# --- anime-segmentation ---
echo ">>> anime-segmentation (v1 + v2)"
./run_batch_anime_seg.sh
echo ""

echo "========================================"
echo "All batches complete — $(date '+%H:%M:%S')"
echo "========================================"

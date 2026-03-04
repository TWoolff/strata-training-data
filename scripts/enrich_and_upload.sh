#!/usr/bin/env bash
# Issue #198: Auto-chain script for RTMPose enrichment + upload
# Runs in order:
#   1. Wait for anime_seg download to finish
#   2. Re-ingest any missing anime_seg images from external HD
#   3. Enrich anime_seg with RTMPose
#   4. Wait for anime_instance_seg enrichment to finish
#   5. Upload joints.json for both datasets to Hetzner bucket

set -euo pipefail

REPO_DIR="/Users/taw/code/strata-training-data"
cd "$REPO_DIR"

# Load credentials
source .env 2>/dev/null || true
export AWS_ACCESS_KEY_ID="${BUCKET_ACCESS_KEY//\"/}"
export AWS_SECRET_ACCESS_KEY="${BUCKET_SECRET//\"/}"
ENDPOINT="https://fsn1.your-objectstorage.com"
BUCKET="s3://strata-training-data"

LOGDIR="/tmp/issue198"
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGDIR/chain.log"; }

# ─── Step 1: Wait for anime_seg download ───────────────────────────────────
log "Step 1: Waiting for anime_seg download (PID 96045)..."
DOWNLOAD_PID=96045
if kill -0 "$DOWNLOAD_PID" 2>/dev/null; then
    while kill -0 "$DOWNLOAD_PID" 2>/dev/null; do
        sleep 30
        COUNT=$(ls output/anime_seg/ 2>/dev/null | wc -l | tr -d ' ')
        log "  anime_seg download: $COUNT dirs so far"
    done
fi
ANIME_SEG_COUNT=$(ls output/anime_seg/ 2>/dev/null | wc -l | tr -d ' ')
log "Step 1 done: anime_seg download complete — $ANIME_SEG_COUNT example dirs"

# ─── Step 2: Re-ingest missing v1 images from external HD ─────────────────
log "Step 2: Checking for missing anime_seg images from external HD..."
HD_DIR="/Volumes/TAMWoolff/data/preprocessed/anime_segmentation"
if [ -d "$HD_DIR" ]; then
    # Count current v1 examples
    V1_COUNT=$(ls output/anime_seg/ 2>/dev/null | grep -c 'animeseg_v1' || true)
    log "  Currently $V1_COUNT v1 examples in output"

    # Re-ingest with only_new to add any missing
    log "  Re-ingesting from HD with --only_new..."
    python3 run_ingest.py \
        --adapter anime_seg \
        --input_dir "$HD_DIR" \
        --output_dir ./output/anime_seg \
        --only_new \
        2>&1 | tee "$LOGDIR/reingest_anime_seg.log"

    NEW_V1_COUNT=$(ls output/anime_seg/ 2>/dev/null | grep -c 'animeseg_v1' || true)
    log "  After re-ingest: $NEW_V1_COUNT v1 examples (+$((NEW_V1_COUNT - V1_COUNT)) new)"
else
    log "  External HD not mounted at $HD_DIR — skipping re-ingest"
fi

TOTAL_SEG=$(ls output/anime_seg/ 2>/dev/null | wc -l | tr -d ' ')
log "Step 2 done: $TOTAL_SEG total anime_seg examples"

# ─── Step 3: Enrich anime_seg with RTMPose ─────────────────────────────────
log "Step 3: Enriching anime_seg with RTMPose..."
python3 run_enrich.py \
    --input_dir ./output/anime_seg \
    --det_model ./models/yolox_m_humanart.onnx \
    --pose_model ./models/rtmpose_m_body7.onnx \
    --device cpu \
    --only_missing \
    2>&1 | tee "$LOGDIR/enrich_anime_seg.log"

ANIME_SEG_JOINTS=$(find output/anime_seg/ -name "joints.json" | wc -l | tr -d ' ')
log "Step 3 done: $ANIME_SEG_JOINTS anime_seg joints.json files created"

# ─── Step 4: Wait for anime_instance_seg enrichment ────────────────────────
log "Step 4: Waiting for anime_instance_seg enrichment (PID 95947)..."
ENRICH_PID=95947
if kill -0 "$ENRICH_PID" 2>/dev/null; then
    while kill -0 "$ENRICH_PID" 2>/dev/null; do
        sleep 60
        DONE=$(find output/anime_instance_seg/ -name "joints.json" | wc -l | tr -d ' ')
        log "  anime_instance_seg enrichment: $DONE/98428 joints.json"
    done
fi
ANIME_INST_JOINTS=$(find output/anime_instance_seg/ -name "joints.json" | wc -l | tr -d ' ')
log "Step 4 done: $ANIME_INST_JOINTS anime_instance_seg joints.json files"

# If enrichment was killed/incomplete, restart it
if [ "$ANIME_INST_JOINTS" -lt 98000 ]; then
    log "  Enrichment incomplete ($ANIME_INST_JOINTS < 98000), restarting..."
    python3 run_enrich.py \
        --input_dir ./output/anime_instance_seg \
        --det_model ./models/yolox_m_humanart.onnx \
        --pose_model ./models/rtmpose_m_body7.onnx \
        --device cpu \
        --only_missing \
        2>&1 | tee "$LOGDIR/enrich_anime_instance_seg_retry.log"
    ANIME_INST_JOINTS=$(find output/anime_instance_seg/ -name "joints.json" | wc -l | tr -d ' ')
    log "  After retry: $ANIME_INST_JOINTS joints.json files"
fi

# ─── Step 5: Upload joints.json to bucket ──────────────────────────────────
log "Step 5: Uploading joints.json files to Hetzner bucket..."

log "  Uploading anime_instance_seg joints..."
aws s3 sync ./output/anime_instance_seg/ "$BUCKET/anime_instance_seg/" \
    --endpoint-url "$ENDPOINT" \
    --exclude "*" --include "*/joints.json" \
    2>&1 | tail -5 | tee -a "$LOGDIR/chain.log"

log "  Uploading anime_seg joints..."
aws s3 sync ./output/anime_seg/ "$BUCKET/anime_seg/" \
    --endpoint-url "$ENDPOINT" \
    --exclude "*" --include "*/joints.json" \
    2>&1 | tail -5 | tee -a "$LOGDIR/chain.log"

# Also upload updated metadata.json (has_joints: true)
log "  Uploading updated metadata.json files..."
aws s3 sync ./output/anime_instance_seg/ "$BUCKET/anime_instance_seg/" \
    --endpoint-url "$ENDPOINT" \
    --exclude "*" --include "*/metadata.json" \
    2>&1 | tail -5 | tee -a "$LOGDIR/chain.log"

aws s3 sync ./output/anime_seg/ "$BUCKET/anime_seg/" \
    --endpoint-url "$ENDPOINT" \
    --exclude "*" --include "*/metadata.json" \
    2>&1 | tail -5 | tee -a "$LOGDIR/chain.log"

# ─── Summary ───────────────────────────────────────────────────────────────
log ""
log "============================================"
log "Issue #198 enrichment complete!"
log "============================================"
log "  anime_instance_seg joints: $ANIME_INST_JOINTS"
log "  anime_seg joints:          $ANIME_SEG_JOINTS"
log "  Total new joint data:      $((ANIME_INST_JOINTS + ANIME_SEG_JOINTS))"
log "============================================"
log ""
log "Full logs in $LOGDIR/"

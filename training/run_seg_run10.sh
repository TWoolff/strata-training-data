#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 10: Seg Bootstrap Round 3 + Back View (Model 6) (A100)
#
# Runs segmentation training followed by back view generation in one session.
#
# Seg changes from run 9:
#   - ADD flux_diverse: ~11K FLUX-generated pseudo-labeled chars (weight 4.0)
#   - BUMP gemini_diverse: weight 3.0 → 4.0 (re-labeled by run 8 model)
#   - Resume from run 8 checkpoint (0.4721 mIoU)
#   - Target: mIoU > 0.52
#
# Seg data composition:
#   - flux_diverse:        ~11,000  FLUX-generated, pseudo-labeled by run 8 (wt 4.0)
#   - cvat_annotated:          49   hand-annotated diverse illustrated chars  (wt 10.0)
#   - gemini_diverse:         854   auto-triaged pseudo-labels from run 7     (wt 4.0)
#   - gemini_li_converted:    694   Dr. Li expert labels                      (wt 3.0)
#   - vroid_cc0:            1,386   GT 22-class VRoid chars                   (wt 2.5)
#   - humanrig:            11,434   GT 22-class T-pose 3D renders             (wt 2.0)
#   - meshy_cc0_textured:  15,281   GT 22-class diverse 3D chars              (wt 1.5)
#
# Back view data:
#   - back_view_pairs:         1,085  rigged triplets (Meshy FBX + VRoid)
#   - back_view_pairs_unrigged:  ~960  unrigged triplets (Meshy GLB)
#   - Total: ~2,045 pairs
#
# Estimated: ~6-8 hrs seg + ~2-3 hrs back view on A100
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_seg_run10.sh
#   ./training/run_seg_run10.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run10_seg_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 10 Seg (Bootstrap 3)"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "[pre] Pre-flight checks..."

if [ -z "${BUCKET_ACCESS_KEY:-}" ] || [ -z "${BUCKET_SECRET:-}" ]; then
    echo "  WARNING: BUCKET_ACCESS_KEY/BUCKET_SECRET not set — upload will fail."
fi

if ! rclone lsd hetzner:strata-training-data/ &>/dev/null; then
    echo "  FAIL: rclone cannot connect to Hetzner bucket"
    exit 1
fi
echo "  OK: rclone bucket connection"

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  FAIL: CUDA not available"
    exit 1
fi
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
GPU_MEM=$(python3 -c "import torch; p=torch.cuda.get_device_properties(0); m=getattr(p,'total_memory',getattr(p,'total_mem',0)); print(f'{m/1e9:.0f}GB')")
echo "  OK: CUDA — $GPU_NAME ($GPU_MEM)"

pip install -q scipy diffusers transformers accelerate 2>/dev/null || true
echo "  Dependencies OK."
echo ""

# ---------------------------------------------------------------------------
# 0. Download run 8 checkpoint
# ---------------------------------------------------------------------------
echo "[0/5] Downloading checkpoint..."

RUN8_CKPT="checkpoints/segmentation/run8_best.pt"
mkdir -p checkpoints/segmentation

if [ -f "$RUN8_CKPT" ]; then
    echo "  run8_best.pt already exists."
else
    echo "  Downloading run 8 checkpoint (0.4721 mIoU)..."
    rclone copy hetzner:strata-training-data/checkpoints_run8_bootstrap/segmentation/best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
    if [ -f "checkpoints/segmentation/best.pt" ]; then
        cp checkpoints/segmentation/best.pt "$RUN8_CKPT"
        echo "  Saved as $RUN8_CKPT"
    else
        echo "  WARNING: No run 8 checkpoint found — will train from scratch."
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# 1. Download datasets
# ---------------------------------------------------------------------------
echo "[1/5] Downloading datasets..."
echo ""

mkdir -p ./data_cloud/tars

download_dataset() {
    local name="$1"
    local tar_name="${2:-$name}"
    local dir="./data_cloud/$name"
    local tar="./data_cloud/tars/${tar_name}.tar"

    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null | head -1)" ]; then
        echo "  $name already exists."
        return 0
    fi

    echo "  Downloading $name..."
    rclone copy "hetzner:strata-training-data/tars/${tar_name}.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "$tar" ]; then
        echo "  Extracting $name..."
        tar xf "$tar" -C ./data_cloud/
        rm -f "$tar"
    else
        echo "  FATAL: $name tar not found in bucket."
        exit 1
    fi
}

download_dataset gemini_diverse
download_dataset gemini_li_converted
download_dataset cvat_annotated
download_dataset humanrig
download_dataset vroid_cc0

# flux_diverse (key new dataset for run 10)
FD_DIR="./data_cloud/flux_diverse"
FD_TAR="./data_cloud/tars/flux_diverse.tar"
if [ -d "$FD_DIR" ] && [ "$(find "$FD_DIR" -maxdepth 2 -name 'segmentation.png' | head -1)" ]; then
    echo "  flux_diverse already exists."
else
    rm -rf "$FD_DIR"
    echo "  Downloading flux_diverse (~11K FLUX-generated chars)..."
    rclone copy "hetzner:strata-training-data/tars/flux_diverse.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "$FD_TAR" ]; then
        echo "  Extracting flux_diverse..."
        tar xf "$FD_TAR" -C ./data_cloud/
        rm -f "$FD_TAR"
        # Fix double-nesting if tar extracted as flux_diverse/flux_diverse/
        if [ -d "./data_cloud/flux_diverse/flux_diverse" ]; then
            mv ./data_cloud/flux_diverse/flux_diverse/* ./data_cloud/flux_diverse/
            rmdir ./data_cloud/flux_diverse/flux_diverse
        fi
        FD_COUNT=$(ls ./data_cloud/flux_diverse/ 2>/dev/null | wc -l | tr -d ' ')
        echo "  flux_diverse: $FD_COUNT examples"
    else
        echo "  FATAL: flux_diverse.tar not found in bucket."
        exit 1
    fi
fi

# meshy_cc0_textured (restructured tar)
MESHY_DIR="./data_cloud/meshy_cc0_textured"
MESHY_TAR="./data_cloud/tars/meshy_cc0_textured_restructured.tar"
if [ -d "$MESHY_DIR" ] && [ "$(find "$MESHY_DIR" -maxdepth 2 -name 'segmentation.png' | head -1)" ]; then
    echo "  meshy_cc0_textured already exists."
else
    echo "  Downloading meshy_cc0_textured_restructured..."
    rclone copy "hetzner:strata-training-data/tars/meshy_cc0_textured_restructured.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "$MESHY_TAR" ]; then
        echo "  Extracting meshy_cc0_textured..."
        rm -rf "$MESHY_DIR"
        tar xf "$MESHY_TAR" -C ./data_cloud/
        rm -f "$MESHY_TAR"
        if [ -d "./data_cloud/meshy_cc0_restructured" ]; then
            mv ./data_cloud/meshy_cc0_restructured "$MESHY_DIR"
        fi
    else
        echo "  FATAL: meshy_cc0_textured_restructured.tar not found."
        exit 1
    fi
fi

echo ""
echo "  Dataset summary:"
for ds in humanrig vroid_cc0 meshy_cc0_textured gemini_li_converted cvat_annotated gemini_diverse flux_diverse; do
    count=$(ls "./data_cloud/$ds/" 2>/dev/null | wc -l | tr -d ' ')
    echo "    $ds: $count examples"
done
echo ""

# ---------------------------------------------------------------------------
# 2. Marigold enrichment (depth + normals on illustrated datasets)
# ---------------------------------------------------------------------------
echo "[2/5] Marigold enrichment..."
echo ""

for ds_dir in "./data_cloud/gemini_diverse" "./data_cloud/gemini_li_converted" "./data_cloud/flux_diverse"; do
    ds_name=$(basename "$ds_dir")

    MISSING=$(find "$ds_dir" -name "image.png" -exec sh -c '
        dir=$(dirname "$1"); [ ! -f "$dir/depth.png" ] && echo m
    ' _ {} \; | wc -l | tr -d ' ')

    if [ "$MISSING" -gt 0 ]; then
        echo "  $ds_name: $MISSING missing depth — running Marigold..."
        python run_normals_enrich.py \
            --input-dir "$ds_dir" \
            --only-missing \
            --batch-size 16 \
            2>&1 | tee "$LOG_DIR/enrich_${ds_name}.log"
    else
        echo "  $ds_name: depth+normals OK."
    fi
done

echo ""

# ---------------------------------------------------------------------------
# 3. Quality filter
# ---------------------------------------------------------------------------
echo "[3/5] Quality filter..."
echo ""

for ds in humanrig vroid_cc0 meshy_cc0_textured gemini_li_converted cvat_annotated gemini_diverse flux_diverse; do
    ds_dir="./data_cloud/$ds"
    if [ -d "$ds_dir" ]; then
        rm -f "$ds_dir/quality_filter.json"
        echo "  Filtering $ds..."
        python scripts/filter_seg_quality.py \
            --data-dirs "$ds_dir" \
            --output-dir "$ds_dir" \
            --min-regions 4 \
            --max-single-region 0.70 \
            --min-foreground 0.05 \
            2>&1 | tee -a "$LOG_DIR/quality_filter.log"
    fi
done

echo "  Quality filter complete."
echo ""

# ---------------------------------------------------------------------------
# 4. Train segmentation
# ---------------------------------------------------------------------------
echo "[4/5] Training SEGMENTATION model..."
echo "  Resuming from: $RUN8_CKPT (run 8, 0.4721 mIoU)"
echo "  Config: training/configs/segmentation_a100_run10.yaml"
echo ""

RESUME_FLAG=""
if [ -f "$RUN8_CKPT" ]; then
    RESUME_FLAG="--resume $RUN8_CKPT"
fi

python -m training.train_segmentation \
    --config training/configs/segmentation_a100_run10.yaml \
    --reset-epochs \
    $RESUME_FLAG \
    2>&1 | tee "$LOG_DIR/segmentation.log"

echo ""
echo "  Segmentation training complete."
echo ""

# ---------------------------------------------------------------------------
# 5. ONNX Export + Upload
# ---------------------------------------------------------------------------
echo "[5/5] Exporting + uploading..."
echo ""

mkdir -p ./models/onnx

if [ -f "checkpoints/segmentation/best.pt" ]; then
    python -m training.export_onnx \
        --model segmentation \
        --checkpoint checkpoints/segmentation/best.pt \
        --output ./models/onnx/segmentation.onnx \
        2>&1 | tee "$LOG_DIR/export.log"
    echo "  Exported segmentation.onnx"
else
    echo "  WARNING: No seg checkpoint found for export."
fi

echo "  Uploading checkpoints, logs, ONNX..."

rclone copy ./checkpoints/segmentation/ hetzner:strata-training-data/checkpoints_run10_seg/segmentation/ \
    --transfers 32 --fast-list -P
rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run10_seg_${TIMESTAMP}/ \
    --transfers 32 --fast-list -P
if [ -f "./models/onnx/segmentation.onnx" ]; then
    rclone copy ./models/onnx/segmentation.onnx hetzner:strata-training-data/models/onnx_run10_seg/ \
        --transfers 32 --fast-list -P
fi

echo ""
echo "============================================"
echo "  Run 10 Seg complete!"
echo "  Finished: $(date)"
echo ""
echo "  Results:"
grep -E "New best mIoU|mIoU=" "$LOG_DIR/segmentation.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Run 11: Fine-tune from run 10 best — low LR, no warmup
# ---------------------------------------------------------------------------
echo "============================================"
echo "  Strata Training — Run 11 Seg (Fine-tune)"
echo "  Resuming from run 10 best checkpoint"
echo "  Config: training/configs/segmentation_a100_run11.yaml"
echo "  Started: $(date)"
echo "============================================"
echo ""

# Save run 10 best before run 11 overwrites it
if [ -f "checkpoints/segmentation/best.pt" ]; then
    cp checkpoints/segmentation/best.pt checkpoints/segmentation/run10_best.pt
    echo "  Saved run 10 best as run10_best.pt"
fi

python -m training.train_segmentation \
    --config training/configs/segmentation_a100_run11.yaml \
    --reset-epochs \
    --resume checkpoints/segmentation/run10_best.pt \
    2>&1 | tee "$LOG_DIR/segmentation_run11.log"

echo ""
echo "  Run 11 complete."
echo "  Results:"
grep -E "New best mIoU|mIoU=" "$LOG_DIR/segmentation_run11.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""

# Upload run 11 checkpoint separately
if [ -f "checkpoints/segmentation/best.pt" ]; then
    cp checkpoints/segmentation/best.pt checkpoints/segmentation/run11_best.pt
    rclone copy checkpoints/segmentation/run11_best.pt \
        hetzner:strata-training-data/checkpoints_run11_seg/segmentation/ \
        --transfers 32 --fast-list -P
    python -m training.export_onnx \
        --model segmentation \
        --checkpoint checkpoints/segmentation/run11_best.pt \
        --output ./models/onnx/segmentation_run11.onnx \
        2>&1 | tee "$LOG_DIR/export_run11.log"
    rclone copy ./models/onnx/segmentation_run11.onnx \
        hetzner:strata-training-data/models/onnx_run11_seg/ \
        --transfers 32 --fast-list -P
    echo "  Run 11 checkpoint + ONNX uploaded."
fi
echo ""

# ---------------------------------------------------------------------------
# Back View: Download data
# ---------------------------------------------------------------------------
echo "============================================"
echo "  Strata Training — Back View (Model 6)"
echo "  Started: $(date)"
echo "============================================"
echo ""

echo "[A/D] Downloading back view pairs..."

mkdir -p ./data/training/back_view_pairs ./data/tars

# Rigged pairs (1,085 triplets)
if [ -z "$(ls -A ./data/training/back_view_pairs 2>/dev/null)" ]; then
    rclone copy hetzner:strata-training-data/tars/back_view_pairs.tar ./data/tars/ \
        --transfers 16 --fast-list --size-only -P 2>&1 | tee "$LOG_DIR/bv_download.log"
    echo "  Extracting back_view_pairs.tar..."
    tar xf ./data/tars/back_view_pairs.tar -C ./data/training/
    if [ -d "./data/training/back_view_pairs_merged" ]; then
        mv ./data/training/back_view_pairs_merged/pair_* ./data/training/back_view_pairs/ 2>/dev/null || true
        rm -rf ./data/training/back_view_pairs_merged
    fi
    rm -f ./data/tars/back_view_pairs.tar
else
    echo "  back_view_pairs already exists, skipping."
fi

# Unrigged pairs (~960 additional triplets)
UNRIGGED_MARKER="./data/training/.back_view_pairs_unrigged_done"
if [ ! -f "$UNRIGGED_MARKER" ]; then
    rclone copy hetzner:strata-training-data/tars/back_view_pairs_unrigged.tar ./data/tars/ \
        --transfers 16 --fast-list --size-only -P 2>&1 | tee -a "$LOG_DIR/bv_download.log"
    echo "  Extracting back_view_pairs_unrigged.tar..."
    tar xf ./data/tars/back_view_pairs_unrigged.tar -C ./data/training/back_view_pairs/ --strip-components=1
    rm -f ./data/tars/back_view_pairs_unrigged.tar
    UNRIGGED_COUNT=$(ls -d ./data/training/back_view_pairs/pair_* 2>/dev/null | wc -l | tr -d ' ')
    echo "  After unrigged extraction: $UNRIGGED_COUNT total pairs"
    touch "$UNRIGGED_MARKER"
else
    echo "  back_view_pairs_unrigged already extracted, skipping."
fi

PAIR_COUNT=$(ls -d ./data/training/back_view_pairs/pair_* 2>/dev/null | wc -l | tr -d ' ')
echo "  Pairs: $PAIR_COUNT (~1085 rigged + ~960 unrigged)"
if [ "$PAIR_COUNT" -lt 1500 ]; then
    echo "  WARN: Expected ~2000+ pairs, got $PAIR_COUNT — unrigged may not have merged correctly"
fi
echo ""

# ---------------------------------------------------------------------------
# Back View: Train
# ---------------------------------------------------------------------------
echo "[B/D] Training back view model..."

python3 -m training.train_back_view \
    --config training/configs/back_view_a100.yaml \
    2>&1 | tee "$LOG_DIR/back_view_train.log"

echo ""

# ---------------------------------------------------------------------------
# Back View: Export ONNX
# ---------------------------------------------------------------------------
echo "[C/D] Exporting back view ONNX..."

mkdir -p ./models/onnx_back_view

python3 -m training.export_onnx \
    --model back_view \
    --checkpoint ./checkpoints/back_view/best.pt \
    --output ./models/onnx_back_view/back_view.onnx \
    2>&1 | tee "$LOG_DIR/bv_export.log"

BV_ONNX_SIZE=$(du -h ./models/onnx_back_view/back_view.onnx 2>/dev/null | cut -f1 || echo "unknown")
echo "  ONNX size: $BV_ONNX_SIZE"
echo ""

# ---------------------------------------------------------------------------
# Back View: Upload
# ---------------------------------------------------------------------------
echo "[D/D] Uploading back view checkpoints + ONNX..."

rclone copy ./checkpoints/back_view/ hetzner:strata-training-data/checkpoints_back_view_run2/ \
    --transfers 32 --fast-list -P 2>&1 | tee "$LOG_DIR/bv_upload.log"
rclone copy ./models/onnx_back_view/ hetzner:strata-training-data/models/back_view_run2/ \
    --transfers 32 --fast-list -P
rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run10_${TIMESTAMP}/ \
    --transfers 32 --fast-list -P

echo ""
echo "============================================"
echo "  Run 10 complete!"
echo "  Finished: $(date)"
echo ""
echo "  Seg results:"
grep -E "New best mIoU|mIoU=" "$LOG_DIR/segmentation.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  Back view best val/l1:"
grep -E "New best|val/l1" "$LOG_DIR/back_view_train.log" 2>/dev/null | tail -3 || echo "  (check logs)"
echo ""
echo "  To download to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_run10_seg/ ./checkpoints_run10_seg/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/onnx_run10_seg/ ./models/onnx_run10_seg/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/checkpoints_back_view_run2/ ./checkpoints_back_view_run2/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/back_view_run2/ ./models/back_view_run2/ --transfers 32 --fast-list -P"
echo "============================================"

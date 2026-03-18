#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 10: Seg Bootstrap + Back View Run 2 (A100)
#
# Part A: Segmentation bootstrap round 3
#   - NEW: flux_diverse (~1,844 FLUX-generated, pseudo-labeled by run 8 model)
#   - Resume from run 8 checkpoint (0.4721 mIoU)
#   - Target: mIoU > 0.52
#
# Part B: Back view generation run 2
#   - NEW: 720 unrigged Meshy multi-angle pairs (total 1,805 pairs)
#   - Resume from run 1 checkpoint (val/l1 = 0.2982)
#   - Target: val/l1 < 0.25
#
# Estimated: ~4-6 hrs on A100
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_10.sh
#   ./training/run_10.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run10_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 10"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "[pre] Pre-flight checks..."

PREFLIGHT_FAIL=0

if ! rclone lsd hetzner:strata-training-data/ &>/dev/null; then
    echo "  FAIL: rclone cannot connect to Hetzner bucket"
    PREFLIGHT_FAIL=1
else
    echo "  OK: rclone bucket connection"
fi

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  FAIL: CUDA not available"
    PREFLIGHT_FAIL=1
else
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEM=$(python3 -c "import torch; p=torch.cuda.get_device_properties(0); m=getattr(p,'total_memory',getattr(p,'total_mem',0)); print(f'{m/1e9:.0f}GB')")
    echo "  OK: CUDA — $GPU_NAME ($GPU_MEM)"
fi

if ! python3 -c "import torchvision" 2>/dev/null; then
    echo "  FAIL: torchvision not installed"
    PREFLIGHT_FAIL=1
else
    echo "  OK: torchvision"
fi

if [ "$PREFLIGHT_FAIL" -ne 0 ]; then
    echo ""
    echo "Pre-flight failed. Fix issues above and retry."
    exit 1
fi

pip install -q scipy diffusers transformers accelerate 2>/dev/null || true
echo "  Dependencies OK."
echo ""

# ==========================================================================
# PART A: SEGMENTATION
# ==========================================================================
echo "=========================================="
echo "  PART A: Segmentation (Bootstrap Round 3)"
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# A0. Download checkpoints
# ---------------------------------------------------------------------------
echo "[A0] Downloading seg checkpoint..."

RUN8_CKPT="checkpoints/segmentation/run8_best.pt"
if [ -f "$RUN8_CKPT" ]; then
    echo "  run8_best.pt already exists."
else
    echo "  Downloading run 8 checkpoint..."
    rclone copy hetzner:strata-training-data/checkpoints_run8_bootstrap/segmentation/best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
    if [ -f "checkpoints/segmentation/best.pt" ]; then
        cp checkpoints/segmentation/best.pt "$RUN8_CKPT"
        echo "  Saved as $RUN8_CKPT"
    else
        echo "  WARNING: No run 8 checkpoint found. Will train from scratch."
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# A1. Download datasets
# ---------------------------------------------------------------------------
echo "[A1] Downloading seg datasets..."

# flux_diverse — NEW: ~1,844 FLUX-generated pseudo-labeled chars
FD_DIR="./data_cloud/flux_diverse"
if [ -d "$FD_DIR" ] && [ "$(ls -A "$FD_DIR" 2>/dev/null | head -1)" ]; then
    echo "  flux_diverse already exists."
else
    mkdir -p "$FD_DIR"
    echo "  Downloading flux_diverse..."
    rclone copy "hetzner:strata-training-data/flux_diverse/" "$FD_DIR/" \
        --transfers 32 --fast-list -P
fi

# gemini_diverse — 874 pseudo-labeled + auto-triaged
GD_DIR="./data_cloud/gemini_diverse"
GD_TAR="./data_cloud/tars/gemini_diverse.tar"
if [ -d "$GD_DIR" ] && [ "$(ls -A "$GD_DIR" 2>/dev/null | head -1)" ]; then
    echo "  gemini_diverse already exists."
else
    mkdir -p ./data_cloud/tars
    echo "  Downloading gemini_diverse tar..."
    rclone copy "hetzner:strata-training-data/tars/gemini_diverse.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "$GD_TAR" ]; then
        echo "  Extracting gemini_diverse..."
        tar xf "$GD_TAR" -C ./data_cloud/
        rm -f "$GD_TAR"
    else
        echo "  FATAL: gemini_diverse tar not found."
        exit 1
    fi
fi

# gemini_li_converted — Dr. Li's expert annotations
LI_DIR="./data_cloud/gemini_li_converted"
LI_TAR="./data_cloud/tars/gemini_li_converted.tar"
if [ -d "$LI_DIR" ] && [ "$(ls -A "$LI_DIR" 2>/dev/null | head -1)" ]; then
    echo "  gemini_li_converted already exists."
else
    mkdir -p ./data_cloud/tars
    echo "  Downloading gemini_li_converted tar..."
    rclone copy "hetzner:strata-training-data/tars/gemini_li_converted.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "$LI_TAR" ]; then
        echo "  Extracting gemini_li_converted..."
        tar xf "$LI_TAR" -C ./data_cloud/
        rm -f "$LI_TAR"
    else
        echo "  FATAL: gemini_li_converted tar not found."
        exit 1
    fi
fi

# CVAT hand-annotated
CVAT_DIR="./data_cloud/cvat_annotated"
CVAT_TAR="./data_cloud/tars/cvat_annotated.tar"
if [ -d "$CVAT_DIR" ] && [ "$(ls -A "$CVAT_DIR" 2>/dev/null | head -1)" ]; then
    echo "  cvat_annotated already exists."
else
    mkdir -p ./data_cloud/tars
    echo "  Downloading cvat_annotated tar..."
    rclone copy "hetzner:strata-training-data/tars/cvat_annotated.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "$CVAT_TAR" ]; then
        echo "  Extracting cvat_annotated..."
        tar xf "$CVAT_TAR" -C ./data_cloud/
        rm -f "$CVAT_TAR"
    fi
fi

# Standard datasets
for ds in humanrig vroid_cc0; do
    ds_dir="./data_cloud/$ds"
    tar_file="./data_cloud/tars/${ds}.tar"
    if [ -d "$ds_dir" ] && [ "$(ls -A "$ds_dir" 2>/dev/null | head -1)" ]; then
        echo "  $ds already exists."
    else
        echo "  Downloading $ds tar..."
        mkdir -p ./data_cloud/tars
        rclone copy "hetzner:strata-training-data/tars/${ds}.tar" ./data_cloud/tars/ \
            --transfers 32 --fast-list -P
        if [ -f "$tar_file" ]; then
            tar xf "$tar_file" -C ./data_cloud/
            rm -f "$tar_file"
        else
            echo "  WARNING: Could not download $ds."
        fi
    fi
done

# meshy_cc0_textured
MESHY_RESTR_TAR="./data_cloud/tars/meshy_cc0_textured_restructured.tar"
MESHY_DIR="./data_cloud/meshy_cc0_textured"
if [ -d "$MESHY_DIR" ] && [ "$(find "$MESHY_DIR" -maxdepth 2 -name 'segmentation.png' | head -1)" ]; then
    echo "  meshy_cc0_textured already exists (restructured)."
else
    echo "  Downloading meshy_cc0_textured_restructured tar..."
    mkdir -p ./data_cloud/tars
    rclone copy "hetzner:strata-training-data/tars/meshy_cc0_textured_restructured.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "$MESHY_RESTR_TAR" ]; then
        echo "  Extracting meshy_cc0_textured_restructured..."
        rm -rf "$MESHY_DIR"
        tar xf "$MESHY_RESTR_TAR" -C ./data_cloud/
        rm -f "$MESHY_RESTR_TAR"
        if [ -d "./data_cloud/meshy_cc0_restructured" ]; then
            mv ./data_cloud/meshy_cc0_restructured "$MESHY_DIR"
        fi
    fi
fi

echo ""

# ---------------------------------------------------------------------------
# A2. Marigold enrichment (depth + normals on new datasets)
# ---------------------------------------------------------------------------
echo "[A2] Marigold enrichment..."

for ds_dir in "$FD_DIR" "$GD_DIR" "$LI_DIR"; do
    ds_name=$(basename "$ds_dir")
    MISSING_DEPTH=$(find "$ds_dir" -name "image.png" -exec sh -c '
        dir=$(dirname "$1"); [ ! -f "$dir/depth.png" ] && echo missing
    ' _ {} \; | wc -l | tr -d ' ')

    if [ "$MISSING_DEPTH" -gt 0 ]; then
        echo "  $ds_name: $MISSING_DEPTH missing depth — running Marigold..."
        python3 run_normals_enrich.py \
            --input-dir "$ds_dir" \
            --only-missing \
            --batch-size 16 \
            2>&1 | tee "$LOG_DIR/enrich_${ds_name}.log"
    else
        echo "  $ds_name: depth + normals OK."
    fi
done

echo ""

# ---------------------------------------------------------------------------
# A3. Quality filter
# ---------------------------------------------------------------------------
echo "[A3] Quality filter..."

for ds in humanrig vroid_cc0 meshy_cc0_textured gemini_li_converted cvat_annotated gemini_diverse flux_diverse; do
    ds_dir="./data_cloud/$ds"
    if [ -d "$ds_dir" ]; then
        rm -f "$ds_dir/quality_filter.json"
        echo "  Filtering $ds..."
        python3 scripts/filter_seg_quality.py \
            --data-dirs "$ds_dir" \
            --output-dir "$ds_dir" \
            --min-regions 4 \
            --max-single-region 0.70 \
            --min-foreground 0.05 \
            2>&1 | tee -a "$LOG_DIR/quality_filter.log"
    fi
done

echo ""

# ---------------------------------------------------------------------------
# A4. Train segmentation
# ---------------------------------------------------------------------------
echo "[A4] Training SEGMENTATION model..."

RESUME_CKPT="$RUN8_CKPT"
if [ ! -f "$RESUME_CKPT" ]; then
    RESUME_CKPT=""
    echo "  No resume checkpoint — training from scratch."
fi

if [ -n "$RESUME_CKPT" ]; then
    echo "  Resuming from: $RESUME_CKPT (run 8, 0.4721 mIoU)"
fi
echo "  Config: training/configs/segmentation_a100_run10.yaml"
echo ""

TRAIN_CMD="python3 -m training.train_segmentation \
    --config training/configs/segmentation_a100_run10.yaml \
    --reset-epochs"

if [ -n "$RESUME_CKPT" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME_CKPT"
fi

eval "$TRAIN_CMD" 2>&1 | tee "$LOG_DIR/segmentation.log"

echo "  Segmentation training complete."
echo ""

# ---------------------------------------------------------------------------
# A5. Export + Upload seg
# ---------------------------------------------------------------------------
echo "[A5] Exporting seg ONNX + uploading..."

ONNX_DIR="./models/onnx"
mkdir -p "$ONNX_DIR"

if [ -f "checkpoints/segmentation/best.pt" ]; then
    python3 -m training.export_onnx \
        --model segmentation \
        --checkpoint checkpoints/segmentation/best.pt \
        --output "$ONNX_DIR/segmentation.onnx" \
        2>&1 | tee "$LOG_DIR/export_seg.log"
fi

rclone copy ./checkpoints/segmentation/ hetzner:strata-training-data/checkpoints_run10/segmentation/ \
    --transfers 32 --fast-list -P
if [ -f "$ONNX_DIR/segmentation.onnx" ]; then
    rclone copy "$ONNX_DIR/segmentation.onnx" hetzner:strata-training-data/models/onnx_run10/ \
        --transfers 32 --fast-list -P
fi

echo ""

# ==========================================================================
# PART B: BACK VIEW
# ==========================================================================
echo "=========================================="
echo "  PART B: Back View (Run 2)"
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# B1. Download back view data
# ---------------------------------------------------------------------------
echo "[B1] Downloading back view data..."

mkdir -p ./data/training/back_view_pairs
mkdir -p ./data/training/back_view_pairs_unrigged

# Original rigged pairs (1,085)
BV_TAR="./data/tars/back_view_pairs.tar"
if [ -d "./data/training/back_view_pairs" ] && [ "$(ls -d ./data/training/back_view_pairs/pair_* 2>/dev/null | head -1)" ]; then
    echo "  back_view_pairs (rigged) already exists."
else
    mkdir -p ./data/tars
    rclone copy hetzner:strata-training-data/tars/back_view_pairs.tar ./data/tars/ \
        --transfers 16 --fast-list --size-only -P
    echo "  Extracting back_view_pairs..."
    tar xf "$BV_TAR" -C ./data/training/
    if [ -d "./data/training/back_view_pairs_merged" ]; then
        mv ./data/training/back_view_pairs_merged/pair_* ./data/training/back_view_pairs/ 2>/dev/null || true
        rm -rf ./data/training/back_view_pairs_merged
    fi
fi

# New unrigged pairs (720)
BV_UNRIGGED_TAR="./data/tars/back_view_pairs_unrigged.tar"
if [ -d "./data/training/back_view_pairs_unrigged" ] && [ "$(ls -d ./data/training/back_view_pairs_unrigged/pair_* 2>/dev/null | head -1)" ]; then
    echo "  back_view_pairs_unrigged already exists."
else
    mkdir -p ./data/tars
    rclone copy hetzner:strata-training-data/tars/back_view_pairs_unrigged.tar ./data/tars/ \
        --transfers 16 --fast-list --size-only -P
    if [ -f "$BV_UNRIGGED_TAR" ]; then
        echo "  Extracting back_view_pairs_unrigged..."
        tar xf "$BV_UNRIGGED_TAR" -C ./data/training/
    fi
fi

PAIR_COUNT_RIGGED=$(ls -d ./data/training/back_view_pairs/pair_* 2>/dev/null | wc -l | tr -d ' ')
PAIR_COUNT_UNRIGGED=$(ls -d ./data/training/back_view_pairs_unrigged/pair_* 2>/dev/null | wc -l | tr -d ' ')
echo "  Rigged pairs: $PAIR_COUNT_RIGGED"
echo "  Unrigged pairs: $PAIR_COUNT_UNRIGGED"
echo "  Total: $((PAIR_COUNT_RIGGED + PAIR_COUNT_UNRIGGED))"
echo ""

# ---------------------------------------------------------------------------
# B2. Download run 1 checkpoint
# ---------------------------------------------------------------------------
echo "[B2] Downloading back view checkpoint..."

BV_CKPT="checkpoints/back_view/run1_best.pt"
if [ -f "$BV_CKPT" ]; then
    echo "  run1_best.pt already exists."
else
    rclone copy hetzner:strata-training-data/checkpoints_back_view/back_view/best.pt \
        ./checkpoints/back_view/ --transfers 32 --fast-list -P
    if [ -f "checkpoints/back_view/best.pt" ]; then
        cp checkpoints/back_view/best.pt "$BV_CKPT"
        echo "  Saved as $BV_CKPT"
    else
        echo "  WARNING: No run 1 checkpoint. Training from scratch."
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# B3. Train back view
# ---------------------------------------------------------------------------
echo "[B3] Training back view model..."

BV_RESUME="$BV_CKPT"
if [ ! -f "$BV_RESUME" ]; then
    BV_RESUME=""
fi

BV_CMD="python3 -m training.train_back_view \
    --config training/configs/back_view_a100_run2.yaml"

if [ -n "$BV_RESUME" ]; then
    echo "  Resuming from: $BV_RESUME (run 1, val/l1 = 0.2982)"
    BV_CMD="$BV_CMD --resume $BV_RESUME --reset-epochs"
fi

eval "$BV_CMD" 2>&1 | tee "$LOG_DIR/back_view.log"

echo "  Back view training complete."
echo ""

# ---------------------------------------------------------------------------
# B4. Export + Upload back view
# ---------------------------------------------------------------------------
echo "[B4] Exporting back view ONNX + uploading..."

mkdir -p ./models/onnx_back_view

if [ -f "checkpoints/back_view/best.pt" ]; then
    python3 -m training.export_onnx \
        --model back_view \
        --checkpoint ./checkpoints/back_view/best.pt \
        --output ./models/onnx_back_view/back_view.onnx \
        2>&1 | tee "$LOG_DIR/export_back_view.log"
fi

rclone copy ./checkpoints/back_view/ hetzner:strata-training-data/checkpoints_run10/back_view/ \
    --transfers 32 --fast-list -P
if [ -f "./models/onnx_back_view/back_view.onnx" ]; then
    rclone copy ./models/onnx_back_view/ hetzner:strata-training-data/models/back_view_run2/ \
        --transfers 32 --fast-list -P
fi

rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run10_${TIMESTAMP}/ \
    --transfers 16 --fast-list -P

echo ""
echo "============================================"
echo "  Run 10 Complete!"
echo "  Finished: $(date)"
echo ""
echo "  Seg results:"
grep -E "Best mIoU|New best|miou" "$LOG_DIR/segmentation.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  Back view results:"
grep -E "best val|val/l1" "$LOG_DIR/back_view.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  Download to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_run10/ ./checkpoints_run10/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/onnx_run10/ ./models/onnx_run10/ --transfers 32 --fast-list -P"
echo "============================================"

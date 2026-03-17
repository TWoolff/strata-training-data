#!/usr/bin/env bash
# =============================================================================
# Strata Training — Ship Run (A100)
#
# Goal: Retrain joints + weights with latest seg model, export all 4 ONNX models
#   - Model 1 (Seg): Keep run 9 checkpoint (0.4843 mIoU) — no retraining
#   - Model 2 (Joints): Retrain from scratch with humanrig + meshy + fbanimehq + gemini
#   - Model 3 (Weights): Recompute encoder features with run 9 seg, retrain
#   - Model 4 (Inpainting): Keep run 6 checkpoint (0.0028 val/l1) — no retraining
#
# Estimated: ~4-6 hrs on A100, ~$2-3
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_ship.sh
#   ./training/run_ship.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run_ship_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Ship Run"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "[pre] Pre-flight checks..."

PREFLIGHT_FAIL=0

# rclone configured
if ! rclone lsd hetzner:strata-training-data/ &>/dev/null; then
    echo "  FAIL: rclone cannot connect to Hetzner bucket"
    PREFLIGHT_FAIL=1
else
    echo "  OK: rclone bucket connection"
fi

# CUDA available
if ! python -c "import torch; assert torch.cuda.is_available()" &>/dev/null; then
    echo "  FAIL: CUDA not available"
    PREFLIGHT_FAIL=1
else
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "  OK: CUDA available ($GPU_NAME)"
fi

# Configs exist
JOINTS_CONFIG="training/configs/joints_ship.yaml"
WEIGHTS_CONFIG="training/configs/weights_ship.yaml"

for cfg in "$JOINTS_CONFIG" "$WEIGHTS_CONFIG"; do
    if [ ! -f "$cfg" ]; then
        echo "  FAIL: Config not found: $cfg"
        PREFLIGHT_FAIL=1
    else
        echo "  OK: $cfg"
    fi
done

# Python imports
if ! python -c "from training.train_joints import *" &>/dev/null; then
    echo "  FAIL: Cannot import training.train_joints"
    PREFLIGHT_FAIL=1
else
    echo "  OK: Python training imports"
fi

# Disk space (need at least 30 GB)
FREE_GB=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "$FREE_GB" -lt 30 ]; then
    echo "  FAIL: Only ${FREE_GB}GB free disk space (need 30GB+)"
    PREFLIGHT_FAIL=1
else
    echo "  OK: ${FREE_GB}GB free disk space"
fi

if [ -z "${BUCKET_ACCESS_KEY:-}" ] || [ -z "${BUCKET_SECRET:-}" ]; then
    echo "  WARNING: BUCKET_ACCESS_KEY/BUCKET_SECRET not set."
    echo "  Upload step will fail. Set them if you want bucket upload."
fi

echo ""

if [ "$PREFLIGHT_FAIL" -ne 0 ]; then
    echo "  PRE-FLIGHT FAILED — fix issues above"
    exit 1
fi

echo "  PRE-FLIGHT PASSED"
echo ""

pip install -q scipy diffusers transformers accelerate 2>/dev/null || true

# ---------------------------------------------------------------------------
# 0. Download checkpoints
# ---------------------------------------------------------------------------
echo "[0/7] Downloading checkpoints..."
echo ""

# Run 9 seg checkpoint (0.4843 mIoU) — no retraining needed
SEG_CKPT="checkpoints/segmentation/run9_best.pt"
mkdir -p checkpoints/segmentation
if [ -f "$SEG_CKPT" ]; then
    echo "  run9_best.pt already exists."
else
    echo "  Downloading run 9 seg checkpoint..."
    rclone copy hetzner:strata-training-data/checkpoints_run9_bootstrap/segmentation/best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
    if [ -f "checkpoints/segmentation/best.pt" ]; then
        cp checkpoints/segmentation/best.pt "$SEG_CKPT"
        echo "  Saved as $SEG_CKPT"
    else
        echo "  FATAL: Run 9 seg checkpoint not found in bucket."
        exit 1
    fi
fi

# Run 6 inpainting checkpoint (0.0028 val/l1) — no retraining needed
INP_CKPT="checkpoints/inpainting/run6_best.pt"
mkdir -p checkpoints/inpainting
if [ -f "$INP_CKPT" ]; then
    echo "  run6_best.pt (inpainting) already exists."
else
    echo "  Downloading run 6 inpainting checkpoint..."
    rclone copy hetzner:strata-training-data/checkpoints_run6/inpainting/best.pt \
        ./checkpoints/inpainting/ --transfers 32 --fast-list -P
    if [ -f "checkpoints/inpainting/best.pt" ]; then
        cp checkpoints/inpainting/best.pt "$INP_CKPT"
        echo "  Saved as $INP_CKPT"
    else
        echo "  WARNING: Run 6 inpainting checkpoint not found. Will skip inpainting ONNX export."
    fi
fi

echo ""

# ---------------------------------------------------------------------------
# 1. Download datasets
# ---------------------------------------------------------------------------
echo "[1/7] Downloading datasets..."
echo ""

# Datasets needed for joints: humanrig, meshy_cc0_textured, fbanimehq, gemini_diverse, gemini_li_converted, cvat_annotated
# Datasets needed for weights: humanrig (only one with weights.json)

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

# meshy_cc0_textured — restructured tar
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
    else
        echo "  WARNING: Could not download meshy_cc0_textured_restructured."
    fi
fi

# fbanimehq — large dataset (~14 GiB)
FBAHQ_DIR="./data_cloud/fbanimehq"
FBAHQ_TAR="./data_cloud/tars/fbanimehq.tar"
if [ -d "$FBAHQ_DIR" ] && [ "$(ls -A "$FBAHQ_DIR" 2>/dev/null | head -1)" ]; then
    echo "  fbanimehq already exists."
else
    echo "  Downloading fbanimehq tar (~14 GiB, this will take a while)..."
    mkdir -p ./data_cloud/tars
    rclone copy "hetzner:strata-training-data/tars/fbanimehq.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "$FBAHQ_TAR" ]; then
        echo "  Extracting fbanimehq..."
        tar xf "$FBAHQ_TAR" -C ./data_cloud/
        rm -f "$FBAHQ_TAR"
    else
        echo "  WARNING: Could not download fbanimehq. Joints will train without it."
    fi
fi

# Smaller datasets
for ds in gemini_diverse gemini_li_converted cvat_annotated; do
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

echo ""

# ---------------------------------------------------------------------------
# 2. Verify data
# ---------------------------------------------------------------------------
echo "[2/7] Verifying downloaded data..."
echo ""

for ds in humanrig vroid_cc0 meshy_cc0_textured fbanimehq gemini_diverse gemini_li_converted cvat_annotated; do
    if [ -d "./data_cloud/$ds" ]; then
        count=$(find "./data_cloud/$ds" -type f | head -200000 | wc -l)
        size=$(du -sh "./data_cloud/$ds" 2>/dev/null | cut -f1)
        echo "  $ds: $count files ($size)"
    else
        echo "  $ds: NOT FOUND"
    fi
done

# humanrig is critical for both joints and weights
if [ ! -d "./data_cloud/humanrig" ] || [ -z "$(ls -A ./data_cloud/humanrig 2>/dev/null | head -1)" ]; then
    echo ""
    echo "  FATAL: humanrig not found. Required for both joints and weights."
    exit 1
fi
echo ""

# ---------------------------------------------------------------------------
# 3. Quality filter
# ---------------------------------------------------------------------------
echo "[3/7] Quality filter..."
echo ""

for ds in humanrig vroid_cc0 meshy_cc0_textured gemini_li_converted cvat_annotated gemini_diverse; do
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
# 4. Train joints
# ---------------------------------------------------------------------------
echo "[4/7] Training JOINTS model..."
echo ""
echo "  Config: $JOINTS_CONFIG"
echo "  Datasets: humanrig, meshy_cc0_textured, fbanimehq, gemini_diverse, gemini_li_converted, cvat_annotated"
echo ""

python -m training.train_joints \
    --config "$JOINTS_CONFIG" \
    2>&1 | tee "$LOG_DIR/joints.log"

echo ""
echo "  Joints training complete."
echo ""

# ---------------------------------------------------------------------------
# 5. Recompute encoder features + train weights
# ---------------------------------------------------------------------------
echo "[5/7] Recomputing encoder features with run 9 seg model..."
echo ""

python -m training.data.precompute_encoder_features \
    --segmentation-checkpoint "$SEG_CKPT" \
    --data-dirs ./data_cloud/humanrig \
    --output-dir ./data_cloud/encoder_features \
    --device cuda \
    2>&1 | tee "$LOG_DIR/precompute_encoder.log"

echo ""
echo "  Training WEIGHTS model..."
echo "  Config: $WEIGHTS_CONFIG"
echo ""

python -m training.train_weights \
    --config "$WEIGHTS_CONFIG" \
    2>&1 | tee "$LOG_DIR/weights.log"

echo ""
echo "  Weights training complete."
echo ""

# ---------------------------------------------------------------------------
# 6. ONNX Export (all 4 models)
# ---------------------------------------------------------------------------
echo "[6/7] Exporting all 4 models to ONNX..."
echo ""

ONNX_DIR="./models/onnx"
mkdir -p "$ONNX_DIR"

# Seg — use run 9 checkpoint directly
if [ -f "$SEG_CKPT" ]; then
    echo "  Exporting segmentation -> segmentation.onnx"
    python -m training.export_onnx \
        --model segmentation \
        --checkpoint "$SEG_CKPT" \
        --output "$ONNX_DIR/segmentation.onnx" \
        2>&1 | tee -a "$LOG_DIR/export.log"
else
    echo "  SKIP segmentation — no checkpoint"
fi

# Joints — freshly trained
if [ -f "checkpoints/joints/best.pt" ]; then
    echo "  Exporting joints -> joint_refinement.onnx"
    python -m training.export_onnx \
        --model joints \
        --checkpoint checkpoints/joints/best.pt \
        --output "$ONNX_DIR/joint_refinement.onnx" \
        2>&1 | tee -a "$LOG_DIR/export.log"
else
    echo "  SKIP joints — no checkpoint"
fi

# Weights — freshly trained
if [ -f "checkpoints/weights/best.pt" ]; then
    echo "  Exporting weights -> weight_prediction.onnx"
    python -m training.export_onnx \
        --model weights_vertex \
        --checkpoint checkpoints/weights/best.pt \
        --output "$ONNX_DIR/weight_prediction.onnx" \
        2>&1 | tee -a "$LOG_DIR/export.log"
else
    echo "  SKIP weights — no checkpoint"
fi

# Inpainting — use run 6 checkpoint
if [ -f "$INP_CKPT" ]; then
    echo "  Exporting inpainting -> inpainting.onnx"
    python -m training.export_onnx \
        --model inpainting \
        --checkpoint "$INP_CKPT" \
        --output "$ONNX_DIR/inpainting.onnx" \
        2>&1 | tee -a "$LOG_DIR/export.log"
elif [ -f "checkpoints/inpainting/best.pt" ]; then
    echo "  Exporting inpainting -> inpainting.onnx (from best.pt)"
    python -m training.export_onnx \
        --model inpainting \
        --checkpoint checkpoints/inpainting/best.pt \
        --output "$ONNX_DIR/inpainting.onnx" \
        2>&1 | tee -a "$LOG_DIR/export.log"
else
    echo "  SKIP inpainting — no checkpoint"
fi

echo ""
echo "  ONNX models:"
ls -lh "$ONNX_DIR/"*.onnx 2>/dev/null || echo "  (no ONNX files)"
echo ""

# ---------------------------------------------------------------------------
# 7. Upload everything to bucket
# ---------------------------------------------------------------------------
echo "[7/7] Uploading checkpoints, logs, ONNX models..."
echo ""

rclone copy ./checkpoints/ hetzner:strata-training-data/checkpoints_ship/ \
    --transfers 32 --fast-list -P
rclone copy ./logs/ hetzner:strata-training-data/logs/ \
    --transfers 32 --fast-list -P
if [ -d "$ONNX_DIR" ] && [ "$(ls -A "$ONNX_DIR" 2>/dev/null)" ]; then
    rclone copy "$ONNX_DIR/" hetzner:strata-training-data/models/onnx_ship/ \
        --transfers 32 --fast-list -P
fi

echo ""
echo "============================================"
echo "  Ship Run complete!"
echo "  Finished: $(date)"
echo ""
echo "  ONNX models:"
ls -lh "$ONNX_DIR/"*.onnx 2>/dev/null || echo "  (none)"
echo ""
echo "  Checkpoints:"
ls -lh checkpoints/*/best.pt 2>/dev/null || echo "  (none)"
echo ""
echo "  Results:"
echo "    Seg: run 9, 0.4843 mIoU (not retrained)"
grep -E "Best|best|mean_offset" "$LOG_DIR/joints.log" 2>/dev/null | tail -3 || echo "    Joints: (check logs)"
grep -E "Best|best|mae" "$LOG_DIR/weights.log" 2>/dev/null | tail -3 || echo "    Weights: (check logs)"
echo "    Inpainting: run 6, 0.0028 val/l1 (not retrained)"
echo ""
echo "  To download ONNX models to Mac:"
echo "    rclone copy hetzner:strata-training-data/models/onnx_ship/ ./models/onnx_ship/ --transfers 32 --fast-list -P"
echo ""
echo "  To copy ONNX to Strata app:"
echo "    cp ./models/onnx_ship/*.onnx ../strata/src-tauri/models/"
echo "============================================"

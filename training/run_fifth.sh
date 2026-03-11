#!/usr/bin/env bash
# =============================================================================
# Strata Training — Fifth Run (A100)
#
# Goal: Improved seg model + retrain inpainting
#   - Model 1 (Seg): Fine-tune from run 4 checkpoint with expanded Gemini data
#     + VRoid CC0 multi-pose + cleaner pseudo-labels
#   - Model 2 (Joints): Keep run 3 checkpoint (0.001206) — no retraining
#   - Model 3 (Weights): Keep run 3 checkpoint (0.023 MAE) — no retraining
#   - Model 4 (Inpainting): Retrain with more diverse occlusion pairs
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_fifth.sh
#   ./training/run_fifth.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run5_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Fifth Run"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# =============================================================================
# PRE-FLIGHT CHECKS — fail fast before wasting GPU time
# =============================================================================
echo "=========================================="
echo "  PRE-FLIGHT CHECKS"
echo "=========================================="
echo ""

PREFLIGHT_FAIL=0

# --- Check 1: rclone configured ---
if ! rclone lsd hetzner:strata-training-data/ &>/dev/null; then
    echo "  FAIL: rclone cannot connect to Hetzner bucket"
    echo "        Run ./training/cloud_setup.sh lean first"
    PREFLIGHT_FAIL=1
else
    echo "  OK: rclone bucket connection"
fi

# --- Check 2: CUDA available ---
if ! python -c "import torch; assert torch.cuda.is_available()" &>/dev/null; then
    echo "  FAIL: CUDA not available"
    PREFLIGHT_FAIL=1
else
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "  OK: CUDA available ($GPU_NAME)"
fi

# --- Check 3: Resume checkpoint exists ---
RUN4_CKPT="checkpoints/segmentation/run4_best.pt"
if [ ! -f "$RUN4_CKPT" ]; then
    echo "  WARN: $RUN4_CKPT not found, will download from bucket"
    echo "        Downloading run 4 seg checkpoint..."
    mkdir -p checkpoints/segmentation
    rclone copy hetzner:strata-training-data/checkpoints_run4/segmentation/best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
    if [ -f "checkpoints/segmentation/best.pt" ]; then
        cp checkpoints/segmentation/best.pt "$RUN4_CKPT"
        echo "  OK: Downloaded and saved as $RUN4_CKPT"
    else
        echo "  FAIL: Could not download run 4 seg checkpoint from bucket"
        echo "        Checked: checkpoints_run4/segmentation/best.pt"
        PREFLIGHT_FAIL=1
    fi
else
    echo "  OK: Resume checkpoint exists ($RUN4_CKPT)"
fi

# --- Check 4: Run 3 joints + weights checkpoints ---
for model_dir in joints weights; do
    ckpt="checkpoints/$model_dir/best.pt"
    if [ ! -f "$ckpt" ]; then
        echo "  WARN: $ckpt not found, downloading from bucket..."
        mkdir -p "checkpoints/$model_dir"
        rclone copy "hetzner:strata-training-data/checkpoints_run3/$model_dir/" \
            "./checkpoints/$model_dir/" --transfers 32 --fast-list -P
        if [ -f "$ckpt" ]; then
            echo "  OK: Downloaded $ckpt"
        else
            echo "  WARN: Could not download $ckpt (non-fatal, we're not retraining)"
        fi
    else
        echo "  OK: $ckpt exists"
    fi
done

# --- Check 5: Training data directories exist and have content ---
REQUIRED_DATASETS="humanrig meshy_cc0 meshy_cc0_textured anime_seg fbanimehq gemini_diverse"
OPTIONAL_DATASETS="unirig meshy_cc0_unrigged vroid_cc0"

for ds in $REQUIRED_DATASETS; do
    ds_dir="./data_cloud/$ds"
    if [ ! -d "$ds_dir" ] || [ -z "$(ls -A "$ds_dir" 2>/dev/null | head -1)" ]; then
        echo "  FAIL: Required dataset missing or empty: $ds_dir"
        PREFLIGHT_FAIL=1
    else
        count=$(find "$ds_dir" -name "*.png" -o -name "*.json" | head -1000 | wc -l)
        echo "  OK: $ds ($count+ files)"
    fi
done

for ds in $OPTIONAL_DATASETS; do
    ds_dir="./data_cloud/$ds"
    if [ ! -d "$ds_dir" ] || [ -z "$(ls -A "$ds_dir" 2>/dev/null | head -1)" ]; then
        echo "  SKIP: Optional dataset not present: $ds (will continue without it)"
    else
        count=$(find "$ds_dir" -name "*.png" -o -name "*.json" | head -1000 | wc -l)
        echo "  OK: $ds ($count+ files)"
    fi
done

# --- Check 6: Training config exists ---
SEG_CONFIG="training/configs/segmentation_a100_run5.yaml"
if [ ! -f "$SEG_CONFIG" ]; then
    echo "  FAIL: Seg config not found: $SEG_CONFIG"
    PREFLIGHT_FAIL=1
else
    echo "  OK: Seg config exists ($SEG_CONFIG)"
fi

INP_CONFIG="training/configs/inpainting_a100_lean.yaml"
if [ ! -f "$INP_CONFIG" ]; then
    echo "  FAIL: Inpainting config not found: $INP_CONFIG"
    PREFLIGHT_FAIL=1
else
    echo "  OK: Inpainting config exists ($INP_CONFIG)"
fi

# --- Check 7: Python imports work ---
if ! python -c "from training.train_segmentation import *" &>/dev/null; then
    echo "  FAIL: Cannot import training.train_segmentation"
    PREFLIGHT_FAIL=1
else
    echo "  OK: Python training imports"
fi

# --- Check 8: Disk space (need at least 20 GB free) ---
FREE_GB=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "$FREE_GB" -lt 20 ]; then
    echo "  FAIL: Only ${FREE_GB}GB free disk space (need 20GB+)"
    PREFLIGHT_FAIL=1
else
    echo "  OK: ${FREE_GB}GB free disk space"
fi

# --- Check 9: Verify resume checkpoint loads ---
if [ -f "$RUN4_CKPT" ]; then
    if python -c "
import torch
ckpt = torch.load('$RUN4_CKPT', map_location='cpu', weights_only=False)
assert 'model_state_dict' in ckpt, 'No model_state_dict in checkpoint'
assert 'epoch' in ckpt, 'No epoch in checkpoint'
epoch = ckpt['epoch']
miou = ckpt.get('metrics', {}).get('val/miou', 'unknown')
print(f'Checkpoint OK: epoch {epoch}, mIoU {miou}')
" 2>/dev/null; then
        echo "  OK: Resume checkpoint is valid"
    else
        echo "  FAIL: Resume checkpoint is corrupted or incompatible"
        PREFLIGHT_FAIL=1
    fi
fi

echo ""

# --- Final verdict ---
if [ "$PREFLIGHT_FAIL" -ne 0 ]; then
    echo "=========================================="
    echo "  PRE-FLIGHT FAILED — fix issues above"
    echo "=========================================="
    exit 1
fi

echo "=========================================="
echo "  PRE-FLIGHT PASSED — starting training"
echo "=========================================="
echo ""

# =============================================================================
# STEP 1: Quality filter on seg masks
# =============================================================================
echo "[1/7] Running segmentation quality filter..."
echo ""

for ds in humanrig unirig meshy_cc0 meshy_cc0_textured gemini_diverse vroid_cc0; do
    ds_dir="./data_cloud/$ds"
    if [ -d "$ds_dir" ] && [ ! -f "$ds_dir/quality_filter.json" ]; then
        echo "  Filtering $ds..."
        python scripts/filter_seg_quality.py \
            --data-dirs "$ds_dir" \
            --output-dir "$ds_dir" \
            --min-regions 4 \
            --max-single-region 0.70 \
            --min-foreground 0.05 \
            2>&1 | tee -a "$LOG_DIR/quality_filter.log"
    else
        echo "  $ds: already filtered or not present, skipping."
    fi
done

echo ""
echo "  Quality filter complete."
echo ""

# =============================================================================
# STEP 2: Normals + depth enrichment for datasets missing them
# =============================================================================
echo "[2/7] Enriching datasets with surface normals + depth (Marigold)..."
echo ""

pip install -q diffusers transformers accelerate 2>/dev/null

for ds in gemini_diverse vroid_cc0; do
    if [ -d "./data_cloud/$ds" ]; then
        echo "  Enriching $ds..."
        python run_normals_enrich.py \
            --input-dir "./data_cloud/$ds" \
            --only-missing \
            --batch-size 16 \
            2>&1 | tee "$LOG_DIR/enrich_normals_${ds}.log"
        echo ""
    fi
done

echo "  Enrichment complete."
echo ""

# =============================================================================
# STEP 3: Train segmentation (resume from run 4)
# =============================================================================
echo "[3/7] Training SEGMENTATION model (resume from run 4)..."
echo ""
echo "  Resuming from: $RUN4_CKPT"
echo "  Config: $SEG_CONFIG"
echo ""

python -m training.train_segmentation \
    --config "$SEG_CONFIG" \
    --resume "$RUN4_CKPT" \
    2>&1 | tee "$LOG_DIR/segmentation.log"

# Verify training actually produced a checkpoint
if [ ! -f "checkpoints/segmentation/best.pt" ]; then
    echo "  ERROR: No best.pt produced by segmentation training!"
    exit 1
fi

echo ""
echo "  Segmentation training complete."
echo ""

# =============================================================================
# STEP 4: Generate inpainting pairs + train inpainting
# =============================================================================
echo "[4/7] Generating inpainting pairs + training INPAINTING model..."
echo ""

PAIRS_DIR="./data_cloud/inpainting_pairs"
PAIRS_COUNT=$(find "$PAIRS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -100 | wc -l)

if [ "$PAIRS_COUNT" -ge 100 ]; then
    echo "  Occlusion pairs already exist ($PAIRS_COUNT+ dirs), skipping generation."
else
    python -m training.data.generate_occlusion_pairs \
        --source-dirs \
            ./data_cloud/humanrig \
            ./data_cloud/gemini_diverse \
        --output-dir "$PAIRS_DIR" \
        --max-images 15000 \
        --masks-per-image 3 \
        2>&1 | tee "$LOG_DIR/generate_inpainting_pairs.log"
fi

echo ""
echo "  Training INPAINTING model..."
echo ""

python -m training.train_inpainting \
    --config "$INP_CONFIG" \
    2>&1 | tee "$LOG_DIR/inpainting.log"

echo ""
echo "  Inpainting training complete."
echo ""

# =============================================================================
# STEP 5: ONNX Export (all 4 models)
# =============================================================================
echo "[5/7] Exporting all models to ONNX..."
echo ""

ONNX_DIR="./models/onnx"
mkdir -p "$ONNX_DIR"

for model_export in \
    "segmentation checkpoints/segmentation/best.pt segmentation.onnx" \
    "joints checkpoints/joints/best.pt joint_refinement.onnx" \
    "weights_vertex checkpoints/weights/best.pt weight_prediction.onnx" \
    "inpainting checkpoints/inpainting/best.pt inpainting.onnx"
do
    set -- $model_export
    model_name=$1 ckpt=$2 onnx_file=$3
    if [ -f "$ckpt" ]; then
        echo "  Exporting $model_name -> $onnx_file"
        python -m training.export_onnx \
            --model "$model_name" \
            --checkpoint "$ckpt" \
            --output "$ONNX_DIR/$onnx_file" \
            2>&1 | tee -a "$LOG_DIR/export.log"
    else
        echo "  SKIP $model_name — no checkpoint at $ckpt"
    fi
done

echo ""

# =============================================================================
# STEP 6: Re-enrich Gemini data with run 5 model (bootstrap for run 6)
# =============================================================================
echo "[6/7] Re-enriching Gemini data with updated seg model..."
echo ""

if [ -d "./data_cloud/gemini_diverse" ]; then
    python run_seg_enrich.py \
        --checkpoint checkpoints/segmentation/best.pt \
        --input-dir ./data_cloud/gemini_diverse \
        --only-missing \
        2>&1 | tee "$LOG_DIR/seg_enrich_gemini.log"
    echo "  Re-enrichment complete."
else
    echo "  No gemini_diverse dir, skipping."
fi
echo ""

# =============================================================================
# STEP 7: Upload everything to bucket
# =============================================================================
echo "[7/7] Uploading checkpoints, logs, and ONNX models..."
echo ""

rclone copy ./checkpoints/ hetzner:strata-training-data/checkpoints_run5/ \
    --transfers 32 --fast-list -P
rclone copy ./logs/ hetzner:strata-training-data/logs/ \
    --transfers 32 --fast-list -P
rclone copy ./models/onnx/ hetzner:strata-training-data/models/onnx_run5/ \
    --transfers 32 --fast-list -P

# Upload re-enriched Gemini data
if [ -d "./data_cloud/gemini_diverse" ]; then
    echo "  Uploading re-enriched gemini_diverse..."
    rclone copy ./data_cloud/gemini_diverse/ hetzner:strata-training-data/gemini_diverse/ \
        --transfers 32 --checkers 64 --fast-list --size-only -P
fi

echo ""
echo "============================================"
echo "  Fifth run complete!"
echo "  Finished: $(date)"
echo ""
echo "  ONNX models:"
ls -lh "$ONNX_DIR/"*.onnx 2>/dev/null || echo "  (no ONNX files found)"
echo ""
echo "  Checkpoints:"
ls -lh checkpoints/*/best.pt 2>/dev/null || echo "  (no checkpoints found)"
echo ""
echo "  To download results to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_run5/ ./checkpoints/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/onnx_run5/ ./models/onnx/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/logs/ ./logs/ --transfers 32 --fast-list -P"
echo "============================================"

#!/usr/bin/env bash
# =============================================================================
# Strata Training — Fifth Run (A100)
#
# Goal: Ground-truth joints upgrade + seg improvement + encoder features refresh
#   - Model 1 (Seg): Resume from run 4, add expanded Gemini data
#   - Model 2 (Joints): Retrain with 45K new posed ground-truth examples
#   - Model 3 (Weights): Recompute encoder features with new seg, retrain
#   - Model 4 (Inpainting): Keep run 4 checkpoint — no retraining needed
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

# --- Check 3: Seg config exists ---
SEG_CONFIG="training/configs/segmentation_a100_run5.yaml"
JOINTS_CONFIG="training/configs/joints_a100_run5.yaml"
INP_CONFIG="training/configs/inpainting_a100_lean.yaml"
WEIGHTS_CONFIG="training/configs/weights_a100_lean.yaml"

for cfg in "$SEG_CONFIG" "$JOINTS_CONFIG" "$INP_CONFIG" "$WEIGHTS_CONFIG"; do
    if [ ! -f "$cfg" ]; then
        echo "  FAIL: Config not found: $cfg"
        PREFLIGHT_FAIL=1
    else
        echo "  OK: $cfg"
    fi
done

# --- Check 4: Python imports work ---
if ! python -c "from training.train_segmentation import *" &>/dev/null; then
    echo "  FAIL: Cannot import training.train_segmentation"
    PREFLIGHT_FAIL=1
else
    echo "  OK: Python training imports"
fi

# --- Check 5: Disk space (need at least 30 GB free for encoder features + checkpoints) ---
FREE_GB=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "$FREE_GB" -lt 30 ]; then
    echo "  FAIL: Only ${FREE_GB}GB free disk space (need 30GB+)"
    PREFLIGHT_FAIL=1
else
    echo "  OK: ${FREE_GB}GB free disk space"
fi

echo ""

if [ "$PREFLIGHT_FAIL" -ne 0 ]; then
    echo "=========================================="
    echo "  PRE-FLIGHT FAILED — fix issues above"
    echo "=========================================="
    exit 1
fi

echo "=========================================="
echo "  PRE-FLIGHT PASSED — starting setup"
echo "=========================================="
echo ""

# =============================================================================
# STEP 0: Download checkpoints + humanrig_posed dataset
# =============================================================================
echo "[0/8] Downloading run 4 checkpoints + humanrig_posed..."
echo ""

# Run 4 seg checkpoint (to resume from)
RUN4_SEG_CKPT="checkpoints/segmentation/run4_best.pt"
if [ -f "$RUN4_SEG_CKPT" ]; then
    echo "  run4_best.pt already exists."
else
    mkdir -p checkpoints/segmentation
    rclone copy hetzner:strata-training-data/checkpoints_run4/segmentation/best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
    if [ -f "checkpoints/segmentation/best.pt" ]; then
        cp checkpoints/segmentation/best.pt "$RUN4_SEG_CKPT"
        echo "  Saved as $RUN4_SEG_CKPT"
    else
        echo "  WARNING: Run 4 seg checkpoint not found. Trying run 1..."
        rclone copy hetzner:strata-training-data/checkpoints_run1/segmentation/best.pt \
            ./checkpoints/segmentation/ --transfers 32 --fast-list -P
        if [ -f "checkpoints/segmentation/best.pt" ]; then
            cp checkpoints/segmentation/best.pt "$RUN4_SEG_CKPT"
            echo "  Fell back to run 1 checkpoint."
        else
            echo "  FATAL: No segmentation checkpoint found."
            exit 1
        fi
    fi
fi

# Run 4 inpainting checkpoint (keep as-is, not retraining)
echo "  Downloading run 4 inpainting checkpoint..."
mkdir -p checkpoints/inpainting
rclone copy hetzner:strata-training-data/checkpoints_run4/inpainting/ \
    ./checkpoints/inpainting/ --transfers 32 --fast-list -P 2>/dev/null || \
    echo "  (no inpainting checkpoint found — will skip ONNX export for inpainting)"

# Download humanrig_posed
echo ""
echo "  Downloading humanrig_posed dataset..."
if [ -d "./data_cloud/humanrig_posed" ] && [ "$(ls -A ./data_cloud/humanrig_posed 2>/dev/null | head -1)" ]; then
    echo "  humanrig_posed already exists, skipping."
else
    mkdir -p ./data_cloud/_tars
    rclone copy hetzner:strata-training-data/tars/humanrig_posed.tar \
        ./data_cloud/_tars/ --transfers 32 --fast-list -P
    if [ -f "./data_cloud/_tars/humanrig_posed.tar" ]; then
        echo "  Extracting humanrig_posed.tar..."
        tar xf ./data_cloud/_tars/humanrig_posed.tar -C ./data_cloud/
        rm -f ./data_cloud/_tars/humanrig_posed.tar
    else
        echo "  Tar not found. Trying loose files..."
        rclone copy hetzner:strata-training-data/humanrig_posed/ \
            ./data_cloud/humanrig_posed/ --transfers 32 --checkers 64 \
            --fast-list --size-only -P
    fi
    rmdir ./data_cloud/_tars 2>/dev/null || true
fi

# Download expanded Gemini diverse (prefer tar, fall back to loose files)
echo ""
echo "  Downloading expanded gemini_diverse..."
if [ -d "./data_cloud/gemini_diverse" ] && [ "$(ls -A ./data_cloud/gemini_diverse 2>/dev/null | head -1)" ]; then
    echo "  gemini_diverse already exists, skipping."
else
    mkdir -p ./data_cloud/_tars
    rclone copy hetzner:strata-training-data/tars/gemini_diverse.tar \
        ./data_cloud/_tars/ --transfers 32 --fast-list -P
    if [ -f "./data_cloud/_tars/gemini_diverse.tar" ]; then
        echo "  Extracting gemini_diverse.tar..."
        tar xf ./data_cloud/_tars/gemini_diverse.tar -C ./data_cloud/
        rm -f ./data_cloud/_tars/gemini_diverse.tar
    else
        echo "  Tar not found. Trying loose files..."
        rclone copy hetzner:strata-training-data/gemini_diverse/ \
            ./data_cloud/gemini_diverse/ --transfers 32 --checkers 64 \
            --fast-list --size-only -P
    fi
    rmdir ./data_cloud/_tars 2>/dev/null || true
fi

echo ""

# =============================================================================
# STEP 1: Verify data
# =============================================================================
echo "[1/8] Verifying downloaded data..."
echo ""

for ds in humanrig humanrig_posed meshy_cc0 meshy_cc0_textured anime_seg fbanimehq gemini_diverse; do
    if [ -d "./data_cloud/$ds" ]; then
        count=$(find "./data_cloud/$ds" -type f | head -200000 | wc -l)
        size=$(du -sh "./data_cloud/$ds" 2>/dev/null | cut -f1)
        echo "  $ds: $count files ($size)"
    else
        echo "  $ds: NOT FOUND"
    fi
done

# humanrig_posed is the whole point of run 5
if [ ! -d "./data_cloud/humanrig_posed" ] || [ -z "$(ls -A ./data_cloud/humanrig_posed 2>/dev/null | head -1)" ]; then
    echo ""
    echo "  FATAL: humanrig_posed not found. This is the core new dataset for run 5."
    echo "  Upload first: rclone copy /path/to/humanrig_posed hetzner:strata-training-data/humanrig_posed/ ..."
    exit 1
fi
echo ""

# =============================================================================
# STEP 2: Quality filter + Marigold enrichment
# =============================================================================
echo "[2/8] Running quality filter + Marigold enrichment..."
echo ""

for ds in humanrig humanrig_posed meshy_cc0 meshy_cc0_textured gemini_diverse; do
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
echo "  Enriching new datasets with normals + depth (Marigold)..."
pip install -q diffusers transformers accelerate 2>/dev/null

for ds in humanrig_posed gemini_diverse; do
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
# STEP 3: Train joints (THE BIG UPGRADE — 45K posed GT examples)
# =============================================================================
echo "[3/8] Training JOINTS model (with humanrig_posed GT data)..."
echo ""
echo "  Config: $JOINTS_CONFIG"
echo "  New: ~45K ground-truth posed examples (Blender raycast)"
echo "  GT ratio: ~40% (up from ~19%)"
echo ""

python -m training.train_joints \
    --config "$JOINTS_CONFIG" \
    2>&1 | tee "$LOG_DIR/joints.log"

echo ""
echo "  Joints training complete."
echo ""

# =============================================================================
# STEP 4: Train segmentation (resume from run 4)
# =============================================================================
echo "[4/8] Training SEGMENTATION model (resume from run 4)..."
echo ""
echo "  Resuming from: $RUN4_SEG_CKPT"
echo "  Config: $SEG_CONFIG"
echo "  New: expanded gemini_diverse (500+)"
echo ""

python -m training.train_segmentation \
    --config "$SEG_CONFIG" \
    --resume "$RUN4_SEG_CKPT" \
    2>&1 | tee "$LOG_DIR/segmentation.log"

if [ ! -f "checkpoints/segmentation/best.pt" ]; then
    echo "  ERROR: No best.pt produced by segmentation training!"
    exit 1
fi

echo ""
echo "  Segmentation training complete."
echo ""

# =============================================================================
# STEP 5: Recompute encoder features + retrain weights
# =============================================================================
echo "[5/8] Recomputing encoder features with new seg model..."
echo ""

python -m training.data.precompute_encoder_features \
    --checkpoint checkpoints/segmentation/best.pt \
    --data-dirs ./data_cloud/humanrig ./data_cloud/humanrig_posed \
    --output-dir ./data_cloud/encoder_features \
    --batch-size 16 \
    --fp16 \
    2>&1 | tee "$LOG_DIR/precompute_encoder.log"

echo ""
echo "  Training WEIGHTS model with new encoder features..."
echo ""

python -m training.train_weights \
    --config "$WEIGHTS_CONFIG" \
    2>&1 | tee "$LOG_DIR/weights.log"

echo ""
echo "  Weights training complete."
echo ""

# =============================================================================
# STEP 6: ONNX Export (all 4 models)
# =============================================================================
echo "[6/8] Exporting all models to ONNX..."
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
# STEP 7: Re-enrich Gemini data with run 5 model (bootstrap for run 6)
# =============================================================================
echo "[7/8] Re-enriching Gemini data with updated seg model..."
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
# STEP 8: Upload everything to bucket
# =============================================================================
echo "[8/8] Uploading checkpoints, logs, ONNX models, and encoder features..."
echo ""

rclone copy ./checkpoints/ hetzner:strata-training-data/checkpoints_run5/ \
    --transfers 32 --fast-list -P
rclone copy ./logs/ hetzner:strata-training-data/logs/ \
    --transfers 32 --fast-list -P
rclone copy ./models/onnx/ hetzner:strata-training-data/models/onnx_run5/ \
    --transfers 32 --fast-list -P
rclone copy ./data_cloud/encoder_features/ hetzner:strata-training-data/encoder_features_run5/ \
    --transfers 32 --fast-list -P

# Upload enriched datasets back
echo ""
echo "  Uploading enriched humanrig_posed + gemini_diverse..."
rclone copy ./data_cloud/humanrig_posed/ hetzner:strata-training-data/humanrig_posed/ \
    --transfers 32 --checkers 64 --fast-list --size-only -P
rclone copy ./data_cloud/gemini_diverse/ hetzner:strata-training-data/gemini_diverse/ \
    --transfers 32 --checkers 64 --fast-list --size-only -P

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

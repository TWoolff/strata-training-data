#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 16 Seg (A100)
#
# Fixed split loader + recovered sora_diverse + GT humanrig_posed (train-only).
# Resume from run 15 (0.5695 mIoU).
#
# Key fixes applied:
#   - humanrig_posed as train_only (doesn't affect val/test split)
#   - sora_diverse placeholder masks → real pseudo-labels (839 recovered)
#   - humanrig_\d+ regex groups posed examples correctly (666 chars, not 81K)
#
# Estimated: ~3 hrs total on A100
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_seg_run16.sh
#   ./training/run_seg_run16.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run16_seg_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 16 Seg"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
echo "[pre] Pre-flight checks..."

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
# 0. Download run 15 checkpoint
# ---------------------------------------------------------------------------
echo "[0/5] Downloading checkpoint..."
mkdir -p checkpoints/segmentation

RUN15_CKPT="checkpoints/segmentation/run15_best.pt"
if [ -f "$RUN15_CKPT" ]; then
    echo "  run15_best.pt already exists."
else
    echo "  Downloading run 15 checkpoint..."
    rclone copy hetzner:strata-training-data/checkpoints_run15_seg/segmentation/run15_best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
    if [ ! -f "$RUN15_CKPT" ]; then
        echo "  FATAL: Could not download run15_best.pt"
        exit 1
    fi
fi

# Download frozen splits file if available in bucket
FROZEN_FILE="./data_cloud/frozen_val_test.json"
if [ ! -f "$FROZEN_FILE" ]; then
    mkdir -p ./data_cloud
    rclone copy hetzner:strata-training-data/data_cloud/frozen_val_test.json ./data_cloud/ \
        --transfers 4 2>/dev/null || true
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
    local check="${3:-}"

    if [ -n "$check" ]; then
        if [ -d "$dir" ] && [ "$(find "$dir" -maxdepth 2 -name "$check" | head -1)" ]; then
            echo "  $name already exists."
            return 0
        fi
    elif [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null | head -1)" ]; then
        echo "  $name already exists."
        return 0
    fi

    rm -rf "$dir"

    # Try enriched tar first (has depth+normals, skips Marigold)
    local enriched_tar="./data_cloud/tars/${name}_enriched.tar"
    echo "  Checking for ${name}_enriched.tar..."
    rclone copy "hetzner:strata-training-data/tars/${name}_enriched.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P 2>/dev/null || true
    if [ -f "$enriched_tar" ]; then
        echo "  Found enriched tar — extracting $name..."
        tar xf "$enriched_tar" -C ./data_cloud/
        rm -f "$enriched_tar"
        COUNT=$(ls "./data_cloud/$name/" 2>/dev/null | wc -l | tr -d ' ')
        echo "  $name (enriched): $COUNT examples"
        return 0
    fi

    # Fall back to regular tar
    local tar="./data_cloud/tars/${tar_name}.tar"
    echo "  Downloading $name..."
    rclone copy "hetzner:strata-training-data/tars/${tar_name}.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "$tar" ]; then
        echo "  Extracting $name..."
        tar xf "$tar" -C ./data_cloud/
        rm -f "$tar"
        COUNT=$(ls "./data_cloud/$name/" 2>/dev/null | wc -l | tr -d ' ')
        echo "  $name: $COUNT examples"
    else
        echo "  FATAL: $name tar not found in bucket."
        exit 1
    fi
}

download_dataset humanrig humanrig "segmentation.png"
download_dataset humanrig_posed humanrig_posed "segmentation.png"
download_dataset vroid_cc0 vroid_cc0
download_dataset gemini_li_converted gemini_li_converted
download_dataset cvat_annotated cvat_annotated
download_dataset flux_diverse_clean flux_diverse_clean "image.png"
download_dataset toon_pseudo toon_pseudo "image.png"
download_dataset sora_diverse sora_diverse "image.png"

# meshy_cc0_textured (restructured tar, special naming)
MESHY_DIR="./data_cloud/meshy_cc0_textured"
if [ -d "$MESHY_DIR" ] && [ "$(find "$MESHY_DIR" -maxdepth 2 -name 'segmentation.png' | head -1)" ]; then
    echo "  meshy_cc0_textured already exists."
else
    echo "  Downloading meshy_cc0_textured..."
    rm -rf "$MESHY_DIR"
    rclone copy "hetzner:strata-training-data/tars/meshy_cc0_textured_restructured.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    MESHY_TAR="./data_cloud/tars/meshy_cc0_textured_restructured.tar"
    if [ -f "$MESHY_TAR" ]; then
        tar xf "$MESHY_TAR" -C ./data_cloud/
        rm -f "$MESHY_TAR"
        [ -d "./data_cloud/meshy_cc0_restructured" ] && mv ./data_cloud/meshy_cc0_restructured "$MESHY_DIR"
    else
        echo "  FATAL: meshy tar not found."
        exit 1
    fi
fi

echo ""
echo "  Dataset summary:"
for ds in humanrig humanrig_posed vroid_cc0 meshy_cc0_textured gemini_li_converted cvat_annotated sora_diverse flux_diverse_clean toon_pseudo; do
    count=$(ls "./data_cloud/$ds/" 2>/dev/null | wc -l | tr -d ' ')
    echo "    $ds: $count examples"
done
echo ""

# ---------------------------------------------------------------------------
# 1b. Pseudo-label sora_diverse (placeholder masks → real seg masks)
# ---------------------------------------------------------------------------
echo "[1b/5] Pseudo-labeling sora_diverse (recovering placeholder masks)..."
echo ""

for ds in sora_diverse; do
    ds_dir="./data_cloud/$ds"
    if [ ! -d "$ds_dir" ]; then continue; fi
    TOTAL=$(find "$ds_dir" -maxdepth 2 -name "image.png" 2>/dev/null | wc -l | tr -d ' ')
    HAVE_SEG=$(find "$ds_dir" -maxdepth 2 -name "segmentation.png" 2>/dev/null | wc -l | tr -d ' ')
    NEED=$((TOTAL - HAVE_SEG))
    if [ "$NEED" -gt 0 ]; then
        echo "  $ds: $TOTAL images, $HAVE_SEG labeled, $NEED need pseudo-labeling..."
        python scripts/batch_pseudo_label.py \
            --input-dir "$ds_dir" \
            --output-dir "$ds_dir" \
            --checkpoint "$RUN15_CKPT" \
            --only-missing \
            2>&1 | tee "$LOG_DIR/pseudo_label_${ds}.log"
    else
        echo "  $ds: all $TOTAL examples have segmentation masks."
    fi
done
echo ""

# ---------------------------------------------------------------------------
# 2. Marigold enrichment (new sora_diverse images only)
# ---------------------------------------------------------------------------
echo "[2/5] Marigold enrichment (new images only)..."
echo ""

for ds_dir in "./data_cloud/sora_diverse" "./data_cloud/flux_diverse_clean" "./data_cloud/gemini_li_converted" "./data_cloud/toon_pseudo"; do
    ds_name=$(basename "$ds_dir")
    if [ ! -d "$ds_dir" ]; then continue; fi
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

# humanrig_posed — GT masks, skip anatomy checks
HP_DIR="./data_cloud/humanrig_posed"
if [ -d "$HP_DIR" ]; then
    if [ -f "$HP_DIR/quality_filter.json" ]; then
        echo "  humanrig_posed: quality_filter.json exists, skipping."
    else
        echo "  Filtering humanrig_posed (--skip-anatomy for GT posed data)..."
        python scripts/filter_seg_quality.py \
            --data-dirs "$HP_DIR" \
            --output-dir "$HP_DIR" \
            --min-regions 3 \
            --max-single-region 0.80 \
            --min-foreground 0.03 \
            --skip-anatomy \
            2>&1 | tee -a "$LOG_DIR/quality_filter.log"
    fi
fi

# GT datasets — strict filter
for ds in humanrig vroid_cc0 meshy_cc0_textured gemini_li_converted cvat_annotated; do
    ds_dir="./data_cloud/$ds"
    if [ -d "$ds_dir" ]; then
        if [ -f "$ds_dir/quality_filter.json" ]; then
            echo "  $ds: quality_filter.json exists, skipping."
        else
            echo "  Filtering $ds..."
            python scripts/filter_seg_quality.py \
                --data-dirs "$ds_dir" \
                --output-dir "$ds_dir" \
                --min-regions 4 \
                --max-single-region 0.70 \
                --min-foreground 0.05 \
                2>&1 | tee -a "$LOG_DIR/quality_filter.log"
        fi
    fi
done

# Pseudo-labeled datasets — relaxed filter (skip anatomy, higher single-region threshold)
rm -f ./data_cloud/sora_diverse/quality_filter.json 2>/dev/null
for ds in sora_diverse flux_diverse_clean toon_pseudo; do
    ds_dir="./data_cloud/$ds"
    if [ -d "$ds_dir" ]; then
        if [ -f "$ds_dir/quality_filter.json" ]; then
            echo "  $ds: quality_filter.json exists, skipping."
        else
            echo "  Filtering $ds (--skip-anatomy, relaxed thresholds)..."
            python scripts/filter_seg_quality.py \
                --data-dirs "$ds_dir" \
                --output-dir "$ds_dir" \
                --min-regions 4 \
                --max-single-region 0.80 \
                --min-foreground 0.05 \
                --skip-anatomy \
                2>&1 | tee -a "$LOG_DIR/quality_filter.log"
        fi
    fi
done
echo "  Quality filter complete."
echo ""

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 3b. Generate frozen val/test splits (if not already present)
# ---------------------------------------------------------------------------
FROZEN_FILE="./data_cloud/frozen_val_test.json"
if [ ! -f "$FROZEN_FILE" ]; then
    echo "  Generating frozen val/test split (without humanrig_posed)..."
    python3 -c "
from pathlib import Path
from training.data.split_loader import load_or_generate_splits
# Generate splits from base datasets only (no humanrig_posed)
dirs = [
    Path('./data_cloud/humanrig'),
    Path('./data_cloud/vroid_cc0'),
    Path('./data_cloud/meshy_cc0_textured'),
    Path('./data_cloud/gemini_li_converted'),
    Path('./data_cloud/cvat_annotated'),
    Path('./data_cloud/sora_diverse'),
    Path('./data_cloud/flux_diverse_clean'),
    Path('./data_cloud/toon_pseudo'),
]
splits = load_or_generate_splits(dirs, seed=42, frozen_splits_file=Path('$FROZEN_FILE'))
print(f'  Val: {len(splits[\"val\"])} chars, Test: {len(splits[\"test\"])} chars')
"
    # Upload frozen file to bucket for future runs
    rclone copy "$FROZEN_FILE" hetzner:strata-training-data/data_cloud/ --transfers 4 -P
    echo "  Frozen splits saved and uploaded."
else
    echo "  Frozen splits file exists."
fi
echo ""

echo "[4/5] Training SEGMENTATION model..."
echo "  Resuming from: $RUN15_CKPT (run 15 best)"
echo "  Config: training/configs/segmentation_a100_run16.yaml"
echo ""

python -m training.train_segmentation \
    --config training/configs/segmentation_a100_run16.yaml \
    --reset-epochs \
    --resume "$RUN15_CKPT" \
    2>&1 | tee "$LOG_DIR/segmentation.log"

echo ""
echo "  Segmentation training complete."
echo ""

# ---------------------------------------------------------------------------
# 5. Export + Upload
# ---------------------------------------------------------------------------
echo "[5/5] Exporting + uploading..."
echo ""

mkdir -p ./models/onnx

if [ -f "checkpoints/segmentation/best.pt" ]; then
    cp checkpoints/segmentation/best.pt checkpoints/segmentation/run16_best.pt
    python -m training.export_onnx \
        --model segmentation \
        --checkpoint checkpoints/segmentation/run16_best.pt \
        --output ./models/onnx/segmentation_run16.onnx \
        2>&1 | tee "$LOG_DIR/export.log"
    echo "  Exported segmentation_run16.onnx"
fi

rclone copy ./checkpoints/segmentation/run16_best.pt \
    hetzner:strata-training-data/checkpoints_run16_seg/segmentation/ \
    --transfers 32 --fast-list -P
rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run16_seg_${TIMESTAMP}/ \
    --transfers 32 --fast-list -P
if [ -f "./models/onnx/segmentation_run16.onnx" ]; then
    rclone copy ./models/onnx/segmentation_run16.onnx \
        hetzner:strata-training-data/models/onnx_run16_seg/ \
        --transfers 32 --fast-list -P
fi

echo ""
echo "============================================"
echo "  Run 16 Seg complete!"
echo "  Finished: $(date)"
echo "  Results:"
grep -E "New best mIoU|mIoU=" "$LOG_DIR/segmentation.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  To download to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_run16_seg/ /Volumes/TAMWoolff/data/checkpoints_run16_seg/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/onnx_run16_seg/ /Volumes/TAMWoolff/data/models/onnx_run16_seg/ --transfers 32 --fast-list -P"
echo "============================================"

#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 15 Seg (A100)
#
# GT humanrig_posed (83K Blender-rendered seg masks) + more illustrated data.
# Resume from run 14 (0.5561 mIoU).
#
# Estimated: ~3.5 hrs total on A100
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_seg_run15.sh
#   ./training/run_seg_run15.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run15_seg_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 15 Seg"
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
# 0. Download run 14 checkpoint
# ---------------------------------------------------------------------------
echo "[0/5] Downloading checkpoint..."
mkdir -p checkpoints/segmentation

RUN14_CKPT="checkpoints/segmentation/run14_best.pt"
if [ -f "$RUN14_CKPT" ]; then
    echo "  run14_best.pt already exists."
else
    echo "  Downloading run 14 checkpoint..."
    rclone copy hetzner:strata-training-data/checkpoints_run14_seg/segmentation/run14_best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
    if [ ! -f "$RUN14_CKPT" ]; then
        echo "  FATAL: Could not download run14_best.pt"
        exit 1
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# 1. Download datasets (enriched tars first, fallback to regular)
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
# humanrig_posed — force fresh download (GT masks replaced old pseudo-labels)
rm -rf ./data_cloud/humanrig_posed
download_dataset humanrig_posed humanrig_posed "segmentation.png"
download_dataset vroid_cc0 vroid_cc0
download_dataset gemini_li_converted gemini_li_converted
download_dataset cvat_annotated cvat_annotated
download_dataset flux_diverse_clean flux_diverse_clean "image.png"
download_dataset toon_pseudo toon_pseudo "image.png"

# sora_diverse — force regular tar (enriched tar has old images)
SORA_DIR="./data_cloud/sora_diverse"
if [ -d "$SORA_DIR" ] && [ "$(find "$SORA_DIR" -maxdepth 2 -name 'image.png' | head -1)" ]; then
    SORA_COUNT=$(ls "$SORA_DIR/" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$SORA_COUNT" -lt 1600 ]; then
        echo "  sora_diverse has $SORA_COUNT examples (stale) — re-downloading..."
        rm -rf "$SORA_DIR"
    else
        echo "  sora_diverse already exists ($SORA_COUNT examples)."
    fi
fi
if [ ! -d "$SORA_DIR" ] || [ -z "$(ls -A "$SORA_DIR" 2>/dev/null)" ]; then
    echo "  Downloading sora_diverse (regular tar)..."
    rclone copy "hetzner:strata-training-data/tars/sora_diverse.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P
    if [ -f "./data_cloud/tars/sora_diverse.tar" ]; then
        tar xf "./data_cloud/tars/sora_diverse.tar" -C ./data_cloud/
        rm -f "./data_cloud/tars/sora_diverse.tar"
        SORA_COUNT=$(ls "$SORA_DIR/" 2>/dev/null | wc -l | tr -d ' ')
        echo "  sora_diverse: $SORA_COUNT examples"
    else
        echo "  FATAL: sora_diverse tar not found."
        exit 1
    fi
fi

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
# 1b. Pseudo-label new sora_diverse images
# ---------------------------------------------------------------------------
echo "[1b/5] Pseudo-labeling new images..."
echo ""

for ds in sora_diverse; do
    ds_dir="./data_cloud/$ds"
    if [ ! -d "$ds_dir" ]; then continue; fi
    TOTAL=$(find "$ds_dir" -maxdepth 2 -name "image.png" 2>/dev/null | wc -l | tr -d ' ')
    HAVE_SEG=$(find "$ds_dir" -maxdepth 2 -name "segmentation.png" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$TOTAL" -gt 0 ] && [ "$HAVE_SEG" -lt "$TOTAL" ]; then
        echo "  $ds: $TOTAL images, $HAVE_SEG labeled — pseudo-labeling remainder..."
        python scripts/batch_pseudo_label.py \
            --input-dir "$ds_dir" \
            --output-dir "$ds_dir" \
            --checkpoint "$RUN14_CKPT" \
            --only-missing \
            2>&1 | tee "$LOG_DIR/pseudo_label_${ds}.log"
    else
        echo "  $ds: all examples have segmentation masks."
    fi
done
echo ""

# ---------------------------------------------------------------------------
# 2. Marigold enrichment (only for new/missing — enriched tars skip this)
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

for ds in humanrig humanrig_posed vroid_cc0 meshy_cc0_textured gemini_li_converted cvat_annotated sora_diverse flux_diverse_clean toon_pseudo; do
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
echo "  Quality filter complete."
echo ""

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
echo "[4/5] Training SEGMENTATION model..."
echo "  Resuming from: $RUN14_CKPT (run 14 best)"
echo "  Config: training/configs/segmentation_a100_run15.yaml"
echo ""

python -m training.train_segmentation \
    --config training/configs/segmentation_a100_run15.yaml \
    --reset-epochs \
    --resume "$RUN14_CKPT" \
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
    cp checkpoints/segmentation/best.pt checkpoints/segmentation/run15_best.pt
    python -m training.export_onnx \
        --model segmentation \
        --checkpoint checkpoints/segmentation/run15_best.pt \
        --output ./models/onnx/segmentation_run15.onnx \
        2>&1 | tee "$LOG_DIR/export.log"
    echo "  Exported segmentation_run15.onnx"
fi

rclone copy ./checkpoints/segmentation/run15_best.pt \
    hetzner:strata-training-data/checkpoints_run15_seg/segmentation/ \
    --transfers 32 --fast-list -P
rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run15_seg_${TIMESTAMP}/ \
    --transfers 32 --fast-list -P
if [ -f "./models/onnx/segmentation_run15.onnx" ]; then
    rclone copy ./models/onnx/segmentation_run15.onnx \
        hetzner:strata-training-data/models/onnx_run15_seg/ \
        --transfers 32 --fast-list -P
fi

echo ""
echo "============================================"
echo "  Run 15 Seg complete!"
echo "  Finished: $(date)"
echo "  Results:"
grep -E "New best mIoU|mIoU=" "$LOG_DIR/segmentation.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
echo "  To download to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_run15_seg/ /Volumes/TAMWoolff/data/checkpoints_run15_seg/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/onnx_run15_seg/ /Volumes/TAMWoolff/data/models/onnx_run15_seg/ --transfers 32 --fast-list -P"
echo "============================================"

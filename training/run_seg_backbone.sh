#!/usr/bin/env bash
# =============================================================================
# Strata Training — Backbone Comparison Run (A100)
#
# Goal: Test EfficientNet-B3 backbone for segmentation (replacing MobileNetV3)
#   - MobileNetV3 (~18M params) plateaued at ~0.37 mIoU across multiple runs
#   - EfficientNet-B3 (~33M params) should provide more capacity
#   - Training from ImageNet-pretrained backbone (fresh seg/depth/normals heads)
#   - Same datasets as seg+meshy run (humanrig + vroid_cc0 + meshy_cc0_textured + anime_seg)
#
# Optional: pass --resnet50 to also train a ResNet-50 variant (~54M params)
#
# Estimated: ~6-8 hrs on A100, ~$2-3
#   - EfficientNet-B3: ~6-8 hrs (80 epochs, early stopping patience 15)
#   - ResNet-50 (optional): ~6-8 hrs additional
#
# Prerequisites:
#   export BUCKET_ACCESS_KEY='...'
#   export BUCKET_SECRET='...'
#   git clone https://github.com/TWoolff/strata-training-data.git && cd strata-training-data
#   ./training/cloud_setup.sh lean
#
# Usage:
#   chmod +x training/run_seg_backbone.sh
#   ./training/run_seg_backbone.sh                # EfficientNet-B3 only
#   ./training/run_seg_backbone.sh --resnet50     # Also train ResNet-50
# =============================================================================
set -euo pipefail

ALSO_RESNET50=false
if [[ "${1:-}" == "--resnet50" ]]; then
    ALSO_RESNET50=true
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/seg_backbone_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Backbone Comparison"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "  EfficientNet-B3: YES"
echo "  ResNet-50: $ALSO_RESNET50"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "[pre] Pre-flight checks..."

if [ -z "${BUCKET_ACCESS_KEY:-}" ] || [ -z "${BUCKET_SECRET:-}" ]; then
    echo "  WARNING: BUCKET_ACCESS_KEY/BUCKET_SECRET not set."
    echo "  Upload step will fail. Set them if you want bucket upload."
fi

pip install -q scipy diffusers transformers accelerate 2>/dev/null || true
echo "  Dependencies OK."
echo ""

# ---------------------------------------------------------------------------
# 1. Download datasets
# ---------------------------------------------------------------------------
echo "[1/5] Downloading datasets..."
echo ""

# Helper: download + extract a tar, delete tar after
download_tar() {
    local name="$1"
    local ds_dir="./data_cloud/$name"

    if [ -d "$ds_dir" ] && [ "$(ls -A "$ds_dir" 2>/dev/null | head -1)" ]; then
        echo "  $name already exists."
        return 0
    fi

    mkdir -p ./data_cloud/tars
    local tar_file="./data_cloud/tars/${name}.tar"

    echo "  Downloading ${name}.tar..."
    rclone copy "hetzner:strata-training-data/tars/${name}.tar" ./data_cloud/tars/ \
        --transfers 32 --fast-list -P

    if [ -f "$tar_file" ]; then
        echo "  Extracting $name..."
        tar xf "$tar_file" -C ./data_cloud/
        rm -f "$tar_file"
        echo "  $name ready. (tar deleted)"
    else
        echo "  WARNING: Could not download $name."
        return 1
    fi
}

download_tar "humanrig"
download_tar "vroid_cc0"
download_tar "anime_seg"

# Meshy CC0 textured restructured
MESHY_DIR="./data_cloud/meshy_cc0_textured"
if [ -d "$MESHY_DIR" ] && [ "$(ls -A "$MESHY_DIR" 2>/dev/null | head -1)" ]; then
    echo "  meshy_cc0_textured already exists."
else
    mkdir -p ./data_cloud/tars
    echo "  Downloading meshy_cc0_textured_restructured.tar..."
    rclone copy "hetzner:strata-training-data/tars/meshy_cc0_textured_restructured.tar" \
        ./data_cloud/tars/ --transfers 32 --fast-list -P
    tar_file="./data_cloud/tars/meshy_cc0_textured_restructured.tar"
    if [ -f "$tar_file" ]; then
        echo "  Extracting..."
        tar xf "$tar_file" -C ./data_cloud/
        rm -f "$tar_file"
        for candidate in meshy_cc0_textured_restructured meshy_cc0_restructured; do
            if [ -d "./data_cloud/$candidate" ] && [ ! -d "$MESHY_DIR" ]; then
                mv "./data_cloud/$candidate" "$MESHY_DIR"
                echo "  Renamed $candidate → meshy_cc0_textured"
            fi
        done
    else
        echo "  WARNING: Could not download meshy_cc0_textured_restructured.tar"
    fi
fi

echo ""
echo "  Disk usage after downloads:"
du -sh ./data_cloud/* 2>/dev/null | head -20
df -h . | tail -1
echo ""

# ---------------------------------------------------------------------------
# 2. Quality filter + Marigold enrichment
# ---------------------------------------------------------------------------
echo "[2/5] Quality filter + Marigold normals..."
echo ""

for ds in humanrig vroid_cc0 meshy_cc0_textured anime_seg; do
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

for ds in meshy_cc0_textured vroid_cc0; do
    if [ -d "./data_cloud/$ds" ]; then
        echo "  Enriching $ds with Marigold normals..."
        python run_normals_enrich.py \
            --input-dir "./data_cloud/$ds" \
            --only-missing \
            --batch-size 16 \
            2>&1 | tee "$LOG_DIR/enrich_normals_${ds}.log"
    fi
done

echo "  Quality filter + enrichment complete."
echo ""

# ---------------------------------------------------------------------------
# 3. Train EfficientNet-B3
# ---------------------------------------------------------------------------
echo "[3/5] Training SEGMENTATION with EfficientNet-B3 backbone..."
echo ""
echo "  Config: training/configs/segmentation_a100_efficientnet.yaml"
echo "  Training from scratch (ImageNet-pretrained backbone, fresh heads)"
echo ""

python -m training.train_segmentation \
    --config training/configs/segmentation_a100_efficientnet.yaml \
    2>&1 | tee "$LOG_DIR/seg_efficientnet_b3.log"

echo ""
echo "  EfficientNet-B3 training complete."
echo ""

# Save EfficientNet checkpoint separately
EFFNET_BEST="./checkpoints/segmentation/best.pt"
if [ -f "$EFFNET_BEST" ]; then
    mkdir -p ./checkpoints/segmentation_efficientnet
    cp "$EFFNET_BEST" ./checkpoints/segmentation_efficientnet/best.pt
    cp ./checkpoints/segmentation/latest.pt ./checkpoints/segmentation_efficientnet/latest.pt 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 3b. (Optional) Train ResNet-50
# ---------------------------------------------------------------------------
if [ "$ALSO_RESNET50" = true ]; then
    echo "[3b/5] Training SEGMENTATION with ResNet-50 backbone..."
    echo ""
    echo "  Config: training/configs/segmentation_a100_resnet50.yaml"
    echo ""

    # Clear checkpoints dir for ResNet run
    rm -f ./checkpoints/segmentation/best.pt ./checkpoints/segmentation/latest.pt

    python -m training.train_segmentation \
        --config training/configs/segmentation_a100_resnet50.yaml \
        2>&1 | tee "$LOG_DIR/seg_resnet50.log"

    echo ""
    echo "  ResNet-50 training complete."
    echo ""

    # Save ResNet-50 checkpoint separately
    RESNET_BEST="./checkpoints/segmentation/best.pt"
    if [ -f "$RESNET_BEST" ]; then
        mkdir -p ./checkpoints/segmentation_resnet50
        cp "$RESNET_BEST" ./checkpoints/segmentation_resnet50/best.pt
        cp ./checkpoints/segmentation/latest.pt ./checkpoints/segmentation_resnet50/latest.pt 2>/dev/null || true
    fi
fi

# ---------------------------------------------------------------------------
# 4. ONNX Export
# ---------------------------------------------------------------------------
echo "[4/5] Exporting models to ONNX..."
echo ""

ONNX_DIR="./models/onnx_backbone"
mkdir -p "$ONNX_DIR"

# Export EfficientNet-B3
if [ -f "./checkpoints/segmentation_efficientnet/best.pt" ]; then
    python -m training.export_onnx \
        --model segmentation \
        --backbone efficientnet_b3 \
        --checkpoint ./checkpoints/segmentation_efficientnet/best.pt \
        --output "$ONNX_DIR/segmentation_efficientnet.onnx" \
        2>&1 | tee "$LOG_DIR/export_efficientnet.log"
    echo "  Exported segmentation_efficientnet.onnx"
fi

# Export ResNet-50 (if trained)
if [ -f "./checkpoints/segmentation_resnet50/best.pt" ]; then
    python -m training.export_onnx \
        --model segmentation \
        --backbone resnet50 \
        --checkpoint ./checkpoints/segmentation_resnet50/best.pt \
        --output "$ONNX_DIR/segmentation_resnet50.onnx" \
        2>&1 | tee "$LOG_DIR/export_resnet50.log"
    echo "  Exported segmentation_resnet50.onnx"
fi

echo ""

# ---------------------------------------------------------------------------
# 5. Upload to bucket
# ---------------------------------------------------------------------------
echo "[5/5] Uploading checkpoints, logs, and ONNX..."
echo ""

rclone copy ./checkpoints/segmentation_efficientnet/ \
    hetzner:strata-training-data/checkpoints_backbone/segmentation_efficientnet/ \
    --transfers 32 --fast-list -P

if [ -d "./checkpoints/segmentation_resnet50" ]; then
    rclone copy ./checkpoints/segmentation_resnet50/ \
        hetzner:strata-training-data/checkpoints_backbone/segmentation_resnet50/ \
        --transfers 32 --fast-list -P
fi

rclone copy ./logs/ hetzner:strata-training-data/logs/ \
    --transfers 32 --fast-list -P

rclone copy "$ONNX_DIR/" hetzner:strata-training-data/models/onnx_backbone/ \
    --transfers 32 --fast-list -P

echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================"
echo "  Backbone Comparison run complete!"
echo "  Finished: $(date)"
echo ""
echo "  EfficientNet-B3 results:"
grep -E "Best mIoU|New best|miou" "$LOG_DIR/seg_efficientnet_b3.log" 2>/dev/null | tail -5 || echo "  (check logs)"
echo ""
if [ "$ALSO_RESNET50" = true ]; then
    echo "  ResNet-50 results:"
    grep -E "Best mIoU|New best|miou" "$LOG_DIR/seg_resnet50.log" 2>/dev/null | tail -5 || echo "  (check logs)"
    echo ""
fi
echo "  To download results to Mac:"
echo "    rclone copy hetzner:strata-training-data/checkpoints_backbone/ ./checkpoints_backbone/ --transfers 32 --fast-list -P"
echo "    rclone copy hetzner:strata-training-data/models/onnx_backbone/ ./models/onnx_backbone/ --transfers 32 --fast-list -P"
echo "============================================"

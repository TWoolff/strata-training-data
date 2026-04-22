#!/usr/bin/env bash
# =============================================================================
# Strata Training — Run 29 Seg (A100) — Run 28 + label cleaning
#
# Conditional on Run 28 underperforming. Tests whether removing obviously-bad
# pseudo-labels improves training.
#
# Audit on April 22 found (scripts/audit_labels.py):
#   - flux_diverse_clean: 15.2% of examples have head-below-torso (impossible)
#   - sora_diverse: 3.9%
#   - gemini_diverse: 5.4%
#   - gemini_li_converted (hand labels): 2.4% (real edge cases)
#
# Run 29 = Run 28 + --drop-head-below-torso in quality filter. Removes ~370
# clearly-bad labels across flux+sora+gemini_diverse without touching the
# hand-labeled or GT datasets.
#
# Estimated: ~5 hrs total on A100 (same as Run 28).
#
# Usage:
#   chmod +x training/run_seg_run29.sh
#   ./training/run_seg_run29.sh
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/run29_seg_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  Strata Training — Run 29 Seg"
echo "  Run 28 hyperparams + cleaned labels"
echo "  Started: $(date)"
echo "  Logs: $LOG_DIR"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
echo "[pre] Pre-flight checks..."

PREFLIGHT_FAIL=0

if ! rclone lsd hetzner:strata-training-data/ &>/dev/null; then
    echo "  FAIL: rclone"; PREFLIGHT_FAIL=1
else
    echo "  OK: rclone bucket connection"
fi

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  FAIL: CUDA"; PREFLIGHT_FAIL=1
else
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEM=$(python3 -c "import torch; p=torch.cuda.get_device_properties(0); m=getattr(p,'total_memory',getattr(p,'total_mem',0)); print(f'{m/1e9:.0f}GB')")
    echo "  OK: CUDA — $GPU_NAME ($GPU_MEM)"

    FREE_MIB=$(python3 -c "import torch; t=torch.cuda.get_device_properties(0); a=torch.cuda.memory_allocated(0); print(int((t.total_memory-a)/1e6))")
    if [ "$FREE_MIB" -lt 30000 ]; then
        echo "  WARN: only ${FREE_MIB} MiB free — zombie CUDA contexts? check: fuser -v /dev/nvidia*"
    fi
fi

RUN20_CKPT="checkpoints/segmentation/run20_best.pt"
if [ ! -f "$RUN20_CKPT" ]; then
    mkdir -p checkpoints/segmentation
    rclone copy hetzner:strata-training-data/checkpoints_run20_seg/segmentation/run20_best.pt \
        ./checkpoints/segmentation/ --transfers 32 --fast-list -P
fi
echo "  OK: Run 20 checkpoint"

download_if_missing() {
    local tar_name="$1"; local extract_dir="$2"
    if [ -d "$extract_dir" ] && [ -n "$(ls "$extract_dir"/ 2>/dev/null | head -1)" ]; then
        return 0
    fi
    echo "  $extract_dir missing — downloading $tar_name..."
    mkdir -p data/tars data_cloud
    rclone copy "hetzner:strata-training-data/tars/$tar_name" ./data/tars/ --transfers 32 --fast-list -P
    tar xf "./data/tars/$tar_name" -C ./data_cloud/
    rm -f "./data/tars/$tar_name"
}

download_if_missing "humanrig.tar" "./data_cloud/humanrig"
download_if_missing "vroid_cc0.tar" "./data_cloud/vroid_cc0"
download_if_missing "meshy_cc0_textured_restructured.tar" "./data_cloud/meshy_cc0_restructured"
download_if_missing "gemini_li_converted.tar" "./data_cloud/gemini_li_converted"
download_if_missing "cvat_annotated.tar" "./data_cloud/cvat_annotated"
download_if_missing "sora_diverse.tar" "./data_cloud/sora_diverse"
# Incremental sora tars merged into the same dir (added after Run 20 baseline).
# If the base tar was already extracted, these add the ~1,613 newer chars from
# Runs 21-22. Without them sora_diverse drops to 854 examples vs Run 20's 2,467.
if [ -d "./data_cloud/sora_diverse" ]; then
    SORA_COUNT=$(ls -d ./data_cloud/sora_diverse/*/ 2>/dev/null | wc -l | tr -d ' ')
    if [ "$SORA_COUNT" -lt 2000 ]; then
        echo "  sora_diverse only has $SORA_COUNT examples — adding sora_diverse_new tars..."
        mkdir -p data/tars
        for t in sora_diverse_new.tar sora_diverse_new2.tar; do
            rclone copy "hetzner:strata-training-data/tars/$t" ./data/tars/ --transfers 32 --fast-list -P 2>&1 | tail -2 || true
            if [ -f "./data/tars/$t" ]; then
                tar xf "./data/tars/$t" -C ./data_cloud/sora_diverse/ --strip-components=1 2>/dev/null || \
                    tar xf "./data/tars/$t" -C ./data_cloud/ 2>/dev/null || true
                rm -f "./data/tars/$t"
            fi
        done
        echo "  sora_diverse now: $(ls -d ./data_cloud/sora_diverse/*/ | wc -l | tr -d ' ') examples"
        # Force quality filter re-run on sora since composition changed
        rm -f ./data_cloud/sora_diverse/quality_filter.json
    fi
fi
download_if_missing "flux_diverse_clean.tar" "./data_cloud/flux_diverse_clean"
download_if_missing "gemini_diverse.tar" "./data_cloud/gemini_diverse"

if [ ! -f "data_cloud/frozen_val_test.json" ]; then
    rclone copy hetzner:strata-training-data/data_cloud/frozen_val_test.json \
        ./data_cloud/ --transfers 4 --fast-list -P
fi
echo "  OK: frozen val/test splits"

if [ "$PREFLIGHT_FAIL" -ne 0 ]; then exit 1; fi
echo ""

# ---------------------------------------------------------------------------
# 1. Re-pseudo-label gemini_diverse with Run 20 (same as Run 28)
# ---------------------------------------------------------------------------
echo "[1/4] Re-pseudo-labeling gemini_diverse with Run 20 checkpoint..."
python3 scripts/batch_pseudo_label.py \
    --input-dir ./data_cloud/gemini_diverse \
    --output-dir ./data_cloud/gemini_diverse \
    --checkpoint "$RUN20_CKPT" \
    --device cuda \
    2>&1 | tee "$LOG_DIR/pseudo_label.log"

# Invalidate all quality filter caches so the stricter filter runs on every dataset
rm -f ./data_cloud/*/quality_filter.json
echo ""

# ---------------------------------------------------------------------------
# 2. Marigold enrichment
# ---------------------------------------------------------------------------
echo "[2/4] Marigold enrichment (gemini_diverse, only-missing)..."
python3 run_normals_enrich.py \
    --input-dir ./data_cloud/gemini_diverse \
    --only-missing \
    --batch-size 16 \
    2>&1 | tee "$LOG_DIR/enrich.log" || true
echo ""

# ---------------------------------------------------------------------------
# 3. Quality filter with label cleaning
# ---------------------------------------------------------------------------
echo "[3/4] Quality filter (with --drop-head-below-torso)..."
echo "  Removes anatomically-impossible labels. Expected rejections:"
echo "    flux_diverse_clean: +~240 (15%)"
echo "    sora_diverse:       +~100 (4%)"
echo "    gemini_diverse:     +~180 (5%)"
echo "    others: barely affected"
echo ""

# GT datasets — skip anatomy checks (posed renders may legitimately hide head/torso)
for ds_dir in ./data_cloud/humanrig ./data_cloud/vroid_cc0 ./data_cloud/meshy_cc0_restructured; do
    ds_name=$(basename "$ds_dir")
    echo "  $ds_name (GT, skip-anatomy)..."
    python3 scripts/filter_seg_quality.py \
        --data-dirs "$ds_dir" \
        --min-regions 4 --max-single-region 0.70 --min-foreground 0.05 \
        --skip-anatomy \
        2>&1 | tee -a "$LOG_DIR/quality_filter.log"
done

# Pseudo-labeled / hand-labeled illustrated datasets — apply validated checks.
# Combination tested against hand-labeled control on April 22:
#   - --drop-head-below-torso: +0.3% on hand labels, +15% on flux (catches flipped heads)
#   - --max-bg-bleed 0.10:     +0.4% on hand labels, small on pseudo (catches labels
#                              painted into image background)
#   - --min-silhouette-coverage 0.50: +1.7% on hand labels, ~3% on pseudo (catches
#                              incomplete silhouettes)
# Combined hand-label rejection: ~4.5% — acceptable baseline for good data.
# Dropped as validated: --max-disconn-per-class (hand labels legitimately have
# many blobs due to converter aggregation), --class-size-ref (picks up distribution
# shift not individual outliers).
for ds_dir in ./data_cloud/gemini_li_converted ./data_cloud/cvat_annotated ./data_cloud/sora_diverse ./data_cloud/flux_diverse_clean ./data_cloud/gemini_diverse; do
    ds_name=$(basename "$ds_dir")
    echo "  $ds_name (illustrated, validated filter combo)..."
    python3 scripts/filter_seg_quality.py \
        --data-dirs "$ds_dir" \
        --min-regions 4 --max-single-region 0.70 --min-foreground 0.05 \
        --drop-head-below-torso \
        --max-bg-bleed 0.10 \
        --min-silhouette-coverage 0.50 \
        2>&1 | tee -a "$LOG_DIR/quality_filter.log"
done
echo ""

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
echo "[4/4] Training SEGMENTATION model..."
echo "  Config: training/configs/segmentation_a100_run29.yaml"
echo "  Resume: $RUN20_CKPT"
echo ""

python3 -m training.train_segmentation \
    --config training/configs/segmentation_a100_run29.yaml \
    --resume "$RUN20_CKPT" \
    --reset-epochs \
    2>&1 | tee "$LOG_DIR/train.log"

echo ""

# ---------------------------------------------------------------------------
# Export + Upload
# ---------------------------------------------------------------------------
echo "[final] Exporting ONNX + uploading..."

cp checkpoints/segmentation/best.pt checkpoints/segmentation/run29_best.pt

python3 -m training.export_onnx \
    --model segmentation \
    --checkpoint checkpoints/segmentation/run29_best.pt \
    --output ./models/onnx/segmentation_run29.onnx \
    2>&1 | tee "$LOG_DIR/export.log"

rclone copy checkpoints/segmentation/run29_best.pt \
    hetzner:strata-training-data/checkpoints_run29_seg/segmentation/ \
    --transfers 4 --fast-list --size-only -P

rclone copy ./models/onnx/segmentation_run29.onnx \
    hetzner:strata-training-data/models/onnx_run29_seg/ \
    --transfers 4 --fast-list --size-only -P

rclone copy "$LOG_DIR/" hetzner:strata-training-data/logs/run29_seg_${TIMESTAMP}/ \
    --transfers 4 --fast-list -P

echo ""
echo "============================================"
echo "  Run 29 complete! Finished: $(date)"
grep -E "best mIoU|New best" "$LOG_DIR/train.log" 2>/dev/null | tail -3 || echo "  (check logs)"
echo ""
echo "  Interpretation (vs Run 28):"
echo "    Run 29 > Run 28 → noisy labels were hurting; cleaner data helps"
echo "    Run 29 ≈ Run 28 → label noise isn't the bottleneck"
echo "    Run 29 < Run 28 → we over-filtered; dropped useful data"
echo ""
echo "  Download: rclone copy hetzner:strata-training-data/checkpoints_run29_seg/ /Volumes/TAMWoolff/data/checkpoints_run29_seg/ --transfers 32 --fast-list -P"
echo "============================================"

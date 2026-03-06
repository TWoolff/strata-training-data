#!/usr/bin/env bash
# Full NOVA-Human ingest pipeline: convert + enrich with joints + depth
set -euo pipefail

OUTPUT_DIR="./output/nova_human"
INPUT_DIR="./data/nova_human"

echo "=== Step 1/3: Convert NOVA-Human to Strata format (ortho only) ==="
python3 ingest/nova_human_adapter.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --only_new

echo ""
echo "=== Step 2/3: Enrich with RTMPose joints ==="
python3 run_enrich.py \
    --input_dir "$OUTPUT_DIR" \
    --det_model ./models/yolox_m_humanart.onnx \
    --pose_model ./models/rtmpose_m_body7.onnx \
    --only_missing

echo ""
echo "=== Step 3/3: Enrich with Depth Anything v2 draw order ==="
python3 run_depth_enrich.py \
    --input_dir "$OUTPUT_DIR" \
    --depth_model ./models/depth_anything_v2_vits.onnx \
    --only_missing

echo ""
echo "=== Done! ==="
echo "Output: $OUTPUT_DIR"
echo ""
echo "Upload to bucket:"
echo "  rclone copy $OUTPUT_DIR/ hetzner:strata-training-data/nova_human/ --transfers 32 --checkers 64 --fast-list --size-only -P"

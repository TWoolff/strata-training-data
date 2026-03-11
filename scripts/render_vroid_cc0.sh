#!/usr/bin/env bash
# Render VRoid CC0 characters through the Strata pipeline.
#
# Produces: image.png, segmentation.png, depth, normals, joints, metadata
# per character x pose x angle x style.
#
# Prerequisites:
#   1. Download CC0 VRM/GLB files to /Volumes/TAMWoolff/data/raw/vroid_cc0/
#   2. For .vrm files: install VRM Add-on for Blender (https://vrm-addon-for-blender.info/)
#      For .glb files: no add-on needed (uses Blender's built-in glTF importer)
#   3. Blender 4.0+ available on PATH (or set BLENDER below)
#
# Usage:
#   bash scripts/render_vroid_cc0.sh              # Full run (all angles, 20 poses)
#   bash scripts/render_vroid_cc0.sh --quick       # Quick test (front only, 3 poses, 1 char)
#   bash scripts/render_vroid_cc0.sh --dry-run     # Show what would be rendered

set -euo pipefail

# --- Configuration ---
BLENDER="${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"
VROID_DIR="${VROID_DIR:-/Volumes/TAMWoolff/data/raw/vroid_cc0}"
POSE_DIR="${POSE_DIR:-/Volumes/TAMWoolff/data/poses}"
OUTPUT_DIR="${OUTPUT_DIR:-/Volumes/TAMWoolff/data/output/vroid_cc0}"
STYLES="${STYLES:-flat,unlit,textured}"
POSES_PER_CHAR="${POSES_PER_CHAR:-20}"
ANGLES="${ANGLES:-front,three_quarter,side,three_quarter_back,back}"
RESOLUTION=512
MAX_CHARS=0  # 0 = all

# --- Parse flags ---
QUICK=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --quick)
            QUICK=true
            POSES_PER_CHAR=3
            ANGLES="front"
            STYLES="flat"
            MAX_CHARS=1
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
    esac
done

# --- Validate ---
if [ ! -d "$VROID_DIR" ]; then
    echo "ERROR: VRM/GLB directory not found: $VROID_DIR"
    echo "Download CC0 VRM/GLB files from VRoid Hub first."
    exit 1
fi

# Count both .vrm and .glb files
VRM_COUNT=$(find "$VROID_DIR" -maxdepth 1 \( -name "*.vrm" -o -name "*.glb" \) | wc -l | tr -d ' ')
if [ "$VRM_COUNT" -eq 0 ]; then
    echo "ERROR: No .vrm or .glb files found in $VROID_DIR"
    exit 1
fi

echo "============================================================"
echo "VRoid CC0 Rendering Pipeline"
echo "============================================================"
echo "Input directory: $VROID_DIR"
echo "Model files:     $VRM_COUNT (.vrm/.glb)"
echo "Output:          $OUTPUT_DIR"
echo "Poses/char:      $POSES_PER_CHAR"
echo "Angles:          $ANGLES"
echo "Styles:          $STYLES"
echo "Resolution:      ${RESOLUTION}x${RESOLUTION}"
if [ "$MAX_CHARS" -gt 0 ]; then
    echo "Max characters:  $MAX_CHARS"
fi
if [ "$QUICK" = true ]; then
    echo "Mode:            QUICK (1 char, front only, 3 poses, flat style)"
fi
echo "============================================================"
echo ""

# Estimate output
ANGLE_COUNT=$(echo "$ANGLES" | tr ',' '\n' | wc -l | tr -d ' ')
STYLE_COUNT=$(echo "$STYLES" | tr ',' '\n' | wc -l | tr -d ' ')
if [ "$MAX_CHARS" -gt 0 ] && [ "$MAX_CHARS" -lt "$VRM_COUNT" ]; then
    CHAR_COUNT=$MAX_CHARS
else
    CHAR_COUNT=$VRM_COUNT
fi
TOTAL_RENDERS=$((CHAR_COUNT * POSES_PER_CHAR * ANGLE_COUNT * STYLE_COUNT))
echo "Estimated renders: ~${TOTAL_RENDERS} images"
echo "Estimated time:    ~$((TOTAL_RENDERS * 3 / 60)) minutes (at ~3s/render on CPU)"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN — listing model files:"
    find "$VROID_DIR" -maxdepth 1 \( -name "*.vrm" -o -name "*.glb" \) | sort
    exit 0
fi

# --- Check Blender ---
if ! command -v "$BLENDER" &>/dev/null; then
    echo "ERROR: Blender not found. Set BLENDER=/path/to/blender or add to PATH."
    exit 1
fi

BLENDER_VERSION=$("$BLENDER" --version 2>/dev/null | head -1)
echo "Using: $BLENDER_VERSION"
echo ""

# --- Execute ---
START_TIME=$(date +%s)

"$BLENDER" --background --python run_vroid_render.py -- \
    --vroid_dir "$VROID_DIR" \
    --pose_dir "$POSE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --styles "$STYLES" \
    --resolution "$RESOLUTION" \
    --poses_per_character "$POSES_PER_CHAR" \
    --angles "$ANGLES" \
    $([ "$MAX_CHARS" -gt 0 ] && echo "--max_characters $MAX_CHARS")

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "============================================================"
echo "VRoid CC0 rendering complete!"
echo "Time: ${MINUTES}m ${SECONDS}s"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Validate: python3 run_validation.py --input_dir $OUTPUT_DIR"
echo "  2. Upload:   rclone copy $OUTPUT_DIR hetzner:strata-training-data/vroid_cc0/ --transfers 32 --checkers 64 --fast-list --size-only -P"
echo "============================================================"

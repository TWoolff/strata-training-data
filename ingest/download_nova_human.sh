#!/usr/bin/env bash
# Download NOVA-Human-Mirror-Purged from HuggingFace
# Source: https://huggingface.co/datasets/ljsabc/NOVA-Human-Mirror-Purged
# Credit: Dr. Chengze Li (ljsabc)
#
# Total: ~38 GB across all directories
# Saves to local SSD (data/nova_human/) — move to external HD later

set -euo pipefail

REPO="ljsabc/NOVA-Human-Mirror-Purged"
OUT_DIR="./data/nova_human"

mkdir -p "$OUT_DIR"

echo "============================================"
echo "  Downloading NOVA-Human-Mirror-Purged"
echo "  Repo: $REPO"
echo "  Destination: $OUT_DIR"
echo "  Total: ~38 GB"
echo "============================================"
echo ""

# Resolve HF CLI command (newer versions use 'hf', older use 'huggingface-cli')
if command -v huggingface-cli &>/dev/null; then
    HF_CLI="huggingface-cli"
elif command -v hf &>/dev/null; then
    HF_CLI="hf"
else
    echo "Installing huggingface_hub via pipx..."
    command -v pipx &>/dev/null || brew install pipx
    pipx install "huggingface_hub[cli]"
    HF_CLI="hf"
fi

# Download everything
echo "[1/7] Downloading ortho/ (~7 GB)..."
$HF_CLI download "$REPO" --repo-type dataset --include "ortho/*" --local-dir "$OUT_DIR"

echo ""
echo "[2/7] Downloading ortho_mask/ (~207 MB)..."
$HF_CLI download "$REPO" --repo-type dataset --include "ortho_mask/*" --local-dir "$OUT_DIR"

echo ""
echo "[3/7] Downloading rgb/ (~20 GB)..."
$HF_CLI download "$REPO" --repo-type dataset --include "rgb/*" --local-dir "$OUT_DIR"

echo ""
echo "[4/7] Downloading rgb_mask/ (~1 GB)..."
$HF_CLI download "$REPO" --repo-type dataset --include "rgb_mask/*" --local-dir "$OUT_DIR"

echo ""
echo "[5/7] Downloading xyza/ (~5.4 GB)..."
$HF_CLI download "$REPO" --repo-type dataset --include "xyza/*" --local-dir "$OUT_DIR"

echo ""
echo "[6/7] Downloading ortho_katepca_chonk/ (~4.7 GB)..."
$HF_CLI download "$REPO" --repo-type dataset --include "ortho_katepca_chonk/*" --local-dir "$OUT_DIR"

echo ""
echo "[7/7] Downloading metadata..."
$HF_CLI download "$REPO" --repo-type dataset --include "human_rutileE_meta.json" --include "README.md" --local-dir "$OUT_DIR"

echo ""
echo "============================================"
echo "  Download complete!"
echo "  Location: $OUT_DIR"
echo "  Next steps:"
echo "    1. Extract .exe archives (they are self-extracting 7z)"
echo "    2. Move to external HD when available"
echo "    3. Run ingest/nova_human_adapter.py"
echo "============================================"

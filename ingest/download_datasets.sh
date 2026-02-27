#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Download pre-processed external datasets for the Strata training pipeline.
#
# Usage:
#   ./ingest/download_datasets.sh all              # download everything
#   ./ingest/download_datasets.sh nova_human        # single dataset
#   ./ingest/download_datasets.sh nova_human stdgen  # multiple datasets
#
# Each dataset is downloaded to data/preprocessed/{dataset_name}/.
# Already-downloaded datasets are skipped (idempotent).
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$REPO_ROOT/data/preprocessed"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
error() { echo "[ERROR] $*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || error "'$1' is required but not installed."
}

skip_if_exists() {
    local dir="$1"
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null | head -1)" ]; then
        info "Skipping — already downloaded: $dir"
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
# Dataset download functions
# ---------------------------------------------------------------------------

download_nova_human() {
    local dest="$DATA_DIR/nova_human"
    skip_if_exists "$dest" && return 0

    info "Downloading NOVA-Human (~50-80 GB) ..."
    require_cmd git

    mkdir -p "$dest"
    # NOVA-Human is hosted via the NOVA-3D GitHub repo with links to
    # external storage.  Clone the repo first, then follow download
    # instructions from their README.
    if [ ! -d "$dest/.repo" ]; then
        git clone --depth 1 \
            https://github.com/NOVA-3D-Anime-Character-Synthesis/NOVA-3D.git \
            "$dest/.repo"
    fi

    info "NOVA-Human repo cloned to $dest/.repo"
    info "Follow the download instructions in $dest/.repo/README.md"
    info "to fetch the full dataset into $dest/"
}

download_stdgen() {
    local dest="$DATA_DIR/stdgen"
    skip_if_exists "$dest" && return 0

    info "Downloading StdGEN ..."
    require_cmd git

    mkdir -p "$dest"
    if [ ! -d "$dest/.repo" ]; then
        git clone --depth 1 \
            https://github.com/hyz317/StdGEN.git \
            "$dest/.repo"
    fi

    # StdGEN weights and data are on HuggingFace
    if command -v huggingface-cli >/dev/null 2>&1; then
        info "Downloading StdGEN data from HuggingFace ..."
        huggingface-cli download hyz317/StdGEN --local-dir "$dest/hf_data" || \
            warn "HuggingFace download failed — install huggingface-cli or download manually."
    else
        warn "huggingface-cli not found. Install with: pip install huggingface_hub[cli]"
        warn "Then run: huggingface-cli download hyz317/StdGEN --local-dir $dest/hf_data"
    fi

    info "StdGEN repo cloned to $dest/.repo"
}

download_animerun() {
    local dest="$DATA_DIR/animerun"
    skip_if_exists "$dest" && return 0

    info "Downloading AnimeRun (~5 GB) ..."
    require_cmd git

    mkdir -p "$dest"
    # AnimeRun data hosted on the project page with Google Drive / Baidu links.
    # Clone the repo for scripts and documentation.
    if [ ! -d "$dest/.repo" ]; then
        git clone --depth 1 \
            https://github.com/lisiyao21/AnimeRun.git \
            "$dest/.repo"
    fi

    info "AnimeRun repo cloned to $dest/.repo"
    info "Follow download links in $dest/.repo/README.md for the full dataset."
}

download_unirig() {
    local dest="$DATA_DIR/unirig"
    skip_if_exists "$dest" && return 0

    info "Downloading UniRig / Rig-XL (~20 GB) ..."
    require_cmd git

    mkdir -p "$dest"
    if [ ! -d "$dest/.repo" ]; then
        git clone --depth 1 \
            https://github.com/VAST-AI-Research/UniRig.git \
            "$dest/.repo"
    fi

    info "UniRig repo cloned to $dest/.repo"
    info "Follow the download instructions in $dest/.repo/README.md for Rig-XL dataset."
}

download_linkto_anime() {
    local dest="$DATA_DIR/linkto_anime"
    skip_if_exists "$dest" && return 0

    info "Downloading LinkTo-Anime (~10 GB) ..."
    require_cmd git

    mkdir -p "$dest"
    # LinkTo-Anime is referenced in arXiv 2506.02733.
    # Check the paper's project page for download links.
    info "LinkTo-Anime dataset download requires manual steps."
    info "See arXiv paper 2506.02733 for download links."
    info "Place downloaded files in: $dest/"
}

download_fbanimehq() {
    local dest="$DATA_DIR/fbanimehq"
    skip_if_exists "$dest" && return 0

    info "Downloading FBAnimeHQ (~25 GB) ..."

    mkdir -p "$dest"
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download skytnt/fbanimehq --local-dir "$dest" || \
            warn "HuggingFace download failed — check network or credentials."
    else
        warn "huggingface-cli not found. Install with: pip install huggingface_hub[cli]"
        warn "Then run: huggingface-cli download skytnt/fbanimehq --local-dir $dest"
    fi
}

download_anime_segmentation() {
    local dest="$DATA_DIR/anime_segmentation"
    skip_if_exists "$dest" && return 0

    info "Downloading anime-segmentation ..."

    mkdir -p "$dest"
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download skytnt/anime-segmentation --local-dir "$dest" || \
            warn "HuggingFace download failed — check network or credentials."
    else
        warn "huggingface-cli not found. Install with: pip install huggingface_hub[cli]"
        warn "Then run: huggingface-cli download skytnt/anime-segmentation --local-dir $dest"
    fi
}

download_anime_instance_seg() {
    local dest="$DATA_DIR/anime_instance_seg"
    skip_if_exists "$dest" && return 0

    info "Downloading anime-instance-segmentation ..."
    require_cmd git

    mkdir -p "$dest"
    if [ ! -d "$dest/.repo" ]; then
        git clone --depth 1 \
            https://github.com/dreMaz/AnimeInstanceSegmentationDataset.git \
            "$dest/.repo"
    fi

    info "Anime instance segmentation repo cloned to $dest/.repo"
    info "Follow download instructions in $dest/.repo/README.md for the full dataset."
}

download_charactergen() {
    local dest="$DATA_DIR/charactergen"
    skip_if_exists "$dest" && return 0

    info "Downloading CharacterGen ..."
    require_cmd git

    mkdir -p "$dest"
    if [ ! -d "$dest/.repo" ]; then
        git clone --depth 1 \
            https://github.com/zjp-shadow/CharacterGen.git \
            "$dest/.repo"
    fi

    info "CharacterGen repo cloned to $dest/.repo"
    info "Follow the download instructions in $dest/.repo/README.md for render data."
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

ALL_DATASETS=(
    nova_human
    stdgen
    animerun
    unirig
    linkto_anime
    fbanimehq
    anime_segmentation
    anime_instance_seg
    charactergen
)

download_dataset() {
    case "$1" in
        nova_human)          download_nova_human ;;
        stdgen)              download_stdgen ;;
        animerun)            download_animerun ;;
        unirig)              download_unirig ;;
        linkto_anime)        download_linkto_anime ;;
        fbanimehq)           download_fbanimehq ;;
        anime_segmentation)  download_anime_segmentation ;;
        anime_instance_seg)  download_anime_instance_seg ;;
        charactergen)        download_charactergen ;;
        *)                   error "Unknown dataset: $1. Available: ${ALL_DATASETS[*]}" ;;
    esac
}

usage() {
    echo "Usage: $0 <dataset_name> [dataset_name ...]"
    echo "       $0 all"
    echo ""
    echo "Available datasets:"
    for ds in "${ALL_DATASETS[@]}"; do
        echo "  $ds"
    done
    exit 1
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if [ $# -eq 0 ]; then
    usage
fi

mkdir -p "$DATA_DIR"

if [ "$1" = "all" ]; then
    info "Downloading all datasets ..."
    for ds in "${ALL_DATASETS[@]}"; do
        info "--- $ds ---"
        download_dataset "$ds"
        echo ""
    done
else
    for ds in "$@"; do
        info "--- $ds ---"
        download_dataset "$ds"
        echo ""
    done
fi

info "Done."

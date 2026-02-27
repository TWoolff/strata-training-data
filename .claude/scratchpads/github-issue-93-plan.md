# Issue #93: Create ingest/ directory with download_datasets.sh and nova_human_adapter.py

## Understanding

The issue asks to bootstrap the `ingest/` directory — the adapter layer for converting pre-processed external datasets into Strata's training format. This is the highest-ROI action per the refreshed PRD: ~355K images across 9 external datasets.

Type: **new feature** (greenfield directory and infrastructure)

Deliverables:
1. `ingest/__init__.py` — package marker
2. `ingest/download_datasets.sh` — master download script for all 9 datasets
3. `ingest/nova_human_adapter.py` — first adapter, converts NOVA-Human renders to Strata format
4. `data/preprocessed/{dataset}/README.md` — 9 subdirectories with README files
5. `.gitignore` updates for `data/preprocessed/`
6. Tests for `nova_human_adapter.py`

## Approach

### download_datasets.sh
- Bash script with functions per dataset
- Support `./download_datasets.sh all` or `./download_datasets.sh nova_human`
- Each dataset function: check dependencies (git, wget/curl), download to `data/preprocessed/{name}/`, verify with checksums where available
- Use `git clone` for GitHub repos, `wget`/`curl` for direct downloads, `huggingface-cli` for HF datasets
- Print clear progress messages and skip already-downloaded datasets

### nova_human_adapter.py
- Pure Python (no Blender dependency) — follows `spine_parser.py` pattern
- Input: NOVA-Human directory structure (per-character dirs with `ortho/`, `ortho_mask/`, `rgb/`, etc.)
- Output: Strata format (`image.png` + `segmentation.png` + `metadata.json`) under output dir
- NOVA-Human provides: ortho renders, masks, XYZA data. Does NOT provide Strata-specific 19-region segmentation, joint positions, or draw order
- Adapter converts available data and flags missing annotations in metadata (`"missing_annotations": ["joints", "draw_order", "strata_segmentation"]`)
- Use PIL for image resizing to 512×512
- Use exporter module for saving (following existing patterns)

### Data structure reference (NOVA-Human)
```
{character_id}/
├── ortho/              ← front/back orthographic renders
├── ortho_mask/         ← segmentation masks for ortho views
├── ortho_xyza/         ← position + alpha data
├── rgb/                ← 16 random-view RGB renders
├── rgb_mask/           ← masks for random views
├── xyza/               ← position + alpha for random views
└── {character_id}_meta.json
```

## Files to Create/Modify

### New files:
- `ingest/__init__.py` — empty package init
- `ingest/download_datasets.sh` — master download script
- `ingest/nova_human_adapter.py` — NOVA-Human → Strata converter
- `data/preprocessed/nova_human/README.md`
- `data/preprocessed/stdgen/README.md`
- `data/preprocessed/unirig/README.md`
- `data/preprocessed/animerun/README.md`
- `data/preprocessed/linkto_anime/README.md`
- `data/preprocessed/fbanimehq/README.md`
- `data/preprocessed/anime_segmentation/README.md`
- `data/preprocessed/anime_instance_seg/README.md`
- `data/preprocessed/charactergen/README.md`
- `tests/test_nova_human_adapter.py`

### Modified files:
- `.gitignore` — add `data/preprocessed/**` rules

## Risks & Edge Cases

- NOVA-Human directory structure may vary — adapter should handle missing subdirs gracefully
- Image resolution varies across datasets — resize to 512×512 consistently
- NOVA-Human masks are NOT Strata 19-region segmentation — they're foreground/background or part-based with different taxonomy. Adapter should store original masks and flag that Strata segmentation is missing
- Large file downloads — script should be idempotent (skip existing downloads)
- Some datasets require `huggingface-cli` — script should check for it

## Open Questions

- None — issue is well-specified with clear acceptance criteria

## Implementation Notes

### What was implemented
All 6 deliverables completed as planned:

1. **`ingest/__init__.py`** — Single-line docstring package marker.
2. **`ingest/download_datasets.sh`** — Bash script with per-dataset functions, `all` mode, idempotent (skips existing downloads), checks for required tools (`git`, `huggingface-cli`). Uses `git clone --depth 1` for GitHub repos, `huggingface-cli download` for HF datasets.
3. **`ingest/nova_human_adapter.py`** — Pure Python adapter (no Blender). Two main entry points: `convert_character()` and `convert_directory()`. Saves per-view output in `{char_id}_{view_name}/` subdirs with `image.png`, `segmentation.png` (fg/bg binary mask), and `metadata.json`. Flags missing Strata annotations in metadata. Supports `only_new`, `include_rgb`, `max_characters`, custom resolution.
4. **9 `data/preprocessed/*/README.md` files** — Each documents download URL, size, license, format, and Strata adapter status.
5. **`.gitignore`** — Added `data/preprocessed/**` and `!data/preprocessed/*/README.md`.
6. **`tests/test_nova_human_adapter.py`** — 27 tests across 5 test classes covering resize, mask conversion, metadata building, parsing, single-character conversion, and batch directory conversion.

### Design decisions
- **Did not use `pipeline/exporter.py`** — The issue's output format uses per-example subdirectories (`{char_id}_{view}/image.png`) rather than the pipeline's flat layout (`images/{char_id}_pose_00_flat.png`). Using exporter would have forced an awkward mapping between NOVA-Human views and the pose-indexed naming convention.
- **Binary mask stored as 0/255 PNG** — Matches visual expectation while metadata clearly documents it's fg/bg, not 19-region segmentation.
- **Ortho-only by default** — `include_rgb=False` default since orthographic views are the primary training data; 16 random perspective views per character would balloon output.
- **Code simplifier removed unused `_ORTHO_VIEWS` constant** and simplified `parse_character` data loading with `dict()` constructor.

### Test results
- 27/27 tests passing
- Linter clean (ruff check passes)

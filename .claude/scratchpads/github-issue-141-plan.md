# Issue #141: Build GitHub Live2D model scraper for training data collection

## Understanding
- Need a CLI script `run_live2d_scrape.py` that searches GitHub for repos containing Live2D model files (.moc3, .model3.json), filters by permissive license, downloads qualifying models via sparse git checkout, and organizes them into `data/live2d/live2d_NNN/` directories.
- Type: **new feature** (data acquisition tooling)
- This is the data collection step that feeds the existing Live2D pipeline (`live2d_renderer.py`, `live2d_mapper.py`, `live2d_review_ui.py`)

## Approach
- Single script `run_live2d_scrape.py` following the existing CLI patterns from `run_pipeline.py` / `run_validation.py`
- Uses `gh` CLI via subprocess for GitHub API access (handles auth, rate limits)
- Three search strategies: `extension:moc3`, `filename:model3.json live2d`, and repo search for `live2d model`
- Sparse checkout to download only model directories (not entire repos)
- CSV manifest at `data/live2d/labels/live2d_manifest.csv` — note this is separate from the mapper's `live2d_mappings.csv`
- Idempotent: tracks downloaded repos in manifest, skips on re-run

## Files to Modify
- **New:** `run_live2d_scrape.py` — the entire scraper script
- No modifications to existing files needed

## Key Design Decisions
1. **Manifest CSV naming**: Issue says `live2d_manifest.csv` — distinct from mapper's `live2d_mappings.csv`. Good separation.
2. **Model directory naming**: `live2d_NNN/` with zero-padded 3-digit IDs, continuing from highest existing ID
3. **`gh` CLI over `requests`**: Avoids token management, handles auth automatically
4. **Sparse checkout**: `git sparse-checkout` to download only model subdirectories, avoiding full repo clones
5. **Rate limiting**: 2.5s between search API calls, 0.5s between core API calls, respect 403/429 with retry

## Risks & Edge Cases
- GitHub search API may not find all repos (search index incomplete)
- Some repos may have models deeply nested — tree traversal handles this
- Large repos (>100MB) may slow down sparse checkout — could add size filter
- Rate limit reset parsing from `gh` CLI output (need to detect 403/429 from subprocess stderr)
- Model directories may have inconsistent structures — need flexible detection
- Some repos may list a license but individual model files have different terms — we can only check repo-level license
- Naming conflict: issue's manifest CSV vs mapper's CSV — using different filenames resolves this

## Open Questions
- None — the issue is very detailed and prescriptive

## Implementation Plan
1. Write `run_live2d_scrape.py` with:
   - Dataclasses for search results (`RepoInfo`, `ModelInfo`)
   - GitHub search functions (3 queries, pagination, dedup)
   - License filtering
   - File tree inspection for model directory detection
   - Sparse checkout download
   - Directory organization into `live2d_NNN/`
   - CSV manifest management
   - CLI with argparse (`--dry_run`, `--max_models`, `--include_unverified`, `--output_dir`)
2. Follow codebase conventions: `from __future__ import annotations`, type hints, pathlib, Google docstrings, logging

## Implementation Notes

### What was implemented
- Single file `run_live2d_scrape.py` (~530 lines after formatting) with all functionality in one module
- `RepoInfo` and `ModelInfo` dataclasses for structured data
- `_run_gh()` wrapper with exponential backoff retry on rate limits
- Three search strategies: code search for `extension:moc3`, code search for `filename:model3.json live2d`, repo search for `live2d model`
- `find_model_dirs()` — fetches full recursive tree, identifies dirs with model files + nearby PNGs
- `sparse_checkout_model()` — `git clone --filter=blob:none --no-checkout --depth=1` + sparse checkout for targeted download
- CSV manifest at `data/live2d/labels/live2d_manifest.csv` with columns matching the issue spec
- Idempotent: loads existing manifest, checks `repo_full_name:repo_path` combos, continues numbering from highest existing ID
- CLI: `--dry_run`, `--max_models`, `--include_unverified`, `--output_dir`

### Design decisions during implementation
- Rate limit detection via stderr parsing ("rate limit", "403", "429") since `gh` doesn't expose headers easily
- Exponential backoff: `SEARCH_DELAY_S * 2^attempt + RATE_LIMIT_BUFFER_S` instead of parsing `X-RateLimit-Reset`
- Model directory detection checks same-dir PNGs, texture subdirs (textures/parts/images), AND parent dir PNGs
- Manifest saved after each download for crash-resilience
- `_count_pngs()` counts both root PNGs and texture subdir PNGs for the `fragment_count` field
- Notes field in manifest stores the repo-relative model path for dedup on re-runs

### Lint/test results
- `ruff check` and `ruff format --check` both pass clean
- All 845 existing tests pass (1 pre-existing failure in `test_proportion_clusterer.py` due to missing sklearn)
- All 170 Live2D-specific tests pass

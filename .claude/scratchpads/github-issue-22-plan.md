# Issue #22: Implement train/val/test dataset split by character

## Understanding
- Split processed characters into train (80%) / val (10%) / test (10%) sets
- All images/masks/joints of a single character must go into the same split (prevent data leakage)
- Output `splits.json` at the dataset root with character IDs per split
- Deterministic via fixed random seed; stratify by source for proportional mix
- Support incremental updates: new characters assigned to maintain ratio balance
- Type: new feature (in the exporter/post-processing phase)

## Approach
- Create a new `pipeline/splitter.py` module (single responsibility, following manifest.py pattern)
- `generate_splits()` function: takes output_dir, discovers characters from `sources/` dir, groups by source, shuffles deterministically, slices proportionally, writes `splits.json`
- `save_splits()` in `exporter.py` is NOT needed — `splitter.py` writes its own file directly (same pattern as `manifest.py` which writes `manifest.json` directly)
- Integrate into `generate_dataset.py` alongside the `generate_manifest()` call
- For stratification: group characters by `source` field from `sources/*.json` metadata, proportionally assign from each source group to train/val/test
- For incremental updates: if `splits.json` already exists, load it, identify new characters, and assign only new ones to maintain ratio balance
- Use `SPLIT_RATIOS` from `config.py` (already defined but unused)
- Use seed=42 for deterministic shuffling

## Files to Modify
1. **`pipeline/splitter.py`** (NEW) — Core split logic:
   - `generate_splits(output_dir, *, seed=42) -> Path` — main entry point
   - Discovers character IDs from `sources/` directory
   - Groups by source for stratification
   - Handles incremental updates (reads existing splits.json)
   - Writes `splits.json`

2. **`pipeline/generate_dataset.py`** — Integration:
   - Import `generate_splits` from `.splitter`
   - Call after `generate_manifest()` at ~line 878

3. **`pipeline/__init__.py`** — May need to check if splitter needs to be listed (probably not, only imported internally)

## Risks & Edge Cases
- Very few characters (< 10): splits may not have exactly 80/10/10 — accept ±1 character
- All characters from one source: stratification degrades to simple shuffle, still works
- Source field missing in metadata: fall back to prefix inference (same pattern as manifest.py's `_infer_source_from_id`)
- Incremental run where existing splits.json has characters no longer in the dataset: preserve them in splits (don't remove absent characters, they may be temporarily missing)
- Edge case: 0 new characters on incremental run — just return existing splits unchanged

## Open Questions
- None — requirements are clear from the issue and PRD

## Implementation Notes
- Created `pipeline/splitter.py` as planned — single-responsibility module following `manifest.py` pattern
- Did NOT add `save_splits()` to `exporter.py` — the splitter handles its own I/O (consistent with how manifest.py works)
- `__init__.py` did not need changes — splitter is imported only by `generate_dataset.py`
- Used `random.Random(seed)` instance instead of `random.seed()` to avoid polluting global RNG state
- Stratification works per-source: each source group is shuffled independently, then proportionally sliced across train/val/test
- Incremental update assigns new characters one at a time to whichever split has the largest deficit vs target ratio
- Code simplifier applied: set comprehension for `assigned`, `max()` with key for `_most_underrepresented_split`, removed unused `Any` import
- `SPLIT_RATIOS` from `config.py` (already existed, lines 680–684) is now consumed by the splitter
- `SPLIT_SEED = 42` defined as module-level constant in `splitter.py`

# Issues #122, #123, #124: Training Infrastructure Setup

## Understanding
Three interconnected issues that create the training pipeline foundation:
- **#122**: Directory structure, requirements.txt, YAML configs, README
- **#123**: `training/data/transforms.py` — L/R flip-aware augmentation, ImageNet normalization, bone order constants
- **#124**: `training/data/split_loader.py` — Character-level dataset split loader (prevents data leakage)

All are infrastructure/foundation work — no model code yet.

## Approach

Build bottom-up: directory structure (#122) → transforms (#123) → split loader (#124).
Follow existing pipeline conventions exactly (pathlib, type hints, Google docstrings, `from __future__ import annotations`).

### Key Design Decisions
- **Transforms**: Use numpy for mask operations (no torch dependency for mask flipping). ImageNet normalization uses torch since it operates on tensors.
- **Split loader**: Reuse patterns from `pipeline/splitter.py` but simplified — no stratification by source (that's the pipeline's job). Focus on loading existing splits.json or generating fresh ones.
- **Character ID extraction**: Regex-based, handle Mixamo (`mixamo_001_pose_05_flat` → `mixamo_001`), FBAnimeHQ (`fbanimehq_0000_000005` → `fbanimehq_0000`), NOVA-Human (`nova_human_12345` → `nova_human_12345`).
- **BONE_ORDER**: Matches Rust's `joints.rs` exactly (20-slot list).
- **num_classes**: Issue says 22 (20 body + background + accessory), configs reflect this.

## Files to Create

### Issue #122 — Infrastructure
- `training/__init__.py` — empty
- `training/README.md` — setup, GPU requirements, training commands, ONNX contracts
- `training/requirements.txt` — torch, torchvision, onnx, etc.
- `training/configs/segmentation.yaml` — from issue spec
- `training/configs/joints.yaml` — joint prediction config
- `training/configs/weights.yaml` — weight prediction config
- `training/models/__init__.py` — empty
- `training/data/__init__.py` — empty
- `training/utils/__init__.py` — empty

### Issue #123 — Transforms
- `training/data/transforms.py` — flip_mask, flip_joints, normalize_imagenet, constants

### Issue #124 — Split Loader
- `training/data/split_loader.py` — load_or_generate_splits, character_id_from_example

### Tests
- `tests/test_training_transforms.py` — flip round-trip, swap pairs, edge cases
- `tests/test_split_loader.py` — split ratios, determinism, multi-directory, character ID extraction

## Risks & Edge Cases
- **Mask flip copy-before-swap**: Must use temporary array to avoid corrupting data when swapping L/R IDs
- **Character ID extraction**: Need robust regex that handles various naming conventions
- **BONE_ORDER accuracy**: Must match Rust runtime exactly — verify against issue spec
- **requirements.txt versions**: Use minimum versions that are current but not bleeding edge

## Open Questions
- None — issues are well-specified

## Implementation Notes

### What was implemented
All three issues completed as planned. No deviations from the plan.

### Design decisions made during implementation
- **Torch as optional dependency**: `transforms.py` uses `TYPE_CHECKING` guard for torch import at module level, with lazy `import torch` inside `normalize_imagenet()`. This lets numpy-only functions (`flip_mask`, `flip_joints`) work without torch installed.
- **LUT-based mask swapping**: Used a 256-element numpy lookup table (`lut[flipped]`) for region ID swapping — avoids the copy-before-swap issue entirely and is vectorized.
- **Test torch guard**: `TestNormalizeImagenet` class uses `@pytest.mark.skipif(not _HAS_TORCH)` so the rest of the test suite runs without torch.
- **Split loader discovery**: Scans `images/`, `sources/`, and `masks/` directories as fallbacks, matching the pipeline's `exporter.py` layout.

### Test results
- 36 passed, 3 skipped (torch normalization tests — torch not installed in dev environment)
- Lint clean: `ruff check` and `ruff format --check` pass

### Follow-up work
- Subsequent issues will add model architectures in `training/models/`
- Dataset loader classes will use `transforms.py` and `split_loader.py`
- `normalize_imagenet` tests need torch installed to run (CI should have it)

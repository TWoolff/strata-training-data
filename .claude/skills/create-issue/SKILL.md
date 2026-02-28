---
name: create-issue
description: This skill should be used when the user asks to "create an issue", "file a bug", "write a feature request", "report a problem", or says "/create-issue". It creates detailed GitHub issues for the Strata synthetic data pipeline.
user-invokable: true
argument-hint: "<feature or bug description>"
---

# Create GitHub Issue

Create a detailed GitHub issue for: $ARGUMENTS

## Analysis Phase

1. **Understand the Request**
   - Parse the feature/bug description
   - Identify affected pipeline modules (see full module list below)
   - Determine scope and complexity

2. **Research Codebase**
   - Find related modules and functions
   - Check existing patterns that could be leveraged
   - Identify constants and configuration involved in `config.py`

3. **Check for Related Work**
   - Search existing issues for duplicates
   - Review PRDs in `.claude/prd/` for context

## Pipeline Modules Reference

**Core pipeline (`pipeline/`):**
`generate_dataset.py`, `config.py`, `importer.py`, `bone_mapper.py`, `pose_applicator.py`, `renderer.py`, `style_augmentor.py`, `joint_extractor.py`, `weight_extractor.py`, `draw_order_extractor.py`, `exporter.py`, `manifest.py`, `splitter.py`, `validator.py`

**Source-specific (`pipeline/`):**
`vroid_importer.py`, `vroid_mapper.py`, `live2d_mapper.py`, `spine_parser.py`, `accessory_detector.py`, `measurement_ground_truth.py`

**External dataset adapters (`ingest/`):**
`nova_human_adapter.py`, `stdgen_semantic_mapper.py`, `stdgen_pipeline_ext.py`

**Key config.py constants:**
`REGION_COLORS`, `REGION_NAMES`, `NUM_REGIONS` (20), `MIXAMO_BONE_MAP`, `VRM_BONE_ALIASES`, `COMMON_BONE_ALIASES`, `VROID_MATERIAL_PATTERNS`, `SPINE_BONE_PATTERNS`, `LIVE2D_FRAGMENT_PATTERNS`, `STDGEN_SEMANTIC_CLASSES`, `CAMERA_ANGLES`, `ART_STYLES`, `RENDER_RESOLUTION`, `ACCESSORY_NAME_PATTERNS`, `FUZZY_KEYWORD_PATTERNS`, `SUBSTRING_KEYWORDS`

## Issue Creation

Create a GitHub issue with:

### Title
Clear, action-oriented title (e.g., "Implement fuzzy bone mapper for non-Mixamo skeletons" or "Fix segmentation mask anti-aliasing artifacts")

### Description

**Problem/Feature**
Describe what needs to be built or fixed.

**Affected Modules**
- List relevant pipeline modules from the reference above
- Note specific pipeline stages involved (Import -> Map -> Pose -> Render -> Style -> Export -> Validate -> Manifest)
- Identify constants that may need updates in `config.py`
- Note data structures affected (region mapping, joint JSON, weight JSON, manifest)

**Technical Approach**
- Suggested implementation strategy
- Key files to modify
- New files or modules needed
- Patterns to follow (bone mapping priority, majority vote for faces, emission shader setup)

**Acceptance Criteria**
- [ ] Specific, testable requirements
- [ ] Output images are 512x512 PNG with transparent background
- [ ] Segmentation masks are 8-bit grayscale with valid region IDs (0-19, 20 total)
- [ ] Joint JSON has all 19 joints within image bounds
- [ ] Pipeline runs headless (`blender --background`)
- [ ] Handles Mixamo and VRM naming conventions
- [ ] Constants defined in `config.py`
- [ ] Tests pass: `python -m pytest tests/ -v`
- [ ] Lint passes: `ruff check . && ruff format --check .`

**Code Standards**
- Type hints on function signatures
- `snake_case` for functions/variables, `ALL_CAPS` for constants
- `pathlib.Path` for file paths
- Google-style docstrings for public functions
- `from __future__ import annotations` at top of each module
- f-strings for formatting

**PRD Reference**
Note relevant sections from `.claude/prd/` and phase number

### Labels
Suggest appropriate labels:
- `bug` / `feature` / `enhancement`
- `importer` / `bone-mapper` / `renderer` / `style-augmentor` / `joint-extractor` / `weight-extractor` / `exporter` / `config` / `vroid-importer` / `vroid-mapper` / `live2d-mapper` / `spine-parser` / `accessory-detector` / `validator` / `manifest` / `splitter` / `draw-order` / `ingest`
- `phase-1` through `phase-6` (per PRD implementation plan)
- Priority level if clear

## Create Issue

```bash
gh issue create --title "Title" --body "Description"
```

Return the issue URL when complete.

---
name: create-issue
description: Create a detailed GitHub issue for a new feature or bug in the Strata synthetic data pipeline. Use when user wants to create an issue, report a bug, or document a feature request.
user-invokable: true
argument-hint: "<feature or bug description>"
---

# Create GitHub Issue

Create a detailed GitHub issue for: $ARGUMENTS

## Analysis Phase

1. **Understand the Request**
   - Parse the feature/bug description
   - Identify affected pipeline modules (importer, bone_mapper, renderer, style_augmentor, joint_extractor, weight_extractor, exporter)
   - Determine scope and complexity

2. **Research Codebase**
   - Find related modules and functions
   - Check existing patterns that could be leveraged
   - Identify constants and configuration involved in `config.py`

3. **Check for Related Work**
   - Search existing issues for duplicates
   - Review `.claude/prd/strata-synthetic-data-pipeline.md` for PRD context

## Issue Creation

Create a GitHub issue with:

### Title
Clear, action-oriented title (e.g., "Implement fuzzy bone mapper for non-Mixamo skeletons" or "Fix segmentation mask anti-aliasing artifacts")

### Description

**Problem/Feature**
Describe what needs to be built or fixed.

**Affected Modules**
- List relevant pipeline modules (e.g., `bone_mapper.py`, `renderer.py`, `style_augmentor.py`)
- Note specific pipeline stages involved (import → map → pose → render → style → export)
- Identify constants that may need updates in `config.py` (REGION_COLORS, RENDER_RESOLUTION, bone mapping tables)
- Note data structures affected (region mapping, joint JSON, weight JSON, manifest)

**Technical Approach**
- Suggested implementation strategy
- Key files to modify
- New files or modules needed
- Patterns to follow (bone mapping priority, majority vote for faces, emission shader setup)

**Acceptance Criteria**
- [ ] Specific, testable requirements
- [ ] Output images are 512×512 PNG with transparent background
- [ ] Segmentation masks are 8-bit grayscale with valid region IDs (0–17)
- [ ] Joint JSON has all 17 joints within image bounds
- [ ] Pipeline runs headless (`blender --background`)
- [ ] Handles Mixamo naming conventions
- [ ] Constants defined in `config.py`

**Code Standards**
- Type hints on function signatures
- `snake_case` for functions/variables, `ALL_CAPS` for constants
- `pathlib.Path` for file paths
- Google-style docstrings for public functions
- f-strings for formatting

**PRD Reference**
Note relevant sections from `.claude/prd/strata-synthetic-data-pipeline.md` and phase number

### Labels
Suggest appropriate labels:
- `bug` / `feature` / `enhancement`
- `importer` / `bone-mapper` / `renderer` / `style-augmentor` / `joint-extractor` / `weight-extractor` / `exporter` / `config`
- `phase-1` through `phase-5` (per PRD implementation plan)
- Priority level if clear

## Create Issue

```bash
gh issue create --title "Title" --body "Description"
```

Return the issue URL when complete.

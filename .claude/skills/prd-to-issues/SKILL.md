---
name: prd-to-issues
description: Parse a PRD markdown file and create GitHub issues for all features. Each issue is scoped to be completable by one developer in one work day. Use when starting a new project phase or breaking down a PRD into actionable tasks.
user-invokable: true
argument-hint: "<path to PRD file or phase name>"
---

# PRD to GitHub Issues

Parse the PRD file and create GitHub issues: $ARGUMENTS

If no file specified, check `.claude/prd/` directory and ask which phase to break down.

## Phase 1: Parse PRD

### Read and Analyze PRD
1. Read the PRD file completely (default: `.claude/prd/strata-synthetic-data-pipeline.md`)
2. Identify major sections and features
3. Note dependencies between features
4. Extract acceptance criteria and specifications

### Feature Extraction
For each feature area, identify:
- Core functionality required
- Pipeline modules affected (importer, bone_mapper, renderer, style_augmentor, joint_extractor, weight_extractor, exporter)
- Data structures needed (region mapping, joint JSON, weight JSON, bone mapping overrides)
- Blender API requirements (FBX import, armature, materials, render passes)
- Post-processing requirements (PIL/OpenCV style transforms)
- Dependencies on other features

## Phase 2: Scope Issues

### One-Day Rule
Each issue should be completable by one developer in one work day (~6-8 hours of focused work). This means:

**Too Large (split it):**
- "Implement full rendering pipeline" → Split into color render, segmentation render, joint extraction, weight extraction
- "Build all style augmentation" → Split into per-style issues (flat, cel, pixel art, painterly, sketch, unlit)
- "Import and normalize all characters" → Split into Mixamo importer, generic FBX importer, fuzzy bone mapper

**Right Size:**
- "Implement FBX import with scale/position normalization"
- "Build Mixamo bone name → Strata region mapping table"
- "Create Emission-based segmentation material assignment"
- "Implement orthographic camera auto-framing"
- "Build pixel art style post-processing (downscale + palette reduction)"
- "Extract per-vertex bone weights to JSON format"
- "Implement dataset train/val/test split by character"

**Too Small (combine it):**
- "Add head region" + "Add neck region" → "Define complete 17-region mapping with colors"
- "Create flat material" + "Create cel material" → "Implement render-time Blender shader styles (flat, cel, unlit)"

### Dependency Ordering
Order issues so dependencies are clear:
1. `config.py` — Region colors, bone mapping table, constants
2. FBX import and normalization (`importer.py`)
3. Bone name → Strata region mapping (`bone_mapper.py`)
4. Segmentation material assignment (`renderer.py` — materials)
5. Orthographic camera auto-framing (`renderer.py` — camera)
6. Color render pass (`renderer.py` — color)
7. Segmentation render pass (`renderer.py` — mask)
8. Joint position extraction (`joint_extractor.py`)
9. Weight data extraction (`weight_extractor.py`)
10. Pose library application (`pose_applicator.py`)
11. Render-time styles: flat, cel/toon, unlit (`renderer.py` — shaders)
12. Post-render styles: pixel art, painterly, sketch (`style_augmentor.py`)
13. Export and file organization (`exporter.py`)
14. Dataset manifest and splits (`exporter.py` — metadata)
15. Batch processing orchestrator (`generate_dataset.py`)
16. Validation and quality checks
17. Scale-up: fuzzy bone mapper for non-Mixamo characters
18. Scale-up: per-character override JSON for manual bone mapping

### Issue Sizing Guidelines

| Complexity | Examples | Target Hours |
|------------|----------|--------------|
| Small | Config constants, single render pass, one style transform | 2-4 hours |
| Medium | Bone mapper, camera auto-framing, joint extraction | 4-6 hours |
| Large (max) | Full batch orchestrator, fuzzy bone mapper with fallbacks | 6-8 hours |

If estimated > 8 hours, split into multiple issues.

## Phase 3: Create Issues

For each identified feature, create an issue following this format:

### Issue Template

```markdown
## Summary
One-line description of what this issue accomplishes.

## Background
Why this feature is needed and how it fits into the pipeline. Reference the relevant PRD section and phase.

## Requirements

### Functional Requirements
- [ ] Specific behavior 1
- [ ] Specific behavior 2
- [ ] Edge cases to handle

### Technical Requirements
- [ ] Pipeline module(s) to create or modify
- [ ] Constants to define in `config.py`
- [ ] Blender API features required (bpy operators, node trees, etc.)
- [ ] Must work in `blender --background` mode (no GUI)

### Output Requirements (if applicable)
- [ ] Image format and dimensions (512×512 RGBA PNG)
- [ ] Mask format (8-bit grayscale, pixel value = region ID 0–17)
- [ ] JSON schema for joint/weight data

## Implementation Notes
- Suggested approach
- Key files to modify/create
- Patterns to follow from existing code
- Blender API considerations

## Dependencies
- List issues that must be completed first
- Or "None — can start immediately"

## Acceptance Criteria
- [ ] All functional requirements met
- [ ] Pipeline runs headless without errors
- [ ] Output files match expected format and dimensions
- [ ] Automated validation checks pass
- [ ] Constants in `config.py`, type hints on functions
- [ ] Code follows project patterns (see CLAUDE.md)

## PRD Reference
Section N, Phase M: [phase title] — `.claude/prd/strata-synthetic-data-pipeline.md`
```

### Labels
Apply appropriate labels to each issue:
- **Type**: `feature`, `enhancement`, `infrastructure`, `bug`
- **Module**: `importer`, `bone-mapper`, `renderer`, `style-augmentor`, `joint-extractor`, `weight-extractor`, `exporter`, `config`, `orchestrator`
- **Priority**: `P0-critical`, `P1-high`, `P2-medium`, `P3-low`
- **Size**: `size-small`, `size-medium`, `size-large`
- **Phase**: `phase-1` through `phase-5` (per PRD §10)

### Milestones
Group issues into milestones matching the PRD implementation phases:

**Phase 1: Skeleton (Week 1)**
- FBX import + normalize
- Bone mapper (Mixamo hardcoded)
- Segmentation materials
- Orthographic camera
- Color + mask render
- Joint extraction
- Export to output directory

**Phase 2: Pose & Scale (Week 2)**
- Pose library integration
- Pose application loop
- Y-axis flip augmentation
- Scale variation
- Batch multi-character processing
- Weight extraction
- Progress logging + error handling

**Phase 3: Style Augmentation (Week 3)**
- Blender shaders: flat, cel/toon, unlit
- Post-render: pixel art, painterly, sketch
- Style application loop
- Training-time augmentation config

**Phase 4: Scale Up (Week 4)**
- Full Mixamo library download
- Sketchfab/Quaternius curation
- Fuzzy bone mapper
- Per-character override JSON
- Full batch run
- Manifest generation
- Train/val/test split
- Quality report

**Phase 5: 2D Source Integration (Week 5)**
- Spine project parser
- Manual annotation pipeline
- Mixed source dataset merging

## Phase 4: Create in GitHub

### Check for Existing Issues
```bash
gh issue list --state all
```

### Create Milestones (if needed)
```bash
gh api repos/{owner}/{repo}/milestones -f title="Phase 1: Skeleton Pipeline" -f state="open"
```

### Create Each Issue
```bash
gh issue create \
  --title "Issue title" \
  --body "Issue body..." \
  --label "feature,renderer,size-medium,phase-1"
```

### Create in Dependency Order
Create foundation issues first so they can be referenced as dependencies in later issues.

## Phase 5: Summary Report

After creating all issues, provide:

### Issues Created
| # | Title | Labels | Milestone | Dependencies |
|---|-------|--------|-----------|--------------|
| 1 | ... | ... | ... | None |
| 2 | ... | ... | ... | #1 |

### Recommended Order
Suggest the order to tackle issues based on dependencies.

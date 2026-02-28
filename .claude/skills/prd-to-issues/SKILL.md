---
name: prd-to-issues
description: This skill should be used when the user asks to "break down the PRD", "create issues from the PRD", "parse the PRD", "convert PRD to issues", or says "/prd-to-issues". It parses a PRD markdown file and creates GitHub issues scoped to one developer per work day.
user-invokable: true
argument-hint: "<path to PRD file or phase name>"
---

# PRD to GitHub Issues

Parse the PRD file and create GitHub issues: $ARGUMENTS

If no file specified, check `.claude/prd/` directory and ask which phase to break down.

## Phase 1: Parse PRD

### Read and Analyze PRD
1. Read the PRD file completely (check `.claude/prd/` for available PRDs)
2. Identify major sections and features
3. Note dependencies between features
4. Extract acceptance criteria and specifications

### Feature Extraction
For each feature area, identify:
- Core functionality required
- Pipeline modules affected:

  **Core pipeline:** `generate_dataset.py`, `config.py`, `importer.py`, `bone_mapper.py`, `pose_applicator.py`, `renderer.py`, `style_augmentor.py`, `joint_extractor.py`, `weight_extractor.py`, `draw_order_extractor.py`, `exporter.py`, `manifest.py`, `splitter.py`, `validator.py`

  **Source-specific:** `vroid_importer.py`, `vroid_mapper.py`, `live2d_mapper.py`, `spine_parser.py`, `accessory_detector.py`, `measurement_ground_truth.py`

  **Ingest adapters:** `nova_human_adapter.py`, `stdgen_semantic_mapper.py`, `stdgen_pipeline_ext.py`

- Data structures needed (region mapping with 20 IDs 0-19, joint JSON with 19 joints, weight JSON, bone mapping overrides)
- Blender API requirements (FBX import, VRM import, armature, materials, render passes, multi-angle camera)
- Post-processing requirements (PIL/OpenCV style transforms)
- Dependencies on other features

## Phase 2: Scope Issues

### One-Day Rule
Each issue should be completable by one developer in one work day (~6-8 hours of focused work). This means:

**Too Large (split it):**
- "Implement full rendering pipeline" -> Split into color render, segmentation render, joint extraction, weight extraction
- "Build all style augmentation" -> Split into per-style issues (flat, cel, pixel art, painterly, sketch, unlit)
- "Import and normalize all characters" -> Split into Mixamo importer, VRM importer, generic FBX importer, fuzzy bone mapper

**Right Size:**
- "Implement FBX import with scale/position normalization"
- "Build Mixamo bone name -> Strata region mapping table"
- "Create Emission-based segmentation material assignment"
- "Implement orthographic camera auto-framing"
- "Build pixel art style post-processing (downscale + palette reduction)"
- "Extract per-vertex bone weights to JSON format"
- "Implement dataset train/val/test split by character"
- "Add VRM/VRoid import with A-pose normalization"
- "Build StdGEN 4-class to Strata 20-class semantic mapper"

**Too Small (combine it):**
- "Add head region" + "Add neck region" -> "Define complete 20-region mapping (IDs 0-19) with colors"
- "Create flat material" + "Create cel material" -> "Implement render-time Blender shader styles (flat, cel, unlit)"

### Dependency Ordering
Order issues so dependencies are clear:
1. `config.py` — Region colors, bone mapping table, constants (20 regions, IDs 0-19)
2. FBX import and normalization (`importer.py`)
3. Bone name -> Strata region mapping (`bone_mapper.py`)
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
14. Dataset manifest and splits (`manifest.py`, `splitter.py`)
15. Batch processing orchestrator (`generate_dataset.py`)
16. Validation and quality checks (`validator.py`, `run_validation.py`)
17. Scale-up: fuzzy bone mapper for non-Mixamo characters
18. Scale-up: per-character override JSON for manual bone mapping
19. VRM/VRoid import and A-pose normalization (`vroid_importer.py`)
20. VRoid material-to-region mapping (`vroid_mapper.py`)
21. Live2D fragment-to-region mapping (`live2d_mapper.py`)
22. Spine 2D project parsing (`spine_parser.py`)
23. Draw order extraction (`draw_order_extractor.py`)
24. Accessory detection and hiding (`accessory_detector.py`)
25. Multi-angle camera rendering
26. External dataset adapters (`ingest/*.py`)

### Issue Sizing Guidelines

| Complexity | Examples | Target Hours |
|------------|----------|--------------|
| Small | Config constants, single render pass, one style transform | 2-4 hours |
| Medium | Bone mapper, camera auto-framing, joint extraction, VRoid mapper | 4-6 hours |
| Large (max) | Full batch orchestrator, fuzzy bone mapper with fallbacks, ingest adapter | 6-8 hours |

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
- [ ] Image format and dimensions (512x512 RGBA PNG)
- [ ] Mask format (8-bit grayscale, pixel value = region ID 0-19, 20 total)
- [ ] JSON schema for joint/weight data (19 joints, regions 1-19)

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
- [ ] Automated validation checks pass (`python run_validation.py`)
- [ ] Tests pass (`python -m pytest tests/ -v`)
- [ ] Lint passes (`ruff check . && ruff format --check .`)
- [ ] Constants in `config.py`, type hints on functions
- [ ] Code follows project patterns (see CLAUDE.md)

## PRD Reference
Section N, Phase M: [phase title] — `.claude/prd/`
```

### Labels
Apply appropriate labels to each issue:
- **Type**: `feature`, `enhancement`, `infrastructure`, `bug`
- **Module**: `importer`, `bone-mapper`, `renderer`, `style-augmentor`, `joint-extractor`, `weight-extractor`, `exporter`, `config`, `orchestrator`, `vroid-importer`, `vroid-mapper`, `live2d-mapper`, `spine-parser`, `accessory-detector`, `draw-order`, `manifest`, `splitter`, `validator`, `ingest`
- **Priority**: `P0-critical`, `P1-high`, `P2-medium`, `P3-low`
- **Size**: `size-small`, `size-medium`, `size-large`
- **Phase**: `phase-1` through `phase-6` (per PRD implementation plan)

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
- Live2D fragment mapper
- Manual annotation pipeline
- Mixed source dataset merging

**Phase 6: External Dataset Ingestion**
- NOVA-Human adapter (`ingest/nova_human_adapter.py`)
- StdGEN semantic mapper (`ingest/stdgen_semantic_mapper.py`)
- StdGEN pipeline extension (`ingest/stdgen_pipeline_ext.py`)
- VRM/VRoid import + VRoid mapper
- Download/verification scripts
- Cross-source dataset merging and validation

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

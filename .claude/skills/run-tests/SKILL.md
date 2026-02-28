---
name: run-tests
description: Run the test suite for the Strata training data pipeline. Use when the user asks to run tests, verify changes, or check for regressions.
user-invokable: true
argument-hint: "<optional: specific test file, module name, or pattern>"
---

# Run Tests

Run the test suite: $ARGUMENTS

If no argument provided, run all tests. If a module name or pattern is given, run matching tests only.

## Quick Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_fuzzy_bone_mapper.py -v

# Run tests matching a keyword pattern
python -m pytest tests/ -k "vroid" -v

# Run with short output (no verbose)
python -m pytest tests/

# Run and stop on first failure
python -m pytest tests/ -x

# Run with coverage (if pytest-cov installed)
python -m pytest tests/ --cov=pipeline --cov=ingest -v
```

## Test Files

| Test File | Module Under Test | What It Tests |
|-----------|-------------------|---------------|
| `test_fuzzy_bone_mapper.py` | `pipeline/bone_mapper.py` | Fuzzy bone name matching (prefix-strip, substring, fuzzy keyword) |
| `test_live2d_mapper.py` | `pipeline/live2d_mapper.py` | Live2D ArtMesh fragment -> Strata label pattern matching |
| `test_vroid_importer.py` | `pipeline/vroid_importer.py` | VRM/VRoid import, A-pose normalization |
| `test_vroid_mapper.py` | `pipeline/vroid_mapper.py` | VRoid material slot -> Strata region mapping |
| `test_bvh_parser.py` | `animation/scripts/bvh_parser` | BVH mocap file parsing |
| `test_bvh_to_strata.py` | `animation/scripts/bvh_to_strata` | BVH retargeting to Strata 19-bone skeleton |
| `test_nova_human_adapter.py` | `ingest/nova_human_adapter.py` | NOVA-Human dataset -> Strata format conversion |
| `test_stdgen_semantic_mapper.py` | `ingest/stdgen_semantic_mapper.py` | StdGEN 4-class -> Strata 20-class mapping |
| `test_blueprint_exporter.py` | Blueprint export module | Blueprint export format correctness |
| `test_proportion_normalizer.py` | Proportion normalization | Character proportion normalization |

## Testing Patterns

- **Pure Python**: All tests run without Blender — bpy is mocked where needed
- **Imports**: Tests import from `pipeline.config` for constants (REGION_NAMES, NUM_REGIONS, etc.)
- **Naming**: Test files follow `tests/test_{module_name}.py`
- **Framework**: pytest (no unittest required, though compatible)
- **Assertions**: Use plain `assert` statements (pytest-style)

## Full Quality Check

After running tests, also verify linting:

```bash
# Lint check
ruff check .

# Format check
ruff format --check .

# Dataset validation (if output exists)
python run_validation.py --dataset_dir ./output/segmentation/
```

## Writing New Tests

When adding a test for a new module:

1. Create `tests/test_{module_name}.py`
2. Import the module under test and constants from `pipeline.config`
3. Test edge cases: empty inputs, malformed data, boundary values
4. For bone/label mapping tests: test exact matches, prefix-stripped matches, substring matches, and unrecognized names
5. For ingest adapters: test with minimal fixture data, verify output structure matches Strata format
6. Keep tests fast — no file I/O or network calls unless testing I/O specifically (use tmp_path fixture)

## Output

Report:
- Number of tests run, passed, failed, skipped
- Any failures with tracebacks
- Lint/format issues if found

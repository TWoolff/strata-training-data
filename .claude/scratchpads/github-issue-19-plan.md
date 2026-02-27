# Issue #19: Implement fuzzy bone mapper for non-Mixamo character skeletons

## Understanding
- Extend `bone_mapper.py` with fuzzy keyword matching that handles non-Mixamo bone naming
  conventions (Blender-style, generic, artist-custom)
- The existing chain is: override → exact (Mixamo) → alias → prefix-strip → substring
- Need to add a **fuzzy keyword** step between substring and "unmapped"
- Key difference from existing substring: fuzzy normalizes the bone name first
  (strips prefixes, splits camelCase, tokenizes on delimiters) then scores keyword matches
- Must avoid false positives (e.g., `"leg"` matching as left-side `"l"`)

## Approach
- Add a new step `_try_fuzzy_keyword` to the matching chain, after substring match
- Name normalization: strip known prefixes → split camelCase → split on `_`, `.`, `-`, ` ` → lowercase tokens
- Laterality detection with word boundaries: `"l"` only at token boundaries (not inside "leg"),
  check for `.l`, `_l`, `"left"` as whole tokens
- Define `FUZZY_KEYWORD_PATTERNS` in config.py: list of `(frozenset_of_keywords, region_id)` tuples
- Score = count of matched keywords / total keywords in pattern; require >= 0.6
- If multiple regions match, pick highest score; ties broken by more-specific pattern (more keywords)
- Track `fuzzy` count in MappingStats; log every fuzzy match for manual review

## Files to Modify
1. **`pipeline/config.py`**
   - Add `FUZZY_KEYWORD_PATTERNS: list[tuple[tuple[str, ...], RegionId]]` — keyword tuples per region
   - Add `FUZZY_MIN_SCORE: float = 0.6`
   - Extend `COMMON_PREFIXES` with any missing prefixes (Bone_, etc.)

2. **`pipeline/bone_mapper.py`**
   - Add `_normalize_bone_name(name: str) -> list[str]` — prefix strip, camelCase split, tokenize
   - Add `_detect_laterality(tokens: list[str]) -> str | None` — returns "l", "r", or None
   - Add `_try_fuzzy_keyword(bone_name: str) -> tuple[RegionId | None, float]` — score-based match
   - Add `fuzzy: int = 0` field to `MappingStats`
   - Insert fuzzy step after substring in `_map_all_bones`
   - Update logging in `map_bones` to include fuzzy count
   - Log each fuzzy match at DEBUG level for review

## Risks & Edge Cases
- `"leg"` substring matching `"l"` (left) — mitigate with token-boundary laterality detection
- Bones like `"DEF-spine.001"` — prefix stripping + number suffix stripping needed
- Very short bone names (e.g., `"L"`, `"R"`) — too ambiguous, should not match
- Multiple high-scoring matches — resolved by picking highest score, then most-specific pattern
- Existing substring step already catches many cases — fuzzy is the safety net

## Open Questions
- None — the issue spec and existing code are clear enough to proceed

## Implementation Notes
- Used `_canonicalize_laterality` instead of `_detect_laterality` — simpler approach that
  replaces whole-token aliases (`"l"` → `"left"`, `"r"` → `"right"`) via `LATERALITY_ALIASES`
  dict, then matches against patterns using canonical `"left"`/`"right"` keywords
- `LATERALITY_ALIASES` only maps `"l"` and `"r"` — tokens already named `"left"`/`"right"`
  pass through unchanged via `.get(t, t)` default
- Added `Bone_`, `bone_`, `RIG_`, `rig_` to `COMMON_PREFIXES`
- Added `("hips",)` pattern alongside `("hip",)` — normalization produces `"hips"` from
  `"Hips"` (no stemming), so both forms are needed
- `_NUMERIC_SUFFIX_RE` strips `.001`, `_02` etc. from bone names — critical for Blender rigs
  that use numeric suffixes for duplicate bones
- 39 tests covering normalization, laterality, and end-to-end matching (22 bone name variants)
- All fuzzy matches logged at DEBUG level for manual review

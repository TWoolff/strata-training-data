# Issue #3: Build Mixamo bone name to Strata region mapper

## Understanding
- Create `bone_mapper.py` that maps every bone in an imported armature to one of Strata's **19 body regions** (+ background = 20 total)
- **Key change from original issue**: Added `shoulder_l` (region 18) and `shoulder_r` (region 19) for cleaner 1:1 bone-to-region mapping. The deltoid/clavicle area gets its own region.
- Mapping priority chain: exact match → prefix match → substring match → manual override JSON
- Also produces per-vertex region assignment via dominant bone weight
- Mixamo characters should map 100% automatically; non-Mixamo ~80% auto

## Updated Region Table (20 total)

| ID | Region | ID | Region |
|----|--------|----|--------|
| 0 | background | 10 | lower_arm_r |
| 1 | head | 11 | hand_r |
| 2 | neck | 12 | upper_leg_l |
| 3 | chest | 13 | lower_leg_l |
| 4 | spine | 14 | foot_l |
| 5 | hips | 15 | upper_leg_r |
| 6 | upper_arm_l | 16 | lower_leg_r |
| 7 | lower_arm_l | 17 | foot_r |
| 8 | hand_l | 18 | shoulder_l |
| 9 | upper_arm_r | 19 | shoulder_r |

## Approach
- **bone_mapper.py**: Single module with one public function `map_bones()` that takes an armature + meshes and returns a `BoneMapping` dataclass
- **Mapping chain**: Try each strategy in order, stop on first match:
  1. **Manual override**: Load `source_characters/{character_id}_overrides.json` if it exists
  2. **Exact match**: Look up full bone name in `MIXAMO_BONE_MAP`
  3. **Alias match**: Look up full bone name in `COMMON_BONE_ALIASES`
  4. **Prefix strip**: Strip known prefixes (`mixamorig:`, `Bip01_`, `Bip001_`, `def_`, `DEF-`) and retry exact + alias
  5. **Substring match**: Case-insensitive keyword matching (e.g., contains "shoulder" + "left" → 18)
- **Vertex assignment**: For each vertex, find the vertex group with highest weight, look up its bone in `bone_to_region`. Vertices with no weights → region 0.
- **Shoulder mapping**: `mixamorig:LeftShoulder` → region 18 (`shoulder_l`), `mixamorig:RightShoulder` → region 19 (`shoulder_r`). Previously these were lumped into upper_arm regions.

## Files to Modify

### `config.py`
- Add regions 18 (`shoulder_l`) and 19 (`shoulder_r`) to `REGION_NAMES`, `REGION_COLORS`
- Update `NUM_REGIONS` from 18 → 20
- Update `MIXAMO_BONE_MAP`: move `LeftShoulder` from region 6 → 18, `RightShoulder` from region 9 → 19
- Update `COMMON_BONE_ALIASES`: move shoulder aliases to regions 18/19
- Add `COMMON_PREFIXES: list[str]` for prefix stripping
- Add `SUBSTRING_KEYWORDS: dict[str, RegionId]` for substring matching

### `bone_mapper.py` (new file)
- `BoneMapping` dataclass: `bone_to_region`, `vertex_to_region`, `unmapped_bones`, `mapping_stats`
- `map_bones(armature, meshes, character_id)` → `BoneMapping`
- Internal helpers: `_try_exact()`, `_try_alias()`, `_try_prefix_strip()`, `_try_substring()`, `_load_overrides()`
- `_assign_vertices(meshes, bone_to_region)` → vertex_to_region dict

### `CLAUDE.md`
- Update the region table to show 19 regions + background (20 total)

## Risks & Edge Cases
- **Finger bones**: Must map to hand regions (8/11), not get confused with substring "hand"
- **Toe bones**: Must map to foot regions (14/17)
- **Ambiguous substring matches**: "arm" appears in "forearm" — need to check for "fore" first, or use more specific keywords
- **Vertices with no bone weights**: Assign to region 0 (background). Log a warning if count > 0.
- **Override JSON missing**: Silently skip, not an error
- **Override JSON malformed**: Log warning, skip the override
- **Multiple meshes**: Process each mesh independently for vertex assignment
- **Shoulder vs upper_arm ambiguity**: Some rigs may name the clavicle bone something like "arm_upper" — substring matching needs careful keyword ordering

## Open Questions
- None — the 19-region scheme is confirmed by the user.

## Implementation Notes
- Implemented as planned with one simplification: `_try_prefix_strip()` returns `RegionId | None` instead of a tuple — the stripped name was never used by the caller
- `_try_prefix_strip` looks up the stripped name in both `MIXAMO_BONE_MAP` and `COMMON_BONE_ALIASES` directly (no redundant re-lookup of the original prefixed name, which `_try_exact` already handles)
- Vertex assignment uses composite keys (`mesh_index * 10_000_000 + vertex_index`) to uniquely identify vertices across multiple meshes
- `SUBSTRING_KEYWORDS` ordering is critical: more specific patterns (e.g., "forearm", "shin") come before general ones ("arm", "leg") to prevent false matches
- Shoulder substring patterns placed before arm patterns to avoid "shoulder" being caught by "arm" rules
- `COMMON_BONE_ALIASES` expanded with clavicle variants (`clavicle.L`, `L_clavicle`, etc.) for non-Mixamo rigs
- All 20 region IDs validated programmatically against config constants

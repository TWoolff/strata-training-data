# Issue #88: Create VRoid label mappings CSV

## Understanding
- Create a hand-curated VRoid label mappings CSV at `data/vroid/labels/vroid_mappings.csv`
- This is a **seed data file** — pre-populated with known VRoid material → Strata label mappings
- Captures override entries for edge cases that don't map cleanly through the standard regex patterns in `VROID_MATERIAL_PATTERNS`
- Type: infrastructure / data file creation (not code changes)

## Approach
- Use the CSV schema already defined by `vroid_mapper.py` → `CSV_HEADER`: `model_id,material_name,strata_label,strata_region_id,confirmed`
- Add an `override_reason` column as the issue requests — this extends the existing schema but `load_csv()` in `vroid_mapper.py` uses `csv.DictReader` so extra columns won't break it
- Seed with the standard VRoid Studio material slots (Face, EyeWhite, EyeIris, Hair, Body, etc.)
- Add known edge case overrides: shoes (L/R), hair extensions, outfit accessories, generic "Body" material
- Mark standard auto-mappable materials as `confirmed=auto` and edge cases as `confirmed=manual`

## Schema Decision
The issue requests: `material_name,bone_name,strata_label,override_reason,confirmed`

But the existing `vroid_mapper.py` CSV schema is: `model_id,material_name,strata_label,strata_region_id,confirmed`

I'll reconcile these by using a schema that:
1. Matches the existing `vroid_mapper.py` export format (so `load_csv()` still works)
2. Adds the `override_reason` column from the issue requirements
3. Uses `_default` for `model_id` to indicate these are default mappings (not per-model)

Final schema: `model_id,material_name,strata_label,strata_region_id,confirmed,override_reason`

## Files to Modify
- `data/vroid/labels/vroid_mappings.csv` — **CREATE** — the main deliverable
- No code changes needed — existing `vroid_mapper.py` can already read this file

## Risks & Edge Cases
- `load_csv()` in `vroid_mapper.py` uses `csv.DictReader` — extra columns are silently ignored, so adding `override_reason` won't break anything
- The "Body" material is a known problem — it covers the entire mesh and maps to "chest" as a coarse fallback. This should be documented in the CSV.
- Shoe/foot materials without L/R suffix need `disambiguate_lr()` at runtime — the CSV can only note this, not solve it

## Open Questions
- None — the issue comment clarifies the location (`data/vroid/labels/`) and the mapper code defines the schema clearly.

## Implementation Notes
- Created `data/vroid/labels/vroid_mappings.csv` with 41 entries covering all standard VRoid Studio materials plus edge case overrides
- Schema: `model_id,material_name,strata_label,strata_region_id,confirmed,override_reason` — compatible with existing `vroid_mapper.load_csv()` (extra `override_reason` column silently ignored by `csv.DictReader`)
- Used `_default` as `model_id` to indicate these are default mappings applicable to all VRoid models
- Breakdown: 23 auto-confirmed standard materials, 18 manual overrides
- Key edge cases documented: Body (coarse full-mesh fallback), Shoe/Boot/Sock/Glove (L/R disambiguation needed at runtime), Hair_Extension (may need future `hair_back` label), accessories (all mapped to background for v1)
- Removed `.gitkeep` from `data/vroid/labels/` since directory now has real content
- Verified: all region IDs valid (0-19), `load_csv()` compatibility confirmed, ruff check passes

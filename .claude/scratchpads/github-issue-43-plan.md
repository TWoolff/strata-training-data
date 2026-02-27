# Issue #43: Write taxonomy comparison document (Strata 20 vs See-through 19)

## Understanding
- Create `docs/taxonomy-comparison.md` mapping Strata's 20-class body region taxonomy against See-through's 19-class taxonomy
- This is a documentation/research task, not a code change
- Decision gate: determines whether any region changes needed before large-scale data generation
- PRD Reference: Research PRD §4.2–4.4

## See-through 19-class taxonomy (extracted from paper figures + GUI)
1. hair
2. headwear
3. face
4. eyes
5. eyewear
6. ears
7. earwear
8. nose
9. mouth
10. neck
11. neckwear
12. topwear
13. handwear
14. bottomwear
15. legwear
16. footwear
17. tail
18. wings
19. objects

Plus "None" as fallback (not counted as a class).
No background class — character-only taxonomy.

## Strata 20-class taxonomy (from config.py)
0. background
1. head
2. neck
3. chest
4. spine
5. hips
6. upper_arm_l
7. lower_arm_l
8. hand_l
9. upper_arm_r
10. lower_arm_r
11. hand_r
12. upper_leg_l
13. lower_leg_l
14. foot_l
15. upper_leg_r
16. lower_leg_r
17. foot_r
18. shoulder_l
19. shoulder_r

## Key Differences
- **Philosophy**: See-through is clothing-aware + anime-specific (face detail, accessories).
  Strata is anatomy-aware + rigging-focused (bones, limb segments, L/R distinction).
- **See-through has, Strata doesn't**: eyewear, earwear, headwear, neckwear, topwear,
  bottomwear, legwear, footwear, handwear, eyes, mouth, nose, tail, wings, objects
- **Strata has, See-through doesn't**: background, chest, spine, hips, shoulder_l/r,
  upper_arm/lower_arm/upper_leg/lower_leg (all L/R), separate foot_l/foot_r
- **Both have**: hair/head, neck, hand (as handwear vs hand_l/r), face, ears

## Approach
- Write all 7 sections required by the issue acceptance criteria
- Focus on practical implications for Strata's animation use case
- Recommend Option C for hair_back (defer as optional detail region)
- Reference issue #24 for accessory handling

## Files to Modify
- NEW: `docs/taxonomy-comparison.md`

## Risks & Edge Cases
- None — pure documentation task

## Implementation Notes
- See-through 19-class list was extracted by visually inspecting paper figures (13–19) and the GUI screenshot (Figure 10). The paper never enumerates all 19 classes in text — they're only shown in figure labels and the annotation GUI dropdown.
- The user provided the final missing classes (eyewear, wings) from Figure 10's GUI dropdown that was partially visible in the PDF render.
- Document covers all 7 sections required by acceptance criteria.
- Key decision: no changes to Strata's 20-region taxonomy for v1. The two taxonomies serve fundamentally different purposes (skeletal anatomy vs visual appearance).
- Hair layering deferred as Option C (optional detail region for future work).
- Accessory handling deferred to issue #24.

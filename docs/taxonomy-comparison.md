# Taxonomy Comparison: Strata 20-Region vs See-through 19-Class

**Date:** 2026-02-27
**Status:** Decision gate — determines whether region changes are needed before large-scale data generation
**References:** See-through paper (arXiv:2602.03749, Feb 2026), Strata config.py, Research PRD §4.2–4.4

---

## 1. Side-by-Side Table

### Strata Standard Skeleton (20 regions, IDs 0–19)

| ID | Strata Region | See-through Equivalent | Notes |
|----|---------------|----------------------|-------|
| 0 | background | *(none)* | See-through is character-only, no background class |
| 1 | head | hair + face + eyes + eyewear + ears + earwear + nose + mouth + headwear | See-through decomposes the head into 9 fine-grained classes |
| 2 | neck | neck + neckwear | See-through separates neck skin from neckwear (scarves, chokers, etc.) |
| 3 | chest | topwear *(partial)* | Strata = upper torso bone region; See-through = visible clothing covering torso |
| 4 | spine | topwear *(partial)* | Strata = mid-torso bone region; See-through doesn't distinguish chest vs spine |
| 5 | hips | bottomwear *(partial)* | Strata = pelvis bone region; See-through = clothing (skirts, pants waistband) |
| 6 | upper_arm_l | topwear *(sleeves)* or handwear *(gloves)* | Strata = left upper arm bone; See-through = whatever covers it |
| 7 | lower_arm_l | topwear *(sleeves)* or handwear | Same pattern — bone vs clothing |
| 8 | hand_l | handwear | Strata = left hand bone region; See-through = gloves/hand covering |
| 9 | upper_arm_r | topwear *(sleeves)* or handwear | Mirror of upper_arm_l |
| 10 | lower_arm_r | topwear *(sleeves)* or handwear | Mirror of lower_arm_l |
| 11 | hand_r | handwear | Mirror of hand_l |
| 12 | upper_leg_l | bottomwear or legwear | Strata = left thigh bone; See-through = pants/stockings covering it |
| 13 | lower_leg_l | legwear or bottomwear | Strata = left shin bone; See-through = stockings/pants |
| 14 | foot_l | footwear | Strata = left foot bone; See-through = shoes/boots |
| 15 | upper_leg_r | bottomwear or legwear | Mirror of upper_leg_l |
| 16 | lower_leg_r | legwear or bottomwear | Mirror of lower_leg_l |
| 17 | foot_r | footwear | Mirror of foot_l |
| 18 | shoulder_l | topwear *(partial)* | Strata = left clavicle bone; See-through has no shoulder class |
| 19 | shoulder_r | topwear *(partial)* | Mirror of shoulder_l |

### See-through Taxonomy (19 classes, extracted from paper GUI — Figure 10)

| # | See-through Class | Strata Equivalent | Notes |
|---|------------------|-------------------|-------|
| 1 | hair | head (ID 1) | Strata merges all hair into head region |
| 2 | headwear | head (ID 1) | Hats, ribbons, hair accessories — Strata absorbs into head |
| 3 | face | head (ID 1) | Strata doesn't separate face from head |
| 4 | eyes | head (ID 1) | Fine-grained facial region — irrelevant for body rigging |
| 5 | eyewear | head (ID 1) | Glasses — Strata absorbs into head |
| 6 | ears | head (ID 1) | Strata absorbs into head |
| 7 | earwear | head (ID 1) | Earrings, ear accessories — Strata absorbs into head |
| 8 | nose | head (ID 1) | Fine-grained facial region |
| 9 | mouth | head (ID 1) | Fine-grained facial region |
| 10 | neck | neck (ID 2) | Direct match |
| 11 | neckwear | neck (ID 2) | Scarves, chokers — Strata absorbs into neck |
| 12 | topwear | chest (3) + spine (4) + shoulder_l/r (18/19) + upper_arm (6/9) | See-through = clothing item; Strata = multiple bone regions underneath |
| 13 | handwear | hand_l (8) + hand_r (11) + lower_arm (7/10) | Gloves, arm covers |
| 14 | bottomwear | hips (5) + upper_leg (12/15) | Skirts, pants, shorts |
| 15 | legwear | lower_leg (13/16) + upper_leg (12/15) | Stockings, leggings, thigh-highs |
| 16 | footwear | foot_l (14) + foot_r (17) | Shoes, boots |
| 17 | tail | *(none)* | Animal/fantasy tails — Strata v1 is humanoid-only |
| 18 | wings | *(none)* | Fantasy wings — Strata v1 is humanoid-only |
| 19 | objects | *(none)* | Held items, props — Strata doesn't label objects |

---

## 2. Classes Present in See-through but Not Strata

### Clothing/Accessory Decomposition (9 classes)
See-through has dedicated classes for **headwear**, **eyewear**, **earwear**, **neckwear**, **topwear**, **handwear**, **bottomwear**, **legwear**, and **footwear**. Strata has none — clothing is assigned to the bone region underneath it. A shirt sleeve on the upper arm is labeled `upper_arm_l`, not "topwear."

**Why See-through needs these:** Their goal is layer decomposition for 2.5D animation. Each clothing item is a separate drawable layer that moves independently. A shirt needs to be its own layer so it can deform differently from the arm underneath.

**Why Strata doesn't need these (for now):** Strata's goal is skeleton-based rigging. The segmentation model needs to know which bone drives each pixel, not which garment covers it. A shirt sleeve and the arm underneath both follow the upper_arm bone. For v1, clothing classes add complexity without improving rig quality.

**Future consideration:** If Strata adds Live2D-style layer decomposition (Research PRD §2), clothing-aware classes become valuable. Defer to that phase.

### Facial Detail Classes (4 classes)
See-through separates the head into **face**, **eyes**, **nose**, and **mouth**. Strata collapses everything into a single `head` region.

**Why See-through needs these:** Anime characters have exaggerated facial features (large eyes, small noses). For talking-head VTubing, eyes and mouth must be independent layers with separate deformation controls.

**Why Strata doesn't need these:** Strata drives head rotation from a single head bone. Sub-face regions don't improve rig quality — the whole face moves as one unit. The exception would be facial animation (blend shapes), which is out of scope for v1.

### Non-Humanoid Classes (2 classes)
- **tail** — animal/fantasy character tails
- **wings** — fantasy character wings

Strata v1 targets humanoid bipeds only. These map to no Strata region. When non-humanoid support is added (future work per CLAUDE.md), these would need new region IDs.

### Held Objects (1 class)
**objects** — weapons, props, food, phones, etc. Strata's pipeline currently hides accessories for cleaner training data (CLAUDE.md: "Accessories: hide for v1"). See issue #24 for accessory handling strategy.

---

## 3. Classes Present in Strata but Not See-through

### Background (1 class)
Strata ID 0 = `background`. See-through operates on pre-segmented characters with no background. This is a non-issue — background is a pipeline concern, not a taxonomy disagreement.

### Anatomical L/R Distinction (12 classes)
Strata distinguishes left vs right for every limb: `upper_arm_l`/`upper_arm_r`, `lower_arm_l`/`lower_arm_r`, `hand_l`/`hand_r`, `upper_leg_l`/`upper_leg_r`, `lower_leg_l`/`lower_leg_r`, `foot_l`/`foot_r`. See-through has none of these — `handwear` covers both hands, `legwear` covers both legs.

**Why Strata needs L/R:** Skeleton rigging requires knowing which bone drives each pixel. The left arm and right arm are driven by different bones, so they must be different regions. Flip augmentation (swapping L/R labels during training) also requires explicit L/R labels.

**Why See-through doesn't:** Their decomposition only needs to separate the arm as a layer — it doesn't matter which arm since both can use the same deformation logic.

### Shoulder Regions (2 classes)
Strata has `shoulder_l` (ID 18) and `shoulder_r` (ID 19) for the clavicle bones. See-through has no shoulder class — the shoulder area falls under `topwear` or is implicitly part of the torso.

**Why Strata needs these:** The clavicle/shoulder bone is a separate joint in the skeleton. Raising an arm involves shoulder rotation distinct from chest rotation. Correct weight painting requires knowing which pixels follow the shoulder bone vs the chest bone.

### Torso Subdivision (3 classes)
Strata splits the torso into `chest` (ID 3), `spine` (ID 4), and `hips` (ID 5). See-through's closest equivalents are `topwear` (covers the whole upper body) and `bottomwear` (covers the hip area), but these are clothing-level, not anatomy-level.

**Why Strata needs these:** A character bending forward deforms the spine region differently from the chest. The hips rotate independently when walking. Three torso segments allow more accurate weight painting and deformation.

---

## 4. Boundary Definition Differences

### Neck–Torso Boundary
- **Strata:** Defined by bone hierarchy. The neck region includes vertices weighted to the `Neck` bone. The chest begins at `Spine2`. Clear boundary from bone weights.
- **See-through:** `neck` = visible skin of the neck. `topwear` begins where the collar/neckline of the clothing is. If the character wears a turtleneck, the neck class shrinks; if they wear a low-cut top, the neck class extends further down.
- **Impact:** Low. Both systems agree on where the neck is anatomically. The difference is that See-through's boundary moves with clothing, while Strata's is fixed to skeletal structure.

### Hips–Upper Leg Boundary
- **Strata:** Defined by the `Hips` bone vs `LeftUpLeg`/`RightUpLeg` bone weights. The boundary is at the hip joint.
- **See-through:** `bottomwear` covers the hip area. `legwear` begins where stockings/leggings start. No explicit hip joint boundary.
- **Impact:** Low. Strata's bone-weight-driven boundary is correct for rigging. See-through's clothing-driven boundary is irrelevant to Strata's use case.

### Head Boundary (significant difference)
- **Strata:** Single `head` region (ID 1) includes the entire head: skull, face, eyes, ears, hair, jaw, everything weighted to the `Head` bone.
- **See-through:** Head is decomposed into 9 classes (hair, face, eyes, eyewear, ears, earwear, nose, mouth, headwear). The boundary between `hair` and `face` is particularly important for their layer sandwich problem (hair_front renders in front of face, hair_back renders behind).
- **Impact:** For Strata's rigging use case, the single `head` region is sufficient — everything moves with the head bone. However, the hair_front/hair_back distinction matters for animation quality on long-haired characters. See Section 5 below.

---

## 5. Hair Layering Analysis

### The Problem
See-through explicitly solves the "hair sandwich" problem: hair strands that wrap around the face, with some in front and some behind. They split the `hair` class into front/back strata using K-Means clustering on pseudo-depth values (paper §4.2.2).

For Strata, all hair maps to `head` (ID 1) and deforms with the head bone. This works when:
- Hair is short or tied up (moves as one unit with the head)
- The character is viewed from the front (no depth stratification needed)

This breaks when:
- Long hair flows behind the shoulders — it should follow the spine/shoulders, not the head
- The character turns — hair_front and hair_back need different draw orders

### Does hair_front/hair_back matter for Strata?
**For segmentation training:** No. The segmentation model only needs to identify "this pixel belongs to the head region" for rigging. Hair sub-regions don't change which bone drives the pixel — the head bone drives all hair.

**For animation/Live2D decomposition (future):** Yes. If Strata adds draw-order prediction (Research PRD §3) or layer decomposition (Research PRD §2), hair stratification becomes essential. The See-through team found this was one of their most important innovations.

**For weight painting (current):** Potentially. Long hair behind the shoulders should arguably have some weight on the spine/shoulder bones rather than 100% head bone. But this is a weight-painting refinement, not a segmentation taxonomy change.

### Recommendation
**Option C: Defer as optional detail region.** Don't add `hair_back` to the base 20-region taxonomy. Instead:
1. Keep `head` (ID 1) as the single head region for v1 segmentation
2. When implementing draw-order prediction (Research PRD §3), use the See-through approach of K-Means depth clustering within the head region to split hair front/back
3. If a future "detail regions" system is added, `hair_front` and `hair_back` would be natural sub-regions of `head`

**Justification:**
- Adding a region changes `NUM_REGIONS` in config.py, which cascades to `create_region_materials()`, `convert_rgb_to_grayscale_mask()`, class maps, and all validation checks
- The v1 training data is 3D Mixamo characters where hair is typically short or modeled as a solid mesh — the hair sandwich problem rarely occurs
- The Live2D data source (Research PRD §2) will naturally provide hair layering supervision when that pipeline is built

---

## 6. Recommendation on `hair_back`

**Decision: Defer.** Do not add `hair_back` to the base taxonomy for v1.

**When to revisit:**
- When Live2D model ingestion pipeline is built (Research PRD §2.5)
- When draw-order prediction is added (Research PRD §3)
- If validation on long-haired characters shows systematic segmentation/rigging errors

**If added later, implementation would be:**
- New region ID 20: `hair_back` (hair behind the neck/shoulder line)
- Update `NUM_REGIONS` to 21
- Add bone mapping: vertices weighted to head bone but positioned behind the neck → `hair_back`
- Minimal pipeline disruption if designed as an additive change

---

## 7. Recommendation on `accessory` Label

**Decision: Defer to issue #24.**

The See-through taxonomy handles accessories by absorbing them into specific classes:
- Hat → `headwear`
- Glasses → `eyewear`
- Earrings → `earwear`
- Scarf → `neckwear`
- Gloves → `handwear`
- Held items → `objects`

Strata currently hides accessories for v1 (cleaner training data) and flags them with `has_accessories: true` in metadata.

The Research PRD (§4.3) already notes that the `accessory` label question is tracked under issue #24. The two approaches are:

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| Strata current | Hide accessories, flag in metadata | Clean training data, simpler model | Loses accessory coverage |
| See-through style | Category-specific classes (headwear, etc.) | Rich annotation, useful for decomposition | 6+ new classes, more annotation work |
| Hybrid | Single `accessory` catch-all class | Minimal taxonomy change | Less useful than specific classes |

**Recommendation:** For v1 segmentation, continue hiding accessories. For the future Live2D pipeline, adopt See-through-style specific accessory classes. This aligns with the Research PRD recommendation and keeps the v1 pipeline simple.

---

## Summary of Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Add clothing-aware classes? | No (v1) | Strata's bone-based taxonomy is correct for rigging |
| Add facial detail classes? | No | Single head region sufficient for head bone |
| Add hair_back? | Defer (Option C) | Revisit when Live2D/draw-order pipelines are built |
| Add accessory classes? | Defer to #24 | Continue hiding for v1, adopt specific classes for Live2D |
| Add tail/wings? | No (v1) | Humanoid-only for v1 per project scope |
| Change any existing regions? | No | Current 20-region taxonomy is well-suited for rigging |

**Bottom line:** The Strata and See-through taxonomies serve fundamentally different purposes. See-through decomposes by *visual appearance* (what you see on the surface). Strata decomposes by *skeletal anatomy* (what bone drives each pixel). Neither taxonomy is a superset of the other. No changes to the base 20-region taxonomy are needed for v1. The See-through taxonomy should inform future work on Live2D ingestion and draw-order prediction.

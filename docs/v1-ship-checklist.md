# Strata v1 Ship Checklist

**Goal:** Get Strata v1 (2D rigging + animation) into beta testers' hands within 4-6 weeks. Use feedback + LOIs to support funding conversations.

**Per advisor (Erhvervhus Sjælland):** users + LOIs > model performance numbers. Ship the product, not the research.

---

## v1 Scope

**In scope:**
- Import a 2D character illustration (PNG/PSD with transparency)
- Auto-segment into 22 anatomy regions
- Auto-place 19-bone skeleton
- Auto-generate skinning weights for 2D mesh deformation
- User adjustments: skeleton edit, weight paint, region cleanup
- Pose / animate via timeline keyframes (already built)
- Export GLB / Spine JSON / custom format

**Out of scope (v2):**
- 3D mesh generation (SAM 3D)
- UV texture inpainting
- Multi-view input
- Anything that requires solving the watercolor cat problem

---

## Status — AI Models (training-data repo)

### Segmentation — `0.6485 mIoU` (Run 20)

- [x] Run 20 baseline shipped
- [x] Boundary softening tuned
- [x] Strata post-processing improvements (confidence-gated cleanup, bilinear upscaling, small-component removal)
- [ ] One more run targeting 0.68-0.70 (Run 28/29 in flight as of April 22)
- [ ] **Decision after current run:** accept current best, stop new training, focus on Strata UX

### Joint Refinement — `0.00121 offset`

- [x] Functional, integrated in Strata
- [x] `.pt` archive in bucket recovered (was missing)
- [ ] No new training needed for v1 unless specific failure mode emerges

### Weight Prediction — `0.0215 MAE`

- [x] Functional MLP shipping in Strata
- [x] Laplacian smoothing post-processing in Strata
- [ ] Add: weight painting fallback UI for cases where AI weights are wrong

### 2D Background Inpainting — `0.0028 val/l1`

- [x] Already at ship quality
- [ ] No work needed

---

## Status — Strata Desktop App (../strata/)

This is where most v1 effort should go. **The blocker for funding is product quality, not AI.**

### Import Flow
- [ ] Drag-drop PNG/PSD into the app
- [ ] Detect transparent background, alpha-aware preprocessing (already done)
- [ ] Run rembg if no transparency
- [ ] Show segmentation preview, let user verify before committing
- [ ] Handle non-character images gracefully (error message, suggestion)

### Skeleton Setup
- [x] Auto-place skeleton (already implemented)
- [x] Drag joints, parent/child editing (already implemented)
- [x] IK chains (already implemented)
- [ ] First-time user tutorial overlay on skeleton editor
- [ ] Reset skeleton to AI defaults button
- [ ] Symmetry mirror button (left → right)

### Weight Painting
- [ ] Auto weights from AI model (already in pipeline)
- [ ] Weight paint brush — let user fix AI errors
- [ ] Show weight overlay (heat map per bone)
- [ ] Smooth weights button
- [ ] Per-vertex weight inspector

### Pose / Animate
- [x] Pose mode (drag bones, deform mesh)
- [x] Timeline keyframing
- [ ] Onion skinning (next/prev frame ghost)
- [ ] Animation curve editor
- [ ] Pre-set pose library (T-pose, A-pose, idle, walk cycle, etc.)

### Export
- [x] GLB export
- [x] FBX export
- [x] Spine JSON export
- [ ] Test exports in Unity, Unreal, Spine
- [ ] Add quick "test in browser" preview using three.js

### First-Run Experience
- [ ] Welcome screen with sample character
- [ ] Model download progress bar (~300MB total)
- [ ] Tooltips on first use of each tool
- [ ] Sample project showing complete workflow

### Polish
- [ ] Undo/redo robustness across all tools
- [ ] Keyboard shortcuts documented
- [ ] Settings panel (model selection, quality presets)
- [ ] Error reporting / crash logging
- [ ] Auto-save and recovery

---

## Status — User Research

### Target Group (TBD — pick one)

Candidates ranked by likelihood to pay:

1. **Indie game devs** making 2D game characters → want easy rigging + Spine/Unity export
2. **VTuber model riggers** → currently use Live2D Cubism (paid, complex). Want easier alternative.
3. **2D animators** doing motion graphics or storybook style → want quick character animation
4. **Hobby illustrators** wanting to bring their art "to life" → larger pool, but lower willingness to pay

**Recommendation:** start with indie game devs. Clear use case, technical comfort with the tooling, willing to pay for production-ready output.

### Beta Recruitment

- [ ] Pick target group (decision needed)
- [ ] Identify communities (Reddit r/gamedev, indiedev Discord, Twitter, itch.io devs)
- [ ] Demo video showing complete workflow (5-10 min)
- [ ] Beta signup form / landing page
- [ ] 5-10 beta testers committed
- [ ] Schedule 30-min user testing sessions (record screen + audio)

### Feedback Loop

- [ ] Discord/Slack for beta testers
- [ ] Weekly office hours
- [ ] Bug tracker (linear / github issues)
- [ ] Feature request voting

---

## Status — Fundraising Prep

### Materials

- [ ] Demo video: 60-second hero video showing import → animated character
- [ ] Demo video: 5-min walkthrough for serious viewers
- [ ] Pitch deck (10-15 slides)
- [ ] One-pager
- [ ] Live demo readiness — can run app smoothly in front of investors

### Letters of Intent

- [ ] Identify 2-3 professionals who would say "I'd pay for this"
- [ ] Draft LOI template
- [ ] Get signatures (no payment, just intent)

### Funding Targets

- [x] Erhvervhus Sjælland advisor relationship established
- [ ] InnoFounder application (430K DKK grant, August 2026 round)
- [ ] PreSeed Ventures conversation
- [ ] Accelerace conversation
- [ ] byFounders conversation
- [ ] Soapbox VC follow-up (Jesse Heasman previously connected)

---

## Time Allocation (Per Advisor)

| Bucket | % | Why |
|--------|---|-----|
| AI work | 20% | One last seg run, then stop. Ship current best. |
| Strata UX polish | 40% | The actual bottleneck for funding |
| User research + beta | 30% | "Real users" is the unlock |
| Fundraising prep | 10% | Demo videos, LOIs, pitch material |

**Anti-pattern:** spending 80% of the next 6 weeks chasing 0.65 → 0.75 mIoU. Even if successful, it doesn't move the funding needle. A 20-minute demo of a real beta user creating a working character will move it more.

---

## Decision Log

- **April 22, 2026:** Pivot to v1-only focus. Defer all 3D / texture inpainting work. Per Erhvervhus Sjælland advisor.
- **April 22, 2026:** Seg target relaxed from 0.75 → 0.70. Run 28/29 is the final seg attempt for v1.
- **April 22, 2026:** Stop work on StyleTex test, geometry maps, SAM 3D integration.

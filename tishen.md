# Ti Shen — Side Tear detection & segmentation work

My part of the group assignment: **crumpled glove**, **side tear**, **improper roll**.
This document covers the **side tear** detector and the **segmentation / background
removal** work it required. Crumpled and improper roll are not started yet.

> Note for the team: my three defects do not match `DEFECT_ASSIGNMENT_PLAN.md`.
> That plan gives tears to Member A and puts improper roll on its *excluded* list.
> The code here is scoped so nothing collides (see **Scope split** below), but the
> plan document still needs updating to match reality.

---

## 1. Current pipeline

```
photo
  │
  ├─ preprocessing.py ─────── resize → median blur → CLAHE on Lab L
  │
  ├─ segmentation.py ──────── skin_mask()          remove the bare hand / forearm
  │                           background_key()     remove the backdrop by HUE
  │                           get_background_colors()  multi-mode background model
  │                           _threshold_against() Otsu on distance to nearest mode
  │                           (× BG_REFINE_ITERS)  re-derive the model, segment again
  │                           _repair_holes()      fill holes that are really glove
  │                           → mask_filled, mask_raw
  │
  ├─ defect_detection.py ──── detect_holes         (Member A)
  │                           detect_side_tear     ← MINE
  │                           detect_open_tears    (Member A)
  │                           detect_stains        (Member B)
  │                           deduplicate()        IoU > 0.5, registration order wins
  │
  └─ gui.py / evaluate.py ─── display and batch scoring
```

### Why segmentation got this complicated

Every stage below exists because the photographs were taken with the glove **worn**,
on a **cluttered, unevenly lit** background. None of it would be needed on a plain
backdrop with the glove off the hand.

| Stage | Problem it solves |
|---|---|
| `skin_mask` | The forearm fuses into the glove, so every shape descriptor is measured on a glove+arm blob |
| `background_key` | Shadowed backdrop sits far from every sampled colour and survives as "glove" |
| Multi-mode background | The backdrop is two materials (seat + floor tile); one median describes neither |
| Iterative refinement | The border strip never sees the large shadowed areas around the glove |
| `_repair_holes` | Knit cotton texture punches the mask full of holes |

---

## 2. What changed in the codebase

Nothing is committed yet. `git status` shows 5 modified files and 2 new ones.

### `src/segmentation.py` — heavily extended

**New: `skin_mask(img)`**
Removes the bare hand and forearm. Works in the **Lab chroma plane** (hue angle +
chroma magnitude) rather than raw `a*`/`b*`, because shadow on the backdrop drops
`b*` into the skin band and absolute thresholds misfired on it.

Measured (patch medians):

| class | chroma | angle |
|---|---|---|
| skin | 22.6 | 69.4 |
| white cotton | 8.5 | 69.3 |
| yellow backdrop | 54.7 | 78.9 |
| yellow, shadowed | 36.8 | 82.5 |
| blue nitrile | 28.9 | −106.7 |

Skin shares an *angle* with cotton and a *chroma band* with the backdrop, but nothing
shares both — so both criteria are required together.

A **structural guard** keeps only skin components that touch the image border: a
forearm is attached to a body outside the frame, a glove is not. Without it a tan or
leather glove gets classified as skin and deleted.

**New: `background_key(img)`** — the biggest single win
Keys out the backdrop by **hue**, because hue barely moves under shadow while
lightness moves enormously.

| class | H | S |
|---|---|---|
| yellow backdrop, lit | 21 | 231 |
| yellow backdrop, **shadowed** | **20** | 160 |
| skin | 12 | 104 |
| white cotton | 14 | 27 |
| white floor tile | 15 | 16 |
| blue nitrile | 101 | 160 |

The reference hue is measured from the border strip, not hardcoded, so the same rule
keys the red synthetic background correctly.

Two guards were needed:
* **Fallback** — if the backdrop hue happens to match the *glove* hue the key eats the
  glove. Detected by the mask collapsing below `MIN_AREA_RATIO`, then re-segmented
  without the key.
* **Lightness-variance gate** (`BG_HUE_MIN_LSTD`) — the key only fires when the
  backdrop is genuinely unevenly lit. Measured L std inside the keyed region: uniform
  synthetic backdrop **0.8**, real photographs **32–40**. This is what keeps the
  synthetic regression suite unaffected.

**New: multi-mode background** (`get_background_colors`, `_modes_of`, `background_distance`)
Coarse histogram peak search over the border strip. A single median of "yellow seat +
grey tile" is a colour present nowhere in the image, so *both* real backgrounds then
measured as foreground. Maps onto Ch 7 multimodal thresholding.

**New: iterative refinement** (`BG_REFINE_*`)
Segment once, re-derive the background colours from **everything the mask called
background**, segment again. Same iterate-and-refine idea as Basic Global Thresholding
in Ch 7. Plausible masks 20/25 → 24/25; median raggedness 68 → 60.

**New: `_repair_holes()`**
Knit texture drops scattered glove pixels, leaving holes. Size cannot separate noise
from real punctures — they overlap completely. What is visible *inside* does:

| | Lab distance from glove colour, inside the hole |
|---|---|
| noise holes | p10 18.4 · p50 29.3 · p90 85.1 |
| real holes | p10 40.5 · p50 79.9 · p90 124.5 |

A cut at 30 fills 127 of 238 noise holes and destroys **none** of the 10 real ones.
Holes per image 9.5 → 5.0.

**Changed: `OPEN_KSIZE` documented after a sweep** — 3 is optimal; anything larger
erodes thin tears away (k=5/7/9 all broke the tear tests and lost defects).

**Kept but unused: `texture_map()`, `textured_mask()`** — see rejected list below.

### `src/defect_detection.py`

**New: `detect_side_tear()`** plus helpers `_basic_rectangle_axis()` and
`_fingers_at_low_end()`. Two branches, unioned:

* **Branch A — notch.** `close(mask, disc) − mask` isolates concavities narrower than
  the disc. A scale-bounded convex deficiency **D = H − S** (Ch 10/11) without the
  convex hull's global reach. On a real spread-finger glove the hull runs straight
  from fingertip to cuff, so every local notch merges into one huge deficiency —
  measured, a tear was swallowed inside a 30,414 px component.
* **Branch B — skin through the glove.** These gloves are worn, so a breach exposes
  the hand. Finds tears that never open wide enough to change the silhouette.

Filters: lateral band, minimum area, minimum depth, colour distance from the glove,
elongation (branch B), plus a relaxed **cuff rule** for small-but-emphatic notches.

**Orientation** (`_fingers_at_low_end`) uses the **skin position** as its primary cue —
the arm attaches at the cuff, never at the fingertips. This scored **8/8** on test
images where shape-based cues managed 5/8. Fill-ratio along the axis is the fallback
for unworn gloves and the synthetic images.

**Changed: `detect_holes`** now compares against *every* background mode, not just the
dominant one, and down-weights lightness (`BG_MATCH_L_WEIGHT = 0.25`). A lighting
gradient is a continuum, so a hole's lightness can land between two sampled modes
while its chroma matches one exactly — measured 49.2 in full Lab vs 12.0 in chroma.

### `src/selftest.py`
`analyse()` counts Open Tear + Side Tear together so all 35 pre-existing expectations
stay valid. Added `tear_labels()` and `build_side_tear_cases()` — 10 scope tests that
assert *which* detector claimed each tear, printed as their own block.

### `src/gui.py`, `src/evaluate.py`
Registered "Side Tear" in the dropdown, the label map, and `LABEL_MAP`
(`side_tear` / `edge_tear`).

### `src/isolate.py` — new
Background-removal viewer. Writes a 3-panel strip per image: photo | glove isolated |
silhouette with the outline traced.

```powershell
python src\isolate.py "path\to\images" output_folder
```

Useful for labelling ground truth and as a report figure for the segmentation stage.

---

## 3. Scope split with Member A

`detect_side_tear` is registered **before** `detect_open_tears` so the more specific
detector wins de-duplication inside its band. The two are disjoint by construction:

* lateral / cuff tear → **Side Tear** (mine)
* fingertip tear → **Open Tear** (Member A's)

Verified by the 10 scope tests.

---

## 4. Results

Scored against 43 hand-labelled defects (37 side tears + 6 holes) across 25
photographs. A detection counts only if the labelled defect centre falls inside the
box — **localisation, not "did the image produce a box"**.

| stage | F1 | precision | recall |
|---|---|---|---|
| first honest measurement | 0.270 | 22% | 33% |
| + skin-through branch | 0.389 | 28% | 62% |
| + tightened notch branch | 0.444 | 35% | 62% |
| + elongation filter | 0.505 | 42% | 62% |
| + segmentation work *(detector unchanged)* | 0.568 | 53% | 61% |
| + cuff acceptance rule | **0.586** | **55%** | **63%** |

Synthetic regression held at **35/35** and **10/10** throughout.

Segmentation: plausible mask area **25/25**, retained backdrop **8.2% → 0.5%**,
defects still inside the mask **38/43**.

**Earlier numbers of 100% precision / 100% recall were image-level** — they only asked
"did this image produce any box". On a set where every tear image yields 2–4 boxes
that metric is close to meaningless. The table above is the honest one.

---

## 5. Techniques evaluated and rejected

All measured, all documented in the source so they are not retried. Useful material
for the report's "rational critical evaluation of candidate techniques".

| technique | result |
|---|---|
| Homomorphic illumination flattening | plausible masks 20/25 → **15/25**, raggedness 68 → **246** |
| Texture as an additive segmentation score | plausible masks 23 → **17** |
| Texture **veto** on the hue key | F1 0.57 → **0.52** |
| Texture **rescue** growing the mask | F1 0.57 → **0.29** — see below |
| Contour roughness as a tear feature | only **1.24×** separation, ranges overlap |
| Enclosure ratio for skin patches | 0.49 vs 0.45 — no separation |
| Relaxing area/depth globally for emphatic colour | F1 0.584 → **0.562** |
| Larger morphological opening (k=5/7/9) | broke tear tests, lost defects |

**The texture rescue is the instructive one.** It produced the cleanest masks of the
whole exercise — `224955` went from 80.5% to **100%** glove coverage — and **halved**
detection F1. A tear's frayed edge is textured too, so repairing "missing glove" seals
the very openings the detector reads.

> Cosmetic mask quality and useful mask quality are not the same thing. A hole in the
> mask where the glove is intact is damage; a hole where the glove is torn is the
> signal. Any repair that cannot tell them apart trades signal for tidiness.

---

## 6. Known limitations

* **White cotton on white floor tile** (`224955`) — unsolvable by colour. No hue to
  key, no colour distance. The mask fragments.
* **Recall stuck around 62%** — 17 tears missed. Attributed: 6 have a notch the
  filters reject, 3 have no signal at all (the hand held the cut closed), 7 sit away
  from the mask boundary (either the mask is wrong there or my ground-truth circle is
  a few pixels out).
* **False positives**: 12 mid-glove (mask artefacts), 12 at the cuff (glove/arm
  junction), 0 in the finger region — the band filter works.
* **Residual contamination in 7 of 25 images** — forearm slivers along the bottom
  edge of the `2241xx` set, yellow wedges near the cuff on `224321`/`224404`.
* **No clean gloves in the dataset**, so the false-positive rate required by the
  assignment cannot currently be measured at all.
* `SIDE_TEAR_MIN_COLOR_DIST = 45` is the least secure threshold — the gap between the
  weakest real tear and the strongest false one is only about 3 Lab units.

---

## 7. Next steps

1. **Reshoot** — decided. Glove **off the hand**, tears nudged **open**, plain
   saturated backdrop in a contrasting hue, nothing white or grey in frame, even
   light, ~10% margin, check focus. Include **4–5 clean gloves per material** for a
   `good/` folder.
2. **Redo segmentation first** and confirm it is right before touching the detector.
   Most of the segmentation complexity above exists to survive conditions the reshoot
   removes.
3. **Re-label ground truth** using the same process — it caught real errors twice:
   I mark expected results → the team verifies → only then measure.
4. **Then** rebuild the detector, scoring localisation from the first iteration.
5. Still to start: **crumpled glove** and **improper roll**.

### Reusable tooling

| file | purpose |
|---|---|
| `src/isolate.py` | background-removed view for labelling and report figures |
| `ground_truth.json` | 43 defects as coordinates + type (`side_tear` / `hole`) |
| `iterate.py` *(scratch)* | runs the detector over a labelled set, scores it, dumps annotated output |

Iteration outputs are on the Desktop in `iteration1out` … `iteration5out`, each with a
`_summary.txt`; the annotated ground truth is in `expected result`.

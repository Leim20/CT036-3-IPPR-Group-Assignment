# Ti Shen — defect detectors & segmentation

My three defects, all implemented and merged into the team's system:

| detector | defect | recall on its own images |
|---|---|---|
| `detect_incomplete_beading` | a stretch of the cuff hem is missing | 11/11 |
| `detect_damage_by_fold` | a crease left where the glove was folded | 7/7 |
| `detect_improper_roll` | the cuff is rolled or bunched, not lying flat | 8/8 |

`detect_side_tear` -- written when tearing was still mine -- was removed from the
runtime, GUI and evaluator because it caused too many false positives; lateral tears
now belong to the existing **Open Tear** class. Sections 2, 4 and 5 keep its method and
measurements as a record, clearly marked. Its two orientation helpers were kept, because
all three detectors above depend on them. All three rest on the **segmentation /
background removal** work below.

> Note for the team: `DEFECT_ASSIGNMENT_PLAN.md` still puts improper roll on its
> *excluded* list ("needs fine-grained edge geometry modelling, false-positive rate
> hard to control"). It is implemented and works -- 8/8 with one false positive on a
> 7-image control -- so that plan entry is out of date and should be corrected.

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
  ├─ defect_detection.py ──── detect_holes               (Member A)
  │                           detect_incomplete_beading  ← MINE
  │                           detect_damage_by_fold      ← MINE
  │                           detect_open_tears          (Member A)
  │                           detect_stains              (Member B)
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

The side tear and segmentation work is committed as `d5761ea` on branch
`tishen/side-tear-and-segmentation`; the beading and fold detectors sit on top of it.

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

**`detect_side_tear()` -- written, then REMOVED.** It was mine while tearing was
still my defect. Member A deleted it in `0f60112` because its false-positive rate was
too high (on one cuff photograph it claimed four separate tears), and lateral tears
now fall to the existing **Open Tear** class. The write-up below is kept because the
method and its measurements are worth reporting; the code is no longer in the system.

Its two helpers, `_basic_rectangle_axis()` and `_fingers_at_low_end()`, were kept --
all three of my current detectors depend on them for glove orientation.

It had two branches, unioned:

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
`analyse()` counted Open Tear + Side Tear together so all 35 pre-existing expectations
stayed valid, and `tear_labels()` / `build_side_tear_cases()` added 10 scope tests
asserting *which* detector claimed each tear. All of that was removed with the detector;
the suite now runs 41/41 without it.

### `src/gui.py`, `src/evaluate.py`
Registered "Incomplete Beading", "Damage By Fold" and "Improper Roll" in the dropdown,
the detector label map, and `LABEL_MAP`. The "Side Tear" entries were removed with the
detector; `LABEL_MAP` still accepts the legacy `side_tear` / `edge_tear` folder names
but now maps them to **Open Tear**, so existing labelled data stays usable.

### `src/isolate.py` — new
Background-removal viewer. Writes a 3-panel strip per image: photo | glove isolated |
silhouette with the outline traced.

```powershell
python src\isolate.py "path\to\images" output_folder
```

Useful for labelling ground truth and as a report figure for the segmentation stage.

---

## 3. Scope split with Member A

This section used to describe how `detect_side_tear` (mine) and `detect_open_tears`
(Member A's) were kept disjoint -- lateral tears to one, fingertip tears to the other,
with mine registered first so it won de-duplication inside its band.

That split no longer exists: side tear was removed and **all tearing is Member A's**.
My three detectors claim regions no tearing detector looks at -- the cuff hem
(beading), the glove surface (fold) and the cuff band (roll) -- so there is no
registration-order dependency between my work and theirs any more. What overlap
remains is between my own three, and is measured in section 10.

---

## 4. Results

> These are the **side tear** figures. The detector has since been removed (see
> section 2), so this is a record of method rather than a current result. It is kept
> because it is the only part of my work scored for LOCALISATION, and because the
> jump from the image-level numbers to the honest ones is the single most useful
> lesson in this document.

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
| Enclosure ratio for skin-through patches | 0.49 vs 0.45 -- no separation at all |
| Directional closing to rejoin a dashed fold ridge | bridged the fragments, but also merged the ridge into neighbouring creases: two images produced one box swallowing most of the glove, two lost detection entirely, and the two it was aimed at still missed |

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

## 7. Incomplete beading

The bead is the finished hem at the cuff: a maroon knitted band on the cotton gloves, a
rolled edge on latex and nitrile. The defect is a stretch of that hem missing, leaving a
ragged opening that looks like a tear at the wrist.

**The idea that made it tractable:** a bead is a CONTINUOUS structure, so the defect is
a DISCONTINUITY in it, and the rest of the same bead is the reference. Every threshold
is self-referential (median +/- k*sd along this glove own cuff), so nothing needs
retuning per material or per lighting.

1. Segment; find the major axis; the cuff end is whichever end the skin sits nearer
2. Extract the **cuff opening edge** -- boundary in the cuff band running alongside skin
3. Reduce it to a **1-D signature** (Ch 10/11): at each station record the colour just
   inside the edge (the bead) and the local roughness of the edge
4. Standardise both against the cuff own median and spread, then sum
5. Flag contiguous runs above threshold, merge runs that nearly touch, box each

Colour and roughness fail in different places, which is why both are used. Colour is
strong on cotton where the maroon band is unmistakable and weak on nitrile where the
roll is nearly the same blue as the body; roughness ignores colour entirely but needs
sharp focus.

Three fixes came out of testing:

| fix | why |
|---|---|
| near-skin radius 26 -> 70 px | when the forearm is only partly in frame the skin mask is a thin strip, and a tight radius clipped the cuff edge before it reached the defect |
| run merging (`BEAD_RUN_MERGE_FRAC`) | a single gap dips back under threshold mid-way, so one defect was reported as 3-4 boxes. Merging took 225539 from 4 boxes to 1 |
| minimum run 0.05 -> 0.08 | removed a spurious box. Swept: at 0.11 real detections start disappearing, so 0.08 is the edge of the safe range |

The red-backdrop photographs also exposed a real bug: **skin detection returned 0%**, so
the arm fused into the glove. Skin under a red cast measures chroma 48-49, above the old
bound of 34, but its hue angle (52-58 deg) is still in range while the red backdrop sits
at 33 deg. Widening chroma to 55 and tightening the angle bound to 74 fixed it without
disturbing the yellow-backdrop set or the synthetic suite.

**Result:** fires on all 11 images, 100% recall across cotton, latex and nitrile. Boxes
land cleanly on 6 of 11, 3 more are close, 2 are weak (one rotated shot, one soft-focus
with a heavy colour cast).

---

## 8. Damage by fold

A fold leaves a long dark crease across the glove SURFACE. Unlike everything else here
it is not a boundary feature, so none of the contour machinery applies.

**Morphological blackhat with a LINE structuring element** (Ch 8) responds to dark
structures narrower than the element, so a line about a tenth of the glove length picks
up a crease while ignoring the woven texture of a fabric glove -- that texture is
fine-scale in EVERY direction and never fills a long line. Sweeping 12 orientations and
keeping the maximum makes the response independent of which way the fold runs.

Two regions must be excluded, both found by looking at the response map: the glove
**boundary** (finger gaps are dark valleys and light up hard) and the **cuff** (knitted
ribbing is a regular line pattern that swamps a real crease).

Candidates are scored on **length x elongation**: a fold is long and straight, a shadow
blob is neither, a strip of grip pattern is straight but short.

Two threshold decisions worth recording:

* **Rank threshold, not median + k*sigma.** On a low-contrast crease the sigma of the
  whole glove buries the defect; two images were missed for exactly that reason even
  though the response traced their folds perfectly. Switched to keeping the strongest 6%.
* **Length floor 0.18 -> 0.10.** Measured, real creases run 119-144 px on gloves whose
  axis put that floor at 154-188 px, so genuine folds were rejected on length alone.

**Result:** 7 of 7 images detect a fold, 100% recall.

**Known limitation:** on two images the box lands on a finger crease rather than the
large fold across the palm. The mechanism is understood -- a broad ridge is dark
unevenly along its length, so it thresholds into a DASHED chain of blobs, each too short
to survive the length filter, while a thinner but unbroken finger crease survives and
wins. Two fixes were tried and both failed (see the rejected list).

Two images were dropped from this set by agreement: they showed general crumpling rather
than a distinct fold, and that ambiguity was pulling the scoring toward the wrong model.

---

## 9. Improper roll

The cuff is rolled or twisted into a thick uneven band at the wrist instead of lying
flat. Exactly the opposite of incomplete beading: beading is a GAP in the hem, improper
roll is EXCESS material bunched up.

Two independent signatures, both physical rather than curve-fitted:

**The cuff stops being darker than the palm.** A flat cuff sits in the shadow of the
wrist. A rolled one bulges towards the camera and catches the light along its ridge.
Measured as palm lightness minus cuff lightness:

| | palm L − cuff L |
|---|---|
| improper roll | −27 .. 11 |
| normal cuff | 10 .. 34 |

**The terminal edge is tilted.** A properly worn cuff ends roughly PERPENDICULAR to the
glove's major axis; a rolled one sits at an angle:

| | degrees off perpendicular |
|---|---|
| improper roll | 0.4 .. 86.2 |
| normal cuff | 0.3 .. 4.5 |

The two are OR'd, because either alone is sufficient: a roll that happens to sit square
to the axis is still caught by its brightness, and a roll on a glove whose cuff is
naturally pale is still caught by its angle.

The control set is the damage-by-fold images, which have undamaged cuffs.

**Result:** 8/8 on the roll images, 1 false positive across the 7 controls.

**Caveat worth stating in the report:** both thresholds were fitted on 15 images
(8 defective, 7 control) from a small number of physical gloves. The separation is
clean on that data, but it is a small sample and the numbers should be re-checked
against a wider set before they are quoted as general.

---

## 10. Cross-talk between detectors

Running everything over the whole dataset:

| detector | recall on its own defect | precision |
|---|---|---|
| Incomplete Beading | 11/11 = 100% | 47.8% |
| Damage By Fold | 7/7 = 100% | 30.4% |
| Improper Roll | 8/8 = 100% | not measured in this run |

(The original run of this table also carried a Side Tear row at 85.7% / 28.6%. That
precision was the lowest of the four and is part of why the detector was dropped.)

**Recall is perfect; precision is not, and the reason is cross-talk rather than bad
detection.** The fold detector fires on beading images because a torn cuff also creases
the material; the beading detector fires on fold images because a fold near the cuff
disturbs the hem profile. Each is finding something real, on an image labelled for a
different defect.

This is exactly the multi-defect recognition problem the bonus marks ask about. It is a
per-region arbitration problem, not a per-detector tuning problem, and cannot be fixed
by adjusting any single detector in isolation.

Two rows in the evaluation are not mine: **Stain (24 false positives)** is Member B
detector firing on cotton knit texture, and **Tear / Hole (5)** is Member A.

---

## 11. Region highlighting

Every one of my detectors used to report only a rectangle, and a rectangle is a poor
description of these defects: a fold is a long thin diagonal crease, a beading gap
follows the curve of the cuff, and an improper roll wraps around the wrist. The box
covers a lot of glove that is not damaged, which overstates the affected area and
makes the annotated figures harder to read.

The team's shared code already solved the rendering half of this. `Detection` carries
a `mask` field -- "a uint8 binary image the same size as the preprocessed picture,
used for pixel-level shading and for affected-area" -- and `draw_results` already
tints those pixels, outlines them, and falls back to filling the box only when a
detector supplies no mask. So my detectors were the ones filling whole rectangles
with colour, because `detection_mask()` had nothing better to fall back on.

Each of mine already computed the right pixels internally; they were simply being
thrown away at the return statement.

| detector | region now shaded |
|---|---|
| `detect_damage_by_fold` | the connected components of the thresholded crease response that survived the length/elongation test, carried through the box-merge so a crease made of several fragments shades as one region |
| `detect_incomplete_beading` | the traced cuff run, stroked to `BEAD_MASK_WIDTH` (the depth the bead occupies) and clipped back to the glove mask |
| `detect_improper_roll` | the cuff-band pixels themselves, which follow the glove outline |

**How much the box was overstating things.** Shaded area as a fraction of the box it
replaced, measured over the 26 images of my three sets:

| detector | mask / box area |
|---|---|
| damage by fold | 0.09 -- 0.82 (typically ~0.2) |
| incomplete beading | 0.15 -- 0.35 |
| improper roll | 0.06 -- 0.94 |

So for a fold the rectangle was claiming roughly five times the damaged area it should
have. This feeds straight into `affected_area_percentage()`, which is reported per
image, so the numbers in the results section were inflated before this change and are
honest after it.

Two details that needed care:

* The beading run lies **on** the boundary, so half the stroked band fell on the
  background. It is intersected with the glove mask (dilated by a few pixels so the
  band still reads against the backdrop) -- otherwise the shaded area counted
  backdrop as damaged glove.
* The box is now derived from the region rather than from the raw contour run, so the
  shading can never spill outside its own outline. Before that fix a few beading
  regions measured 1.04-1.20x their box.

**Evidence scores.** The same return statements had been leaving `evidence` at its
default of 0.0, so every one of my detections was labelled "... 0" in the annotated
image and contributed nothing to `overall_evidence_score()`. Each now reports how far
past its own threshold the measurement sat, on the team's existing 50-100 convention
(50 = exactly at the threshold, 100 = comfortably past it):

| detector | measured against |
|---|---|
| incomplete beading | mean standardised departure over the run, vs `BEAD_K_SIGMA` |
| damage by fold | length x elongation, vs the product of the two floors |
| improper roll | whichever of the two signatures fired harder |

**Colours.** All of them were falling through to the default green, so they could not be
told apart when several fired on one glove. They now have entries in `DEFECT_COLORS`:
beading blue, fold green, roll cyan. Roll was pink at first and
had to be changed -- against a magenta `Thin / Overstretched` region on the same glove
the two were indistinguishable.

**Verified unchanged:** selftest 41/41, pytest 40 passed / 13 subtests, and recall on my
own sets still 11/11 beading, 7/7 fold, 8/8 roll. This changes what is drawn and what
`affected_area_percentage` measures, not what is detected.

---

## 12. Next steps

1. **Ground truth for all three detectors.** None has been scored for LOCALISATION --
   "100% recall" here means the right label on the right image, not that the box sits on
   the defect. The side tear work showed how misleading that can be: image-level scoring
   read 100%/100% while box-level was 22%/33%. The process that worked was: I mark the
   expected results, the team verifies them, then we measure.
2. **Clean gloves.** There are still none in the dataset, so the false-positive rate the
   assignment asks for cannot be measured at all. 4-5 undamaged gloves per material.
3. **Cross-talk arbitration** -- see section 10. Needed for the bonus marks, and it is a
   group-level design decision rather than something any one detector can fix.
4. **Reshoot** if time allows: glove off the hand, tears nudged open, plain saturated
   backdrop in a contrasting hue, nothing white or grey in frame, even light, ~10%
   margin, check focus. Most of the segmentation complexity above exists purely to
   survive conditions a clean shot removes.


### Reusable tooling

| file | purpose |
|---|---|
| `src/isolate.py` | background-removed view for labelling and report figures |
| `ground_truth.json` | 43 defects as coordinates + type (`side_tear` / `hole`) |
| `iterate.py` *(scratch)* | runs the detector over a labelled set, scores it, dumps annotated output |

Iteration outputs are on the Desktop in `iteration1out` … `iteration5out`, each with a
`_summary.txt`; the annotated ground truth is in `expected result`.

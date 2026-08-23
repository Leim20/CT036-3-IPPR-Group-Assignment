# How my three detectors work, front to back

Reference document for two purposes: rebuilding this work inside the team's system later,
and writing the report. Every number quoted here is measured, not estimated, and the
source it came from is named.

My defects: **incomplete beading**, **damage by fold**, **improper roll**.
Constraint: **classical image processing only, no machine learning.**

---

## Part 1 -- The path a photograph takes

```
  USER
   |
   |  gui.py -- GloveDefectApp
   |    pick image  ->  pick detector (dropdown, or "all")  ->  Run Detection
   v
  process_image_array(image, material, detectors)          pipeline.py
   |
   +-- 1. preprocess(image)                                preprocessing.py
   |      resize to width 800 (cap height 2400)
   |      medianBlur k=5            -- keeps tear edges sharper than Gaussian
   |      CLAHE on Lab L channel    -- cancels side lighting, leaves colour alone
   |      returns (img_norm, img_plain)
   |
   +-- 2. segment_glove(img_norm)                          segmentation.py
   |      -> mask_filled  glove outline, holes filled
   |      -> mask_raw     glove pixels only, holes black
   |
   +-- 3. glove_found(mask_filled)      -- area sanity check, else stop here
   |
   +-- 4. get_background_color(img_norm)  -- plain border median
   |
   +-- 5. run_all_detectors(...)                           defect_detection.py
   |      each detector in DETECTORS order, each in its own try/except
   |      deduplicate(): IoU > 0.5, FIRST REGISTERED WINS
   |      -> list[Detection(name, box, mask, evidence)]
   |
   +-- 6. build_defect_masks(...)  -- rebuilds a pixel mask per detection
   |      defect_mask = OR of all of them
   |
   +-- 7. draw_results(img_plain, defects)
   |      per defect: translucent tint (alpha 0.38) + traced outline
   |                  + thin bounding box + "<name> <evidence>" label
   v
  result dict -> GUI
     side-by-side view (original | annotated) or overlay view
     affected_area_percentage(), overall_evidence_score(), processing time
     Save button writes the annotated image
```

**Result dict keys** the GUI and evaluator rely on: `original_image`, `normalized_image`,
`glove_mask`, `raw_glove_mask`, `defect_mask`, `defects`, `defect_labels`,
`defect_locations`, `features`, `debug_images`, `result_image`, `glove_found`, `errors`,
`status_message`.

### The contract every detector must satisfy

```python
def detect_x(img, mask_filled, mask_raw, bg_color, img_plain=None, material=None):
    return [Detection(name, (x, y, w, h), mask, evidence), ...]
```

* `img` is `img_norm` (CLAHE applied), `img_plain` is without it
* `mask` -- uint8 binary, same size as `img`; drives the shading and the affected-area figure
* `evidence` -- rule strength 0-100 on the team's convention: **50 = exactly at the
  detector's own acceptance threshold, 100 = comfortably past it.** It is not a probability
  and must not be described as one in the report.
* **The `img_plain` and `material` keyword arguments are not optional.** A detector missing
  them raises, and `run_all_detectors` swallows the exception per-detector -- so it is
  silently skipped and simply never fires. This cost me a full debugging session.

---

## Part 2 -- Segmentation, which everything else rests on

Not one of my three defects, but all three depend on it, and most of my effort went here.
The photographs are of **worn** gloves on coloured backdrops, so two things must be removed
before any detector runs: the backdrop, and the bare arm.

**Skin removal** works in the **Lab chroma plane** (hue angle + magnitude), because chroma
survives shadow where RGB does not. Measured:

| surface | chroma | hue angle |
|---|---|---|
| skin | 22.6 | 69.4 |
| skin under red cast | 48-49 | 52-58 |
| white cotton glove | 8.5 | 69.3 |
| yellow backdrop, lit | 54.7 | 78.9 |
| yellow backdrop, shadowed | 36.8 | 82.5 |
| blue nitrile | 28.9 | -106.7 |
| red backdrop | -- | 33 |

`SKIN_CHROMA = (13.0, 55.0)`, `SKIN_ANGLE = (42.0, 74.0)`. The upper chroma bound had to
reach 55 for skin under a red cast; the angle bound had to tighten to 74 so the red
backdrop at 33 deg stayed out.

**Structural guard:** only skin components **touching the image border** are kept. A
forearm is attached to a body outside the frame so it always enters from an edge; a tan or
leather glove does not. Without this the leather sample in the regression suite was deleted
as skin.

**Background removal** is a hue key plus multi-mode Lab distance. Measured H/S: yellow lit
21/231, yellow shadowed 20/160, skin 12/104, white cotton 14/27, tile 15/16, nitrile
101/160. Hence `BG_HUE_TOLERANCE=6`, `BG_HUE_MIN_SAT=110`. `BG_HUE_MIN_LSTD=10.0`
separates a real photograph (L std 32-40) from a synthetic flat one (0.8).

Two guards that exist because of specific failures:

* `BG_KEY_MIN_KEEP = 0.55` -- when the backdrop hue is near the glove's, the key deletes
  the glove. An absolute floor is not enough: on white cotton over a coloured mat the key
  left 5.3% of the frame, which clears a 5% floor while the correct mask is 53%. So segment
  both ways and keep the key only if it does not collapse the glove *relative to not using
  it*.
* `BG_MODE_MIN_CHROMA = 10.0` -- distance is taken to chromatic modes only. A near-neutral
  mode (black shadow, grey tile) matched every neutral pixel including a white glove, and
  the mask collapsed 49.7% -> 11.2%.

`_repair_holes` uses `HOLE_REPAIR_MAX_DIST = 30.0`, measured: noise holes p10 18.4 / p50
29.3 / p90 85.1, real holes p10 40.5 / p50 79.9 / p90 124.5. Fills 127 of 238 noise holes,
destroys 0 of 10 real ones.

**Result:** plausible masks went from 24/51 to 25/25, retained backdrop 8.2% -> 0.5%.

---

## Part 3 -- The three detectors

All three start the same way, because all three are defined relative to the glove's
geometry rather than to the image frame:

```python
cnt  = largest external contour of mask_filled
axis, perp, lo, hi = _basic_rectangle_axis(cnt)     # cv2.minAreaRect -> major axis
low_is_distal      = _fingers_at_low_end(...)       # which end holds the fingers
frac               = (pt . axis - lo) / (hi - lo)   # 0.0 = fingertips, 1.0 = cuff
```

`_fingers_at_low_end` decides orientation from **where the bare arm is** -- an arm attaches
at the cuff, never at the fingertips. Scored **8/8** against 5/8 for shape-based cues. Fill
ratio along the axis is the fallback for unworn gloves and the synthetic images (fingertip
end 0.58-0.82, cuff end 0.88-1.00).

---

### 3.1 Incomplete beading

**Defect:** a stretch of the finished hem at the cuff is missing, leaving a ragged opening.

**This was built three times. The two failures are the most useful part of the write-up.**

**v1 -- boundary signature (Ch 10/11).** Walk the cuff outline, record at each station the
colour just inside it and the local roughness, standardise against the cuff's own median,
flag contiguous runs. Labelled 11/11. But on the clearest image -- a hem torn open across a
third of the cuff -- it marked a 700 px sliver at one tip.
*Why:* **a hem gap does not change the silhouette.** The material either side still bounds
the outline, so the glove mask is a smooth solid hand shape with no notch. A boundary
signature has nothing to read.

**v2 -- threshold the cuff band against its own median colour.** Every number improved:
flagged area separated cleanly from intact-cuff controls (0.00068-0.00497 vs
0.00000-0.00050), recall stayed 11/11, false positives on the fold set fell 4/7 -> 0/7.
**It was still completely wrong.** Thresholding a band against its median finds whatever
colour is in the *minority*. On the cotton gloves that is the maroon bead -- so it outlined
**the part of the hem that is still there**, the exact inverse of the defect.
*The lesson: a metric that separates is not the same as a metric measuring the right thing.
Only zooming into four images at high magnification revealed it.*

**v3 -- find the missing material.** A gap is an *absence*, not a colour anomaly. The
gloves are worn, so what shows through is the hand.

```
1. arm = skin_mask(img)                        (guarded: the forearm)
   stand down if it is smaller than 500 px -- no reference to check against
   arm_hue = median hue over the arm
2. hull = convexHull(cnt), filled
3. breach = the unguarded skin chroma test AND hull AND NOT mask_raw
             ^ UNGUARDED skin test: skin through a hole is a mid-frame island,
               and skin_mask's border-touching guard would discard it
4. morphological OPEN 5x5 then CLOSE 15x15       (Ch 8)
5. connected components; reject each unless
      a. it does not touch the image border      -> that is the arm
      b. area >= BEAD_MIN_AREA_FRAC * axis_len^2
      c. centroid frac >= BEAD_CUFF_BAND         -> it is at the cuff
      d. |median hue - arm_hue| <= BEAD_MAX_HUE_SHIFT
6. keep the largest BEAD_MAX_REGIONS; mask = the component itself
```

Steps 5c and 5d are each there for a specific measured failure:

**5c, position.** The skin test also fires on the shadow the glove casts on a warm
backdrop, and those shadows are *larger* than the real tears, so size alone picks the wrong
one:

| | axis fraction |
|---|---|
| real breach at the cuff | 0.81 .. 1.00 |
| backdrop shadow elsewhere | 0.05 .. 0.66 |

**5d, hue.** One shadow survives 5c -- on the yellow-backdrop nitrile photograph it sits at
0.81 and is 5,159 px against the real tear's 41 px. Skin seen through a hole is the *same
skin under the same light* as the visible forearm, so its hue must match; and hue is what
survives shadow, which is the whole basis of the background key:

| | hue distance from the forearm |
|---|---|
| real breach (all 11) | 0 .. 2 |
| shadow on the backdrop | 4 |

Rejected alternatives, both measured: **enclosure** (fraction of a component's surrounding
ring that is glove) gave 0.46-0.73 for real breaches vs 0.10-0.53 for shadows -- overlaps.
**Lightness vs the forearm** gave -17 for a real breach vs -9 for the shadow -- inverted.

**Constants:** `BEAD_CUFF_BAND=0.75`, `BEAD_MIN_AREA_FRAC=0.0001`, `BEAD_OPEN=5`,
`BEAD_CLOSE=15`, `BEAD_MAX_HUE_SHIFT=3.0`, `BEAD_MAX_REGIONS=2`, `BEAD_BOX_PAD=8`.

**Result:** 11/11, and the region is on the actual defect in 11 of 11 (v1: 6 of 11,
v2: 0 of 11).

---

### 3.2 Damage by fold

**Defect:** a crease left across the glove surface where it was folded. Unlike the others
this is not a boundary feature at all, so none of the contour machinery applies.

```
1. inner = erode(mask_filled, FOLD_ERODE_FRAC * axis_len)   -- stay off the boundary
2. drop everything with frac > FOLD_CUFF_EXCLUDE            -- the cuff
3. gray = medianBlur(grayscale, 5)
4. crease_response: for 12 orientations, MORPH_BLACKHAT with a 1-px-wide LINE
   structuring element of length FOLD_LINE_FRAC * axis_len; keep the pixelwise max
5. threshold at the top FOLD_TOP_PERCENT of the response inside `inner`
6. CLOSE 7x7 ellipse, OPEN 3x3
7. connected components, keep those with
      length >= FOLD_MIN_LEN_FRAC * axis_len  AND  elongation >= FOLD_MIN_ELONG
   scored by length x elongation
8. merge boxes closer than FOLD_MERGE_GAP_FRAC * axis_len, keep top FOLD_MAX_BOXES
   mask = union of the components that went into each merged box
```

**Why blackhat with a line element (Ch 8 morphology, Ch 7 line detection):** blackhat
responds to dark structures *narrower than the structuring element*. A line about a tenth
of the glove's length picks up a crease while ignoring woven fabric texture, which is
fine-scale in *every* direction and so never fills a long line. Sweeping 12 orientations
and keeping the max makes it independent of which way the fold runs.

**Two decisions worth defending in the report:**

* **Rank threshold, not median + k*sigma.** On a low-contrast crease the sigma of the whole
  glove buries the defect. Two images were missed for exactly that reason even though the
  response traced their folds perfectly.
* **The length floor came down from 0.18 to 0.10.** Measured: real creases run 119-144 px
  on gloves whose axis put the 0.18 floor at 154-188 px, so genuine folds were being
  rejected on length alone.

**Rejected, measured:** directional closing to rejoin a fold that thresholds into a dashed
chain. It did bridge the fragments, but it merged the ridge into neighbouring creases --
two images produced a single box swallowing most of the glove, two lost their detection
entirely, and the two it was aimed at still missed. Worse on every count.

**Constants:** `FOLD_N_ORIENT=12`, `FOLD_LINE_FRAC=0.10`, `FOLD_ERODE_FRAC=0.045`,
`FOLD_CUFF_EXCLUDE=0.72`, `FOLD_TOP_PERCENT=6.0`, `FOLD_MIN_LEN_FRAC=0.10`,
`FOLD_MIN_ELONG=2.5`, `FOLD_MERGE_GAP_FRAC=0.06`, `FOLD_MAX_BOXES=2`.

**Result:** 7/7.

---

### 3.3 Improper roll

**Defect:** the cuff is rolled or bunched into a thick uneven band instead of lying flat.
The exact opposite of incomplete beading -- beading is a *gap* in the hem, improper roll is
*excess* material.

Two independent physical signatures, **OR'd** because either alone is sufficient:

**A. The cuff stops being darker than the palm.** A flat cuff sits in the shadow of the
wrist; a rolled one bulges towards the camera and catches the light along its ridge.

| | palm L - cuff L |
|---|---|
| improper roll | -27 .. 11 |
| normal cuff | 10 .. 34 |

**B. The terminal edge is tilted.** A properly worn cuff ends roughly *perpendicular* to
the major axis. Fitted by SVD on the pixels beyond `ROLL_EDGE_BAND`.

| | degrees off perpendicular |
|---|---|
| improper roll | 0.4 .. 86.2 |
| normal cuff | 0.3 .. 4.5 |

```python
if darkness >= ROLL_DARK_MAX and edge_angle <= ROLL_EDGE_ANGLE_MIN:
    return []          # both say normal -> nothing to report
```

OR'd, because a roll square to the axis is still caught by its brightness, and a roll on a
naturally pale cuff is still caught by its angle. The mask is the cuff band pixels
themselves, which follow the glove outline.

**Constants:** `ROLL_CUFF_BAND=(0.80,0.98)`, `ROLL_PALM_BAND=(0.40,0.65)`,
`ROLL_EDGE_BAND=0.96`, `ROLL_DARK_MAX=12.0`, `ROLL_EDGE_ANGLE_MIN=8.0`,
`ROLL_MIN_BAND_PX=40`.

**Result:** 8/8, 1 false positive across the 7 fold controls.

**Caveat for the report:** both thresholds were fitted on 15 images (8 defective,
7 control) from a small number of physical gloves. The separation is clean on that data,
but it is a small sample.

---

## Part 4 -- Region highlighting

Every detector originally reported only a rectangle, and a rectangle describes these
defects badly: a fold is a thin diagonal crease, a beading gap follows the curve of the
cuff, a roll wraps around the wrist. `Detection.mask` now carries the actual pixels.

Shaded area as a fraction of the box it replaced, over 26 images:

| detector | mask / box |
|---|---|
| damage by fold | 0.09 .. 0.82 (typically ~0.2) |
| incomplete beading | 0.15 .. 0.35 |
| improper roll | 0.06 .. 0.94 |

So for a fold the rectangle was claiming roughly **five times** the damaged area. This
feeds `affected_area_percentage()`, which is reported per image -- the figures were
inflated before this change and are honest after it.

---

## Part 5 -- Things that were tried and rejected

All measured, all documented in the source so they are not retried. Good report material:
they show the method was tested rather than assumed.

| technique | result |
|---|---|
| homomorphic flattening | plausible masks 20/25 -> 15/25, raggedness 68 -> 246 |
| texture as an additive score | 23 -> 17 |
| texture as a veto | F1 0.57 -> 0.52 |
| texture as a rescue | F1 0.57 -> 0.29 |
| contour roughness | only 1.24x separation |
| enclosure ratio | 0.49 vs 0.45 -- no separation |
| global relaxation for emphatic colour | F1 0.584 -> 0.562 |
| grey-world white balance | test pass rate 13 -> 11 |
| opening kernel k=5/7/9 | all broke the tear tests; k=3 kept |

*"Cosmetic mask quality and useful mask quality are not the same thing"* -- the texture
rescue produced visibly tidier masks and halved the F1.

---

## Part 6 -- Honest limitations

These belong in the report's critical analysis, not hidden.

1. **Nothing is scored for localisation.** "11/11" means the right label on the right
   image. The regions were checked by eye at high magnification, not against marked ground
   truth. The side-tear work showed how far apart those two can be: image-level scoring read
   100%/100% while box-level was 22%/33%.
2. **Incomplete beading needs a worn glove.** The cue is the hand behind the gap; an empty
   glove with the same defect shows backdrop and is missed.
3. **Beading and improper roll are not separated.** A rolled cuff also exposes skin at the
   wrist, so 5 of 8 roll photographs also report beading. All three versions of the beading
   detector confused them.
4. **No clean gloves in the dataset**, so a true false-positive rate cannot be measured at
   all. The fold images serve as the only intact-cuff control.
5. **Cross-talk between detectors is unresolved** -- it is a per-region arbitration problem,
   not something any single detector can fix. `deduplicate()` resolves overlaps by
   registration order, which is a blunt instrument: it silently deleted a teammate's
   detection until the order was corrected.
6. **Small samples throughout.** 11 / 7 / 8 images per defect, from a handful of physical
   gloves, several thresholds fitted on those same images.

---

## Part 7 -- Implementation checklist for porting

See `TISHEN_INTEGRATION.md` for the full dependency trace. Short version:

This was superseded. The detectors are no longer ported into the team's
`defect_detection.py`; they live in their own package with their own segmentation, and
are registered into the one shared detector list from a block appended to the end of that
file. See `TISHEN_INTEGRATION.md` for the current arrangement.

- [ ] Copy `src/tishen/` (three files: `__init__`, `detection`, `segmentation`)
- [ ] Append the registration block to the end of `src/defect_detection.py` -- it drops
      the stale entries by name, appends the package versions last, and adapts the
      return type so pixel masks survive
- [ ] Verify: selftest 41, pytest 40, beading 18/18, fold 7/7, roll 8/8
- [ ] Verify the two integration checks: **glove masks differ on 0 images**, and **no
      detection of theirs disappears**

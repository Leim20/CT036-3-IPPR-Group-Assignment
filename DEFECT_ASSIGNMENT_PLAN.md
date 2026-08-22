# Defect Assignment Plan (12 required + 1 buffer)

> Companion files: [SYSTEM_COMPLETION_CHECKLIST.md](SYSTEM_COMPLETION_CHECKLIST.md)
> (overall completion cross-check), [README.md](README.md) (how to add a
> detector, `ctx` dict reference).

## Design principles

1. **12 is the floor, not the target** (the assignment says "minimum 3
   defects per member"). Doing more earns no extra marks, and the report
   is capped at 2500-3000 words -- more defect types means shallower
   justification for each, which actually hurts the 50%
   method-justification score.
2. **Group by technique family, not by name.** 50% of the marks is for
   "selecting between candidate techniques with justification" -- if all
   12 defects are the same trick wearing different labels, this section's
   marks collapse.
3. **Keep 1 buffer (the 13th)**, in case a defect turns out to be
   infeasible with classical CV halfway through. Better to have an unused
   buffer than to come up short on the required 12.
4. Every defect is annotated with pitfalls found through actual testing,
   not armchair difficulty ratings.

## Defects excluded from Figure 1 (and why)

The following are hard to implement reliably with classical image
processing and are not recommended:

| Defect | Why it's hard |
|---|---|
| Inside Out | No colour/shape cue can tell which side is facing up |
| Improper Roll | Needs fine-grained edge geometry modelling, false-positive rate hard to control |
| Plastic Contamination | Transparent foreign material, colour is close to the glove's own -- Lab-distance methods largely fail |
| Damaged by Fold | Almost indistinguishable from normal wrinkling using classical features |
| Double Dipping | Shows up as localised thickening, no stable colour/edge signature |

---

## Assignment table

| Family | Defect (Figure 1 naming) | Core algorithm | Difficulty | Status | Suggested owner |
|---|---|---|---|---|---|
| 1. Mask subtraction + background-colour check | Tearing (enclosed rupture) | `mask_filled - mask_raw`, candidate region's average colour approx equal to the background colour | Easy | Done: `detect_tearing` | **Member A** |
| 2. Convexity defects (narrow + sharp) | Tearing / Open Tear | convexity defects, mouth-width/depth ratio < 0.45 and apex angle < 24 deg | Medium | Done: `detect_open_tears` | **Member A** |
| 2. Convexity defects (narrow + sharp, restricted to the fingertip region) | Tearing (fingertip) | same algorithm + a position test ("the notch apex sits in the glove's top quarter") | Medium | Covered by the regression suite | **Member A** |
| 3. Convexity-defect counting (peaks) | Finger Not Enough / Missing Finger | count convex-hull peaks (fingertips); report if != 5 | Medium | Not started | **Member B** |
| 3. Convexity-defect counting (valleys) | Touching / Fused Fingers | the deep notch that should exist between two adjacent fingers is missing or too shallow | Medium | Not started | **Member B** |
| 4. Material-adaptive Lab/HSV + shape filtering | Stain | material/background chroma plus density/compactness on light material; dominant hue on coloured material, with local Lab only as fallback | Easy | Done: `detect_stains` | **Member B** |
| 4. Lab colour distance (large area, low intensity) | Discoloration | same distance formula, but a much larger area threshold and a much smaller colour-distance threshold | Easy | Not started | **Member C** |
| 4. Lab colour distance (count of small blobs) | Spotting | the criterion is "count of small blobs >= N", not the area of a single blob | Medium | Not started | **Member C** |
| 5. Contour geometry statistics | Oversize / Shape Abnormality | contour area and aspect ratio compared against the mean +/- n standard deviations from clean samples (`good/`) | Easy | Not started | **Member C** |
| 5. Distance transform (local width) | Thin Area | `cv2.distanceTransform`, find where the glove's local width drops sharply | Medium | Not started | **Member D** |
| 6. Edge density (texture) | Wrinkles / Dent | Canny edge density inside the glove region, significantly above the baseline from clean samples | Medium | Not started | **Member D** |
| 7. Edge arc analysis | Incomplete Beading | curvature/arc-smoothness analysis of the wrist cuff segment of the contour | Medium | Not started | **Member D** |
| 6. Local curvature spike (**buffer**) | Knocking | a local curvature spike in the contour, less sharp than a tear (same family as Wrinkle, different thresholds) | Hard, buffer | Optional | whoever finishes first |

**Headcount check**: A=3 (done), B=3 (1 done + 2 new), C=3 (all new), D=3
(all new) + 1 buffer = 13 defects. Everyone has >= 3, the group has >= 12,
satisfying the assignment's hard requirement.

---

## Known implementation pitfalls (read before starting, so you don't rediscover them the hard way)

- **Family 1, Tearing (enclosed rupture)**: segmentation fails under "strong side lighting +
  dark glove" (reproducible via the "known limitations" list at the
  bottom of `selftest.py`). This is a segmentation-layer problem, not an
  issue with the hole detector itself.
- **Family 4, Stain result and trade-off**: the detector rebuilds a complete
  surface from normal light material so large brown marks removed by foreground
  segmentation can re-enter the search. Candidates must differ in Lab chroma
  from both material and background and pass density/compactness checks, which
  suppresses yellow background visible through knit holes, finger gaps and
  thin edges. Coloured gloves use dominant hue, with local Lab only when the
  primary rule finds nothing. The tuned set has a **24/24 image-level hit**,
  **0/2 clean-image false positives**, and **36/36 synthetic regressions**.
  Image-level hit means at least one stain was found, not pixel-level accuracy;
  visual overlay review still shows some missed faint/edge marks. These 26 real
  images were used during tuning, so a separate untouched set is required for
  final accuracy. Stricter compactness reduces stray labels but can miss white,
  faint, tiny or edge-adjacent marks. Large gradual shifts belong to the
  Discoloration detector.
- **Family 4, Discoloration vs Stain**: do NOT just relabel
  `detect_stains`'s output and return it twice under a different name --
  that only counts as 1 technique, and this is exactly the red line
  called out in the checklist. The area/intensity thresholds must
  genuinely differ, with test images that prove the two get classified
  separately under your criteria.
- **Family 6, Wrinkles/Dent**: CLAHE in the preprocessing stage amplifies
  local contrast, which can turn normal fabric texture into a false
  "wrinkle". The `img` parameter a detector receives has already been
  through CLAHE -- texture-based detectors shouldn't use it directly;
  call `preprocessing.preprocess(original_img, fix_light=False)` yourself
  and use its second return value instead.
- **Family 5, Oversize / Thin area**: both need a baseline for "what a
  normal glove looks like", which can only be calibrated once real photos
  exist in `dataset/raw/<material>/good/` -- synthetic images can't
  produce trustworthy numbers here.

## Tearing detector naming

`detect_tearing` finds enclosed openings and outputs the assignment-compliant
label **`"Tearing"`**.
`detect_open_tears` outputs `"Open Tear"` for boundary-reaching tears. The
legacy `dataset/raw/hole/` folder remains supported by the evaluator and maps
to `"Tearing"`.

---

## What each person does next (using Member B as the example)

1. Follow the pattern in the README's "team division of labour" section:
   add `detect_missing_finger(ctx)` / `detect_fused_fingers(ctx)` to
   `src/defect_detection.py`, and register them in `DETECTORS`.
2. Add the corresponding folder-name -> label rows to `LABEL_MAP` at the
   top of `src/evaluate.py`.
3. Add synthetic test scenarios for your defect(s) in `src/selftest.py`
   (see how `open_tear`/`fingertip_tear` are drawn for reference).
4. When shooting real photos, put them under
   `dataset/raw/<material>/<your defect folder>/`, following the rules in
   [dataset/README.md](dataset/README.md).
5. Run `.venv\Scripts\python src\evaluate.py` to check the recall/precision
   for your rows.

## Suggested: turn this table into GitHub Issues

Open one Issue per row (excluding the 3 already-done ones), titled
"[technique family] defect name", with the body copied from that row's
algorithm approach + known pitfalls, assigned to the corresponding
member. Repository: https://github.com/Leim20/glove-defect-detection

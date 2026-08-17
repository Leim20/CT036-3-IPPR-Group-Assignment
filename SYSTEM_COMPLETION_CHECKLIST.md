# GDD System Completion & Cross-Check Checklist

> Scope: only the glove defect detection system, source code, algorithms, parameters,
> dataset and testing. Project report, presentation and individual reflection are excluded.
>
> Purpose: this file records what must still be completed and gives Claude Code a
> reproducible checklist for independent cross-checking.

## 1. Current verdict

The current project is a runnable prototype skeleton, not a submission-ready GDD system.

What currently works:

- [x] Python preprocessing -> segmentation -> detection -> GUI pipeline runs.
- [x] GUI can open an image, run detection, draw boxes and show defect names.
- [x] The synthetic self-test detects one enclosed hole and one obvious stain.
- [x] No Haar Cascade, TensorFlow or template/pattern matching implementation was found.
- [x] Source files compile in the current environment.

Major incomplete requirements:

- [ ] At least 12 distinct defect types for a four-member group.
- [ ] At least 5 test images for every defect type.
- [ ] Approximately 60 self-created test images in total.
- [ ] Images divided across at least 3 glove materials.
- [ ] Accuracy and per-class evaluation against labelled ground truth.
- [ ] Robustness to environmental changes.
- [ ] Safe handling of segmentation failure and images containing no glove.

Important distinction: Figure 1 shows possible defects. It is not necessary to
implement every defect pictured, but the assignment explicitly requires at least
3 defects per member. With 4 members, the group minimum is 12 distinct defects.

## 2. Current implementation inventory

### Implemented pipeline

| Stage | File | Current implementation |
|---|---|---|
| Preprocessing | `src/preprocessing.py` | Resize to 800 px width and 5x5 median blur |
| Segmentation | `src/segmentation.py` | Centre-patch HSV reference colour, fixed thresholds, morphology and largest contour |
| Defect detection | `src/defect_detection.py` | Enclosed hole/tear and colour-based stain detectors |
| GUI | `src/gui.py` | Open image, detect, show annotated image and list detections |
| Smoke test | `src/selftest.py` | One synthetic blue glove with one hole and one stain |

### Registered defect types

Only these two detectors are registered in `DETECTORS`:

1. `Tear / Hole`
2. `Stain`

`detect_missing_finger` and `detect_wrinkles` are comments/placeholders, not
implemented features.

Nominal defect coverage is currently 2/12 = 16.7% of the assignment minimum.
Actual coverage is lower because `Tear / Hole` does not detect an open tear that
reaches the glove boundary.

## 3. P0 blockers - must be fixed before calling the system complete

### P0.1 Define and implement the final 12-defect taxonomy

- [ ] Agree on exactly 12 or more distinct defect labels.
- [ ] Assign at least 3 defect types to every group member.
- [ ] Implement a dedicated detector or justified shared detector with distinct
      classification logic for every selected defect.
- [ ] Register every completed detector in `DETECTORS`.
- [ ] Ensure the GUI reports the correct defect name, not a generic substitute.
- [ ] Add positive and negative automated tests for every detector.

Suggested candidates from Figure 1:

- Tearing (fingertip)
- Tearing/open tear
- Hole/puncture
- Knocking
- Incomplete Beading
- Dirty
- Stain
- Double Dipping
- Finger Not Enough/missing or incomplete finger
- Touching/fused fingers
- Discoloration
- Wrinkles/Dent
- Oversize/shape abnormality
- Plastic Contamination
- Thin area
- Damaged by Fold
- Spotting
- Inside Out
- Improper Roll

Do not count `Dirty`, `Stain`, `Discoloration`, `Spotting` and `Plastic
Contamination` as separate completed types if all of them merely return the label
`Stain`. Distinct rubric credit requires the system to recognise and display the
intended type, supported by test evidence.

### P0.2 Replace or harden glove segmentation

Current critical dependency: `segment_glove()` assumes the glove occupies the
centre 20% of the image. This contradicts the requirement to detect gloves in an
arbitrary supplied image and causes background selection when the glove is not
centred.

- [ ] Remove the centre-patch-as-ground-truth assumption, or explicitly detect
      when that assumption is invalid.
- [ ] Add a segmentation confidence/validity result.
- [ ] Reject or report `No glove detected` when no plausible glove contour exists.
- [ ] Never convert segmentation failure into `glove PASSED inspection`.
- [ ] Reject masks that cover an implausibly small or large fraction of the frame.
- [ ] Handle multiple large regions and background regions touching image borders.
- [ ] Handle HSV hue circular wrap-around at 179 -> 0.
- [ ] Validate segmentation on all three selected materials and backgrounds.
- [ ] Measure mask IoU or Dice score using labelled masks.

Minimum segmentation acceptance criteria:

- [ ] No-glove images produce `No glove detected`, never `PASSED`.
- [ ] Centred and off-centre gloves are both accepted.
- [ ] Similar-colour glove/background cases either segment correctly or return a
      clear low-confidence failure.
- [ ] Red hue wrap-around does not select the background.
- [ ] A failed mask cannot proceed silently to defect detection.
- [ ] Define and meet a documented mask IoU/Dice target on a held-out dataset.

### P0.3 Fix tearing/hole scope

The current subtraction method finds enclosed background-coloured regions inside
the filled outer contour. It does not detect a tear open to the glove boundary.

- [ ] Separate `enclosed hole/puncture` from `open tear` if both are selected.
- [ ] Detect boundary notches, fingertip tears and edge-connected tears.
- [ ] Test small, medium and large defects.
- [ ] Test tears on fingertips, palm, wrist/beading and side boundaries.
- [ ] Test irregular/elongated tears, not only perfect circles.
- [ ] Ensure normal finger gaps and contour concavities are not reported as tears.

### P0.4 Build the assignment dataset

The current `dataset/` directory only contains debug masks and a generated
self-test result. These are not the required assignment dataset.

- [ ] Create approximately 60 original glove images.
- [ ] Include at least 5 labelled images for every selected defect.
- [ ] Cover at least 3 materials, for example latex, rubber and leather.
- [ ] Include defect-free images for false-positive measurement.
- [ ] Include images with multiple simultaneous defects.
- [ ] Include environmental variations: position, rotation, scale, lighting,
      shadow, background, blur, noise and camera distance.
- [ ] Preserve original images; do not test only on generated result images.
- [ ] Record ground-truth class and bounding box/mask for every test image.
- [ ] Prevent the same or near-duplicate image from appearing in both tuning and
      final test subsets.

Recommended structure:

```text
dataset/
  raw/
    latex/<defect_name>/
    rubber/<defect_name>/
    leather/<defect_name>/
  annotations/
  splits/
    tune.txt
    validation.txt
    test.txt
  negatives/
  multi_defect/
```

### P0.5 Add measurable evaluation

- [ ] Define what counts as a correct detection, including bounding-box IoU.
- [ ] Report per-class TP, FP, FN and TN where applicable.
- [ ] Calculate per-class precision, recall and F1.
- [ ] Calculate macro averages so frequent classes do not hide weak classes.
- [ ] Produce a confusion matrix for defect classification.
- [ ] Measure false-pass rate on defective gloves.
- [ ] Measure false-positive rate on defect-free gloves.
- [ ] Measure segmentation IoU/Dice separately from defect classification.
- [ ] Save machine-readable results, not only screenshots.
- [ ] Record failure cases and the exact parameters/version used.

The assignment minimum of five images per defect is a rubric minimum, not enough
to establish strong statistical confidence. Add more images where practical.

## 4. P1 algorithm and parameter work

The current parameters exist, but there is no evidence that they were calibrated
against labelled images.

| Parameter | Current value | Required work |
|---|---:|---|
| Resize width | 800 px | Confirm sufficient detail for smallest selected defect |
| Median blur | 5x5 | Test whether fine tears/spots are erased |
| HSV H tolerance | 12 | Fix hue wrap-around and tune on all materials |
| HSV S tolerance | 90 | Tune against low-saturation latex/leather cases |
| HSV V tolerance | 110 | Current range easily merges similar backgrounds |
| Segmentation morphology | 7x7, close x2, open x1 | Test bridging fingers and removing small defects |
| Hole minimum area | 80 px2 | Current synthetic test missed circular holes of radius <=6 px |
| Hole morphology | 3x3 open | Test whether it removes narrow tears |
| Stain minimum area | 60 px2 | Tune for small spotting and sensor noise |
| Lab colour distance | 45 | Current synthetic sweep missed mild colour changes |
| Stain erosion | 9x9 | Test stains close to glove boundaries and fingertips |

Required parameter process:

- [ ] Move thresholds into a documented configuration object/file.
- [ ] Define the valid scale/unit of every threshold.
- [ ] Tune only on the tune/validation split.
- [ ] Freeze parameters before final test evaluation.
- [ ] Record per-material performance.
- [ ] Use a reproducible parameter-search or threshold-sweep script.
- [ ] Add regression tests for the selected final thresholds.
- [ ] Pin dependency versions in `requirements.txt` or a lock file.

## 5. P1 GUI and failure-state improvements

- [ ] Add a visible `No glove detected` state.
- [ ] Add a visible `Segmentation uncertain` state.
- [ ] Do not show `PASSED` unless glove detection/segmentation passed validation.
- [ ] Display every recognised defect using the exact agreed taxonomy.
- [ ] Show multiple simultaneous defects without overwriting results.
- [ ] Optionally show segmentation preview/debug information for demonstration.
- [ ] Catch processing exceptions and show a useful message rather than crashing.
- [ ] Confirm image loading for JPG, JPEG, PNG and BMP; treat GIF support carefully.
- [ ] Test unusual aspect ratios and very large/small source images.

## 6. Automated tests that are still required

### Pipeline tests

- [ ] Valid image completes the complete pipeline.
- [ ] Invalid/corrupt image is rejected.
- [ ] No image selected is handled safely.
- [ ] No glove in image returns `No glove detected`.
- [ ] Empty or invalid mask cannot produce `PASSED`.
- [ ] All detector outputs have valid names and in-bounds boxes.
- [ ] Drawing results does not modify the input image unexpectedly.

### Segmentation tests

- [ ] Centre, left, right, top and bottom placements.
- [ ] Multiple rotations and scales.
- [ ] Bright, dark and uneven lighting.
- [ ] Shadows and highlights.
- [ ] Similar-colour and high-contrast backgrounds.
- [ ] Low-saturation/white latex, dark rubber and textured leather.
- [ ] Hue values near 0/179.
- [ ] Partial glove visibility and cropped gloves.

### Per-defect tests

For each selected defect:

- [ ] At least 5 real positive images.
- [ ] Defect-free negatives of the same material/background.
- [ ] Visually similar competing defect classes.
- [ ] Different positions, sizes, orientations and severities.
- [ ] All three materials where the defect is physically applicable.
- [ ] Multiple-defect combinations.

## 7. Known reproduced failures for Claude Code to cross-check

These were reproduced using temporary synthetic images; the temporary harness was
removed after validation. Claude Code should independently recreate the cases.

| Case | Observed result |
|---|---|
| Centred clean glove | Segmentation IoU approximately 0.9977; no defect |
| Internal circular hole | `Tear / Hole` detected |
| Hole plus stain | Both current labels detected |
| Edge-connected/open tear | Tear missed |
| Off-centre clean glove | IoU approximately 0.1625; often false `Tear / Hole` |
| Extreme horizontal offset | IoU approximately 0.0002; no valid failure warning |
| No glove | Filled mask covers 100% of frame; GUI can report `PASSED` |
| Light glove/light background | Filled mask covers 100% of frame |
| Dark glove/dark background | Filled mask covers 100% of frame |
| Red hue split around H=179/H=1 | Background selected; false tear behaviour |
| Hole radius 3-6 px at 800 px width | Missed |
| Hole radius 7 px and above | Detected in the synthetic sweep |
| Mild synthetic stain colour change | Missed until colour difference became larger |

Synthetic tests are useful regression tests, but they are not a replacement for
accuracy evaluation on the self-created real dataset.

## 8. Reproducible baseline checks

Run from the project root in PowerShell:

```powershell
.venv\Scripts\python.exe -m compileall -q src
.venv\Scripts\python.exe src\selftest.py
```

Expected current self-test output includes both:

```text
Tear / Hole
Stain
```

Static detector registration check:

```powershell
rg -n "DETECTORS|detect_" src\defect_detection.py
```

Prohibited-method check:

```powershell
rg -n -i "haarcascade|cascadeclassifier|tensorflow|keras|matchtemplate|template matching|pattern matching" src requirements.txt
```

Dataset count check:

```powershell
Get-ChildItem dataset -Recurse -File -Include *.jpg,*.jpeg,*.png,*.bmp,*.gif |
    Measure-Object
```

## 9. Definition of done for the system

Do not mark the system complete until all of the following are true:

- [ ] At least 12 distinct defects are implemented and correctly named.
- [ ] Every group member owns at least 3 demonstrated defects.
- [ ] At least 3 glove materials are represented and tested.
- [ ] Every defect has at least 5 labelled test images.
- [ ] Approximately 60 or more original test images are provided.
- [ ] Defect-free and environmental-variation negatives are included.
- [ ] Segmentation failure cannot be reported as a passed glove.
- [ ] Off-centre and valid environmental-variation cases meet agreed targets.
- [ ] Per-class precision, recall and F1 are produced from held-out images.
- [ ] Critical false-pass cases are fixed or explicitly rejected by the system.
- [ ] Parameter values are calibrated, frozen and documented.
- [ ] Dependency versions and test commands are reproducible.
- [ ] GUI displays correct defect types and safe failure states.
- [ ] All relevant source code and test files are present.

## 10. Claude Code cross-check instructions

Claude Code should not mark an item complete merely because a function name exists.
For every completed checkbox, require at least one of:

1. Source-code evidence with file and line number.
2. A reproducible automated test and its output.
3. Dataset evidence with image count and labels.
4. A calculated metric from held-out ground truth.

Claude Code should explicitly challenge these claims:

- Whether `Tear / Hole` detects open boundary tears, not only enclosed holes.
- Whether generic `Stain` output can legitimately count as multiple defect types.
- Whether three materials are actually tested rather than only mentioned in README.
- Whether environmental robustness is measured rather than inferred from using HSV.
- Whether `PASSED` is possible after failed segmentation.
- Whether parameter values were selected using data rather than arbitrary defaults.

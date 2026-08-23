# Glove Defect Detection System (GDD) -- IPPR Group Assignment

APU CT036-3-IPPR group assignment: detect defects in gloves using
**classical image processing methods** (no deep learning -- the assignment
prohibits Haar Cascade / TensorFlow / template matching). Due: 2026-08-16.

## Libraries used (just 3, no heavyweight frameworks)

| Library | What it's for |
|---|---|
| OpenCV (`opencv-python`) | Reading images, filtering, colour segmentation, contour finding -- the core |
| NumPy | Images are number matrices under the hood; NumPy handles the matrix math |
| Pillow + Tkinter | GUI (Tkinter ships with Python) |

## First-time setup (only needs doing once)

Open a terminal (PowerShell) in this folder and run, in order:

```powershell
python -m venv .venv                      # create an isolated Python environment
.venv\Scripts\pip install -r requirements.txt   # install the three libraries
.venv\Scripts\python src\selftest.py      # sanity check: confirm the environment works
```

## Day-to-day usage

```powershell
.venv\Scripts\python src\gui.py
```

First place test photos directly under `dataset/raw/`, or in any subfolders;
the dropdown scans recursively. Once the
window opens, choose a photo from **Test Image**, select one defect or
`All Defects` under **Defect / Detection Mode**, then click **Run Detection**.
The original image appears as soon as it is selected; the result panel shows
fixed-colour pixel overlays, contours, boxes and defect labels. Below it the
GUI reports affected-area percentage, rule evidence and processing time. Use
**Save Result** to export the full-resolution annotated image.

## Project structure (what each file does)

```
IPPR/
├── requirements.txt        list of libraries to install
├── dataset/                self-collected dataset goes here (naming rules below)
└── src/
    ├── preprocessing.py    (1) preprocessing: uniform size + denoise
    ├── segmentation.py     (2) glove segmentation: cut the glove out of the background
    ├── defect_detection.py (3) defect detection: one function per defect -- team members mainly write here
    ├── gui.py               (4) GUI (the main program, run this)
    └── selftest.py          environment sanity-check script
```

Pipeline: `image -> (1) preprocess (denoise + illumination normalisation) -> (2) segment the glove mask -> (3) run each defect detector -> (4) display results in the GUI`

## Verifying nothing has broken

```powershell
.venv\Scripts\python src\selftest.py
```

Runs 36 simulated scenarios (low/bright light, side lighting, noise,
background colour changes, glove offsets, off-colour stains, a clean two-tone material, open tears,
fingertip tears, multiple simultaneous defects, blank background with no
glove, plus a **4 materials x 4 lighting conditions** matrix), compares
each against the expected result, and prints a pass rate. **Run this every
time the algorithms change** -- a failing run exits non-zero.
Currently: 36/36 passing.

At the bottom it also prints a separate **"known limitations"** list (not
counted in the pass rate). The 2 combinations still failing both involve
side lighting with a dark glove -- segmentation relies on a single global
background reference colour plus a global Otsu threshold, which fails when
the background itself has a strong brightness gradient. **These are not gaps
in test coverage, they are known unfixed
issues** -- write them up in the report's critical analysis.

> Caution: synthetic images only validate algorithm logic, they don't
> replace real photographs. Real glove texture, shadows and reflections
> must be validated against the self-collected dataset in `dataset/`.

## Dataset naming convention (assignment requires ~60 images, 3 materials)

**The folder name IS the ground truth** -- no manual bounding-box
annotation needed:

```
dataset/raw/
├── latex/               latex gloves
│   ├── tearing/         tearing photos (>= 5)
│   ├── open_tear/       open tear photos (>= 5)
│   ├── stain/           stain photos (>= 5)
│   ├── tearing+stain/   an image with multiple defects, joined with +
│   └── good/            clean/defect-free gloves (used to measure false positives)
├── rubber/              rubber gloves
└── leather/             leather gloves
```

The folder-name -> defect-label lookup table lives in `LABEL_MAP` at the
top of `src/evaluate.py`. **Whenever a team member adds a new detector,
add a matching row there too**, or the evaluation script won't recognise
the folder.

Shooting requirements (the current segmentation algorithm assumes these,
must be followed):
- **Leave a margin of at least 6% around the glove** -- don't let it touch
  the edges or fill the frame, since segmentation estimates the background
  colour from the image borders
- Use a **plain, single-colour background** that clearly contrasts with
  the glove
- Keep the glove roughly centred (doesn't need to be exact -- measured IoU
  stays at 0.999 even when offset to a corner)

Also deliberately take some photos under different lighting/angles, to
demonstrate the system's robustness (this earns extra marks in the report).

## Batch evaluation (produces the accuracy figures the report needs)

```powershell
.venv\Scripts\python src\evaluate.py
.venv\Scripts\python src\evaluate.py --save-failures   # also save annotated failing images
```

Output: detected / missed / false-positive counts, recall, precision and
F1 per defect type; pass rate per material; false-positive rate on clean
gloves; a list of failure cases. Per-image results are written to
`dataset/evaluation_result.csv`.

Maps onto assignment Sec 4 "test the system to evaluate the accuracy",
Sec 5 "describe the results of testing" / "critical analysis for cases of
images that fail" -- i.e. the 40%-of-marks section.

## Team division of labour (3 defects per person, 12 total across the group)

1. In `defect_detection.py`, model your detector function after
   `detect_holes` / `detect_stains`. **The signature must follow this
   shape:**

```python
def detect_wrinkles(img, mask_filled, mask_raw, bg_color):
    """Wrinkle detection"""
    ...
    return [("Wrinkle", (x, y, w, h)), ...]   # return [] if nothing was found
```

All four parameters are passed to every detector; ignore whichever you
don't need (e.g. `detect_open_tears` only uses `mask_filled`).

2. Add the function's name to the `DETECTORS` list at the bottom of the file;
3. The GUI calls it automatically -- no need to touch the interface code.

### What the four parameters are

| Parameter | Contents | When to use it |
|---|---|---|
| `img` | preprocessed + lighting-fixed BGR image | **colour-based** detection (stains, discolouration) |
| `mask_filled` | the glove's full outline (holes filled in) | testing "is this inside the glove" |
| `mask_raw` | the glove's actual pixels (holes are black) | finding missing regions |
| `bg_color` | background reference colour (Lab) | testing "is this the background showing through" |

For the glove's normal colour, call
`segmentation.get_glove_color(img, mask_raw)`; for its area fraction, call
`segmentation.glove_found(mask_filled)`.

> Texture/edge-based detection (e.g. wrinkles) needs the image WITHOUT the
> lighting fix -- call `preprocessing.preprocess(original_img,
> fix_light=False)` yourself and use its second return value (`img_plain`),
> don't use the `img` you were passed (it's already been through CLAHE,
> which amplifies fabric texture and can turn normal texture into a false
> wrinkle).

`defect_detection.py` already has the tunable threshold constants at the
top; follow the pattern in `detect_holes` / `detect_stains` rather than
writing a separate helper module.

### A crashing detector won't take the rest of the system down with it

`run_all_detectors` wraps every detector in its own try/except. If your
function raises an exception, or its return format is wrong, only yours
gets skipped -- the rest still produce results as normal, and the error
shows up in the GUI as `WARNING - N detector(s) failed`, and is tallied
separately in `evaluate.py`.

So **you don't need to worry about breaking the whole system while
debugging** -- but do note: seeing that WARNING means your code has a bug,
and that image's result is incomplete, so it shouldn't be used to compute
accuracy.

### Duplicate detections are removed automatically

The same defect is often reported by more than one detector at once (e.g.
a large hole can also satisfy a "thin area" test). `run_all_detectors`
calls `deduplicate()` at the end, which keeps the higher-priority hit
based on the **order defects are registered** in `DETECTORS`, so the same
spot isn't double-counted and dragging precision down. **You don't need to
worry about this when writing a detector**, but it's worth thinking about
registration order: put more specific, more reliable detectors earlier in
the list.

Classical techniques you can draw on per defect:
- Enclosed hole/puncture -> mask subtraction + **verifying the region's
  colour equals the background colour** (already implemented)
- Open tear (reaches the edge) -> **convexity defects** + narrow/sharp
  shape criteria (already implemented; note that a normal finger gap is
  also a deep notch, distinguished by "mouth width / depth ratio" and
  "apex angle")
- Stain -> material-adaptive rules: rebuild light-glove material, compare Lab
  chroma against both material and background, then reject sparse knit holes
  and thin edges by density/compactness; coloured gloves use dominant-hue
  deviation with local Lab only as a fallback (implemented)
- Missing/incomplete finger -> convexity defects, count the fingers
- Wrinkle/dent -> edge density (Canny) or texture features (GLCM/LBP)
  within the glove region
- Oversize/shape abnormality -> contour area, aspect ratio compared
  against normal samples

## How this maps onto the report (50% of the marks is method justification!)

Every module's comments explain "why it's done this way" -- expand
directly on these when writing the report:
- Why HSV/Lab instead of RGB (resistant to lighting changes -> satisfies
  the robustness requirement)
- Why median blur, Otsu, morphological operations
- The principle behind each defect detector + failure-case analysis (40%
  of the marks is in experiments and critical analysis)

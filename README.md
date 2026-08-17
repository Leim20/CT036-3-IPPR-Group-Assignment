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

Once the window opens: click **Open Image** -> pick a glove photo -> click
**Detect Defects**.

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

Runs 35 simulated scenarios (low/bright light, side lighting, noise,
background colour changes, glove offsets, off-colour stains, open tears,
fingertip tears, multiple simultaneous defects, blank background with no
glove, plus a **4 materials x 4 lighting conditions** matrix), compares
each against the expected result, and prints a pass rate. **Run this every
time the algorithms change** -- a failing run exits non-zero.
Currently: 35/35 passing.

At the bottom it also prints a separate **"known limitations"** list (not
counted in the pass rate). There are currently 4 entries, all related to
"side lighting" -- segmentation relies on a single global background
reference colour plus a global Otsu threshold, which fails when the
background itself has a strong brightness gradient, especially with dark
gloves. **These are not gaps in test coverage, they are known unfixed
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
│   ├── hole/            hole photos (>= 5)
│   ├── open_tear/       open tear photos (>= 5)
│   ├── stain/           stain photos (>= 5)
│   ├── hole+stain/      an image with multiple defects, joined with +
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
def detect_wrinkles(ctx):
    """Wrinkle detection"""
    ...
    return [("Wrinkle", (x, y, w, h)), ...]   # return [] if nothing was found
```

2. Add the function's name to the `DETECTORS` list at the bottom of the file;
3. The GUI calls it automatically -- no need to touch the interface code.

### What's in the `ctx` dict (pull out whatever you need, no need to recompute anything)

| Key | Contents | When to use it |
|---|---|---|
| `ctx["img"]` | preprocessed + illumination-normalised BGR image | **colour-based** detection (stains, discolouration) |
| `ctx["img_plain"]` | resized + denoised only, no illumination normalisation | **texture/edge-based** detection (wrinkles) |
| `ctx["lab"]` | `img`'s Lab array (already converted) | computing colour distance |
| `ctx["gray"]` | grayscale version | Canny, contours |
| `ctx["mask_filled"]` | the glove's full outline (holes filled in) | testing "is this inside the glove" |
| `ctx["mask_raw"]` | the glove's actual pixels (holes are black) | finding missing regions |
| `ctx["bg_lab"]` | background reference colour | testing "is this the background showing through" |
| `ctx["glove_lab"]` | the glove's normal colour | testing "is this colour normal" |
| `ctx["area_ratio"]` | the glove's area as a fraction of the frame | size/shape abnormality detection |
| `ctx["ok"]` | whether a glove was successfully found | usually no need to worry about this, the framework already handles it |

> Why a dict instead of a long parameter list: the whole team is writing
> 12 detectors, and every time one more thing is needed, a long parameter
> list would mean changing all 12 function signatures and all 4 people
> touching the code. With a dict, you just add a key -- no need to touch
> anyone else's code.

`defect_detection.py` also has a ready-made helper at the top,
`_boxes_from_mask(mask, min_area)`: give it a black-and-white image and it
automatically finds connected blobs, filters out noise, and returns a list
of bounding boxes.

### A crashing detector won't take the rest of the system down with it

Every detector runs in **isolation**. If your function raises an
exception, or its return format is wrong, only yours gets skipped -- the
rest still produce results as normal, and the error shows up in the GUI as
`WARNING - N detector(s) failed`, and is tallied separately in
`evaluate.py`.

So **you don't need to worry about breaking the whole system while
debugging** -- but do note: seeing that WARNING means your code has a bug,
and that image's result is incomplete, so it shouldn't be used to compute
accuracy.

### Duplicate detections are removed automatically

The same defect is often reported by more than one detector at once (e.g.
a large hole can also satisfy a "thin area" test). The framework
automatically calls `deduplicate()` inside `run_all_detectors`, which
keeps the higher-priority hit based on the **order defects are registered**
in `DETECTORS`, so the same spot isn't double-counted and dragging
precision down. **You don't need to worry about this when writing a
detector**, but it's worth thinking about registration order: put more
specific, more reliable detectors earlier in the list.

Classical techniques you can draw on per defect:
- Enclosed hole/puncture -> mask subtraction + **verifying the region's
  colour equals the background colour** (already implemented)
- Open tear (reaches the edge) -> **convexity defects** + narrow/sharp
  shape criteria (already implemented; note that a normal finger gap is
  also a deep notch, distinguished by "mouth width / depth ratio" and
  "apex angle")
- Stain/spot/discolouration -> Lab colour distance + excluding the
  background colour + area filtering (already implemented)
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

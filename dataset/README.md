# Dataset directory

The assignment requires roughly 60 self-collected photos, spread across
**>= 3 materials**, with **>= 5 photos per defect**.

The source-code repository intentionally contains no dataset photographs.
The dataset is distributed as a separate archive. After receiving it, extract
the archive so that the photographs appear under `dataset/raw/` before running
the GUI, self-test, evaluation script, or image-dependent regression tests.

## Directory structure (the folder name IS the ground truth, no manual bounding-box annotation needed)

```
dataset/raw/
├── latex/               latex gloves
│   ├── hole/            holes (>= 5)
│   ├── open_tear/       open tears (>= 5)
│   ├── stain/           stains (>= 5)
│   ├── hole+stain/      an image with multiple defects, joined with +
│   └── good/            clean gloves, used to measure the false-positive rate
├── rubber/              rubber gloves
└── leather/             leather gloves
```

The folder-name -> defect-label lookup table lives in `LABEL_MAP` at the
top of `src/evaluate.py`. **Add a matching row whenever you add a new
detector**, or the evaluation script won't recognise the folder.

## Shooting requirements (the current segmentation algorithm assumes these, must be followed)

- **Leave a margin of at least 6% around the glove** -- don't let it touch
  the edges or fill the frame, since segmentation estimates the background
  colour from the image borders
- Use a **plain, single-colour background** that clearly contrasts with
  the glove
- Keep the glove roughly centred in the frame (doesn't need to be exact)

Also deliberately take some photos under different lighting/angles, to
demonstrate the system's robustness (this earns extra marks in the report).

## Keep dataset photographs out of Git

A single phone photo can be 3-5 MB; 60 of them adds up to 200-300 MB,
which bloats the repository and makes cloning slow. Do not commit photographs
under `dataset/raw/`; `.gitignore` excludes them while retaining the empty
folder structure. Share the dataset archive separately from the source-code
ZIP. If storage size matters, photographs may be downscaled to roughly
**1600px on the long edge** because the application resizes them for processing.

## Files auto-generated in this directory (not tracked in version control)

- `selftest_result.jpg` -- annotated result from `selftest.py`
- `evaluation_result.csv` -- per-image results from `evaluate.py`
- `failures/` -- failing images saved by `evaluate.py --save-failures`

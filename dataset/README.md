# Dataset directory

The assignment requires roughly 60 self-collected photos, spread across
**>= 3 materials**, with **>= 5 photos per defect**.

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

## Downscale images before committing to Git

A single phone photo can be 3-5 MB; 60 of them adds up to 200-300 MB,
which bloats the repo and makes cloning slow. Downscale everything to
roughly **1600px on the long edge** before committing -- the system
already resizes to an 800px width for processing, so this costs no
detection accuracy.

## Files auto-generated in this directory (not tracked in version control)

- `selftest_result.jpg` -- annotated result from `selftest.py`
- `evaluation_result.csv` -- per-image results from `evaluate.py`
- `failures/` -- failing images saved by `evaluate.py --save-failures`

# -*- coding: utf-8 -*-
"""
Preprocessing module
Purpose: bring any input image into a uniform, clean, illumination-consistent
state before it goes any further down the pipeline.

Reasons worth writing up in the report:
  - Uniform size    -> downstream area thresholds (min_area etc.) only mean
                       something consistent if every image is the same size
  - Median blur     -> removes salt-and-pepper noise while preserving edges
                       better than a Gaussian blur (hole edges need to survive)
  - CLAHE           -> cancels out "bright on one side, dark on the other"
                       lighting, which maps directly onto the assignment's
                       hard requirement that "the system must not be
                       sensitive to environmental changes"

Note this returns two images, not one (see normalize_illumination below
for why).
"""
import cv2


# Maximum allowed height after resizing. Guards against pathological aspect
# ratios blowing up memory: a 1x600 image resized to "width = 800" would
# become 800x480000, which triggered a real MemoryError in testing.
MAX_HEIGHT = 2400


def preprocess(img, target_width=800):
    """Step 1: uniform size + denoise. Returns the processed image
    (illumination normalisation has not been applied yet)."""
    h, w = img.shape[:2]
    scale = target_width / w

    # Extreme-aspect-ratio guard: if the resulting height would exceed the
    # cap, scale by height instead. Note that once this guard triggers, the
    # width is no longer 800, so the meaning of the absolute-pixel area
    # thresholds (MIN_AREA_HOLE etc.) shifts. Normal photographs never
    # trigger this branch.
    if h * scale > MAX_HEIGHT:
        scale = MAX_HEIGHT / h

    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)
    img = cv2.resize(img, (new_w, new_h))

    # Median blur: each pixel takes the median of its 5x5 neighbourhood,
    # denoising while keeping edges intact. The kernel can't be larger than
    # the image itself or OpenCV raises an error.
    ksize = min(5, new_w, new_h)
    if ksize % 2 == 0:
        ksize -= 1
    if ksize >= 3:
        img = cv2.medianBlur(img, ksize)
    return img


def normalize_illumination(img, clip_limit=2.0, grid=8):
    """Step 2: illumination normalisation (CLAHE = Contrast Limited
    Adaptive Histogram Equalisation).

    Approach: convert to Lab colour space and equalise only the lightness
    channel L, leaving the colour channels a/b untouched. This flattens out
    uneven brightness without altering the glove's actual colour.

    Experiment log (worth citing in the report's critical analysis):
      We also tried grey-world white balance, but across our 13 synthetic
      test scenarios the pass count actually *dropped* from 13 to 11. The
      reason: when the background is strongly coloured, white balance
      forcibly desaturates it towards grey, which weakens the colour
      contrast between glove and background. We ended up not using it.

    Caution: CLAHE amplifies local contrast, which on real photographs will
    also amplify fabric texture and sensor noise. That's a risk for
    texture/edge-based detectors like wrinkle detection, so we keep the
    un-normalised image around too (see img_plain in
    segmentation.build_context).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def prepare(img, target_width=800, use_clahe=True):
    """One-stop entry point: returns (img_norm, img_plain).

    - img_plain: resize + denoise only            -> texture detectors (wrinkles etc.) use this
    - img_norm : img_plain plus illumination norm  -> segmentation and colour detectors use this

    Set use_clahe=False to switch off illumination normalisation with one
    flag, handy for A/B experiments.
    """
    img_plain = preprocess(img, target_width)
    img_norm = normalize_illumination(img_plain) if use_clahe else img_plain.copy()
    return img_norm, img_plain

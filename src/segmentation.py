# -*- coding: utf-8 -*-
"""
Glove segmentation module
Purpose: cut the glove out of the background and produce a black-and-white
mask: white (255) = glove, black (0) = background.

Design rationale (usable directly in the report):
  1. The reference colour is sampled from the four IMAGE BORDERS and treated
     as the "background colour".
     -> This is the key design decision in this module, see the comparison
        below:

     Old approach: sample a small patch at the centre of the frame as the
     "glove colour". Two problems:
       (a) if the glove drifts slightly off-centre, or a hole happens to sit
           exactly at the centre, the reference colour is wrong and the
           whole segmentation collapses;
       (b) more seriously, a logical contradiction: segmentation throws away
           pixels that "don't look like the glove colour" -- and a stain is
           exactly that kind of pixel. So the stain gets excluded from the
           glove mask, the stain detector can never find it inside the mask,
           and it instead gets reported by the hole detector as a "hole"
           (the wrong defect type).

     New approach: use the background colour as the reference, and treat
     pixels that "don't look like the background" as the glove. As a result:
       (a) we no longer assume the glove sits at the centre, only that the
           four edges are background (just leave a margin when shooting);
       (b) a stain "is not the background colour" -> it stays inside the
           glove mask -> the stain detector can see it; a hole "reveals
           exactly the background colour" -> only genuine holes fall into
           the hole-candidate region.
       The logical contradiction above disappears on its own; one change
       fixes both problems at once.

  2. Otsu's automatic threshold decides "how far from the background counts
     as glove", instead of a hand-tuned fixed colour range, so it adapts
     across materials and background colours -> satisfies the "must not be
     sensitive to the environment" requirement.
  3. Morphological operations clean up noise, keeping only the single
     largest connected region (that's the glove).
  4. Finally, a sanity check: if the glove's area fraction is too small or
     too large, the verdict is "no glove found" rather than blindly
     reporting "inspection passed".

Experiment log: on the same synthetic test suite, the pass count before vs
after this change went from 6/14 to 14/14 (reproduce with
.venv\\Scripts\\python src\\selftest.py).
"""
import cv2
import numpy as np

# How wide a strip along each edge of the frame to sample as the background
# region (fraction of the frame's width/height).
BORDER_RATIO = 0.06

# Plausible range for the glove's area fraction of the whole frame; outside
# this range we conclude "there is no glove in this image".
MIN_AREA_RATIO, MAX_AREA_RATIO = 0.05, 0.95

# Morphology kernel sizes (see step 3 of segment_glove for why these must
# NOT be set equal to each other).
CLOSE_KSIZE = 7   # closing small gaps: larger is fine
OPEN_KSIZE = 3    # removing noise: must stay small, or it erases away thin
                  # tears and fingertip slivers


def _border_pixels(arr, ratio=BORDER_RATIO):
    """Extract the pixels along the top/bottom/left/right edge strips of
    the frame and concatenate them into one long list."""
    h, w = arr.shape[:2]
    bh, bw = max(int(h * ratio), 1), max(int(w * ratio), 1)
    return np.concatenate([
        arr[:bh].reshape(-1, 3), arr[-bh:].reshape(-1, 3),
        arr[:, :bw].reshape(-1, 3), arr[:, -bw:].reshape(-1, 3),
    ])


def segment_glove(img):
    """Input a BGR image, return (mask_filled, mask_raw, bg_lab):
    - mask_raw   : the glove's actual pixels (holes are black, because a
                   hole reveals the background)
    - mask_filled: the glove's full outer contour filled in (holes are
                   filled white too)
    - bg_lab     : the background reference colour (Lab), needed later to
                   tell "hole" and "stain" apart
    Subtracting the two masks = "inside the glove outline, but coloured
    like the background" -> hole candidates.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # --- 1. Estimate the background colour from the four border strips
    #     (median is more noise/outlier-resistant than the mean) ---
    bg_lab = np.median(_border_pixels(lab), axis=0)

    # --- 2. Compute each pixel's "colour distance" from the background,
    #     then let Otsu automatically find the cut-off ---
    #     Euclidean distance in Lab space tracks human colour-difference
    #     perception more closely than comparing in RGB
    dist = np.linalg.norm(lab - bg_lab, axis=2)
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist_u8, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- 3. Morphology: closing fills small gaps, opening removes small
    #     noise specks ---
    # Warning: the two kernel sizes must NOT be equal -- this was found by
    #   testing, not assumed:
    #   with a 7x7 opening kernel, the thin sliver (~11px wide) left behind
    #   after a fingertip tear opens up gets eroded away completely, and the
    #   tear disappears for good (20 regression scenarios only passed 17/20).
    #   Dropping to 3x3 keeps the tear intact (18/20 -> 20/20 once the test
    #   image itself was fixed).
    # Trade-off: weaker noise suppression on real photographs. The 5x5
    #   median blur in preprocessing already removes a first pass of noise,
    #   so this should be acceptable -- but it must be re-validated once
    #   real photos are available.
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KSIZE,) * 2)
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KSIZE,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k, iterations=1)

    # --- 4. Keep only the largest connected region and fill it in -> mask_filled ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_filled, [biggest], -1, 255, cv2.FILLED)

    # Also restrict the raw mask to the largest region, dropping stray
    # background specks
    mask_raw = cv2.bitwise_and(mask, mask_filled)
    return mask_filled, mask_raw, bg_lab


def build_context(img_norm, img_plain):
    """Run segmentation and package everything the detectors will need
    into a single dict, ctx.

    Why a dict instead of a long parameter list?
      The whole team is writing 12 detectors. If every detector were
      written as detect_xxx(img, mask_filled, mask_raw, bg_lab, ...), then
      every time one more thing is needed, all 12 signatures would have to
      change and all 4 people would have to touch the code. Collecting
      everything into a ctx dict means future additions are just new keys,
      without touching anyone else's function.

    What's in ctx (pull out whatever your detector needs):
      img         preprocessed + illumination-normalised BGR image -> colour detectors use this
      img_plain   resized + denoised only, no illumination norm -> texture/edge detectors use this
      lab         img's Lab float array (already converted, so each
                  detector doesn't have to redo it)
      gray        img's grayscale version
      mask_filled the glove's full outline mask (holes filled in)
      mask_raw    the glove's actual pixel mask (holes are black)
      bg_lab      background reference colour (Lab)
      glove_lab   reference colour for the glove's normal colour (Lab),
                  the median of the glove's interior pixels
      area_ratio  the glove's area as a fraction of the whole frame
      ok          whether a glove was successfully found (False -> the GUI
                  shows "no glove detected")
      errors      list of detectors that failed (filled in by
                  run_all_detectors, shown by the GUI)
    """
    mask_filled, mask_raw, bg_lab = segment_glove(img_norm)
    lab = cv2.cvtColor(img_norm, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Glove reference colour: erode the mask inward first, to avoid the
    # "glove colour mixed with background colour" pixels along the edge
    inner = cv2.erode(mask_raw, np.ones((9, 9), np.uint8)) > 0
    glove_lab = np.median(lab[inner], axis=0) if inner.any() else bg_lab

    area_ratio = float(mask_filled.mean() / 255)
    return {
        "img": img_norm,
        "img_plain": img_plain,
        "lab": lab,
        "gray": cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY),
        "mask_filled": mask_filled,
        "mask_raw": mask_raw,
        "bg_lab": bg_lab,
        "glove_lab": glove_lab,
        "area_ratio": area_ratio,
        "ok": MIN_AREA_RATIO < area_ratio < MAX_AREA_RATIO,
        "errors": [],
    }

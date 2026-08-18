# -*- coding: utf-8 -*-
"""
Glove segmentation: cut the glove out of the background and produce a
black-and-white mask (white = glove, black = background).

Approach: sample the colour along the four image borders as a "background
reference colour", compute each pixel's colour distance from it, and use
Otsu's automatic threshold to find the cut-off.

Why the background colour is used as the reference instead of a patch at
the centre of the frame:
  - the glove isn't always centred, so a centre patch is an easy way to
    grab the wrong reference colour
  - more importantly, it's a logical problem: if the "glove colour" were
    used as the reference, segmentation would exclude any pixel that
    doesn't look like the glove -- and a stain is exactly that kind of
    pixel, so the stain detector could never find it, and it would
    instead get misreported by the hole detector as a "hole" (wrong
    defect type)
  Using the background colour as the reference fixes both problems at
  once: a stain "isn't the background colour" so it stays in the mask,
  and a hole "reveals exactly the background colour" so only genuine
  holes become hole candidates.
"""
import cv2
import numpy as np

BORDER_RATIO = 0.06                          # width of the border strip sampled as background
MIN_AREA_RATIO, MAX_AREA_RATIO = 0.05, 0.95  # plausible range for the glove's area fraction
CLOSE_KSIZE = 7   # morphology closing kernel: fills small gaps, larger is fine
OPEN_KSIZE = 3    # morphology opening kernel: removes noise, must stay small --
                  # 7 erodes away thin tears too (found through testing)


def get_background_color(img):
    """Median colour (Lab space) of the pixels along the four image
    borders -- used as the background reference."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    bh, bw = max(int(h * BORDER_RATIO), 1), max(int(w * BORDER_RATIO), 1)
    border = np.concatenate([
        lab[:bh].reshape(-1, 3), lab[-bh:].reshape(-1, 3),
        lab[:, :bw].reshape(-1, 3), lab[:, -bw:].reshape(-1, 3),
    ])
    return np.median(border, axis=0)


def segment_glove(img):
    """Return (mask_filled, mask_raw):
    - mask_raw   : the glove's actual pixels (holes are black, since a
                   hole reveals the background)
    - mask_filled: the glove's full outline filled in (holes are filled
                   white too)
    Subtracting the two = "inside the outline, but coloured like the
    background" -> hole candidates.
    """
    bg_color = get_background_color(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # each pixel's distance from the background colour, Otsu finds "how
    # far counts as glove" automatically
    dist = np.linalg.norm(lab - bg_color, axis=2)
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KSIZE,) * 2)
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KSIZE,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)

    # keep only the largest connected region, and fill it in
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_filled, [biggest], -1, 255, cv2.FILLED)

    mask_raw = cv2.bitwise_and(mask, mask_filled)
    return mask_filled, mask_raw


def glove_found(mask_filled):
    """Whether the glove's area fraction falls in a plausible range; if
    not, this image doesn't contain a glove. Returns (found, area ratio)."""
    ratio = float(mask_filled.mean() / 255)
    return MIN_AREA_RATIO < ratio < MAX_AREA_RATIO, ratio


def get_glove_color(img, mask_raw):
    """Reference colour for the glove's normal appearance (Lab median).
    Erode the mask inward first, to avoid the mixed glove/background
    pixels along the edge. Returns None if no glove pixels are found.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    inner = cv2.erode(mask_raw, np.ones((9, 9), np.uint8)) > 0
    return np.median(lab[inner], axis=0) if inner.any() else None

# -*- coding: utf-8 -*-
"""
Glove segmentation: separate the glove from the background and return a
binary mask (white = glove, black = background).

Approach: sample the colour of the four image borders as a "background
reference colour", measure how far every pixel is from it, and let Otsu
pick the cut-off automatically.

Why the background colour is the reference, and not a centre patch taken
as the "glove colour":
  - the glove is not necessarily centred, so a centre patch easily picks
    up the wrong reference colour;
  - more importantly it is logically wrong: with a "glove colour"
    reference, segmentation throws away every pixel that does not look
    like the glove -- and a stain is exactly such a pixel. The stain
    detector would then never see it, while the hole detector would
    report it as a "hole" (wrong defect type).
  With a background reference, a stain stays inside the mask because it
  "is not the background colour", and a hole becomes a hole candidate
  because what shows through it *is* the background colour. Both problems
  disappear on their own, with no special-casing.
"""
import cv2
import numpy as np

BORDER_RATIO = 0.06                          # how wide a border strip to sample
MIN_AREA_RATIO, MAX_AREA_RATIO = 0.05, 0.95  # plausible range for the glove area
CLOSE_KSIZE = 7   # closing kernel: fills gaps, being generous is fine
OPEN_KSIZE = 3    # opening kernel: removes speckles, must stay small -- at 7 it
                  # also wipes out the thin sliver of a tear (found the hard way)

# Once the 90th percentile of the chroma difference reaches this, segment on the
# a/b chroma alone and ignore lightness L.
# Why there are two modes:
#   With a coloured background (say a yellow mat) and a white glove, lightness
#   is a liability -- shadows on the mat differ strongly in brightness and get
#   taken for glove, while a real stain sits close to the material in lightness
#   and falls under the threshold. Measured on 4 white-cotton photos, switching
#   to pure chroma raised stain coverage inside the mask from 2-3/4 to 4/4.
#   But a grey glove on a grey-white background is near-neutral in chroma on
#   both sides and can only be told apart by lightness, so there we have to
#   fall back to the full Lab distance.
CHROMA_SEG_MIN_SPREAD = 12.0


def get_background_color(img):
    """Median colour (in Lab) of the four image borders: the background reference."""
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
    - mask_raw   : the glove's actual pixels (a hole is black there, because
                   what shows through a hole is the background)
    - mask_filled: the glove's outline filled in solid (holes turn white too)
    The difference between them = "inside the outline but coloured like the
    background" -> hole candidates.
    """
    bg_color = get_background_color(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Distance from every pixel to the background colour; Otsu decides how far
    # "far enough to be glove" is. When the background itself has a clear hue we
    # compare chroma only, so brightness differences from shadows and highlights
    # cannot interfere (see the note above).
    chroma = np.hypot(lab[:, :, 1] - bg_color[1], lab[:, :, 2] - bg_color[2])
    if float(np.percentile(chroma, 90)) >= CHROMA_SEG_MIN_SPREAD:
        dist = chroma
    else:
        dist = np.linalg.norm(lab - bg_color, axis=2)
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KSIZE,) * 2)
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KSIZE,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)

    # Keep only the largest connected region and fill its interior
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_filled, [biggest], -1, 255, cv2.FILLED)

    mask_raw = cv2.bitwise_and(mask, mask_filled)
    return mask_filled, mask_raw


def glove_found(mask_filled):
    """Whether the glove's area fraction is plausible; if it is not, no glove
    was found in this image. Returns (found, area_fraction).
    """
    ratio = float(mask_filled.mean() / 255)
    return MIN_AREA_RATIO < ratio < MAX_AREA_RATIO, ratio


def get_glove_color(img, mask_raw):
    """Reference colour of the undamaged glove material (median in Lab). The
    mask is eroded inwards first, to avoid the rim of pixels where glove and
    background colours blend. Returns None when no glove pixels are found.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    inner = cv2.erode(mask_raw, np.ones((9, 9), np.uint8)) > 0
    return np.median(lab[inner], axis=0) if inner.any() else None

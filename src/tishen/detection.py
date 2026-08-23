# -*- coding: utf-8 -*-
"""Defect detection for my three defects: incomplete beading, damage by fold
and improper roll.

Self-contained on purpose. This module imports only OpenCV, NumPy and our own
segmentation, never the team's ``defect_detection`` or ``segmentation``. Their
detectors and ours therefore share no code and no state: nothing tuned here can
change any result of theirs.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .segmentation import skin_mask, skin_chroma_mask, segment_glove


# ============================================================
# Our own segmentation, used instead of the one we are handed
# ============================================================
# These three detectors take the team's (img, mask_filled, mask_raw, ...)
# signature so that they can be registered in the one shared GUI. They do NOT
# use the masks that signature delivers.
#
# The reason is that a detector is only as good as the mask it reasons over,
# and the shared mask is tuned for the team's structural defects. Beading works
# on skin visible through the cuff, fold on the glove surface, roll on the cuff
# band -- all three need the segmentation tuned for them, and tuning the shared
# one would move every teammate's results. So each detector re-segments the
# image with OUR segmentation and ignores the mask it was given.
#
# There is precedent for this in the team's own pipeline: the Stain and Plastic
# detectors already fall back to _owned_colour_detector_segmentation() when the
# shared mask does not suit them.
#
# Segmentation is the expensive step and all three detectors want the same
# result, so it is computed once per image and cached. The key is a cheap
# digest of a subsampled copy: hashing a full 800x1400 frame on every call
# would cost more than it saves.
_MASK_CACHE = {}
_MASK_CACHE_LIMIT = 8


def our_masks(img):
    """(mask_filled, mask_raw) from OUR segmentation, cached per image."""
    key = (img.shape, hash(img[::16, ::16].tobytes()))
    hit = _MASK_CACHE.get(key)
    if hit is None:
        hit = segment_glove(img)
        if len(_MASK_CACHE) >= _MASK_CACHE_LIMIT:
            _MASK_CACHE.pop(next(iter(_MASK_CACHE)))
        _MASK_CACHE[key] = hit
    return hit


# Annotation reference size: a landscape image at the standard 800px width
# counts as scale 1.0.
DRAW_REF_SIZE = 800.0


# BGR colours: each defect keeps one colour in the result image, which doubles
# as the GUI's legend.
DEFECT_COLORS = {
    "Stain": (0, 140, 255),          # orange
    "Tearing": (40, 40, 220),       # red
    "Open Tear": (190, 55, 150),    # purple
    "Finger Not Enough": (0, 165, 255),
    "Thin / Overstretched": (255, 0, 255),
    "Spotting": (40, 200, 255),
    "Plastic Contamination": (210, 180, 40),  # cyan-blue
    "Incomplete Beading": (255, 90, 60),      # blue
    "Damage By Fold": (60, 200, 60),          # green
    "Improper Roll": (255, 255, 0),           # cyan
}


DEFAULT_DEFECT_COLOR = (35, 160, 70)  # any detector added later: green


# ============================================================
# Incomplete beading: the cuff hem is interrupted
# ============================================================
# The bead is the finished hem at the cuff -- a maroon knitted band on the
# cotton gloves, a rolled edge on latex and nitrile. "Incomplete beading"
# means a stretch of that hem is missing, leaving a ragged opening at the
# wrist through which the hand is visible.
#
# TWO EARLIER VERSIONS FAILED. Both are worth recording because the way
# they failed is what pointed at the right cue.
#
#   1. A 1-D SIGNATURE ALONG THE CUFF BOUNDARY (Ch 10/11): walk the outline
#      and flag runs where the colour just inside it, or its roughness,
#      departs from the rest of the same cuff. It labelled all 11
#      photographs, but on the clearest one -- a hem torn open across a
#      third of the cuff -- it marked a 700 px sliver at one tip and missed
#      the rest. A hem gap does not change the silhouette: the material
#      either side of it still bounds the outline, so a boundary signature
#      has almost nothing to read.
#
#   2. THRESHOLDING THE CUFF BAND AGAINST ITS OWN MEDIAN COLOUR. This was
#      worse, and in an instructive way: it marks whatever colour is in the
#      MINORITY inside the band. On the cotton gloves that is the maroon
#      bead -- so it drew its region neatly around the part of the hem that
#      is STILL THERE, which is the exact inverse of the defect. On the
#      nitrile ones it found the shadow under the cuff or a crease in the
#      bunched material. It never once found the gap, because a gap is not
#      a colour anomaly; it is an ABSENCE.
#
# The defect is missing material. These gloves are photographed being worn,
# so what shows through the gap is the hand:
#
#   1. take the unguarded skin-colour test (`skin_chroma_mask`, see the note
#      there -- `skin_mask` would discard the patch showing through a breach
#      because it keeps only components that touch the image border);
#   2. keep the skin lying INSIDE the glove's convex hull but OUTSIDE the
#      glove itself. That is skin where material ought to be;
#   3. drop any component that runs off the frame -- that is the forearm,
#      which always does, while a breach never does;
#   4. keep what is left near the cuff end of the major axis.
#
# Step 4 is what makes it work. The skin test also fires on the shadow
# between glove and backdrop on the yellow-backdrop photographs, and those
# shadows are LARGER than the real tears, so size alone picks the wrong one.
# Position separates them completely -- measured over the 11 photographs:
#
#     real breach at the cuff      axis fraction 0.81 .. 1.00
#     backdrop shadow elsewhere    axis fraction 0.05 .. 0.66
#
# so the cut-off sits between them at 0.75. Enclosure (how much of a
# component's surrounding ring is glove) was measured as an alternative and
# rejected: 0.46-0.73 for real breaches against 0.10-0.53 for the shadows,
# which overlaps.
#
# KNOWN LIMITATION, and it needs stating in the report: this detector reads
# the HAND through the gap, so it only works on a glove being worn. An empty
# glove with the same defect would show backdrop through the gap instead and
# would not be found. Every photograph in our set is of a worn glove, so the
# limitation is invisible in these results -- which is exactly why it has to
# be written down rather than discovered by whoever tests it next.
BEAD_CUFF_BAND = 0.75       # breach centroid this far along the major axis


BEAD_MIN_AREA_FRAC = 0.0001     # smallest believable breach / axis^2


BEAD_OPEN = 5               # drop speckle


BEAD_CLOSE = 15             # close the breach into one region


BEAD_MAX_REGIONS = 2


BEAD_BOX_PAD = 8


# The skin-colour test also passes the SHADOW the glove casts on a warm
# backdrop, and on one photograph that shadow is larger than the tear and
# sits at the cuff end of the axis, so neither size nor position rejects it.
# What rejects it is that skin seen through a hole is the SAME skin, under
# the same light, as the forearm already visible in the picture -- so its
# HUE must match, and hue is what survives shadow (that is the whole basis
# of the background key in segmentation.py). Measured against the forearm's
# own median hue:
#     real breach            0 .. 2
#     shadow on the backdrop 4
# The margin is narrow and rests on a single counter-example, so this is a
# threshold to re-check if the detector is ever run on new photographs.
BEAD_MAX_HUE_SHIFT = 3.0


# ============================================================
# Damage by fold: a crease left where the glove was folded
# ============================================================
# A fold leaves a long dark crease across the glove SURFACE -- unlike the
# other defects here it is not a boundary feature at all, so none of the
# contour machinery applies.
#
# Morphological BLACKHAT with a LINE structuring element responds to dark
# structures narrower than the element. A line roughly a tenth of the
# glove's length therefore picks up a crease while ignoring the woven
# texture of a fabric glove, which is fine-scale in EVERY direction and so
# never fills a long line. Sweeping the element over 12 orientations and
# keeping the maximum makes the response independent of which way the
# fold runs. (Ch 8 morphology; Ch 7 line detection.)
#
# Two regions have to be excluded, both found by looking at the response:
#   * the glove boundary -- finger gaps are dark valleys and light up hard
#   * the cuff -- knitted ribbing is a regular line pattern that swamps a
#     real crease
FOLD_N_ORIENT = 12          # line elements every 180/12 degrees


FOLD_LINE_FRAC = 0.10       # length of the line element / glove major axis


FOLD_ERODE_FRAC = 0.045     # stay this far inside the glove boundary


FOLD_CUFF_EXCLUDE = 0.72    # ignore the cuff band entirely


# Rank threshold, not median+k*sigma: on a low-contrast crease the sigma of
# the WHOLE glove buries the defect. Two images were missed for exactly
# that reason even though the response traced their folds perfectly.
FOLD_TOP_PERCENT = 6.0


# A fold is LONG and STRAIGHT, so candidates are scored on length x
# elongation: a shadow blob is neither, a strip of grip pattern is straight
# but short. The length floor was 0.18 and had to come down -- measured,
# real creases run 119-144 px on gloves whose axis put that floor at
# 154-188 px, so genuine folds were rejected on length alone.
FOLD_MIN_LEN_FRAC = 0.10


FOLD_MIN_ELONG = 2.5


FOLD_MERGE_GAP_FRAC = 0.06  # boxes closer than this are fragments of one crease


FOLD_MAX_BOXES = 2


# ============================================================
# Improper roll: the cuff is rolled or bunched instead of lying flat
# ============================================================
# The opposite of incomplete beading. Beading is a GAP in the hem;
# improper roll is EXCESS material, rolled or twisted into a thick uneven
# band at the wrist.
#
# Two independent signatures, both physical rather than fitted:
#
#   * A flat cuff sits in the shadow of the wrist and reads DARKER than
#     the palm. A rolled cuff bulges towards the camera, catches the light
#     along its ridge, and stops being darker. Measured as the lightness
#     of the palm band minus the lightness of the cuff band:
#         improper roll   -27 .. 11
#         normal cuff      10 .. 34
#
#   * A properly worn cuff ends in an edge roughly PERPENDICULAR to the
#     glove's major axis. A rolled one is tilted:
#         improper roll   0.4 .. 86.2 degrees off perpendicular
#         normal cuff     0.3 ..  4.5
#
# Either one alone is enough, so the two are OR'd: a roll that happens to
# sit square to the axis is still caught by its brightness, and a roll on
# a glove whose cuff is naturally pale is still caught by its angle.
ROLL_CUFF_BAND = (0.80, 0.98)   # the cuff, as a fraction along the major axis


ROLL_PALM_BAND = (0.40, 0.65)   # the palm, used as the brightness reference


ROLL_EDGE_BAND = 0.96           # terminal edge = beyond this fraction


ROLL_DARK_MAX = 12.0            # cuff lighter than this margin below the palm -> rolled


ROLL_EDGE_ANGLE_MIN = 8.0       # cuff edge tilted more than this off perpendicular -> rolled


ROLL_MIN_BAND_PX = 40           # too little cuff visible to judge


@dataclass
class Detection:
    """One located defect.

    ``mask`` is a uint8 binary image the same size as the preprocessed picture,
    used for pixel-level shading and for affected-area; ``evidence`` is a
    rule-based strength from 0 to 100, not a machine-learning probability.

    ``__iter__`` exists so that older code can still write
    ``for name, box in defects``.
    """

    name: str
    box: tuple
    mask: np.ndarray | None = None
    evidence: float = 0.0

    def __iter__(self):
        yield self.name
        yield self.box

    def __getitem__(self, index):
        """Keep legacy tests and tuple-style callers working."""
        return (self.name, self.box)[index]


# ============================================================
# Shared glove-axis geometry helpers
# =====================================================
def _basic_rectangle_axis(cnt):
    """Major and minor axis of the glove, taken from its basic rectangle
    (Ch 10/11 boundary descriptors: diameter -> major axis -> minor axis
    perpendicular to it -> basic rectangle).

    Returns (axis, perp, lo, hi): `axis` is the unit vector along the
    major axis, `lo`/`hi` bracket the glove's extent along it, so any
    point's position can be expressed as a fraction of the glove's length
    rather than in pixels -- which is what makes the position test work
    at any resolution and any glove rotation.
    """
    (_, _), (w, h), ang = cv2.minAreaRect(cnt)
    if w >= h:          # make sure `ang` refers to the LONGER side
        ang += 90
    th = np.radians(ang)
    axis = np.array([-np.sin(th), np.cos(th)], np.float32)
    perp = np.array([np.cos(th), np.sin(th)], np.float32)
    proj = cnt.reshape(-1, 2).astype(np.float32) @ axis
    return axis, perp, float(proj.min()), float(proj.max())


def _fingers_at_low_end(mask_filled, axis, perp, lo, hi, stations=60, img=None):
    """Which end of the major axis holds the fingers?

    We cannot assume the glove points "up" in the photo, and the lateral
    band test is meaningless until we know which end is which.

    The cue is FILL RATIO along a slab cut across the glove: at the
    fingertip end a cut crosses several separate fingers with background
    between them, so the glove occupies only part of the span it covers;
    at the cuff end it crosses one solid band and fills the span almost
    completely. Run counting was tried first and is the same idea stated
    discretely, but it needs a pixel-gap threshold and a hole or a speck
    of noise inside the mask invents an extra run. Fill ratio is
    continuous, needs no such threshold, and degrades gracefully.

    Measured fill ratio, fingertip end vs cuff end:
        synthetic glove   0.68 vs 1.00
        real 224641       0.80 vs 0.97
        real 224604       0.58 vs 0.88
        real 224955       0.82 vs 0.97
    Both cues agreed on every image tested, which is why only the more
    robust of the two is kept here.
    """
    # PRIMARY CUE: where the bare arm is.
    # These gloves are photographed being WORN, and an arm is attached at
    # the cuff -- never at the fingertips. So whichever end of the axis
    # the skin region sits nearer is the cuff end, and the fingers are at
    # the other one. This is a physical fact about the scene rather than a
    # heuristic about shape, and it was right on 8 of 8 test images where
    # the shape-based cues below were right on only 5.
    if img is not None:
        skin = skin_mask(img)
        sy, sx = np.nonzero(skin)
        if len(sx) >= 200:
            sp = np.stack([sx, sy], 1).astype(np.float32)
            skin_frac = (float((sp @ axis).mean()) - lo) / (hi - lo + 1e-6)
            return skin_frac > 0.5          # skin high => fingers low

    # FALLBACK for an unworn glove (and for the synthetic regression
    # images, which contain no skin at all): fill ratio along the axis.
    # A cut across the fingertip end crosses several separate fingers with
    # background between them, so the glove fills only part of the span;
    # a cut across the cuff fills it almost completely.
    ys, xs = np.nonzero(mask_filled)
    if len(xs) < 50:
        return True
    pts = np.stack([xs, ys], 1).astype(np.float32)
    along = pts @ axis
    across = pts @ perp
    span = hi - lo
    if span < 20:
        return True
    thickness = max(span / stations, 1.0)

    def mean_fill(f0, f1):
        fills = []
        for t in np.linspace(lo + f0 * span, lo + f1 * span, stations // 3):
            sel = np.abs(along - t) <= thickness * 0.5
            if sel.sum() < 5:
                continue
            v = across[sel]
            extent = float(v.max() - v.min()) + 1.0
            # pixels present, divided by the area of the slab they span
            fills.append(float(sel.sum()) / (extent * thickness))
        return float(np.mean(fills)) if fills else 1.0

    # the emptier end is the fingertip end
    return mean_fill(0.02, 0.28) < mean_fill(0.72, 0.98)


def _axis_fraction(mask_filled, axis, perp, lo, hi, low_is_distal):
    """Every mask pixel's position along the major axis, 0 = fingertips."""
    ys, xs = np.nonzero(mask_filled)
    if len(xs) == 0:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    frac = (pts @ axis - lo) / (hi - lo + 1e-6)
    if not low_is_distal:
        frac = 1.0 - frac
    return ys, xs, pts, frac


def _line_element(length, angle_deg):
    """A one-pixel-wide line of the given length and orientation."""
    k = np.zeros((length, length), np.uint8)
    c = length // 2
    a = np.radians(angle_deg)
    dx, dy = np.cos(a), np.sin(a)
    for t in np.linspace(-c, c, length * 2):
        x, y = int(round(c + t * dx)), int(round(c + t * dy))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1
    return k


def crease_response(gray, axis_len):
    """Maximum blackhat response over orientations: how much each pixel
    looks like part of a dark linear valley."""
    length = max(int(FOLD_LINE_FRAC * axis_len) | 1, 9)
    best = np.zeros(gray.shape, np.float32)
    for i in range(FOLD_N_ORIENT):
        element = _line_element(length, i * 180.0 / FOLD_N_ORIENT)
        response = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, element)
        best = np.maximum(best, response.astype(np.float32))
    return best


def detect_incomplete_beading(img, mask_filled, mask_raw, bg_color,
                              img_plain=None, material=None):
    """A stretch of the cuff hem is missing.

    See the note above the BEAD_* constants. In short: find the hand showing
    through the glove near the cuff -- that is where the hem is not.
    """
    mask_filled, mask_raw = our_masks(img)      # ignore the mask we were given
    contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    axis, perp, lo, hi = _basic_rectangle_axis(cnt)
    axis_len = hi - lo
    if axis_len < 40:
        return []
    low_is_distal = _fingers_at_low_end(mask_filled, axis, perp, lo, hi, img=img)

    # The forearm is the colour reference: whatever shows through a breach has
    # to look like it. Without one there is nothing to check against, so the
    # detector stands down rather than guessing.
    arm = skin_mask(img) > 0
    if np.count_nonzero(arm) < 500:
        return []
    hue = cv2.cvtColor(cv2.medianBlur(img, 5), cv2.COLOR_BGR2HSV)[:, :, 0].astype(np.float32)
    arm_hue = float(np.median(hue[arm]))

    hull = np.zeros(mask_filled.shape, np.uint8)
    cv2.drawContours(hull, [cv2.convexHull(cnt)], -1, 255, cv2.FILLED)

    # skin inside the glove's outline but where the glove is not
    breach = cv2.bitwise_and(cv2.bitwise_and(skin_chroma_mask(img), hull),
                             cv2.bitwise_not(mask_raw))
    breach = cv2.morphologyEx(breach, cv2.MORPH_OPEN,
                              np.ones((BEAD_OPEN,) * 2, np.uint8))
    breach = cv2.morphologyEx(breach, cv2.MORPH_CLOSE,
                              np.ones((BEAD_CLOSE,) * 2, np.uint8))

    h, w = mask_filled.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(breach, 8)
    min_area = BEAD_MIN_AREA_FRAC * axis_len * axis_len
    keep = []
    for i in range(1, n):
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        cw, ch = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
            continue                       # runs off frame: this is the arm
        if area < min_area:
            continue
        frac = (float(np.asarray(centroids[i]) @ axis) - lo) / (axis_len + 1e-6)
        if not low_is_distal:
            frac = 1.0 - frac
        if frac < BEAD_CUFF_BAND:
            continue                       # not at the cuff
        if abs(float(np.median(hue[labels == i])) - arm_hue) > BEAD_MAX_HUE_SHIFT:
            continue                       # wrong hue for skin: a shadow
        keep.append((area, i, x, y, cw, ch))
    keep.sort(reverse=True)

    results = []
    for area, i, x, y, cw, ch in keep[:BEAD_MAX_REGIONS]:
        region = (labels == i).astype(np.uint8) * 255
        bx = max(x - BEAD_BOX_PAD, 0)
        by = max(y - BEAD_BOX_PAD, 0)
        bw = min(cw + 2 * BEAD_BOX_PAD, w - bx)
        bh = min(ch + 2 * BEAD_BOX_PAD, h - by)
        # Evidence: how far past the smallest believable breach this one is.
        # At the acceptance floor it is 50, at ten times it saturates.
        evidence = 50.0 + 50.0 * float(np.clip(
            (area / max(min_area, 1.0) - 1.0) / 9.0, 0.0, 1.0))
        results.append(Detection("Incomplete Beading", (bx, by, bw, bh),
                                 region, round(evidence, 1)))
    return results


def detect_damage_by_fold(img, mask_filled, mask_raw, bg_color,
                          img_plain=None, material=None):
    """A crease left across the glove where it was folded."""
    mask_filled, mask_raw = our_masks(img)      # ignore the mask we were given
    contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    axis, perp, lo, hi = _basic_rectangle_axis(cnt)
    axis_len = hi - lo
    if axis_len < 40:
        return []

    er = max(int(FOLD_ERODE_FRAC * axis_len) | 1, 3)
    inner = cv2.erode(mask_filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er, er)))

    low_is_distal = _fingers_at_low_end(mask_filled, axis, perp, lo, hi, img=img)
    ys, xs = np.nonzero(inner)
    if len(xs):
        pts = np.stack([xs, ys], 1).astype(np.float32)
        frac = (pts @ axis - lo) / (axis_len + 1e-6)
        if not low_is_distal:
            frac = 1.0 - frac
        drop = frac > FOLD_CUFF_EXCLUDE
        inner[ys[drop], xs[drop]] = 0
    if inner.sum() < 500:
        return []

    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    response = crease_response(gray, axis_len)
    response[inner == 0] = 0

    values = response[inner > 0]
    threshold = np.percentile(values, 100.0 - FOLD_TOP_PERCENT)
    binary = ((response > threshold) & (inner > 0)).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # NOTE: directional closing to rejoin a fold that thresholds into a
    # DASHED chain of blobs was tried here and reverted. It did bridge the
    # fragments, but it also merged the ridge into neighbouring creases:
    # two images then produced a single box swallowing most of the glove,
    # two lost their detection entirely, and the two it was aimed at still
    # missed. Worse on every count than leaving the fragments alone.

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    candidates = []
    for i in range(1, n):
        pts = np.argwhere(labels == i)[:, ::-1].astype(np.int32)
        if len(pts) < 30:
            continue
        (_, _), (bw, bh), _ = cv2.minAreaRect(pts)
        length = max(bw, bh)
        elongation = length / (min(bw, bh) + 1e-6)
        if length < FOLD_MIN_LEN_FRAC * axis_len or elongation < FOLD_MIN_ELONG:
            continue
        candidates.append((length * elongation,
                           [int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                            int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])],
                           i))
    candidates.sort(reverse=True, key=lambda c: c[0])

    # one crease often survives as several components, so nearby boxes are
    # merged before the top-N cut -- otherwise the budget is spent on
    # fragments of a single fold rather than on separate defects
    gap = FOLD_MERGE_GAP_FRAC * axis_len
    # Each entry carries the component labels that went into it, so the
    # shaded region can be the crease pixels themselves rather than the box
    # that bounds them -- a fold is a thin diagonal line, and its box is
    # mostly undamaged glove.
    merged = []
    for strength, (x, y, bw, bh), label in candidates:
        for m, labels_in, best in merged:
            if (x < m[0] + m[2] + gap and m[0] < x + bw + gap and
                    y < m[1] + m[3] + gap and m[1] < y + bh + gap):
                nx, ny = min(m[0], x), min(m[1], y)
                m[2] = max(m[0] + m[2], x + bw) - nx
                m[3] = max(m[1] + m[3], y + bh) - ny
                m[0], m[1] = nx, ny
                labels_in.append(label)
                break
        else:
            merged.append(([x, y, bw, bh], [label], strength))

    floor = max(FOLD_MIN_LEN_FRAC * axis_len, 1.0) * FOLD_MIN_ELONG
    results = []
    for box, labels_in, strength in merged[:FOLD_MAX_BOXES]:
        region = np.isin(labels, labels_in).astype(np.uint8) * 255
        # A crease thresholds to a line a few pixels wide; widen it slightly
        # so the shading is visible without swallowing the surrounding glove.
        region = cv2.dilate(region, np.ones((5, 5), np.uint8))
        evidence = 50.0 + 50.0 * float(np.clip(strength / floor - 1.0, 0.0, 1.0))
        results.append(Detection("Damage By Fold", tuple(box), region,
                                 round(evidence, 1)))
    return results


def detect_improper_roll(img, mask_filled, mask_raw, bg_color,
                         img_plain=None, material=None):
    """The cuff is rolled or bunched rather than lying flat."""
    mask_filled, mask_raw = our_masks(img)      # ignore the mask we were given
    contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    axis, perp, lo, hi = _basic_rectangle_axis(cnt)
    axis_len = hi - lo
    if axis_len < 40:
        return []
    low_is_distal = _fingers_at_low_end(mask_filled, axis, perp, lo, hi, img=img)
    got = _axis_fraction(mask_filled, axis, perp, lo, hi, low_is_distal)
    if got is None:
        return []
    ys, xs, pts, frac = got

    cuff_lo, cuff_hi = ROLL_CUFF_BAND
    palm_lo, palm_hi = ROLL_PALM_BAND
    in_cuff = (frac > cuff_lo) & (frac < cuff_hi)
    in_palm = (frac > palm_lo) & (frac < palm_hi)
    if in_cuff.sum() < ROLL_MIN_BAND_PX or in_palm.sum() < ROLL_MIN_BAND_PX:
        return []

    # --- signature 1: the cuff has stopped being darker than the palm ---
    lightness = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    inner = cv2.erode(mask_filled, np.ones((7, 7), np.uint8)) > 0
    cuff_mask = np.zeros(mask_filled.shape, bool)
    cuff_mask[ys[in_cuff], xs[in_cuff]] = True
    palm_mask = np.zeros(mask_filled.shape, bool)
    palm_mask[ys[in_palm], xs[in_palm]] = True
    cuff_mask &= inner
    palm_mask &= inner
    if not cuff_mask.any() or not palm_mask.any():
        return []
    darkness = float(np.median(lightness[palm_mask]) - np.median(lightness[cuff_mask]))

    # --- signature 2: the terminal edge is tilted off perpendicular ---
    edge_sel = frac > ROLL_EDGE_BAND
    edge_angle = 0.0
    if edge_sel.sum() >= ROLL_MIN_BAND_PX:
        edge_pts = pts[edge_sel]
        centred = edge_pts - edge_pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        edge_angle = np.degrees(np.arccos(min(abs(float(vt[0] @ perp)), 1.0)))

    if darkness >= ROLL_DARK_MAX and edge_angle <= ROLL_EDGE_ANGLE_MIN:
        return []

    box = cv2.boundingRect(np.stack([xs[in_cuff], ys[in_cuff]], 1).astype(np.int32))
    # The rolled material is the cuff band itself, which follows the glove's
    # outline. Its bounding box also covers the background on either side of
    # the wrist, so the band is shaded directly instead.
    region = np.zeros(mask_filled.shape, np.uint8)
    region[ys[in_cuff], xs[in_cuff]] = 255

    # Evidence: how far past whichever signature fired. Either alone is
    # sufficient, so the stronger one is taken.
    dark_margin = (ROLL_DARK_MAX - darkness) / max(ROLL_DARK_MAX, 1e-6)
    angle_margin = ((edge_angle - ROLL_EDGE_ANGLE_MIN)
                    / max(3.0 * ROLL_EDGE_ANGLE_MIN, 1e-6))
    evidence = 50.0 + 50.0 * float(np.clip(max(dark_margin, angle_margin), 0.0, 1.0))
    return [Detection("Improper Roll", tuple(int(v) for v in box), region,
                      round(evidence, 1))]


def _annotation_scale(shape):
    """Annotation scale, taken from the image's longest side.

    Why this is needed: preprocessing normalises every image to 800px wide, but a
    portrait shot then runs to about 1400px tall. The GUI panel is a fixed size,
    so a portrait result has to shrink to roughly 0.32x to fit while a landscape
    one only shrinks to 0.58x. With font size and line width hard-coded in
    pixels, the annotations on a portrait photo end up almost invisible.
    Scaling them by the longest side makes both orientations read the same after
    the shrink.
    """
    longest = max(shape[0], shape[1])
    return max(1.0, longest / DRAW_REF_SIZE)


def detection_color(name):
    """The fixed BGR colour this defect gets in the result image."""
    return DEFECT_COLORS.get(name, DEFAULT_DEFECT_COLOR)


def detection_mask(defect, shape):
    """Get the pixel-level defect mask; fall back to the rectangle only for older
    detectors that do not provide one."""
    if isinstance(defect, Detection) and defect.mask is not None:
        if defect.mask.shape[:2] == shape[:2]:
            return defect.mask > 0
    _, (x, y, w, h) = defect
    mask = np.zeros(shape[:2], dtype=bool)
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, shape[1]), min(y + h, shape[0])
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def affected_area_percentage(defects, glove_mask):
    """Defect pixels as a percentage of the reconstructed inspection region.

    Most defect masks lie inside the segmented glove, so this is identical to
    dividing by the filled glove outline. An uncovered finger can legitimately
    lie just outside a material-only cotton mask; including accepted defect
    pixels in the denominator prevents that real region from being reported as
    0.00% affected.
    """
    glove = glove_mask > 0
    glove_pixels = int(np.count_nonzero(glove))
    if glove_pixels == 0 or not defects:
        return 0.0
    affected = np.zeros(glove.shape, dtype=bool)
    for defect in defects:
        affected |= detection_mask(defect, glove_mask.shape)
    inspection_region = glove | affected
    return (
        100.0 * np.count_nonzero(affected)
        / max(int(np.count_nonzero(inspection_region)), 1)
    )


def draw_results(img, defects, alpha=0.38, defect_masks=None):
    """Draw each defect in its own colour: a translucent pixel region, its
    outline, the bounding box and the evidence score.

    Older callers passed a list of masks as the third positional argument.
    Accept that form while the shared pipeline uses masks stored directly on
    ``Detection`` objects.
    """
    legacy_mask_mode = not np.isscalar(alpha)
    if legacy_mask_mode:
        defect_masks = alpha
        alpha = 0.38
    if defect_masks is not None:
        defects = [
            Detection(
                str(name),
                tuple(int(value) for value in box),
                defect_masks[index] if index < len(defect_masks) else None,
                getattr(defect, "evidence", 0.0),
            )
            for index, defect in enumerate(defects)
            for name, box in [tuple(defect)]
        ]
    out = img.copy()
    scale = _annotation_scale(img.shape)
    font_scale = 0.5 * scale
    thin = max(1, int(round(1 * scale)))
    thick = max(2, int(round(2 * scale)))
    pad = max(3, int(round(3 * scale)))
    for defect in defects:
        name, (x, y, w, h) = defect
        color = detection_color(name)
        mask = detection_mask(defect, img.shape)
        if mask.any():
            original_pixels = out[mask].astype(np.float32)
            tint = np.asarray(color, dtype=np.float32)
            out[mask] = np.clip(
                original_pixels * (1.0 - alpha) + tint * alpha, 0, 255,
            ).astype(np.uint8)
            mask_u8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(out, contours, -1, color, thick)

        if legacy_mask_mode:
            continue

        cv2.rectangle(out, (x, y), (x + w, y + h), color, thin)
        evidence = defect.evidence if isinstance(defect, Detection) else 0.0
        label = f"{name} {evidence:.0f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thin,
        )
        gap = pad + 2
        label_y = (y - pad if y - text_h - baseline - gap >= 0
                   else y + text_h + baseline + gap)
        top = max(0, label_y - text_h - baseline - pad)
        bottom = min(out.shape[0] - 1, label_y + pad)
        # Slide the label left when it would run off the right-hand edge.
        # Clamping only the filled rectangle (which is what used to happen)
        # leaves the TEXT hanging outside the picture, so a defect near the
        # right margin loses its name in every saved screenshot.
        label_x = max(0, min(x, out.shape[1] - 1 - text_w - 2 * pad))
        right = min(out.shape[1] - 1, label_x + text_w + 2 * pad)
        cv2.rectangle(out, (label_x, top), (right, bottom), color, cv2.FILLED)
        text_color = (20, 20, 20) if name == "Stain" else (255, 255, 255)
        cv2.putText(
            out, label, (label_x + pad, label_y - 1), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, text_color, thin, cv2.LINE_AA,
        )
    return out


DETECTORS = [
    detect_incomplete_beading,
    detect_damage_by_fold,
    detect_improper_roll,
]

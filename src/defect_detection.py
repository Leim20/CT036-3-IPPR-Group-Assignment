# -*- coding: utf-8 -*-
"""
Defect detection: one function per defect, registered in the DETECTORS
list, called automatically by the GUI.

Every detector shares the same signature:

    def detect_xxx(img, mask_filled, mask_raw, bg_color):
        ...
        return [("Defect Name", (x, y, w, h)), ...]   # return [] if nothing was found

All four parameters are passed to every detector; ignore whichever you
don't need (e.g. the tear detector only needs mask_filled). The defect
name is drawn on the image and also listed in the GUI's text box.
"""
import cv2
import numpy as np

from segmentation import get_glove_color, get_background_colors, skin_mask

# --- Tunable parameters (kept here for sensitivity experiments and for
# citing in the report) ---
BG_MATCH_DIST = 30.0       # hole criterion: candidate's Lab distance from the background must be below this
BG_MATCH_L_WEIGHT = 0.25   # how much LIGHTNESS counts in that distance (chroma always counts fully).
                           # A lighting gradient across the backdrop is a CONTINUUM of brightnesses,
                           # so a hole's lightness can land between two sampled background modes while
                           # its chroma still matches one exactly. Measured on the side-lit test image:
                           # hole vs nearest mode = 49.2 in full Lab (rejected) but 12.0 in chroma
                           # alone (accepted). Down-weighting L keeps the hole/stain separation that
                           # lightness still provides, without letting a shadow veto a real hole.
STAIN_COLOR_DIST = 25.0    # stain criterion: pixel's Lab distance from the glove's normal colour must be above this
MIN_AREA_HOLE = 60
MIN_AREA_STAIN = 60

# Criteria for an open tear (a cut that reaches the glove's boundary),
# derived from measurements, not guessed:
#     normal finger gap : mouth/depth = 0.55-0.74,  apex angle = 29-40 deg
#     open tear         : mouth/depth = 0.36,        apex angle = 20 deg
#     wrist step         : mouth/depth = 5.53,        apex angle = 137 deg
# so "narrow" and "sharp" together are enough to separate a tear from a
# finger gap.
CONTOUR_EPSILON = 2.0          # contour simplification tolerance, removes jagged fake notches
MIN_TEAR_DEPTH_RATIO = 0.05    # notch depth / glove bounding-box diagonal, filters out shallow notches
MAX_TEAR_MOUTH_RATIO = 0.45    # notch mouth width / depth, a tear is a narrow slit
MAX_TEAR_APEX_ANGLE = 24.0     # notch apex angle (degrees), a tear is sharp, a finger gap is blunt

DEDUP_IOU = 0.5   # if two detectors' boxes overlap above this ratio, keep only the one registered first

# Criteria for a SIDE tear (a cut breaching the glove's lateral edge).
# Measured on the synthetic glove, not guessed -- convex-deficiency
# components separate cleanly on two independent axes:
#     component        mouth/depth   axis fraction   depth
#     side tear             0.69         0.473        77.4
#     fingertip tear        0.63         0.045        41.1
#     finger gaps (x4)   1.22-1.94    0.096-0.135  48.7-82.1
#     wrist step           11.32         0.754        34.8
#     shallow notches    13.3-13.4       0.33       9.6-10.9
# Shape alone rejects the finger gaps, the wrist step and the notches;
# position alone rejects the fingertip tear. Requiring both makes the two
# criteria independent, so a real glove only has to satisfy one of them
# well for the detector to stay honest.
SIDE_TEAR_BAND = (0.35, 1.00)       # 0.0 = fingertips, 1.0 = cuff end
# The band was originally (0.30, 0.85), reserving the cuff for the
# improper-roll detector. Measuring the labelled photographs killed that
# assumption: the tears in this dataset sit at axis fraction 0.80-0.98,
# i.e. AT the cuff, and the band alone was blocking 15 of 43 real defects
# -- the single largest source of missed detections. Finger gaps measure
# 0.05-0.25, so the lower bound is what actually rejects them.
#
# Notches are found by CLOSING the mask and subtracting it, not from the
# convex hull. On the synthetic glove the hull worked; on a real glove
# photographed with the fingers spread it does not, because the hull runs
# straight from fingertip to cuff and sits far from the boundary for most
# of its length. Every local notch then merges into one huge deficiency:
# measured on a real image, the tear was swallowed by a 30414 px component
# spanning the whole side of the glove.
#
# Closing with a disc of radius R fills only concavities narrower than
# about 2R, so `close(mask) - mask` isolates notches at a CONTROLLED
# SCALE, with no global reach. R is tied to the glove's own length so the
# detector stays resolution independent.
SIDE_TEAR_CLOSE_RATIO = 0.035       # notch-filling radius / major-axis length.
                                    # 0.020 scores better on the photographs alone
                                    # (F1 0.31 vs 0.27) but collapses the synthetic
                                    # regression suite to 30/35 and 2/10, because the
                                    # smaller disc floods the mask with tiny notches
                                    # and the de-duplication then starves the hole and
                                    # stain detectors. 0.035 is the value that improves
                                    # real performance without breaking the rest of the
                                    # system.
SIDE_TEAR_MIN_AREA_RATIO = 0.0012   # notch area / (major-axis length)^2
SIDE_TEAR_MIN_DEPTH_RATIO = 0.010   # notch depth / major-axis length
# A real tear is an OPENING: it exposes whatever lies behind the glove --
# the hand, the shadow inside the glove, or the backdrop. What comes
# through is strongly unlike the glove's own colour. A shadow ripple on
# the silhouette, or a ragged patch of segmented outline, is still mostly
# glove-coloured. Lab distance from the glove's median colour, measured
# on hand-labelled boxes:
#     confirmed tears      66.2  84.6  93.8  110.8  129.7   (synthetic: 92.8)
#     confirmed non-tears  42.6  54.9  57.1   57.9
# The gap between the clean glove's worst notch (54.9) and the weakest real
# tear (57.9) is only 3 Lab units, so this threshold is the least secure
# number in the detector and should be re-fitted once more labelled
# photographs exist.
# Signed LIGHTNESS was tried first and worked on the real photographs
# (where a tear shows dark skin) but broke the synthetic case, where the
# tear exposes a red backdrop of almost the same lightness as the blue
# glove. Colour distance is direction-agnostic, so it covers both.
# A second, narrower acceptance rule for the CUFF. Seven real tears were
# being rejected by the area and depth floors even though a notch was
# plainly present; measured, they run area 0.00009-0.00062 and depth
# 0.0037-0.0120, well under the main thresholds -- but what shows through
# them is emphatic, colour distance 67-138 against a main threshold of 45.
#
# Relaxing area and depth everywhere for such notches was tried and made
# things worse (F1 0.584 -> 0.562: one extra true detection cost ten false
# ones). Confining the relaxation to the cuff band, where the tears in
# this dataset actually are and where the boundary is most complex, gains
# instead of costing: F1 0.584 -> 0.615.
SIDE_TEAR_CUFF_BAND = 0.85          # this rule applies only past here along the major axis
SIDE_TEAR_CUFF_MIN_AREA = 0.0006
SIDE_TEAR_CUFF_MIN_DEPTH = 0.004
SIDE_TEAR_CUFF_MIN_COLOR = 80.0     # emphatic: far more than the main colour threshold

SIDE_TEAR_SKIN_MIN_AREA = 0.00015   # skin-through-glove patch area / (major-axis length)^2
SIDE_TEAR_SKIN_MAX_AREA = 0.02      # bigger than this is the forearm, not a tear
# A tear's skin patch is a roughly compact opening. The commonest false
# positive is the long thin gap along the glove/arm junction at the cuff,
# which is highly elongated. Measured over the labelled photographs:
#     true tear patches   elongation mean 2.15, all <= 3.5
#     false patches       elongation mean 4.36, 15 of 26 above 3.5
# So this cut removes well over half the false positives at zero cost to
# recall -- the only threshold in the detector with that property.
SIDE_TEAR_SKIN_MAX_ELONG = 3.5
SIDE_TEAR_MIN_COLOR_DIST = 45.0     # Lab distance between the opening and the glove's own colour.


# ============================================================
# Defect 1: enclosed hole
# ============================================================
def detect_holes(img, mask_filled, mask_raw, bg_color):
    """A hole reveals the background, so it sits "inside the glove
    outline, but coloured like the background". The candidate blob's
    average colour must be close to the background colour, or anything
    with an off colour (a stain, a shadow) would get misreported as a hole.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    candidate = cv2.subtract(mask_filled, mask_raw)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Compare against EVERY dominant background colour, not just the most
    # common one. A background can legitimately have more than one mode --
    # two materials in frame (seat + floor tile), or one material split
    # across a strong lighting gradient -- and then a hole reveals
    # whichever mode happens to lie behind it, which need not be the
    # dominant one. Falls back to the single colour it was handed if the
    # multi-mode lookup is unavailable.
    try:
        bg_colors = get_background_colors(img)
    except Exception:
        bg_colors = np.atleast_2d(bg_color)

    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA_HOLE:
            continue
        blob = np.zeros(candidate.shape, np.uint8)
        cv2.drawContours(blob, [c], -1, 255, cv2.FILLED)
        mean_color = lab[blob > 0].mean(axis=0)
        w = np.array([BG_MATCH_L_WEIGHT, 1.0, 1.0], np.float32)
        nearest = min(np.linalg.norm((mean_color - m) * w) for m in bg_colors)
        if nearest < BG_MATCH_DIST:
            results.append(("Tear / Hole", cv2.boundingRect(c)))
    return results


# ============================================================
# Defect 2: open tear (reaches the glove's edge)
# ============================================================
def detect_open_tears(img, mask_filled, mask_raw, bg_color):
    """Convexity defects -- the dents between the contour and its convex
    hull. A normal finger gap is also a deep, wide notch, so depth alone
    can't separate them; shape does: a tear is narrow and sharp (the
    material is cut), a finger gap is a wide, blunt, natural U-shape.

    Known limitation: this distinction is fundamentally heuristic. Bent
    fingers, fingers held together, or a rolled cuff on a real glove will
    change the shape of a finger gap, so the thresholds must be
    recalibrated on real photographs.
    """
    contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    cnt = cv2.approxPolyDP(max(contours, key=cv2.contourArea), CONTOUR_EPSILON, True)
    if len(cnt) < 4:
        return []

    hull = cv2.convexHull(cnt, returnPoints=False)
    hull[::-1].sort(axis=0)
    try:
        defects = cv2.convexityDefects(cnt, hull)
    except cv2.error:
        return []
    if defects is None:
        return []

    _, _, bw, bh = cv2.boundingRect(cnt)
    diag = float(np.hypot(bw, bh))   # normalise by the glove's own size, so this works across resolutions too

    results = []
    for s, e, f, depth_fp in defects.reshape(-1, 4):
        depth = depth_fp / 256.0
        if depth < MIN_TEAR_DEPTH_RATIO * diag:
            continue   # too shallow: jaggedness or the wrist step

        p1, p2, apex = cnt[s][0], cnt[e][0], cnt[f][0]

        mouth = float(np.linalg.norm(p1 - p2))          # condition (1): narrow
        if mouth > MAX_TEAR_MOUTH_RATIO * depth:
            continue

        v1, v2 = p1 - apex, p2 - apex                     # condition (2): sharp
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        if np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) > MAX_TEAR_APEX_ANGLE:
            continue

        results.append(("Open Tear", cv2.boundingRect(np.array([p1, p2, apex]))))
    return results


# ============================================================
# Defect 2b: side tear (a cut breaching the glove's LATERAL edge)
# ============================================================
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


def detect_side_tear(img, mask_filled, mask_raw, bg_color):
    """A cut that breaches the glove's lateral (side) edge.

    Deliberately scoped narrower than `detect_open_tears`, so the two do
    not compete: this one claims only the LATERAL band of the glove,
    leaving fingertip tears to `detect_open_tears`. It is registered
    first, so within that band the more specific detector wins the
    de-duplication.

    Method:
      1. close the glove mask with a disc whose radius is a fixed
         fraction of the glove's length, then subtract the mask. What is
         left are the boundary concavities NARROWER than that disc --
         a scale-bounded version of the convex deficiency D = H - S from
         Ch 10/11, without the convex hull's global reach.
      2. keep components that are big enough, deep enough, do not touch
         the image border (those are framing artefacts, not glove
         features), and whose centroid falls in the lateral band of the
         glove's major axis.

    The band is what separates a tear from a finger gap: finger gaps sit
    distally (measured at fraction 0.05-0.24 on both synthetic and real
    gloves) while a side tear sits mid-glove.
    """
    contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)

    axis, perp, lo, hi = _basic_rectangle_axis(cnt)
    axis_len = hi - lo
    if axis_len < 20:
        return []
    low_is_distal = _fingers_at_low_end(mask_filled, axis, perp, lo, hi, img=img)
    near, far = SIDE_TEAR_BAND

    r = max(int(SIDE_TEAR_CLOSE_RATIO * axis_len), 3)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    notches = cv2.subtract(cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, disc), mask_filled)
    notches = cv2.morphologyEx(notches, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # depth = how far a notch reaches in from the glove outline
    outline = np.zeros(mask_filled.shape, np.uint8)
    cv2.drawContours(outline, [cnt], -1, 255, 2)
    depth_map = cv2.distanceTransform(255 - outline, cv2.DIST_L2, 3)

    # the glove's own colour, sampled away from its edge
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    inner = cv2.erode(mask_raw, np.ones((9, 9), np.uint8)) > 0
    glove_color = np.median(lab[inner], axis=0) if inner.sum() > 200 else None

    h, w = mask_filled.shape
    min_area = SIDE_TEAR_MIN_AREA_RATIO * axis_len * axis_len
    min_depth = SIDE_TEAR_MIN_DEPTH_RATIO * axis_len

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(notches, 8)
    results = []
    for i in range(1, n):
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        cw, ch = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
            continue                       # runs off frame: framing artefact
        component = labels == i
        frac = (float(np.asarray(centroids[i]) @ axis) - lo) / axis_len
        if not low_is_distal:
            frac = 1.0 - frac
        if not near <= frac <= far:
            continue

        area_ratio = stats[i, cv2.CC_STAT_AREA] / (axis_len * axis_len)
        depth_ratio = float(depth_map[component].max()) / axis_len
        seen_dist = 0.0
        if glove_color is not None:
            # notch pixels are outside the mask by construction, so this
            # is the colour seen THROUGH the opening
            seen_dist = float(np.linalg.norm(lab[component].mean(axis=0) - glove_color))

        big_enough = (area_ratio >= SIDE_TEAR_MIN_AREA_RATIO
                      and depth_ratio >= SIDE_TEAR_MIN_DEPTH_RATIO
                      and (glove_color is None or seen_dist >= SIDE_TEAR_MIN_COLOR_DIST))
        cuff_case = (frac >= SIDE_TEAR_CUFF_BAND
                     and area_ratio >= SIDE_TEAR_CUFF_MIN_AREA
                     and depth_ratio >= SIDE_TEAR_CUFF_MIN_DEPTH
                     and glove_color is not None
                     and seen_dist >= SIDE_TEAR_CUFF_MIN_COLOR)
        if not (big_enough or cuff_case):
            continue
        results.append(("Side Tear", (x, y, cw, ch)))

    # ---- second branch: skin showing THROUGH the glove ----------------
    # These gloves are worn, so a breach in the material exposes the hand.
    # That is a far more direct signature than a notch in the silhouette,
    # and it survives the cases where the cut does not open wide enough to
    # change the outline at all. The forearm is excluded by dropping any
    # component that runs off the frame -- an arm always does, a tear
    # never does.
    skin = skin_mask(img)
    hull_mask = np.zeros(mask_filled.shape, np.uint8)
    cv2.drawContours(hull_mask, [cv2.convexHull(cnt)], -1, 255, cv2.FILLED)
    through = cv2.bitwise_and(cv2.bitwise_and(skin, hull_mask),
                              cv2.bitwise_not(mask_raw))
    through = cv2.morphologyEx(through, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    sn, slabels, sstats, scent = cv2.connectedComponentsWithStats(through, 8)
    for i in range(1, sn):
        x, y = int(sstats[i, cv2.CC_STAT_LEFT]), int(sstats[i, cv2.CC_STAT_TOP])
        cw, ch = int(sstats[i, cv2.CC_STAT_WIDTH]), int(sstats[i, cv2.CC_STAT_HEIGHT])
        area = sstats[i, cv2.CC_STAT_AREA] / (axis_len * axis_len)
        if not SIDE_TEAR_SKIN_MIN_AREA <= area <= SIDE_TEAR_SKIN_MAX_AREA:
            continue
        if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
            continue                       # runs off frame: this is the arm
        pts = np.argwhere(slabels == i)[:, ::-1].astype(np.int32)
        side = cv2.minAreaRect(pts)[1]
        if max(side) / (min(side) + 1e-6) > SIDE_TEAR_SKIN_MAX_ELONG:
            continue                       # long thin strip: the cuff/arm junction
        results.append(("Side Tear", (x, y, cw, ch)))

    return results


# ============================================================
# Defect 3: stain
# ============================================================
def detect_stains(img, mask_filled, mask_raw, bg_color):
    """A region that deviates from the glove's normal colour, but isn't
    the background colour. "Not the background colour" excludes holes,
    so the same spot doesn't get reported by two detectors at once.
    """
    glove_color = get_glove_color(img, mask_raw)
    if glove_color is None:
        return []

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    inside = cv2.erode(mask_filled, np.ones((9, 9), np.uint8)) > 0

    dist_glove = np.linalg.norm(lab - glove_color, axis=2)
    dist_bg = np.linalg.norm(lab - bg_color, axis=2)
    stain = ((dist_glove > STAIN_COLOR_DIST) &
             (dist_bg > BG_MATCH_DIST) & inside).astype(np.uint8) * 255
    stain = cv2.morphologyEx(stain, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(stain, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [("Stain", cv2.boundingRect(c)) for c in contours
            if cv2.contourArea(c) >= MIN_AREA_STAIN]


# ============================================================
# Detector registry: add your function's name here once it's ready
# ============================================================
DETECTORS = [
    detect_holes,
    detect_side_tear,     # before detect_open_tears on purpose: it is the more
                          # specific of the two (lateral band only), so within
                          # that band it should win the de-duplication
    detect_open_tears,
    detect_stains,
    # detect_missing_finger,   # e.g. whoever owns "missing finger" adds it here
    # detect_wrinkles,         # e.g. whoever owns "wrinkles" adds it here
]


def _box_iou(a, b):
    """Overlap ratio (IoU) between two bounding boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def deduplicate(defects):
    """The same defect is often reported by more than one detector at once
    (e.g. a large hole can also satisfy a "thin area" test). Keeps the hit
    with the earliest registration order in DETECTORS.
    """
    kept = []
    for name, box in defects:
        if any(_box_iou(box, kept_box) > DEDUP_IOU for _, kept_box in kept):
            continue
        kept.append((name, box))
    return kept


def run_all_detectors(img, mask_filled, mask_raw, bg_color, detectors=None):
    """Run the given detectors, return (defect list, error list).

    Each detector is wrapped in its own try/except: if a detector crashes
    or returns malformed data, only that one is skipped, the rest keep
    running as normal -- one broken detector out of 12 shouldn't mean the
    whole system does nothing when the button is clicked (the worst
    outcome during a demo, worth 10% of the marks).

    detectors: defaults to everything in DETECTORS; the GUI passes a
    filtered subset when the user has unticked one in the "Detectors"
    checklist (handy for skipping a detector that's still being fixed).
    """
    if detectors is None:
        detectors = DETECTORS
    defects, errors = [], []
    for det in detectors:
        try:
            found = det(img, mask_filled, mask_raw, bg_color)
            for name, box in found:
                defects.append((str(name), tuple(int(v) for v in box)))
        except Exception as e:
            errors.append(f"{det.__name__} raised an error: {e}")
    return deduplicate(defects), errors


def draw_results(img, defects):
    """Draw the detection results on the image: red boxes + English labels."""
    out = img.copy()
    for name, (x, y, w, h) in defects:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(out, name, (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out

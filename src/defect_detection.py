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

from segmentation import get_glove_color

# --- Tunable parameters (kept here for sensitivity experiments and for
# citing in the report) ---
BG_MATCH_DIST = 30.0       # hole criterion: candidate's Lab distance from the background must be below this
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

    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA_HOLE:
            continue
        blob = np.zeros(candidate.shape, np.uint8)
        cv2.drawContours(blob, [c], -1, 255, cv2.FILLED)
        mean_color = lab[blob > 0].mean(axis=0)
        if np.linalg.norm(mean_color - bg_color) < BG_MATCH_DIST:
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

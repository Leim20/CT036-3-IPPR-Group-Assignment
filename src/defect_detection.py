# -*- coding: utf-8 -*-
"""
Defect detection module
Each defect = one function. Every team member adds their own 3 defects as
3 functions, registers them in the DETECTORS list, and the GUI calls them
automatically.

===== Conventions for writing a detector (read this if you're on the team) =====
Every function signature follows the same shape:

    def detect_xxx(ctx):
        ...
        return [("Defect Name", (x, y, w, h)), ...]

  - The only input is the ctx dict, which already has the image, masks,
    reference colours etc. ready to use (full list in the docstring of
    segmentation.build_context)
  - Return a list where each item is (defect name, bounding box). Return []
    if nothing was found
  - The defect name is drawn directly on the image and also listed in the
    GUI's text box (a hard requirement of the assignment)

_boxes_from_mask below is a ready-made helper: give it a black-and-white
image and it automatically finds connected blobs, filters out ones that are
too small, and returns a list of bounding boxes.
"""
import cv2
import numpy as np

# --- Tunable parameters (kept together here for sensitivity experiments
# and for citing in the report) ---
# Hole criterion: a candidate blob counts as a hole if its average colour's
# Lab distance from the background colour is below this
BG_MATCH_DIST = 30.0
# Stain criterion: a pixel counts as discoloured if its Lab distance from
# the glove's normal colour is above this
STAIN_COLOR_DIST = 25.0
# Minimum area (pixels, based on the standardised 800px width after
# preprocessing)
MIN_AREA_HOLE = 60
MIN_AREA_STAIN = 60

# --- Criteria parameters for open tears (a cut that reaches the glove's
# boundary) ---
# These three values were derived from measurements, not guessed. On
# synthetic gloves we measured:
#     normal finger gap : mouth/depth = 0.55-0.74,  apex angle = 29-40 deg
#     open tear         : mouth/depth = 0.36,        apex angle = 20 deg
#     wrist step         : mouth/depth = 5.53,        apex angle = 137 deg
# So the two conditions "narrow" and "sharp" are enough to separate a tear
# from a finger gap (see detect_open_tears).
CONTOUR_EPSILON = 2.0          # contour simplification tolerance (px), removes jagged fake notches
MIN_TEAR_DEPTH_RATIO = 0.05    # notch depth / glove bounding-box diagonal, filters out shallow notches
MAX_TEAR_MOUTH_RATIO = 0.45    # notch mouth width / notch depth, a tear is a narrow slit
MAX_TEAR_APEX_ANGLE = 24.0     # notch apex angle (degrees), a tear is sharp, a finger gap is blunt

# De-duplication: if two detectors report overlapping boxes above this
# ratio, keep only the one registered first
DEDUP_IOU = 0.5


def _boxes_from_mask(mask, min_area, open_ksize=3):
    """Helper: black-and-white mask -> list of bounding boxes (automatically
    drops blobs whose area is below the threshold)."""
    if open_ksize:
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((open_ksize, open_ksize), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours
            if cv2.contourArea(c) >= min_area]


# ============================================================
# Defect 1: Enclosed hole / puncture
# ============================================================
def detect_holes(ctx):
    """Principle: a hole reveals the background, so it sits "inside the
    glove outline, but coloured like the background".

    Two-step test (the second step is the key fix, the old version didn't
    have it):
      (1) inside the outline, but NOT in the glove mask -> hole candidate
      (2) the candidate blob's average colour must be CLOSE TO the
          background colour for it to count as a hole
          Without step (2), anything with an off colour (stains, printed
          logos, shadows) gets reported as a hole, causing a wrong defect
          type -- this was the exact bug measured before the fix.
    """
    lab, bg = ctx["lab"], ctx["bg_lab"]
    candidate = cv2.subtract(ctx["mask_filled"], ctx["mask_raw"])
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN,
                                 np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA_HOLE:
            continue
        blob = np.zeros(candidate.shape, np.uint8)
        cv2.drawContours(blob, [c], -1, 255, cv2.FILLED)
        mean_color = lab[blob > 0].mean(axis=0)
        if np.linalg.norm(mean_color - bg) < BG_MATCH_DIST:   # <- step (2)
            results.append(("Tear / Hole", cv2.boundingRect(c)))
    return results


# ============================================================
# Defect 2: Open tear (a cut that reaches the glove's edge)
# ============================================================
def detect_open_tears(ctx):
    """Principle: convexity defects -- the dents between the glove's
    contour and its convex hull.

    Why the mask-subtraction trick from detect_holes doesn't work here:
      mask_filled is "the contour polygon, filled in", and the contour
      itself follows a notch inward, so a tear that reaches the boundary
      is never excluded from mask_filled in the first place -- the
      subtraction is always 0 (measured: 0 candidate pixels). This is a
      criterion problem, not something parameter-tuning can fix; a
      different method is required.

    The hard part: a normal FINGER GAP is also a deep, large boundary
    notch, so "depth" alone can't separate them. The measured difference is
    in shape (data at the top of this file):
      (1) Narrow: a tear's mouth width is small relative to its depth (the
          material is cut, the two sides sit almost against each other)
      (2) Sharp: a tear's apex is a sharp V, a finger gap is a rounded U
    Neither condition looks at position, so both fingertip tears and palm
    tears are detected.

    Known limitation (must be written up in the report's critical
    analysis): the distinction between a finger gap and a tear is
    fundamentally heuristic, with no absolute boundary. Bent fingers,
    fingers held together, or a rolled cuff on a real glove will all change
    the shape of the finger gaps. The three thresholds above must be
    recalibrated on real photographs; loosen them and finger gaps get
    misreported as tears, tighten them and shallow fingertip tears get
    missed.
    """
    contours, _ = cv2.findContours(ctx["mask_filled"], cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Simplify the contour first: the raw contour has a lot of single-pixel
    # jaggedness that produces dozens of fake notches
    cnt = cv2.approxPolyDP(max(contours, key=cv2.contourArea),
                           CONTOUR_EPSILON, True)
    if len(cnt) < 4:
        return []

    hull = cv2.convexHull(cnt, returnPoints=False)
    hull[::-1].sort(axis=0)        # convexityDefects requires monotonic hull indices
    try:
        defects = cv2.convexityDefects(cnt, hull)
    except cv2.error:              # OpenCV raises when the contour is degenerate
        return []
    if defects is None:
        return []

    _, _, bw, bh = cv2.boundingRect(cnt)
    diag = float(np.hypot(bw, bh))     # normalise by the glove's own size, so this also works across resolutions

    results = []
    for s, e, f, depth_fp in defects.reshape(-1, 4):   # OpenCV 5 returns (N,4)
        depth = depth_fp / 256.0                       # per the API: depth is a fixed-point value
        if depth < MIN_TEAR_DEPTH_RATIO * diag:
            continue                                   # too shallow: jaggedness or the wrist step

        p_start, p_end, p_apex = cnt[s][0], cnt[e][0], cnt[f][0]

        # Condition (1): narrow -- mouth width must be small relative to depth
        mouth = float(np.linalg.norm(p_start - p_end))
        if mouth > MAX_TEAR_MOUTH_RATIO * depth:
            continue

        # Condition (2): sharp -- the angle between the two edges at the apex must be small
        v1, v2 = p_start - p_apex, p_end - p_apex
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        if np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) > MAX_TEAR_APEX_ANGLE:
            continue

        box = cv2.boundingRect(np.array([p_start, p_end, p_apex]))
        results.append(("Open Tear", box))
    return results


# ============================================================
# Defect 3: Stain / dirt spot
# ============================================================
def detect_stains(ctx):
    """Principle: compute the glove's "normal colour" (median) in Lab
    space; a small blob that deviates too far from the normal colour but is
    also NOT the background colour is a stain.

    All three conditions must hold at once for a pixel to count as a stain:
      (1) colour deviates from the glove's normal colour (dist > STAIN_COLOR_DIST)
      (2) colour is not the background colour (excludes holes, avoids the
          same blob being reported twice with the wrong type)
      (3) located inside the glove (mask_filled eroded inward, to avoid the
          mixed-colour pixels along the edge)

    Note that step (3) uses mask_filled, not mask_raw:
    a visually obvious stain may never have made it into mask_raw in the
    first place, so using mask_raw means the more obvious a stain is, the
    less likely it gets found -- that was exactly the bug before the fix.
    """
    lab, ref, bg = ctx["lab"], ctx["glove_lab"], ctx["bg_lab"]

    inside = cv2.erode(ctx["mask_filled"], np.ones((9, 9), np.uint8)) > 0
    if not inside.any():
        return []

    dist_glove = np.linalg.norm(lab - ref, axis=2)   # distance from the glove's normal colour
    dist_bg = np.linalg.norm(lab - bg, axis=2)       # distance from the background colour

    stain = ((dist_glove > STAIN_COLOR_DIST) &
             (dist_bg > BG_MATCH_DIST) &
             inside).astype(np.uint8) * 255
    return [("Stain", box) for box in
            _boxes_from_mask(stain, MIN_AREA_STAIN)]


# ============================================================
# Detector registry: once you've written a new function, add its name here
# ============================================================
DETECTORS = [
    detect_holes,
    detect_open_tears,
    detect_stains,
    # detect_missing_finger,   <- e.g. whoever owns "missing finger" adds it here
    # detect_wrinkles,         <- e.g. whoever owns "wrinkles" adds it here
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


def deduplicate(defects, iou_threshold=DEDUP_IOU):
    """The same physical defect is often reported by more than one detector
    at once (e.g. a large hole can also satisfy a "thin area" test). This
    keeps the higher-priority hit -- priority follows the order defects are
    registered in DETECTORS -- so the same defect isn't double-counted and
    dragging precision down.

    Team members don't need to worry about this function when adding a new
    detector; the framework calls it automatically.
    """
    kept = []
    for name, box in defects:
        if any(_box_iou(box, kept_box) > iou_threshold for _, kept_box in kept):
            continue
        kept.append((name, box))
    return kept


def run_all_detectors(ctx):
    """Run every registered detector in turn, then merge and de-duplicate.

    If the segmentation stage decided "there is no glove in this image",
    return an empty list right away, and let the GUI say "no glove
    detected" -- it must never be reported as "inspection passed".

    Each detector is run in isolation: if a team member's detector crashes
    (raises an exception) or returns malformed data, only that one is
    skipped -- the rest keep running normally, and the error is recorded in
    ctx["errors"] for the GUI to display.
      Without this, one broken detector out of 12 would mean the whole
      system does nothing when the button is clicked -- the worst possible
      outcome during the demo (worth 10% of the marks).
    """
    if not ctx.get("ok", True):
        return []

    errors = ctx.setdefault("errors", [])
    all_defects = []

    for detector in DETECTORS:
        name = getattr(detector, "__name__", str(detector))
        try:
            found = detector(ctx)
        except Exception as exc:                      # the detector itself crashed
            errors.append(f"{name} raised an error: {type(exc).__name__}: {exc}")
            continue
        try:                                          # validate the return format while we're at it
            for label, (x, y, w, h) in found:
                all_defects.append((str(label), (int(x), int(y), int(w), int(h))))
        except (TypeError, ValueError) as exc:
            errors.append(f"{name} returned malformed output "
                          f"(expected [(name, (x,y,w,h)), ...]): {exc}")

    return deduplicate(all_defects)


def draw_results(img, defects):
    """Draw the detection results on the image: red boxes + English labels."""
    out = img.copy()
    for name, (x, y, w, h) in defects:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(out, name, (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out

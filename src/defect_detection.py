# -*- coding: utf-8 -*-
"""
Defect detection: one function per defect, registered in the DETECTORS
list, called automatically by the GUI.

Every detector shares the same signature:

    def detect_xxx(img, mask_filled, mask_raw, bg_color,
                   img_plain=None, material=None):
        ...
        return [("Defect Name", (x, y, w, h)), ...]   # return [] if nothing was found

The last two parameters are optional so older four-argument calls remain
valid. ``img`` is lighting-normalised; ``img_plain`` retains the original
colour/texture after resize and denoising. The defect name is drawn on the
image and also listed in the GUI's text box.
"""
import cv2
import numpy as np

from segmentation import get_glove_color

# --- Tunable parameters (kept here for sensitivity experiments and for
# citing in the report) ---
BG_MATCH_DIST = 30.0       # used to keep stains separate from background-coloured regions
STAIN_COLOR_DIST = 25.0    # stain criterion: pixel's Lab distance from the glove's normal colour must be above this
MIN_AREA_STAIN = 60

# Hole detector parameters. The photographed gloves are worn on a hand, so a
# puncture reveals skin rather than the green background. Candidate skin pixels
# must form an enclosed, locally high-contrast region well inside the glove.
HOLE_BOUNDARY_KSIZE = 13
HOLE_RING_KSIZE = 11
HOLE_DEFAULT_RULE = {
    "min_area": 60,
    "min_local_contrast": 30.0,
    "min_interior_ratio": 0.20,
}
HOLE_MATERIAL_RULES = {
    # Cotton weave can expose many tiny skin-coloured gaps, so genuine damage
    # must be larger and have a sharper boundary.
    "cotton": {
        "min_area": 240,
        "min_local_contrast": 50.0,
        "min_interior_ratio": 0.20,
    },
    # Latex foam is opaque and uniform; even a small enclosed skin region is
    # strong evidence, and the dataset produced no non-hole candidates.
    "latex_foam": {
        "min_area": 60,
        "min_local_contrast": 0.0,
        "min_interior_ratio": 0.20,
    },
    # Thin nitrile shows skin through the material, so require a larger region
    # before classifying it as an actual opening.
    "nitrile": {
        "min_area": 400,
        "min_local_contrast": 20.0,
        "min_interior_ratio": 0.20,
    },
}

# Broad, lighting-tolerant skin ranges in YCrCb and HSV. Requiring both rules
# avoids accepting blue glove highlights that happen to satisfy only one space.
SKIN_Y_MIN = 30
SKIN_CR_MIN, SKIN_CR_MAX = 125, 185
SKIN_CB_MIN, SKIN_CB_MAX = 65, 140
SKIN_H_MAX, SKIN_H_WRAP_MIN = 25, 170
SKIN_S_MIN, SKIN_V_MIN = 25, 45

# Finger-not-enough detector parameters. Each rule is dimensionless, except
# the expected feature counts, so resizing an image does not change the
# decision. Cotton needs stricter evidence because its open weave and flexible
# fingers create more skin-coloured regions and silhouette variation.
FINGER_NOT_ENOUGH_DEFAULT_RULE = {
    "indent_min_area_ratio": 0.004,
    "indent_max_y_ratio": 0.55,
    "indent_target_count": 3,
    "row_min_width_ratio": 0.06,
    "row_max_y_ratio": 0.65,
    "row_support_ratio": 0.08,
    "row_target_count": 2,
    "skin_min_area_ratio": 0.012,
    "skin_min_boundary_ratio": 0.05,
    "skin_max_y_ratio": 0.72,
}
FINGER_NOT_ENOUGH_MATERIAL_RULES = {
    "cotton": {
        "indent_min_area_ratio": 0.010,
        "indent_max_y_ratio": 0.75,
        "indent_target_count": 4,
        "row_min_width_ratio": 0.06,
        "row_max_y_ratio": 0.55,
        "row_support_ratio": 0.08,
        "row_target_count": 2,
        "skin_min_area_ratio": 0.024,
        "skin_min_boundary_ratio": 0.05,
        "skin_max_y_ratio": 0.72,
    },
    "latex_foam": {
        "indent_min_area_ratio": 0.001,
        "indent_max_y_ratio": 0.55,
        "indent_target_count": 3,
        "row_min_width_ratio": 0.04,
        "row_max_y_ratio": 0.75,
        "row_support_ratio": 0.06,
        "row_target_count": 2,
        "skin_min_area_ratio": 0.004,
        "skin_min_boundary_ratio": 0.10,
        "skin_max_y_ratio": 0.72,
    },
    "nitrile": {
        "indent_min_area_ratio": 0.001,
        "indent_max_y_ratio": 0.55,
        "indent_target_count": 2,
        "row_min_width_ratio": 0.04,
        "row_max_y_ratio": 0.55,
        "row_support_ratio": 0.08,
        "row_target_count": 3,
        "skin_min_area_ratio": 0.004,
        "skin_min_boundary_ratio": 0.05,
        "skin_max_y_ratio": 0.72,
    },
}
FINGER_REGION_HEIGHT_RATIO = 0.80
FINGER_SKIN_BOUNDARY_KSIZE = 13

# Thin / overstretched detector. These thresholds were measured on the
# development photographs after isolating glove-coloured pixels from the
# controlled green inspection mat. Cotton and nitrile use transparency cues;
# latex foam has no reliable transparency cue, so its lower-confidence branch
# uses only a coarse edge-density measurement and is documented as experimental.
THIN_BLUE_H_MIN, THIN_BLUE_H_MAX = 85, 145
THIN_BLUE_S_MIN, THIN_BLUE_V_MIN = 40, 35
THIN_WHITE_S_MAX, THIN_WHITE_V_MIN = 35, 95
THIN_ROI_CLOSE_KSIZE = 9
THIN_ROI_ERODE_KSIZE = 11
THIN_MIN_ROI_AREA_RATIO = 0.01

THIN_COTTON_BLUE_S25_MAX = 120.0
THIN_COTTON_WHITE_SKIN_MIN_RATIO = 0.0008
THIN_COTTON_WHITE_SKIN_MIN_COMPONENTS = 3
THIN_COTTON_WHITE_SKIN_MAX_LARGEST_SHARE = 0.15
THIN_COTTON_WHITE_GRID_MIN_COVERAGE = 0.034

THIN_NITRILE_SKIN_MIN_RATIO = 0.0007
THIN_NITRILE_SKIN_MAX_RATIO = 0.002
THIN_NITRILE_SKIN_MIN_COMPONENTS = 4
THIN_NITRILE_LIGHT_P25_MIN = 120.0
THIN_NITRILE_LIGHT_P25_SHADOW_MAX = 135.0
THIN_NITRILE_S_MEDIAN_MAX = 130.0
THIN_NITRILE_SHADOW_S_MEDIAN_MAX = 133.0

THIN_LATEX_LIGHT_P25_MIN = 90.0
THIN_LATEX_LAPLACIAN_MEAN_MAX = 18.0
THIN_LATEX_LOW_S_MEDIAN_MAX = 140.0
THIN_LATEX_BRIGHT_LIGHT_P25_MIN = 112.0
THIN_LATEX_BRIGHT_S_MEDIAN_MAX = 160.0

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

# Result visualisation. Detectors keep the assignment's required
# ``(label, bounding_box)`` return contract; after recognition, the shared
# pipeline rebuilds the accepted pixel evidence as a binary segmentation mask.
# The semi-transparent colours make the original glove texture remain visible.
DEFECT_OVERLAY_ALPHA = 0.45
DEFECT_OVERLAY_COLORS = {
    "Hole": (0, 0, 255),                   # red (BGR)
    "Open Tear": (0, 80, 255),             # orange-red
    "Finger Not Enough": (0, 165, 255),     # orange
    "Thin / Overstretched": (255, 0, 255),  # magenta
    "Stain": (255, 80, 0),                 # blue
}
THIN_SEGMENT_DENSITY_KSIZE = 41
THIN_SEGMENT_DENSITY_MIN = 2


# ============================================================
# Defect 1: enclosed hole
# ============================================================
def detect_holes(img, mask_filled, mask_raw, bg_color,
                 img_plain=None, material=None):
    """Detect enclosed punctures that expose the wearer's skin.

    The previous implementation searched for green-background pixels inside
    the glove. That assumption did not match the real dataset: the gloves are
    worn, so their holes reveal skin. This detector therefore uses two
    classical skin-colour rules, then rejects candidates that touch the glove's
    outside boundary, lack a sharp colour transition, or sit too close to the
    silhouette edge. Those checks prevent exposed fingertips from being
    labelled as palm holes.

    When no material metadata is supplied (as in the synthetic regression
    tests), a strict background-revealing fallback is also run. Real dataset
    calls always provide their material and use the calibrated skin rule.
    """
    source = img_plain if img_plain is not None else img
    rule = HOLE_MATERIAL_RULES.get(material, HOLE_DEFAULT_RULE)

    ycrcb = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    skin_ycrcb = (
        (y > SKIN_Y_MIN)
        & (cr >= SKIN_CR_MIN) & (cr <= SKIN_CR_MAX)
        & (cb >= SKIN_CB_MIN) & (cb <= SKIN_CB_MAX)
    )

    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    skin_hsv = (
        ((h <= SKIN_H_MAX) | (h >= SKIN_H_WRAP_MIN))
        & (s >= SKIN_S_MIN)
        & (v >= SKIN_V_MIN)
    )

    candidate = (skin_ycrcb & skin_hsv & (mask_filled > 0)).astype(np.uint8)
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    if count <= 1:
        return []

    # Any colour anomaly touching this inner boundary band is an exposed outer
    # edge/fingertip, not an enclosed hole.
    eroded = cv2.erode(
        mask_filled,
        np.ones((HOLE_BOUNDARY_KSIZE, HOLE_BOUNDARY_KSIZE), np.uint8),
    )
    boundary_band = (mask_filled > 0) & (eroded == 0)
    touching_boundary = set(np.unique(labels[boundary_band]))

    lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    interior_distance = cv2.distanceTransform(
        (mask_filled > 0).astype(np.uint8), cv2.DIST_L2, 5
    )
    max_interior_distance = max(float(interior_distance.max()), 1.0)

    results = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < rule["min_area"] or label in touching_boundary:
            continue

        component = (labels == label).astype(np.uint8)
        ring = cv2.dilate(
            component,
            np.ones((HOLE_RING_KSIZE, HOLE_RING_KSIZE), np.uint8),
        ) - component
        ring_pixels = (ring > 0) & (mask_filled > 0)
        if not ring_pixels.any():
            continue

        local_contrast = float(
            np.linalg.norm(
                lab[component > 0].mean(axis=0) - lab[ring_pixels].mean(axis=0)
            )
        )
        if local_contrast < rule["min_local_contrast"]:
            continue

        interior_ratio = float(
            interior_distance[component > 0].max() / max_interior_distance
        )
        if interior_ratio < rule["min_interior_ratio"]:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y0 = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        results.append(("Hole", (x, y0, width, height)))
    if material is None:
        # Generic fallback for an unworn glove: an enclosed hole reveals the
        # photographed background. Keep this separate from the real-data rule
        # so porous/translucent materials do not acquire its false positives.
        background_candidate = cv2.subtract(mask_filled, mask_raw)
        background_candidate = cv2.morphologyEx(
            background_candidate,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )
        contours, _ = cv2.findContours(
            background_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        lab_normalized = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        for contour in contours:
            if cv2.contourArea(contour) < HOLE_DEFAULT_RULE["min_area"]:
                continue
            blob = np.zeros(background_candidate.shape, np.uint8)
            cv2.drawContours(blob, [contour], -1, 255, cv2.FILLED)
            mean_color = lab_normalized[blob > 0].mean(axis=0)
            if np.linalg.norm(mean_color - bg_color) < BG_MATCH_DIST:
                results.append(("Hole", cv2.boundingRect(contour)))

    return results


# ============================================================
# Defect 2: open tear (reaches the glove's edge)
# ============================================================
def detect_open_tears(img, mask_filled, mask_raw, bg_color,
                      img_plain=None, material=None):
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
# Defect 3: finger not enough
# ============================================================
def detect_finger_not_enough(img, mask_filled, mask_raw, bg_color,
                             img_plain=None, material=None):
    """Detect a shortened, hidden, or absent glove finger.

    Three independent, explainable measurements are used because the real
    photographs contain two versions of this defect: some gloves expose a
    bare finger, while others have one glove finger folded out of view.

    1. A sufficiently large skin-coloured component inside the upper glove
       silhouette indicates an exposed finger.
    2. Convex-hull indentation count describes the missing space between the
       remaining finger shapes.
    3. The number of persistent foreground runs across upper rows describes
       how many separate finger columns are visible.

    A curled or folded finger that leaves a visible empty space is deliberately
    accepted as this defect, even when the physical finger is still attached.
    The per-material rules account for the different stiffness and texture of
    cotton, latex foam, and nitrile gloves.
    """
    contours, _ = cv2.findContours(
        mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []

    glove_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(glove_contour) <= 0:
        return []

    x, y0, width, height = cv2.boundingRect(glove_contour)
    if width <= 0 or height <= 0:
        return []

    material_key = str(material).lower() if material is not None else None
    rule = FINGER_NOT_ENOUGH_MATERIAL_RULES.get(
        material_key, FINGER_NOT_ENOUGH_DEFAULT_RULE
    )

    # Count sizeable gaps between the glove and its convex hull. Only gaps in
    # the upper part of the glove are relevant; cuff/wrist gaps are ignored.
    hull_points = cv2.convexHull(glove_contour)
    hull_mask = np.zeros_like(mask_filled, dtype=np.uint8)
    cv2.drawContours(hull_mask, [hull_points], -1, 255, cv2.FILLED)
    hull_area = max(int(cv2.countNonZero(hull_mask)), 1)

    indentation_mask = cv2.subtract(hull_mask, mask_filled)
    indent_count, _, indent_stats, indent_centroids = (
        cv2.connectedComponentsWithStats(
            (indentation_mask > 0).astype(np.uint8), 8
        )
    )
    qualifying_indentations = 0
    largest_indentation_box = None
    largest_indentation_area = -1
    for label in range(1, indent_count):
        component_area_ratio = (
            float(indent_stats[label, cv2.CC_STAT_AREA]) / hull_area
        )
        centroid_y_ratio = (
            float(indent_centroids[label, 1]) - y0
        ) / max(height, 1)
        if (
            component_area_ratio >= rule["indent_min_area_ratio"]
            and centroid_y_ratio <= rule["indent_max_y_ratio"]
        ):
            qualifying_indentations += 1
            component_area = int(indent_stats[label, cv2.CC_STAT_AREA])
            if component_area > largest_indentation_area:
                largest_indentation_area = component_area
                largest_indentation_box = (
                    int(indent_stats[label, cv2.CC_STAT_LEFT]),
                    int(indent_stats[label, cv2.CC_STAT_TOP]),
                    int(indent_stats[label, cv2.CC_STAT_WIDTH]),
                    int(indent_stats[label, cv2.CC_STAT_HEIGHT]),
                )

    # Count vertical finger columns that persist through several upper rows.
    glove_crop = mask_filled[y0:y0 + height, x:x + width] > 0
    inspected_height = max(
        1, min(
            height,
            int(round(height * rule["row_max_y_ratio"])),
        )
    )
    minimum_run_width = max(
        1, int(round(width * rule["row_min_width_ratio"]))
    )
    run_histogram = np.zeros(7, dtype=np.int32)
    for row in glove_crop[:inspected_height]:
        padded = np.pad(row.astype(np.int8), (1, 1))
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        visible_runs = int(np.count_nonzero(
            (ends - starts) >= minimum_run_width
        ))
        if 1 <= visible_runs <= 6:
            run_histogram[visible_runs] += 1

    minimum_support_rows = max(
        2, int(round(height * rule["row_support_ratio"]))
    )
    persistent_run_count = 0
    for run_count in range(6, 0, -1):
        # A row with more visible columns also supports every lower count.
        # This keeps the measurement stable when a narrow extra sliver appears
        # briefly near a fingertip.
        if run_histogram[run_count:].sum() >= minimum_support_rows:
            persistent_run_count = run_count
            break

    # Use the original (non-lighting-normalised) colour for skin evidence.
    source = img_plain if img_plain is not None else img
    ycrcb = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    skin_ycrcb = (
        (y_channel > SKIN_Y_MIN)
        & (cr_channel >= SKIN_CR_MIN) & (cr_channel <= SKIN_CR_MAX)
        & (cb_channel >= SKIN_CB_MIN) & (cb_channel <= SKIN_CB_MAX)
    )
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    skin_hsv = (
        ((hue <= SKIN_H_MAX) | (hue >= SKIN_H_WRAP_MIN))
        & (saturation >= SKIN_S_MIN)
        & (value >= SKIN_V_MIN)
    )

    upper_region = np.zeros(mask_filled.shape, dtype=bool)
    finger_region_height = max(
        1, int(round(height * FINGER_REGION_HEIGHT_RATIO))
    )
    upper_region[y0:y0 + finger_region_height, x:x + width] = True
    skin_candidate = (
        skin_ycrcb & skin_hsv & (mask_filled > 0) & upper_region
    ).astype(np.uint8)
    skin_candidate = cv2.morphologyEx(
        skin_candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    skin_candidate = cv2.morphologyEx(
        skin_candidate, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
    )
    skin_component_count, skin_labels, skin_stats, skin_centroids = (
        cv2.connectedComponentsWithStats(skin_candidate, 8)
    )
    # A missing or shortened finger exposes skin at the glove's outer edge.
    # Requiring boundary contact prevents an enclosed palm hole or internal
    # translucent patch from becoming Finger Not Enough merely because it is
    # skin-coloured.
    eroded_glove = cv2.erode(
        mask_filled,
        np.ones(
            (FINGER_SKIN_BOUNDARY_KSIZE, FINGER_SKIN_BOUNDARY_KSIZE),
            np.uint8,
        ),
    )
    glove_boundary = (mask_filled > 0) & (eroded_glove == 0)
    exposed_skin = False
    exposed_skin_box = None
    for label in range(1, skin_component_count):
        component_area = int(skin_stats[label, cv2.CC_STAT_AREA])
        component_area_ratio = float(component_area) / hull_area
        if component_area_ratio < rule["skin_min_area_ratio"]:
            continue

        component = skin_labels == label
        boundary_ratio = (
            float(np.count_nonzero(component & glove_boundary))
            / max(component_area, 1)
        )
        component_y_ratio = (
            float(skin_centroids[label, 1]) - y0
        ) / max(height, 1)
        if (
            boundary_ratio >= rule["skin_min_boundary_ratio"]
            and component_y_ratio <= rule["skin_max_y_ratio"]
        ):
            exposed_skin = True
            exposed_skin_box = (
                int(skin_stats[label, cv2.CC_STAT_LEFT]),
                int(skin_stats[label, cv2.CC_STAT_TOP]),
                int(skin_stats[label, cv2.CC_STAT_WIDTH]),
                int(skin_stats[label, cv2.CC_STAT_HEIGHT]),
            )
            break
    missing_space = (
        qualifying_indentations == rule["indent_target_count"]
    )
    missing_column = persistent_run_count == rule["row_target_count"]
    if not (exposed_skin or missing_space or missing_column):
        return []

    # Localise the evidence for the display stage. Exposed skin is already a
    # real pixel region; otherwise the largest abnormal hull gap is the best
    # estimate of where a completely absent finger should have been. The full
    # upper zone remains a safe last resort for a row-count-only recognition.
    result_box = exposed_skin_box or largest_indentation_box
    if result_box is None:
        result_box = (
            x,
            y0,
            width,
            min(finger_region_height, mask_filled.shape[0] - y0),
        )
    return [("Finger Not Enough", result_box)]


# ============================================================
# Defect 4: thin / overstretched
# ============================================================
def _thin_material_region(source, material):
    """Return a glove-colour ROI and whether cotton is blue or white.

    Some photographs show a green inspection card surrounded by a black table.
    A border-only background estimate can then select part of the card instead
    of the glove. This small material-colour refinement stays inside the
    detector and uses only HSV thresholding, morphology and the largest contour.
    It does not alter the shared segmentation used by the other detectors.
    """
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    blue = (
        (hue >= THIN_BLUE_H_MIN) & (hue <= THIN_BLUE_H_MAX)
        & (saturation >= THIN_BLUE_S_MIN)
        & (value >= THIN_BLUE_V_MIN)
    )
    white = (
        (saturation <= THIN_WHITE_S_MAX)
        & (value >= THIN_WHITE_V_MIN)
    )

    if material == "cotton":
        if np.count_nonzero(blue) >= np.count_nonzero(white):
            selected, subtype = blue, "blue"
        else:
            selected, subtype = white, "white"
    elif material in {"nitrile", "latex_foam"}:
        selected, subtype = blue, "blue"
    else:
        return None, None

    raw = selected.astype(np.uint8) * 255
    connected = cv2.morphologyEx(
        raw,
        cv2.MORPH_CLOSE,
        np.ones((THIN_ROI_CLOSE_KSIZE, THIN_ROI_CLOSE_KSIZE), np.uint8),
    )
    connected = cv2.morphologyEx(
        connected, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    contours, _ = cv2.findContours(
        connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, subtype

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < source.shape[0] * source.shape[1] * THIN_MIN_ROI_AREA_RATIO:
        return None, subtype

    filled = np.zeros(source.shape[:2], dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
    return filled, subtype


def _thin_grid_coverage(candidate, region, minimum_fraction=0.01):
    """Fraction of valid 10 x 10 glove blocks containing candidate pixels."""
    contours, _ = cv2.findContours(
        region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0
    x, y, width, height = cv2.boundingRect(
        max(contours, key=cv2.contourArea)
    )
    valid_blocks = 0
    occupied_blocks = 0
    region_bool = region > 0
    candidate_bool = candidate > 0
    for row in range(10):
        y1 = y + round(row * height / 10)
        y2 = y + round((row + 1) * height / 10)
        for column in range(10):
            x1 = x + round(column * width / 10)
            x2 = x + round((column + 1) * width / 10)
            block_region = region_bool[y1:y2, x1:x2]
            region_pixels = int(np.count_nonzero(block_region))
            if region_pixels < max(10, int(0.20 * block_region.size)):
                continue
            valid_blocks += 1
            block_candidate = candidate_bool[y1:y2, x1:x2]
            if (
                np.count_nonzero(block_candidate & block_region)
                / region_pixels
                >= minimum_fraction
            ):
                occupied_blocks += 1
    return float(occupied_blocks) / max(valid_blocks, 1)


def detect_thin_area(img, mask_filled, mask_raw, bg_color,
                     img_plain=None, material=None):
    """Detect diffuse material thinning or overstretching.

    The defect is not a change in the glove outline, so silhouette or template
    comparison would be unsuitable. Instead, this detector measures physical
    effects of stretched material inside a glove-colour region:

    * blue cotton loses saturation as skin shows through the opened weave;
    * white cotton exposes many dispersed skin-coloured weave openings;
    * nitrile either leaks dispersed skin colour or becomes broadly pale and
      low-saturation while stretched tightly over the hand;
    * latex foam is normally opaque, so an experimental branch measures an
      unusually smooth coating. It is intentionally lower priority than the
      cotton and nitrile rules.

    All decisions use named HSV/YCrCb/Lightness/edge-density statistics and
    fixed development thresholds. One compact skin patch is rejected for the
    cotton and nitrile transparency rules so a puncture is not relabelled as
    diffuse thinning.
    """
    source = img_plain if img_plain is not None else img
    material_key = str(material).lower() if material is not None else None
    region, cotton_subtype = _thin_material_region(source, material_key)
    if region is None:
        return []

    interior = cv2.erode(
        region,
        np.ones((THIN_ROI_ERODE_KSIZE, THIN_ROI_ERODE_KSIZE), np.uint8),
    ) > 0
    interior_area = int(np.count_nonzero(interior))
    if interior_area == 0:
        return []

    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]

    ycrcb = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    skin = (
        (y_channel > SKIN_Y_MIN)
        & (cr_channel >= SKIN_CR_MIN) & (cr_channel <= SKIN_CR_MAX)
        & (cb_channel >= SKIN_CB_MIN) & (cb_channel <= SKIN_CB_MAX)
        & ((hue <= SKIN_H_MAX) | (hue >= SKIN_H_WRAP_MIN))
        & (saturation >= SKIN_S_MIN)
        & (value >= SKIN_V_MIN)
        & interior
    ).astype(np.uint8)
    skin = cv2.morphologyEx(
        skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(
        skin, 8
    )
    sizeable_skin_components = sum(
        int(component_stats[label, cv2.CC_STAT_AREA]) >= 10
        for label in range(1, component_count)
    )
    skin_area = int(np.count_nonzero(skin))
    skin_ratio = float(skin_area) / interior_area
    largest_skin_area = max(
        (
            int(component_stats[label, cv2.CC_STAT_AREA])
            for label in range(1, component_count)
        ),
        default=0,
    )
    largest_skin_share = float(largest_skin_area) / max(skin_area, 1)

    is_thin = False
    if material_key == "cotton" and cotton_subtype == "blue":
        saturation_p25 = float(np.percentile(saturation[interior], 25))
        is_thin = saturation_p25 < THIN_COTTON_BLUE_S25_MAX
    elif material_key == "cotton":
        grid_coverage = _thin_grid_coverage(skin, region)
        is_thin = (
            skin_ratio >= THIN_COTTON_WHITE_SKIN_MIN_RATIO
            and sizeable_skin_components
            >= THIN_COTTON_WHITE_SKIN_MIN_COMPONENTS
            and largest_skin_share
            <= THIN_COTTON_WHITE_SKIN_MAX_LARGEST_SHARE
            and grid_coverage >= THIN_COTTON_WHITE_GRID_MIN_COVERAGE
        )
    elif material_key == "nitrile":
        dispersed_skin = (
            skin_ratio >= THIN_NITRILE_SKIN_MIN_RATIO
            and skin_ratio < THIN_NITRILE_SKIN_MAX_RATIO
            and sizeable_skin_components >= THIN_NITRILE_SKIN_MIN_COMPONENTS
        )
        lightness_p25 = float(np.percentile(lightness[interior], 25))
        saturation_median = float(np.median(saturation[interior]))
        broadly_pale = (
            lightness_p25 > THIN_NITRILE_LIGHT_P25_MIN
            and saturation_median < THIN_NITRILE_S_MEDIAN_MAX
            and skin_ratio < THIN_NITRILE_SKIN_MAX_RATIO
        )
        shadow_pale = (
            THIN_NITRILE_LIGHT_P25_MIN < lightness_p25
            <= THIN_NITRILE_LIGHT_P25_SHADOW_MAX
            and saturation_median < THIN_NITRILE_SHADOW_S_MEDIAN_MAX
            and skin_ratio < THIN_NITRILE_SKIN_MAX_RATIO
        )
        is_thin = dispersed_skin or broadly_pale or shadow_pale
    elif material_key == "latex_foam":
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        laplacian_mean = float(np.mean(laplacian[interior]))
        lightness_p25 = float(np.percentile(lightness[interior], 25))
        saturation_median = float(np.median(saturation[interior]))
        is_thin = (
            lightness_p25 > THIN_LATEX_LIGHT_P25_MIN
            and laplacian_mean < THIN_LATEX_LAPLACIAN_MEAN_MAX
            and (
                saturation_median < THIN_LATEX_LOW_S_MEDIAN_MAX
                or (
                    lightness_p25 > THIN_LATEX_BRIGHT_LIGHT_P25_MIN
                    and saturation_median
                    < THIN_LATEX_BRIGHT_S_MEDIAN_MAX
                )
            )
        )

    if not is_thin:
        return []

    contours, _ = cv2.findContours(
        region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    return [("Thin / Overstretched", cv2.boundingRect(
        max(contours, key=cv2.contourArea)
    ))]


# ============================================================
# Defect 5: stain
# ============================================================
def detect_stains(img, mask_filled, mask_raw, bg_color,
                  img_plain=None, material=None):
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
    detect_finger_not_enough,
    detect_thin_area,
    detect_stains,
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


def run_all_detectors(img, mask_filled, mask_raw, bg_color, detectors=None,
                      img_plain=None, material=None):
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
            found = det(
                img,
                mask_filled,
                mask_raw,
                bg_color,
                img_plain=img_plain,
                material=material,
            )
            for name, box in found:
                defects.append((str(name), tuple(int(v) for v in box)))
        except Exception as e:
            errors.append(f"{det.__name__} raised an error: {e}")
    return deduplicate(defects), errors


def _skin_colour_mask(source, support_mask=None):
    """Segment skin-coloured pixels with the detector's two colour rules."""
    ycrcb = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    skin = (
        (y_channel > SKIN_Y_MIN)
        & (cr_channel >= SKIN_CR_MIN) & (cr_channel <= SKIN_CR_MAX)
        & (cb_channel >= SKIN_CB_MIN) & (cb_channel <= SKIN_CB_MAX)
        & ((hue <= SKIN_H_MAX) | (hue >= SKIN_H_WRAP_MIN))
        & (saturation >= SKIN_S_MIN)
        & (value >= SKIN_V_MIN)
    )
    if support_mask is not None:
        skin &= support_mask > 0
    return skin.astype(np.uint8) * 255


def _mask_inside_box(mask, box):
    """Clip a binary candidate mask to a safe image-space bounding box."""
    height, width = mask.shape[:2]
    x, y, box_width, box_height = (int(value) for value in box)
    x1 = min(max(x, 0), width)
    y1 = min(max(y, 0), height)
    x2 = min(max(x + box_width, 0), width)
    y2 = min(max(y + box_height, 0), height)
    clipped = np.zeros((height, width), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped


def _segment_hole(source, mask_filled, mask_raw, box):
    """Return only the accepted skin/background opening inside a hole box."""
    skin = _skin_colour_mask(source, mask_filled)
    skin = cv2.morphologyEx(
        skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    skin = cv2.morphologyEx(
        skin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    background_opening = cv2.subtract(mask_filled, mask_raw)
    evidence = cv2.bitwise_or(skin, background_opening)
    return _mask_inside_box(evidence, box)


def _segment_hull_gap(mask_filled, box):
    """Segment missing material between a glove silhouette and its hull."""
    contours, _ = cv2.findContours(
        mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros_like(mask_filled)
    glove_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(glove_contour)
    hull_mask = np.zeros_like(mask_filled)
    cv2.drawContours(hull_mask, [hull], -1, 255, cv2.FILLED)
    return _mask_inside_box(cv2.subtract(hull_mask, mask_filled), box)


def _visible_short_finger(mask_filled, gap_component):
    """Find a glove-covered short finger protruding into a missing-space gap.

    The dilated gap touches the fingers on both sides. Removing the lower gap
    closure separates those contacts; an interior contact (not the two outside
    walls) is evidence of a curled/shortened finger still physically present.
    """
    gap_u8 = (gap_component > 0).astype(np.uint8) * 255
    points = cv2.findNonZero(gap_u8)
    if points is None:
        return np.zeros_like(mask_filled)
    gap_x, gap_y, gap_width, gap_height = cv2.boundingRect(points)
    if gap_width < 9 or gap_height < 9:
        return np.zeros_like(mask_filled)

    contact = cv2.bitwise_and(
        cv2.dilate(
            gap_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        ),
        mask_filled,
    )
    search = np.zeros_like(mask_filled)
    search_bottom = min(
        mask_filled.shape[0], gap_y + max(1, round(0.90 * gap_height))
    )
    side_margin = max(2, round(0.10 * gap_width))
    search[
        gap_y:search_bottom,
        min(mask_filled.shape[1], gap_x + side_margin):
        min(mask_filled.shape[1], gap_x + gap_width - side_margin),
    ] = 255
    contact = cv2.bitwise_and(contact, search)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (contact > 0).astype(np.uint8), 8
    )
    candidates = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= 20:
            candidates.append((area, label))
    if not candidates:
        return np.zeros_like(mask_filled)

    _, selected_label = max(candidates)
    selected_contact = (labels == selected_label).astype(np.uint8) * 255
    # Grow inward from the detected boundary contact to colour the finger's
    # material, while intersection with the glove mask prevents background fill.
    grown = cv2.dilate(
        selected_contact,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)),
    )
    return cv2.bitwise_and(grown, mask_filled)


def _segment_finger_not_enough(source, mask_filled, box, material):
    """Colour a visible short finger; leave true absence for box fallback."""
    contours, _ = cv2.findContours(
        mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros_like(mask_filled)

    glove_contour = max(contours, key=cv2.contourArea)
    _, glove_y, _, glove_height = cv2.boundingRect(glove_contour)
    hull = cv2.convexHull(glove_contour)
    hull_mask = np.zeros_like(mask_filled)
    cv2.drawContours(hull_mask, [hull], -1, 255, cv2.FILLED)
    hull_area = max(cv2.countNonZero(hull_mask), 1)
    gap = cv2.subtract(hull_mask, mask_filled)

    material_key = str(material).lower() if material is not None else None
    rule = FINGER_NOT_ENOUGH_MATERIAL_RULES.get(
        material_key, FINGER_NOT_ENOUGH_DEFAULT_RULE
    )
    box_mask = _mask_inside_box(
        np.full_like(mask_filled, 255, dtype=np.uint8), box
    )

    # A missing/curled finger normally creates the largest qualifying upper
    # hull indentation. It is used to locate a visible short finger, but the
    # empty hull gap itself is not coloured as though it were glove material.
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (gap > 0).astype(np.uint8), 8
    )
    best_label = None
    best_area = -1
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        area_ratio = float(area) / hull_area
        y_ratio = (
            float(centroids[label, 1]) - glove_y
        ) / max(glove_height, 1)
        component = labels == label
        if (
            area_ratio >= rule["indent_min_area_ratio"]
            and y_ratio <= rule["indent_max_y_ratio"]
            and np.any(component & (box_mask > 0))
            and area > best_area
        ):
            best_label = label
            best_area = area

    segmented = np.zeros_like(mask_filled)

    # If the shortened finger exposes skin, include only components satisfying
    # the same size, boundary-contact and vertical-position evidence used by
    # the recogniser.
    skin = _skin_colour_mask(source, mask_filled)
    skin = cv2.morphologyEx(
        skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    skin = cv2.morphologyEx(
        skin, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
    )
    eroded = cv2.erode(
        mask_filled,
        np.ones(
            (FINGER_SKIN_BOUNDARY_KSIZE, FINGER_SKIN_BOUNDARY_KSIZE),
            np.uint8,
        ),
    )
    glove_boundary = (mask_filled > 0) & (eroded == 0)
    skin_count, skin_labels, skin_stats, skin_centroids = (
        cv2.connectedComponentsWithStats((skin > 0).astype(np.uint8), 8)
    )
    for label in range(1, skin_count):
        area = int(skin_stats[label, cv2.CC_STAT_AREA])
        component = skin_labels == label
        boundary_ratio = (
            float(np.count_nonzero(component & glove_boundary)) / max(area, 1)
        )
        y_ratio = (
            float(skin_centroids[label, 1]) - glove_y
        ) / max(glove_height, 1)
        if (
            float(area) / hull_area >= rule["skin_min_area_ratio"]
            and boundary_ratio >= rule["skin_min_boundary_ratio"]
            and y_ratio <= rule["skin_max_y_ratio"]
        ):
            segmented[component & (box_mask > 0)] = 255

    if cv2.countNonZero(segmented) == 0 and best_label is not None:
        segmented = _visible_short_finger(
            mask_filled, labels == best_label
        )
    # An all-zero mask is intentional for a completely absent finger. The
    # drawing function then uses the detector's localised missing-space box.
    return segmented


def _segment_thin_area(source, mask_filled, box, material):
    """Segment the transparency/paleness evidence of accepted thinning."""
    material_key = str(material).lower() if material is not None else None
    region, cotton_subtype = _thin_material_region(source, material_key)
    if region is None:
        return np.zeros_like(mask_filled)

    interior = cv2.erode(
        region,
        np.ones((THIN_ROI_ERODE_KSIZE, THIN_ROI_ERODE_KSIZE), np.uint8),
    )
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    _, saturation, _ = cv2.split(hsv)
    lightness = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)[:, :, 0]
    skin = _skin_colour_mask(source, interior)
    skin = cv2.morphologyEx(
        skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    interior_bool = interior > 0

    if material_key == "cotton" and cotton_subtype == "blue":
        candidate = (
            (saturation < THIN_COTTON_BLUE_S25_MAX) & interior_bool
        ).astype(np.uint8) * 255
    elif material_key == "cotton":
        # Open white-cotton weave appears as many nearby skin dots. A local
        # mean converts those dots into the continuous affected fabric region.
        density = cv2.blur(
            skin,
            (THIN_SEGMENT_DENSITY_KSIZE, THIN_SEGMENT_DENSITY_KSIZE),
        )
        candidate = (
            (density >= THIN_SEGMENT_DENSITY_MIN) & interior_bool
        ).astype(np.uint8) * 255
    elif material_key == "nitrile":
        pale = (
            (lightness > THIN_NITRILE_LIGHT_P25_MIN)
            & (saturation < THIN_NITRILE_SHADOW_S_MEDIAN_MAX)
            & interior_bool
        ).astype(np.uint8) * 255
        transparent = cv2.dilate(skin, np.ones((7, 7), np.uint8))
        candidate = cv2.bitwise_or(pale, transparent)
    elif material_key == "latex_foam":
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        local_edges = cv2.boxFilter(
            np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)),
            cv2.CV_32F,
            (15, 15),
        )
        candidate = (
            (lightness > THIN_LATEX_LIGHT_P25_MIN)
            & (saturation < THIN_LATEX_BRIGHT_S_MEDIAN_MAX)
            & (local_edges < THIN_LATEX_LAPLACIAN_MEAN_MAX)
            & interior_bool
        ).astype(np.uint8) * 255
    else:
        candidate = np.zeros_like(mask_filled)

    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
    )
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8)
    )
    candidate = _mask_inside_box(candidate, box)
    if cv2.countNonZero(candidate) == 0:
        # Classification already accepted the image. For exceptionally diffuse
        # latex/pale evidence, the eroded material ROI is the honest region of
        # support and is preferable to inventing a filled rectangle.
        candidate = _mask_inside_box(interior, box)
    return candidate


def _segment_stain(img, mask_filled, bg_color, box):
    """Recreate the accepted stain colour-distance pixels inside its box."""
    glove_color = get_glove_color(img, mask_filled)
    if glove_color is None:
        return np.zeros_like(mask_filled)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    inside = cv2.erode(mask_filled, np.ones((9, 9), np.uint8)) > 0
    dist_glove = np.linalg.norm(lab - glove_color, axis=2)
    dist_bg = np.linalg.norm(lab - bg_color, axis=2)
    stain = (
        (dist_glove > STAIN_COLOR_DIST)
        & (dist_bg > BG_MATCH_DIST)
        & inside
    ).astype(np.uint8) * 255
    stain = cv2.morphologyEx(
        stain, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    return _mask_inside_box(stain, box)


def build_defect_masks(img, mask_filled, mask_raw, bg_color, defects,
                       img_plain=None, material=None):
    """Build one pixel-level binary mask for every recognised defect.

    This is a post-recognition segmentation stage. Keeping it separate means
    every detector still follows the fixed assignment contract while the GUI,
    evaluator and saved failure images all receive the same coloured result.
    """
    source = img_plain if img_plain is not None else img
    masks = []
    reusable = {}
    full_box = (0, 0, mask_filled.shape[1], mask_filled.shape[0])
    for name, box in defects:
        if name == "Hole":
            if name not in reusable:
                reusable[name] = _segment_hole(
                    source, mask_filled, mask_raw, full_box
                )
            mask = _mask_inside_box(reusable[name], box)
        elif name == "Open Tear":
            if name not in reusable:
                reusable[name] = _segment_hull_gap(mask_filled, full_box)
            mask = _mask_inside_box(reusable[name], box)
        elif name == "Finger Not Enough":
            mask = _segment_finger_not_enough(
                source, mask_filled, box, material
            )
        elif name == "Thin / Overstretched":
            mask = _segment_thin_area(source, mask_filled, box, material)
        elif name == "Stain":
            if name not in reusable:
                reusable[name] = _segment_stain(
                    img, mask_filled, bg_color, full_box
                )
            mask = _mask_inside_box(reusable[name], box)
        else:
            mask = np.zeros_like(mask_filled)
        masks.append((mask > 0).astype(np.uint8) * 255)
    return masks


def draw_results(img, defects, defect_masks=None):
    """Draw coloured pixel segmentation, contours and English labels.

    ``defect_masks`` is optional for backward compatibility with the original
    self-test helper. When omitted, the previous bounding-box display is used.
    """
    out = img.copy()
    if defect_masks is None:
        defect_masks = [None] * len(defects)

    for index, (name, (x, y, w, h)) in enumerate(defects):
        mask = defect_masks[index] if index < len(defect_masks) else None
        has_mask = (
            isinstance(mask, np.ndarray)
            and mask.shape == out.shape[:2]
            and cv2.countNonZero((mask > 0).astype(np.uint8)) > 0
        )
        color = DEFECT_OVERLAY_COLORS.get(name, (0, 0, 255))
        if has_mask:
            selected = mask > 0
            colour_layer = np.empty_like(out)
            colour_layer[:] = color
            blended = cv2.addWeighted(
                out, 1.0 - DEFECT_OVERLAY_ALPHA,
                colour_layer, DEFECT_OVERLAY_ALPHA,
                0.0,
            )
            out[selected] = blended[selected]
            contours, _ = cv2.findContours(
                selected.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(out, contours, -1, color, 2)
            mask_points = cv2.findNonZero(selected.astype(np.uint8))
            label_x, label_y, _, _ = cv2.boundingRect(mask_points)
        else:
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            label_x, label_y = x, y

        text_origin = (label_x, max(label_y - 8, 18))
        cv2.putText(
            out, name, text_origin,
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4,
        )
        cv2.putText(
            out, name, text_origin,
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
    return out

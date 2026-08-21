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
from dataclasses import dataclass

import cv2
import numpy as np

# --- Tunable parameters (kept here for sensitivity experiments and for
# citing in the report) ---
BG_MATCH_DIST = 30.0       # hole criterion: candidate's Lab distance from the background must be below this
STAIN_MASK_ERODE_KSIZE = 7     # exclude colour mixing at the glove/background outline
STAIN_NEUTRAL_S_MAX = 45       # HSV: white/grey/black material saturation ceiling
STAIN_LIGHT_V_MIN = 90         # only a sufficiently bright neutral glove uses the light branch
STAIN_NEUTRAL_RATIO = 0.20     # minimum light-neutral fraction for the light-glove branch
STAIN_NEUTRAL_BASE_CLOSE_KSIZE = 101  # bridge large marks while rebuilding knitted material
STAIN_NEUTRAL_REGION_ERODE_KSIZE = 15 # reject colour mixing at the rebuilt outer edge
STAIN_NEUTRAL_CHROMA_DIST = 10.0      # Lab a/b departure from normal material
STAIN_NEUTRAL_BG_CHROMA_DIST = 20.0   # reject holes/gaps with background chroma
STAIN_NEUTRAL_DENSITY_KSIZE = 17      # neighbourhood used to measure colour-block density
STAIN_NEUTRAL_DENSITY_MIN = 0.18      # sparse knit texture is not a stain
STAIN_NEUTRAL_CLOSE_KSIZE = 11        # join small gaps inside one knitted stain
STAIN_NEUTRAL_MIN_RADIUS = 4.0        # reject thin finger-edge bands at 800 px width
STAIN_NEUTRAL_MIN_COMPACTNESS = 0.35  # 4*pi*A/P^2; solid regions score closer to 1
STAIN_COLOR_S_MIN = 45         # minimum saturation for reliable hue on a coloured glove
STAIN_COLOR_V_MIN = 35         # reject pixels too dark for reliable hue
STAIN_BASE_HUE_TOL = 15        # OpenCV hue tolerance around the material's dominant hue
STAIN_HUE_DIST = 20            # hue departure required for an off-colour mark
STAIN_BASE_CLOSE_KSIZE = 41    # bridge texture/marks while rebuilding the material region
STAIN_LOCAL_KSIZE = 41         # local window for black/white/same-hue marks on coloured gloves
STAIN_LOCAL_DIST = 20.0        # weighted local Lab-distance threshold
STAIN_LUMA_WEIGHT = 0.5        # down-weight illumination while retaining black/white marks
STAIN_OPEN_KSIZE = 3           # remove isolated pixel noise
STAIN_CLOSE_KSIZE = 15         # join small gaps inside one mark
MIN_AREA_HOLE = 60
MIN_AREA_STAIN = 200           # minimum area at the normalised 800-pixel width

# Fixed BGR colours used for masks, contours, boxes, labels and the GUI key.
DEFECT_COLORS = {
    "Stain": (0, 140, 255),
    "Tear / Hole": (40, 40, 220),
    "Open Tear": (190, 55, 150),
}
DEFAULT_DEFECT_COLOR = (35, 160, 70)


@dataclass
class Detection:
    """One located defect with an optional pixel mask and rule evidence.

    ``evidence`` is a 0..100 heuristic rule-strength score, not a learned
    probability. Iteration keeps compatibility with legacy ``name, box`` code.
    """

    name: str
    box: tuple
    mask: np.ndarray | None = None
    evidence: float = 0.0

    def __iter__(self):
        yield self.name
        yield self.box

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
        color_distance = float(np.linalg.norm(mean_color - bg_color))
        if color_distance < BG_MATCH_DIST:
            color_fit = 1.0 - color_distance / BG_MATCH_DIST
            size_fit = min(1.0, cv2.contourArea(c) / (MIN_AREA_HOLE * 4.0))
            evidence = 50.0 + 50.0 * (0.75 * color_fit + 0.25 * size_fit)
            results.append(Detection(
                "Tear / Hole", cv2.boundingRect(c), blob, round(evidence, 1),
            ))
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
        angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        if angle > MAX_TEAR_APEX_ANGLE:
            continue

        points = np.array([p1, p2, apex])
        blob = np.zeros(mask_filled.shape, np.uint8)
        cv2.fillPoly(blob, [points], 255)
        depth_fit = min(1.0, depth / (2.0 * MIN_TEAR_DEPTH_RATIO * diag))
        mouth_fit = np.clip(
            1.0 - mouth / (MAX_TEAR_MOUTH_RATIO * depth), 0.0, 1.0,
        )
        angle_fit = np.clip(1.0 - angle / MAX_TEAR_APEX_ANGLE, 0.0, 1.0)
        evidence = 50.0 + 50.0 * (
            0.30 * depth_fit + 0.35 * mouth_fit + 0.35 * angle_fit
        )
        results.append(Detection(
            "Open Tear", cv2.boundingRect(points), blob, round(float(evidence), 1),
        ))
    return results


# ============================================================
# Defect 3: stain
# ============================================================
def _odd_kernel(preferred, h, w):
    """Clamp a morphology/median kernel to the image and keep it odd."""
    size = min(preferred, h, w)
    if size % 2 == 0:
        size -= 1
    return size


def _largest_component(mask):
    """Keep only the largest connected component in a binary image."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return np.zeros_like(mask)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == index).astype(np.uint8) * 255


def _region_from_base(base, close_ksize):
    """Rebuild the glove from normal material pixels, excluding carpet/skin."""
    base = cv2.morphologyEx(
        base, cv2.MORPH_CLOSE,
        np.ones((close_ksize, close_ksize), np.uint8),
    )
    base = cv2.morphologyEx(base, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    base = _largest_component(base)
    contours, _ = cv2.findContours(base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = np.zeros_like(base)
    if contours:
        cv2.drawContours(
            region, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED,
        )
    return region > 0


def detect_stains(img, mask_filled, mask_raw, bg_color):
    """Select colour rules by material appearance and return pixel masks.

    For light knitted/latex gloves, normal neutral pixels rebuild the whole
    material surface. A candidate must differ in Lab chroma from both the
    material and the background and form a dense, compact colour block. This
    recovers large marks removed by foreground segmentation while rejecting
    yellow background visible through knit holes. Coloured gloves use a
    dominant-hue departure, with a strict local-Lab fallback only when that
    primary rule finds no credible region.

    Trade-off: white powder, very faint marks and small edge-adjacent stains
    may be missed. Thresholds assume preprocessing to an 800-pixel width.
    """
    h, w = img.shape[:2]
    erode_ksize = _odd_kernel(STAIN_MASK_ERODE_KSIZE, h, w)
    base_close_ksize = _odd_kernel(STAIN_BASE_CLOSE_KSIZE, h, w)
    neutral_base_close_ksize = _odd_kernel(STAIN_NEUTRAL_BASE_CLOSE_KSIZE, h, w)
    neutral_region_erode_ksize = _odd_kernel(
        STAIN_NEUTRAL_REGION_ERODE_KSIZE, h, w,
    )
    neutral_close_ksize = _odd_kernel(STAIN_NEUTRAL_CLOSE_KSIZE, h, w)
    local_ksize = _odd_kernel(STAIN_LOCAL_KSIZE, h, w)
    close_ksize = _odd_kernel(STAIN_CLOSE_KSIZE, h, w)
    if min(
        erode_ksize, base_close_ksize, neutral_base_close_ksize,
        neutral_region_erode_ksize, neutral_close_ksize,
        local_ksize, close_ksize,
    ) < 3:
        return []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hue, sat, val = cv2.split(hsv)
    raw_foreground = mask_raw > 0
    inside = cv2.erode(
        mask_raw, np.ones((erode_ksize, erode_ksize), np.uint8),
    ) > 0
    if not inside.any():
        return []

    neutral_light = inside & (sat <= STAIN_NEUTRAL_S_MAX) & (val >= STAIN_LIGHT_V_MIN)
    neutral_ratio = neutral_light.sum() / inside.sum()
    candidate = np.zeros(mask_raw.shape, np.uint8)
    evidence_map = np.zeros(mask_raw.shape, np.float32)
    glove_region = np.zeros(mask_raw.shape, dtype=bool)
    colorful_branch = neutral_ratio < STAIN_NEUTRAL_RATIO

    if not colorful_branch:
        glove_region = _region_from_base(
            neutral_light.astype(np.uint8) * 255, neutral_base_close_ksize,
        )
        if glove_region.any():
            glove_inside = cv2.erode(
                glove_region.astype(np.uint8) * 255,
                np.ones(
                    (neutral_region_erode_ksize, neutral_region_erode_ksize),
                    np.uint8,
                ),
            ) > 0
            base_lab = np.median(lab_u8[neutral_light], axis=0).astype(np.float32)
            lab_float = lab_u8.astype(np.float32)
            chroma_dist = np.hypot(
                lab_float[:, :, 1] - base_lab[1],
                lab_float[:, :, 2] - base_lab[2],
            )
            background_chroma_dist = np.hypot(
                lab_float[:, :, 1] - bg_color[1],
                lab_float[:, :, 2] - bg_color[2],
            )
            direct_pixels = (
                glove_inside
                & (chroma_dist >= STAIN_NEUTRAL_CHROMA_DIST)
                & (background_chroma_dist >= STAIN_NEUTRAL_BG_CHROMA_DIST)
            )
            density_kernel = (
                STAIN_NEUTRAL_DENSITY_KSIZE, STAIN_NEUTRAL_DENSITY_KSIZE,
            )
            local_density = cv2.boxFilter(
                direct_pixels.astype(np.float32), -1,
                density_kernel, normalize=True,
            )
            stain_pixels = direct_pixels & (
                local_density >= STAIN_NEUTRAL_DENSITY_MIN
            )
            candidate[stain_pixels] = 255
            material_strength = np.clip(
                (chroma_dist - STAIN_NEUTRAL_CHROMA_DIST)
                / max(STAIN_NEUTRAL_CHROMA_DIST * 2.0, 1.0),
                0.0, 1.0,
            )
            background_strength = np.clip(
                (background_chroma_dist - STAIN_NEUTRAL_BG_CHROMA_DIST)
                / max(STAIN_NEUTRAL_BG_CHROMA_DIST * 1.5, 1.0),
                0.0, 1.0,
            )
            density_strength = np.clip(
                (local_density - STAIN_NEUTRAL_DENSITY_MIN) / 0.50,
                0.0, 1.0,
            )
            strength = (
                0.50 * material_strength
                + 0.25 * background_strength
                + 0.25 * density_strength
            )
            evidence_map[stain_pixels] = 0.55 + 0.45 * strength[stain_pixels]
    else:
        colorful = inside & (sat >= STAIN_COLOR_S_MIN) & (val >= STAIN_COLOR_V_MIN)
        if colorful.any():
            hist = np.bincount(hue[colorful], minlength=180).astype(np.float32)
            # Hue is circular: join both ends, smooth nine bins, then take the peak.
            smooth = np.convolve(
                np.r_[hist[-4:], hist, hist[:4]], np.ones(9), mode="valid",
            )
            dominant_hue = int(np.argmax(smooth) % 180)
            raw_delta = np.abs(hue.astype(np.int16) - dominant_hue)
            hue_delta = np.minimum(raw_delta, 180 - raw_delta)
            base = colorful & (hue_delta <= STAIN_BASE_HUE_TOL)
            glove_region = _region_from_base(
                base.astype(np.uint8) * 255, base_close_ksize,
            )
            stain_pixels = (
                glove_region & raw_foreground
                & (sat >= STAIN_COLOR_S_MIN) & (val >= STAIN_COLOR_V_MIN)
                & (hue_delta >= STAIN_HUE_DIST)
            )
            candidate[stain_pixels] = 255
            strength = np.clip(
                (hue_delta.astype(np.float32) - STAIN_HUE_DIST)
                / max(90.0 - STAIN_HUE_DIST, 1.0),
                0.0, 1.0,
            )
            evidence_map[stain_pixels] = 0.55 + 0.45 * strength[stain_pixels]

    if not glove_region.any():
        neutral_base = inside & (sat <= STAIN_NEUTRAL_S_MAX)
        glove_region = _region_from_base(
            neutral_base.astype(np.uint8) * 255, base_close_ksize,
        )

    # Do not stack local-Lab fold/highlight false positives on top of a hue
    # result. Use the fallback only when the primary colour rule found no
    # credible region.
    if colorful_branch and glove_region.any():
        primary = cv2.morphologyEx(
            candidate, cv2.MORPH_OPEN,
            np.ones((STAIN_OPEN_KSIZE, STAIN_OPEN_KSIZE), np.uint8),
        )
        primary = cv2.morphologyEx(
            primary, cv2.MORPH_CLOSE,
            np.ones((close_ksize, close_ksize), np.uint8),
        )
        primary_contours, _ = cv2.findContours(
            primary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        has_primary = any(
            cv2.contourArea(contour) >= MIN_AREA_STAIN
            for contour in primary_contours
        )
        if not has_primary:
            local_lab = cv2.medianBlur(lab_u8, local_ksize).astype(np.float32)
            delta = lab_u8.astype(np.float32) - local_lab
            local_dist = np.sqrt(
                (STAIN_LUMA_WEIGHT * delta[:, :, 0]) ** 2
                + delta[:, :, 1] ** 2
                + delta[:, :, 2] ** 2
            )
            local_region = cv2.erode(
                mask_raw, np.ones((local_ksize, local_ksize), np.uint8),
            ) > 0
            local_pixels = (
                local_region & glove_region & (local_dist >= STAIN_LOCAL_DIST)
            )
            candidate[local_pixels] = 255
            strength = np.clip(
                (local_dist - STAIN_LOCAL_DIST) / max(STAIN_LOCAL_DIST * 2.0, 1.0),
                0.0, 1.0,
            )
            evidence_map[local_pixels] = 0.55 + 0.45 * strength[local_pixels]

    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_OPEN,
        np.ones((STAIN_OPEN_KSIZE, STAIN_OPEN_KSIZE), np.uint8),
    )
    final_close_ksize = neutral_close_ksize if not colorful_branch else close_ksize
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE,
        np.ones((final_close_ksize, final_close_ksize), np.uint8),
    )
    contours, _ = cv2.findContours(
        candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    results = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA_STAIN:
            continue
        filled = np.zeros(candidate.shape, np.uint8)
        cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
        blob = cv2.bitwise_and(candidate, filled)
        if not colorful_branch:
            perimeter = cv2.arcLength(contour, True)
            compactness = (
                4.0 * np.pi * cv2.contourArea(contour)
                / max(perimeter * perimeter, 1.0)
            )
            radius = float(cv2.distanceTransform(
                blob, cv2.DIST_L2, 3,
            ).max())
            if (
                compactness < STAIN_NEUTRAL_MIN_COMPACTNESS
                or radius < STAIN_NEUTRAL_MIN_RADIUS
            ):
                continue
        scored = evidence_map[(blob > 0) & (evidence_map > 0)]
        evidence = (
            55.0 if scored.size == 0
            else 100.0 * float(np.percentile(scored, 75))
        )
        results.append(Detection(
            "Stain", cv2.boundingRect(contour), blob, round(evidence, 1),
        ))
    return sorted(results, key=lambda result: (result.box[1], result.box[0]))


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
    for defect in defects:
        name, box = defect
        if any(_box_iou(box, kept_defect.box) > DEDUP_IOU for kept_defect in kept):
            continue
        kept.append(defect)
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
            for item in found:
                name, box = item
                clean_box = tuple(int(v) for v in box)
                if isinstance(item, Detection):
                    defects.append(Detection(
                        str(name), clean_box, item.mask, float(item.evidence),
                    ))
                else:
                    defects.append(Detection(str(name), clean_box))
        except Exception as e:
            errors.append(f"{det.__name__} raised an error: {e}")
    return deduplicate(defects), errors


def detection_color(name):
    """Return the fixed BGR display colour for one defect type."""
    return DEFECT_COLORS.get(name, DEFAULT_DEFECT_COLOR)


def detection_mask(defect, shape):
    """Return a pixel mask, falling back to the box for legacy detectors."""
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
    """Return union(defect pixels) / complete glove outline as a percentage."""
    glove = glove_mask > 0
    glove_pixels = int(np.count_nonzero(glove))
    if glove_pixels == 0 or not defects:
        return 0.0
    affected = np.zeros(glove.shape, dtype=bool)
    for defect in defects:
        affected |= detection_mask(defect, glove_mask.shape)
    return 100.0 * np.count_nonzero(affected & glove) / glove_pixels


def overall_evidence_score(defects, image_shape):
    """Return the pixel-area-weighted heuristic evidence score (not probability)."""
    weighted_sum = 0.0
    total_weight = 0
    for defect in defects:
        weight = int(np.count_nonzero(detection_mask(defect, image_shape)))
        evidence = defect.evidence if isinstance(defect, Detection) else 0.0
        if weight > 0:
            weighted_sum += float(evidence) * weight
            total_weight += weight
    return weighted_sum / total_weight if total_weight else 0.0


def draw_results(img, defects, alpha=0.38):
    """Draw colour-coded pixel overlays, contours, boxes and evidence labels."""
    out = img.copy()
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
            contours, _ = cv2.findContours(
                mask.astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(out, contours, -1, color, 2)

        cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)
        evidence = defect.evidence if isinstance(defect, Detection) else 0.0
        label = f"{name} {evidence:.0f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
        )
        label_y = (
            y - 5 if y - text_h - baseline - 6 >= 0
            else y + text_h + baseline + 6
        )
        top = max(0, label_y - text_h - baseline - 4)
        bottom = min(out.shape[0] - 1, label_y + 3)
        right = min(out.shape[1] - 1, x + text_w + 6)
        cv2.rectangle(out, (x, top), (right, bottom), color, cv2.FILLED)
        text_color = (20, 20, 20) if name == "Stain" else (255, 255, 255)
        cv2.putText(
            out, label, (x + 3, label_y - 1), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, text_color, 1, cv2.LINE_AA,
        )
    return out

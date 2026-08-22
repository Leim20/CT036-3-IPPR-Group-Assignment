# -*- coding: utf-8 -*-
"""
Defect detection: every defect is one function, registered in the DETECTORS
list, and the GUI calls them all automatically.

The rule for writing a detector is that the signature is always:

    def detect_xxx(img, mask_filled, mask_raw, bg_color):
        ...
        return [("Defect Name", (x, y, w, h)), ...]   # [] when nothing is found

All four arguments go to every detector; ignore the ones you do not need (the
tear detector, for instance, only uses mask_filled). The defect name is drawn
on the image and listed in the GUI's text panel.
"""
from dataclasses import dataclass

import cv2
import numpy as np

# --- Tunable parameters, kept together so a sensitivity study is easy to run
#     and easy to write up in the report ---
BG_MATCH_DIST = 30.0       # hole rule: a candidate counts as a hole only if its
                           # Lab distance to the background colour is below this
STAIN_MASK_ERODE_KSIZE = 7     # drop the rim where glove and background blend
STAIN_NEUTRAL_S_MAX = 45       # HSV: below this the material counts as neutral
                               # (white / grey / black)
STAIN_LIGHT_V_MIN = 90         # neutral AND bright enough -> white/light-grey branch
STAIN_NEUTRAL_RATIO = 0.20     # this fraction of light-neutral pixels means a light glove
STAIN_NEUTRAL_BASE_CLOSE_KSIZE = 101  # bridge over a large stain to rebuild the
                                      # complete knitted glove surface
STAIN_NEUTRAL_REGION_ERODE_KSIZE = 15 # drop the blended rim of the rebuilt outline
STAIN_NEUTRAL_CHROMA_DIST = 8.0      # Lab a/b deviation from the normal material
STAIN_NEUTRAL_BG_CHROMA_DIST = 20.0   # reject mesh holes and finger gaps that share
                                      # the background's chroma
STAIN_NEUTRAL_DENSITY_KSIZE = 17      # window for measuring solid-patch density
STAIN_NEUTRAL_DENSITY_MIN = 0.12      # sparse knit texture must not become a stain
STAIN_NEUTRAL_CLOSE_KSIZE = 11        # close the small gaps inside one knitted stain
STAIN_NEUTRAL_MIN_RADIUS = 6.0        # reject thin finger edges and crease shadows
                                      # (pixels, at the standard 800px width)
STAIN_NEUTRAL_MIN_COMPACTNESS = 0.20  # 4*pi*A/P^2; the closer to 1, the more solid
# The shape thresholds are stricter for coloured (smooth nitrile) gloves: paint
# and mud sit on a smooth surface as sharp-edged blobs, while cotton soaks the
# stain up so its edges spread and its shape loosens. The two branches therefore
# cannot share one set of thresholds.
# Measured: a nitrile cuff shadow false alarm has compactness 0.20 / inscribed
# radius 6.4, while the weakest real stain is 0.27 / 11.5. On cotton a real stain
# can drop to 0.20, so 0.25 there would throw real stains away.
STAIN_COLOR_MIN_COMPACTNESS = 0.25
STAIN_COLOR_MIN_RADIUS = 8.0
# The area threshold also has to be set separately for the coloured branch.
# Measured: on smooth nitrile a real stain runs 2734-63396 px, while finger-edge
# shadow false alarms are only 660-1449 px, so 2000 leaves room on both sides.
# But on cotton (the neutral branch) the smallest real stain is 575 px, so
# sharing this number would throw real stains away.
STAIN_COLOR_MIN_AREA = 2000
STAIN_COLOR_S_MIN = 45         # minimum saturation for a hue to be meaningful
STAIN_COLOR_V_MIN = 35         # reject pixels too dark for their hue to be trusted
STAIN_BASE_HUE_TOL = 15        # how many OpenCV hue units around the dominant hue
                               # still count as normal material
STAIN_HUE_DIST = 20            # how far from the dominant hue counts as a stain
STAIN_BASE_CLOSE_KSIZE = 41    # join the glove's main-colour regions, bridging
                               # texture and stains to rebuild the outline
STAIN_LOCAL_KSIZE = 41         # local colour window on coloured gloves, to also
                               # catch black, white and same-hue dark stains
STAIN_LOCAL_DIST = 20.0        # threshold on the locally weighted Lab distance
STAIN_LUMA_WEIGHT = 0.5        # down-weight lightness, but stay responsive to
                               # black and white stains
STAIN_OPEN_KSIZE = 3           # remove single-pixel noise
STAIN_CLOSE_KSIZE = 15         # close the small cracks inside one stain
# How far the segmentation mask can be trusted: once this fraction of the mask's
# pixels are literally the background colour, segmentation has swallowed some
# background, and only then do we fall back to the conservative plan of
# rebuilding the glove region from the material's dominant colour.
# Measured over 24 real photos: yellow-mat shots are 0.0-3.5% contaminated,
# stone-tile shots 34.8-60.2%. Nothing falls in between, so 8% is a safe line.
STAIN_SEG_POLLUTION_MAX = 0.08
STAIN_SEG_CLEAN_CHROMA = 15.0  # closer than this in chroma to the background
                               # counts as "literally the background colour"
STAIN_SEG_ERODE_KSIZE = 21      # when using the segmentation outline directly,
                                # erode only slightly to avoid the blended rim
# Dark-stain rule (black paint, ink): how much darker in Lab L than the normal
# material a pixel has to be.
# Why this needs its own rule: the coloured branch finds stains by hue deviation,
# but the hue of a very dark pixel is unreliable to begin with. Measured, black
# paint on a blue glove comes out at hue 110-113 against the material's 104 --
# a few degrees apart, so the hue rule fails outright. On top of that the paint's
# V of 11-25 is filtered out by STAIN_COLOR_V_MIN anyway. Asking "how much
# darker than the material" is far more stable (see the ABS note below).
STAIN_DARK_L_DROP = 85.0
# The absolute lightness cap is only a generous backstop; the relative drop above
# is what actually discriminates.
# Why an absolute threshold cannot carry it: the absolute lightness of black
# paint moves a lot with the lighting. Measured, the first batch of photos gave
# L=9-15 and the second batch, same paint, gave L=55-62 (brighter light, glossier
# surface). An absolute threshold of 45 makes the second batch's largest and most
# obvious stain disappear completely.
# The relative drop is far steadier: 128-142 and 87-94 for the two batches, while
# the darker tone of a genuinely two-tone glove only drops 57. A threshold of 85
# leaves room on both sides.
STAIN_DARK_L_ABS = 110.0
# --- Spotting rules ---
# What separates this from Stain is *count*, not area: a Stain is one or two
# large patches, Spotting is many small dots scattered about. So the primary
# criterion is "how many small-enough dots are there", and only a high enough
# count counts as Spotting -- which is what stops the two from being the same
# technique under two names.
SPOTTING_MIN_COUNT = 5           # at least this many dots before it is Spotting
SPOTTING_MIN_AREA = 50           # smallest allowed dot, to filter noise (at 800px)
SPOTTING_MAX_AREA_RATIO = 0.015  # largest dot as a fraction of the glove area;
                                 # anything bigger is a Stain, not a dot
SPOTTING_MIN_COMPACTNESS = 0.30  # a dot should be a compact little blob; this
                                 # rejects thin edges and creases
# A dot also has to be vivid enough: Spotting is splashed-on colour, so it
# deviates strongly. Measured, real dots (yellow watercolour on a blue glove)
# sit 120 away from the material's chroma, while faint marks on cotton broken up
# by the weave are only 11-15 away -- a factor of 8, so 40 leaves plenty of room
# on both sides.
SPOTTING_MIN_CHROMA_DEV = 40.0
# The dot's colour must also *not* be the background colour. A little background
# bleeds through at the glove's edge, and in the coloured branch that also
# satisfies "hue deviates from the glove's main colour". Measured, those false
# alarms have hue 159-175 (red backdrop) while the real green dots are 72-85, so
# "how far from the background's chroma" removes them cleanly.
SPOTTING_MIN_BG_CHROMA_DIST = 30.0
# Spotting's merge kernel has to be smaller than Stain's: too large a kernel
# glues two neighbouring dots into one. Measured, Stain's 11 loses adjacent dots
# while 7 is right. Stain, on the other hand, needs 11 to put a stain fragmented
# by the knit texture back together -- opposite needs, so they cannot share it.
SPOTTING_CLOSE_KSIZE = 7

# --- Plastic Contamination rules ---
# Cause: packaging film or plastic scraps sticking to the glove on the line.
# What it looks like: transparent film has many sharp creases, and the specular
# reflection along a crease drives saturation down towards 0 (nearly pure white).
# Matte latex and nitrile never produce such pixels. So the criterion is not
# "bright", it is "*unsaturated* bright pixels packed densely into a small area".
# Measured over 1 film photo and 37 existing photos:
#     local highlight density   film 0.52-1.00   |  bare glove max 0.091
#     dense-region area         film 16433 px    |  usable photos all <= 398 px
PLASTIC_MIN_MATERIAL_S = 120.0  # material gate: on a white or grey glove the
                                # material is unsaturated already, so the rule
                                # would be meaningless
PLASTIC_S_DROP = 80.0           # this much less saturated than the material
                                # before it counts as a film reflection
PLASTIC_V_KEEP = 0.80           # and at least 80% of the material's brightness --
                                # this rules out shadows (also unsaturated, but dark)
PLASTIC_MASK_ERODE_KSIZE = 15   # drop the blended rim around the glove outline
PLASTIC_DENSITY_KSIZE = 31      # window for counting reflections in a small patch
PLASTIC_DENSITY_MIN = 0.15      # density threshold; bare glove peaks at 0.091,
                                # so this keeps a 1.6x margin
PLASTIC_CLOSE_KSIZE = 21        # fill the gaps between creases into one patch
# The area thresholds are fractions of the glove area, never hard pixel counts:
# how much of the frame the glove fills depends on shooting distance, so a fixed
# pixel count stops working the moment the framing changes. Measured, as a
# fraction of the glove area:
#     film 3.44%   |   largest false region in any usable photo 0.09%
# The lower bound of 0.5% is 5.5x above the false alarms and 7x below the real
# film, leaving room at both ends.
PLASTIC_MIN_AREA_RATIO = 0.005
PLASTIC_MIN_AREA = 300          # absolute floor, only to reject specks on a tiny image
# Upper bound: film is a *local* foreign object. Backlighting that blows a whole
# glove out into white covers 11-16%, which is a shooting problem rather than a
# defect, so the cap rejects it.
PLASTIC_MAX_AREA_RATIO = 0.08
# Transparent film hazes *everything* it covers, so saturation drops across the
# whole area; a white powder stain only whitens its own little blob and leaves
# the surroundings in the original colour. Hence the requirement that the
# region's median saturation sits clearly below the material's.
# Measured: film region = 42-58% of the material  |  white powder stain = 100%
PLASTIC_MAX_REGION_SAT_RATIO = 0.75
# The decision comes from the density region, but the *extent* is grown from the
# haze region: only creases reflect, the transparent parts of the film do not, so
# a mask built from the density region alone covers only part of the film and
# affected-area comes out too low.
# Measured on this photo: the density region is 6.1% of the glove, growing to the
# whole film gives 11.8%, and loosening the threshold to 90% leaks across the
# entire glove (36%). 85% is the knee.
PLASTIC_HAZE_SAT_RATIO = 0.85
PLASTIC_GROW_CLOSE_KSIZE = 15
PLASTIC_GROW_MAX_RATIO = 0.25   # if it grows past this it has probably leaked, so
                                # fall back to the density region rather than force it

MIN_AREA_HOLE = 60
MIN_AREA_STAIN = 500           # smallest stain at the standard 800px width.
                               # Crease-shadow false alarms all sit at 500-600px
                               # and the smallest real stain is 1700px, so the
                               # threshold goes between them; 9 of 14 annotated
                               # photos then reach 100/100.

# BGR colours: each defect keeps one colour in the result image, which doubles
# as the GUI's legend.
DEFECT_COLORS = {
    "Stain": (0, 140, 255),          # orange
    "Tear / Hole": (40, 40, 220),   # red
    "Open Tear": (190, 55, 150),    # purple
    "Plastic Contamination": (210, 180, 40),  # cyan-blue
}
DEFAULT_DEFECT_COLOR = (35, 160, 70)  # any detector added later: green


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

# Thresholds for an open tear (one that reaches the glove's edge). The numbers
# are measured, not guessed:
#     normal finger gap: mouth/depth = 0.55-0.74, apex angle = 29-40 degrees
#     open tear:         mouth/depth = 0.36,      apex angle = 20 degrees
#     wrist step:        mouth/depth = 5.53,      apex angle = 137 degrees
# So "narrow" plus "sharp" is enough to tell a tear from a finger gap.
CONTOUR_EPSILON = 2.0          # contour simplification, to drop notches that are
                               # just staircase artefacts
MIN_TEAR_DEPTH_RATIO = 0.05    # notch depth / glove bounding-box diagonal, to
                               # filter out shallow notches
MAX_TEAR_MOUTH_RATIO = 0.45    # notch mouth width / depth; a tear is a narrow slit
MAX_TEAR_APEX_ANGLE = 24.0     # apex angle in degrees; a tear is sharp, a finger
                               # gap is blunt

DEDUP_IOU = 0.5   # when two detectors' boxes overlap by more than this, keep only
                  # the one registered first


# ============================================================
# Defect 1: closed hole
# ============================================================
def detect_holes(img, mask_filled, mask_raw, bg_color):
    """A hole shows the background through it, so it is "inside the glove's
    outline yet coloured like the background".
    The candidate's mean colour has to be close to the background colour; without
    that check, anything whose colour deviates at all -- stains, shadows -- would
    be reported as a hole.
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
# Defect 2: open tear (one that reaches the glove's edge)
# ============================================================
def detect_open_tears(img, mask_filled, mask_raw, bg_color):
    """Convexity defects -- the dents between a contour and its convex hull.
    A normal finger gap is also a deep, large notch, so depth alone cannot
    separate them; the shape has to decide. A tear is narrow and sharp (material
    cut open), a finger gap is wide and blunt (a natural U-shaped space).

    Known limitation: this distinction is a heuristic. On real gloves, bent or
    closed fingers and a rolled cuff all change the shape of a finger gap, so the
    thresholds have to be recalibrated against real photos.
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
    diag = float(np.hypot(bw, bh))   # normalise by the glove's own size, so the
                                     # thresholds survive a change of resolution

    results = []
    for s, e, f, depth_fp in defects.reshape(-1, 4):
        depth = depth_fp / 256.0
        if depth < MIN_TEAR_DEPTH_RATIO * diag:
            continue   # too shallow: a staircase artefact or the wrist step

        p1, p2, apex = cnt[s][0], cnt[e][0], cnt[f][0]

        mouth = float(np.linalg.norm(p1 - p2))          # condition 1: narrow
        if mouth > MAX_TEAR_MOUTH_RATIO * depth:
            continue

        v1, v2 = p1 - apex, p2 - apex                     # condition 2: sharp
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        if angle > MAX_TEAR_APEX_ANGLE:
            continue

        points = np.array([p1, p2, apex])
        blob = np.zeros(mask_filled.shape, np.uint8)
        cv2.fillPoly(blob, [points], 255)
        depth_fit = min(1.0, depth / (2.0 * MIN_TEAR_DEPTH_RATIO * diag))
        mouth_fit = np.clip(1.0 - mouth / (MAX_TEAR_MOUTH_RATIO * depth), 0.0, 1.0)
        angle_fit = np.clip(1.0 - angle / MAX_TEAR_APEX_ANGLE, 0.0, 1.0)
        evidence = 50.0 + 50.0 * (
            0.30 * depth_fit + 0.35 * mouth_fit + 0.35 * angle_fit
        )
        results.append(Detection(
            "Open Tear", cv2.boundingRect(points), blob, round(float(evidence), 1),
        ))
    return results


# ============================================================
# Defect 5: Plastic Contamination
# ============================================================
def detect_plastic_contamination(img, mask_filled, mask_raw, bg_color):
    """Find *unsaturated* highlights, then require them to be packed densely
    into a small area.

    When transparent film lies on the glove, the specular reflection along each
    crease is close to pure white -- saturation collapses while brightness does
    not. A matte glove surface never does this. Single such pixels turn up all
    over a real photo (sensor noise, blended edges), so the real criterion is
    *density*: only when a small window is packed with them is there actually a
    creased film lying there.

    How the detectors in this module divide the work (this is exactly what the
    report's "choice of technique" section is about):
        detect_stains      chroma *deviates* from the material   -> coloured stains
        detect_spotting    *counts* small round coloured dots    -> scattered spots
        detect_holes       candidate colour *equals* the backdrop -> holes right through
        this function      *local density* of unsaturated glare  -> transparent film

    Known limitations:
      * on a white or grey glove the rule is meaningless (the material is already
        unsaturated), so the detector abstains;
      * strong backlighting blows the glove itself out into large white patches.
        The area cap rejects those, but an extremely over-exposed photo can still
        fool it -- shooting with diffuse light avoids this completely.
    """
    h, w = img.shape[:2]
    erode_ksize = _odd_kernel(PLASTIC_MASK_ERODE_KSIZE, h, w)
    density_ksize = _odd_kernel(PLASTIC_DENSITY_KSIZE, h, w)
    close_ksize = _odd_kernel(PLASTIC_CLOSE_KSIZE, h, w)
    if min(erode_ksize, density_ksize, close_ksize) < 3:
        return []

    inside = cv2.erode(
        mask_filled, np.ones((erode_ksize, erode_ksize), np.uint8)) > 0
    if not inside.any():
        return []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    material_sat = float(np.median(sat[inside]))
    material_val = float(np.median(val[inside]))
    if material_sat < PLASTIC_MIN_MATERIAL_S:
        return []   # neutral-coloured glove: no evidence, so no guess

    # Reflections off the film's creases: saturation drops, brightness does not
    # (which is what tells them apart from shadows)
    specular = (
        inside
        & (sat <= material_sat - PLASTIC_S_DROP)
        & (val >= material_val * PLASTIC_V_KEEP)
    )
    if not specular.any():
        return []

    density = cv2.boxFilter(
        specular.astype(np.float32), -1, (density_ksize, density_ksize))
    region = ((density >= PLASTIC_DENSITY_MIN) & inside).astype(np.uint8) * 255
    region = cv2.morphologyEx(
        region, cv2.MORPH_CLOSE, np.ones((close_ksize, close_ksize), np.uint8))

    glove_area = float(inside.sum())
    min_area = max(PLASTIC_MIN_AREA, glove_area * PLASTIC_MIN_AREA_RATIO)
    max_area = glove_area * PLASTIC_MAX_AREA_RATIO

    # Haze region: wherever the film covers, saturation is lower overall, whether
    # or not a crease reflects there. It is used only to trace the film's full
    # extent -- the decision still belongs to the density region above. Haze on
    # its own would take in shadows and dark weave too, so it is only trustworthy
    # with a density seed behind it.
    grow_close = _odd_kernel(PLASTIC_GROW_CLOSE_KSIZE, h, w)
    hazed = (inside & (sat <= material_sat * PLASTIC_HAZE_SAT_RATIO)).astype(np.uint8)
    if grow_close >= 3:
        hazed = cv2.morphologyEx(
            hazed, cv2.MORPH_CLOSE, np.ones((grow_close, grow_close), np.uint8))
    haze_count, haze_labels, _, _ = cv2.connectedComponentsWithStats(hazed, 8)
    grow_cap = glove_area * PLASTIC_GROW_MAX_RATIO

    count, labels, stats, _ = cv2.connectedComponentsWithStats(region, 8)
    used_haze = set()
    results = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        picked = labels == index
        # Transparency check: everything the film covers should be greyed out. A
        # few white lines with the original colour still around them is a white
        # stain (detect_stains' job), not a covering film.
        if float(np.median(sat[picked])) > material_sat * PLASTIC_MAX_REGION_SAT_RATIO:
            continue
        # Use the haze component the seed sits in as the final extent; if it grows
        # too far, or there is none, fall back to the seed.
        extent = picked
        ids = {int(v) for v in np.unique(haze_labels[picked]) if v}
        if ids:
            grown = np.isin(haze_labels, list(ids))
            if grown.sum() <= grow_cap:
                if ids & used_haze:
                    continue      # a second seed on the same film: report it once
                used_haze |= ids
                extent = grown
        blob = extent.astype(np.uint8) * 255
        ys, xs = np.where(extent)
        box = (int(xs.min()), int(ys.min()),
               int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        # How well each of the three pieces of evidence fits, weighted into a
        # 0-100 rule score (not a probability)
        peak_density = float(density[picked].max())
        mean_density = float(density[picked].mean())
        sat_drop = material_sat - float(np.median(sat[picked]))
        density_fit = np.clip(
            (mean_density - PLASTIC_DENSITY_MIN) / max(PLASTIC_DENSITY_MIN, 1e-6),
            0.0, 1.0)
        area_fit = np.clip(area / (3.0 * min_area), 0.0, 1.0)
        sat_fit = np.clip(
            (sat_drop - PLASTIC_S_DROP) / max(PLASTIC_S_DROP, 1e-6), 0.0, 1.0)
        evidence = 50.0 + 50.0 * (
            0.40 * float(density_fit) + 0.25 * float(area_fit)
            + 0.35 * float(sat_fit)
        )
        _ = peak_density   # peak is only for watching while tuning, not scored
        results.append(Detection(
            "Plastic Contamination", box, blob, round(evidence, 1)))
    return results


# ============================================================
# Defect 4: Spotting -- many small coloured dots scattered about
# ============================================================
def _find_spots(img, mask_filled, mask_raw, bg_color):
    """Find the small coloured dots that satisfy the definition of a "spot", and
    return [(contour, area, compactness), ...].

    This is a shared helper so that detect_spotting and detect_stains apply the
    *same* criterion: Stain uses it to decide whether a batch of small dots
    should be handed over to Spotting. If the two criteria ever disagreed, a real
    stain could be handed over by Stain and then refused by Spotting, so nothing
    would report it at all.
    """
    h, w = img.shape[:2]
    glove_area = float((mask_filled > 0).sum())
    if glove_area <= 0:
        return []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    erode_ksize = _odd_kernel(STAIN_MASK_ERODE_KSIZE, h, w)
    inside = cv2.erode(mask_raw, np.ones((erode_ksize, erode_ksize), np.uint8)) > 0
    if not inside.any():
        return []

    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    material_lab = np.median(lab_img[inside], axis=0)

    colorful = inside & (sat >= STAIN_COLOR_S_MIN) & (val >= STAIN_COLOR_V_MIN)
    if colorful.sum() < inside.sum() * STAIN_NEUTRAL_RATIO:
        deviation = np.hypot(lab_img[:, :, 1] - material_lab[1],
                             lab_img[:, :, 2] - material_lab[2])
        candidate = inside & (deviation >= STAIN_NEUTRAL_CHROMA_DIST)
    else:
        hist = np.bincount(hue[colorful], minlength=180).astype(np.float32)
        smooth = np.convolve(
            np.r_[hist[-4:], hist, hist[:4]], np.ones(9), mode="valid",
        )
        dominant_hue = int(np.argmax(smooth) % 180)
        raw_delta = np.abs(hue.astype(np.int16) - dominant_hue)
        hue_delta = np.minimum(raw_delta, 180 - raw_delta)
        candidate = (
            inside & (hue_delta >= STAIN_HUE_DIST)
            & (sat >= STAIN_COLOR_S_MIN) & (val >= STAIN_COLOR_V_MIN)
        )

    blob_mask = cv2.morphologyEx(
        candidate.astype(np.uint8) * 255, cv2.MORPH_OPEN,
        np.ones((STAIN_OPEN_KSIZE, STAIN_OPEN_KSIZE), np.uint8),
    )
    # Fragments have to be merged before counting, otherwise the knit texture cuts
    # one whole stain into several pieces, the count passes, and it gets reported
    # as Spotting. Measured, 4 real stains on a cotton photo break into 7-8 pieces.
    close_ksize = _odd_kernel(SPOTTING_CLOSE_KSIZE, h, w)
    if close_ksize >= 3:
        blob_mask = cv2.morphologyEx(
            blob_mask, cv2.MORPH_CLOSE,
            np.ones((close_ksize, close_ksize), np.uint8),
        )
    contours, _ = cv2.findContours(
        blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    max_area = glove_area * SPOTTING_MAX_AREA_RATIO
    spots = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < SPOTTING_MIN_AREA or area > max_area:
            continue          # too small is noise, too large should be a Stain
        perimeter = cv2.arcLength(contour, True)
        compactness = 4.0 * np.pi * area / max(perimeter * perimeter, 1.0)
        if compactness < SPOTTING_MIN_COMPACTNESS:
            continue
        blob = np.zeros(blob_mask.shape, np.uint8)
        cv2.drawContours(blob, [contour], -1, 255, cv2.FILLED)
        pixels = blob > 0
        deviation = float(np.median(np.hypot(
            lab_img[pixels, 1] - material_lab[1],
            lab_img[pixels, 2] - material_lab[2],
        )))
        if deviation < SPOTTING_MIN_CHROMA_DEV:
            continue      # too faint: most likely material texture, not a splash
        bg_distance = float(np.median(np.hypot(
            lab_img[pixels, 1] - bg_color[1],
            lab_img[pixels, 2] - bg_color[2],
        )))
        if bg_distance < SPOTTING_MIN_BG_CHROMA_DIST:
            continue      # it is the background colour: backdrop showing through
                          # at the glove's edge
        spots.append((contour, area, compactness))
    return spots


def detect_spotting(img, mask_filled, mask_raw, bg_color):
    """Many small coloured dots scattered over the glove (splashed paint, say).

    What separates this from Stain is the criterion itself, not just a threshold:
      Stain    asks "is there a region deviating from the material's colour",
               and one or two patches already qualify;
      Spotting asks "are there enough small, vivid dots", and too few does not
               qualify -- one or two isolated dots should be reported by Stain,
               not counted twice as two different defects.
    """
    spots = _find_spots(img, mask_filled, mask_raw, bg_color)
    # Primary criterion: too few dots is not Spotting, so leave it to Stain
    if len(spots) < SPOTTING_MIN_COUNT:
        return []

    results = []
    for contour, area, compactness in spots:
        blob = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(blob, [contour], -1, 255, cv2.FILLED)
        count_fit = min(1.0, len(spots) / (SPOTTING_MIN_COUNT * 2.0))
        evidence = 50.0 + 50.0 * (0.6 * count_fit + 0.4 * compactness)
        results.append(Detection(
            "Spotting", cv2.boundingRect(contour), blob, round(evidence, 1),
        ))
    return results


# ============================================================
# Defect 3: stains
# ============================================================
def _odd_kernel(preferred, h, w):
    """Clamp a morphology / median kernel to the current image size, keeping it odd."""
    size = min(preferred, h, w)
    if size % 2 == 0:
        size -= 1
    return size


def _largest_component(mask):
    """Keep only the largest connected component of a binary image."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return np.zeros_like(mask)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == index).astype(np.uint8) * 255


def _segmentation_is_clean(mask_filled, lab, bg_color):
    """Decide whether the segmentation mask has swallowed some background.

    Method: count how many pixels inside the mask are literally the background
    colour. The glove itself should have no large background-coloured region, so
    a high fraction means the segmentation cannot be trusted.
    """
    mask = mask_filled > 0
    if not mask.any():
        return False
    bg_chroma = np.hypot(lab[:, :, 1] - bg_color[1], lab[:, :, 2] - bg_color[2])
    polluted = mask & (bg_chroma < STAIN_SEG_CLEAN_CHROMA)
    return (polluted.sum() / mask.sum()) <= STAIN_SEG_POLLUTION_MAX


def _region_from_base(base, close_ksize):
    """Rebuild the glove region from pixels of the normal material colour, so a
    carpet or a forearm cannot be taken for a stain."""
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
    """Pick the colour rule from how the material looks, and only look for stains
    inside the reconstructed glove region.

    Light knitted or latex gloves: rebuild the full surface from the normal white
    or grey material, then require a candidate to deviate in chroma from both the
    material *and* the background, and to form a continuous, not-elongated patch.
    That way a large stain is recovered even when foreground segmentation deleted
    it, while a yellow background showing through the mesh is not taken for one.
    Coloured gloves: find the dominant hue first, then the regions deviating from
    it; a strict local Lab distance additionally catches black, white, or
    same-hue-but-much-darker stains. Candidates are still intersected with
    ``mask_raw``, so the background between fingers cannot be enclosed by the
    rebuilt outline.

    Trade-off: white powder on a neutral glove, and very faint or edge-hugging
    small stains, may be missed. Thresholds are calibrated on preprocessed images
    800px wide.
    """
    h, w = img.shape[:2]
    erode_ksize = _odd_kernel(STAIN_MASK_ERODE_KSIZE, h, w)
    base_close_ksize = _odd_kernel(STAIN_BASE_CLOSE_KSIZE, h, w)
    neutral_base_close_ksize = _odd_kernel(STAIN_NEUTRAL_BASE_CLOSE_KSIZE, h, w)
    neutral_region_erode_ksize = _odd_kernel(STAIN_NEUTRAL_REGION_ERODE_KSIZE, h, w)
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
    # Remember which pixels the darkness rule caught; those candidates get a
    # stricter area threshold later on
    dark_source = np.zeros(mask_raw.shape, bool)
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

    # When segmentation is clean, use its outline directly as the search region.
    # The older "rebuild from the material colour" approach pushes a large stain
    # near the glove's edge outside the region -- measured on real photos, it
    # loses 50-94% of the stain pixels. Only when segmentation itself is polluted
    # by background is the rebuild the safer choice.
    seg_clean = _segmentation_is_clean(mask_filled, lab_u8.astype(np.float32), bg_color)
    seg_erode_ksize = _odd_kernel(STAIN_SEG_ERODE_KSIZE, h, w)

    if not colorful_branch:
        if seg_clean:
            glove_region = mask_filled > 0
        else:
            glove_region = _region_from_base(
                neutral_light.astype(np.uint8) * 255, neutral_base_close_ksize,
            )
        if glove_region.any():
            erode_k = seg_erode_ksize if seg_clean else neutral_region_erode_ksize
            glove_inside = cv2.erode(
                glove_region.astype(np.uint8) * 255,
                np.ones((erode_k, erode_k), np.uint8),
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
            # Hue is circular: join the two ends, smooth over 9 bins, take the peak.
            smooth = np.convolve(
                np.r_[hist[-4:], hist, hist[:4]], np.ones(9), mode="valid",
            )
            dominant_hue = int(np.argmax(smooth) % 180)
            raw_delta = np.abs(hue.astype(np.int16) - dominant_hue)
            hue_delta = np.minimum(raw_delta, 180 - raw_delta)
            base = colorful & (hue_delta <= STAIN_BASE_HUE_TOL)
            if seg_clean:
                glove_region = cv2.erode(
                    mask_filled,
                    np.ones((seg_erode_ksize, seg_erode_ksize), np.uint8),
                ) > 0
            else:
                glove_region = _region_from_base(
                    base.astype(np.uint8) * 255, base_close_ksize,
                )
            # Rule 1: hue deviates clearly from the dominant one -- mud, coloured stains
            hue_stain = (
                glove_region & raw_foreground
                & (sat >= STAIN_COLOR_S_MIN) & (val >= STAIN_COLOR_V_MIN)
                & (hue_delta >= STAIN_HUE_DIST)
            )
            # Rule 2: far darker than the material -- black paint and ink, whose
            # hue cannot be trusted
            base_l = float(np.median(lab_u8[base, 0])) if base.any() else 0.0
            l_channel = lab_u8[:, :, 0].astype(np.float32)
            dark_stain = (
                glove_region & raw_foreground
                & (l_channel <= base_l - STAIN_DARK_L_DROP)   # much darker than material
                & (l_channel <= STAIN_DARK_L_ABS)              # and dark in absolute terms
            )
            dark_source |= dark_stain
            stain_pixels = hue_stain | dark_stain

            candidate[stain_pixels] = 255
            hue_strength = np.clip(
                (hue_delta.astype(np.float32) - STAIN_HUE_DIST)
                / max(90.0 - STAIN_HUE_DIST, 1.0),
                0.0, 1.0,
            )
            dark_strength = np.clip(
                (base_l - lab_u8[:, :, 0].astype(np.float32) - STAIN_DARK_L_DROP)
                / max(STAIN_DARK_L_DROP, 1.0),
                0.0, 1.0,
            )
            strength = np.maximum(hue_strength, dark_strength)
            evidence_map[stain_pixels] = 0.55 + 0.45 * strength[stain_pixels]

    if not glove_region.any():
        neutral_base = inside & (sat <= STAIN_NEUTRAL_S_MAX)
        glove_region = _region_from_base(
            neutral_base.astype(np.uint8) * 255, base_close_ksize,
        )

    # First check whether the dominant-hue rule already produced a credible
    # candidate. Once a real stain has been found, do not layer the local-Lab
    # rule on top of it, or normal creases and highlights get marked as extra
    # Stains. Only when the primary rule found nothing does local Lab step in to
    # catch black, white and same-hue dark stains.
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

    final_close_ksize = neutral_close_ksize if not colorful_branch else close_ksize

    # The two rules (hue deviation / darkness) get their own morphology and
    # contour pass; they must not be merged into one mask first.
    # Learned the hard way: on a blue latex glove the darkness rule picks up a
    # thin shadow running down a finger, and closing glues it to the yellow paint
    # beside it into one 86x314 strip. Compactness falls from 0.53 to 0.23, so
    # the shape filter throws a real stain away as an elongated shadow. Kept
    # apart, a thin shadow can only ruin its own blob.
    passes = []
    if colorful_branch and dark_source.any():
        passes.append((cv2.bitwise_and(candidate, (~dark_source).astype(np.uint8) * 255), False))
        passes.append((cv2.bitwise_and(candidate, dark_source.astype(np.uint8) * 255), True))
    else:
        passes.append((candidate, False))

    contours = []
    contour_from_dark = []
    for pass_mask, pass_is_dark in passes:
        pass_mask = cv2.morphologyEx(
            pass_mask, cv2.MORPH_OPEN,
            np.ones((STAIN_OPEN_KSIZE, STAIN_OPEN_KSIZE), np.uint8),
        )
        pass_mask = cv2.morphologyEx(
            pass_mask, cv2.MORPH_CLOSE,
            np.ones((final_close_ksize, final_close_ksize), np.uint8),
        )
        found, _ = cv2.findContours(
            pass_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in found:
            contours.append(contour)
            contour_from_dark.append(pass_is_dark)
        candidate = cv2.bitwise_or(candidate, pass_mask)
    # First decide whether the picture as a whole is a "many small dots" pattern.
    # If it is, those dots belong to Spotting and Stain gives them up of its own
    # accord -- the two detectors are separated by their own criteria rather than
    # relying on deduplicate()'s registration order as a safety net (which also
    # means running Stain alone will not double-report them).
    glove_area = float((mask_filled > 0).sum())
    spot_max_area = glove_area * SPOTTING_MAX_AREA_RATIO
    # Exactly the same criterion detect_spotting uses, so whatever is handed over
    # is guaranteed to be accepted
    is_spotting_pattern = (
        len(_find_spots(img, mask_filled, mask_raw, bg_color)) >= SPOTTING_MIN_COUNT
    )

    results = []
    for contour, pass_is_dark in zip(contours, contour_from_dark):
        if is_spotting_pattern and cv2.contourArea(contour) <= spot_max_area:
            continue      # hand to detect_spotting; large patches stay as Stain
        filled = np.zeros(candidate.shape, np.uint8)
        cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
        blob = cv2.bitwise_and(candidate, filled)
        # Regions caught by the darkness rule get a larger area threshold: finger
        # edges and crease shadows are dark too. Measured, those false alarms are
        # only 660-1449px, while a real stain recognised by its lightness drop
        # starts at 2734px. But a same-hue dark stain is found by the local-Lab
        # fallback and can be genuinely small, so the stricter threshold must not
        # apply to every candidate -- only to what the darkness rule produced.
        blob_mask = blob > 0
        # Now that each rule runs on its own, which rule produced this blob is
        # known exactly; no need to guess it from a pixel-majority vote
        from_dark = pass_is_dark or (
            bool(blob_mask.any())
            and (dark_source & blob_mask).sum() / blob_mask.sum() > 0.5
        )
        min_area = STAIN_COLOR_MIN_AREA if from_dark else MIN_AREA_STAIN
        if cv2.contourArea(contour) < min_area:
            continue
        # The shape filter applies to both branches: crease shadows and glove
        # edges are long thin strips, a real stain is a compact blob. It used to
        # run only in the neutral branch, which is why the crease shadows in the
        # black-paint photos were caught by the darkness rule as a pile of
        # elongated false alarms.
        if True:
            perimeter = cv2.arcLength(contour, True)
            compactness = (
                4.0 * np.pi * cv2.contourArea(contour)
                / max(perimeter * perimeter, 1.0)
            )
            radius = float(cv2.distanceTransform(
                blob, cv2.DIST_L2, 3,
            ).max())
            min_compactness = (
                STAIN_COLOR_MIN_COMPACTNESS if colorful_branch
                else STAIN_NEUTRAL_MIN_COMPACTNESS
            )
            min_radius = (
                STAIN_COLOR_MIN_RADIUS if colorful_branch
                else STAIN_NEUTRAL_MIN_RADIUS
            )
            if compactness < min_compactness or radius < min_radius:
                continue
        scored = evidence_map[(blob > 0) & (evidence_map > 0)]
        evidence = 55.0 if scored.size == 0 else 100.0 * float(np.percentile(scored, 75))
        results.append(Detection(
            "Stain", cv2.boundingRect(contour), blob, round(evidence, 1),
        ))
    return sorted(results, key=lambda result: (result.box[1], result.box[0]))


# ============================================================
# Detector registry: once you have written your function, add its name here
# ============================================================
DETECTORS = [
    detect_holes,
    detect_open_tears,
    # Spotting has to come before Stain: both detectors match the same batch of
    # small dots, and deduplicate() keeps whichever is registered first, so this
    # is what stops them being counted twice.
    detect_spotting,
    detect_stains,
    detect_plastic_contamination,
    # detect_missing_finger,   # e.g. whoever owns "missing finger" adds it here
    # detect_wrinkles,         # e.g. whoever owns "wrinkles" adds it here
]


def _box_iou(a, b):
    """Intersection over union of two bounding boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def deduplicate(defects):
    """The same defect is often reported by several detectors at once (a large
    hole also satisfies "thin area", for instance). Keep whichever comes first in
    the DETECTORS registration order.
    """
    kept = []
    for defect in defects:
        name, box = defect
        if any(
            _box_iou(box, kept_defect.box) > DEDUP_IOU
            for kept_defect in kept
        ):
            continue
        kept.append(defect)
    return kept


def run_all_detectors(img, mask_filled, mask_raw, bg_color, detectors=None):
    """Run the registered detectors in order and return (defects, error messages).

    Each detector gets its own try/except: if one crashes, or returns the wrong
    shape, only that one is skipped and the rest carry on. One broken detector
    out of twelve must not leave the whole system doing nothing when the button
    is pressed -- the demo is worth 10% of the marks, and that is the worst way
    to lose it.

    detectors: omit it to run everything registered in DETECTORS. When a detector
    is unticked in the GUI, a subset containing only the ticked ones is passed in
    (handy for skipping a detector that is not working yet).
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
            errors.append(f"{det.__name__} failed: {e}")
    return deduplicate(defects), errors


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
    """Union of all defect pixels divided by the glove's filled outline, as a
    percentage."""
    glove = glove_mask > 0
    glove_pixels = int(np.count_nonzero(glove))
    if glove_pixels == 0 or not defects:
        return 0.0
    affected = np.zeros(glove.shape, dtype=bool)
    for defect in defects:
        affected |= detection_mask(defect, glove_mask.shape)
    return 100.0 * np.count_nonzero(affected & glove) / glove_pixels


def overall_evidence_score(defects, image_shape):
    """Rule evidence score, weighted by each region's pixel count. Not a
    probability."""
    weighted_sum = 0.0
    total_weight = 0
    for defect in defects:
        weight = int(np.count_nonzero(detection_mask(defect, image_shape)))
        evidence = defect.evidence if isinstance(defect, Detection) else 0.0
        if weight > 0:
            weighted_sum += float(evidence) * weight
            total_weight += weight
    return weighted_sum / total_weight if total_weight else 0.0


# Annotation reference size: a landscape image at the standard 800px width
# counts as scale 1.0.
DRAW_REF_SIZE = 800.0


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


def draw_results(img, defects, alpha=0.38):
    """Draw each defect in its own colour: a translucent pixel region, its
    outline, the bounding box and the evidence score."""
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
        right = min(out.shape[1] - 1, x + text_w + 2 * pad)
        cv2.rectangle(out, (x, top), (right, bottom), color, cv2.FILLED)
        text_color = (20, 20, 20) if name == "Stain" else (255, 255, 255)
        cv2.putText(
            out, label, (x + pad, label_y - 1), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, text_color, thin, cv2.LINE_AA,
        )
    return out

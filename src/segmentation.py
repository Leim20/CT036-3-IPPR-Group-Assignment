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

# --- Bare-skin rejection ------------------------------------------------
# Real photographs are often taken with the glove being WORN, so the hand
# and forearm are in frame and touch the glove. Colour distance from the
# background cannot separate them -- skin is just as "not background" as
# the glove is -- so the arm gets fused into the glove region and every
# shape descriptor computed downstream (convex hull, convex deficiency,
# major axis, compactness) is measured on a glove+arm blob instead.
#
# Skin is separated here in the CIE Lab CHROMA PLANE, using the hue angle
# and chroma magnitude rather than raw a*/b* values. Absolute thresholds
# were tried first and failed: shadow on the background lowers b* into the
# skin band, so shadowed backdrop was classified as skin. Angle and chroma
# are ratios, so they survive the shadow.
#
# Measured on the dataset (patch means):
#     class            chroma   angle
#     skin              22.6     69.4
#     white cotton       8.5     69.3   <- same angle as skin, 2.7x less chroma
#     yellow backdrop   54.7     78.9   <- similar chroma band, 10 deg off in angle
#     yellow in shadow  36.8     82.5
#     blue nitrile      28.9   -106.7   <- nowhere near
# Skin shares an angle with cotton and a chroma band with the backdrop,
# but nothing shares BOTH, so the two criteria are required together.
SKIN_CHROMA = (13.0, 34.0)     # magnitude in the a*b* plane: above cotton, below the backdrop
SKIN_ANGLE = (42.0, 76.0)      # hue angle (degrees) in the a*b* plane: below the backdrop's 79-83
SKIN_KSIZE = 7                 # morphology kernel for cleaning the skin mask
MIN_SKIN_RATIO = 0.02          # ignore a skin mask this small -- probably just noise

# --- Multi-mode background model ---------------------------------------
BG_MODES = 3               # how many dominant background colours to model
BG_QUANT = 16              # Lab quantisation step used to find those modes
BG_MIN_MODE_RATIO = 0.05   # a mode must hold at least this share of the border strip
BG_MODE_MIN_SEP = 70.0     # two modes closer than this in Lab are the same background

# --- Iterative background refinement -----------------------------------
# The border strip is a poor sample of a real background: it is thin, and
# on these photographs it misses the large shadowed areas that sit in the
# middle of the frame around the glove. Shadowed backdrop is then far from
# every sampled mode, measures as "not background", and gets swallowed
# into the glove mask.
#
# So the background model is refined by ITERATION, in the same spirit as
# the Basic Global Thresholding algorithm in Ch 7: segment once, take
# everything the mask now calls background, re-derive the colour modes
# from that much larger and more representative sample, and segment again.
#
# Measured over the 25 labelled photographs: plausible-area masks went
# from 20/25 to 24/25 and median boundary raggedness (P^2/A) fell from 68
# to 60. The worst case, an image whose mask had swallowed 69.7% of the
# frame, came back to 21.7%.
#
# Homomorphic illumination flattening (dividing out a heavily blurred
# copy) was tried first as the shadow fix and made things clearly WORSE --
# plausible masks 20/25 -> 15/25, raggedness 68 -> 246 -- because the
# division amplifies noise across flat regions and weakens exactly the
# colour contrast segmentation depends on. Recorded here so it is not
# retried.
BG_REFINE_ITERS = 2        # 1 disables refinement (border strip only)
BG_REFINE_MODES = 6        # more modes are affordable once the whole background is sampled
BG_REFINE_QUANT = 14
BG_REFINE_MIN_RATIO = 0.03
BG_REFINE_SEP = 30.0
BG_REFINE_SAMPLE = 40000   # cap the pixel sample so the mode search stays cheap

# --- Background hue key -------------------------------------------------
# Lab distance plus Otsu cannot remove a SHADOWED backdrop: shadow moves
# lightness a long way, so shadowed backdrop sits far from every sampled
# background colour and survives as "glove" -- most visibly in the dark
# gaps between the fingers.
#
# Hue barely moves under shadow, and the backdrop here is strongly
# saturated while every glove is not. Measured medians (OpenCV HSV):
#     yellow backdrop, lit        H 21   S 231
#     yellow backdrop, shadowed   H 20   S 160   <- same hue, much darker
#     bare skin                   H 12   S 104
#     white cotton glove          H 14   S  27
#     white floor tile            H 15   S  16
#     blue nitrile glove          H 101  S 160
#     synthetic red backdrop      H 0    S 175
#     synthetic blue glove        H 104  S 177
# So "within a few degrees of the backdrop's own hue AND strongly
# saturated" keys out lit and shadowed backdrop together, while leaving
# skin (8 degrees away), the white gloves and the tile (far too grey), and
# the blue gloves (nowhere near) untouched. The reference hue is measured
# from the border strip rather than hardcoded, so the same rule keys the
# red synthetic background correctly too.
BG_HUE_TOLERANCE = 6        # degrees either side of the backdrop hue (OpenCV scale, 0-180)
BG_HUE_MIN_SAT = 110         # backdrop is saturated; the gloves and tile are not
BG_HUE_KSIZE = 5            # morphology kernel for tidying the key
# The key exists to remove a backdrop that is UNEVENLY LIT. On a backdrop
# of uniform brightness the ordinary Lab distance already handles things,
# and keying adds nothing but a chance to clip something by accident. So
# it is applied only when the keyed region actually varies in lightness.
# Measured std of L inside the keyed region:
#     uniform synthetic backdrop      0.8
#     side-lit synthetic backdrop    31.3
#     the dataset photographs     32.3 - 39.9
# A gate at 10 separates them with a wide margin, and it is what keeps the
# synthetic regression suite unaffected by this stage.
BG_HUE_MIN_LSTD = 10.0

# --- Texture: measured, and deliberately NOT used ----------------------
# Texture separates glove from backdrop cleanly on paper -- local standard
# deviation measured 10.86-15.28 on glove patches against 1.41-7.10 on
# background -- and it is tempting, because the two remaining segmentation
# failures are both colour failures (yellow light reflecting off white
# cotton, and white cotton lying on white floor tile).
#
# It was tried twice and both forms made the SYSTEM worse:
#   * as a veto stopping textured pixels being keyed out ..... F1 0.57 -> 0.52
#   * as a rescue growing the mask into textured pixels ...... F1 0.57 -> 0.29
# The rescue is the instructive one: a tear's frayed edge is textured too,
# so rescuing "missing glove" seals the very openings the detector reads.
# It produced a visibly cleaner, better-covered mask and halved detection
# performance. Cosmetic mask quality and useful mask quality are not the
# same thing, which is worth stating plainly in the report.
#
# texture_map() is kept below because it is genuinely useful for
# inspection and for the write-up, but nothing in the pipeline calls it.             # how far the rescue may grow from trusted glove

# --- Repairing holes punched through the glove --------------------------
# Knit cotton has enough local variation that the Otsu cut drops scattered
# patches of genuine glove, leaving the mask riddled with small holes. On
# the worst image only 75% of known glove area survived. Those holes are
# indistinguishable from real punctures BY SIZE -- measured, a real defect
# hole runs 0.0005-0.0009 of glove area and the noise holes 0.0004-0.003,
# completely overlapping.
#
# What does separate them is what is visible INSIDE the hole. A noise hole
# is glove that was misclassified, so it still looks like glove; a real
# opening shows skin or backdrop. Lab distance from the glove's own colour,
# measured inside each hole:
#     noise holes  p10 18.4   p50 29.3   p90  85.1
#     real holes   p10 40.5   p50 79.9   p90 124.5
# A cut at 30 fills 127 of 238 noise holes and destroys NONE of the 10
# real ones.
HOLE_REPAIR_MAX_DIST = 30.0   # holes whose interior is closer than this to glove colour get filled
HOLE_REPAIR_MAX_AREA = 0.01   # ...and are no bigger than this fraction of the glove

BORDER_RATIO = 0.06                          # width of the border strip sampled as background
MIN_AREA_RATIO, MAX_AREA_RATIO = 0.05, 0.95  # plausible range for the glove's area fraction
CLOSE_KSIZE = 7   # morphology closing kernel: fills small gaps, larger is fine
OPEN_KSIZE = 3    # morphology opening kernel: removes noise, must stay small.
                  # Swept over 3/5/7/9 against the 25 labelled photographs:
                  #   k=3  35/35 synth, 10/10 tear, 23/25 plausible, ragged 58, 38/43 defects kept
                  #   k=5  34/35,  8/10,           23/25,            ragged 57, 37/43
                  #   k=7  34/35,  8/10,           23/25,            ragged 72, 36/43
                  #   k=9  34/35,  8/10,           23/25,            ragged 70, 34/43
                  # Anything above 3 erodes thin tears away, which is the one
                  # thing this mask must not do.


def skin_mask(img):
    """Bare hand / forearm pixels, as a binary mask.

    Skin-colour segmentation is demonstrated in the Ch 7 segmentation
    lecture ("Segmented image, skin colour is shown"); this is the same
    idea moved into the Lab chroma plane so it survives the shadows that
    real photographs contain.

    Returns an all-zero mask when the result is implausibly small, so
    that images with no bare skin in them (and the synthetic regression
    images) are left completely untouched.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    chroma = np.hypot(a, b)
    angle = np.degrees(np.arctan2(b, a))

    lo_c, hi_c = SKIN_CHROMA
    lo_a, hi_a = SKIN_ANGLE
    mask = ((chroma >= lo_c) & (chroma <= hi_c) &
            (angle >= lo_a) & (angle <= hi_a)).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SKIN_KSIZE,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)    # drop speckle
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)   # close the arm into one region

    # Structural guard: keep only skin components that TOUCH THE IMAGE
    # BORDER. A forearm is attached to a body outside the frame, so it
    # always enters from an edge; a glove lying inside the frame does not
    # have to. Without this guard a tan or leather glove -- which really
    # is skin-coloured -- gets classified as skin and deleted, which is
    # exactly what happened to the leather sample in the regression suite.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    h, w = mask.shape
    kept = np.zeros_like(mask)
    for i in range(1, n):
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
            kept[labels == i] = 255

    if kept.mean() / 255 < MIN_SKIN_RATIO:
        return np.zeros_like(kept)
    return kept


def texture_map(img, window=None):
    """Local standard deviation of intensity -- high on woven or moulded
    glove surfaces, low on a painted seat or a floor tile."""
    k = window or TEXTURE_WINDOW
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(g, (k, k))
    mean_sq = cv2.blur(g * g, (k, k))
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def textured_mask(img):
    """Pixels textured enough that they cannot be background."""
    t = texture_map(img)
    hi = float(np.percentile(t, 95))
    thresh = max(TEXTURE_VETO_FRAC * hi, TEXTURE_VETO_MIN)
    return (t >= thresh).astype(np.uint8) * 255


def background_key(img, skin=None):
    """Pixels that match the backdrop's own hue at high saturation.

    Returns a binary mask of "definitely background", which segmentation
    subtracts before thresholding. See the note on BG_HUE_TOLERANCE above
    for the measured colour statistics this is built on.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1].astype(np.int16)

    if skin is None:
        skin = skin_mask(img)
    H, W = h.shape
    bh, bw = max(int(H * BORDER_RATIO), 1), max(int(W * BORDER_RATIO), 1)
    strips = [(slice(0, bh), slice(None)), (slice(H - bh, H), slice(None)),
              (slice(None), slice(0, bw)), (slice(None), slice(W - bw, W))]
    keep = (skin == 0) & (sat >= BG_HUE_MIN_SAT)
    sample = np.concatenate([h[ys, xs][keep[ys, xs]] for ys, xs in strips])
    if len(sample) < 200:
        return np.zeros_like(skin)          # no saturated backdrop: nothing to key

    # circular median of the border hue
    ang = np.radians(sample.astype(np.float32) * 2.0)
    ref = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360.0) / 2.0

    diff = np.abs(h - ref)
    diff = np.minimum(diff, 180 - diff)     # hue is circular
    key = ((diff <= BG_HUE_TOLERANCE) & (sat >= BG_HUE_MIN_SAT)).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BG_HUE_KSIZE,) * 2)
    key = cv2.morphologyEx(key, cv2.MORPH_OPEN, k)
    key = cv2.morphologyEx(key, cv2.MORPH_CLOSE, k)

    if not key.any():
        return key
    lightness = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    if float(lightness[key > 0].std()) < BG_HUE_MIN_LSTD:
        return np.zeros_like(key)          # evenly lit backdrop: key not needed
    return key


def _border_pixels(img, skin=None):
    """Lab pixels from the four border strips, with skin excluded.

    Skin is dropped because when the glove is worn the forearm usually
    runs out of the frame, so it lands in the border strip and drags the
    "background" reference towards skin colour. Measured on one dataset
    image, the bottom border strip was 52.9% skin.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    bh, bw = max(int(h * BORDER_RATIO), 1), max(int(w * BORDER_RATIO), 1)

    if skin is None:
        skin = skin_mask(img)
    keep = skin == 0
    strips = [(slice(0, bh), slice(None)), (slice(h - bh, h), slice(None)),
              (slice(None), slice(0, bw)), (slice(None), slice(w - bw, w))]
    border = np.concatenate([lab[ys, xs][keep[ys, xs]] for ys, xs in strips])
    if len(border) < 50:                      # skin filled the whole border: fall back
        border = np.concatenate([lab[ys, xs].reshape(-1, 3) for ys, xs in strips])
    return border


def _modes_of(pixels, k, quant, min_ratio, sep):
    """Dominant colours of a pixel set, by coarse histogram peak search."""
    q = np.floor(pixels / quant).astype(np.int32)
    _, inverse, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    modes = []
    for i in np.argsort(counts)[::-1]:
        if counts[i] < min_ratio * len(pixels):
            continue
        c = pixels[inverse == i].mean(axis=0)
        if any(np.linalg.norm(c - m) < sep for m in modes):
            continue
        modes.append(c)
        if len(modes) >= k:
            break
    if not modes:
        modes = [np.median(pixels, axis=0)]
    return np.array(modes, np.float32)


def get_background_colors(img, skin=None):
    """The dominant background colours (Lab), as an (N, 3) array.

    A single median cannot describe a background made of more than one
    thing. These photographs have a yellow seat AND grey floor tile in the
    border strip; the median of the two is a colour that exists nowhere in
    the image, so BOTH real backgrounds then measure as "far from
    background" and get segmented as glove.

    Ch 7 covers exactly this case under multimodal histograms: when the
    histogram has several dominant modes it must be partitioned with
    several thresholds rather than one. Here the modes are found by
    quantising the border colours onto a coarse Lab grid and keeping the
    most populated cells -- a histogram peak search, not a clustering
    algorithm, so no learning is involved.
    """
    # A smooth brightness gradient across ONE background splits into
    # several quantised cells, so modes are required to be well separated:
    # a uniform-but-unevenly-lit backdrop stays a single mode, while a
    # genuinely two-material backdrop (seat + floor tile) still yields two.
    return _modes_of(_border_pixels(img, skin), BG_MODES, BG_QUANT,
                     BG_MIN_MODE_RATIO, BG_MODE_MIN_SEP)


def background_distance(lab, bg_colors):
    """Distance from every pixel to the NEAREST background colour."""
    bg_colors = np.atleast_2d(bg_colors)
    d = np.stack([np.linalg.norm(lab - c, axis=2) for c in bg_colors])
    return d.min(axis=0)


def get_background_color(img, skin=None):
    """The single most common background colour (Lab).

    Kept as-is so existing detectors and evaluate.py continue to work
    unchanged; segmentation itself uses the multi-mode version above.
    """
    return get_background_colors(img, skin)[0]


def _threshold_against(lab, bg_colors, skin, bg_key):
    """One segmentation pass against a given background colour model."""
    dist = background_distance(lab, bg_colors)
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # A texture-based RESCUE of glove pixels that failed this cut was
    # tried here and removed: it doubles back on itself, because the
    # frayed edge of a tear is textured too, so rescuing "missing glove"
    # also seals the tear openings the detector depends on. Measured, it
    # halved detection F1 from 0.52 to 0.29 while improving the mask's
    # cosmetic coverage -- a clear case where the prettier mask is the
    # worse input.

    # Remove bare skin BEFORE the largest-component step. Order matters:
    # if the arm is still attached when we pick the largest component, the
    # glove and the arm are one component and there is nothing left to
    # separate afterwards.
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(bg_key))

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KSIZE,) * 2)
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KSIZE,) * 2)
    # open first, then close: opening severs the thin skin-coloured bridge
    # that can survive at the cuff, and doing it before closing stops that
    # bridge being sealed back up again
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

    # keep only the largest connected region, and fill it in
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_filled, [biggest], -1, 255, cv2.FILLED)
    return mask_filled, cv2.bitwise_and(mask, mask_filled)


def _repair_holes(img, mask_filled, mask_raw):
    """Fill holes that are just misclassified glove, keep the real ones.

    See the note on HOLE_REPAIR_MAX_DIST above for the measured statistics
    this separation is built on.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    inner = cv2.erode(mask_raw, np.ones((9, 9), np.uint8)) > 0
    if inner.sum() < 300:
        return mask_raw
    glove_color = np.median(lab[inner], axis=0)

    holes = cv2.subtract(mask_filled, mask_raw)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
    glove_area = max(mask_filled.sum() / 255.0, 1.0)
    repaired = mask_raw.copy()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] / glove_area > HOLE_REPAIR_MAX_AREA:
            continue                       # too big to be texture noise
        component = labels == i
        if np.linalg.norm(lab[component].mean(axis=0) - glove_color) < HOLE_REPAIR_MAX_DIST:
            repaired[component] = 255      # this was glove all along
    return repaired


def segment_glove(img):
    """Return (mask_filled, mask_raw):
    - mask_raw   : the glove's actual pixels (holes are black, since a
                   hole reveals the background)
    - mask_filled: the glove's full outline filled in (holes are filled
                   white too)
    Subtracting the two = "inside the outline, but coloured like the
    background" -> hole candidates.

    Runs BG_REFINE_ITERS passes: the first uses the border strip as the
    background sample, and each further pass re-derives the background
    colours from everything the previous mask called background. That is
    what lets large shadowed areas -- which never touch the border strip --
    join the background model instead of being swallowed by the glove.
    """
    skin = skin_mask(img)
    bg_key = background_key(img, skin)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_colors = get_background_colors(img, skin)

    mask_filled, mask_raw = _threshold_against(lab, bg_colors, skin, bg_key)

    # Guard: if the backdrop's hue happens to be close to the GLOVE's hue,
    # the key removes the glove instead of the background and the result
    # collapses. Detect that by the mask becoming implausibly small, and
    # fall back to segmenting without the key.
    if bg_key.any() and mask_filled.mean() / 255 < MIN_AREA_RATIO:
        bg_key = np.zeros_like(bg_key)
        mask_filled, mask_raw = _threshold_against(lab, bg_colors, skin, bg_key)

    rng = np.random.default_rng(0)     # fixed seed: segmentation stays reproducible
    for _ in range(max(BG_REFINE_ITERS - 1, 0)):
        outside = lab[(mask_filled == 0) & (skin == 0)]
        if len(outside) < 500:
            break
        if len(outside) > BG_REFINE_SAMPLE:
            outside = outside[rng.choice(len(outside), BG_REFINE_SAMPLE, replace=False)]
        bg_colors = _modes_of(outside, BG_REFINE_MODES, BG_REFINE_QUANT,
                              BG_REFINE_MIN_RATIO, BG_REFINE_SEP)
        mask_filled, mask_raw = _threshold_against(lab, bg_colors, skin, bg_key)

    return mask_filled, _repair_holes(img, mask_filled, mask_raw)


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

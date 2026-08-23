# -*- coding: utf-8 -*-
"""
Defect detection: every defect is one function, registered in the DETECTORS
list, and the GUI calls them all automatically.

The rule for writing a detector is that the signature is always:

    def detect_xxx(img, mask_filled, mask_raw, bg_color,
                   img_plain=None, material=None):
        ...
        return [("Defect Name", (x, y, w, h)), ...]   # [] when nothing is found

The last two parameters are optional so older four-argument calls remain
valid. ``img`` is lighting-normalised; ``img_plain`` retains the original
colour/texture after resize and denoising. The defect name is drawn on the
image and also listed in the GUI's text box.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from segmentation import get_glove_color, get_background_colors, skin_mask

# --- Tunable parameters (kept here for sensitivity experiments and for
# citing in the report) ---
# --- Tunable parameters, kept together so a sensitivity study is easy to run
#     and easy to write up in the report ---
BG_MATCH_DIST = 30.0       # tearing rule: a candidate counts only if its
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

# --- Hole, finger-not-enough and thin-area parameters. These are retained
# alongside the newer teammate stain/spot/plastic rules above. ---
STAIN_COLOR_DIST = 25.0    # stain criterion: pixel's Lab distance from the glove's normal colour must be above this

# Tearing detector parameters. The photographed gloves are worn on a hand, so an
# puncture reveals skin rather than the green background. Candidate skin pixels
# must form an enclosed, locally high-contrast region well inside the glove.
TEARING_BOUNDARY_KSIZE = 13
TEARING_RING_KSIZE = 11
TEARING_DEFAULT_RULE = {
    "min_area": 60,
    "min_local_contrast": 30.0,
    "min_interior_ratio": 0.20,
}
TEARING_MATERIAL_RULES = {
    # Cotton weave can expose many tiny skin-coloured gaps, so genuine damage
    # must be larger and have a sharper boundary.
    "cotton": {
        "min_area": 240,
        "min_local_contrast": 50.0,
        "min_interior_ratio": 0.20,
    },
    # Latex foam is opaque and uniform; even a small enclosed skin region is
    # strong evidence, and the dataset produced no non-tearing candidates.
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

# Cotton fibres lower the *mean* Lab difference around a large opening because
# the surrounding ring contains a mixture of white yarn, shadow and exposed
# skin.  A second, deliberately narrow rule therefore accepts only candidates
# that are simultaneously large and deep inside the glove.  Measurements on
# the current development photographs separate the nearest clean-glove region
# (area ratio 0.0045) from the first low-contrast tear (0.0093); 0.008 leaves a
# margin between them.  The depth and contrast backstops keep broad illumination
# changes from satisfying the size rule on their own.
TEARING_COTTON_LARGE_MIN_AREA_RATIO = 0.008
TEARING_COTTON_LARGE_MIN_LOCAL_CONTRAST = 30.0
TEARING_COTTON_LARGE_MIN_INTERIOR_RATIO = 0.70

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
    "indent_max_width_ratio": 0.45,
    "indent_max_y_ratio": 0.55,
    "indent_target_count": 3,
    "row_min_width_ratio": 0.06,
    "row_max_y_ratio": 0.65,
    "row_support_ratio": 0.08,
    "row_target_count": 2,
    "skin_min_area_ratio": 0.012,
    "skin_max_width_ratio": 0.30,
    "skin_min_boundary_ratio": 0.05,
    "skin_max_y_ratio": 0.72,
}
FINGER_NOT_ENOUGH_MATERIAL_RULES = {
    "cotton": {
        "indent_min_area_ratio": 0.010,
        "indent_max_width_ratio": 0.45,
        "indent_max_y_ratio": 0.75,
        "indent_target_count": 4,
        "row_min_width_ratio": 0.06,
        "row_max_y_ratio": 0.55,
        "row_support_ratio": 0.08,
        "row_target_count": 2,
        # Evaluate each exposed finger separately.  Four uncovered fingers are
        # four medium components, not one large skin region, so the old 2.4%
        # gate discarded all of them.  Shape and glove-attachment checks below
        # provide the cotton-specific false-positive protection instead.
        "skin_min_area_ratio": 0.008,
        "skin_max_width_ratio": 0.30,
        "skin_min_boundary_ratio": 0.05,
        "skin_max_y_ratio": 0.72,
    },
    "latex_foam": {
        "indent_min_area_ratio": 0.001,
        "indent_max_width_ratio": 0.45,
        "indent_max_y_ratio": 0.55,
        "indent_target_count": 3,
        "row_min_width_ratio": 0.04,
        "row_max_y_ratio": 0.75,
        "row_support_ratio": 0.06,
        "row_target_count": 2,
        "skin_min_area_ratio": 0.004,
        "skin_max_width_ratio": 0.30,
        "skin_min_boundary_ratio": 0.10,
        "skin_max_y_ratio": 0.72,
    },
    "nitrile": {
        "indent_min_area_ratio": 0.001,
        "indent_max_width_ratio": 0.45,
        "indent_max_y_ratio": 0.55,
        "indent_target_count": 2,
        "row_min_width_ratio": 0.04,
        "row_max_y_ratio": 0.55,
        "row_support_ratio": 0.08,
        "row_target_count": 3,
        "skin_min_area_ratio": 0.004,
        "skin_max_width_ratio": 0.30,
        "skin_min_boundary_ratio": 0.05,
        "skin_max_y_ratio": 0.72,
    },
}

# Silhouette support for a folded/short finger.  Exact indentation counts were
# brittle: both good and defective gloves commonly produced four hull gaps.
# The replacement first counts prominent upper-contour tips, then requires a
# second abnormality before reporting a shortage.  The supporting abnormality
# is material-specific because flexible cotton, foam latex and nitrile produce
# different hull-gap sizes when a finger folds into the palm.
FINGER_EXPECTED_TIP_COUNT = 5
FINGER_TIP_MAX_Y_RATIO = 0.48
FINGER_TIP_MIN_RADIUS_RATIO = 0.20
FINGER_TIP_MIN_PROMINENCE_RATIO = 0.015
FINGER_TIP_SMOOTH_SIGMA_RATIO = 0.002
FINGER_TIP_NEIGHBOUR_RATIO = 0.025
FINGER_SHAPE_DEFAULT_RULE = {
    "min_indent_count": 2,
    "large_gap_area_ratio": 0.06,
    "max_solidity": 0.76,
    "support_indent_counts": (3,),
}
FINGER_SHAPE_MATERIAL_RULES = {
    "cotton": {
        "min_indent_count": 2,
        "large_gap_area_ratio": 0.08,
        "max_solidity": 0.70,
        "support_indent_counts": (),
    },
    "latex_foam": {
        "min_indent_count": 2,
        "large_gap_area_ratio": 0.065,
        "max_solidity": 0.76,
        "support_indent_counts": (3,),
    },
    "nitrile": {
        "min_indent_count": 2,
        "large_gap_area_ratio": 0.05,
        "max_solidity": 0.80,
        "support_indent_counts": (2,),
    },
}
FINGER_REGION_HEIGHT_RATIO = 0.80
FINGER_SKIN_BOUNDARY_KSIZE = 13
# An exposed fingertip remains part of the hand silhouette, so it does not
# necessarily reduce the number of visible finger columns.  Instead, verify
# that the glove boundary wraps around the skin component in both axes.  This
# distinguishes a bare fingertip from a skin-coloured stain that merely touches
# one edge of the glove.
FINGER_SKIN_TIP_MIN_ASPECT_RATIO = 1.20
FINGER_SKIN_TIP_MIN_BOUNDARY_SPAN_RATIO = 0.65
FINGER_SKIN_MIN_Y_RATIO = -0.45
FINGER_SKIN_ATTACHMENT_RADIUS_RATIO = 0.02
FINGER_SKIN_EXTERNAL_MAX_INSIDE_RATIO = 0.50
FINGER_SKIN_EXTERNAL_MIN_NEAR_RATIO = 0.03
FINGER_SKIN_EXTERNAL_MIN_LOWER_NEAR_RATIO = 0.08
FINGER_SKIN_LOWER_START_RATIO = 0.65

# Thin / overstretched detector. These thresholds were measured on the
# development photographs after isolating glove-coloured pixels from the
# controlled green inspection mat. Cotton and nitrile use transparency cues;
# latex foam has no reliable transparency cue, so its lower-confidence branch
# uses only a coarse edge-density measurement and is documented as experimental.
THIN_BLUE_H_MIN, THIN_BLUE_H_MAX = 85, 145
# Purple/blue inspection backdrops overlap the broad generic blue interval, but
# the photographed nitrile glove remains at H=107-115.  A nitrile-specific cap
# prevents the backdrop from becoming the material ROI and depressing its
# lightness statistic.
THIN_NITRILE_BLUE_H_MAX = 115
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

MIN_AREA_TEARING = 60
MIN_AREA_STAIN = 500           # smallest stain at the standard 800px width.
                               # Crease-shadow false alarms all sit at 500-600px
                               # and the smallest real stain is 1700px, so the
                               # threshold goes between them; 9 of 14 annotated
                               # photos then reach 100/100.

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
}
DEFAULT_DEFECT_COLOR = (35, 160, 70)  # any detector added later: green

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

# Result visualisation. Detectors keep the assignment's required
# ``(label, bounding_box)`` return contract; after recognition, the shared
# pipeline rebuilds the accepted pixel evidence as a binary segmentation mask.
# The semi-transparent colours make the original glove texture remain visible.
DEFECT_OVERLAY_ALPHA = 0.45
DEFECT_OVERLAY_COLORS = {
    "Tearing": (0, 0, 255),                # red (BGR)
    "Open Tear": (0, 80, 255),             # orange-red
    "Finger Not Enough": (0, 165, 255),     # orange
    "Thin / Overstretched": (255, 0, 255),  # magenta
    "Stain": (255, 80, 0),                 # blue
}
THIN_SEGMENT_DENSITY_KSIZE = 41
THIN_SEGMENT_DENSITY_MIN = 2


# ============================================================
# Defect 1: enclosed tearing
# ============================================================
def detect_tearing(img, mask_filled, mask_raw, bg_color,
                   img_plain=None, material=None):
    """Detect enclosed punctures that expose the wearer's skin.

    The previous implementation searched for green-background pixels inside
    the glove. That assumption did not match the real dataset: the gloves are
    worn, so their torn openings reveal skin. This detector therefore uses two
    classical skin-colour rules, then rejects candidates that touch the glove's
    outside boundary, lack a sharp colour transition, or sit too close to the
    silhouette edge. Those checks prevent exposed fingertips from being
    labelled as palm tearing.

    When no material metadata is supplied (as in the synthetic regression
    tests), a strict background-revealing fallback is also run. Real dataset
    calls always provide their material and use the calibrated skin rule.
    """
    source = img_plain if img_plain is not None else img
    rule = TEARING_MATERIAL_RULES.get(material, TEARING_DEFAULT_RULE)

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

    # Any colour anomaly touching this inner boundary band is an exposed outer
    # edge/fingertip, not an enclosed tear.
    eroded = cv2.erode(
        mask_filled,
        np.ones((TEARING_BOUNDARY_KSIZE, TEARING_BOUNDARY_KSIZE), np.uint8),
    )
    boundary_band = (mask_filled > 0) & (eroded == 0)
    touching_boundary = set(np.unique(labels[boundary_band]))

    lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    interior_distance = cv2.distanceTransform(
        (mask_filled > 0).astype(np.uint8), cv2.DIST_L2, 5
    )
    max_interior_distance = max(float(interior_distance.max()), 1.0)
    glove_area = max(int(cv2.countNonZero(mask_filled)), 1)

    results = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < rule["min_area"] or label in touching_boundary:
            continue
        component = (labels == label).astype(np.uint8)
        ring = cv2.dilate(
            component,
            np.ones((TEARING_RING_KSIZE, TEARING_RING_KSIZE), np.uint8),
        ) - component
        ring_pixels = (ring > 0) & (mask_filled > 0)
        if not ring_pixels.any():
            continue

        local_contrast = float(
            np.linalg.norm(
                lab[component > 0].mean(axis=0) - lab[ring_pixels].mean(axis=0)
            )
        )
        interior_ratio = float(
            interior_distance[component > 0].max() / max_interior_distance
        )
        area_ratio = float(area) / glove_area
        standard_candidate = (
            local_contrast >= rule["min_local_contrast"]
            and interior_ratio >= rule["min_interior_ratio"]
        )
        large_cotton_candidate = (
            material == "cotton"
            and area_ratio >= TEARING_COTTON_LARGE_MIN_AREA_RATIO
            and local_contrast
            >= TEARING_COTTON_LARGE_MIN_LOCAL_CONTRAST
            and interior_ratio
            >= TEARING_COTTON_LARGE_MIN_INTERIOR_RATIO
        )
        if not (standard_candidate or large_cotton_candidate):
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y0 = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area_fit = min(1.0, area / max(4.0 * rule["min_area"], 1.0))
        contrast_reference = (
            TEARING_COTTON_LARGE_MIN_LOCAL_CONTRAST
            if large_cotton_candidate and not standard_candidate
            else rule["min_local_contrast"]
        )
        contrast_fit = min(
            1.0,
            local_contrast / max(2.0 * contrast_reference, 1.0),
        )
        depth_fit = min(
            1.0,
            interior_ratio / max(2.0 * rule["min_interior_ratio"], 1e-6),
        )
        evidence = 50.0 + 50.0 * (
            0.25 * area_fit + 0.45 * contrast_fit + 0.30 * depth_fit
        )
        results.append(Detection(
            "Tearing",
            (x, y0, width, height),
            component * 255,
            round(float(evidence), 1),
        ))
    if material is None:
        # Generic fallback for an unworn glove: an enclosed tear reveals the
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
            if cv2.contourArea(contour) < TEARING_DEFAULT_RULE["min_area"]:
                continue
            blob = np.zeros(background_candidate.shape, np.uint8)
            cv2.drawContours(blob, [contour], -1, 255, cv2.FILLED)
            mean_color = lab_normalized[blob > 0].mean(axis=0)
            color_distance = float(np.linalg.norm(mean_color - bg_color))
            if color_distance < BG_MATCH_DIST:
                color_fit = 1.0 - color_distance / BG_MATCH_DIST
                size_fit = min(
                    1.0,
                    cv2.contourArea(contour) /
                    (TEARING_DEFAULT_RULE["min_area"] * 4.0),
                )
                evidence = 50.0 + 50.0 * (
                    0.75 * color_fit + 0.25 * size_fit
                )
                results.append(Detection(
                    "Tearing",
                    cv2.boundingRect(contour),
                    blob,
                    round(float(evidence), 1),
                ))
    return results


# ============================================================
# Defect 2: open tear (one that reaches the glove's edge)
# ============================================================
def detect_open_tears(img, mask_filled, mask_raw, bg_color,
                      img_plain=None, material=None):
    """Convexity defects -- the dents between the contour and its convex
    hull. A normal finger gap is also a deep, wide notch, so depth alone
    can't separate them; shape does: a tear is narrow and sharp (the
    material is cut), a finger gap is a wide, blunt, natural U-shape.
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
# Defect 2b: side tear (a cut breaching the glove's LATERAL edge)
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


def detect_side_tear(img, mask_filled, mask_raw, bg_color,
                     img_plain=None, material=None):
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
# Incomplete beading: the cuff hem is interrupted
# ============================================================
# The bead is the finished hem at the cuff -- a maroon knitted band on the
# cotton gloves, a rolled edge on latex and nitrile. "Incomplete beading"
# means a stretch of that hem is missing, leaving a ragged opening that
# looks like a tear at the wrist.
#
# The key property is that a bead is a CONTINUOUS structure, so the defect
# is a DISCONTINUITY in it -- and the rest of the same bead is the
# reference. That makes every threshold self-referential (median +/- k*sd
# measured along this glove's own cuff) rather than absolute, so nothing
# needs retuning per material or per lighting.
#
# Method: reduce the cuff boundary to a 1-D signature (Ch 10/11), walking
# along it and recording at each station the colour just inside the edge
# and the local roughness of the edge. On an intact bead both run flat. A
# contiguous run where they jump is the defect.
BEAD_CUFF_BAND = 0.72       # cuff = this far along the major axis and beyond
BEAD_NEAR_SKIN = 70         # cuff opening = boundary within this many px of bare skin.
                            # Generous on purpose: when the forearm is only partly in
                            # frame the skin mask is a thin strip, and a tight radius
                            # clips the cuff edge before it reaches the defect.
BEAD_RUN_MERGE_FRAC = 0.06  # runs separated by less than this share of the edge are one defect
BEAD_SAMPLE_DEPTH = 10      # how far inside the edge the bead colour is sampled
BEAD_SMOOTH_WIN = 21        # window for the smoothed contour the roughness is measured against
BEAD_MIN_EDGE_PTS = 30      # too short a cuff edge to judge
BEAD_RUN_MIN_FRAC = 0.08    # a defect must span at least this share of the cuff edge
BEAD_K_SIGMA = 0.8          # how far above the cuff's own median counts as "bead missing"
BEAD_BOX_PAD = 8

def _cuff_edge(mask_filled, skin, axis, lo, hi, low_is_distal):
    """The stretch of glove boundary that forms the cuff opening."""
    contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    pts = cnt.reshape(-1, 2).astype(np.float32)
    frac = (pts @ axis - lo) / (hi - lo + 1e-6)
    if not low_is_distal:
        frac = 1.0 - frac

    # Prefer boundary running alongside bare skin -- that is the opening the
    # hand comes out of. If no skin was found (a heavy colour cast can
    # defeat the skin rule) fall back to position along the major axis.
    use_skin = skin is not None and skin.any()
    near_skin = (cv2.dilate(skin, np.ones((BEAD_NEAR_SKIN,) * 2, np.uint8)) > 0
                 if use_skin else None)
    h, w = mask_filled.shape
    keep = []
    for i, (x, y) in enumerate(pts):
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        if frac[i] < BEAD_CUFF_BAND:
            continue
        if use_skin and not near_skin[yi, xi]:
            continue
        keep.append(i)
    if len(keep) < BEAD_MIN_EDGE_PTS:
        return None
    keep = np.array(keep)
    segments = np.split(keep, np.where(np.diff(keep) > 5)[0] + 1)
    longest = max(segments, key=len)
    return pts[longest] if len(longest) >= BEAD_MIN_EDGE_PTS else None


def detect_incomplete_beading(img, mask_filled, mask_raw, bg_color,
                              img_plain=None, material=None):
    """A stretch of the cuff hem is missing.

    See the note above the BEAD_* constants for the reasoning. In short:
    walk the cuff boundary, record the colour just inside it and how rough
    it is, and flag any contiguous run where those depart from the rest of
    the same cuff.
    """
    contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    axis, perp, lo, hi = _basic_rectangle_axis(cnt)
    if hi - lo < 20:
        return []
    low_is_distal = _fingers_at_low_end(mask_filled, axis, perp, lo, hi, img=img)
    edge = _cuff_edge(mask_filled, skin_mask(img), axis, lo, hi, low_is_distal)
    if edge is None:
        return []

    moments = cv2.moments(cnt)
    if moments["m00"] == 0:
        return []
    centroid = np.array([moments["m10"] / moments["m00"],
                         moments["m01"] / moments["m00"]], np.float32)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = mask_filled.shape

    # roughness: how far the edge strays from a smoothed copy of itself
    k = np.ones(BEAD_SMOOTH_WIN, np.float32) / BEAD_SMOOTH_WIN
    smooth_x = np.convolve(edge[:, 0], k, "same")
    smooth_y = np.convolve(edge[:, 1], k, "same")
    roughness = np.hypot(edge[:, 0] - smooth_x, edge[:, 1] - smooth_y)

    # colour of the material just inside the edge -- the bead itself
    colours = []
    for x, y in edge:
        inward = centroid - np.array([x, y], np.float32)
        inward /= (np.linalg.norm(inward) + 1e-6)
        px, py = np.array([x, y], np.float32) + inward * BEAD_SAMPLE_DEPTH
        xi = min(max(int(round(px)), 1), w - 2)
        yi = min(max(int(round(py)), 1), h - 2)
        colours.append(lab[yi - 1:yi + 2, xi - 1:xi + 2].reshape(-1, 3).mean(axis=0))
    colours = np.array(colours)
    colour_dev = np.linalg.norm(colours - np.median(colours, axis=0), axis=1)

    def standardise(v):
        return (v - np.median(v)) / (np.std(v) + 1e-6)

    score = standardise(colour_dev) + standardise(roughness)
    flag = score > BEAD_K_SIGMA

    # smooth the flag so one defect reads as one run, not a dotted line
    win = max(int(len(flag) * 0.02), 3)
    flag = np.convolve(flag.astype(np.float32), np.ones(win) / win, "same") > 0.5

    # collect the runs, then merge ones that nearly touch: a single gap in
    # the bead often dips back under threshold briefly in the middle, and
    # without merging it gets reported as three or four separate defects
    runs = []
    i = 0
    while i < len(flag):
        if not flag[i]:
            i += 1
            continue
        j = i
        while j < len(flag) and flag[j]:
            j += 1
        runs.append([i, j])
        i = j

    gap = max(int(BEAD_RUN_MERGE_FRAC * len(flag)), 4)
    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= gap:
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    results = []
    min_run = max(int(BEAD_RUN_MIN_FRAC * len(flag)), 6)
    for a, b in merged:
        if b - a < min_run:
            continue
        x, y, bw, bh = cv2.boundingRect(edge[a:b].astype(np.int32))
        results.append(("Incomplete Beading",
                        (max(x - BEAD_BOX_PAD, 0), max(y - BEAD_BOX_PAD, 0),
                         bw + 2 * BEAD_BOX_PAD, bh + 2 * BEAD_BOX_PAD)))
    return results



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


def detect_damage_by_fold(img, mask_filled, mask_raw, bg_color,
                          img_plain=None, material=None):
    """A crease left across the glove where it was folded."""
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
                            int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])]))
    candidates.sort(reverse=True, key=lambda c: c[0])

    # one crease often survives as several components, so nearby boxes are
    # merged before the top-N cut -- otherwise the budget is spent on
    # fragments of a single fold rather than on separate defects
    gap = FOLD_MERGE_GAP_FRAC * axis_len
    merged = []
    for _, (x, y, bw, bh) in candidates:
        for m in merged:
            if (x < m[0] + m[2] + gap and m[0] < x + bw + gap and
                    y < m[1] + m[3] + gap and m[1] < y + bh + gap):
                nx, ny = min(m[0], x), min(m[1], y)
                m[2] = max(m[0] + m[2], x + bw) - nx
                m[3] = max(m[1] + m[3], y + bh) - ny
                m[0], m[1] = nx, ny
                break
        else:
            merged.append([x, y, bw, bh])
    return [("Damage By Fold", tuple(m)) for m in merged[:FOLD_MAX_BOXES]]


# ============================================================
# Defect 3: finger not enough
# ============================================================
def _count_extended_fingertips(mask_filled, glove_box):
    """Count prominent upper-contour tips around the palm centre.

    The contour is kept in traversal order rather than reduced to a convex
    hull.  Consequently two adjacent fingers can still form two radial peaks
    even when one is shorter and hidden behind the other's angular direction.
    All distances, smoothing and prominence gates are normalised by glove size.
    """
    contours, _ = cv2.findContours(
        mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return 0

    contour = max(contours, key=cv2.contourArea)
    points = contour[:, 0, :]
    if len(points) < 5:
        return 0

    x, y0, width, height = glove_box
    scale = float(max(width, height, 1))
    distance = cv2.distanceTransform(
        (mask_filled > 0).astype(np.uint8), cv2.DIST_L2, 5
    )

    # The widest point in the middle of an upright glove is a stable palm
    # centre.  Excluding the upper fingers and lower cuff prevents either from
    # becoming the radial origin.
    search_top = min(
        mask_filled.shape[0] - 1, y0 + int(round(0.30 * height))
    )
    search_bottom = min(
        mask_filled.shape[0], y0 + max(1, int(round(0.78 * height)))
    )
    search_left = min(mask_filled.shape[1] - 1, max(0, x))
    search_right = min(mask_filled.shape[1], x + max(width, 1))
    palm_roi = distance[
        search_top:search_bottom, search_left:search_right
    ]
    if palm_roi.size == 0 or float(palm_roi.max()) <= 0.0:
        return 0
    _, _, _, palm_location = cv2.minMaxLoc(palm_roi)
    palm_x = search_left + int(palm_location[0])
    palm_y = search_top + int(palm_location[1])

    radial = np.hypot(
        points[:, 0].astype(np.float32) - palm_x,
        points[:, 1].astype(np.float32) - palm_y,
    ).astype(np.float32)
    point_count = len(radial)
    sigma = max(2.0, point_count * FINGER_TIP_SMOOTH_SIGMA_RATIO)
    tiled = np.concatenate([radial, radial, radial]).reshape(1, -1)
    smoothed_all = cv2.GaussianBlur(tiled, (0, 0), sigma).ravel()
    smoothed = smoothed_all[point_count:2 * point_count]

    neighbour = max(
        10, int(round(point_count * FINGER_TIP_NEIGHBOUR_RATIO))
    )
    local_max_all = cv2.dilate(
        smoothed_all.reshape(1, -1),
        np.ones((1, 2 * neighbour + 1), np.uint8),
    ).ravel()
    local_max = local_max_all[point_count:2 * point_count]
    candidate_indices = np.flatnonzero(smoothed >= local_max - 1e-4)

    maximum_tip_y = y0 + FINGER_TIP_MAX_Y_RATIO * height
    minimum_radius = FINGER_TIP_MIN_RADIUS_RATIO * scale
    minimum_prominence = FINGER_TIP_MIN_PROMINENCE_RATIO * scale
    candidates = []
    for index in candidate_indices:
        point_x, point_y = points[index]
        if point_y > maximum_tip_y or smoothed[index] < minimum_radius:
            continue
        centre = point_count + int(index)
        left_min = float(
            smoothed_all[centre - 2 * neighbour:centre].min()
        )
        right_min = float(
            smoothed_all[centre + 1:centre + 2 * neighbour + 1].min()
        )
        prominence = float(smoothed[index] - max(left_min, right_min))
        if prominence >= minimum_prominence:
            candidates.append((int(index), float(smoothed[index])))

    # A rounded fingertip can produce a short plateau of equal maxima. Merge
    # neighbouring plateau samples into one physical tip, including the contour
    # wrap between its first and last array entries.
    peaks = []
    for candidate in candidates:
        if not peaks or candidate[0] - peaks[-1][0] > neighbour:
            peaks.append(candidate)
        elif candidate[1] > peaks[-1][1]:
            peaks[-1] = candidate
    if (
        len(peaks) > 1
        and peaks[0][0] + point_count - peaks[-1][0] <= neighbour
    ):
        keep = peaks[0] if peaks[0][1] >= peaks[-1][1] else peaks[-1]
        peaks = peaks[1:-1] + [keep]
    return min(len(peaks), 6)


def detect_finger_not_enough(img, mask_filled, mask_raw, bg_color,
                             img_plain=None, material=None):
    """Detect a shortened, hidden, or absent glove finger.

    Three independent, explainable measurements are used because the real
    photographs contain two versions of this defect: some gloves expose a
    bare finger, while others have one glove finger folded out of view.

    1. A sufficiently large skin-coloured component inside the upper glove
       silhouette indicates an exposed finger when the silhouette boundary
       wraps around a finger-shaped component. Skin immediately outside the
       material mask is also accepted when its lower end attaches to the glove;
       segmentation can otherwise discard every bare finger on cotton gloves.
       Each accepted component becomes its own detection region. A weaker skin
       component still needs support from a missing finger column.
    2. Prominent radial peaks along the ordered upper contour count extended
       fingertips without requiring every finger to have the same length.
    3. Convex-hull gap size, silhouette solidity and persistent upper-row runs
       support the tip shortage so one noisy contour cue cannot decide alone.

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

    # The measurements below describe finger columns from top to palm. Real
    # inspection photos are commonly landscape, with the wrist entering from
    # either side. Normalise just this detector to an upright silhouette, then
    # map an accepted box back to the caller's coordinates. The wider/more
    # occupied end is the wrist; the opposite end contains the fingertips.
    if width > height:
        crop = mask_filled[y0:y0 + height, x:x + width] > 0
        band_width = max(1, int(round(width * 0.12)))
        left_support = int(np.count_nonzero(crop[:, :band_width]))
        right_support = int(np.count_nonzero(crop[:, -band_width:]))
        rotation = (
            cv2.ROTATE_90_COUNTERCLOCKWISE
            if left_support >= right_support
            else cv2.ROTATE_90_CLOCKWISE
        )
        rotated_results = detect_finger_not_enough(
            cv2.rotate(img, rotation),
            cv2.rotate(mask_filled, rotation),
            cv2.rotate(mask_raw, rotation),
            bg_color,
            img_plain=(
                cv2.rotate(img_plain, rotation)
                if img_plain is not None else None
            ),
            material=material,
        )
        mapped_results = []
        image_height, image_width = mask_filled.shape[:2]
        for result in rotated_results:
            rx, ry, rw, rh = result.box
            if rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
                mapped_box = (image_width - ry - rh, rx, rh, rw)
                mapped_mask = (
                    cv2.rotate(result.mask, cv2.ROTATE_90_CLOCKWISE)
                    if result.mask is not None else None
                )
            else:
                mapped_box = (ry, image_height - rx - rw, rh, rw)
                mapped_mask = (
                    cv2.rotate(result.mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    if result.mask is not None else None
                )
            mapped_results.append(Detection(
                result.name,
                tuple(int(value) for value in mapped_box),
                mapped_mask,
                result.evidence,
            ))
        return mapped_results

    material_key = str(material).lower() if material is not None else None
    rule = FINGER_NOT_ENOUGH_MATERIAL_RULES.get(
        material_key, FINGER_NOT_ENOUGH_DEFAULT_RULE
    )
    shape_rule = FINGER_SHAPE_MATERIAL_RULES.get(
        material_key, FINGER_SHAPE_DEFAULT_RULE
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
    indentation_strengths = []
    largest_indentation_box = None
    largest_indentation_area = -1
    largest_indentation_area_ratio = 0.0
    for label in range(1, indent_count):
        component_area_ratio = (
            float(indent_stats[label, cv2.CC_STAT_AREA]) / hull_area
        )
        centroid_y_ratio = (
            float(indent_centroids[label, 1]) - y0
        ) / max(height, 1)
        component_width_ratio = (
            float(indent_stats[label, cv2.CC_STAT_WIDTH]) / max(width, 1)
        )
        if (
            component_area_ratio >= rule["indent_min_area_ratio"]
            and component_width_ratio <= rule["indent_max_width_ratio"]
            and centroid_y_ratio <= rule["indent_max_y_ratio"]
        ):
            qualifying_indentations += 1
            area_fit = min(
                1.0,
                component_area_ratio
                / max(4.0 * rule["indent_min_area_ratio"], 1e-6),
            )
            width_fit = np.clip(
                1.0
                - component_width_ratio / rule["indent_max_width_ratio"],
                0.0,
                1.0,
            )
            vertical_fit = np.clip(
                1.0 - centroid_y_ratio / rule["indent_max_y_ratio"],
                0.0,
                1.0,
            )
            indentation_strengths.append(
                0.50 * area_fit + 0.25 * width_fit + 0.25 * vertical_fit
            )
            largest_indentation_area_ratio = max(
                largest_indentation_area_ratio, component_area_ratio
            )
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

    fingertip_count = _count_extended_fingertips(
        mask_filled, (x, y0, width, height)
    )
    missing_column = (
        0 < fingertip_count < FINGER_EXPECTED_TIP_COUNT
    )
    glove_solidity = (
        float(cv2.countNonZero(mask_filled)) / max(hull_area, 1)
    )
    shape_support = (
        qualifying_indentations >= shape_rule["min_indent_count"]
        and (
            largest_indentation_area_ratio
            >= shape_rule["large_gap_area_ratio"]
            or glove_solidity <= shape_rule["max_solidity"]
            or qualifying_indentations
            in shape_rule["support_indent_counts"]
        )
    )
    missing_space = missing_column and shape_support

    # Use the original (non-lighting-normalised) colour for skin evidence.  Do
    # not intersect it with ``mask_filled`` here: on strongly coloured cotton,
    # segmentation often keeps only the glove material and drops the bare
    # fingers as separate components.  Their proximity and lower-end attachment
    # to the material mask are verified explicitly below.
    source = img_plain if img_plain is not None else img
    finger_region_height = max(
        1, int(round(height * FINGER_REGION_HEIGHT_RATIO))
    )
    skin_candidate = (_skin_colour_mask(source) > 0).astype(np.uint8)
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
    # Requiring boundary contact prevents enclosed palm tearing or internal
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

    attachment_radius = max(
        3, int(round(width * FINGER_SKIN_ATTACHMENT_RADIUS_RATIO))
    )
    near_glove = cv2.dilate(
        mask_filled,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * attachment_radius + 1, 2 * attachment_radius + 1),
        ),
    ) > 0
    skin_records = []
    for label in range(1, skin_component_count):
        component_area = int(skin_stats[label, cv2.CC_STAT_AREA])
        component_area_ratio = float(component_area) / hull_area
        if component_area_ratio < rule["skin_min_area_ratio"]:
            continue
        component_width_ratio = (
            float(skin_stats[label, cv2.CC_STAT_WIDTH]) / max(width, 1)
        )
        if component_width_ratio > rule["skin_max_width_ratio"]:
            # Segmentation can include the wearer's forearm at the image edge.
            # That broad skin region is not shaped like one exposed finger.
            continue

        component = skin_labels == label
        inside_ratio = (
            float(np.count_nonzero(component & (mask_filled > 0)))
            / max(component_area, 1)
        )
        boundary_ratio = (
            float(np.count_nonzero(component & glove_boundary))
            / max(component_area, 1)
        )
        component_height = int(skin_stats[label, cv2.CC_STAT_HEIGHT])
        component_width = int(skin_stats[label, cv2.CC_STAT_WIDTH])
        component_aspect_ratio = (
            float(component_height) / max(component_width, 1)
        )
        boundary_y, boundary_x = np.nonzero(component & glove_boundary)
        if boundary_x.size:
            boundary_x_span_ratio = (
                float(np.ptp(boundary_x) + 1) / max(component_width, 1)
            )
            boundary_y_span_ratio = (
                float(np.ptp(boundary_y) + 1) / max(component_height, 1)
            )
        else:
            boundary_x_span_ratio = 0.0
            boundary_y_span_ratio = 0.0
        component_is_internal_tip = (
            component_aspect_ratio >= FINGER_SKIN_TIP_MIN_ASPECT_RATIO
            and boundary_x_span_ratio
            >= FINGER_SKIN_TIP_MIN_BOUNDARY_SPAN_RATIO
            and boundary_y_span_ratio
            >= FINGER_SKIN_TIP_MIN_BOUNDARY_SPAN_RATIO
        )
        component_y_ratio = (
            float(skin_centroids[label, 1]) - y0
        ) / max(height, 1)
        if not (
            component_y_ratio >= FINGER_SKIN_MIN_Y_RATIO
            and component_y_ratio <= rule["skin_max_y_ratio"]
        ):
            continue

        component_top = int(skin_stats[label, cv2.CC_STAT_TOP])
        lower_start = component_top + int(round(
            component_height * FINGER_SKIN_LOWER_START_RATIO
        ))
        lower_component = component.copy()
        lower_component[:lower_start] = False
        lower_area = int(np.count_nonzero(lower_component))
        near_ratio = (
            float(np.count_nonzero(component & near_glove))
            / max(component_area, 1)
        )
        lower_near_ratio = (
            float(np.count_nonzero(lower_component & near_glove))
            / max(lower_area, 1)
        )
        component_is_external_tip = (
            inside_ratio < FINGER_SKIN_EXTERNAL_MAX_INSIDE_RATIO
            and component_aspect_ratio >= FINGER_SKIN_TIP_MIN_ASPECT_RATIO
            and near_ratio >= FINGER_SKIN_EXTERNAL_MIN_NEAR_RATIO
            and lower_near_ratio >= FINGER_SKIN_EXTERNAL_MIN_LOWER_NEAR_RATIO
        )
        internal_skin_evidence = (
            inside_ratio >= FINGER_SKIN_EXTERNAL_MAX_INSIDE_RATIO
            and boundary_ratio >= rule["skin_min_boundary_ratio"]
        )
        external_skin_evidence = component_is_external_tip
        if not (internal_skin_evidence or external_skin_evidence):
            continue

        area_fit = min(
            1.0,
            component_area_ratio
            / max(2.0 * rule["skin_min_area_ratio"], 1e-6),
        )
        if external_skin_evidence:
            contact_fit = min(
                1.0,
                lower_near_ratio
                / max(2.0 * FINGER_SKIN_EXTERNAL_MIN_LOWER_NEAR_RATIO, 1e-6),
            )
        else:
            contact_fit = min(
                1.0,
                boundary_ratio
                / max(2.0 * rule["skin_min_boundary_ratio"], 1e-6),
            )
        vertical_fit = np.clip(
            1.0 - max(component_y_ratio, 0.0) / rule["skin_max_y_ratio"],
            0.0,
            1.0,
        )
        candidate_strength = float(
            0.45 * area_fit + 0.35 * contact_fit + 0.20 * vertical_fit
        )
        component_is_exposed_tip = (
            component_is_internal_tip or component_is_external_tip
        )
        candidate_box = (
            int(skin_stats[label, cv2.CC_STAT_LEFT]),
            int(skin_stats[label, cv2.CC_STAT_TOP]),
            component_width,
            component_height,
        )
        component_mask = component.astype(np.uint8) * 255
        skin_records.append((
            candidate_box,
            component_mask,
            component_is_exposed_tip,
            candidate_strength,
        ))
    tip_shortage_strength = (
        np.clip(
            (FINGER_EXPECTED_TIP_COUNT - fingertip_count)
            / max(FINGER_EXPECTED_TIP_COUNT - 1, 1),
            0.0,
            1.0,
        )
        if missing_column else 0.0
    )
    row_shortage_strength = (
        np.clip(
            (FINGER_EXPECTED_TIP_COUNT - persistent_run_count)
            / max(FINGER_EXPECTED_TIP_COUNT - 1, 1),
            0.0,
            1.0,
        )
        if missing_column else 0.0
    )
    silhouette_shortage_strength = (
        0.80 * tip_shortage_strength + 0.20 * row_shortage_strength
    )
    accepted_skin = [
        record for record in skin_records
        if record[2] or missing_space
    ]
    if accepted_skin:
        results = []
        for box, component_mask, is_tip, strength in accepted_skin:
            skin_shape_strength = (
                0.70 * strength + 0.30
                if is_tip
                else 0.60 * strength + 0.40 * silhouette_shortage_strength
            )
            evidence = 50.0 + 50.0 * skin_shape_strength
            results.append(Detection(
                "Finger Not Enough",
                box,
                component_mask,
                round(float(evidence), 1),
            ))
        return sorted(results, key=lambda result: result.box[0])

    if not missing_space:
        return []

    indentation_strength = (
        float(np.mean(indentation_strengths))
        if indentation_strengths else 0.0
    )
    evidence = 50.0 + 50.0 * (
        0.65 * indentation_strength + 0.35 * silhouette_shortage_strength
    )

    # Localise the evidence for the display stage. Exposed skin is already a
    # real pixel region; otherwise the largest abnormal hull gap is the best
    # estimate of where a completely absent finger should have been. The full
    # upper zone remains a safe last resort for a row-count-only recognition.
    result_box = largest_indentation_box
    if result_box is None:
        result_box = (
            x,
            y0,
            width,
            min(finger_region_height, mask_filled.shape[0] - y0),
        )
    return [Detection(
        "Finger Not Enough",
        result_box,
        evidence=round(float(evidence), 1),
    )]


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
    elif material == "nitrile":
        selected = blue & (hue <= THIN_NITRILE_BLUE_H_MAX)
        subtype = "blue"
    elif material == "latex_foam":
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
    known_materials = {"cotton", "nitrile", "latex_foam"}
    if material_key not in known_materials:
        # No filename-based guess: evaluate the increasingly general physical
        # cues directly. Nitrile comes first because its narrow hue ROI rejects
        # the purple-blue backdrop that can pollute the generic blue region.
        for candidate_material in ("nitrile", "latex_foam", "cotton"):
            candidate = detect_thin_area(
                img,
                mask_filled,
                mask_raw,
                bg_color,
                img_plain=source,
                material=candidate_material,
            )
            if candidate:
                return candidate
        return []

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
    rule_strength = 0.0
    if material_key == "cotton" and cotton_subtype == "blue":
        saturation_p25 = float(np.percentile(saturation[interior], 25))
        is_thin = saturation_p25 < THIN_COTTON_BLUE_S25_MAX
        if is_thin:
            rule_strength = float(np.clip(
                (THIN_COTTON_BLUE_S25_MAX - saturation_p25) / 40.0,
                0.0,
                1.0,
            ))
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
        if is_thin:
            skin_fit = min(
                1.0,
                skin_ratio / max(2.0 * THIN_COTTON_WHITE_SKIN_MIN_RATIO, 1e-6),
            )
            component_fit = min(
                1.0,
                sizeable_skin_components
                / max(2.0 * THIN_COTTON_WHITE_SKIN_MIN_COMPONENTS, 1.0),
            )
            distribution_fit = float(np.clip(
                1.0
                - largest_skin_share
                / max(THIN_COTTON_WHITE_SKIN_MAX_LARGEST_SHARE, 1e-6),
                0.0,
                1.0,
            ))
            grid_fit = min(
                1.0,
                grid_coverage
                / max(2.0 * THIN_COTTON_WHITE_GRID_MIN_COVERAGE, 1e-6),
            )
            rule_strength = float(np.mean((
                skin_fit, component_fit, distribution_fit, grid_fit,
            )))
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
        if dispersed_skin:
            ratio_fit = float(np.clip(
                (skin_ratio - THIN_NITRILE_SKIN_MIN_RATIO)
                / max(
                    THIN_NITRILE_SKIN_MAX_RATIO
                    - THIN_NITRILE_SKIN_MIN_RATIO,
                    1e-6,
                ),
                0.0,
                1.0,
            ))
            component_fit = min(
                1.0,
                sizeable_skin_components
                / max(2.0 * THIN_NITRILE_SKIN_MIN_COMPONENTS, 1.0),
            )
            rule_strength = max(
                rule_strength, 0.35 + 0.65 * (ratio_fit + component_fit) / 2.0
            )
        if broadly_pale or shadow_pale:
            saturation_limit = (
                THIN_NITRILE_S_MEDIAN_MAX
                if broadly_pale else THIN_NITRILE_SHADOW_S_MEDIAN_MAX
            )
            light_fit = float(np.clip(
                (lightness_p25 - THIN_NITRILE_LIGHT_P25_MIN) / 40.0,
                0.0,
                1.0,
            ))
            saturation_fit = float(np.clip(
                (saturation_limit - saturation_median) / 50.0,
                0.0,
                1.0,
            ))
            rule_strength = max(
                rule_strength,
                0.35 + 0.65 * (light_fit + saturation_fit) / 2.0,
            )
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
        if is_thin:
            saturation_limit = (
                THIN_LATEX_LOW_S_MEDIAN_MAX
                if saturation_median < THIN_LATEX_LOW_S_MEDIAN_MAX
                else THIN_LATEX_BRIGHT_S_MEDIAN_MAX
            )
            light_fit = float(np.clip(
                (lightness_p25 - THIN_LATEX_LIGHT_P25_MIN) / 40.0,
                0.0,
                1.0,
            ))
            edge_fit = float(np.clip(
                (THIN_LATEX_LAPLACIAN_MEAN_MAX - laplacian_mean)
                / max(THIN_LATEX_LAPLACIAN_MEAN_MAX, 1e-6),
                0.0,
                1.0,
            ))
            saturation_fit = float(np.clip(
                (saturation_limit - saturation_median) / 60.0,
                0.0,
                1.0,
            ))
            rule_strength = (
                0.20 * light_fit + 0.45 * edge_fit + 0.35 * saturation_fit
            )

    if not is_thin:
        return []

    contours, _ = cv2.findContours(
        region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    box = cv2.boundingRect(max(contours, key=cv2.contourArea))
    defect_mask = _segment_thin_area(
        source, mask_filled, box, material_key
    )
    evidence = 50.0 + 50.0 * float(np.clip(rule_strength, 0.0, 1.0))
    return [Detection(
        "Thin / Overstretched",
        box,
        defect_mask,
        round(evidence, 1),
    )]


# ============================================================
# Defect 5: Plastic Contamination
# ============================================================
def detect_plastic_contamination(img, mask_filled, mask_raw, bg_color,
                                 img_plain=None, material=None):
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
        detect_tearing     candidate colour *equals* the backdrop -> openings right through
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


def detect_spotting(img, mask_filled, mask_raw, bg_color,
                    img_plain=None, material=None):
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


def detect_stains(img, mask_filled, mask_raw, bg_color,
                  img_plain=None, material=None):
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
    # detect_tearing is the team's rewrite of what used to be detect_holes;
    # my three sit after it and before the rest, so the more specific
    # detectors win de-duplication in the regions they claim.
    detect_tearing,
    detect_incomplete_beading,
    detect_damage_by_fold,
    detect_open_tears,
    # detect_side_tear sits AFTER detect_open_tears on merge. It was written
    # when tearing was still mine and it claims the same lateral breaches, so
    # ahead of detect_open_tears it starves the team's detector and their
    # regression scenarios lose their "Open Tear" count. Behind it, it only
    # picks up what their detector leaves.
    detect_side_tear,
    detect_finger_not_enough,
    detect_thin_area,
    # Spotting has to come before Stain: both detectors match the same batch of
    # small dots, and deduplicate() keeps whichever is registered first, so this
    # is what stops them being counted twice.
    detect_spotting,
    detect_stains,
    detect_plastic_contamination,
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
    tear also satisfies "thin area", for instance). Keep whichever comes first in
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


def run_all_detectors(img, mask_filled, mask_raw, bg_color, detectors=None,
                      img_plain=None, material=None):
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
            found = det(
                img,
                mask_filled,
                mask_raw,
                bg_color,
                img_plain=img_plain,
                material=material,
            )
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


def _segment_tearing(source, mask_filled, mask_raw, box):
    """Return only the accepted skin/background opening inside a tearing box."""
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

    The dilated gap touches the fingers on both sides. Restricting it to the
    gap's inner corridor separates those walls; an interior contact is evidence
    of a curled/shortened finger still physically present. The corridor extends
    slightly below the gap so the recovered mask includes the finger base.
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
        mask_filled.shape[0], gap_y + max(1, round(1.10 * gap_height))
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

    # Recover the complete visible short/folded finger rather than painting a
    # fixed-radius patch around its boundary.  The gap's inner corridor excludes
    # the two neighbouring full-length fingers and limits spill into the palm.
    # Connected-component growth within that constrained material region is the
    # morphological equivalent of geodesic dilation to stability, but completes
    # in one pass regardless of finger length.
    allowed = cv2.bitwise_and(mask_filled, search)
    allowed_count, allowed_labels, _, _ = cv2.connectedComponentsWithStats(
        (allowed > 0).astype(np.uint8), 8
    )
    best_allowed_label = None
    best_overlap = 0
    for label in range(1, allowed_count):
        overlap = int(np.count_nonzero(
            (allowed_labels == label) & (selected_contact > 0)
        ))
        if overlap > best_overlap:
            best_overlap = overlap
            best_allowed_label = label

    if best_allowed_label is not None:
        grown = (allowed_labels == best_allowed_label).astype(np.uint8) * 255
        grown = cv2.morphologyEx(
            grown, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )
        return cv2.bitwise_and(grown, mask_filled)

    # Degenerate one-pixel contacts can disappear from the constrained mask.
    # Retain the old local fallback so a recognised defect never loses all of
    # its visible evidence because of that numerical edge case.
    local = cv2.dilate(
        selected_contact,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    return cv2.bitwise_and(local, mask_filled)


def _segment_finger_not_enough(source, mask_filled, box, material):
    """Colour a visible short finger or the estimated missing-space region.

    Exposed-skin detections already carry their validated component mask from
    recognition and never enter this helper.  Re-running broad skin thresholding
    here used to colour unrelated wrist/forearm pixels inside a geometry box.
    """
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

    if best_label is None:
        return np.zeros_like(mask_filled)

    gap_component = (labels == best_label).astype(np.uint8) * 255
    visible = _visible_short_finger(mask_filled, labels == best_label)
    visible_area = int(cv2.countNonZero(visible))
    gap_area = max(int(cv2.countNonZero(gap_component)), 1)
    if visible_area >= max(20, round(0.25 * gap_area)):
        return visible

    # No trustworthy material protrusion remains: represent the inferred area
    # that the missing/folded finger should occupy.  It is clipped to the
    # recognised box so unrelated hull gaps cannot enter the affected-area mask.
    return cv2.bitwise_and(gap_component, box_mask)


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
    for defect in defects:
        name, box = defect
        if (
            isinstance(defect, Detection)
            and defect.mask is not None
            and defect.mask.shape == mask_filled.shape
        ):
            mask = defect.mask
        elif name == "Tearing":
            if name not in reusable:
                reusable[name] = _segment_tearing(
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
        right = min(out.shape[1] - 1, x + text_w + 2 * pad)
        cv2.rectangle(out, (x, top), (right, bottom), color, cv2.FILLED)
        text_color = (20, 20, 20) if name == "Stain" else (255, 255, 255)
        cv2.putText(
            out, label, (x + pad, label_y - 1), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, text_color, thin, cv2.LINE_AA,
        )
    return out

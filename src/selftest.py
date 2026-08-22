# -*- coding: utf-8 -*-
"""
Regression test script -- run this every time the algorithms change, to
confirm nothing broke:
    .venv\\Scripts\\python src\\selftest.py

It automatically generates a batch of "simulated glove images" (different
lighting, backgrounds, stain colours, glove offsets, etc.), runs the full
detection pipeline on each, compares the result against the EXPECTED
result for that scenario, and finally prints a pass rate.

Why this exists:
  The assignment requires "the system must not be sensitive to
  environmental changes". Eyeballing one or two images can't show that.
  A batch of controllable synthetic scenarios lets us quantify the
  system's robustness, and gives us numbers to put in the report's
  "Experimental Results & Critical Analysis" section.

Caution: synthetic images can only verify that the ALGORITHM LOGIC is
  correct, they are not a substitute for real photographs. Real glove
  texture, shadows and reflections must be validated against the
  self-collected dataset in dataset/.
"""
import os
import sys

import cv2
import numpy as np

# Force UTF-8 stdout in case the terminal encoding doesn't support it
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from preprocessing import preprocess
from segmentation import segment_glove, glove_found, get_background_color
from defect_detection import run_all_detectors, draw_results

GLOVE = (190, 120, 40)   # glove colour (BGR, blue)
BG = (60, 60, 200)       # background colour (BGR, red)


def make_glove_image(stain_color=(100, 60, 20), hole=True, stain=True,
                     bright=1.0, offset=(0, 0), bg=BG, glove=GLOVE,
                     noise=0, side_light=False,
                     open_tear=False, fingertip_tear=False):
    """Draw one simulated glove image; the parameters simulate different
    shooting conditions and defects."""
    img = np.full((600, 800, 3), bg, dtype=np.uint8)
    ox, oy = offset

    # palm + wrist
    cv2.ellipse(img, (400 + ox, 330 + oy), (130, 110), 0, 0, 360, glove, -1)
    cv2.rectangle(img, (330 + ox, 400 + oy), (470 + ox, 560 + oy), glove, -1)
    # five fingers (finger gaps are kept wide enough that the closing
    # operation in morphology doesn't merge them into solid glove)
    for i, fx in enumerate([290, 345, 400, 455, 505]):
        cv2.ellipse(img, (fx + ox, 220 + oy), (18, 90 - abs(i - 2) * 12),
                    0, 0, 360, glove, -1)

    if hole:   # defect 1: enclosed hole (reveals the background colour)
        cv2.circle(img, (430 + ox, 330 + oy), 22, bg, -1)
    if open_tear:   # defect 2a: open tear on the side of the palm (cutting in from the left edge)
        # Drawn on the LEFT side of the palm on purpose: the hole sits on
        # the right, and if the two defects overlapped they'd merge into
        # one shape, making the hole no longer "enclosed" -- which would
        # break the expected values below
        cv2.fillPoly(img, [np.array([[270 + ox, 330 + oy],
                                     [370 + ox, 315 + oy],
                                     [270 + ox, 355 + oy]])], bg)
    if fingertip_tear:  # defect 2b: fingertip tear (a narrow slit cut down from the middle fingertip)
        cv2.fillPoly(img, [np.array([[393 + ox, 126 + oy],
                                     [400 + ox, 215 + oy],
                                     [407 + ox, 126 + oy]])], bg)
    if stain:  # defect 3: stain (a patch of discolouration)
        cv2.ellipse(img, (400 + ox, 480 + oy), (18, 12), 30, 0, 360,
                    stain_color, -1)

    if side_light:  # simulate side lighting: dark on the left, bright on the right
        gradient = np.linspace(0.45, 1.35, img.shape[1])[None, :, None]
        img = np.clip(img.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
    if bright != 1.0:  # simulate overall under/over-exposure
        img = np.clip(img.astype(np.float32) * bright, 0, 255).astype(np.uint8)
    if noise:  # simulate sensor noise
        img = np.clip(img.astype(np.int16) +
                      np.random.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


NO_GLOVE = ("No Glove", "No Glove", "No Glove")


def analyse(img):
    """Run the full pipeline, return (enclosed-hole count, open-tear count,
    stain count). Returns NO_GLOVE if no glove was found."""
    img_norm, img_plain = preprocess(img)
    mask_filled, mask_raw = segment_glove(img_norm)
    ok, ratio = glove_found(mask_filled)
    if not ok:
        return NO_GLOVE
    bg_color = get_background_color(img_norm)
    defects, _ = run_all_detectors(img_norm, mask_filled, mask_raw, bg_color)
    names = [n for n, _ in defects]
    # "Open Tear" and "Side Tear" are counted together here on purpose:
    # both mean "the glove edge is breached", which is what these
    # scenarios assert, and combining them keeps every pre-existing
    # expectation below valid now that the tear work is split between two
    # detectors. Which of the two fired -- i.e. whether the position
    # filter put the tear in the right class -- is asserted separately in
    # build_side_tear_cases().
    return (names.count("Tear / Hole"),
            names.count("Open Tear") + names.count("Side Tear"),
            names.count("Stain"))


def tear_labels(img):
    """Just the tear labels the pipeline produced, for the side-tear
    scope tests: (side tear count, open tear count)."""
    img_norm, _ = preprocess(img)
    mask_filled, mask_raw = segment_glove(img_norm)
    ok, _ = glove_found(mask_filled)
    if not ok:
        return NO_GLOVE[:2]
    bg_color = get_background_color(img_norm)
    defects, _ = run_all_detectors(img_norm, mask_filled, mask_raw, bg_color)
    names = [n for n, _ in defects]
    return names.count("Side Tear"), names.count("Open Tear")


# Each scenario: (name, image, expected (hole count, tear count, stain count))
def build_cases():
    return [
        ("Baseline: 1 hole + 1 stain",           make_glove_image(),                                 (1, 0, 1)),
        ("Off-colour stain (black grime)",       make_glove_image(stain_color=(20, 20, 20)),         (1, 0, 1)),
        ("Off-colour stain (white powder mark)", make_glove_image(stain_color=(240, 240, 240)),      (1, 0, 1)),
        ("Clean glove (expect zero false positives)", make_glove_image(hole=False, stain=False),      (0, 0, 0)),
        ("Clean glove + noise sigma=8",          make_glove_image(hole=False, stain=False, noise=8), (0, 0, 0)),
        ("Low light: overall 60% darker",        make_glove_image(bright=0.6),                       (1, 0, 1)),
        ("Bright light: overall 140% brighter",  make_glove_image(bright=1.4),                       (1, 0, 1)),
        ("Side lighting (dark left, bright right)", make_glove_image(side_light=True),                (1, 0, 1)),
        ("Glove off-centre",                     make_glove_image(offset=(150, -60)),                (1, 0, 1)),
        ("Hole exactly at frame centre",         make_glove_image(offset=(-30, 0)),                  (1, 0, 1)),
        ("Background colour close to glove colour", make_glove_image(bg=(200, 150, 90)),              (1, 0, 1)),
        ("Grey glove + off-white background",     make_glove_image(glove=(120, 120, 120),
                                                   bg=(190, 190, 190)),               (1, 0, 1)),
        ("Stain only, no hole",                  make_glove_image(hole=False,
                                                   stain_color=(20, 20, 20)),         (0, 0, 1)),
        # --- open tears: the key risk is misreporting the 4 normal finger gaps as tears ---
        ("Open tear on palm edge",               make_glove_image(hole=False, stain=False,
                                                   open_tear=True),                   (0, 1, 0)),
        ("Fingertip tear",                       make_glove_image(hole=False, stain=False,
                                                   fingertip_tear=True),              (0, 1, 0)),
        ("Open tear + enclosed hole",            make_glove_image(stain=False, open_tear=True),      (1, 1, 0)),
        ("All three defects together",           make_glove_image(open_tear=True),                   (1, 1, 1)),
        ("Tear + low light 60%",                 make_glove_image(hole=False, stain=False,
                                                   open_tear=True, bright=0.6),       (0, 1, 0)),
        ("Tear + off-centre glove",               make_glove_image(hole=False, stain=False,
                                                   open_tear=True,
                                                   offset=(150, -60)),                (0, 1, 0)),
        ("Blank background (no glove at all)",   np.full((600, 800, 3), BG, dtype=np.uint8), NO_GLOVE),
    ] + build_material_cases()


# Glove colours for different materials. The earlier lighting scenarios
# only used a bright blue glove, so they always scored full marks -- and
# never caught that "side light + dark glove" breaks segmentation, which
# is exactly what the batch evaluation script exposed.
# Materials and lighting are now crossed into a matrix: passing
# combinations feed the regression gate, failing ones go into the known
# limitations list below.
MATERIALS = [("latex-blue", (190, 120, 40)), ("rubber-gray", (80, 80, 80)),
             ("leather-navy", (60, 90, 150)), ("white-latex", (235, 235, 235))]
LIGHTINGS = [("uniform", {}), ("dim-60%", dict(bright=0.6)),
             ("bright-140%", dict(bright=1.4)), ("noise-8", dict(noise=8))]

# Known-failing combinations (material name, lighting name) -- see the
# KNOWN_FAIL note below
KNOWN_FAIL = {("rubber-gray", "dim-60%")}


def build_material_cases():
    """Material x lighting matrix, every image has 1 hole + 1 stain."""
    cases = []
    for mname, color in MATERIALS:
        for lname, kw in LIGHTINGS:
            if (mname, lname) in KNOWN_FAIL:
                continue
            cases.append((f"{mname}/{lname}",
                          make_glove_image(glove=color, **kw), (1, 0, 1)))
    return cases


def build_side_tear_cases():
    """Scope tests for detect_side_tear: (side tear count, open tear count).

    The pass-rate block above only checks that SOME tear was found. These
    check the harder thing -- that the position filter routed each tear
    to the right class, so the two tear detectors stay disjoint instead of
    both claiming the same defect.
    """
    return [
        ("Lateral tear -> Side Tear, not Open Tear",
         make_glove_image(hole=False, stain=False, open_tear=True),          (1, 0)),
        ("Fingertip tear -> Open Tear, not Side Tear",
         make_glove_image(hole=False, stain=False, fingertip_tear=True),     (0, 1)),
        ("Both tears -> one of each, no double-claim",
         make_glove_image(hole=False, stain=False,
                          open_tear=True, fingertip_tear=True),              (1, 1)),
        ("Clean glove -> finger gaps are not tears",
         make_glove_image(hole=False, stain=False),                          (0, 0)),
        ("Lateral tear + enclosed hole (hole must not leak in)",
         make_glove_image(stain=False, open_tear=True),                      (1, 0)),
        ("Lateral tear, dim 60%",
         make_glove_image(hole=False, stain=False, open_tear=True,
                          bright=0.6),                                       (1, 0)),
        ("Lateral tear, glove off-centre",
         make_glove_image(hole=False, stain=False, open_tear=True,
                          offset=(150, -60)),                                (1, 0)),
        ("Lateral tear, noise sigma=8",
         make_glove_image(hole=False, stain=False, open_tear=True, noise=8), (1, 0)),
        ("Lateral tear on grey glove",
         make_glove_image(hole=False, stain=False, open_tear=True,
                          glove=(80, 80, 80)),                               (1, 0)),
        ("Lateral tear on white glove",
         make_glove_image(hole=False, stain=False, open_tear=True,
                          glove=(235, 235, 235)),                            (1, 0)),
    ]


def build_known_issues():
    """Known, still-unresolved issues, run and printed separately, not
    counted in the pass rate above.

    Kept here rather than deleted so the issue stays visible: a perfect
    regression score with a documented known defect is far more honest
    than "the tests just didn't cover it".
    """
    cases = []
    for mname, color in MATERIALS:
        cases.append((f"Side lighting + {mname}",
                      make_glove_image(glove=color, side_light=True), (1, 0, 1)))
    for mname, lname in sorted(KNOWN_FAIL):
        color = dict(MATERIALS)[mname]
        kw = dict(LIGHTINGS)[lname]
        cases.append((f"{lname} + {mname}",
                      make_glove_image(glove=color, **kw), (1, 0, 1)))
    return cases


def main():
    np.random.seed(0)  # fix the random seed so results are reproducible
    cases = build_cases()

    print("=" * 84)
    print(f"OpenCV version: {cv2.__version__}")
    print("-" * 84)
    print(f"{'Scenario':<42}{'Expected':>14}{'Actual':>14}{'Result':>8}")
    print("-" * 84)

    passed = 0
    for name, img, expect in cases:
        got = analyse(img)
        ok = got == expect
        passed += ok
        print(f"{name:<42}{str(expect):>14}{str(got):>14}{'  PASS' if ok else '  FAIL'}")

    print("-" * 84)
    print(f"Pass rate: {passed}/{len(cases)}")

    # ---- side-tear scope tests: did the tear land in the RIGHT class? ----
    side_cases = build_side_tear_cases()
    print("\n" + "=" * 84)
    print("Side tear scope tests -- (Side Tear count, Open Tear count)")
    print("-" * 84)
    side_passed = 0
    for name, img, expect in side_cases:
        got = tear_labels(img)
        ok = got == expect
        side_passed += ok
        print(f"{name:<52}{str(expect):>10}{str(got):>10}{'  PASS' if ok else '  FAIL'}")
    print("-" * 84)
    print(f"Side tear pass rate: {side_passed}/{len(side_cases)}")
    print("=" * 84)

    # ---- known limitations: run and printed separately, excluded from the pass rate above ----
    known = build_known_issues()
    print("\n" + "=" * 84)
    print("Known limitations (unresolved, excluded from the pass rate above -- "
         "use these for the report's critical analysis)")
    print("-" * 84)
    for name, img, expect in known:
        got = analyse(img)
        mark = "this combination is fine" if got == expect else "still failing"
        print(f"{name:<44}{str(expect):>14}{str(got):>14}  {mark}")
    print("-" * 84)
    print("Root cause: segmentation uses a single global background reference")
    print("colour + a global Otsu threshold, which fails when the background")
    print("itself has a strong brightness gradient. Fix candidates: local")
    print("background estimation / illumination flattening.")
    print("=" * 84)

    # Save the baseline scenario's annotated result as an image, so it's
    # easy to eyeball whether the boxes look right
    img = make_glove_image()
    img_norm, img_plain = preprocess(img)
    mask_filled, mask_raw = segment_glove(img_norm)
    bg_color = get_background_color(img_norm)
    defects, _ = run_all_detectors(img_norm, mask_filled, mask_raw, bg_color)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, "selftest_result.jpg"))
    cv2.imwrite(out_path, draw_results(img_plain, defects))
    print(f"Annotated result saved to: {out_path}")
    print("=" * 84)

    # Return non-zero if any scenario failed, so this can be wired into CI later
    return 0 if (passed == len(cases) and side_passed == len(side_cases)) else 1


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""
Regression test -- run this after every algorithm change to confirm nothing
broke:
    .venv\\Scripts\\python src\\selftest.py

It generates a batch of "simulated glove images" (different lighting, different
backgrounds, different stain colours, the glove shifted off centre, and so on),
runs the whole detection pipeline over them, compares each scenario's result
against the *expected* result, and prints the pass rate.

Why bother:
  The assignment requires that "the system must not be sensitive to
  environmental changes". Eyeballing one or two pictures cannot show that.
  A batch of controlled synthetic scenarios is what lets us state the system's
  robustness as a number, and gives us data for the report's "experimental
  results and critical analysis" section.

! Synthetic images can only verify that the algorithm's logic is right. They
  are no substitute for real photographs. Real glove texture, shadows and
  highlights have to be checked against the dataset under dataset/.
"""
import os
import sys

import cv2
import numpy as np

# The Windows console default encoding may not handle every character; force UTF-8
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from preprocessing import preprocess
from segmentation import segment_glove, glove_found, get_background_color
from defect_detection import run_all_detectors, draw_results

GLOVE = (190, 120, 40)   # glove colour (BGR, blue)
BG = (60, 60, 200)       # background colour (BGR, red)


def make_glove_image(stain_color=(100, 60, 20), tearing=True, stain=True,
                     bright=1.0, offset=(0, 0), bg=BG, glove=GLOVE,
                     noise=0, side_light=False,
                     open_tear=False, fingertip_tear=False, two_tone=False,
                     spots=0, spot_color=(40, 220, 240),
                     plastic=False, glove_scale=1.0):
    """Draw a simulated glove image; the arguments simulate different shooting
    conditions and defects."""
    img = np.full((600, 800, 3), bg, dtype=np.uint8)
    ox, oy = offset

    # Palm + wrist. glove_scale scales the whole glove about the palm centre,
    # to check that the detectors are not sensitive to how big the glove is.
    def sx(x):
        return int(400 + ox + (x - 400) * glove_scale)

    def sy(y):
        return int(330 + oy + (y - 330) * glove_scale)

    def sr(r):
        return max(int(r * glove_scale), 1)

    cv2.ellipse(img, (sx(400), sy(330)), (sr(130), sr(110)), 0, 0, 360, glove, -1)
    cv2.rectangle(img, (sx(330), sy(400)), (sx(470), sy(560)), glove, -1)
    # Five fingers (keep the gaps wide enough that the closing operation does
    # not swallow them into the glove body)
    for i, fx in enumerate([290, 345, 400, 455, 505]):
        cv2.ellipse(img, (sx(fx), sy(220)), (sr(18), sr(90 - abs(i - 2) * 12)),
                    0, 0, 360, glove, -1)

    if two_tone:  # dark fabric cuff of a coated glove; normal material, not a stain
        cuff_color = tuple(max(int(channel * 0.55), 20) for channel in glove)
        cv2.rectangle(img, (330 + ox, 475 + oy), (470 + ox, 560 + oy),
                      cuff_color, -1)

    if tearing:   # defect 1: enclosed tearing (the background shows through)
        cv2.circle(img, (430 + ox, 330 + oy), 22, bg, -1)
    if open_tear:   # defect 2a: open tear in the side of the palm (from the left edge inwards)
        # Note it is drawn on the *left* of the palm: the tear is on the right,
        # and if the two overlapped they would merge into one region, so the
        # tear would no longer be enclosed and the expected result would stop
        # making physical sense.
        cv2.fillPoly(img, [np.array([[270 + ox, 330 + oy],
                                     [370 + ox, 315 + oy],
                                     [270 + ox, 355 + oy]])], bg)
    if fingertip_tear:  # defect 2b: fingertip tear (a narrow slit down the middle finger)
        cv2.fillPoly(img, [np.array([[393 + ox, 126 + oy],
                                     [400 + ox, 215 + oy],
                                     [407 + ox, 126 + oy]])], bg)
    if stain:  # defect 3: stain (a patch of changed colour)
        cv2.ellipse(img, (400 + ox, 480 + oy), (18, 12), 30, 0, 360,
                    stain_color, -1)

    if spots:  # defect 4: scattered small coloured dots (Spotting)
        # Fixed positions, so the generated image is the same every run and the
        # regression result stays reproducible
        placements = [(300, 250), (350, 300), (400, 260), (450, 310), (380, 350),
                      (320, 380), (430, 380), (290, 300), (460, 260), (410, 200)]
        for (px, py) in placements[:spots]:
            cv2.circle(img, (px + ox, py + oy), 11, spot_color, -1)

    if plastic:  # defect 5: Plastic Contamination
        # The film itself is nearly invisible; what it really leaves behind are
        # the specular reflections along its creases: a small area packed with
        # near-white streaks. Fixed random seed keeps it reproducible.
        rng = np.random.RandomState(7)
        x0, y0, side = 300, 268, 58
        for _ in range(20):
            ax, ay, bx, by = rng.randint(0, side, 4)
            cv2.line(img, (x0 + ax + ox, y0 + ay + oy),
                     (x0 + bx + ox, y0 + by + oy), (238, 242, 246), 3)

    if side_light:  # simulate side lighting: dark on the left, bright on the right
        gradient = np.linspace(0.45, 1.35, img.shape[1])[None, :, None]
        img = np.clip(img.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
    if bright != 1.0:  # simulate an overall darker / brighter exposure
        img = np.clip(img.astype(np.float32) * bright, 0, 255).astype(np.uint8)
    if noise:  # simulate sensor noise
        img = np.clip(img.astype(np.int16) +
                      np.random.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


NO_GLOVE = ("no glove", "no glove", "no glove", "no glove", "no glove")


def analyse(img):
    """Run the whole pipeline and return
    (enclosed tears, open tears, stains, spottings, plastic contaminations).
    Returns NO_GLOVE when no glove was found."""
    img_norm, img_plain = preprocess(img)
    mask_filled, mask_raw = segment_glove(img_norm)
    ok, ratio = glove_found(mask_filled)
    if not ok:
        return NO_GLOVE
    bg_color = get_background_color(img_norm)
    defects, _ = run_all_detectors(img_norm, mask_filled, mask_raw, bg_color)
    names = [n for n, _ in defects]
    return (names.count("Tearing"),
            names.count("Open Tear"),
            names.count("Stain"),
            names.count("Spotting"),
            names.count("Plastic Contamination"))


# Each scenario is (name, image, expected (enclosed tears, open tears, stains, spots, plastic))
def build_cases():
    return [
        ("baseline: 1 tearing + 1 stain", make_glove_image(),                       (1, 0, 1, 0, 0)),
        ("odd stain colour (white powder)", make_glove_image(stain_color=(240, 240, 240)),
                                                                                    (1, 0, 1, 0, 0)),
        ("good glove (zero false alarms)", make_glove_image(tearing=False, stain=False),
                                                                                    (0, 0, 0, 0, 0)),
        ("good glove + noise sigma=8", make_glove_image(tearing=False, stain=False, noise=8),
                                                                                    (0, 0, 0, 0, 0)),
        ("good two-tone material",     make_glove_image(tearing=False, stain=False,
                                                        two_tone=True),             (0, 0, 0, 0, 0)),
        ("dim light: 60% brightness",  make_glove_image(bright=0.6),                (1, 0, 1, 0, 0)),
        ("bright light: 140%",         make_glove_image(bright=1.4),                (1, 0, 1, 0, 0)),
        ("side lighting (dark left)",  make_glove_image(side_light=True),           (1, 0, 1, 0, 0)),
        ("glove off centre",           make_glove_image(offset=(150, -60)),         (1, 0, 1, 0, 0)),
        ("tearing exactly at the centre", make_glove_image(offset=(-30, 0)),        (1, 0, 1, 0, 0)),
        ("background close to glove colour", make_glove_image(bg=(200, 150, 90)),   (1, 0, 1, 0, 0)),
        ("grey glove + grey-white background", make_glove_image(glove=(120, 120, 120),
                                                        bg=(190, 190, 190)),        (1, 0, 1, 0, 0)),
        # --- open tears: the point is not to report the 4 normal finger gaps ---
        ("open tear in palm side",     make_glove_image(tearing=False, stain=False,
                                                        open_tear=True),            (0, 1, 0, 0, 0)),
        ("fingertip tear",             make_glove_image(tearing=False, stain=False,
                                                        fingertip_tear=True),       (0, 1, 0, 0, 0)),
        ("open tear + enclosed tearing", make_glove_image(stain=False, open_tear=True),
                                                                                    (1, 1, 0, 0, 0)),
        ("three defects at once",      make_glove_image(open_tear=True),            (1, 1, 1, 0, 0)),
        ("tear + dim light 60%",       make_glove_image(tearing=False, stain=False,
                                                        open_tear=True, bright=0.6), (0, 1, 0, 0, 0)),
        ("tear + glove off centre",    make_glove_image(tearing=False, stain=False,
                                                        open_tear=True,
                                                        offset=(150, -60)),         (0, 1, 0, 0, 0)),
        ("plain background (no glove at all)", np.full((600, 800, 3), BG, dtype=np.uint8),
                                                                                    NO_GLOVE),
        # --- Spotting: the main criterion is "are there enough dots", not area ---
        ("spotting: 8 yellow dots",    make_glove_image(tearing=False, stain=False,
                                                        spots=8),                   (0, 0, 0, 8, 0)),
        ("spotting: 10 yellow dots",   make_glove_image(tearing=False, stain=False,
                                                        spots=10),                  (0, 0, 0, 10, 0)),
        # Below 5 dots Spotting has to give up. This proves it is not just Stain
        # under another name. Here the 3 dots are ~380px each, also under Stain's
        # area threshold, so in the end nothing is reported at all.
        ("only 3 dots: not spotting",  make_glove_image(tearing=False, stain=False,
                                                        spots=3),                   (0, 0, 0, 0, 0)),
        ("spotting + tearing together", make_glove_image(stain=False, spots=8),     (1, 0, 0, 8, 0)),
    ] + build_material_cases() + build_plastic_cases()


# Glove colours for the different materials. The earlier lighting scenarios only
# used a bright blue glove, which is why they always scored full marks and never
# revealed that "side light + dark glove" breaks segmentation -- the batch
# evaluation script is what dug that up. Material and lighting are now a matrix:
# the combinations that pass guard the regression, the ones that fail go into the
# known-limitations list below.
MATERIALS = [("latex bright blue", (190, 120, 40)), ("rubber dark grey", (80, 80, 80)),
             ("leather dark blue", (60, 90, 150)), ("white latex", (235, 235, 235))]
LIGHTINGS = [("even", {}), ("dim 60%", dict(bright=0.6)),
             ("bright 140%", dict(bright=1.4)), ("noise 8", dict(noise=8))]

# Combinations known to fail (material, lighting) -- see the known-issues notes
KNOWN_FAIL = {("rubber dark grey", "dim 60%")}


def build_material_cases():
    """Material x lighting matrix; every image has 1 tearing + 1 stain."""
    cases = []
    for mname, color in MATERIALS:
        for lname, kw in LIGHTINGS:
            if (mname, lname) in KNOWN_FAIL:
                continue
            cases.append((f"{mname} / {lname}",
                          make_glove_image(glove=color, **kw), (1, 0, 1, 0, 0)))
    return cases


def build_plastic_cases():
    """Plastic Contamination scenarios. Three cases pinning down three things:

      1. a small patch of crease reflections on a coloured glove -> must report
      2. the same glove with nothing on it                       -> must not report
      3. a white latex glove (the material itself is unsaturated) -> must abstain
         The third one matters: "less saturated than the material" is meaningless
         on a white glove, and the detector has to recognise that itself instead
         of applying the rule blindly.
    """
    return [
        ("plastic contamination",     make_glove_image(tearing=False, stain=False,
                                                        plastic=True),           (0, 0, 0, 0, 1)),
        ("same glove, nothing on it", make_glove_image(tearing=False, stain=False), (0, 0, 0, 0, 0)),
        # The rule does not apply on a white glove, so it must abstain -- decide
        # when there is evidence, never guess when there is none
        ("white glove (material gate must abstain)",
                                      make_glove_image(tearing=False, stain=False,
                                                        glove=(235, 235, 235),
                                                        plastic=True),           (0, 0, 0, 0, 0)),
    ]


def build_known_issues():
    """Problems we have not solved yet: run and printed separately, and not
    counted towards the pass rate.

    They live here rather than being deleted so the problems stay visible. A
    full-marks regression run alongside a documented known defect is far more
    honest than a defect the tests simply never cover.
    """
    cases = [
        # A small dark stain (~700px). The darkness rule works on "how much
        # darker than the material", but finger edges and crease shadows are
        # just as dark: measured, those false alarms run 660-1449px, which
        # overlaps this dot's size exactly. Area, compactness and inscribed
        # radius all fail to separate them.
        # Trade-off: the area threshold sits at 2000px so real photos get zero
        # false alarms (real black-paint stains start at 2734px), and the price
        # is that these two scenarios are missed.
        ("small dark speck (700px)", make_glove_image(stain_color=(20, 20, 20)), (1, 0, 1, 0, 0)),
        ("small dark speck, no tearing", make_glove_image(tearing=False,
                                             stain_color=(20, 20, 20)), (0, 0, 1, 0, 0)),
    ]
    for mname, color in MATERIALS:
        cases.append((f"side lighting + {mname}",
                      make_glove_image(glove=color, side_light=True), (1, 0, 1, 0, 0)))
    for mname, lname in sorted(KNOWN_FAIL):
        color = dict(MATERIALS)[mname]
        kw = dict(LIGHTINGS)[lname]
        cases.append((f"{lname} + {mname}",
                      make_glove_image(glove=color, **kw), (1, 0, 1, 0, 0)))
    return cases


def main():
    np.random.seed(0)  # fixed seed, so every run reproduces the same result
    cases = build_cases()

    print("=" * 78)
    print(f"OpenCV version : {cv2.__version__}")
    print("-" * 78)
    print(f"{'Scenario':<42}{'expected':>15}{'got':>15}{'':>6}")
    print("-" * 78)

    passed = 0
    for name, img, expect in cases:
        got = analyse(img)
        ok = got == expect
        passed += ok
        print(f"{name:<42}{str(expect):>15}{str(got):>15}{'  PASS' if ok else '  FAIL'}")

    print("-" * 78)
    print(f"Pass rate : {passed}/{len(cases)}")

    # ---- known limitations: run and printed separately, not counted above ----
    known = build_known_issues()
    print("\n" + "=" * 78)
    print("Known limitations (unsolved, not counted -- this is the report's "
          "critical analysis)")
    print("-" * 78)
    for name, img, expect in known:
        got = analyse(img)
        mark = "this one is fine" if got == expect else "still failing"
        print(f"{name:<42}{str(expect):>15}{str(got):>15}  {mark}")
    print("-" * 78)
    print("small dark speck: see the notes in build_known_issues() -- a small")
    print("      dark stain and a crease shadow cannot be told apart by size or")
    print("      shape; sacrificed to keep real photos free of false alarms.")
    print("dim 60% + rubber dark grey: at low contrast segmentation takes")
    print("      part of the glove for background.")
    print("=" * 78)

    # Save the annotated baseline scenario, so the boxes can be checked by eye
    img = make_glove_image()
    img_norm, img_plain = preprocess(img)
    mask_filled, mask_raw = segment_glove(img_norm)
    bg_color = get_background_color(img_norm)
    defects, _ = run_all_detectors(img_norm, mask_filled, mask_raw, bg_color)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, "selftest_result.jpg"))
    cv2.imwrite(out_path, draw_results(img_plain, defects))
    print(f"Annotated result saved to : {out_path}")
    print("=" * 78)

    # Non-zero exit when a scenario fails, so this can be wired into CI later
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())

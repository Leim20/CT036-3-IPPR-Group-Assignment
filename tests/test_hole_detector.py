"""Focused regression tests for the explainable tearing detector."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from defect_detection import detect_tearing
from pipeline import process_image


class TearingDetectorTests(unittest.TestCase):
    BACKGROUND = (45, 190, 80)   # green BGR
    GLOVE = (200, 90, 35)        # blue BGR
    SKIN = (90, 145, 205)        # skin-like BGR

    def make_scene(self, skin_center=None):
        image = np.full((400, 400, 3), self.BACKGROUND, dtype=np.uint8)
        mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (350, 350), 255, cv2.FILLED)
        image[mask > 0] = self.GLOVE
        if skin_center is not None:
            cv2.circle(image, skin_center, 22, self.SKIN, cv2.FILLED)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_color = lab[0, 0]
        return image, mask, bg_color

    def make_finger_scene(self, exposed_finger=False):
        image = np.full((500, 500, 3), self.BACKGROUND, dtype=np.uint8)
        mask = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(mask, (70, 250), (430, 460), 255, cv2.FILLED)
        for x1, top in (
            (90, 110), (160, 70), (230, 80), (300, 70), (370, 110)
        ):
            if exposed_finger and x1 == 230:
                continue
            cv2.rectangle(mask, (x1, top), (x1 + 50, 270), 255, cv2.FILLED)
        image[mask > 0] = self.GLOVE
        if exposed_finger:
            # A complete uncovered finger is long and attaches at the palm.
            cv2.rectangle(image, (235, 70), (275, 255), self.SKIN, cv2.FILLED)
        else:
            # A torn glove fingertip exposes only a short skin cap immediately
            # outside the material silhouette.
            cv2.rectangle(image, (235, 62), (275, 84), self.SKIN, cv2.FILLED)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        return image, mask, lab[0, 0]

    def make_narrow_finger_pad_scene(self):
        image = np.full((600, 500, 3), self.BACKGROUND, dtype=np.uint8)
        mask = np.zeros((600, 500), dtype=np.uint8)
        cv2.rectangle(mask, (50, 300), (450, 560), 255, cv2.FILLED)
        for x1, top in (
            (90, 145), (160, 110), (230, 90), (300, 110), (370, 145)
        ):
            cv2.rectangle(mask, (x1, top), (x1 + 40, 320), 255, cv2.FILLED)
        image[mask > 0] = self.GLOVE
        cv2.ellipse(image, (250, 190), (10, 18), 0, 0, 360,
                    self.SKIN, cv2.FILLED)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        return image, mask, lab[0, 0]

    def test_enclosed_skin_region_is_tearing(self):
        image, mask, bg_color = self.make_scene((200, 200))
        result = detect_tearing(
            image, mask, mask, bg_color, img_plain=image, material="nitrile"
        )
        self.assertEqual(1, len(result))
        self.assertEqual("Tearing", result[0][0])

    def test_clean_glove_has_no_tearing(self):
        image, mask, bg_color = self.make_scene()
        result = detect_tearing(
            image, mask, mask, bg_color, img_plain=image, material="nitrile"
        )
        self.assertEqual([], result)

    def test_skin_region_touching_outer_edge_is_not_enclosed_tearing(self):
        image, mask, bg_color = self.make_scene((52, 200))
        result = detect_tearing(
            image, mask, mask, bg_color, img_plain=image, material="nitrile"
        )
        self.assertEqual([], result)

    def test_shallow_skin_cap_at_fingertip_is_tearing(self):
        image, mask, bg_color = self.make_finger_scene()

        result = detect_tearing(
            image, mask, mask, bg_color, img_plain=image, material="nitrile"
        )

        self.assertEqual(1, len(result))
        self.assertEqual("Tearing", result[0].name)
        self.assertLess(result[0].box[3], result[0].box[2])
        self.assertGreater(cv2.countNonZero(result[0].mask), 400)

    def test_skin_hole_on_narrow_finger_pad_uses_local_depth(self):
        image, mask, bg_color = self.make_narrow_finger_pad_scene()
        distance = cv2.distanceTransform(
            (mask > 0).astype(np.uint8), cv2.DIST_L2, 5
        )
        patch = np.zeros_like(mask)
        cv2.ellipse(patch, (250, 190), (10, 18), 0, 0, 360,
                    255, cv2.FILLED)
        global_depth_ratio = (
            float(distance[patch > 0].max()) / float(distance.max())
        )
        self.assertLess(global_depth_ratio, 0.20)

        result = detect_tearing(
            image, mask, mask, bg_color,
            img_plain=image, material="latex_foam"
        )

        self.assertEqual(1, len(result))
        self.assertEqual("Tearing", result[0].name)
        self.assertGreater(cv2.countNonZero(result[0].mask), 400)

    def test_long_exposed_finger_is_not_fingertip_tearing(self):
        image, mask, bg_color = self.make_finger_scene(exposed_finger=True)

        result = detect_tearing(
            image, mask, mask, bg_color, img_plain=image, material="nitrile"
        )

        self.assertEqual([], result)

    def test_real_cotton_fingertip_caps_are_recovered(self):
        for image_name in (
            "white_cotton_025.jpg",
            "white_cotton_026.jpg",
            "white_cotton_027.jpg",
        ):
            with self.subTest(image=image_name):
                result = process_image(
                    str(
                        ROOT / "dataset" / "raw" / "tearing" / "cotton"
                        / image_name
                    ),
                    detectors=[detect_tearing],
                )
                detections = [
                    item for item in result["defects"]
                    if item.name == "Tearing"
                ]
                self.assertGreaterEqual(len(detections), 1)
                self.assertTrue(any(
                    item.mask is not None
                    and cv2.countNonZero(item.mask) >= 700
                    for item in detections
                ))

    def test_real_long_fingers_and_clean_glove_are_not_new_tears(self):
        cases = (
            "finger_not_enough/cotton/blue_cotton_051.jpg",
            "finger_not_enough/cotton/white_cotton_050.jpg",
            "finger_not_enough/latex_foam/latex_foam_038.jpg",
            "good/latex_foam/latex_foam_006.jpg",
        )
        for relative_path in cases:
            with self.subTest(image=relative_path):
                result = process_image(
                    str(ROOT / "dataset" / "raw" / relative_path),
                    detectors=[detect_tearing],
                )
                self.assertEqual([], result["defects"])

    def test_real_finger_pad_tears_are_recovered(self):
        cases = {
            "tearing/latex_foam/latex_foam_029.jpg": {
                (447, 327, 26, 65),
            },
            "tearing/latex_foam/latex_foam_020.jpg": {
                (406, 292, 19, 28),
                (545, 438, 24, 93),
            },
            "tearing/cotton/white_cotton_040.jpg": {
                (686, 433, 54, 33),
            },
        }
        for relative_path, expected_boxes in cases.items():
            with self.subTest(image=relative_path):
                result = process_image(
                    str(ROOT / "dataset" / "raw" / relative_path),
                    detectors=[detect_tearing],
                )
                detected_boxes = {
                    item.box for item in result["defects"]
                    if item.name == "Tearing"
                }
                self.assertTrue(expected_boxes.issubset(detected_boxes))

    def test_large_deep_low_contrast_cotton_tear_is_retained(self):
        image = np.full((500, 500, 3), self.BACKGROUND, dtype=np.uint8)
        mask = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (450, 450), 255, cv2.FILLED)
        image[mask > 0] = (210, 210, 210)
        # Passes the skin-colour rule but sits below cotton's normal 50-point
        # local-contrast threshold, matching the large photographed palm tear.
        cv2.circle(image, (250, 220), 60, (160, 160, 210), cv2.FILLED)
        bg_color = cv2.cvtColor(
            image, cv2.COLOR_BGR2LAB
        ).astype(np.float32)[0, 0]

        result = detect_tearing(
            image, mask, mask, bg_color, img_plain=image, material="cotton"
        )

        self.assertEqual(1, len(result))
        self.assertGreater(cv2.countNonZero(result[0].mask), 10000)

    def test_small_low_contrast_cotton_patch_stays_rejected(self):
        image = np.full((500, 500, 3), self.BACKGROUND, dtype=np.uint8)
        mask = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (450, 450), 255, cv2.FILLED)
        image[mask > 0] = (210, 210, 210)
        cv2.circle(image, (250, 220), 15, (160, 160, 210), cv2.FILLED)
        bg_color = cv2.cvtColor(
            image, cv2.COLOR_BGR2LAB
        ).astype(np.float32)[0, 0]

        result = detect_tearing(
            image, mask, mask, bg_color, img_plain=image, material="cotton"
        )

        self.assertEqual([], result)


if __name__ == "__main__":
    unittest.main()

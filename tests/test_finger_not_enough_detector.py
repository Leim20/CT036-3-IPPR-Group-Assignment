"""Focused regression tests for the Finger Not Enough detector."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from defect_detection import affected_area_percentage, detect_finger_not_enough
from pipeline import process_image


class FingerNotEnoughDetectorTests(unittest.TestCase):
    BACKGROUND = (45, 190, 80)   # green BGR
    GLOVE = (200, 90, 35)        # blue BGR
    SKIN = (90, 145, 205)        # skin-like BGR

    def make_scene(self, mask, skin_box=None):
        image = np.full((*mask.shape, 3), self.BACKGROUND, dtype=np.uint8)
        image[mask > 0] = self.GLOVE
        if skin_box is not None:
            x1, y1, x2, y2 = skin_box
            cv2.rectangle(image, (x1, y1), (x2, y2), self.SKIN, cv2.FILLED)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        return image, lab[0, 0]

    def detect(self, image, mask, bg_color, material="latex_foam"):
        return detect_finger_not_enough(
            image,
            mask,
            mask,
            bg_color,
            img_plain=image,
            material=material,
        )

    def test_curled_finger_space_is_finger_not_enough(self):
        mask = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(mask, (100, 240), (400, 450), 255, cv2.FILLED)
        cv2.rectangle(mask, (150, 60), (220, 240), 255, cv2.FILLED)
        cv2.rectangle(mask, (280, 60), (350, 240), 255, cv2.FILLED)
        image, bg_color = self.make_scene(mask)

        result = self.detect(image, mask, bg_color)

        self.assertEqual(1, len(result))
        self.assertEqual("Finger Not Enough", result[0][0])
        self.assertGreater(result[0].evidence, 0.0)

    def test_skin_coloured_edge_patch_without_shape_change_is_not_missing_finger(self):
        mask = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(mask, (100, 50), (400, 450), 255, cv2.FILLED)
        # Deliberately elongated like a finger, but the glove boundary touches
        # only its top edge instead of wrapping around it like a bare fingertip.
        image, bg_color = self.make_scene(mask, skin_box=(220, 50, 275, 165))

        result = self.detect(image, mask, bg_color)

        self.assertEqual([], result)

    def test_multiple_exposed_fingertips_return_separate_regions(self):
        """Bare skin still forms finger columns; every exposed tip is a region."""
        mask = np.zeros((600, 500), dtype=np.uint8)
        cv2.rectangle(mask, (80, 270), (410, 550), 255, cv2.FILLED)
        for x1, top, x2 in (
            (95, 120, 150),
            (170, 60, 225),
            (245, 80, 300),
            (320, 130, 375),
        ):
            cv2.rectangle(mask, (x1, top), (x2, 300), 255, cv2.FILLED)
        cv2.rectangle(mask, (385, 210), (470, 335), 255, cv2.FILLED)
        image, bg_color = self.make_scene(mask)
        skin_boxes = (
            (95, 120, 150, 235),
            (170, 60, 225, 205),
            (320, 130, 375, 245),
        )
        for skin_box in skin_boxes:
            cv2.rectangle(
                image, skin_box[:2], skin_box[2:], self.SKIN, cv2.FILLED
            )

        result = self.detect(image, mask, bg_color, material="nitrile")

        self.assertEqual(3, len(result))
        self.assertEqual([95, 170, 320], [item.box[0] for item in result])
        self.assertTrue(all(item.name == "Finger Not Enough" for item in result))
        self.assertTrue(all(cv2.countNonZero(item.mask) > 0 for item in result))

    def test_multiple_external_fingers_attach_to_cotton_material_mask(self):
        """Skin excluded from material segmentation must still be detected."""
        mask = np.zeros((600, 500), dtype=np.uint8)
        cv2.rectangle(mask, (80, 270), (410, 550), 255, cv2.FILLED)
        cv2.rectangle(mask, (245, 80), (300, 300), 255, cv2.FILLED)
        image, bg_color = self.make_scene(mask)
        for skin_box in (
            (95, 120, 150, 280),
            (170, 60, 225, 280),
            (320, 130, 375, 280),
            (400, 210, 460, 320),
        ):
            cv2.rectangle(
                image, skin_box[:2], skin_box[2:], self.SKIN, cv2.FILLED
            )

        result = self.detect(image, mask, bg_color, material="cotton")

        self.assertEqual(4, len(result))
        self.assertTrue(all(cv2.countNonZero(item.mask) > 0 for item in result))
        self.assertGreater(affected_area_percentage(result, mask), 0.0)

    def test_enclosed_skin_patch_is_not_a_missing_finger(self):
        mask = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(mask, (100, 50), (400, 450), 255, cv2.FILLED)
        image, bg_color = self.make_scene(mask, skin_box=(220, 150, 275, 205))

        result = self.detect(image, mask, bg_color)

        self.assertEqual([], result)

    def test_clean_uniform_glove_has_no_finger_defect(self):
        mask = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(mask, (100, 50), (400, 450), 255, cv2.FILLED)
        image, bg_color = self.make_scene(mask)

        result = self.detect(image, mask, bg_color)

        self.assertEqual([], result)

    def test_clean_horizontal_five_finger_glove_has_no_finger_defect(self):
        """A broad wrist-to-hull gap is not a gap between two fingers."""
        mask = np.zeros((500, 800), dtype=np.uint8)
        cv2.rectangle(mask, (0, 140), (440, 370), 255, cv2.FILLED)
        for x1, y1, x2, y2 in (
            (380, 65, 650, 140),
            (430, 140, 730, 180),
            (430, 190, 760, 230),
            (430, 240, 735, 280),
            (410, 290, 680, 335),
        ):
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, cv2.FILLED)
        image, bg_color = self.make_scene(mask)

        result = self.detect(image, mask, bg_color)

        self.assertEqual([], result)

    def test_visible_wrist_is_not_an_exposed_finger(self):
        mask = np.zeros((500, 800), dtype=np.uint8)
        cv2.rectangle(mask, (0, 140), (440, 370), 255, cv2.FILLED)
        for x1, y1, x2, y2 in (
            (380, 65, 650, 140),
            (430, 140, 730, 180),
            (430, 190, 760, 230),
            (430, 240, 735, 280),
            (410, 290, 680, 335),
        ):
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, cv2.FILLED)
        image, bg_color = self.make_scene(mask, skin_box=(0, 140, 170, 370))

        result = self.detect(image, mask, bg_color)

        self.assertEqual([], result)

    def test_horizontal_missing_finger_is_orientation_normalised(self):
        upright = np.zeros((500, 350), dtype=np.uint8)
        cv2.rectangle(upright, (25, 240), (325, 450), 255, cv2.FILLED)
        cv2.rectangle(upright, (75, 60), (145, 240), 255, cv2.FILLED)
        cv2.rectangle(upright, (205, 60), (275, 240), 255, cv2.FILLED)
        mask = cv2.rotate(upright, cv2.ROTATE_90_CLOCKWISE)
        image, bg_color = self.make_scene(mask)

        result = self.detect(image, mask, bg_color)

        self.assertEqual(1, len(result))
        self.assertEqual("Finger Not Enough", result[0].name)
        self.assertGreater(result[0].evidence, 0.0)

    def test_real_folded_finger_regressions_do_not_require_exact_counts(self):
        cases = (
            "finger_not_enough/cotton/white_cotton_011.jpg",
            "finger_not_enough/latex_foam/latex_foam_010.jpg",
            "finger_not_enough/latex_foam/latex_foam_018.jpg",
        )
        for relative_path in cases:
            with self.subTest(image=relative_path):
                result = process_image(
                    str(ROOT / "dataset" / "raw" / relative_path),
                    detectors=[detect_finger_not_enough],
                )
                detections = [
                    item for item in result["defects"]
                    if item.name == "Finger Not Enough"
                ]
                self.assertTrue(detections)
                self.assertTrue(
                    any(cv2.countNonZero(item.mask) > 0 for item in detections)
                )


if __name__ == "__main__":
    unittest.main()

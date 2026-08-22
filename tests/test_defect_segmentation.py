"""Regression tests for lecturer-required coloured defect segmentation."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defect_detection import build_defect_masks, draw_results


class DefectSegmentationTests(unittest.TestCase):
    BACKGROUND = (45, 190, 80)
    GLOVE = (205, 85, 35)
    PALE_BLUE = (205, 170, 155)
    SKIN = (90, 145, 205)

    @staticmethod
    def background_lab(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]

    def test_hole_mask_colours_only_the_opening(self):
        image = np.full((400, 400, 3), self.BACKGROUND, np.uint8)
        glove = np.zeros((400, 400), np.uint8)
        cv2.rectangle(glove, (50, 50), (350, 350), 255, cv2.FILLED)
        image[glove > 0] = self.GLOVE
        cv2.circle(image, (200, 200), 22, self.SKIN, cv2.FILLED)

        masks = build_defect_masks(
            image,
            glove,
            glove,
            self.background_lab(image),
            [("Hole", (178, 178, 45, 45))],
            img_plain=image,
            material="nitrile",
        )

        self.assertGreater(masks[0][200, 200], 0)
        self.assertEqual(0, masks[0][100, 100])

    def test_completely_missing_finger_uses_box_fallback(self):
        image = np.full((500, 500, 3), self.BACKGROUND, np.uint8)
        glove = np.zeros((500, 500), np.uint8)
        cv2.rectangle(glove, (100, 240), (400, 450), 255, cv2.FILLED)
        cv2.rectangle(glove, (150, 60), (220, 240), 255, cv2.FILLED)
        cv2.rectangle(glove, (280, 60), (350, 240), 255, cv2.FILLED)
        image[glove > 0] = self.GLOVE

        masks = build_defect_masks(
            image,
            glove,
            glove,
            self.background_lab(image),
            [("Finger Not Enough", (100, 60, 301, 313))],
            img_plain=image,
            material="latex_foam",
        )

        self.assertEqual(0, cv2.countNonZero(masks[0]))

    def test_visible_curled_finger_colours_the_finger_material(self):
        image = np.full((500, 500, 3), self.BACKGROUND, np.uint8)
        glove = np.zeros((500, 500), np.uint8)
        cv2.rectangle(glove, (100, 240), (420, 450), 255, cv2.FILLED)
        cv2.rectangle(glove, (150, 50), (220, 260), 255, cv2.FILLED)
        cv2.rectangle(glove, (350, 180), (420, 300), 255, cv2.FILLED)
        cv2.rectangle(glove, (250, 160), (300, 270), 255, cv2.FILLED)
        image[glove > 0] = self.GLOVE

        contour = max(
            cv2.findContours(
                glove, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0],
            key=cv2.contourArea,
        )
        hull_mask = np.zeros_like(glove)
        cv2.drawContours(
            hull_mask, [cv2.convexHull(contour)], -1, 255, cv2.FILLED
        )
        gap_contours = cv2.findContours(
            cv2.subtract(hull_mask, glove),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )[0]
        gap_box = cv2.boundingRect(max(gap_contours, key=cv2.contourArea))

        masks = build_defect_masks(
            image,
            glove,
            glove,
            self.background_lab(image),
            [("Finger Not Enough", gap_box)],
            img_plain=image,
            material="latex_foam",
        )

        self.assertGreater(masks[0][200, 275], 0)
        self.assertEqual(0, masks[0][100, 180])

    def test_thin_mask_marks_pale_patch_not_opaque_glove(self):
        image = np.full((420, 520, 3), self.BACKGROUND, np.uint8)
        glove = np.zeros((420, 520), np.uint8)
        cv2.rectangle(glove, (100, 55), (420, 365), 255, cv2.FILLED)
        image[glove > 0] = self.GLOVE
        cv2.rectangle(image, (155, 105), (365, 315), self.PALE_BLUE, cv2.FILLED)

        masks = build_defect_masks(
            image,
            glove,
            glove,
            self.background_lab(image),
            [("Thin / Overstretched", (100, 55, 321, 311))],
            img_plain=image,
            material="cotton",
        )

        self.assertGreater(masks[0][200, 250], 0)
        self.assertEqual(0, masks[0][80, 120])

    def test_overlay_changes_masked_pixels_without_filling_box(self):
        image = np.full((100, 100, 3), 100, np.uint8)
        mask = np.zeros((100, 100), np.uint8)
        cv2.circle(mask, (50, 50), 10, 255, cv2.FILLED)
        result = draw_results(
            image,
            [("Hole", (40, 40, 21, 21))],
            [mask],
        )

        self.assertFalse(np.array_equal(result[50, 50], image[50, 50]))
        self.assertTrue(np.array_equal(result[40, 40], image[40, 40]))


if __name__ == "__main__":
    unittest.main()

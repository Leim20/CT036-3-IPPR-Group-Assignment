"""Focused regression tests for the explainable hole detector."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from defect_detection import detect_holes


class HoleDetectorTests(unittest.TestCase):
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

    def test_enclosed_skin_region_is_hole(self):
        image, mask, bg_color = self.make_scene((200, 200))
        result = detect_holes(
            image, mask, mask, bg_color, img_plain=image, material="nitrile"
        )
        self.assertEqual(1, len(result))
        self.assertEqual("Hole", result[0][0])

    def test_clean_glove_has_no_hole(self):
        image, mask, bg_color = self.make_scene()
        result = detect_holes(
            image, mask, mask, bg_color, img_plain=image, material="nitrile"
        )
        self.assertEqual([], result)

    def test_skin_region_touching_outer_edge_is_not_enclosed_hole(self):
        image, mask, bg_color = self.make_scene((52, 200))
        result = detect_holes(
            image, mask, mask, bg_color, img_plain=image, material="nitrile"
        )
        self.assertEqual([], result)


if __name__ == "__main__":
    unittest.main()

"""Focused regression tests for the Finger Not Enough detector."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from defect_detection import detect_finger_not_enough


class FingerNotEnoughDetectorTests(unittest.TestCase):
    BACKGROUND = (45, 190, 80)   # green BGR
    GLOVE = (200, 90, 35)        # blue BGR
    SKIN = (90, 145, 205)        # skin-like BGR

    def make_scene(self, mask, skin_box=None):
        image = np.full((500, 500, 3), self.BACKGROUND, dtype=np.uint8)
        image[mask > 0] = self.GLOVE
        if skin_box is not None:
            x1, y1, x2, y2 = skin_box
            cv2.rectangle(image, (x1, y1), (x2, y2), self.SKIN, cv2.FILLED)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        return image, lab[0, 0]

    def detect(self, image, mask, bg_color):
        return detect_finger_not_enough(
            image,
            mask,
            mask,
            bg_color,
            img_plain=image,
            material="latex_foam",
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

    def test_exposed_skin_in_finger_zone_is_finger_not_enough(self):
        mask = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(mask, (100, 50), (400, 450), 255, cv2.FILLED)
        image, bg_color = self.make_scene(mask, skin_box=(220, 50, 275, 105))

        result = self.detect(image, mask, bg_color)

        self.assertEqual(1, len(result))
        self.assertEqual("Finger Not Enough", result[0][0])

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


if __name__ == "__main__":
    unittest.main()

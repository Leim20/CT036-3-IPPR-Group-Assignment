"""Focused regression tests for the Thin / Overstretched detector."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defect_detection import detect_thin_area


class ThinDetectorTests(unittest.TestCase):
    OPAQUE_BLUE = (205, 85, 35)
    PALE_BLUE = (205, 170, 155)
    SKIN = (90, 145, 205)

    def make_scene(self, glove_color):
        image = np.full((420, 520, 3), (30, 150, 30), np.uint8)
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.rectangle(mask, (100, 55), (420, 365), 255, cv2.FILLED)
        image[mask > 0] = glove_color
        return image, mask

    def detect(self, image, mask, material):
        return detect_thin_area(
            image,
            mask,
            mask,
            np.array([0, 0, 0], np.float32),
            img_plain=image,
            material=material,
        )

    def test_opaque_blue_cotton_is_not_thin(self):
        image, mask = self.make_scene(self.OPAQUE_BLUE)
        self.assertEqual([], self.detect(image, mask, "cotton"))

    def test_broad_pale_area_in_blue_cotton_is_thin(self):
        image, mask = self.make_scene(self.OPAQUE_BLUE)
        cv2.rectangle(image, (155, 105), (365, 315), self.PALE_BLUE, cv2.FILLED)
        hits = self.detect(image, mask, "cotton")
        self.assertEqual("Thin / Overstretched", hits[0][0])

    def test_dispersed_skin_openings_in_white_cotton_are_thin(self):
        image, mask = self.make_scene((235, 235, 235))
        for y in range(90, 340, 35):
            for x in range(135, 400, 38):
                cv2.circle(image, (x, y), 4, self.SKIN, cv2.FILLED)
        hits = self.detect(image, mask, "cotton")
        self.assertEqual("Thin / Overstretched", hits[0][0])

    def test_one_white_cotton_hole_is_not_diffuse_thinning(self):
        image, mask = self.make_scene((235, 235, 235))
        cv2.circle(image, (260, 210), 18, self.SKIN, cv2.FILLED)
        self.assertEqual([], self.detect(image, mask, "cotton"))

    def test_broad_pale_area_in_nitrile_is_thin(self):
        image, mask = self.make_scene(self.OPAQUE_BLUE)
        cv2.rectangle(image, (120, 75), (400, 345), self.PALE_BLUE, cv2.FILLED)
        hits = self.detect(image, mask, "nitrile")
        self.assertEqual("Thin / Overstretched", hits[0][0])


if __name__ == "__main__":
    unittest.main()

"""Focused regression tests for the Thin / Overstretched detector."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defect_detection import detect_thin_area
from pipeline import infer_material


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

    def test_one_white_cotton_tear_is_not_diffuse_thinning(self):
        image, mask = self.make_scene((235, 235, 235))
        cv2.circle(image, (260, 210), 18, self.SKIN, cv2.FILLED)
        self.assertEqual([], self.detect(image, mask, "cotton"))

    def test_broad_pale_area_in_nitrile_is_thin(self):
        image, mask = self.make_scene(self.OPAQUE_BLUE)
        cv2.rectangle(image, (120, 75), (400, 345), self.PALE_BLUE, cv2.FILLED)
        hits = self.detect(image, mask, "nitrile")
        self.assertEqual("Thin / Overstretched", hits[0][0])

    def test_pale_nitrile_isolated_from_blue_purple_background(self):
        """A hue-adjacent backdrop must not replace the glove material ROI."""
        image = np.full((420, 520, 3), (116, 81, 81), np.uint8)
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.rectangle(mask, (100, 55), (420, 365), 255, cv2.FILLED)
        image[mask > 0] = self.PALE_BLUE

        hits = self.detect(image, mask, "nitrile")

        self.assertEqual("Thin / Overstretched", hits[0][0])

    def test_flat_dataset_filename_is_not_used_as_material_metadata(self):
        self.assertIsNone(
            infer_material("dataset/raw/nitrile_005.jpg", "raw")
        )

    def test_material_folder_remains_trusted_metadata(self):
        self.assertEqual(
            "nitrile",
            infer_material("dataset/raw/thin/nitrile/example.jpg", "nitrile"),
        )

    def test_plain_latex_folder_remains_trusted_metadata(self):
        self.assertEqual(
            "latex",
            infer_material(
                "dataset/raw/Plastic Contamination/latex/example.jpg",
                "latex",
            ),
        )

    def test_auto_mode_detects_pale_nitrile_without_metadata(self):
        image = np.full((420, 520, 3), (116, 81, 81), np.uint8)
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.rectangle(mask, (100, 55), (420, 365), 255, cv2.FILLED)
        image[mask > 0] = self.PALE_BLUE

        hits = self.detect(image, mask, None)

        self.assertEqual("Thin / Overstretched", hits[0].name)
        self.assertGreater(hits[0].evidence, 0.0)

    def test_auto_mode_rejects_opaque_blue_glove(self):
        image, mask = self.make_scene(self.OPAQUE_BLUE)
        self.assertEqual([], self.detect(image, mask, None))


if __name__ == "__main__":
    unittest.main()

"""Regression tests for material-specific plastic-contamination rules."""
import sys
import unittest
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defect_detection import detect_plastic_contamination  # noqa: E402
from pipeline import process_image  # noqa: E402


class PlasticContaminationDetectorTests(unittest.TestCase):
    def test_latex_central_film_is_not_rejected_by_generic_area_cap(self):
        path = (
            PROJECT_ROOT
            / "dataset"
            / "raw"
            / "Plastic Contamination"
            / "latex"
            / "IMG_20260823_111204.jpg"
        )
        if not path.exists():
            self.skipTest("Labelled latex regression image is not available")

        result = process_image(
            path,
            material="latex",
            detectors=[detect_plastic_contamination],
        )

        self.assertTrue(result["glove_found"])
        self.assertEqual([], result["errors"])
        self.assertEqual(4, len(result["defects"]))
        central_roi = (350, 155, 195, 125)
        cx, cy, cw, ch = central_roi
        central = result["defect_mask"][cy:cy + ch, cx:cx + cw]
        self.assertGreater(cv2.countNonZero(central), 3000)

    def test_nitrile_metadata_keeps_the_generic_branch(self):
        path = (
            PROJECT_ROOT
            / "dataset"
            / "raw"
            / "Plastic Contamination"
            / "nitrile"
            / "IMG_20260822_191140.jpg"
        )
        if not path.exists():
            self.skipTest("Local nitrile comparison image is not available")

        generic = process_image(
            path,
            material=None,
            detectors=[detect_plastic_contamination],
        )
        nitrile = process_image(
            path,
            material="nitrile",
            detectors=[detect_plastic_contamination],
        )

        self.assertEqual(
            [(item.name, item.box, item.evidence) for item in generic["defects"]],
            [(item.name, item.box, item.evidence) for item in nitrile["defects"]],
        )
        self.assertEqual(
            0,
            cv2.countNonZero(
                cv2.bitwise_xor(generic["defect_mask"], nitrile["defect_mask"])
            ),
        )


if __name__ == "__main__":
    unittest.main()

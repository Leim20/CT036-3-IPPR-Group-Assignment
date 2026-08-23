"""Regression tests for strict detector selection in the GUI."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from defect_detection import detect_open_tears, detect_tearing
from gui import GloveDefectApp


class DetectorSelectionTests(unittest.TestCase):
    def selected(self, label):
        app = SimpleNamespace(
            detector_var=SimpleNamespace(get=lambda: label),
            detector_by_label={
                "Tearing": detect_tearing,
                "Open Tear": detect_open_tears,
            },
        )
        return GloveDefectApp._selected_detectors(app)

    def test_tearing_mode_does_not_run_open_tear_detector(self):
        self.assertEqual([detect_tearing], self.selected("Tearing"))

    def test_open_tear_remains_separately_selectable(self):
        self.assertEqual([detect_open_tears], self.selected("Open Tear"))


if __name__ == "__main__":
    unittest.main()

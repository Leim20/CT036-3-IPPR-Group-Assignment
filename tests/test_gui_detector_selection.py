"""Regression tests for strict detector selection in the GUI."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from defect_detection import DETECTORS, detect_open_tears, detect_tearing
from evaluate import LABEL_MAP
from gui import DEFECT_OPTIONS, GloveDefectApp


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

    def test_side_tear_detector_is_not_registered_or_selectable(self):
        self.assertNotIn("detect_side_tear", [det.__name__ for det in DETECTORS])
        self.assertNotIn("Side Tear", DEFECT_OPTIONS)

    def test_legacy_side_tear_folders_use_open_tear_label(self):
        self.assertEqual("Open Tear", LABEL_MAP["side_tear"])
        self.assertEqual("Open Tear", LABEL_MAP["edge_tear"])


if __name__ == "__main__":
    unittest.main()

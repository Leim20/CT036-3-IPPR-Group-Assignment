# -*- coding: utf-8 -*-
"""
Batch evaluation script -- runs the full dataset and tallies detected /
missed / false-positive counts for every defect type:

    .venv\\Scripts\\python src\\evaluate.py
    .venv\\Scripts\\python src\\evaluate.py --save-failures   (also saves the failing images)

Maps onto these assignment requirements:
  Sec 4 "test the system to evaluate the accuracy of the proposed techniques"
  Sec 5 "describe the results of testing using various test images"
  Sec 5 "critical analysis for cases of images that fail"
  Sec 6 Experimental Results & Critical analysis = 40% of the marks

===== Dataset layout (the ground truth comes from folder names, no manual
bounding-box annotation needed) =====

    dataset/raw/
      hole/                  <- defect name
        cotton/     *.jpg    <- material name
        nitrile/    *.jpg
        latex_foam/ *.jpg
      finger_not_enough/
        cotton/     *.jpg
        nitrile/    *.jpg
      thin/
        cotton/     *.jpg
        nitrile/    *.jpg
      good/                  <- clean gloves, used to measure false positives
        cotton/     *.jpg

Multiple labels can be joined with "+", for example
``hole+thin/cotton/example.jpg``.

The folder-name -> defect-label lookup table is LABEL_MAP below.
Whenever a team member adds a new detector, add a matching row to it.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline import process_image

# folder name (lowercase) -> the defect label a detector actually returns
# The left side can be whatever you like; the right side MUST exactly
# match the string returned by the detector.
LABEL_MAP = {
    "hole": "Hole",
    "puncture": "Hole",
    "tear": "Open Tear",
    "open_tear": "Open Tear",
    "fingertip_tear": "Open Tear",
    "stain": "Stain",
    "dirty": "Stain",
    "finger_not_enough": "Finger Not Enough",
    "missing_finger": "Finger Not Enough",  # legacy folder-name alias
    "thin": "Thin / Overstretched",
    "overstretched": "Thin / Overstretched",
    "thin_overstretched": "Thin / Overstretched",
    # add new detectors here as team members finish them, e.g. "wrinkle": "Wrinkle",
}

GOOD_DIR_NAMES = {"good", "ok", "normal", "pass"}   # folder names that mean "clean glove"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def parse_expected(folder_name):
    """folder name -> the set of expected defect labels. Returns an empty
    set for a "good/clean" folder."""
    name = folder_name.strip().lower()
    if name in GOOD_DIR_NAMES:
        return set()
    labels = set()
    for part in name.split("+"):
        part = part.strip()
        if part in LABEL_MAP:
            labels.add(LABEL_MAP[part])
        else:
            labels.add(f"<unregistered:{part}>")   # a reminder to add this to LABEL_MAP
    return labels


def collect_images(root):
    """Scan dataset/raw and return [(image path, material, expected defect set), ...]."""
    items = []
    if not os.path.isdir(root):
        return items
    for defect_dir in sorted(os.listdir(root)):
        defect_path = os.path.join(root, defect_dir)
        if not os.path.isdir(defect_path):
            continue
        expected = parse_expected(defect_dir)
        for material in sorted(os.listdir(defect_path)):
            material_path = os.path.join(defect_path, material)
            if not os.path.isdir(material_path):
                continue
            for fn in sorted(os.listdir(material_path)):
                if os.path.splitext(fn)[1].lower() in IMG_EXT:
                    items.append((os.path.join(material_path, fn), material, expected))
    return items


def evaluate(root, save_failures=False):
    items = collect_images(root)
    if not items:
        print(f"No images found under {root}.")
        print(__doc__.split("===== Dataset layout")[1])
        return 1

    # counters
    stat = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})   # per defect type
    mat_stat = defaultdict(lambda: {"hit": 0, "need": 0})     # per-material recall
    rows = []                       # rows written to the CSV
    failures = []                   # failing images, for the report's failure-case analysis
    detector_errors = defaultdict(int)   # which detector failed, and how many times
    n_good = n_good_fp = n_noglove = 0

    fail_dir = os.path.join(os.path.dirname(root), "failures")
    if save_failures:
        os.makedirs(fail_dir, exist_ok=True)

    for path, material, expected in items:
        result = process_image(path, material=material)
        if result["original_image"] is None:
            rows.append([path, material, "|".join(sorted(expected)), "", "failed to read image"])
            failures.append((path, "failed to read image"))
            continue

        if not result["glove_found"]:
            n_noglove += 1
            detected = set()
            defects = []
            verdict = "segmentation failed (no glove detected)"
            failures.append((path, verdict))
        else:
            defects = result["defects"]
            errs = result["errors"]
            detected = {n for n, _ in defects}
            verdict = "correct" if detected == expected else "mismatch"
            # a detector crashing is a code bug, not an accuracy issue --
            # it must be flagged separately
            for err in errs:
                detector_errors[err.split(":")[0]] += 1

        # --- per-class tally ---
        for label in expected:
            if label in detected:
                stat[label]["tp"] += 1
            else:
                stat[label]["fn"] += 1
        for label in detected - expected:
            stat[label]["fp"] += 1

        # --- per-material (whether this image's defects were all detected) ---
        if expected:
            mat_stat[material]["need"] += 1
            if expected <= detected:
                mat_stat[material]["hit"] += 1

        # --- false positives on clean gloves ---
        if not expected:
            n_good += 1
            if detected:
                n_good_fp += 1

        if verdict != "correct":
            if verdict == "mismatch":
                failures.append((path, f"expected {sorted(expected)} got {sorted(detected)}"))
            if save_failures and result["glove_found"]:
                out = os.path.join(fail_dir, material + "_" + os.path.basename(path))
                cv2.imwrite(out, result["result_image"])

        rows.append([path, material, "|".join(sorted(expected)),
                     "|".join(sorted(detected)), verdict])

    # ================= print the results =================
    print("=" * 84)
    print(f"Dataset: {root}    {len(items)} image(s) total")
    print("-" * 84)
    print(f"{'Defect Type':<20}{'Total':>8}{'TP':>7}{'FN':>7}{'FP':>7}"
          f"{'Recall':>9}{'Precision':>11}{'F1':>8}")
    print("-" * 84)
    for label in sorted(stat):
        s = stat[label]
        tp, fn, fp = s["tp"], s["fn"], s["fp"]
        recall = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
        print(f"{label:<20}{tp+fn:>8}{tp:>7}{fn:>7}{fp:>7}"
              f"{recall:>9.1%}{prec:>11.1%}{f1:>8.2f}")

    print("-" * 84)
    if mat_stat:
        print("By material (were all of this image's defects detected):")
        for m in sorted(mat_stat):
            v = mat_stat[m]
            if v["need"]:
                print(f"  {m:<16}{v['hit']}/{v['need']}   {v['hit']/v['need']:.1%}")
            else:
                print(f"  {m:<16}(no defective images)")
    if n_good:
        print(f"False-positive rate on clean gloves: {n_good_fp}/{n_good} = {n_good_fp/n_good:.1%}")
    if n_noglove:
        print(f"Segmentation failures (no glove detected): {n_noglove} image(s)")
    if detector_errors:
        print("-" * 84)
        print("Detector(s) raised runtime errors (this is a code bug -- fix it "
             "before trusting the accuracy numbers above):")
        for name, count in sorted(detector_errors.items(), key=lambda kv: -kv[1]):
            print(f"    {name}  failed {count} time(s)")

    # ================= failure case list (for the report) =================
    if failures:
        print("-" * 84)
        print(f"{len(failures)} failure case(s) (use these for the report's critical analysis):")
        for path, why in failures[:15]:
            print(f"  {os.path.relpath(path, root)}  ->  {why}")
        if len(failures) > 15:
            print(f"  ... {len(failures)-15} more, see the CSV")

    # ================= write the CSV =================
    csv_path = os.path.join(os.path.dirname(root), "evaluation_result.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Image", "Material", "Expected Defects", "Detected Defects", "Verdict"])
        w.writerows(rows)
    print("-" * 84)
    print(f"Per-image results saved to: {csv_path}")
    if save_failures:
        print(f"Annotated failure images saved to: {fail_dir}")
    print("=" * 84)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Glove Defect Detection system -- batch evaluation")
    default_root = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw")
    ap.add_argument("dataset", nargs="?", default=os.path.abspath(default_root),
                    help="dataset root directory (default: dataset/raw)")
    ap.add_argument("--save-failures", action="store_true",
                    help="also save annotated failing images to dataset/failures/")
    args = ap.parse_args()
    return evaluate(args.dataset, args.save_failures)


if __name__ == "__main__":
    sys.exit(main())

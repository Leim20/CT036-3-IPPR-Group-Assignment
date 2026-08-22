# -*- coding: utf-8 -*-
"""
Batch evaluation -- run the whole dataset and count hits, misses and false
alarms for every defect type:

    .venv\\Scripts\\python src\\evaluate.py
    .venv\\Scripts\\python src\\evaluate.py --save-failures   (also saves the failing images)

Covers these assignment requirements:
  S4 "test the system to evaluate the accuracy of the proposed techniques"
  S5 "describe the results of testing using various test images"
  S5 "critical analysis for cases of images that fail"
  S6 Experimental Results & Critical analysis = 40% of the marks

===== How to lay out the dataset =====
(ground truth comes from the folder names, so no boxes have to be drawn by hand)

    dataset/raw/
      tearing/               <- defect name
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
``tearing+thin/cotton/example.jpg``.

The folder-name to defect-name table is LABEL_MAP below.
When you add a new detector, remember to add a line to LABEL_MAP.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline import process_image

# folder name (lower-case) -> the defect name the detector returns
# The left side is up to you; the right side must match the detector exactly.
LABEL_MAP = {
    "hole": "Tearing",       # legacy dataset-folder name
    "puncture": "Tearing",
    "tearing": "Tearing",
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
    "spotting": "Spotting",
    "spots": "Spotting",
    # every spelling is accepted, so a differently written folder name
    # cannot silently skip a whole defect class
    "plastic": "Plastic Contamination",
    "plasticcontamination": "Plastic Contamination",
    "plastic contamination": "Plastic Contamination",
    "plastic_contamination": "Plastic Contamination",
    # add new detectors here as team members finish them,
    # e.g. "wrinkle": "Wrinkle",
}

GOOD_DIR_NAMES = {"good", "ok", "normal", "pass"}   # folder names that mean "clean glove"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def parse_expected(folder_name):
    """Folder name -> the set of defect names expected. Empty set for good ones."""
    name = folder_name.strip().lower()
    if name in GOOD_DIR_NAMES:
        return set()
    labels = set()
    for part in name.split("+"):
        part = part.strip()
        if part in LABEL_MAP:
            labels.add(LABEL_MAP[part])
        else:
            labels.add(f"<unregistered:{part}>")   # reminder to extend LABEL_MAP
    return labels


def collect_images(root):
    """Scan dataset/raw and return [(image path, material, expected defects), ...]"""
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
        print(__doc__.split("===== How to lay out the dataset")[1])
        return 1

    # counters
    stat = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})   # per defect type
    mat_stat = defaultdict(lambda: {"hit": 0, "need": 0})     # recall per material
    rows = []                       # per-image records written to the CSV
    failures = []                   # failing images, for the failure analysis
    detector_errors = defaultdict(int)   # which detector crashed, and how often
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
            verdict = "segmentation failed (no glove found)"
            failures.append((path, verdict))
        else:
            defects = result["defects"]
            errs = result["errors"]
            detected = {n for n, _ in defects}
            verdict = "correct" if detected == expected else "mismatch"
            # A crashing detector is a code bug, not an accuracy problem, so
            # it has to be reported separately.
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

        # --- per material (did all of this image's defects get found?) ---
        if expected:
            mat_stat[material]["need"] += 1
            if expected <= detected:
                mat_stat[material]["hit"] += 1

        # --- false alarms on defect-free images ---
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
    print("=" * 78)
    print(f"Dataset: {root}    {len(items)} images")
    print("-" * 78)
    print(f"{'Defect type':<24}{'expected':>9}{'found':>7}{'missed':>8}"
          f"{'false':>7}{'recall':>9}{'precision':>11}{'F1':>7}")
    print("-" * 78)
    for label in sorted(stat):
        s = stat[label]
        tp, fn, fp = s["tp"], s["fn"], s["fp"]
        recall = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
        print(f"{label:<24}{tp+fn:>9}{tp:>7}{fn:>8}{fp:>7}"
              f"{recall:>9.1%}{prec:>11.1%}{f1:>7.2f}")

    print("-" * 78)
    if mat_stat:
        print("By material (were all of this image's defects found?):")
        for m in sorted(mat_stat):
            v = mat_stat[m]
            if v["need"]:
                print(f"  {m:<16}{v['hit']}/{v['need']}   {v['hit']/v['need']:.1%}")
            else:
                print(f"  {m:<16}(no images with defects)")
    if n_good:
        print(f"False alarms on good gloves : {n_good_fp}/{n_good} = {n_good_fp/n_good:.1%}")
    if n_noglove:
        print(f"Segmentation failed (no glove found) : {n_noglove} images")
    if detector_errors:
        print("-" * 78)
        print("! A detector crashed -- that is a code bug, fix it before "
              "trusting the numbers above:")
        for name, count in sorted(detector_errors.items(), key=lambda kv: -kv[1]):
            print(f"    {name}  failed {count} times")

    # ============= list of failures (used by the report) =============
    if failures:
        print("-" * 78)
        print(f"{len(failures)} failing images -- these are the critical analysis:")
        for path, why in failures[:15]:
            print(f"  {os.path.relpath(path, root)}  ->  {why}")
        if len(failures) > 15:
            print(f"  ... the remaining {len(failures)-15} are in the CSV")

    # ================= write the CSV =================
    csv_path = os.path.join(os.path.dirname(root), "evaluation_result.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "material", "expected", "detected", "verdict"])
        w.writerows(rows)
    print("-" * 78)
    print(f"Per-image results saved to : {csv_path}")
    if save_failures:
        print(f"Annotated failures saved to : {fail_dir}")
    print("=" * 78)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Glove defect detection -- batch evaluation")
    default_root = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw")
    ap.add_argument("dataset", nargs="?", default=os.path.abspath(default_root),
                    help="dataset root directory (default: dataset/raw)")
    ap.add_argument("--save-failures", action="store_true",
                    help="save failing images with their boxes into dataset/failures/")
    args = ap.parse_args()
    return evaluate(args.dataset, args.save_failures)


if __name__ == "__main__":
    sys.exit(main())

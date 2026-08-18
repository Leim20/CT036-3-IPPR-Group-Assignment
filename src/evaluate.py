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
      latex/                 <- material name (at least 3)
        hole/       *.jpg    <- folder name = the defect(s) these images should contain
        open_tear/  *.jpg
        stain/      *.jpg
        hole+stain/ *.jpg    <- a "+" joins multiple defects present in one image
        good/       *.jpg    <- clean gloves, used to measure the false-positive rate
      rubber/ ...
      leather/ ...

The folder-name -> defect-label lookup table is LABEL_MAP below.
Whenever a team member adds a new detector, add a matching row to it.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from preprocessing import preprocess
from segmentation import segment_glove, glove_found, get_background_color
from defect_detection import run_all_detectors, draw_results

# folder name (lowercase) -> the defect label a detector actually returns
# The left side can be whatever you like; the right side MUST exactly
# match the string returned by the detector.
LABEL_MAP = {
    "hole": "Tear / Hole",
    "tear": "Tear / Hole",
    "puncture": "Tear / Hole",
    "open_tear": "Open Tear",
    "fingertip_tear": "Open Tear",
    "stain": "Stain",
    "dirty": "Stain",
    # add new detectors here as team members finish them, e.g. "wrinkle": "Wrinkle",
}

GOOD_DIR_NAMES = {"good", "ok", "normal", "pass"}   # folder names that mean "clean glove"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def imread_unicode(path):
    """Read an image, tolerant of non-ASCII paths (cv2.imread returns None
    for those)."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


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
    for material in sorted(os.listdir(root)):
        mat_dir = os.path.join(root, material)
        if not os.path.isdir(mat_dir):
            continue
        for defect_dir in sorted(os.listdir(mat_dir)):
            d = os.path.join(mat_dir, defect_dir)
            if not os.path.isdir(d):
                continue
            expected = parse_expected(defect_dir)
            for fn in sorted(os.listdir(d)):
                if os.path.splitext(fn)[1].lower() in IMG_EXT:
                    items.append((os.path.join(d, fn), material, expected))
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
        img = imread_unicode(path)
        if img is None:
            rows.append([path, material, "|".join(sorted(expected)), "", "failed to read image"])
            failures.append((path, "failed to read image"))
            continue

        img_norm, img_plain = preprocess(img)
        mask_filled, mask_raw = segment_glove(img_norm)
        ok, ratio = glove_found(mask_filled)

        if not ok:
            n_noglove += 1
            detected = set()
            defects = []
            verdict = "segmentation failed (no glove detected)"
            failures.append((path, verdict))
        else:
            bg_color = get_background_color(img_norm)
            defects, errs = run_all_detectors(img_norm, mask_filled, mask_raw, bg_color)
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
            if save_failures and ok:
                out = os.path.join(fail_dir, material + "_" + os.path.basename(path))
                cv2.imwrite(out, draw_results(img_plain, defects))

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

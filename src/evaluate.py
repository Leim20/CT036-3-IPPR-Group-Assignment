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
      latex/                 <- material name (at least 3 of them)
        hole/       *.jpg    <- folder name = the defects these images contain
        open_tear/  *.jpg
        stain/      *.jpg
        hole+stain/ *.jpg    <- join with + when one image has several defects
        good/       *.jpg    <- defect-free, used to measure the false alarm rate
      rubber/ ...
      leather/ ...

The folder-name to defect-name table is LABEL_MAP below.
When you add a new detector, remember to add a line to LABEL_MAP.
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

# folder name (lower-case) -> the defect name the detector returns
# The left side is up to you; the right side must match the detector exactly.
LABEL_MAP = {
    "hole": "Tear / Hole",
    "tear": "Tear / Hole",
    "puncture": "Tear / Hole",
    "open_tear": "Open Tear",
    "fingertip_tear": "Open Tear",
    "stain": "Stain",
    "dirty": "Stain",
    "spotting": "Spotting",
    "spots": "Spotting",
    # every spelling is accepted, so a differently written folder name
    # cannot silently skip a whole defect class
    "plastic": "Plastic Contamination",
    "plasticcontamination": "Plastic Contamination",
    "plastic contamination": "Plastic Contamination",
    "plastic_contamination": "Plastic Contamination",
    # add yours here when you write a new detector:  "wrinkle": "Wrinkle",
}

GOOD_DIR_NAMES = {"good", "ok", "normal", "pass"}   # names of defect-free folders
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def imread_unicode(path):
    """Read an image whose path may contain non-ASCII characters
    (cv2.imread returns None for those)."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


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
        img = imread_unicode(path)
        if img is None:
            rows.append([path, material, "|".join(sorted(expected)), "", "unreadable"])
            failures.append((path, "unreadable"))
            continue

        img_norm, img_plain = preprocess(img)
        mask_filled, mask_raw = segment_glove(img_norm)
        ok, ratio = glove_found(mask_filled)

        if not ok:
            n_noglove += 1
            detected = set()
            defects = []
            verdict = "segmentation failed (no glove found)"
            failures.append((path, verdict))
        else:
            bg_color = get_background_color(img_norm)
            defects, errs = run_all_detectors(img_norm, mask_filled, mask_raw, bg_color)
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
            if save_failures and ok:
                out = os.path.join(fail_dir, material + "_" + os.path.basename(path))
                cv2.imwrite(out, draw_results(img_plain, defects))

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

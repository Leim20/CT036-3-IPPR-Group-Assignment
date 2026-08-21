# -*- coding: utf-8 -*-
"""
批量评估脚本 —— 跑完整个数据集,统计每种缺陷的检出/漏检/误报:

    .venv\\Scripts\\python src\\evaluate.py
    .venv\\Scripts\\python src\\evaluate.py --save-failures   (顺便把失败的图存下来)

对应作业要求:
  §4 "test the system to evaluate the accuracy of the proposed techniques"
  §5 "describe the results of testing using various test images"
  §5 "critical analysis for cases of images that fail"
  §6 Experimental Results & Critical analysis = 40% 分数

===== 数据集怎么放(标准答案从文件夹名字来,不需要手工标注框)=====

    dataset/raw/
      latex/                 <- 材质名(至少 3 种)
        hole/       *.jpg    <- 文件夹名 = 这些图里应该有的缺陷
        open_tear/  *.jpg
        stain/      *.jpg
        hole+stain/ *.jpg    <- 用 + 连接表示一张图里有多种缺陷
        good/       *.jpg    <- 合格品,用来量误报率
      rubber/ ...
      leather/ ...

文件夹名 → 缺陷英文名的对照表在下面的 LABEL_MAP。
组员加了新检测器之后,记得往 LABEL_MAP 里补一行。
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

# 文件夹名(小写) → 检测器输出的缺陷英文名
# 左边随便你们起,右边必须和检测器 return 的字符串完全一致
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
    # 组员加新检测器后在这里补:  "wrinkle": "Wrinkle",
}

GOOD_DIR_NAMES = {"good", "ok", "normal", "pass"}   # 合格品文件夹的名字
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def imread_unicode(path):
    """支持中文路径的读图(cv2.imread 遇到中文路径会返回 None)。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def parse_expected(folder_name):
    """文件夹名 → 期望出现的缺陷英文名集合。合格品返回空集合。"""
    name = folder_name.strip().lower()
    if name in GOOD_DIR_NAMES:
        return set()
    labels = set()
    for part in name.split("+"):
        part = part.strip()
        if part in LABEL_MAP:
            labels.add(LABEL_MAP[part])
        else:
            labels.add(f"<未登记:{part}>")   # 提醒你们去补 LABEL_MAP
    return labels


def collect_images(root):
    """扫描 dataset/raw,返回 [(图片路径, 材质, 期望缺陷集合), ...]"""
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
        print(f"没有在 {root} 找到任何图片。")
        print(__doc__.split("===== 数据集怎么放")[1])
        return 1

    # 统计容器
    stat = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})   # 每种缺陷
    mat_stat = defaultdict(lambda: {"hit": 0, "need": 0})     # 每种材质的召回
    rows = []                       # 写进 CSV 的逐图记录
    failures = []                   # 失败的图,给报告的失败案例分析用
    detector_errors = defaultdict(int)   # 哪个检测器出错了、出错几次
    n_good = n_good_fp = n_noglove = 0

    fail_dir = os.path.join(os.path.dirname(root), "failures")
    if save_failures:
        os.makedirs(fail_dir, exist_ok=True)

    for path, material, expected in items:
        img = imread_unicode(path)
        if img is None:
            rows.append([path, material, "|".join(sorted(expected)), "", "读图失败"])
            failures.append((path, "读图失败"))
            continue

        img_norm, img_plain = preprocess(img)
        mask_filled, mask_raw = segment_glove(img_norm)
        ok, ratio = glove_found(mask_filled)

        if not ok:
            n_noglove += 1
            detected = set()
            defects = []
            verdict = "分割失败(未检测到手套)"
            failures.append((path, verdict))
        else:
            bg_color = get_background_color(img_norm)
            defects, errs = run_all_detectors(img_norm, mask_filled, mask_raw, bg_color)
            detected = {n for n, _ in defects}
            verdict = "正确" if detected == expected else "不符"
            # 检测器崩溃是代码 bug,不是算法准确率问题,必须单独提醒
            for err in errs:
                detector_errors[err.split(":")[0]] += 1

        # --- 逐类累计 ---
        for label in expected:
            if label in detected:
                stat[label]["tp"] += 1
            else:
                stat[label]["fn"] += 1
        for label in detected - expected:
            stat[label]["fp"] += 1

        # --- 材质维度(只看该图的缺陷有没有全部检出)---
        if expected:
            mat_stat[material]["need"] += 1
            if expected <= detected:
                mat_stat[material]["hit"] += 1

        # --- 合格品误报 ---
        if not expected:
            n_good += 1
            if detected:
                n_good_fp += 1

        if verdict != "正确":
            if verdict == "不符":
                failures.append((path, f"期望{sorted(expected)} 实测{sorted(detected)}"))
            if save_failures and ok:
                out = os.path.join(fail_dir, material + "_" + os.path.basename(path))
                cv2.imwrite(out, draw_results(img_plain, defects))

        rows.append([path, material, "|".join(sorted(expected)),
                     "|".join(sorted(detected)), verdict])

    # ================= 打印结果 =================
    print("=" * 78)
    print(f"数据集: {root}    共 {len(items)} 张图")
    print("-" * 78)
    print(f"{'缺陷类型':<18}{'应检出':>8}{'检出':>7}{'漏检':>7}{'误报':>7}"
          f"{'召回率':>9}{'精确率':>9}{'F1':>8}")
    print("-" * 78)
    for label in sorted(stat):
        s = stat[label]
        tp, fn, fp = s["tp"], s["fn"], s["fp"]
        recall = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
        print(f"{label:<18}{tp+fn:>8}{tp:>7}{fn:>7}{fp:>7}"
              f"{recall:>9.1%}{prec:>9.1%}{f1:>8.2f}")

    print("-" * 78)
    if mat_stat:
        print("按材质(该图的缺陷是否全部检出):")
        for m in sorted(mat_stat):
            v = mat_stat[m]
            if v["need"]:
                print(f"  {m:<16}{v['hit']}/{v['need']}   {v['hit']/v['need']:.1%}")
            else:
                print(f"  {m:<16}(没有带缺陷的图)")
    if n_good:
        print(f"合格品误报率 : {n_good_fp}/{n_good} = {n_good_fp/n_good:.1%}")
    if n_noglove:
        print(f"分割失败(未检测到手套) : {n_noglove} 张")
    if detector_errors:
        print("-" * 78)
        print("⚠ 有检测器运行出错(这是代码 bug,先修好再看上面的准确率):")
        for name, count in sorted(detector_errors.items(), key=lambda kv: -kv[1]):
            print(f"    {name}  出错 {count} 次")

    # ================= 失败案例清单(报告要用)=================
    if failures:
        print("-" * 78)
        print(f"失败案例 {len(failures)} 张(报告的 critical analysis 就写这些):")
        for path, why in failures[:15]:
            print(f"  {os.path.relpath(path, root)}  ->  {why}")
        if len(failures) > 15:
            print(f"  ... 其余 {len(failures)-15} 张见 CSV")

    # ================= 存 CSV =================
    csv_path = os.path.join(os.path.dirname(root), "evaluation_result.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["图片", "材质", "期望缺陷", "检出缺陷", "判定"])
        w.writerows(rows)
    print("-" * 78)
    print(f"逐图结果已保存 : {csv_path}")
    if save_failures:
        print(f"失败图已标注保存 : {fail_dir}")
    print("=" * 78)
    return 0


def main():
    ap = argparse.ArgumentParser(description="手套缺陷检测系统 —— 批量评估")
    default_root = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw")
    ap.add_argument("dataset", nargs="?", default=os.path.abspath(default_root),
                    help="数据集根目录(默认 dataset/raw)")
    ap.add_argument("--save-failures", action="store_true",
                    help="把检测失败的图连同标注框存到 dataset/failures/")
    args = ap.parse_args()
    return evaluate(args.dataset, args.save_failures)


if __name__ == "__main__":
    sys.exit(main())

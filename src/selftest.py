# -*- coding: utf-8 -*-
"""
回归测试脚本 —— 每次改完算法都跑一次,确认没有改坏:
    .venv\\Scripts\\python src\\selftest.py

它会自动生成一批"模拟手套图"(不同光照、不同背景、不同污渍颜色、
手套位置偏移等),跑完整条检测流程,并把每个场景的检测结果和
【期望结果】对比,最后打印通过率。

为什么要做这个:
  作业要求"系统不能对环境敏感"。光靠肉眼看一两张图看不出问题,
  用一批可控的合成场景做回归测试,才能量化地说明系统的鲁棒性,
  也才有数据写进报告的"实验结果与批判分析"部分。

⚠ 合成图只能验证"算法逻辑对不对",不能代替真实照片。
  真实手套的纹理、阴影、反光要用 dataset/ 里的自建数据集来验证。
"""
import os
import sys

import cv2
import numpy as np

# Windows 终端默认编码可能不支持中文,强制用 UTF-8 输出
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from preprocessing import preprocess
from segmentation import segment_glove, glove_found, get_background_color
from defect_detection import run_all_detectors, draw_results

GLOVE = (190, 120, 40)   # 手套颜色(BGR,蓝色)
BG = (60, 60, 200)       # 背景颜色(BGR,红色)


def make_glove_image(stain_color=(100, 60, 20), hole=True, stain=True,
                     bright=1.0, offset=(0, 0), bg=BG, glove=GLOVE,
                     noise=0, side_light=False,
                     open_tear=False, fingertip_tear=False, two_tone=False,
                     spots=0, spot_color=(40, 220, 240),
                     card=False, glove_scale=1.0):
    """画一张模拟手套图,各参数用来模拟不同的拍摄环境和缺陷。"""
    img = np.full((600, 800, 3), bg, dtype=np.uint8)
    ox, oy = offset

    # 手掌 + 手腕。glove_scale 用来模拟"尺寸偏大"的手套:以手掌中心为基准
    # 等比放大,这样长度变了但形状不变 —— 正好是 Oversize 该有的样子。
    def sx(x):
        return int(400 + ox + (x - 400) * glove_scale)

    def sy(y):
        return int(330 + oy + (y - 330) * glove_scale)

    def sr(r):
        return max(int(r * glove_scale), 1)

    cv2.ellipse(img, (sx(400), sy(330)), (sr(130), sr(110)), 0, 0, 360, glove, -1)
    cv2.rectangle(img, (sx(330), sy(400)), (sx(470), sy(560)), glove, -1)
    # 五根手指(指缝留够宽度,避免形态学闭运算把指缝也当成手套)
    for i, fx in enumerate([290, 345, 400, 455, 505]):
        cv2.ellipse(img, (sx(fx), sy(220)), (sr(18), sr(90 - abs(i - 2) * 12)),
                    0, 0, 360, glove, -1)

    if two_tone:  # 模拟涂层手套的深色布料袖口;属于正常材质,不是污渍
        cuff_color = tuple(max(int(channel * 0.55), 20) for channel in glove)
        cv2.rectangle(img, (330 + ox, 475 + oy), (470 + ox, 560 + oy),
                      cuff_color, -1)

    if hole:   # 缺陷 1:封闭破洞(露出背景色)
        cv2.circle(img, (430 + ox, 330 + oy), 22, bg, -1)
    if open_tear:   # 缺陷 2a:手掌侧面开放性撕裂(从左边缘往里裂)
        # 注意画在手掌【左】侧:右侧是破洞的位置,两个缺陷画重叠的话会连成
        # 一片,破洞就不再是"封闭"的了,测试期望值会不符合物理常识
        cv2.fillPoly(img, [np.array([[270 + ox, 330 + oy],
                                     [370 + ox, 315 + oy],
                                     [270 + ox, 355 + oy]])], bg)
    if fingertip_tear:  # 缺陷 2b:指尖撕裂(从中指指尖往下割一道窄缝)
        cv2.fillPoly(img, [np.array([[393 + ox, 126 + oy],
                                     [400 + ox, 215 + oy],
                                     [407 + ox, 126 + oy]])], bg)
    if stain:  # 缺陷 3:污渍(一块变色)
        cv2.ellipse(img, (400 + ox, 480 + oy), (18, 12), 30, 0, 360,
                    stain_color, -1)

    if spots:  # 缺陷 4:散布的小色点(Spotting)
        # 固定位置,保证每次生成的图一样,回归结果可复现
        placements = [(300, 250), (350, 300), (400, 260), (450, 310), (380, 350),
                      (320, 380), (430, 380), (290, 300), (460, 260), (410, 200)]
        for (sx, sy) in placements[:spots]:
            cv2.circle(img, (sx + ox, sy + oy), 11, spot_color, -1)

    if card:  # 标准尺寸参照卡(ID-1 85.6x54mm),放在画面左下角空白处
        # 画成 107x67 像素 -> 每毫米 1.25 像素,长宽比 1.597 接近标准 1.585
        cv2.rectangle(img, (40, 500), (147, 567), (230, 230, 230), -1)

    if side_light:  # 模拟侧面打光:左边暗、右边亮
        gradient = np.linspace(0.45, 1.35, img.shape[1])[None, :, None]
        img = np.clip(img.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
    if bright != 1.0:  # 模拟整体偏暗/偏亮
        img = np.clip(img.astype(np.float32) * bright, 0, 255).astype(np.uint8)
    if noise:  # 模拟传感器噪声
        img = np.clip(img.astype(np.int16) +
                      np.random.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


NO_GLOVE = ("无手套", "无手套", "无手套", "无手套", "无手套")


def analyse(img):
    """跑完整条流水线,返回 (封闭破洞数, 开放撕裂数, 污渍数)。
    没找到手套则返回 NO_GLOVE。"""
    img_norm, img_plain = preprocess(img)
    mask_filled, mask_raw = segment_glove(img_norm)
    ok, ratio = glove_found(mask_filled)
    if not ok:
        return NO_GLOVE
    bg_color = get_background_color(img_norm)
    defects, _ = run_all_detectors(img_norm, mask_filled, mask_raw, bg_color)
    names = [n for n, _ in defects]
    return (names.count("Tear / Hole"),
            names.count("Open Tear"),
            names.count("Stain"),
            names.count("Spotting"),
            names.count("Oversize"))


# 每个场景:(名称, 图片, 期望的 (破洞数, 撕裂数, 污渍数))
def build_cases():
    return [
        ("基准:1破洞+1污渍",      make_glove_image(),                                 (1, 0, 1, 0, 0)),
        ("异色污渍(白色粉迹)",     make_glove_image(stain_color=(240, 240, 240)),      (1, 0, 1, 0, 0)),
        ("合格品(应零误报)",       make_glove_image(hole=False, stain=False),          (0, 0, 0, 0, 0)),
        ("合格品+噪声sigma=8",    make_glove_image(hole=False, stain=False, noise=8), (0, 0, 0, 0, 0)),
        ("双色材质合格品",          make_glove_image(hole=False, stain=False,
                                                   two_tone=True),                   (0, 0, 0, 0, 0)),
        ("弱光:整体变暗60%",      make_glove_image(bright=0.6),                       (1, 0, 1, 0, 0)),
        ("强光:整体变亮140%",     make_glove_image(bright=1.4),                       (1, 0, 1, 0, 0)),
        ("侧面打光(左暗右亮)",     make_glove_image(side_light=True),                  (1, 0, 1, 0, 0)),
        ("手套偏离画面中心",       make_glove_image(offset=(150, -60)),                (1, 0, 1, 0, 0)),
        ("破洞正好落在正中心",     make_glove_image(offset=(-30, 0)),                  (1, 0, 1, 0, 0)),
        ("背景色接近手套色",       make_glove_image(bg=(200, 150, 90)),                (1, 0, 1, 0, 0)),
        ("灰手套+灰白背景",        make_glove_image(glove=(120, 120, 120),
                                                   bg=(190, 190, 190)),               (1, 0, 1, 0, 0)),
        # --- 开放性撕裂:关键在于不能把 4 条正常指缝误报成撕裂 ---
        ("手掌侧面开放撕裂",       make_glove_image(hole=False, stain=False,
                                                   open_tear=True),                   (0, 1, 0, 0, 0)),
        ("指尖撕裂",              make_glove_image(hole=False, stain=False,
                                                   fingertip_tear=True),              (0, 1, 0, 0, 0)),
        ("开放撕裂+封闭破洞",      make_glove_image(stain=False, open_tear=True),      (1, 1, 0, 0, 0)),
        ("三种缺陷同时出现",       make_glove_image(open_tear=True),                   (1, 1, 1, 0, 0)),
        ("撕裂+弱光60%",         make_glove_image(hole=False, stain=False,
                                                   open_tear=True, bright=0.6),       (0, 1, 0, 0, 0)),
        ("撕裂+手套偏心",         make_glove_image(hole=False, stain=False,
                                                   open_tear=True,
                                                   offset=(150, -60)),                (0, 1, 0, 0, 0)),
        ("纯背景图(根本没手套)",   np.full((600, 800, 3), BG, dtype=np.uint8), NO_GLOVE),
        # --- Spotting:主判据是"点够不够多",不是面积 ---
        ("斑点:8个黄点",          make_glove_image(hole=False, stain=False,
                                                   spots=8),                     (0, 0, 0, 8, 0)),
        ("斑点:10个黄点",         make_glove_image(hole=False, stain=False,
                                                   spots=10),                    (0, 0, 0, 10, 0)),
        # 点数不够(<5)时 Spotting 必须放弃,验证它跟 Stain 不是同一招换名字。
        # 这里 3 个点每个约 380px,也低于 Stain 的面积门槛,所以最终什么都不报。
        ("只有3个点:不算斑点",     make_glove_image(hole=False, stain=False,
                                                   spots=3),                     (0, 0, 0, 0, 0)),
        ("斑点+破洞共存",         make_glove_image(stain=False, spots=8),        (1, 0, 0, 8, 0)),
    ] + build_material_cases() + build_oversize_cases()


# 不同材质的手套颜色。之前的光照场景只用了亮蓝色手套,所以一直是满分,
# 却没发现"侧光 + 深色手套"会让分割失效 —— 是批量评估脚本捅出来的。
# 现在把材质和光照做成矩阵,通过的进回归门禁,失败的进下面的已知局限清单。
MATERIALS = [("latex亮蓝", (190, 120, 40)), ("rubber深灰", (80, 80, 80)),
             ("leather深蓝", (60, 90, 150)), ("白色乳胶", (235, 235, 235))]
LIGHTINGS = [("均匀", {}), ("暗60%", dict(bright=0.6)),
             ("亮140%", dict(bright=1.4)), ("噪声8", dict(noise=8))]

# 已知失败的组合(材质名, 光照名) —— 见 KNOWN_ISSUES 说明
KNOWN_FAIL = {("rubber深灰", "暗60%")}


def build_material_cases():
    """材质 × 光照 矩阵,每张图都是 1 破洞 + 1 污渍。"""
    cases = []
    for mname, color in MATERIALS:
        for lname, kw in LIGHTINGS:
            if (mname, lname) in KNOWN_FAIL:
                continue
            cases.append((f"{mname}/{lname}",
                          make_glove_image(glove=color, **kw), (1, 0, 1, 0, 0)))
    return cases


def build_oversize_cases():
    """Oversize 场景。基准长度用同一套合成图标定,保证测试可复现。

    没有参照卡、或者没标定基准时,detect_oversize 必须什么都不报 ——
    这两个用例把"没有依据就不猜"这条钉死。
    """
    import defect_detection as dd
    from preprocessing import preprocess
    from segmentation import segment_glove, get_background_color

    normal = make_glove_image(hole=False, stain=False, card=True)
    img_norm, _ = preprocess(normal)
    mask_filled, _ = segment_glove(img_norm)
    baseline = dd.measure_glove_length_mm(
        img_norm, mask_filled, get_background_color(img_norm),
    )
    dd.OVERSIZE_BASELINE_LENGTH_MM = baseline

    return [
        ("正常尺寸+参照卡",        make_glove_image(hole=False, stain=False,
                                                   card=True),              (0, 0, 0, 0, 0)),
        ("偏大25%+参照卡",        make_glove_image(hole=False, stain=False,
                                                   card=True,
                                                   glove_scale=1.25),       (0, 0, 0, 0, 1)),
        # 没有参照卡就无法标定尺度,必须放弃判断而不是瞎猜
        ("偏大25%但没参照卡",      make_glove_image(hole=False, stain=False,
                                                   glove_scale=1.25),       (0, 0, 0, 0, 0)),
    ]


def build_known_issues():
    """已知还没解决的问题,单独跑、单独打印,不计入通过率。

    放在这里而不是删掉,是为了让问题一直摆在明面上:回归测试满分
    但系统有已知缺陷,比"测试没覆盖到"要诚实得多。
    """
    cases = [
        # 小块黑色污渍(约 700px)。深色规则靠"比材料暗多少"识别,但手指边缘和
        # 指缝的阴影同样暗,实测这类误报是 660~1449px —— 跟这个尺寸的小黑点
        # 完全重叠,面积/紧致度/内切半径三个指标都分不开。
        # 权衡:面积门槛定在 2000px,保证真实照片零误报(黑漆真污渍从 2734px 起),
        # 代价就是这两个场景漏检。
        ("小块黑泥点(700px)", make_glove_image(stain_color=(20, 20, 20)), (1, 0, 1, 0, 0)),
        ("小块黑泥点、无破洞", make_glove_image(hole=False,
                                             stain_color=(20, 20, 20)), (0, 0, 1, 0, 0)),
    ]
    for mname, color in MATERIALS:
        cases.append((f"侧面打光 + {mname}",
                      make_glove_image(glove=color, side_light=True), (1, 0, 1, 0, 0)))
    for mname, lname in sorted(KNOWN_FAIL):
        color = dict(MATERIALS)[mname]
        kw = dict(LIGHTINGS)[lname]
        cases.append((f"{lname} + {mname}",
                      make_glove_image(glove=color, **kw), (1, 0, 1, 0, 0)))
    return cases


def main():
    np.random.seed(0)  # 固定随机数,保证每次跑结果可复现
    cases = build_cases()

    print("=" * 68)
    print(f"OpenCV 版本 : {cv2.__version__}")
    print("-" * 68)
    print(f"{'场景':<22}{'期望':>14}{'实测':>14}{'结果':>8}")
    print("-" * 68)

    passed = 0
    for name, img, expect in cases:
        got = analyse(img)
        ok = got == expect
        passed += ok
        print(f"{name:<22}{str(expect):>16}{str(got):>16}{'  通过' if ok else '  失败'}")

    print("-" * 68)
    print(f"通过率 : {passed}/{len(cases)}")

    # ---- 已知局限:单独跑、单独打印,不计入上面的通过率 ----
    known = build_known_issues()
    print("\n" + "=" * 68)
    print("已知局限(还没解决,不计入通过率 —— 报告的 critical analysis 写这些)")
    print("-" * 68)
    for name, img, expect in known:
        got = analyse(img)
        mark = "此组合正常" if got == expect else "仍失败"
        print(f"{name:<24}{str(expect):>16}{str(got):>16}  {mark}")
    print("-" * 68)
    print("小块黑泥点:见 build_known_issues() 里的说明 —— 小的深色污渍和")
    print("      指缝阴影在尺寸/形状上无法区分,为保证真实照片零误报而牺牲。")
    print("暗60%+rubber深灰:分割在低对比度下把部分手套判成背景。")
    print("=" * 68)

    # 把基准场景的标注结果存成图片,方便肉眼确认框画得对不对
    img = make_glove_image()
    img_norm, img_plain = preprocess(img)
    mask_filled, mask_raw = segment_glove(img_norm)
    bg_color = get_background_color(img_norm)
    defects, _ = run_all_detectors(img_norm, mask_filled, mask_raw, bg_color)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, "selftest_result.jpg"))
    cv2.imwrite(out_path, draw_results(img_plain, defects))
    print(f"标注结果已保存 : {out_path}")
    print("=" * 68)

    # 有场景失败时返回非 0,方便以后接自动化检查
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())

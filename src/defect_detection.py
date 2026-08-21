# -*- coding: utf-8 -*-
"""
缺陷检测模块:每种缺陷写成一个函数,登记进 DETECTORS 列表,GUI 自动调用。

写检测器的规则,函数签名统一是:

    def detect_xxx(img, mask_filled, mask_raw, bg_color):
        ...
        return [("Defect Name", (x, y, w, h)), ...]   # 没检测到就返回 []

四个参数所有检测器统一接收,用不到的可以直接忽略(比如撕裂检测只用得到
mask_filled)。缺陷英文名会画在图上、也会列在 GUI 的文字框里。
"""
from dataclasses import dataclass

import cv2
import numpy as np

# --- 可调参数(集中放这里,方便做参数敏感度实验、写进报告) ---
BG_MATCH_DIST = 30.0       # 破洞判据:候选块与背景色的 Lab 距离小于它才算破洞
STAIN_MASK_ERODE_KSIZE = 7     # 排除手套/背景混色的外轮廓
STAIN_NEUTRAL_S_MAX = 45       # HSV:低于它视为白/灰/黑等中性色材料
STAIN_LIGHT_V_MIN = 90         # 中性且够亮,才走白色/浅灰手套分支
STAIN_NEUTRAL_RATIO = 0.20     # 浅中性色像素占比达到它,判为浅色手套
STAIN_NEUTRAL_BASE_CLOSE_KSIZE = 101  # 跨过大片污渍,重建完整针织手套表面
STAIN_NEUTRAL_REGION_ERODE_KSIZE = 15 # 排除重建轮廓最外侧的背景混色
STAIN_NEUTRAL_CHROMA_DIST = 8.0      # Lab a/b 偏离正常材料的门槛
STAIN_NEUTRAL_BG_CHROMA_DIST = 20.0   # 排除与背景色度相同的网孔/指缝
STAIN_NEUTRAL_DENSITY_KSIZE = 17      # 连续色块密度窗口
STAIN_NEUTRAL_DENSITY_MIN = 0.12      # 稀疏针织纹理不会成为污渍
STAIN_NEUTRAL_CLOSE_KSIZE = 11        # 合并同一针织污渍内的小缝隙
STAIN_NEUTRAL_MIN_RADIUS = 6.0        # 排除细长手指边缘/指缝阴影(像素,800px 标准宽)
STAIN_NEUTRAL_MIN_COMPACTNESS = 0.20  # 4πA/P²;越接近 1 越像实心色块
# 彩色手套(光滑丁腈)的形状门槛更严:油漆/泥巴在光滑表面是锐利的块状,
# 而棉布吸水后污渍边缘发散、形状更松散,所以两个分支不能共用一套阈值。
# 实测:丁腈袖口阴影误报 紧致度=0.20 / 内切半径=6.4,真污渍最低 0.27 / 11.5;
# 棉布真污渍紧致度可低到 0.20,用 0.25 会误杀。
STAIN_COLOR_MIN_COMPACTNESS = 0.25
STAIN_COLOR_MIN_RADIUS = 8.0
# 彩色分支的面积门槛也要单独设,不能跟中性分支共用。
# 实测:光滑丁腈上真污渍面积 2734~63396,而手指边缘阴影误报只有 660~1449,
# 取 2000 两边都有余量;但棉布(中性分支)真污渍最小只有 575,共用会误杀。
STAIN_COLOR_MIN_AREA = 2000
STAIN_COLOR_S_MIN = 45         # 彩色手套的有效色相像素最低饱和度
STAIN_COLOR_V_MIN = 35         # 排除太暗、色相不可靠的像素
STAIN_BASE_HUE_TOL = 15        # 主色相左右多少 OpenCV hue 单位算正常材料
STAIN_HUE_DIST = 20            # 偏离主色相多少才算异色污渍
STAIN_BASE_CLOSE_KSIZE = 41    # 连接手套主色区域,跨过纹理/污渍重建轮廓
STAIN_LOCAL_KSIZE = 41         # 彩色手套的局部颜色窗口,补检黑/白/同色深污渍
STAIN_LOCAL_DIST = 20.0        # 局部加权 Lab 距离阈值
STAIN_LUMA_WEIGHT = 0.5        # 亮度降权,但保留对黑白污渍的响应
STAIN_OPEN_KSIZE = 3           # 去掉单像素噪声
STAIN_CLOSE_KSIZE = 15         # 合并同一污渍内部的小裂缝
# 分割掩膜可信度:掩膜内"颜色就是背景色"的像素占比超过它,说明分割把背景
# 吃进来了,这时才退回"用材料主色重建手套区"的保守方案。
# 实测(24 张真实照片):黄垫照片污染率 0.0~3.5%,石纹地砖照片 34.8~60.2%,
# 中间没有重叠,所以 8% 这个门槛很安全。
STAIN_SEG_POLLUTION_MAX = 0.08
STAIN_SEG_CLEAN_CHROMA = 15.0  # 与背景色度差小于它,视为"就是背景色"
STAIN_SEG_ERODE_KSIZE = 21      # 直接用分割轮廓时,只做小幅腐蚀避开边缘混色
# 深色污渍(黑色油漆/墨渍)判据:比材料正常亮度暗多少 Lab L 才算。
# 为什么需要单独一条:彩色分支靠"色相偏离"找污渍,但极暗像素的色相本来
# 就不可靠 —— 实测黑漆在蓝手套上算出的 hue 是 110~113,跟蓝色材料的
# 104 只差几度,色相判据完全失效;而且黑漆 V=11~25 会被 STAIN_COLOR_V_MIN
# 直接滤掉。改看"比材料暗多少"就稳定得多(具体数值见下面 ABS 的说明)。
STAIN_DARK_L_DROP = 85.0
# 绝对亮度上限只作为宽松兜底,真正起区分作用的是上面的相对落差。
# 为什么不能靠绝对门槛:黑漆的绝对亮度随打光变化很大 —— 实测第一批
# 照片 L=9~15,第二批同样的黑漆 L=55~62(打光更亮/漆面反光)。用绝对
# 门槛 45 会让第二批中间那块最明显的大污渍完全漏检。
# 相对落差则稳定得多:两批分别是 128~142 和 87~94,而双色材质手套较暗
# 那一色只有 57,门槛取 85 两边都留有余量。
STAIN_DARK_L_ABS = 110.0
# --- Spotting(斑点)判据 ---
# 跟 Stain 的本质区别是"数量",不是"面积":Stain 是一两块大的,
# Spotting 是散布很多小点。所以主判据是"够小的点有几个",
# 只有数量达标才算 Spotting —— 这样两者不会变成同一招换个名字。
SPOTTING_MIN_COUNT = 5           # 至少这么多个点才判为 Spotting
SPOTTING_MIN_AREA = 50           # 单点面积下限,滤掉噪声(800px 标准宽)
SPOTTING_MAX_AREA_RATIO = 0.015  # 单点面积上限占手套面积比例;超过就是 Stain 不是点
SPOTTING_MIN_COMPACTNESS = 0.30  # 点应该是紧致的小圆块,排除细长的边缘/褶皱
# 点还必须"够鲜艳":Spotting 是溅上去的异色点,颜色偏离很强。
# 实测真斑点(黄水彩点在蓝手套上)色度偏离材料主色 120,而棉布上被纹理
# 切碎的浅色痕迹只有 11~15 —— 差 8 倍,门槛取 40 两边余量都很大。
SPOTTING_MIN_CHROMA_DEV = 40.0
# 点的颜色还必须【不是背景色】。手套边缘会透出一点背景,在彩色分支里
# 同样满足"色相偏离手套主色",实测这类误报色相 159~175(红背景),
# 而真绿点是 72~85 —— 用"离背景色度多远"就能干净剔除。
SPOTTING_MIN_BG_CHROMA_DIST = 30.0
# Spotting 的碎片合并核要比 Stain 的小:核太大会把两个挨得近的点粘成一个,
# 实测用 Stain 的 11 会漏掉相邻的点,用 7 刚好。Stain 那边则需要 11 才能
# 把针织纹理切碎的污渍合回去 —— 两者需求相反,不能共用。
SPOTTING_CLOSE_KSIZE = 7

# --- Oversize(尺寸异常)判据 ---
# 核心难题:照片里的大小 ≠ 真实大小 —— 同一只手套,相机靠近拍就大、
# 退远拍就小。所以必须在画面里放一个【已知尺寸的参照物】做标定,
# 否则任何基于像素的尺寸判断都没有意义。
# 用银行卡/学生证:ID-1 是国际标准 85.6 x 54 mm,而且是规整矩形,好检测。
CARD_LONG_MM = 85.6            # ID-1 标准卡长边(毫米)
CARD_SHORT_MM = 54.0           # ID-1 标准卡短边(毫米)
CARD_ASPECT_TOL = 0.18         # 长宽比允许偏离标准值多少(俯拍角度略斜时的容差)
CARD_MIN_AREA_RATIO = 0.005    # 卡片至少占画面这么大,滤掉小杂物
CARD_MAX_AREA_RATIO = 0.15     # 也不能太大,否则多半认错了
# 手套实际长度超过基准多少判为偏大。12% 大约相当于差一个尺码
# (常见手套 M->L 长度差约 10~15mm,占总长约 8~12%)。
OVERSIZE_LENGTH_TOL = 1.12
# 基准值:同款【正常尺寸】手套的实测长度(毫米)。必须先用 good/ 照片标定,
# 没标定就不报 Oversize —— 没有基准时任何"偏大"的判断都是无根据的。
# 标定方法:跑 calibrate_oversize(),把它打印的数字填到这里。
OVERSIZE_BASELINE_LENGTH_MM = None

MIN_AREA_HOLE = 60
MIN_AREA_STAIN = 500           # 800px 标准宽度下的最小污渍面积
                               # 指缝阴影误报都在 500~600px,真污渍最小 1700px,
                               # 门槛卡在中间;14 张标注照片里 9 张达到 100/100。

# BGR 颜色:结果图中每种缺陷使用固定颜色,同时作为 GUI 图例。
DEFECT_COLORS = {
    "Stain": (0, 140, 255),          # 橙色
    "Tear / Hole": (40, 40, 220),   # 红色
    "Open Tear": (190, 55, 150),    # 紫色
}
DEFAULT_DEFECT_COLOR = (35, 160, 70)  # 其他以后新增的检测器:绿色


@dataclass
class Detection:
    """一个缺陷的定位结果。

    ``mask`` 是与预处理图同尺寸的 uint8 二值图,用于像素级着色和 affected-area;
    ``evidence`` 是规则证据强度 0..100,不是机器学习概率。

    实现 ``__iter__`` 是为了兼容旧代码里的 ``for name, box in defects``。
    """

    name: str
    box: tuple
    mask: np.ndarray | None = None
    evidence: float = 0.0

    def __iter__(self):
        yield self.name
        yield self.box

# 开放性撕裂(裂到手套边界)的判据,数值来自实测,不是拍脑袋定的:
#     正常指缝: 开口宽/深度 = 0.55~0.74, 顶角 = 29~40 度
#     开放撕裂: 开口宽/深度 = 0.36,      顶角 = 20 度
#     手腕台阶: 开口宽/深度 = 5.53,      顶角 = 137 度
# 所以用"窄"和"尖"两个条件就能把撕裂和指缝分开。
CONTOUR_EPSILON = 2.0          # 轮廓简化容差,去掉锯齿产生的假凹口
MIN_TEAR_DEPTH_RATIO = 0.05    # 凹口深度 ÷ 手套外接框对角线,滤掉浅凹
MAX_TEAR_MOUTH_RATIO = 0.45    # 凹口开口宽度 ÷ 深度,撕裂是窄缝
MAX_TEAR_APEX_ANGLE = 24.0     # 凹口顶点夹角(度),撕裂尖、指缝钝

DEDUP_IOU = 0.5   # 两个检测器报的框重叠超过这个比例,只保留先登记的那个


# ============================================================
# 缺陷 1:封闭破洞
# ============================================================
def detect_holes(img, mask_filled, mask_raw, bg_color):
    """原理:破洞露出的是背景,所以它"在手套轮廓之内、颜色却是背景色"。
    候选块的平均颜色必须接近背景色才算数,否则任何颜色偏离的东西
    (污渍、阴影)都会被误报成破洞。
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    candidate = cv2.subtract(mask_filled, mask_raw)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA_HOLE:
            continue
        blob = np.zeros(candidate.shape, np.uint8)
        cv2.drawContours(blob, [c], -1, 255, cv2.FILLED)
        mean_color = lab[blob > 0].mean(axis=0)
        color_distance = float(np.linalg.norm(mean_color - bg_color))
        if color_distance < BG_MATCH_DIST:
            color_fit = 1.0 - color_distance / BG_MATCH_DIST
            size_fit = min(1.0, cv2.contourArea(c) / (MIN_AREA_HOLE * 4.0))
            evidence = 50.0 + 50.0 * (0.75 * color_fit + 0.25 * size_fit)
            results.append(Detection(
                "Tear / Hole", cv2.boundingRect(c), blob, round(evidence, 1),
            ))
    return results


# ============================================================
# 缺陷 2:开放性撕裂(裂到手套边缘)
# ============================================================
def detect_open_tears(img, mask_filled, mask_raw, bg_color):
    """原理:凸包缺陷(convexity defects)—— 轮廓和它凸包之间的凹陷。
    正常指缝也是又深又大的凹口,靠深度分不开,得看形状:
    撕裂又窄又尖(材料被割开),指缝又宽又钝(自然的 U 形空隙)。

    已知局限:这个区分本质是启发式的,真实手套手指弯曲、并拢、
    袖口卷边都会改变指缝形状,阈值必须用真实照片重新标定。
    """
    contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    cnt = cv2.approxPolyDP(max(contours, key=cv2.contourArea), CONTOUR_EPSILON, True)
    if len(cnt) < 4:
        return []

    hull = cv2.convexHull(cnt, returnPoints=False)
    hull[::-1].sort(axis=0)
    try:
        defects = cv2.convexityDefects(cnt, hull)
    except cv2.error:
        return []
    if defects is None:
        return []

    _, _, bw, bh = cv2.boundingRect(cnt)
    diag = float(np.hypot(bw, bh))   # 用手套自身尺寸归一化,换分辨率也适用

    results = []
    for s, e, f, depth_fp in defects.reshape(-1, 4):
        depth = depth_fp / 256.0
        if depth < MIN_TEAR_DEPTH_RATIO * diag:
            continue   # 太浅,是锯齿或手腕台阶

        p1, p2, apex = cnt[s][0], cnt[e][0], cnt[f][0]

        mouth = float(np.linalg.norm(p1 - p2))          # 条件①:窄
        if mouth > MAX_TEAR_MOUTH_RATIO * depth:
            continue

        v1, v2 = p1 - apex, p2 - apex                     # 条件②:尖
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        if angle > MAX_TEAR_APEX_ANGLE:
            continue

        points = np.array([p1, p2, apex])
        blob = np.zeros(mask_filled.shape, np.uint8)
        cv2.fillPoly(blob, [points], 255)
        depth_fit = min(1.0, depth / (2.0 * MIN_TEAR_DEPTH_RATIO * diag))
        mouth_fit = np.clip(1.0 - mouth / (MAX_TEAR_MOUTH_RATIO * depth), 0.0, 1.0)
        angle_fit = np.clip(1.0 - angle / MAX_TEAR_APEX_ANGLE, 0.0, 1.0)
        evidence = 50.0 + 50.0 * (
            0.30 * depth_fit + 0.35 * mouth_fit + 0.35 * angle_fit
        )
        results.append(Detection(
            "Open Tear", cv2.boundingRect(points), blob, round(float(evidence), 1),
        ))
    return results


# ============================================================
# 缺陷 5:尺寸异常(Oversize)—— 需要画面里有参照卡做标定
# ============================================================
def _find_reference_card(img, mask_filled, bg_color):
    """在背景里找那张标准尺寸卡,返回 (毫米/像素, 卡片轮廓)。找不到返回 None。

    判据:矩形(四边形)、长宽比接近 ID-1 的 85.6/54=1.585、面积适中,
    而且不能压在手套上(压着的话轮廓会被手套截断,量出来不准)。
    """
    h, w = img.shape[:2]
    frame_area = float(h * w)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[:, :, 1] - bg_color[1], lab[:, :, 2] - bg_color[2])
    scaled = cv2.normalize(chroma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, foreground = cv2.threshold(
        scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8),
    )
    # 手套本身不是卡,先排除掉
    foreground[mask_filled > 0] = 0

    contours, _ = cv2.findContours(
        foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    target_aspect = CARD_LONG_MM / CARD_SHORT_MM
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (CARD_MIN_AREA_RATIO * frame_area
                <= area <= CARD_MAX_AREA_RATIO * frame_area):
            continue
        (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
        long_px, short_px = max(rect_w, rect_h), min(rect_w, rect_h)
        if short_px <= 1:
            continue
        aspect = long_px / short_px
        if abs(aspect - target_aspect) > CARD_ASPECT_TOL * target_aspect:
            continue
        # 轮廓要够"填满"它的外接矩形,才像一张实心卡片
        if area / max(rect_w * rect_h, 1.0) < 0.80:
            continue
        mm_per_px = CARD_LONG_MM / long_px
        if best is None or area > best[2]:
            best = (mm_per_px, contour, area)
    return (best[0], best[1]) if best else None


def measure_glove_length_mm(img, mask_filled, bg_color):
    """量出手套的真实长度(毫米)。没找到参照卡时返回 None。

    长度取手套最小外接矩形的长边 —— 比外接框稳定,手套斜放也不受影响。
    """
    card = _find_reference_card(img, mask_filled, bg_color)
    if card is None:
        return None
    mm_per_px, _ = card
    contours, _ = cv2.findContours(
        mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    return max(rect_w, rect_h) * mm_per_px


def detect_oversize(img, mask_filled, mask_raw, bg_color):
    """手套尺寸超出正常范围。

    先用画面里的标准尺寸卡标定出"毫米/像素",把手套长度换算成真实毫米,
    再跟 good/ 照片标定出来的基准长度比。这样相机远近不影响结果。

    没有参照卡、或者还没标定基准,就什么都不报 —— 没有基准时判断"偏大"
    是没有依据的,宁可不报也不能瞎猜。
    """
    if OVERSIZE_BASELINE_LENGTH_MM is None:
        return []
    length_mm = measure_glove_length_mm(img, mask_filled, bg_color)
    if length_mm is None:
        return []
    ratio = length_mm / OVERSIZE_BASELINE_LENGTH_MM
    if ratio < OVERSIZE_LENGTH_TOL:
        return []

    contours, _ = cv2.findContours(
        mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    contour = max(contours, key=cv2.contourArea)
    blob = np.zeros(mask_filled.shape, np.uint8)
    cv2.drawContours(blob, [contour], -1, 255, cv2.FILLED)
    # 超出越多证据越强;超出一倍容差时给满分
    excess = (ratio - OVERSIZE_LENGTH_TOL) / max(OVERSIZE_LENGTH_TOL - 1.0, 1e-6)
    evidence = 50.0 + 50.0 * float(np.clip(excess, 0.0, 1.0))
    return [Detection(
        "Oversize", cv2.boundingRect(contour), blob, round(evidence, 1),
    )]


def calibrate_oversize(image_paths):
    """用 good/ 里的正常手套照片算出基准长度,打印出来供填进常量。

    这一步只做【一次】,是离线标定,不是检测时要做的事。标定完把数字填进
    OVERSIZE_BASELINE_LENGTH_MM,之后检测就只需要单张照片 —— 照片里的
    参照卡提供尺度,基准值来自这个常量,不需要把两张照片放一起比。

    用法见 README 的 Oversize 那一节。
    """
    from preprocessing import preprocess
    from segmentation import segment_glove, get_background_color

    lengths = []
    for path in image_paths:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            continue
        img_norm, _ = preprocess(image)
        mask_filled, _ = segment_glove(img_norm)
        length = measure_glove_length_mm(
            img_norm, mask_filled, get_background_color(img_norm),
        )
        status = f"{length:.1f} mm" if length else "没找到参照卡"
        print(f"  {path}: {status}")
        if length:
            lengths.append(length)
    if not lengths:
        print("没有一张照片能标定 —— 检查参照卡是否完整入镜、是否压在手套上。")
        return None
    baseline = float(np.median(lengths))
    print(f"基准长度中位数 = {baseline:.1f} mm  (共 {len(lengths)} 张)")
    print("把这个数字填进 OVERSIZE_BASELINE_LENGTH_MM")
    return baseline


# ============================================================
# 缺陷 4:斑点(Spotting)—— 散布的多个小色点
# ============================================================
def _find_spots(img, mask_filled, mask_raw, bg_color):
    """找出符合"斑点"定义的小色点,返回 [(轮廓, 面积, 紧致度), ...]。

    抽成共享函数是为了让 detect_spotting 和 detect_stains 用【同一套】判据:
    Stain 靠它判断"这批小点该不该让给 Spotting"。如果两边判据不一致,
    会出现真污渍被 Stain 让掉、又被 Spotting 拒收,最后谁都不报的漏洞。
    """
    h, w = img.shape[:2]
    glove_area = float((mask_filled > 0).sum())
    if glove_area <= 0:
        return []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    erode_ksize = _odd_kernel(STAIN_MASK_ERODE_KSIZE, h, w)
    inside = cv2.erode(mask_raw, np.ones((erode_ksize, erode_ksize), np.uint8)) > 0
    if not inside.any():
        return []

    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    material_lab = np.median(lab_img[inside], axis=0)

    colorful = inside & (sat >= STAIN_COLOR_S_MIN) & (val >= STAIN_COLOR_V_MIN)
    if colorful.sum() < inside.sum() * STAIN_NEUTRAL_RATIO:
        deviation = np.hypot(lab_img[:, :, 1] - material_lab[1],
                             lab_img[:, :, 2] - material_lab[2])
        candidate = inside & (deviation >= STAIN_NEUTRAL_CHROMA_DIST)
    else:
        hist = np.bincount(hue[colorful], minlength=180).astype(np.float32)
        smooth = np.convolve(
            np.r_[hist[-4:], hist, hist[:4]], np.ones(9), mode="valid",
        )
        dominant_hue = int(np.argmax(smooth) % 180)
        raw_delta = np.abs(hue.astype(np.int16) - dominant_hue)
        hue_delta = np.minimum(raw_delta, 180 - raw_delta)
        candidate = (
            inside & (hue_delta >= STAIN_HUE_DIST)
            & (sat >= STAIN_COLOR_S_MIN) & (val >= STAIN_COLOR_V_MIN)
        )

    blob_mask = cv2.morphologyEx(
        candidate.astype(np.uint8) * 255, cv2.MORPH_OPEN,
        np.ones((STAIN_OPEN_KSIZE, STAIN_OPEN_KSIZE), np.uint8),
    )
    # 必须先合并碎片再数点,否则针织纹理会把一块完整污渍切成好几小块,
    # 凑够数量后被误判成 Spotting。实测棉布照片上 4 块真污渍会碎成 7~8 块。
    close_ksize = _odd_kernel(SPOTTING_CLOSE_KSIZE, h, w)
    if close_ksize >= 3:
        blob_mask = cv2.morphologyEx(
            blob_mask, cv2.MORPH_CLOSE,
            np.ones((close_ksize, close_ksize), np.uint8),
        )
    contours, _ = cv2.findContours(
        blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    max_area = glove_area * SPOTTING_MAX_AREA_RATIO
    spots = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < SPOTTING_MIN_AREA or area > max_area:
            continue          # 太小是噪声,太大就该算 Stain
        perimeter = cv2.arcLength(contour, True)
        compactness = 4.0 * np.pi * area / max(perimeter * perimeter, 1.0)
        if compactness < SPOTTING_MIN_COMPACTNESS:
            continue
        blob = np.zeros(blob_mask.shape, np.uint8)
        cv2.drawContours(blob, [contour], -1, 255, cv2.FILLED)
        pixels = blob > 0
        deviation = float(np.median(np.hypot(
            lab_img[pixels, 1] - material_lab[1],
            lab_img[pixels, 2] - material_lab[2],
        )))
        if deviation < SPOTTING_MIN_CHROMA_DEV:
            continue      # 颜色太淡,多半是材质纹理而不是溅上去的点
        bg_distance = float(np.median(np.hypot(
            lab_img[pixels, 1] - bg_color[1],
            lab_img[pixels, 2] - bg_color[2],
        )))
        if bg_distance < SPOTTING_MIN_BG_CHROMA_DIST:
            continue      # 颜色就是背景色,多半是手套边缘透出来的背景
        spots.append((contour, area, compactness))
    return spots


def detect_spotting(img, mask_filled, mask_raw, bg_color):
    """散布在手套上的多个小色点(例如溅上的颜料点)。

    跟 Stain 的区别在判据本身,不只是阈值:
      Stain    看的是"有没有偏离材料主色的区域",一两块就算;
      Spotting 看的是"够小、够鲜艳的点有没有达到一定数量",少了就不算 ——
               单独一两个点应该由 Stain 报出来,不该重复算成两种缺陷。
    """
    spots = _find_spots(img, mask_filled, mask_raw, bg_color)
    # 主判据:点数不够就不是 Spotting,交给 Stain 处理
    if len(spots) < SPOTTING_MIN_COUNT:
        return []

    results = []
    for contour, area, compactness in spots:
        blob = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(blob, [contour], -1, 255, cv2.FILLED)
        count_fit = min(1.0, len(spots) / (SPOTTING_MIN_COUNT * 2.0))
        evidence = 50.0 + 50.0 * (0.6 * count_fit + 0.4 * compactness)
        results.append(Detection(
            "Spotting", cv2.boundingRect(contour), blob, round(evidence, 1),
        ))
    return results


# ============================================================
# 缺陷 3:污渍
# ============================================================
def _odd_kernel(preferred, h, w):
    """把形态学/中位滤波核限制在当前图片尺寸内,并保持奇数。"""
    size = min(preferred, h, w)
    if size % 2 == 0:
        size -= 1
    return size


def _largest_component(mask):
    """只保留二值图中面积最大的连通块。"""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return np.zeros_like(mask)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == index).astype(np.uint8) * 255


def _segmentation_is_clean(mask_filled, lab, bg_color):
    """判断分割掩膜有没有把背景吃进来。

    做法:数一数掩膜内部有多少像素"颜色根本就是背景色"。手套本身不该有
    大片背景色的区域,所以这个比例高就说明分割不可信。
    """
    mask = mask_filled > 0
    if not mask.any():
        return False
    bg_chroma = np.hypot(lab[:, :, 1] - bg_color[1], lab[:, :, 2] - bg_color[2])
    polluted = mask & (bg_chroma < STAIN_SEG_CLEAN_CHROMA)
    return (polluted.sum() / mask.sum()) <= STAIN_SEG_POLLUTION_MAX


def _region_from_base(base, close_ksize):
    """用正常材料主色像素重建手套区域,避免把地毯/手臂当成污渍。"""
    base = cv2.morphologyEx(
        base, cv2.MORPH_CLOSE,
        np.ones((close_ksize, close_ksize), np.uint8),
    )
    base = cv2.morphologyEx(base, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    base = _largest_component(base)
    contours, _ = cv2.findContours(base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = np.zeros_like(base)
    if contours:
        cv2.drawContours(
            region, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED,
        )
    return region > 0


def detect_stains(img, mask_filled, mask_raw, bg_color):
    """按材料外观选择颜色规则,并只在重建出的手套区域内找污渍。

    浅色针织/乳胶手套:从正常白灰材料重建完整表面,再同时要求候选色度
    偏离材料、也偏离背景,并形成连续且不细长的色块。这样大片污渍即使
    被前景分割误删仍可恢复,黄色背景从网孔透出则不会被当成污渍。
    彩色手套:先找主色相,再找偏离主色的区域;另用严格的局部 Lab 距离
    补检黑色、白色或与主色相近但明显更深的污渍。候选仍与 ``mask_raw``
    相交,避免手指间背景被重建轮廓包进去。

    Trade-off:中性色手套上的白色粉迹、非常淡或贴近边缘的小污渍可能漏检。
    阈值以预处理后的 800px 宽图片标定。
    """
    h, w = img.shape[:2]
    erode_ksize = _odd_kernel(STAIN_MASK_ERODE_KSIZE, h, w)
    base_close_ksize = _odd_kernel(STAIN_BASE_CLOSE_KSIZE, h, w)
    neutral_base_close_ksize = _odd_kernel(STAIN_NEUTRAL_BASE_CLOSE_KSIZE, h, w)
    neutral_region_erode_ksize = _odd_kernel(STAIN_NEUTRAL_REGION_ERODE_KSIZE, h, w)
    neutral_close_ksize = _odd_kernel(STAIN_NEUTRAL_CLOSE_KSIZE, h, w)
    local_ksize = _odd_kernel(STAIN_LOCAL_KSIZE, h, w)
    close_ksize = _odd_kernel(STAIN_CLOSE_KSIZE, h, w)
    if min(
        erode_ksize, base_close_ksize, neutral_base_close_ksize,
        neutral_region_erode_ksize, neutral_close_ksize,
        local_ksize, close_ksize,
    ) < 3:
        return []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hue, sat, val = cv2.split(hsv)
    raw_foreground = mask_raw > 0
    # 记录哪些像素是深色规则抓到的,后面对这类候选用更严的面积门槛
    dark_source = np.zeros(mask_raw.shape, bool)
    inside = cv2.erode(
        mask_raw, np.ones((erode_ksize, erode_ksize), np.uint8),
    ) > 0
    if not inside.any():
        return []

    neutral_light = inside & (sat <= STAIN_NEUTRAL_S_MAX) & (val >= STAIN_LIGHT_V_MIN)
    neutral_ratio = neutral_light.sum() / inside.sum()
    candidate = np.zeros(mask_raw.shape, np.uint8)
    evidence_map = np.zeros(mask_raw.shape, np.float32)
    glove_region = np.zeros(mask_raw.shape, dtype=bool)
    colorful_branch = neutral_ratio < STAIN_NEUTRAL_RATIO

    # 分割干净时直接用分割轮廓当检测区。用"材料主色重建"的老办法会把靠近
    # 手套边缘的大块污渍排到区域外 —— 实测在真实照片上丢掉 50%~94% 的污渍
    # 像素。只有分割被背景污染时,重建方案才是更安全的选择。
    seg_clean = _segmentation_is_clean(mask_filled, lab_u8.astype(np.float32), bg_color)
    seg_erode_ksize = _odd_kernel(STAIN_SEG_ERODE_KSIZE, h, w)

    if not colorful_branch:
        if seg_clean:
            glove_region = mask_filled > 0
        else:
            glove_region = _region_from_base(
                neutral_light.astype(np.uint8) * 255, neutral_base_close_ksize,
            )
        if glove_region.any():
            erode_k = seg_erode_ksize if seg_clean else neutral_region_erode_ksize
            glove_inside = cv2.erode(
                glove_region.astype(np.uint8) * 255,
                np.ones((erode_k, erode_k), np.uint8),
            ) > 0
            base_lab = np.median(lab_u8[neutral_light], axis=0).astype(np.float32)
            lab_float = lab_u8.astype(np.float32)
            chroma_dist = np.hypot(
                lab_float[:, :, 1] - base_lab[1],
                lab_float[:, :, 2] - base_lab[2],
            )
            background_chroma_dist = np.hypot(
                lab_float[:, :, 1] - bg_color[1],
                lab_float[:, :, 2] - bg_color[2],
            )
            direct_pixels = (
                glove_inside
                & (chroma_dist >= STAIN_NEUTRAL_CHROMA_DIST)
                & (background_chroma_dist >= STAIN_NEUTRAL_BG_CHROMA_DIST)
            )
            density_kernel = (
                STAIN_NEUTRAL_DENSITY_KSIZE, STAIN_NEUTRAL_DENSITY_KSIZE,
            )
            local_density = cv2.boxFilter(
                direct_pixels.astype(np.float32), -1,
                density_kernel, normalize=True,
            )
            stain_pixels = direct_pixels & (
                local_density >= STAIN_NEUTRAL_DENSITY_MIN
            )
            candidate[stain_pixels] = 255
            material_strength = np.clip(
                (chroma_dist - STAIN_NEUTRAL_CHROMA_DIST)
                / max(STAIN_NEUTRAL_CHROMA_DIST * 2.0, 1.0),
                0.0, 1.0,
            )
            background_strength = np.clip(
                (background_chroma_dist - STAIN_NEUTRAL_BG_CHROMA_DIST)
                / max(STAIN_NEUTRAL_BG_CHROMA_DIST * 1.5, 1.0),
                0.0, 1.0,
            )
            density_strength = np.clip(
                (local_density - STAIN_NEUTRAL_DENSITY_MIN) / 0.50,
                0.0, 1.0,
            )
            strength = (
                0.50 * material_strength
                + 0.25 * background_strength
                + 0.25 * density_strength
            )
            evidence_map[stain_pixels] = 0.55 + 0.45 * strength[stain_pixels]
    else:
        colorful = inside & (sat >= STAIN_COLOR_S_MIN) & (val >= STAIN_COLOR_V_MIN)
        if colorful.any():
            hist = np.bincount(hue[colorful], minlength=180).astype(np.float32)
            # hue 是环形的:把两端接起来后做 9-bin 平滑,取主峰。
            smooth = np.convolve(
                np.r_[hist[-4:], hist, hist[:4]], np.ones(9), mode="valid",
            )
            dominant_hue = int(np.argmax(smooth) % 180)
            raw_delta = np.abs(hue.astype(np.int16) - dominant_hue)
            hue_delta = np.minimum(raw_delta, 180 - raw_delta)
            base = colorful & (hue_delta <= STAIN_BASE_HUE_TOL)
            if seg_clean:
                glove_region = cv2.erode(
                    mask_filled,
                    np.ones((seg_erode_ksize, seg_erode_ksize), np.uint8),
                ) > 0
            else:
                glove_region = _region_from_base(
                    base.astype(np.uint8) * 255, base_close_ksize,
                )
            # 规则①:色相明显偏离主色 —— 抓泥巴、彩色污渍
            hue_stain = (
                glove_region & raw_foreground
                & (sat >= STAIN_COLOR_S_MIN) & (val >= STAIN_COLOR_V_MIN)
                & (hue_delta >= STAIN_HUE_DIST)
            )
            # 规则②:亮度远低于材料 —— 抓黑色油漆/墨渍,这类色相不可靠
            base_l = float(np.median(lab_u8[base, 0])) if base.any() else 0.0
            l_channel = lab_u8[:, :, 0].astype(np.float32)
            dark_stain = (
                glove_region & raw_foreground
                & (l_channel <= base_l - STAIN_DARK_L_DROP)   # 比材料暗得多
                & (l_channel <= STAIN_DARK_L_ABS)              # 且绝对够暗
            )
            dark_source |= dark_stain
            stain_pixels = hue_stain | dark_stain

            candidate[stain_pixels] = 255
            hue_strength = np.clip(
                (hue_delta.astype(np.float32) - STAIN_HUE_DIST)
                / max(90.0 - STAIN_HUE_DIST, 1.0),
                0.0, 1.0,
            )
            dark_strength = np.clip(
                (base_l - lab_u8[:, :, 0].astype(np.float32) - STAIN_DARK_L_DROP)
                / max(STAIN_DARK_L_DROP, 1.0),
                0.0, 1.0,
            )
            strength = np.maximum(hue_strength, dark_strength)
            evidence_map[stain_pixels] = 0.55 + 0.45 * strength[stain_pixels]

    if not glove_region.any():
        neutral_base = inside & (sat <= STAIN_NEUTRAL_S_MAX)
        glove_region = _region_from_base(
            neutral_base.astype(np.uint8) * 255, base_close_ksize,
        )

    # 先检查主色相规则是否已经得到可信候选。真实污渍已经找到时不再叠加
    # 局部 Lab,避免把正常褶皱/反光再标成额外 Stain。只有主规则没有结果时,
    # 才用局部 Lab 补检黑/白/同色深污渍。
    if colorful_branch and glove_region.any():
        primary = cv2.morphologyEx(
            candidate, cv2.MORPH_OPEN,
            np.ones((STAIN_OPEN_KSIZE, STAIN_OPEN_KSIZE), np.uint8),
        )
        primary = cv2.morphologyEx(
            primary, cv2.MORPH_CLOSE,
            np.ones((close_ksize, close_ksize), np.uint8),
        )
        primary_contours, _ = cv2.findContours(
            primary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        has_primary = any(
            cv2.contourArea(contour) >= MIN_AREA_STAIN
            for contour in primary_contours
        )
        if not has_primary:
            local_lab = cv2.medianBlur(lab_u8, local_ksize).astype(np.float32)
            delta = lab_u8.astype(np.float32) - local_lab
            local_dist = np.sqrt(
                (STAIN_LUMA_WEIGHT * delta[:, :, 0]) ** 2
                + delta[:, :, 1] ** 2
                + delta[:, :, 2] ** 2
            )
            local_region = cv2.erode(
                mask_raw, np.ones((local_ksize, local_ksize), np.uint8),
            ) > 0
            local_pixels = (
                local_region & glove_region & (local_dist >= STAIN_LOCAL_DIST)
            )
            candidate[local_pixels] = 255
            strength = np.clip(
                (local_dist - STAIN_LOCAL_DIST) / max(STAIN_LOCAL_DIST * 2.0, 1.0),
                0.0, 1.0,
            )
            evidence_map[local_pixels] = 0.55 + 0.45 * strength[local_pixels]

    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_OPEN,
        np.ones((STAIN_OPEN_KSIZE, STAIN_OPEN_KSIZE), np.uint8),
    )
    final_close_ksize = neutral_close_ksize if not colorful_branch else close_ksize
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE,
        np.ones((final_close_ksize, final_close_ksize), np.uint8),
    )
    contours, _ = cv2.findContours(
        candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    # 先看整幅图是不是"很多小点"的斑点图案。是的话,这些小点属于 Spotting,
    # Stain 主动放弃它们 —— 让两个检测器靠各自的判据分开,而不是只靠
    # deduplicate() 的登记顺序兜底(单独只跑 Stain 时也才不会重复报)。
    glove_area = float((mask_filled > 0).sum())
    spot_max_area = glove_area * SPOTTING_MAX_AREA_RATIO
    # 用跟 detect_spotting 完全相同的判据,确保"让出去"的点对方一定会收
    is_spotting_pattern = (
        len(_find_spots(img, mask_filled, mask_raw, bg_color)) >= SPOTTING_MIN_COUNT
    )

    results = []
    for contour in contours:
        if is_spotting_pattern and cv2.contourArea(contour) <= spot_max_area:
            continue      # 交给 detect_spotting;大块的仍然按 Stain 报
        filled = np.zeros(candidate.shape, np.uint8)
        cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
        blob = cv2.bitwise_and(candidate, filled)
        # 深色规则抓到的区域用更大的面积门槛:手指边缘和指缝的阴影同样是暗的,
        # 实测这类误报只有 660~1449px,而靠亮度落差认出的真污渍从 2734px 起。
        # 但同色系深污渍是局部 Lab 兜底规则找到的,尺寸可以很小,所以不能
        # 对所有候选一刀切,只对深色规则的结果加严。
        blob_mask = blob > 0
        from_dark = bool(blob_mask.any()) and (
            (dark_source & blob_mask).sum() / blob_mask.sum() > 0.5
        )
        min_area = STAIN_COLOR_MIN_AREA if from_dark else MIN_AREA_STAIN
        if cv2.contourArea(contour) < min_area:
            continue
        # 形状过滤对两个分支都适用:指缝阴影、手套边缘是又细又长的条状,
        # 真污渍是紧致的块。原先只在中性色分支做,导致黑漆照片里的指缝
        # 阴影被"深色污渍"规则抓成一堆细长误报。
        if True:
            perimeter = cv2.arcLength(contour, True)
            compactness = (
                4.0 * np.pi * cv2.contourArea(contour)
                / max(perimeter * perimeter, 1.0)
            )
            radius = float(cv2.distanceTransform(
                blob, cv2.DIST_L2, 3,
            ).max())
            min_compactness = (
                STAIN_COLOR_MIN_COMPACTNESS if colorful_branch
                else STAIN_NEUTRAL_MIN_COMPACTNESS
            )
            min_radius = (
                STAIN_COLOR_MIN_RADIUS if colorful_branch
                else STAIN_NEUTRAL_MIN_RADIUS
            )
            if compactness < min_compactness or radius < min_radius:
                continue
        scored = evidence_map[(blob > 0) & (evidence_map > 0)]
        evidence = 55.0 if scored.size == 0 else 100.0 * float(np.percentile(scored, 75))
        results.append(Detection(
            "Stain", cv2.boundingRect(contour), blob, round(evidence, 1),
        ))
    return sorted(results, key=lambda result: (result.box[1], result.box[0]))


# ============================================================
# 检测器登记处:组员写好新函数后,把函数名加到这个列表里就行
# ============================================================
DETECTORS = [
    detect_holes,
    detect_open_tears,
    # Spotting 要排在 Stain 前面:同一批小点两个检测器都会命中,
    # deduplicate() 按登记顺序保留优先级高的,避免重复计数。
    detect_spotting,
    detect_stains,
    detect_oversize,
    # detect_missing_finger,   # 例如:负责"缺指"的组员加在这里
    # detect_wrinkles,         # 例如:负责"褶皱"的组员加在这里
]


def _box_iou(a, b):
    """两个包围框的重叠比例(IoU)。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def deduplicate(defects):
    """同一处缺陷常被多个检测器同时报出来(例如大破洞也符合"薄区"),
    按 DETECTORS 的登记顺序保留优先级最高的那个。
    """
    kept = []
    for defect in defects:
        name, box = defect
        if any(
            _box_iou(box, kept_defect.box) > DEDUP_IOU
            for kept_defect in kept
        ):
            continue
        kept.append(defect)
    return kept


def run_all_detectors(img, mask_filled, mask_raw, bg_color, detectors=None):
    """依次跑登记的检测器,返回 (缺陷列表, 出错信息列表)。

    每个检测器单独 try/except:某个检测器崩了(抛异常)或者返回格式
    写错,只跳过它自己,其余照常运行 —— 12 个检测器坏 1 个,不该让
    整个系统点按钮没反应(demo 占 10% 分数,这是最糟的情况)。

    detectors: 不传就跑 DETECTORS 里登记的全部;GUI 上勾掉某个检测器时,
    会传一个只包含勾中项的子列表进来(方便跳过还没修好的检测器)。
    """
    if detectors is None:
        detectors = DETECTORS
    defects, errors = [], []
    for det in detectors:
        try:
            found = det(img, mask_filled, mask_raw, bg_color)
            for item in found:
                name, box = item
                clean_box = tuple(int(v) for v in box)
                if isinstance(item, Detection):
                    defects.append(Detection(
                        str(name), clean_box, item.mask, float(item.evidence),
                    ))
                else:
                    defects.append(Detection(str(name), clean_box))
        except Exception as e:
            errors.append(f"{det.__name__} 运行出错: {e}")
    return deduplicate(defects), errors


def detection_color(name):
    """返回某种缺陷在结果图中的固定 BGR 颜色。"""
    return DEFECT_COLORS.get(name, DEFAULT_DEFECT_COLOR)


def detection_mask(defect, shape):
    """取得像素级缺陷 mask;旧检测器没有 mask 时才退回矩形区域。"""
    if isinstance(defect, Detection) and defect.mask is not None:
        if defect.mask.shape[:2] == shape[:2]:
            return defect.mask > 0
    _, (x, y, w, h) = defect
    mask = np.zeros(shape[:2], dtype=bool)
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, shape[1]), min(y + h, shape[0])
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def affected_area_percentage(defects, glove_mask):
    """所有缺陷像素并集 ÷ 手套完整轮廓,返回百分比。"""
    glove = glove_mask > 0
    glove_pixels = int(np.count_nonzero(glove))
    if glove_pixels == 0 or not defects:
        return 0.0
    affected = np.zeros(glove.shape, dtype=bool)
    for defect in defects:
        affected |= detection_mask(defect, glove_mask.shape)
    return 100.0 * np.count_nonzero(affected & glove) / glove_pixels


def overall_evidence_score(defects, image_shape):
    """按各检测区域像素数加权的规则证据分数;不是概率。"""
    weighted_sum = 0.0
    total_weight = 0
    for defect in defects:
        weight = int(np.count_nonzero(detection_mask(defect, image_shape)))
        evidence = defect.evidence if isinstance(defect, Detection) else 0.0
        if weight > 0:
            weighted_sum += float(evidence) * weight
            total_weight += weight
    return weighted_sum / total_weight if total_weight else 0.0


# 标注基准尺寸:预处理后统一宽度 800px 的横构图作为 1.0 倍。
DRAW_REF_SIZE = 800.0


def _annotation_scale(shape):
    """按图片最长边算标注放大倍数。

    为什么需要:预处理把所有图统一成 800px 宽,但竖构图的高度会到 1400px。
    GUI 面板是固定大小,竖图要缩到约 0.32 倍才放得下,横图只缩到 0.58 倍 ——
    如果字号和线宽写死成固定像素,竖图上的标注缩完就几乎看不见了。
    按最长边放大标注,缩放之后两种构图看起来才一样清楚。
    """
    longest = max(shape[0], shape[1])
    return max(1.0, longest / DRAW_REF_SIZE)


def draw_results(img, defects, alpha=0.38):
    """按 defect 固定颜色绘制半透明像素区域、轮廓、定位框与证据分数。"""
    out = img.copy()
    scale = _annotation_scale(img.shape)
    font_scale = 0.5 * scale
    thin = max(1, int(round(1 * scale)))
    thick = max(2, int(round(2 * scale)))
    pad = max(3, int(round(3 * scale)))
    for defect in defects:
        name, (x, y, w, h) = defect
        color = detection_color(name)
        mask = detection_mask(defect, img.shape)
        if mask.any():
            original_pixels = out[mask].astype(np.float32)
            tint = np.asarray(color, dtype=np.float32)
            out[mask] = np.clip(
                original_pixels * (1.0 - alpha) + tint * alpha, 0, 255,
            ).astype(np.uint8)
            mask_u8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(out, contours, -1, color, thick)

        cv2.rectangle(out, (x, y), (x + w, y + h), color, thin)
        evidence = defect.evidence if isinstance(defect, Detection) else 0.0
        label = f"{name} {evidence:.0f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thin,
        )
        gap = pad + 2
        label_y = (y - pad if y - text_h - baseline - gap >= 0
                   else y + text_h + baseline + gap)
        top = max(0, label_y - text_h - baseline - pad)
        bottom = min(out.shape[0] - 1, label_y + pad)
        right = min(out.shape[1] - 1, x + text_w + 2 * pad)
        cv2.rectangle(out, (x, top), (right, bottom), color, cv2.FILLED)
        text_color = (20, 20, 20) if name == "Stain" else (255, 255, 255)
        cv2.putText(
            out, label, (x + pad, label_y - 1), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, text_color, thin, cv2.LINE_AA,
        )
    return out

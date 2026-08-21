# -*- coding: utf-8 -*-
"""
手套分割模块:把手套从背景里分出来,得到一张黑白掩膜(白=手套,黑=背景)。

思路:取画面四条边框的颜色当"背景参考色",算每个像素跟它的颜色距离,
用 Otsu 自动阈值找分界线。

为什么用背景色做参考、不用画面中心块当"手套色":
  - 手套不一定在正中央,取中心块容易取错参考色
  - 更关键的是逻辑问题:如果拿"手套色"当参考,分割会把颜色不像手套的
    像素排除掉 —— 污渍恰好就是这种像素,污渍检测器就永远找不到它,
    反而会被破洞检测器误报成"破洞"(缺陷类型报错)
  用背景色做参考后,污渍"不是背景色"会留在掩膜里,破洞"露出的正是
  背景色"才会被当成破洞候选,这两个问题不用额外处理就自动解决了。
"""
import cv2
import numpy as np

BORDER_RATIO = 0.06                          # 画面四边取多宽的一圈当背景采样区
MIN_AREA_RATIO, MAX_AREA_RATIO = 0.05, 0.95  # 手套面积占比的合理范围
CLOSE_KSIZE = 7   # 形态学闭运算核:补缝隙,大一点没关系
OPEN_KSIZE = 3    # 形态学开运算核:去噪点,必须小 —— 用7的话会把撕裂的
                  # 细缝也一起抹掉(实测发现的坑)

# 色度差的 90 分位达到它,就只用 a/b 色度做分割,不看亮度 L。
# 为什么要分两种:
#   彩色背景(比如黄色垫子)+ 白手套时,亮度反而是干扰 —— 垫子上的阴影
#   亮度差很大会被误判成手套,而真污渍亮度接近材料却掉到阈值以下。实测
#   4 张白棉照片,改用纯色度后污渍被掩膜覆盖率从 2~3/4 升到 4/4。
#   但灰手套配灰白背景时两者色度都接近中性,只能靠亮度区分,这时必须
#   退回完整 Lab 距离。
CHROMA_SEG_MIN_SPREAD = 12.0


def get_background_color(img):
    """取画面四条边框像素的颜色中位数(Lab 空间),当背景参考色。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    bh, bw = max(int(h * BORDER_RATIO), 1), max(int(w * BORDER_RATIO), 1)
    border = np.concatenate([
        lab[:bh].reshape(-1, 3), lab[-bh:].reshape(-1, 3),
        lab[:, :bw].reshape(-1, 3), lab[:, -bw:].reshape(-1, 3),
    ])
    return np.median(border, axis=0)


def segment_glove(img):
    """返回 (mask_filled, mask_raw):
    - mask_raw   : 手套实际像素(破洞处是黑的,因为破洞露出的是背景)
    - mask_filled: 手套完整外轮廓填满(破洞也被填成白色)
    两者相减 = "轮廓内、但颜色是背景色"的区域 -> 破洞候选。
    """
    bg_color = get_background_color(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 每个像素跟背景色的距离,Otsu 自动找"多远算手套"的分界线。
    # 背景本身有明显色相时只比色度,避免阴影/反光的亮度差干扰(见上面说明)。
    chroma = np.hypot(lab[:, :, 1] - bg_color[1], lab[:, :, 2] - bg_color[2])
    if float(np.percentile(chroma, 90)) >= CHROMA_SEG_MIN_SPREAD:
        dist = chroma
    else:
        dist = np.linalg.norm(lab - bg_color, axis=2)
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KSIZE,) * 2)
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KSIZE,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)

    # 只保留最大连通区域,并填满内部
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_filled, [biggest], -1, 255, cv2.FILLED)

    mask_raw = cv2.bitwise_and(mask, mask_filled)
    return mask_filled, mask_raw


def glove_found(mask_filled):
    """手套面积占比是否在合理范围;不合理就代表这张图没找到手套。
    返回 (是否找到, 面积占比)。
    """
    ratio = float(mask_filled.mean() / 255)
    return MIN_AREA_RATIO < ratio < MAX_AREA_RATIO, ratio


def get_glove_color(img, mask_raw):
    """手套正常颜色的参考色(Lab 中位数)。先把掩膜向内腐蚀,避开边缘上
    "手套色和背景色混合"的像素。找不到手套像素时返回 None。
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    inner = cv2.erode(mask_raw, np.ones((9, 9), np.uint8)) > 0
    return np.median(lab[inner], axis=0) if inner.any() else None

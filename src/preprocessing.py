# -*- coding: utf-8 -*-
"""
Preprocessing: resize the image, remove noise, fix uneven lighting.
"""
import cv2

MAX_HEIGHT = 2400  # cap the height after resizing, so a pathological aspect
                    # ratio can't blow up memory (e.g. a 1x600 image resized
                    # to width=800 would become 800x480000)


def resize_image(img, target_width=800):
    """Resize to a fixed width, keeping the aspect ratio."""
    h, w = img.shape[:2]
    scale = target_width / w
    if h * scale > MAX_HEIGHT:          # extreme aspect ratio: scale by height instead
        scale = MAX_HEIGHT / h
    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)
    return cv2.resize(img, (new_w, new_h))


def denoise(img):
    """Median blur: removes noise but keeps edges sharper than a Gaussian
    blur would (hole edges need to survive)."""
    h, w = img.shape[:2]
    ksize = min(5, h, w)
    if ksize % 2 == 0:
        ksize -= 1
    return cv2.medianBlur(img, ksize) if ksize >= 3 else img


def fix_lighting(img, clip_limit=2.0, grid=8):
    """CLAHE: equalise only the L (lightness) channel in Lab space, leaving
    colour untouched, to cancel out "bright on one side, dark on the
    other" lighting without changing the glove's actual colour.

    We also tried grey-world white balance as a comparison, but it
    actually dropped the test pass rate from 13 to 11 (it desaturates a
    strongly-coloured background towards grey, weakening the glove/
    background colour contrast), so it wasn't used in the end.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess(img, target_width=800, fix_light=True):
    """Full preprocessing pipeline, returns (img_norm, img_plain):
    - img_plain: resize + denoise only
    - img_norm : img_plain plus lighting fix
    Colour-based detectors (stains) use img_norm; texture-based detectors
    (wrinkles, later on) should use img_plain, since CLAHE can amplify
    fabric texture into a false "wrinkle".

    Set fix_light=False to switch off the lighting fix with one flag, for
    A/B comparisons.
    """
    img_plain = denoise(resize_image(img, target_width))
    img_norm = fix_lighting(img_plain) if fix_light else img_plain.copy()
    return img_norm, img_plain

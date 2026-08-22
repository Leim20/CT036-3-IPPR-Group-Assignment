# -*- coding: utf-8 -*-
"""
Glove isolation view: strip the background away and show ONLY the glove,
so a cut in the silhouette is obvious to the eye.

Run it on a folder or a single image:

    python src/isolate.py <path-to-image-or-folder> [output-folder]

For each input it writes a 3-panel strip:
    photo | glove on flat background | silhouette with the outline traced

Why this exists: on a cluttered photograph a small cut in the glove edge
is easy to miss, both for a person labelling ground truth and for anyone
reviewing detector output. Removing the background turns "is there a cut
here?" into a question about one clean shape.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import preprocess
from segmentation import segment_glove

FLAT_BG = (32, 32, 32)      # neutral ground the isolated glove is placed on
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def imread_unicode(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def isolate(img_bgr, mask):
    """Glove pixels kept, everything else replaced by a flat colour."""
    out = np.full_like(img_bgr, FLAT_BG, dtype=np.uint8)
    np.copyto(out, img_bgr, where=(mask > 0)[:, :, None])
    return out


def silhouette(mask):
    """White glove shape on black, with its outline traced in red so a
    notch in the boundary reads clearly."""
    out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), 2)
    return out


def build_view(path, width=760):
    img_norm, plain = preprocess(imread_unicode(path), target_width=width)
    mask_filled, mask_raw = segment_glove(img_norm)
    panels = [("photo", plain),
              ("glove isolated", isolate(plain, mask_raw)),
              ("silhouette", silhouette(mask_filled))]
    out = []
    for title, panel in panels:
        p = panel.copy()
        cv2.rectangle(p, (0, 0), (p.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(p, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        out.append(p)
    return np.hstack(out)


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    target = argv[0]
    out_dir = argv[1] if len(argv) > 1 else "isolated"
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isdir(target):
        files = [os.path.join(target, f) for f in sorted(os.listdir(target))
                 if os.path.splitext(f)[1].lower() in IMG_EXT]
    else:
        files = [target]

    for f in files:
        view = build_view(f)
        name = os.path.splitext(os.path.basename(f))[0] + "_isolated.png"
        cv2.imwrite(os.path.join(out_dir, name), view)
        print("  ", os.path.join(out_dir, name))
    print(f"{len(files)} image(s) written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

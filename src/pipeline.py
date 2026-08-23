# -*- coding: utf-8 -*-
"""Shared end-to-end glove inspection pipeline.

Both the GUI and batch evaluator call this module. Keeping acquisition,
preprocessing, segmentation and detection in one place prevents the demo from
silently using different logic from the reported evaluation results.
"""
from pathlib import Path

import cv2
import numpy as np

from preprocessing import preprocess
from segmentation import segment_glove, glove_found, get_background_color
from defect_detection import (
    build_defect_masks,
    draw_results,
    run_all_detectors,
)


def infer_material(image_path=None, material=None):
    """Resolve trusted material metadata without inspecting the filename.

    A recognised explicit value or material folder is retained. Images stored
    directly in ``dataset/raw`` and uploaded photos return ``None`` so the Thin
    detector evaluates its image-content rules automatically.
    """
    candidates = []
    if material is not None:
        candidates.append(str(material))
    if image_path is not None:
        path = Path(image_path)
        candidates.append(path.parent.name)

    for candidate in candidates:
        normalized = "".join(
            character.lower() if character.isalnum() else "_"
            for character in candidate
        )
        normalized = "_".join(part for part in normalized.split("_") if part)
        if "latex_foam" in normalized:
            return "latex_foam"
        if normalized == "latex":
            return "latex"
        if "nitrile" in normalized:
            return "nitrile"
        if "cotton" in normalized:
            return "cotton"
    return None


def read_image(image_path):
    """Read an image from a path, including paths with non-ASCII characters."""
    try:
        data = np.fromfile(str(Path(image_path)), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _empty_result(status_message, material=None):
    """Create a predictable result object for acquisition failures."""
    return {
        "original_image": None,
        "normalized_image": None,
        "glove_mask": None,
        "raw_glove_mask": None,
        "defect_mask": None,
        "defects": [],
        "defect_labels": [],
        "defect_locations": [],
        "features": {"material": material, "glove_area_ratio": 0.0},
        "debug_images": {},
        "result_image": None,
        "glove_found": False,
        "errors": [],
        "status_message": status_message,
    }


def process_image_array(image, material=None, detectors=None):
    """Run the complete pipeline on an already decoded BGR image.

    ``detectors`` can limit execution to one selected GUI detector. When it is
    omitted, every registered detector runs, which is the batch-evaluation
    behaviour.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return _empty_result("failed to read image", material)

    img_norm, img_plain = preprocess(image)
    mask_filled, mask_raw = segment_glove(img_norm)
    found, area_ratio = glove_found(mask_filled)

    result = _empty_result("no glove detected", material)
    result.update(
        {
            "original_image": img_plain,
            "normalized_image": img_norm,
            "glove_mask": mask_filled,
            "raw_glove_mask": mask_raw,
            "features": {
                "material": material,
                "glove_area_ratio": area_ratio,
                "defect_count": 0,
            },
            "debug_images": {
                "original": img_plain,
                "normalized": img_norm,
                "glove_mask": mask_filled,
                "raw_glove_mask": mask_raw,
            },
            "result_image": img_plain.copy(),
            "glove_found": found,
        }
    )
    if not found:
        result["defect_mask"] = np.zeros(mask_filled.shape, dtype=np.uint8)
        result["debug_images"]["defect_mask"] = result["defect_mask"]
        return result

    bg_color = get_background_color(img_norm)
    defects, errors = run_all_detectors(
        img_norm,
        mask_filled,
        mask_raw,
        bg_color,
        detectors=detectors,
        img_plain=img_plain,
        material=material,
    )

    # Recognition keeps the required (label, bounding-box) detector contract.
    # Rebuild the accepted colour/shape evidence as pixel-level masks for the
    # lecturer-required segmented output.
    segmented_regions = build_defect_masks(
        img_norm,
        mask_filled,
        mask_raw,
        bg_color,
        defects,
        img_plain=img_plain,
        material=material,
    )
    # Teammate detectors already provide their pixel masks. The three detectors
    # that retain the assignment's legacy tuple return contract receive their
    # reconstructed segmentation masks here, so every downstream display and
    # metric uses the same pixel-level representation.
    for defect, region_mask in zip(defects, segmented_regions):
        if hasattr(defect, "mask") and defect.mask is None:
            defect.mask = region_mask

    defect_mask = np.zeros(mask_filled.shape, dtype=np.uint8)
    for region_mask in segmented_regions:
        defect_mask = cv2.bitwise_or(defect_mask, region_mask)

    result.update(
        {
            "defect_mask": defect_mask,
            "defects": defects,
            "defect_labels": [name for name, _ in defects],
            "defect_locations": [box for _, box in defects],
            "features": {
                "material": material,
                "glove_area_ratio": area_ratio,
                "defect_count": len(defects),
            },
            "result_image": draw_results(img_plain, defects),
            "errors": errors,
            "status_message": "defect detected" if defects else "no defects detected",
        }
    )
    result["debug_images"]["defect_mask"] = defect_mask
    return result


def process_image(image_path, material=None, detectors=None):
    """Read ``image_path`` and run the shared inspection pipeline."""
    return process_image_array(
        read_image(image_path),
        material=infer_material(image_path, material),
        detectors=detectors,
    )

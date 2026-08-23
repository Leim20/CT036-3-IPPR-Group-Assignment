# -*- coding: utf-8 -*-
"""Choong Ti Shen's detectors: incomplete beading, damage by fold, improper roll.

Self-contained on purpose. This package carries its own glove segmentation and
imports nothing from the team's ``segmentation``, ``pipeline`` or
``defect_detection`` modules, so nothing tuned here can move a teammate's
result and nothing tuned there can move ours.

The three detectors take the team's detector signature and are registered in
the one shared DETECTORS list, so the system keeps a single GUI. They ignore
the glove mask that signature hands them and re-segment the image themselves --
see the note at the top of ``detection.py``.
"""
from .detection import DETECTORS, Detection      # noqa: F401

__all__ = ["DETECTORS", "Detection"]

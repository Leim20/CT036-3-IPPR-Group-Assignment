# -*- coding: utf-8 -*-
"""Glove Defect Detection System GUI.

Run from the project root with:
    .venv/Scripts/python src/gui.py

The image picker scans ``dataset/raw`` recursively, and the upload button accepts
external inspection photos. Selecting either source previews it immediately.
The user can then run one detector for a focused inspection, or run all
registered detectors for a complete inspection.
"""
from pathlib import Path
import re
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from preprocessing import preprocess
from defect_detection import (
    DETECTORS,
    affected_area_percentage,
    overall_evidence_score,
)
from pipeline import infer_material, process_image_array, read_image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "raw"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

PANEL_W = 390
PANEL_H = 260
OVERLAY_PANEL_W = PANEL_W * 2 + 20
ZOOM_MIN = 0.5
ZOOM_MAX = 3.0
ZOOM_STEP = 0.25

APP_BG = "#EEF2F7"
SURFACE = "#FFFFFF"
NAVY = "#0F172A"
SLATE = "#334155"
MUTED = "#64748B"
BORDER = "#DCE3EC"
PANEL_BG = "#F8FAFC"
PRIMARY = "#2563EB"
PRIMARY_HOVER = "#1D4ED8"
PRIMARY_SOFT = "#DBEAFE"
SUCCESS = "#22C55E"

IMAGE_PLACEHOLDER = "Select an inspection image..."
DETECTOR_PLACEHOLDER = "Select a defect..."
ALL_DETECTORS_LABEL = "All Defects"
SIDE_BY_SIDE_VIEW = "side_by_side"
OVERLAY_VIEW = "overlay"

QUESTION_PAPER_DEFECTS = (
    "Discoloration",
    "Wrinkles/Dent",
    "Oversize",
    "Plastic Contamination",
    "Thin",
    "Damaged by Fold",
    "Spotting",
    "Inside Out",
    "Improper Roll",
)

ADDITIONAL_GROUP_DEFECTS = (
    "Tearing",
    "Open Tear",
    "Side Tear",
    "Incomplete Beading",
    "Damage By Fold",
    "Stain",
    "Finger Not Enough",
    "Fused Fingers",
)

DEFECT_OPTIONS = QUESTION_PAPER_DEFECTS + ADDITIONAL_GROUP_DEFECTS

DETECTOR_LABELS = {
    "detect_tearing": "Tearing",
    "detect_open_tears": "Open Tear",
    "detect_side_tear": "Side Tear",
    "detect_incomplete_beading": "Incomplete Beading",
    "detect_damage_by_fold": "Damage By Fold",
    "detect_improper_roll": "Improper Roll",
    "detect_stains": "Stain",
    "detect_finger_not_enough": "Finger Not Enough",
    "detect_missing_finger": "Finger Not Enough",  # legacy function-name alias
    "detect_fused_fingers": "Fused Fingers",
    "detect_discoloration": "Discoloration",
    "detect_plastic_contamination": "Plastic Contamination",
    "detect_spotting": "Spotting",
    "detect_thin_area": "Thin",
    "detect_wrinkles": "Wrinkles/Dent",
}


def _label_for(detector):
    """Return a concise user-facing label for a detector function."""
    name = detector.__name__
    if name in DETECTOR_LABELS:
        return DETECTOR_LABELS[name]
    if name.startswith("detect_"):
        name = name[len("detect_"):]
    return name.replace("_", " ").title()


def find_dataset_images(dataset_dir=DATASET_DIR):
    """Return every supported image below the dataset directory."""
    if not dataset_dir.exists():
        return []
    return sorted(
        (
            path
            for path in dataset_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.as_posix().lower(),
    )


def display_path(path, dataset_dir=DATASET_DIR):
    """Show a short ``defect / material / filename`` path in the dropdown."""
    try:
        return path.relative_to(dataset_dir).as_posix()
    except ValueError:
        return path.name


class GloveDefectApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Glove Defect Detection System (GDD)")
        self.root.geometry("1180x800")
        self.root.minsize(1080, 720)
        self.root.configure(background=APP_BG)

        self.img_source = None
        self.img_plain = None
        self.selected_image_path = None
        self.selected_image_material = None
        self.selected_image_is_dataset = False
        self.annotated_result = None
        self.result_view_image = None
        self.view_mode = SIDE_BY_SIDE_VIEW
        self.zoom_level = 1.0
        self.image_paths = {}
        self.detector_by_label = {_label_for(det): det for det in DETECTORS}

        self.image_var = tk.StringVar(value=IMAGE_PLACEHOLDER)
        self.detector_var = tk.StringVar(value=DETECTOR_PLACEHOLDER)
        self.image_count_var = tk.StringVar()
        self.affected_area_var = tk.StringVar(value="Affected area: —")
        self.evidence_var = tk.StringVar(value="Rule evidence: —")
        self.processing_time_var = tk.StringVar(value="Processing time: —")
        self.zoom_var = tk.StringVar(value="100%")
        available = len(set(DEFECT_OPTIONS) & set(self.detector_by_label))
        self.detector_count_var = tk.StringVar(
            value=f"{available} of {len(DEFECT_OPTIONS)} detectors available"
        )

        self._configure_styles()
        self._build_interface()
        self.refresh_image_list()
        self._show_ready_message()

    def _configure_styles(self):
        """Apply a consistent visual system without changing GUI behaviour."""
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("App.TFrame", background=APP_BG)
        style.configure(
            "Card.TFrame",
            background=SURFACE,
            relief=tk.SOLID,
            borderwidth=1,
        )
        style.configure("Surface.TFrame", background=SURFACE)
        style.configure("Header.TFrame", background=NAVY)
        style.configure("HeaderTitle.TLabel", background=NAVY, foreground="#F8FAFC", font=("Segoe UI", 19, "bold"))
        style.configure("HeaderSubtitle.TLabel", background=NAVY, foreground="#94A3B8", font=("Segoe UI", 9))
        style.configure("HeaderStatus.TLabel", background="#172033", foreground="#E2E8F0", font=("Segoe UI", 9, "bold"), padding=(10, 7))
        style.configure("StatusDot.TLabel", background=NAVY, foreground=SUCCESS, font=("Segoe UI", 13, "bold"))
        style.configure("Eyebrow.TLabel", background=SURFACE, foreground=PRIMARY, font=("Segoe UI", 9, "bold"))
        style.configure("SectionTitle.TLabel", background=SURFACE, foreground=NAVY, font=("Segoe UI", 13, "bold"))
        style.configure("Body.TLabel", background=SURFACE, foreground=SLATE, font=("Segoe UI", 9))
        style.configure("Muted.TLabel", background=SURFACE, foreground=MUTED, font=("Segoe UI", 8))
        style.configure("FieldLabel.TLabel", background=SURFACE, foreground=SLATE, font=("Segoe UI", 9, "bold"))
        style.configure("PanelTitle.TLabel", background=SURFACE, foreground=NAVY, font=("Segoe UI", 10, "bold"))
        style.configure("Metric.TFrame", background=PANEL_BG, relief=tk.SOLID, borderwidth=1)
        style.configure("Metric.TLabel", background=PANEL_BG, foreground=NAVY, font=("Segoe UI", 10, "bold"))
        style.configure("Hint.TLabel", background=SURFACE, foreground=MUTED, font=("Segoe UI", 8))
        style.configure(
            "ZoomValue.TLabel",
            background=SURFACE,
            foreground=NAVY,
            font=("Segoe UI", 9, "bold"),
            padding=(7, 0),
        )

        style.configure(
            "Primary.TButton",
            background=PRIMARY,
            foreground="#FFFFFF",
            bordercolor=PRIMARY,
            lightcolor=PRIMARY,
            darkcolor=PRIMARY,
            font=("Segoe UI", 10, "bold"),
            padding=(14, 10),
        )
        style.map(
            "Primary.TButton",
            background=[("active", PRIMARY_HOVER), ("disabled", "#94A3B8")],
            foreground=[("disabled", "#E2E8F0")],
        )
        style.configure(
            "Secondary.TButton",
            background=SURFACE,
            foreground=SLATE,
            bordercolor=BORDER,
            lightcolor=SURFACE,
            darkcolor=SURFACE,
            font=("Segoe UI", 9, "bold"),
            padding=(11, 8),
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#F1F5F9"), ("disabled", "#F8FAFC")],
            foreground=[("disabled", "#94A3B8")],
        )
        style.configure(
            "Toggle.TButton",
            background=SURFACE,
            foreground=SLATE,
            bordercolor=BORDER,
            lightcolor=SURFACE,
            darkcolor=SURFACE,
            font=("Segoe UI", 9, "bold"),
            padding=(12, 7),
        )
        style.map(
            "Toggle.TButton",
            background=[("active", PRIMARY_SOFT), ("disabled", PRIMARY_SOFT)],
            foreground=[("active", PRIMARY), ("disabled", PRIMARY)],
        )
        style.configure(
            "Inspection.TCombobox",
            fieldbackground="#FFFFFF",
            background="#FFFFFF",
            foreground=NAVY,
            bordercolor=BORDER,
            arrowcolor=MUTED,
            padding=7,
        )
        style.map(
            "Inspection.TCombobox",
            bordercolor=[("focus", PRIMARY)],
            arrowcolor=[("active", PRIMARY)],
        )

    def _build_interface(self):
        """Build the styled inspection sidebar and result workspace."""
        container = ttk.Frame(self.root, style="App.TFrame", padding=16)
        container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(container, style="Header.TFrame", padding=(20, 14))
        header.pack(fill=tk.X, pady=(0, 14))
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text="GLOVE DEFECT DETECTION",
            style="HeaderTitle.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Visual inspection workspace  /  image processing and defect analysis",
            style="HeaderSubtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        status = ttk.Frame(header, style="Header.TFrame")
        status.grid(row=0, column=1, rowspan=2, sticky="e")
        ttk.Label(status, text="●", style="StatusDot.TLabel").pack(side=tk.LEFT)
        ttk.Label(
            status,
            text="SYSTEM READY",
            style="HeaderStatus.TLabel",
        ).pack(side=tk.LEFT, padx=(5, 0))

        content = ttk.Frame(container, style="App.TFrame")
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(0, minsize=270)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(
            content,
            style="Card.TFrame",
            padding=(18, 18),
        )
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 14))

        ttk.Label(sidebar, text="INSPECTION", style="Eyebrow.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            sidebar,
            text="Configure analysis",
            style="SectionTitle.TLabel",
        ).pack(anchor="w", pady=(3, 4))
        ttk.Label(
            sidebar,
            text="Choose a dataset image or upload a photo, then select the inspection mode.",
            style="Body.TLabel",
            wraplength=225,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 18))

        ttk.Label(
            sidebar,
            text="Inspection image",
            style="FieldLabel.TLabel",
        ).pack(anchor="w", pady=(0, 6))
        self.image_combo = ttk.Combobox(
            sidebar,
            textvariable=self.image_var,
            state="readonly",
            style="Inspection.TCombobox",
            postcommand=self.refresh_image_list,
        )
        self.image_combo.pack(fill=tk.X)
        self.image_combo.bind("<<ComboboxSelected>>", self.on_image_selected)

        self.upload_button = ttk.Button(
            sidebar,
            text="Upload Photo",
            command=self.upload_photo,
            style="Secondary.TButton",
        )
        self.upload_button.pack(fill=tk.X, pady=(8, 5))
        ttk.Label(
            sidebar,
            textvariable=self.image_count_var,
            style="Muted.TLabel",
            wraplength=225,
            justify=tk.LEFT,
        ).pack(anchor="w")

        ttk.Separator(sidebar).pack(fill=tk.X, pady=18)

        ttk.Label(
            sidebar,
            text="Detection mode",
            style="FieldLabel.TLabel",
        ).pack(anchor="w", pady=(0, 6))
        detector_values = [ALL_DETECTORS_LABEL, *DEFECT_OPTIONS]
        self.detector_combo = ttk.Combobox(
            sidebar,
            textvariable=self.detector_var,
            values=detector_values,
            state="readonly",
            style="Inspection.TCombobox",
        )
        self.detector_combo.pack(fill=tk.X)
        self.detector_combo.bind("<<ComboboxSelected>>", self.on_detector_selected)
        ttk.Label(
            sidebar,
            textvariable=self.detector_count_var,
            style="Muted.TLabel",
            wraplength=225,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(5, 0))

        self.run_button = ttk.Button(
            sidebar,
            text="Run Detection",
            command=self.detect,
            state=tk.DISABLED,
            style="Primary.TButton",
        )
        self.run_button.pack(fill=tk.X, pady=(20, 0))

        ttk.Separator(sidebar).pack(fill=tk.X, pady=18)
        ttk.Label(sidebar, text="WORKFLOW", style="Eyebrow.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            sidebar,
            text="01  Select an inspection image\n\n02  Choose a detection mode\n\n03  Run and review the result",
            style="Body.TLabel",
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(8, 0))

        workspace = ttk.Frame(
            content,
            style="Card.TFrame",
            padding=(16, 14),
        )
        workspace.grid(row=0, column=1, sticky="nsew")

        workspace_header = ttk.Frame(workspace, style="Surface.TFrame")
        workspace_header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            workspace_header,
            text="Inspection workspace",
            style="SectionTitle.TLabel",
        ).pack(side=tk.LEFT)
        ttk.Label(
            workspace_header,
            text="Original input and annotated detection output",
            style="Muted.TLabel",
        ).pack(side=tk.RIGHT)

        self.panels = ttk.Frame(
            workspace,
            style="Surface.TFrame",
            height=285,
        )
        self.panels.pack(fill=tk.BOTH, expand=True)
        self.panels.grid_propagate(False)
        self.panels.columnconfigure(0, weight=1, uniform="image-panels")
        self.panels.columnconfigure(1, weight=1, uniform="image-panels")
        self.panels.rowconfigure(0, weight=1)

        self.original_frame = ttk.Frame(
            self.panels, style="Card.TFrame", padding=8
        )
        self.original_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        ttk.Label(
            self.original_frame,
            text="ORIGINAL IMAGE",
            style="PanelTitle.TLabel",
        ).pack(anchor="w", pady=(1, 7))
        self.result_frame = ttk.Frame(
            self.panels, style="Card.TFrame", padding=8
        )
        self.result_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        ttk.Label(
            self.result_frame,
            text="DETECTION RESULT",
            style="PanelTitle.TLabel",
        ).pack(anchor="w", pady=(1, 7))

        self.panel_left = self._create_image_canvas(
            self.original_frame,
            "Choose or upload an inspection image",
        )

        self.panel_right = self._create_image_canvas(
            self.result_frame,
            "Run detection to view the result",
        )

        view_controls = ttk.Frame(workspace, style="Surface.TFrame")
        view_controls.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(
            view_controls,
            text="RESULT VIEW",
            style="Eyebrow.TLabel",
        ).pack(
            side=tk.LEFT, padx=(0, 7)
        )

        self.side_by_side_button = ttk.Button(
            view_controls,
            text="Side by Side",
            command=lambda: self._set_view_mode(SIDE_BY_SIDE_VIEW),
            state=tk.DISABLED,
            style="Toggle.TButton",
        )
        self.side_by_side_button.pack(side=tk.LEFT, padx=(0, 6))

        self.overlay_button = ttk.Button(
            view_controls,
            text="Overlay",
            command=lambda: self._set_view_mode(OVERLAY_VIEW),
            state=tk.DISABLED,
            style="Toggle.TButton",
        )
        self.overlay_button.pack(side=tk.LEFT)

        ttk.Separator(view_controls, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=12
        )
        ttk.Label(
            view_controls,
            text="ZOOM",
            style="Eyebrow.TLabel",
        ).pack(side=tk.LEFT, padx=(0, 7))

        self.zoom_out_button = ttk.Button(
            view_controls,
            text="−",
            width=3,
            command=lambda: self._change_zoom(-ZOOM_STEP),
            state=tk.DISABLED,
            style="Secondary.TButton",
        )
        self.zoom_out_button.pack(side=tk.LEFT)
        ttk.Label(
            view_controls,
            textvariable=self.zoom_var,
            style="ZoomValue.TLabel",
        ).pack(side=tk.LEFT)
        self.zoom_in_button = ttk.Button(
            view_controls,
            text="+",
            width=3,
            command=lambda: self._change_zoom(ZOOM_STEP),
            state=tk.DISABLED,
            style="Secondary.TButton",
        )
        self.zoom_in_button.pack(side=tk.LEFT)

        self.save_button = ttk.Button(
            view_controls,
            text="Save Result",
            command=self.save_result,
            state=tk.DISABLED,
            style="Secondary.TButton",
        )
        self.save_button.pack(side=tk.RIGHT)

        metrics = ttk.Frame(workspace, style="Surface.TFrame")
        metrics.pack(fill=tk.X, pady=(10, 0))
        metrics.columnconfigure(0, weight=1)
        metrics.columnconfigure(1, weight=1)
        metrics.columnconfigure(2, weight=1)
        metric_specs = (
            (self.affected_area_var, 0, (0, 5)),
            (self.evidence_var, 1, 5),
            (self.processing_time_var, 2, (5, 0)),
        )
        for variable, column, padx in metric_specs:
            metric = ttk.Frame(metrics, style="Metric.TFrame", padding=(12, 9))
            metric.grid(row=0, column=column, sticky="ew", padx=padx)
            ttk.Label(
                metric,
                textvariable=variable,
                style="Metric.TLabel",
            ).pack(anchor="w")
        ttk.Label(
            metrics,
            text=(
                "Colour key: Tearing = red   Finger Not Enough = orange   "
                "Thin = magenta   Stain = orange   Spotting = yellow   Plastic = cyan"
            ),
            style="Hint.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

        summary = ttk.Frame(workspace, style="Card.TFrame", padding=9)
        summary.pack(fill=tk.X, pady=(9, 0))
        ttk.Label(
            summary,
            text="DETECTION SUMMARY",
            style="PanelTitle.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        summary_body = ttk.Frame(summary, style="Surface.TFrame")
        summary_body.pack(fill=tk.BOTH, expand=True)

        self.result_box = tk.Text(
            summary_body,
            height=5,
            font=("Consolas", 9),
            wrap=tk.WORD,
            state=tk.DISABLED,
            background=PANEL_BG,
            foreground=SLATE,
            relief=tk.FLAT,
            padx=10,
            pady=8,
            highlightthickness=1,
            highlightbackground=BORDER,
            selectbackground=PRIMARY_SOFT,
        )
        result_scroll = ttk.Scrollbar(
            summary_body, orient=tk.VERTICAL, command=self.result_box.yview
        )
        self.result_box.configure(yscrollcommand=result_scroll.set)
        self.result_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_image_canvas(self, parent, placeholder):
        """Create a bordered canvas that supports click-drag image panning."""
        canvas = tk.Canvas(
            parent,
            width=PANEL_W,
            height=PANEL_H,
            background=PANEL_BG,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=BORDER,
            cursor="arrow",
        )
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.image = None
        canvas.image_item = None
        canvas.placeholder_item = None
        canvas.display_size = None
        canvas.bind("<ButtonPress-1>", self._start_panel_pan)
        canvas.bind("<B1-Motion>", self._drag_panel)
        canvas.bind(
            "<Configure>",
            lambda _event, panel=canvas: self._layout_panel_content(panel),
        )
        self._clear_panel(canvas, placeholder)
        return canvas

    def refresh_image_list(self):
        """Rescan the dataset whenever the image dropdown is opened."""
        paths = find_dataset_images()
        self.image_paths = {display_path(path): path for path in paths}
        self.image_combo.configure(values=list(self.image_paths))

        count = len(paths)
        if count:
            self.image_count_var.set(
                f"{count} dataset image{'s' if count != 1 else ''} available"
            )
        else:
            self.image_count_var.set(
                "No dataset images found. Use Upload Photo or add images under dataset/raw/."
            )

        dataset_path_missing = (
            self.selected_image_is_dataset
            and self.selected_image_path not in self.image_paths.values()
        )
        if dataset_path_missing:
            self.image_var.set(IMAGE_PLACEHOLDER)
            self.selected_image_path = None
            self.selected_image_material = None
            self.selected_image_is_dataset = False
            self.img_source = None
            self.img_plain = None
            self.zoom_level = 1.0
            self.zoom_var.set("100%")
            self._clear_panel(
                self.panel_left, "Choose or upload an inspection image"
            )
            self._reset_detection_result()
            self._update_zoom_buttons()
            self._show_ready_message()
        self._update_run_button()

    def on_image_selected(self, _event=None):
        selected = self.image_var.get()
        path = self.image_paths.get(selected)
        if path is None:
            self._update_run_button()
            return

        self._load_inspection_image(
            path,
            display_name=selected,
            material=path.parent.name,
        )

    def upload_photo(self):
        """Choose an external inspection photo without copying it into the dataset."""
        chosen = filedialog.askopenfilename(
            title="Upload Inspection Photo",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                ("PNG image", "*.png"),
                ("JPEG image", "*.jpg *.jpeg"),
                ("Bitmap image", "*.bmp"),
                ("GIF image", "*.gif"),
                ("WebP image", "*.webp"),
            ],
        )
        if not chosen:
            return

        path = Path(chosen)
        self._load_inspection_image(path, display_name=path.name, material=None)

    def _load_inspection_image(self, path, display_name, material=None):
        """Load one dataset or external image through the same preview workflow.

        State is changed only after both decoding and preprocessing succeed, so a
        cancelled upload or invalid file cannot replace the active inspection.
        """
        path = Path(path)
        try:
            img = read_image(path)
            if img is None:
                raise ValueError("The selected file is not a readable image.")
            _, img_plain = preprocess(img)
        except Exception as exc:
            messagebox.showerror(
                "Image Error",
                f"Cannot read this inspection image:\n{path}\n\n{exc}",
            )
            return False

        self.selected_image_path = path
        self.selected_image_material = infer_material(path, material)
        self.selected_image_is_dataset = material is not None
        self.img_source = img
        self.img_plain = img_plain
        self.image_var.set(display_name)
        self.zoom_level = 1.0
        self.zoom_var.set("100%")
        self._reset_detection_result()
        self._update_zoom_buttons()
        self._show_ready_message()
        self._update_run_button()
        return True

    def on_detector_selected(self, _event=None):
        if self.img_plain is not None:
            self._reset_detection_result()
        self._show_ready_message()
        self._update_run_button()

    def _selected_detectors(self):
        selected = self.detector_var.get()
        if selected == ALL_DETECTORS_LABEL:
            return list(DETECTORS)
        detector = self.detector_by_label.get(selected)
        return [detector] if detector is not None else []

    def _update_run_button(self):
        ready = self.img_source is not None and bool(self._selected_detectors())
        self.run_button.configure(state=tk.NORMAL if ready else tk.DISABLED)

    def _show_ready_message(self):
        image_name = self.image_var.get()
        detector_name = self.detector_var.get()

        detector_unavailable = (
            detector_name in DEFECT_OPTIONS
            and detector_name not in self.detector_by_label
        )
        lines = [
            "Status: DETECTOR NOT AVAILABLE"
            if detector_unavailable
            else "Status: READY"
        ]
        lines.append(
            f"Selected image: {image_name}"
            if self.selected_image_path is not None
            else "Selected image: Not selected"
        )
        lines.append(
            f"Detection mode: {detector_name}"
            if detector_name == ALL_DETECTORS_LABEL or detector_name in DEFECT_OPTIONS
            else "Detection mode: Not selected"
        )
        lines.append("")
        if detector_unavailable:
            lines.append(
                f"{detector_name} is listed for this project, but its detector "
                "has not been implemented yet."
            )
        else:
            lines.append(
                "Choose or upload an inspection image and detection mode, "
                "then click Run Detection."
            )
        self.say("\n".join(lines))

    def _clear_panel(self, panel, text):
        panel.delete("all")
        panel.image = None
        panel.image_item = None
        panel.display_size = None
        panel.configure(cursor="arrow")
        panel.placeholder_item = panel.create_text(
            max(panel.winfo_width(), PANEL_W) / 2,
            max(panel.winfo_height(), PANEL_H) / 2,
            text=text,
            fill=MUTED,
            font=("Segoe UI", 10),
            justify=tk.CENTER,
        )
        self._layout_panel_content(panel)

    def _start_panel_pan(self, event):
        """Remember the pointer position before dragging a zoomed image."""
        if event.widget.image_item is not None:
            event.widget.scan_mark(event.x, event.y)

    def _drag_panel(self, event):
        """Pan a zoomed image without changing its detection data."""
        if event.widget.image_item is not None:
            event.widget.scan_dragto(event.x, event.y, gain=1)

    def _layout_panel_content(self, panel, center_view=False):
        """Centre small images and expose a pannable region for large ones."""
        viewport_w = max(panel.winfo_width(), 1)
        viewport_h = max(panel.winfo_height(), 1)
        if viewport_w <= 1:
            viewport_w = PANEL_W
        if viewport_h <= 1:
            viewport_h = PANEL_H

        if panel.image_item is None or panel.display_size is None:
            if panel.placeholder_item is not None:
                panel.coords(
                    panel.placeholder_item,
                    viewport_w / 2,
                    viewport_h / 2,
                )
            panel.configure(scrollregion=(0, 0, viewport_w, viewport_h))
            return

        display_w, display_h = panel.display_size
        content_w = max(viewport_w, display_w)
        content_h = max(viewport_h, display_h)
        image_x = max(0, (viewport_w - display_w) / 2)
        image_y = max(0, (viewport_h - display_h) / 2)
        panel.coords(panel.image_item, image_x, image_y)
        panel.configure(scrollregion=(0, 0, content_w, content_h))

        if center_view:
            x_offset = max(0, (content_w - viewport_w) / 2)
            y_offset = max(0, (content_h - viewport_h) / 2)
            panel.xview_moveto(x_offset / content_w if content_w else 0)
            panel.yview_moveto(y_offset / content_h if content_h else 0)

    def _change_zoom(self, delta):
        """Change display scale for loaded images without rerunning detection."""
        if self.img_plain is None:
            return
        new_level = min(ZOOM_MAX, max(ZOOM_MIN, self.zoom_level + delta))
        if abs(new_level - self.zoom_level) < 1e-9:
            return
        self.zoom_level = new_level
        self.zoom_var.set(f"{round(self.zoom_level * 100):.0f}%")
        self._set_view_mode(self.view_mode)
        self._update_zoom_buttons()

    def _update_zoom_buttons(self):
        has_image = self.img_plain is not None
        self.zoom_out_button.configure(
            state=(
                tk.NORMAL
                if has_image and self.zoom_level > ZOOM_MIN
                else tk.DISABLED
            )
        )
        self.zoom_in_button.configure(
            state=(
                tk.NORMAL
                if has_image and self.zoom_level < ZOOM_MAX
                else tk.DISABLED
            )
        )

    def _reset_detection_result(self):
        """Clear the previous result, so switching image or detector cannot save
        last run's annotated picture by mistake."""
        self.annotated_result = None
        self.result_view_image = None
        self.save_button.configure(state=tk.DISABLED)
        self.affected_area_var.set("Affected area: —")
        self.evidence_var.set("Rule evidence: —")
        self.processing_time_var.set("Processing time: —")
        self._set_view_mode(SIDE_BY_SIDE_VIEW)

    def _set_view_mode(self, mode):
        """Switch the existing result between two-panel and expanded views."""
        if mode == OVERLAY_VIEW and self.result_view_image is None:
            return

        self.view_mode = mode
        if mode == OVERLAY_VIEW:
            self.original_frame.grid_remove()
            self.result_frame.grid_configure(
                row=0, column=0, columnspan=2, sticky="nsew", padx=0
            )
            self.show_on_panel(
                self.panel_right,
                self.result_view_image,
                max_width=OVERLAY_PANEL_W,
            )
            self.side_by_side_button.configure(state=tk.NORMAL)
            self.overlay_button.configure(state=tk.DISABLED)
            return

        self.original_frame.grid()
        self.original_frame.grid_configure(
            row=0, column=0, columnspan=1, sticky="nsew", padx=(0, 5)
        )
        self.result_frame.grid_configure(
            row=0, column=1, columnspan=1, sticky="nsew", padx=(5, 0)
        )
        if self.img_plain is not None:
            self.show_on_panel(self.panel_left, self.img_plain)
        else:
            self._clear_panel(
                self.panel_left, "Choose or upload an inspection image"
            )
        if self.result_view_image is not None:
            self.show_on_panel(self.panel_right, self.result_view_image)
        else:
            self._clear_panel(self.panel_right, "Run detection to view the result")
        self.side_by_side_button.configure(state=tk.DISABLED)
        self.overlay_button.configure(
            state=tk.NORMAL if self.result_view_image is not None else tk.DISABLED
        )

    def show_on_panel(self, panel, img_bgr, max_width=PANEL_W, max_height=PANEL_H):
        h, w = img_bgr.shape[:2]
        scale = min(max_width / w, max_height / h) * self.zoom_level
        display_w = max(1, int(w * scale))
        display_h = max(1, int(h * scale))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        img_small = cv2.resize(
            img_bgr,
            (display_w, display_h),
            interpolation=interpolation,
        )
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        panel.delete("all")
        panel.image = photo
        panel.placeholder_item = None
        panel.display_size = (display_w, display_h)
        panel.image_item = panel.create_image(0, 0, image=photo, anchor=tk.NW)
        panel.configure(cursor="fleur")
        self._layout_panel_content(panel, center_view=True)

    def save_result(self):
        """Let the user save the full-resolution annotated image on the right
        as a PNG or JPEG."""
        if self.annotated_result is None or self.selected_image_path is None:
            messagebox.showwarning(
                "No Result", "Run detection before saving an annotated result."
            )
            return

        output_dir = PROJECT_ROOT / "results"
        output_dir.mkdir(exist_ok=True)
        mode = re.sub(r"[^A-Za-z0-9]+", "_", self.detector_var.get()).strip("_")
        default_name = f"{self.selected_image_path.stem}_{mode or 'detection'}_annotated.png"
        chosen = filedialog.asksaveasfilename(
            title="Save Annotated Detection Result",
            initialdir=str(output_dir),
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("JPEG image", "*.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )
        if not chosen:
            return

        path = Path(chosen)
        extension = path.suffix.lower()
        if extension not in {".png", ".jpg", ".jpeg"}:
            path = path.with_suffix(".png")
            extension = ".png"
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if extension in {".jpg", ".jpeg"} else []
        success, encoded = cv2.imencode(extension, self.annotated_result, params)
        if not success:
            messagebox.showerror("Save Error", "The annotated image could not be encoded.")
            return
        try:
            encoded.tofile(str(path))
        except OSError as exc:
            messagebox.showerror("Save Error", f"Cannot save the result:\n{exc}")
            return
        messagebox.showinfo("Result Saved", f"Annotated result saved to:\n{path}")

    def say(self, text):
        self.result_box.configure(state=tk.NORMAL)
        self.result_box.delete("1.0", tk.END)
        self.result_box.insert(tk.END, text + "\n")
        self.result_box.configure(state=tk.DISABLED)

    def detect(self):
        if self.img_source is None or self.selected_image_path is None:
            messagebox.showwarning(
                "Image Required", "Please select an inspection image."
            )
            return

        detectors = self._selected_detectors()
        if not detectors:
            selected = self.detector_var.get()
            if selected in DEFECT_OPTIONS:
                messagebox.showwarning(
                    "Detector Not Available",
                    f"The {selected} detector has not been implemented yet.",
                )
            else:
                messagebox.showwarning("Detector Required", "Please select a defect.")
            return

        self.run_button.configure(text="Detecting...", state=tk.DISABLED)
        self.root.update_idletasks()
        try:
            self._detect(detectors)
        except Exception as exc:
            self._reset_detection_result()
            self._clear_panel(self.panel_right, "Detection could not be completed")
            self.say(
                "Status: ERROR\n"
                f"Selected image: {self.image_var.get()}\n"
                f"Detection mode: {self.detector_var.get()}\n\n"
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            self.run_button.configure(text="Run Detection")
            self._update_run_button()

    def _detect(self, detectors):
        started = time.perf_counter()
        result = process_image_array(
            self.img_source,
            material=self.selected_image_material,
            detectors=detectors,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        ratio = result["features"]["glove_area_ratio"]

        if not result["glove_found"]:
            result_img = result["result_image"]
            if result_img is not None:
                self.result_view_image = result_img.copy()
                self._set_view_mode(self.view_mode)
            self.processing_time_var.set(f"Processing time: {elapsed_ms:.1f} ms")
            self.say(
                "Status: NO GLOVE DETECTED\n"
                f"Selected image: {self.image_var.get()}\n"
                f"Detection mode: {self.detector_var.get()}\n"
                f"Glove area: {ratio:.1%} of the frame\n\n"
                "Use an image with the glove centred on a plain, contrasting background."
            )
            return

        defects = result["defects"]
        errors = result["errors"]
        mask_filled = result["glove_mask"]
        result_img = result["result_image"]
        affected_pct = affected_area_percentage(defects, mask_filled)
        evidence_score = overall_evidence_score(defects, result_img.shape)

        self.annotated_result = result_img.copy()
        self.result_view_image = result_img.copy()
        self.save_button.configure(state=tk.NORMAL)
        self.affected_area_var.set(f"Affected area: {affected_pct:.2f}%")
        self.evidence_var.set(
            f"Rule evidence: {evidence_score:.1f}/100" if defects
            else "Rule evidence: Not applicable"
        )
        self.processing_time_var.set(f"Processing time: {elapsed_ms:.1f} ms")
        self._set_view_mode(self.view_mode)

        mode = self.detector_var.get()
        if defects:
            lines = ["Status: DEFECT DETECTED"]
        elif mode == ALL_DETECTORS_LABEL:
            lines = ["Status: PASSED - NO DEFECTS DETECTED"]
        else:
            lines = [f"Status: NO {mode.upper()} DETECTED"]

        lines.extend(
            [
                f"Selected image: {self.image_var.get()}",
                f"Detection mode: {mode}",
                f"Detected regions: {len(defects)}",
                f"Affected area: {affected_pct:.2f}% of glove",
                (
                    f"Rule evidence score: {evidence_score:.1f}/100 "
                    "(rule strength, not probability)"
                    if defects else "Rule evidence score: Not applicable"
                ),
                f"Processing time: {elapsed_ms:.1f} ms",
                "",
            ]
        )

        if defects:
            for index, defect in enumerate(defects, 1):
                name, (x, y, w, h) = defect
                region_pct = affected_area_percentage([defect], mask_filled)
                region_evidence = getattr(defect, "evidence", 0.0)
                lines.append(
                    f"{index}. {name} | affected={region_pct:.2f}% | "
                    f"evidence={region_evidence:.1f}/100 | "
                    f"box x={x}, y={y}, w={w}, h={h}"
                )
            if mode == "Tearing":
                lines.append(
                    "Tearing mode checks both enclosed and open/fingertip geometry."
                )
        elif mode != ALL_DETECTORS_LABEL:
            lines.append(
                f"Only the {mode} detector was run; this is not a full glove inspection."
            )
        else:
            lines.append("All registered detectors completed without finding a defect.")

        if errors:
            lines.extend(
                [
                    "",
                    f"WARNING: {len(errors)} detector(s) failed; the result may be incomplete.",
                    *(f"- {error}" for error in errors),
                ]
            )

        self.say("\n".join(lines))


if __name__ == "__main__":
    root = tk.Tk()
    app = GloveDefectApp(root)
    root.mainloop()

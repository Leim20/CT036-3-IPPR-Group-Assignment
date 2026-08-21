# -*- coding: utf-8 -*-
"""Glove Defect Detection System GUI.

Run from the project root with:
    .venv/Scripts/python src/gui.py

The image picker scans ``dataset/raw`` recursively. Selecting an image previews
it immediately. The user can then run one detector for a focused test, or run
all registered detectors for a complete inspection.
"""
from pathlib import Path
import re
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from preprocessing import preprocess
from segmentation import segment_glove, glove_found, get_background_color
from defect_detection import (
    DETECTORS,
    affected_area_percentage,
    draw_results,
    overall_evidence_score,
    run_all_detectors,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "raw"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

PANEL_W = 460
PANEL_H = 340

IMAGE_PLACEHOLDER = "Select a test image..."
DETECTOR_PLACEHOLDER = "Select a defect..."
ALL_DETECTORS_LABEL = "All Defects"

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
    "Hole / Puncture",
    "Open Tear",
    "Stain",
    "Missing Finger",
    "Fused Fingers",
)

DEFECT_OPTIONS = QUESTION_PAPER_DEFECTS + ADDITIONAL_GROUP_DEFECTS

DETECTOR_LABELS = {
    "detect_holes": "Hole / Puncture",
    "detect_open_tears": "Open Tear",
    "detect_stains": "Stain",
    "detect_missing_finger": "Missing Finger",
    "detect_fused_fingers": "Fused Fingers",
    "detect_discoloration": "Discoloration",
    "detect_oversize": "Oversize",
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
    """Show a short ``material / defect / filename`` path in the dropdown."""
    try:
        return path.relative_to(dataset_dir).as_posix()
    except ValueError:
        return path.name


class GloveDefectApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Glove Defect Detection System (GDD)")
        self.root.minsize(980, 720)

        self.img_norm = None
        self.img_plain = None
        self.selected_image_path = None
        self.annotated_result = None
        self.image_paths = {}
        self.detector_by_label = {_label_for(det): det for det in DETECTORS}

        self.image_var = tk.StringVar(value=IMAGE_PLACEHOLDER)
        self.detector_var = tk.StringVar(value=DETECTOR_PLACEHOLDER)
        self.image_count_var = tk.StringVar()
        self.affected_area_var = tk.StringVar(value="Affected area: —")
        self.evidence_var = tk.StringVar(value="Rule evidence: —")
        self.processing_time_var = tk.StringVar(value="Processing time: —")
        available = len(set(DEFECT_OPTIONS) & set(self.detector_by_label))
        self.detector_count_var = tk.StringVar(
            value=f"{available} of {len(DEFECT_OPTIONS)} detectors available"
        )

        self._build_interface()
        self.refresh_image_list()
        self._show_ready_message()

    def _build_interface(self):
        """Build the three-part interface: controls, images and result summary."""
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            container,
            text="Glove Defect Detection System",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        controls = ttk.LabelFrame(container, text="Detection Setup", padding=10)
        controls.pack(fill=tk.X, pady=(0, 10))
        controls.columnconfigure(0, weight=3)
        controls.columnconfigure(1, weight=2)

        ttk.Label(controls, text="Test Image").grid(
            row=0, column=0, sticky="w", padx=(0, 10)
        )
        ttk.Label(controls, text="Defect / Detection Mode").grid(
            row=0, column=1, sticky="w", padx=(0, 10)
        )
        ttk.Label(controls, text="Action").grid(row=0, column=2, sticky="w")

        self.image_combo = ttk.Combobox(
            controls,
            textvariable=self.image_var,
            state="readonly",
            width=48,
            postcommand=self.refresh_image_list,
        )
        self.image_combo.grid(row=1, column=0, sticky="ew", padx=(0, 10), pady=(3, 0))
        self.image_combo.bind("<<ComboboxSelected>>", self.on_image_selected)

        detector_values = [ALL_DETECTORS_LABEL, *DEFECT_OPTIONS]
        self.detector_combo = ttk.Combobox(
            controls,
            textvariable=self.detector_var,
            values=detector_values,
            state="readonly",
            width=29,
        )
        self.detector_combo.grid(
            row=1, column=1, sticky="ew", padx=(0, 10), pady=(3, 0)
        )
        self.detector_combo.bind("<<ComboboxSelected>>", self.on_detector_selected)

        action_buttons = ttk.Frame(controls)
        action_buttons.grid(row=1, column=2, sticky="ew", pady=(3, 0))
        action_buttons.columnconfigure(0, weight=1)
        action_buttons.columnconfigure(1, weight=1)

        self.run_button = ttk.Button(
            action_buttons,
            text="Run Detection",
            command=self.detect,
            state=tk.DISABLED,
        )
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.save_button = ttk.Button(
            action_buttons,
            text="Save Result",
            command=self.save_result,
            state=tk.DISABLED,
        )
        self.save_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        ttk.Label(
            controls,
            textvariable=self.image_count_var,
            foreground="#4b5563",
        ).grid(row=2, column=0, sticky="w", pady=(5, 0))
        ttk.Label(
            controls,
            textvariable=self.detector_count_var,
            foreground="#4b5563",
        ).grid(row=2, column=1, sticky="w", pady=(5, 0))

        panels = ttk.Frame(container)
        panels.pack(fill=tk.BOTH, expand=True)
        panels.columnconfigure(0, weight=1, uniform="image-panels")
        panels.columnconfigure(1, weight=1, uniform="image-panels")
        panels.rowconfigure(0, weight=1)

        original_frame = ttk.LabelFrame(panels, text="Original Image", padding=8)
        original_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        result_frame = ttk.LabelFrame(panels, text="Detection Result", padding=8)
        result_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        self.panel_left = tk.Label(
            original_frame,
            text="Select a dataset image above",
            width=60,
            height=20,
            bg="#f3f4f6",
            fg="#374151",
            relief=tk.SUNKEN,
        )
        self.panel_left.pack(fill=tk.BOTH, expand=True)

        self.panel_right = tk.Label(
            result_frame,
            text="Run detection to view the result",
            width=60,
            height=20,
            bg="#f3f4f6",
            fg="#374151",
            relief=tk.SUNKEN,
        )
        self.panel_right.pack(fill=tk.BOTH, expand=True)

        metrics = ttk.Frame(container, padding=(2, 0))
        metrics.pack(fill=tk.X, pady=(9, 0))
        metrics.columnconfigure(0, weight=1)
        metrics.columnconfigure(2, weight=1)
        metrics.columnconfigure(4, weight=1)
        metric_font = ("Segoe UI", 10, "bold")
        ttk.Label(metrics, textvariable=self.affected_area_var, font=metric_font).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Separator(metrics, orient=tk.VERTICAL).grid(
            row=0, column=1, sticky="ns", padx=12
        )
        ttk.Label(metrics, textvariable=self.evidence_var, font=metric_font).grid(
            row=0, column=2, sticky="w"
        )
        ttk.Separator(metrics, orient=tk.VERTICAL).grid(
            row=0, column=3, sticky="ns", padx=12
        )
        ttk.Label(metrics, textvariable=self.processing_time_var, font=metric_font).grid(
            row=0, column=4, sticky="w"
        )
        ttk.Label(
            metrics,
            text="Colour key: Stain = orange   Hole = red   Open Tear = purple",
            foreground="#4b5563",
        ).grid(row=1, column=0, columnspan=5, sticky="w", pady=(4, 0))

        summary = ttk.LabelFrame(container, text="Detection Summary", padding=8)
        summary.pack(fill=tk.X, pady=(8, 0))

        self.result_box = tk.Text(
            summary,
            height=9,
            font=("Segoe UI", 10),
            wrap=tk.NONE,
            state=tk.DISABLED,
        )
        result_scroll = ttk.Scrollbar(
            summary, orient=tk.VERTICAL, command=self.result_box.yview
        )
        self.result_box.configure(yscrollcommand=result_scroll.set)
        self.result_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)

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
                "No images found. Add test images under dataset/raw/."
            )

        current = self.image_var.get()
        if current not in self.image_paths and current != IMAGE_PLACEHOLDER:
            self.image_var.set(IMAGE_PLACEHOLDER)
            self.selected_image_path = None
            self.img_norm = None
            self.img_plain = None
            self._clear_panel(self.panel_left, "Select a dataset image above")
            self._reset_detection_result()
            self._show_ready_message()
        self._update_run_button()

    def on_image_selected(self, _event=None):
        selected = self.image_var.get()
        path = self.image_paths.get(selected)
        if path is None:
            self._update_run_button()
            return

        img = cv2.imdecode(np.fromfile(path, dtype="uint8"), cv2.IMREAD_COLOR)
        if img is None:
            self.selected_image_path = None
            self.img_norm = None
            self.img_plain = None
            self._update_run_button()
            messagebox.showerror("Image Error", f"Cannot read this image:\n{path}")
            return

        self.selected_image_path = path
        self.img_norm, self.img_plain = preprocess(img)
        self.show_on_panel(self.panel_left, self.img_plain)
        self._reset_detection_result()
        self._show_ready_message()
        self._update_run_button()

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
        ready = self.img_norm is not None and bool(self._selected_detectors())
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
            if image_name in self.image_paths
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
            lines.append("Choose an image and detection mode, then click Run Detection.")
        self.say("\n".join(lines))

    def _clear_panel(self, panel, text):
        panel.configure(image="", text=text, width=60, height=20)
        panel.image = None

    def _reset_detection_result(self):
        """Clear stale output so it cannot be saved after changing inputs."""
        self.annotated_result = None
        self.save_button.configure(state=tk.DISABLED)
        self.affected_area_var.set("Affected area: —")
        self.evidence_var.set("Rule evidence: —")
        self.processing_time_var.set("Processing time: —")
        self._clear_panel(self.panel_right, "Run detection to view the result")

    def show_on_panel(self, panel, img_bgr):
        h, w = img_bgr.shape[:2]
        scale = min(PANEL_W / w, PANEL_H / h)
        display_w = max(1, int(w * scale))
        display_h = max(1, int(h * scale))
        img_small = cv2.resize(img_bgr, (display_w, display_h))
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        panel.configure(image=photo, text="", width=display_w, height=display_h)
        panel.image = photo

    def save_result(self):
        """Save the current full-resolution annotated result as PNG or JPEG."""
        if self.annotated_result is None or self.selected_image_path is None:
            messagebox.showwarning(
                "No Result", "Run detection before saving an annotated result."
            )
            return

        output_dir = PROJECT_ROOT / "results"
        output_dir.mkdir(exist_ok=True)
        mode = re.sub(r"[^A-Za-z0-9]+", "_", self.detector_var.get()).strip("_")
        default_name = (
            f"{self.selected_image_path.stem}_{mode or 'detection'}_annotated.png"
        )
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
        params = (
            [cv2.IMWRITE_JPEG_QUALITY, 95]
            if extension in {".jpg", ".jpeg"} else []
        )
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
        if self.img_norm is None or self.selected_image_path is None:
            messagebox.showwarning("Image Required", "Please select a test image.")
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
        mask_filled, mask_raw = segment_glove(self.img_norm)
        ok, ratio = glove_found(mask_filled)

        if not ok:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.annotated_result = None
            self.save_button.configure(state=tk.DISABLED)
            self.affected_area_var.set("Affected area: —")
            self.evidence_var.set("Rule evidence: —")
            self.processing_time_var.set(f"Processing time: {elapsed_ms:.1f} ms")
            self.show_on_panel(self.panel_right, self.img_plain)
            self.say(
                "Status: NO GLOVE DETECTED\n"
                f"Selected image: {self.image_var.get()}\n"
                f"Detection mode: {self.detector_var.get()}\n"
                f"Glove area: {ratio:.1%} of the frame\n\n"
                "Use an image with the glove centred on a plain, contrasting background."
            )
            return

        bg_color = get_background_color(self.img_norm)
        defects, errors = run_all_detectors(
            self.img_norm,
            mask_filled,
            mask_raw,
            bg_color,
            detectors=detectors,
        )
        result_img = draw_results(self.img_plain, defects)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        affected_pct = affected_area_percentage(defects, mask_filled)
        evidence_score = overall_evidence_score(defects, self.img_plain.shape)

        self.annotated_result = result_img.copy()
        self.save_button.configure(state=tk.NORMAL)
        self.affected_area_var.set(f"Affected area: {affected_pct:.2f}% of glove")
        self.evidence_var.set(
            f"Rule evidence: {evidence_score:.1f}/100" if defects
            else "Rule evidence: —"
        )
        self.processing_time_var.set(f"Processing time: {elapsed_ms:.1f} ms")
        self.show_on_panel(self.panel_right, result_img)

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

# -*- coding: utf-8 -*-
"""Glove Defect Detection System GUI.

Run from the project root with:
    .venv/Scripts/python src/gui.py

The image picker scans ``dataset/raw`` recursively. Selecting an image previews
it immediately. The user can then run one detector for a focused test, or run
all registered detectors for a complete inspection.
"""
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from preprocessing import preprocess
from defect_detection import DETECTORS
from pipeline import read_image, process_image_array


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "raw"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

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
    "Hole",
    "Open Tear",
    "Stain",
    "Finger Not Enough",
    "Fused Fingers",
)

DEFECT_OPTIONS = QUESTION_PAPER_DEFECTS + ADDITIONAL_GROUP_DEFECTS

DETECTOR_LABELS = {
    "detect_holes": "Hole",
    "detect_open_tears": "Open Tear",
    "detect_stains": "Stain",
    "detect_finger_not_enough": "Finger Not Enough",
    "detect_missing_finger": "Finger Not Enough",  # legacy function-name alias
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
    """Show a short ``defect / material / filename`` path in the dropdown."""
    try:
        return path.relative_to(dataset_dir).as_posix()
    except ValueError:
        return path.name


class GloveDefectApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Glove Defect Detection System (GDD)")
        self.root.minsize(980, 720)

        self.img_source = None
        self.img_plain = None
        self.selected_image_path = None
        self.image_paths = {}
        self.detector_by_label = {_label_for(det): det for det in DETECTORS}

        self.image_var = tk.StringVar(value=IMAGE_PLACEHOLDER)
        self.detector_var = tk.StringVar(value=DETECTOR_PLACEHOLDER)
        self.image_count_var = tk.StringVar()
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

        self.run_button = ttk.Button(
            controls,
            text="Run Detection",
            command=self.detect,
            state=tk.DISABLED,
        )
        self.run_button.grid(row=1, column=2, sticky="ew", pady=(3, 0))

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

        summary = ttk.LabelFrame(container, text="Detection Summary", padding=8)
        summary.pack(fill=tk.X, pady=(10, 0))

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
            self.img_source = None
            self.img_plain = None
            self._clear_panel(self.panel_left, "Select a dataset image above")
            self._clear_panel(self.panel_right, "Run detection to view the result")
            self._show_ready_message()
        self._update_run_button()

    def on_image_selected(self, _event=None):
        selected = self.image_var.get()
        path = self.image_paths.get(selected)
        if path is None:
            self._update_run_button()
            return

        img = read_image(path)
        if img is None:
            self.selected_image_path = None
            self.img_source = None
            self.img_plain = None
            self._update_run_button()
            messagebox.showerror("Image Error", f"Cannot read this image:\n{path}")
            return

        self.selected_image_path = path
        self.img_source = img
        _, self.img_plain = preprocess(img)
        self.show_on_panel(self.panel_left, self.img_plain)
        self._clear_panel(self.panel_right, "Run detection to view the result")
        self._show_ready_message()
        self._update_run_button()

    def on_detector_selected(self, _event=None):
        if self.img_plain is not None:
            self._clear_panel(self.panel_right, "Run detection to view the result")
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

    def say(self, text):
        self.result_box.configure(state=tk.NORMAL)
        self.result_box.delete("1.0", tk.END)
        self.result_box.insert(tk.END, text + "\n")
        self.result_box.configure(state=tk.DISABLED)

    def detect(self):
        if self.img_source is None or self.selected_image_path is None:
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
        material = Path(self.selected_image_path).parent.name
        result = process_image_array(
            self.img_source,
            material=material,
            detectors=detectors,
        )
        ratio = result["features"]["glove_area_ratio"]

        if not result["glove_found"]:
            self.show_on_panel(self.panel_right, self.img_plain)
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
        self.show_on_panel(self.panel_right, result["result_image"])

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
                "",
            ]
        )

        if defects:
            for index, (name, (x, y, w, h)) in enumerate(defects, 1):
                lines.append(
                    f"{index}. {name} - x={x}, y={y}, width={w}, height={h}"
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

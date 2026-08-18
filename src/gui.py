# -*- coding: utf-8 -*-
"""
GUI entry point -- run this file directly to start the system:
    .venv\\Scripts\\python src\\gui.py

Layout: original image on the left, annotated result on the right (red
boxes mark defects), and a text list underneath listing the detected
defect types (the assignment requires the GUI to display defect types).
"""
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from preprocessing import preprocess
from segmentation import segment_glove, glove_found, get_background_color
from defect_detection import run_all_detectors, draw_results

PANEL_W = 460  # width of each image panel, in pixels


class GloveDefectApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Glove Defect Detection System (GDD)")
        self.img_norm = None   # illumination-normalised image (segmentation/colour detectors)
        self.img_plain = None  # un-normalised image (texture detectors)

        # ---- top button bar ----
        bar = tk.Frame(root)
        bar.pack(pady=8)
        tk.Button(bar, text="Open Image", width=14,
                  command=self.open_image).pack(side=tk.LEFT, padx=6)
        tk.Button(bar, text="Detect Defects", width=14,
                  command=self.detect).pack(side=tk.LEFT, padx=6)

        # ---- two image panels in the middle ----
        panels = tk.Frame(root)
        panels.pack(padx=8)
        self.panel_left = tk.Label(panels, text="Original", width=60,
                                   height=20, relief=tk.SUNKEN)
        self.panel_left.pack(side=tk.LEFT, padx=4, pady=4)
        self.panel_right = tk.Label(panels, text="Result", width=60,
                                    height=20, relief=tk.SUNKEN)
        self.panel_right.pack(side=tk.LEFT, padx=4, pady=4)

        # ---- results text box at the bottom ----
        self.result_box = tk.Text(root, height=6, font=("Segoe UI", 11))
        self.result_box.pack(fill=tk.X, padx=8, pady=8)
        self.say("Open a glove image, then click 'Detect Defects'.")

    # ---------- helper: write text into the result box ----------
    def say(self, text):
        self.result_box.delete("1.0", tk.END)
        self.result_box.insert(tk.END, text + "\n")

    # ---------- helper: display an OpenCV image on a Tkinter panel ----------
    def show_on_panel(self, panel, img_bgr):
        h, w = img_bgr.shape[:2]
        scale = PANEL_W / w
        img_small = cv2.resize(img_bgr, (PANEL_W, int(h * scale)))
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        panel.config(image=photo, width=PANEL_W, height=int(h * scale))
        panel.image = photo  # keep a reference so the image isn't garbage-collected

    # ---------- button: open an image ----------
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select a glove image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if not path:
            return
        # imdecode is used here so that non-ASCII file paths are supported
        img = cv2.imdecode(np.fromfile(path, dtype="uint8"), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Cannot read this image file.")
            return
        self.img_norm, self.img_plain = preprocess(img)
        self.show_on_panel(self.panel_left, self.img_plain)
        self.say("Image loaded. Click 'Detect Defects' to analyse.")

    # ---------- button: detect defects ----------
    def detect(self):
        if self.img_norm is None:
            messagebox.showwarning("Notice", "Please open an image first.")
            return
        # Safety net: any unexpected exception becomes an on-screen message
        # instead of the button silently doing nothing
        try:
            self._detect()
        except Exception as exc:
            self.say(f"Unexpected error while processing this image:\n"
                     f"  {type(exc).__name__}: {exc}")

    def _detect(self):
        mask_filled, mask_raw = segment_glove(self.img_norm)   # 1. segment the glove
        ok, ratio = glove_found(mask_filled)

        # Segmentation sanity check: if no glove was found, this must be
        # stated explicitly -- "no defects found" must never be reported as
        # "passed" (otherwise any random background photo would show PASSED)
        if not ok:
            self.show_on_panel(self.panel_right, self.img_plain)
            self.say("No glove detected in this image "
                     f"(glove area = {ratio:.1%} of the frame).\n"
                     "Please use a photo with the glove centred on a plain "
                     "background.")
            return

        bg_color = get_background_color(self.img_norm)
        defects, errors = run_all_detectors(               # 2. find defects
            self.img_norm, mask_filled, mask_raw, bg_color)
        result_img = draw_results(self.img_plain, defects)   # 3. draw the results
        self.show_on_panel(self.panel_right, result_img)

        # 4. list the defect types as text (hard requirement of the assignment)
        if defects:
            lines = [f"Detected {len(defects)} defect(s):"]
            for i, (name, (x, y, w, h)) in enumerate(defects, 1):
                lines.append(f"  {i}. {name}   at x={x}, y={y}, size {w}x{h}")
        else:
            lines = ["No defects detected - glove PASSED inspection."]

        # 5. detector errors must be surfaced explicitly: otherwise "nothing
        #    found" and "a detector crashed" look identical on screen, and a
        #    fault gets mistaken for a clean glove
        if errors:
            lines.append("")
            lines.append(f"WARNING - {len(errors)} detector(s) failed "
                         f"(result may be incomplete):")
            lines.extend(f"  ! {e}" for e in errors)

        self.say("\n".join(lines))


if __name__ == "__main__":
    root = tk.Tk()
    app = GloveDefectApp(root)
    root.mainloop()

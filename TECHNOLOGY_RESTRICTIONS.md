# Technology Use Restrictions (Do / Do Not)

## Purpose and authority

This project must remain an explainable **classical image-processing** solution for glove defect detection. These rules are based on the assignment technical guide and the technologies currently used in this repository.

If requirements conflict, follow this order:

1. The official assignment brief and the lecturer's written approval.
2. These project restrictions and the current shared interfaces.
3. Individual implementation preferences.

Any lecturer-approved exception must be recorded in the report and this file before it is merged.

## Approved project stack

| Technology | Approved purpose | Restriction |
|---|---|---|
| Python 3 | Application, pipeline, evaluation, and tests | Keep the project runnable in the documented virtual environment. |
| OpenCV (`opencv-python`) | Image input, filtering, colour conversion, thresholding, morphology, connected components, contours, and drawing results | Use explainable classical operations only. |
| NumPy | Image arrays, masks, colour-distance calculations, and numerical features | Do not use it to hide an unexplained model or opaque decision process. |
| Tkinter / `ttk` | Desktop GUI | Keep image-processing logic outside GUI callbacks. |
| Pillow | Converting processed images for display in Tkinter | Do not use it as a second, conflicting processing pipeline. |
| Python standard library | Paths, CLI arguments, CSV output, collections, and `unittest` | Prefer the standard library when it avoids another dependency. |

The only external runtime packages currently listed in `requirements.txt` are `opencv-python`, `numpy`, and `pillow`. Adding another package, framework, service, or programming language requires team agreement and lecturer approval first.

## Do

### Image-processing methods

- Use classical, explainable stages: acquisition, preprocessing, glove segmentation, mask cleanup, defect candidate extraction, feature extraction, rule-based recognition, and output.
- Use methods supported by the course and project, including median filtering, CLAHE when justified, global/Otsu/adaptive thresholding, colour or intensity segmentation, morphology, connected components, contours, convexity analysis, and region filling where appropriate.
- Use measurable features such as area, perimeter, component or hole count, bounding box, aspect ratio, compactness, centroid, normalised position, and colour/intensity statistics.
- Keep every detection decision traceable to named features and documented thresholds.
- Preserve image aspect ratio when resizing. Use padding if a fixed canvas is required; do not stretch the glove.
- State colour spaces accurately. OpenCV reads colour images as BGR; convert explicitly before RGB, Lab, HSV, or YCrCb calculations.
- Treat HSV and HSI as different colour models. If the report or lecturer specifically requires HSI, implement the taught HSI formula or obtain approval to use HSV instead.
- Keep both forms of the glove mask when needed: the raw mask retains holes, while the filled mask represents the complete glove outline.
- Normalise scale-sensitive measurements, for example `defect_area / glove_area` and positions relative to glove width and height.
- Derive thresholds from development images, record the selected values and conditions, then freeze them before final testing.

### Code and architecture

- Keep the shared pipeline in `src/pipeline.py`. The GUI and batch evaluator must call `process_image` or `process_image_array` so they use the same algorithm.
- Keep preprocessing, segmentation, detection, GUI, and evaluation responsibilities separated by module.
- Keep algorithm code out of Tkinter button handlers. GUI code should collect input, call the pipeline, and display the returned result.
- Follow the detector contract used by `run_all_detectors`: accept the shared image/mask/background inputs and supported keyword inputs, and return a list of `(label, (x, y, width, height))` results. Return an empty list when no target defect is found.
- Register a completed detector in `DETECTORS` and add its exact output label to `LABEL_MAP` in `src/evaluate.py`.
- Put tunable values in clearly named constants or a documented configuration structure. Include units or valid ranges where they are not obvious.
- Use relative paths or project configuration. Handle unreadable images, empty masks, missing contours, unknown materials, segmentation failure, and detector exceptions with clear status messages.
- Preserve the safe outcome: a failed or uncertain segmentation must never be reported as a passed glove.

### Dataset and testing

- Preserve original photographs. Write masks, annotated images, failures, and CSV results to generated-output locations instead of overwriting raw data.
- Follow the directory structure used by the active evaluator: `dataset/raw/<defect>/<material>/<image>`; use `good` as the defect folder for clean gloves. Multiple expected defects may be joined with `+`.
- Keep folder names and `LABEL_MAP` synchronized because the evaluator uses the defect folder as ground truth.
- Use self-collected images covering at least three materials and at least five frozen final-test images per assigned defect, as required by the assignment.
- Keep development images used for threshold selection separate from final-test images. Do not tune parameters on the final test set.
- Include clean gloves, different lighting/background conditions, successful cases, boundary cases, and failed cases. Record whether each failure occurred during acquisition, segmentation, feature extraction, or recognition.
- Run the relevant checks after algorithm changes:

```powershell
.venv\Scripts\python -m compileall -q src
.venv\Scripts\python -m unittest discover -s tests
.venv\Scripts\python src\selftest.py
.venv\Scripts\python src\evaluate.py
```

- Treat detector crashes as software defects, not accuracy results. Fix or separately report them before using the affected images in performance calculations.

## Do not

### Prohibited methods and services

- **Do not use Haar Cascade**, including `cv2.CascadeClassifier` or downloaded cascade XML models.
- **Do not use TensorFlow, Keras, PyTorch, YOLO, CNNs, or other deep-learning frameworks/models.**
- **Do not use pattern matching or template matching**, including `cv2.matchTemplate`, correlation against a normal-glove template, or image alignment followed by direct template comparison.
- Do not use pretrained models, online/cloud detection APIs, or external black-box detectors disguised as part of the local pipeline.
- Do not choose a method only because it improves accuracy. Every method must be explainable from the taught image-processing concepts and supported by evidence from this dataset.

### Implementation restrictions

- Do not add an external library, a web framework, a database, another GUI toolkit, or another programming language without approval.
- Do not duplicate the processing pipeline inside `gui.py`, `evaluate.py`, or a member-specific script.
- Do not silently change the shared result dictionary, detector signature, label spelling, dataset layout, or detector registration order.
- Do not scatter unexplained numeric thresholds throughout functions or copy threshold values from examples without measuring this dataset.
- Do not apply every available filter or enhancement by default. Each operation must address an observed noise, lighting, segmentation, or feature problem.
- Do not overuse closing or region filling before hole/tear detection; it can erase the defect evidence the detector needs.
- Do not calculate glove defect colour statistics from the background or outside the glove region of interest.
- Do not hard-code a member's Desktop, Downloads, drive letter, or virtual-environment path in source code.
- Do not overwrite original images or commit `.venv`, caches, generated evaluation files, or unnecessary full-resolution phone images.
- Do not report `PASSED`, `No defect`, or an accuracy score when the image failed to load, no plausible glove was segmented, or a required detector crashed.
- Do not remove difficult or failed examples to make the results look better; they are required evidence for critical analysis.

## Ask the lecturer before using

The following are not automatically approved. Use them only after explicit lecturer confirmation and document the approval:

- A full Fourier/frequency-domain processing pipeline.
- Saliency detection as the main defect detector.
- Full GLCM calculations or other advanced/extra-only texture methods.
- A trained statistical classifier, Bayes decision method, or other learned decision model.
- Advanced detectors not covered by the normal course material.
- Any new external package, language, pretrained component, or major replacement of the approved stack.

Approval cannot override the assignment's explicit bans on Haar Cascade, TensorFlow, and pattern/template matching unless the official assignment requirement itself is formally changed.

## Merge checklist

Before merging a technology or algorithm change, confirm:

- [ ] It uses only the approved stack, or written approval is documented.
- [ ] It contains no Haar Cascade, TensorFlow/deep learning, or pattern/template matching.
- [ ] The method and thresholds are explainable and supported by development evidence.
- [ ] The GUI and evaluator still use the shared pipeline.
- [ ] Dataset folders, detector labels, `DETECTORS`, and `LABEL_MAP` agree.
- [ ] Load, segmentation, and detector failures cannot produce a false pass.
- [ ] Compilation, unit tests, self-test, and batch evaluation have been run as applicable.
- [ ] Raw images remain unchanged and final-test images were not used for tuning.

# How my three detectors sit in the team system

My part is **incomplete beading**, **damage by fold** and **improper roll**. This
describes exactly where that code lives, what it touches, and why it is arranged this
way, so anyone can check it without reading the diff.

---

## 1. The shape of it

My detectors live in their own package, `src/tishen/`, which imports **nothing** from
`segmentation.py`, `pipeline.py` or `defect_detection.py`. It carries its own copy of the
glove segmentation.

The reason is separation of results, not tidiness. `skin_mask()` feeds `segment_glove()`,
`background_key()` and `_border_pixels()`, so every detector in the system depends on it
through the shared pipeline. A detector is only as good as the mask it reasons over, and
the shared mask is tuned for the team's structural defects -- tuning it for beading would
move everyone's numbers. Keeping a separate copy means:

* nothing tuned in `src/tishen/` can change a teammate's result
* nothing tuned in `segmentation.py` can change mine

The two segmentations produce identical masks today, because mine began as a copy. The
point is not that they differ now; it is that they are free to diverge without collateral
damage.

| file | what it is |
|---|---|
| `src/tishen/segmentation.py` | our glove segmentation |
| `src/tishen/detection.py` | the three detectors, their helpers, and the drawing code |
| `src/tishen/__init__.py` | exports `DETECTORS` and `Detection` |

---

## 2. One GUI, one detector list

The system keeps a single GUI. The three detectors take the team's standard signature

```python
detect_x(img, mask_filled, mask_raw, bg_color, img_plain=None, material=None)
```

so they can be registered in the one shared `DETECTORS` list and appear in the dropdown
like any other detector.

**They ignore the glove mask that signature hands them.** Each one calls `our_masks(img)`
and re-segments the image with our segmentation. Segmentation is the expensive step and
all three want the same answer, so it is computed once per image and cached.

There is precedent for this in the team's own pipeline: `detect_stains` and
`detect_plastic_contamination` already fall back to
`_owned_colour_detector_segmentation()` when the shared mask does not suit them.

---

## 3. The only change to a file I did not write

**`src/defect_detection.py`, ~56 lines appended at the very end. Nothing above is
modified.** The block does three things:

1. **Drops the stale entries.** An earlier version of these three still sits further up
   that file, from before they moved into a package. They are removed from `DETECTORS`
   by name so the GUI lists each defect exactly once. The dead functions are left in
   place rather than deleted.

2. **Appends the package versions last.** `deduplicate()` keeps whichever detector is
   registered first when two boxes overlap by more than IoU 0.5. Being last means these
   three can never displace a teammate's detection. This is deliberate -- registered
   first, beading once deleted a Finger Not Enough detection, because a patch of bare
   skin at the cuff and a missing finger produced boxes of `(306,314,225,58)` and
   `(313,321,234,44)`.

3. **Adapts the return type.** `run_all_detectors()` keeps a pixel mask only if
   `isinstance(item, Detection)` passes, against *its own* `Detection` class. The tishen
   package defines its own -- same fields, different class -- so that check failed, the
   mask was dropped, and `build_defect_masks()` returned an empty region for defect names
   it does not know. `draw_results()` skips shading when the mask is empty, so the region
   highlight vanished and only a thin box was drawn. `_adapt_tishen()` re-wraps each
   detection at the boundary, which fixes it without the package importing anything from
   that module.

If `src/tishen/` is absent the import fails quietly and the team system runs exactly as
before.

---

## 4. Dataset

My photographs sit under `dataset/raw/<defect>/<material>/`, which is the layout the
team's `collect_images()` reads and the spelling their `LABEL_MAP` recognises.

| folder | images |
|---|---|
| `incomplete_beading/cotton` | 9 |
| `incomplete_beading/latex` | 2 |
| `incomplete_beading/nitrile` | 7 |
| `damage_by_fold/cotton` | 4 |
| `damage_by_fold/latex_foam` | 3 |
| `improper_roll/cotton` | 3 |
| `improper_roll/nitrile` | 5 |

Only the folder structure and `.gitkeep` are committed -- `.gitignore` keeps raw
photographs out of the repo, and that is respected.

Two notes for whoever checks this:

* **The seven `cotton/side_tear` photographs are now `incomplete_beading/cotton`.** Every
  one shows a tear at the cuff hem beside the bead, which is the same defect. Their
  `LABEL_MAP` maps `side_tear` to **Open Tear**, so those seven no longer count as Open
  Tear positives -- whoever owns tearing will see their image count drop by seven.
* Materials were checked against the team's own samples. Their `latex_foam` is the blue
  diamond-embossed coated glove, so fold images F5-F7 are `latex_foam` and F1-F4 are
  knitted `cotton`. The two white translucent gloves are none of the team's three
  materials and keep a `latex` folder; `infer_material()` does not recognise that name,
  so they report no material, which is the honest answer.

---

## 5. Verification

Run all of these. The first two are the team's and must not move.

| check | expected |
|---|---|
| `python src/selftest.py` | 41 PASS |
| `python -m pytest tests -q` | 40 passed, 13 subtests |
| beading / fold / roll on their own folders | 18/18, 7/7, 8/8 |

And the two that actually catch integration damage, run before and after any change:

* **the shared glove mask must be identical on every image** -- if it moves,
  `segmentation.py` was touched
* **no detection belonging to a teammate may disappear** -- if one does, it is a
  de-duplication collision like the one in section 3

---

## 6. A bug I did not fix, because it is not my code

In `draw_results()` the filled rectangle behind a label is clamped to the image edge but
the text is not, so a defect near the right margin loses its name in every saved
screenshot. It affects only the annotated picture, never a detection or a number. The fix
is three lines: compute `label_x = max(0, min(x, width - 1 - text_w - 2 * pad))` and use
it for both the rectangle and the text.

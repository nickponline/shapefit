# shapefit

**Fit clean polygons to irregular shapes with a neural network.**

📖 **[Read the illustrated explainer](https://claude.ai/artifact/TQyf1TVKUrR8SDPAA9RXfq)**: how it works step by step, with diagrams and a glossary (also in [`docs/explainer.html`](docs/explainer.html)).

Many shapes must be polygons. Examples are building outlines from aerial
photos, rooms in scanned floor plans, and parts in drawings. But the data
often shows these shapes with bad edges: bumps, jagged steps, gaps, holes and
rounded corners.

The usual tool is Douglas–Peucker simplification (`cv2.approxPolyDP`). It can
only remove points from the outline. It cannot tell a bump from a corner. Thus
it keeps the bumps, or it cuts off the real corners.

shapefit uses a different method. It is easy to make a clean polygon and then
damage it. This gives an unlimited number of training pairs: a damaged shape
and its correct clean polygon. A neural network learns from these pairs to
remove the damage.

```python
from shapefit import ShapeFitter

fitter = ShapeFitter("shapefit/weights/shapefit.pt")
polygon = fitter.fit(mask)          # (N, 2) array of (x, y) corners
```

![pipeline](docs/images/pipeline.png)

> [!IMPORTANT]
> **Status.** The data generator, network, vectoriser, API and training script
> are complete and tested. The full training run is not done yet. The only
> checkpoint in the package is `shapefit/weights/overfit_demo.pt`. We trained
> it on only 48 shapes, to show that all parts work together. It learned those
> 48 shapes by memory (this is **overfitting**). It does *not* work on new
> shapes. The figures and the first table below use those 48 shapes. To make a
> general model, do the full training run (see [Training](#training)).

---

## Contents

- [Terms](#terms)
- [How it works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Results (overfit demo)](#results-overfit-demo)
- [Training data](#training-data)
- [Training](#training)
- [Library layout](#library-layout)
- [Limits and next steps](#limits-and-next-steps)

## Terms

Each term has one meaning in this document. The
[explainer](docs/explainer.html#glossary) has a full glossary.

| Term | Meaning |
|---|---|
| **mask** | An image where each pixel is 1 (inside the shape) or 0 (outside the shape). |
| **corner** | A point where two edges of a polygon meet. The code calls it a *vertex*. |
| **canvas** | The fixed 128 × 128 image that the network always uses. |
| **heatmap** | An image where a high pixel value means "a corner is probably here". |
| **contour** | The outer edge of a mask, as a closed chain of points in order. |
| **vectoriser** | The part of shapefit that changes the network output images into a list of corners. |
| **checkpoint** | A file that contains the trained network weights. |
| **IoU** | Overlap area of two shapes divided by the area that either shape covers. 1.0 means identical. |
| **overfitting** | A network learns its training examples by memory and does not work on new examples. |

## How it works

```
 mask of any size,       canvas                    U-Net (7.8 M weights)       vectoriser                  polygon
 or noisy point list ──▶ 128×128, box at 72 % ──▶ ├─ clean mask      ──▶ contour (4× larger) ───────┐
                         (frame.py)               └─ corner heatmap  ──▶ peaks → put on contour ────┤
                                                                           sort by position          │
                                                                           fit one line per edge ◀───┘
                                                                           intersect the lines
                                                                           (optional angle rule) ──▶ move back to input coordinates
```

1. **Framing.** shapefit finds the bounding box of the input. The input can be
   a mask of any size, an `(N, 2)` list of points, or a shapely `Polygon`.
   shapefit scales the shape so that the longest side of the box is 72 % of a
   128 × 128 canvas. It ignores very small dots. The training data uses the
   same framing.
2. **The network.** A U-Net looks at the canvas. It makes two images:
   - a **clean mask**, which shows where the true edge is;
   - a **corner heatmap**, which shows where the corners are.
3. **The vectoriser.** This step uses geometry, not learning.
   1. Find the contour of the clean mask.
   2. Find the peaks of the heatmap. Move each peak onto the contour.
   3. Sort the corners by their position along the contour. This gives the
      correct order around the shape.
   4. For each edge, fit a straight line through the contour points between its
      two corners. Ignore 15 % of the points at each end.
   5. Each corner is the point where the lines of its two edges cross. The
      result is a sharp corner, with an accuracy better than one pixel.
4. **Regularisation (optional).** `regularize="orthogonal"` makes all corners
   90°. Use it for buildings and rooms. `regularize="parallel"` makes edges that
   are almost parallel exactly parallel.

**Why two images and not a list of corners?** A list has a different length
for each shape, and it has no fixed start point. These facts make a list
difficult to learn. Two images are easier to learn. The sort in step 3 then
gives the order.

**Upper limit.** We gave the vectoriser perfect network outputs. It then gets
**IoU 0.994** and the **correct number of corners on 100 %** of shapes
(`tests/test_vectorize.py`). Thus nearly all remaining error comes from the
network.

## Installation

```bash
git clone <this repo> shape-fit && cd shape-fit
pip install -e ".[dev]"      # numpy, torch, opencv-python-headless, shapely, scipy (+ pytest, matplotlib)
pytest                       # 25 tests, approximately 10 s on CPU
```

## Usage

```python
import numpy as np
from shapefit import ShapeFitter, fit_polygon, iou

fitter = ShapeFitter("path/to/checkpoint.pt", device="cpu")   # or "cuda" / "mps"

# 1. a mask of any size (uint8, bool, or float from 0 to 1; RGB and RGBA are also accepted)
poly = fitter.fit(mask)                       # -> (N, 2) float array, (x, y) = (column, row)

# 2. a noisy outline as a list of points, or a shapely Polygon
poly = fitter.fit(noisy_points)               # (M, 2) array, any units
poly = fitter.fit(shapely_polygon)

# 3. regularisation for shapes made by people
poly = fitter.fit(mask, regularize="orthogonal")   # all corners 90°
poly = fitter.fit(mask, regularize="parallel")     # almost-parallel edges become parallel

# 4. test-time augmentation: the mean of 8 rotated and flipped copies (slower)
poly = fitter.fit(mask, tta=True)

# 5. the intermediate outputs, to find problems or to make figures
r = fitter.fit_detailed(mask)
r.canvas, r.clean_mask, r.vertex_heatmap, r.canvas_polygon, r.frame

# short form: loads shapefit/weights/shapefit.pt one time and keeps it
poly = fit_polygon(mask)
```

The output corners are in the input pixel coordinates. Pixel centres are at
whole numbers. The corners go counter-clockwise as you see them on the screen
(x to the right, y down).

You can adjust two values in `fit(...)`:

- `peak_threshold` (default 0.25). Make it lower to find weak corners.
- `min_turn_deg` (default 8°). shapefit removes a corner if the direction
  changes by less than this angle.

[`examples/quickstart.py`](examples/quickstart.py) is a complete example:

```
$ python examples/quickstart.py
4 vertices:
[[184.8  55.5]
 [ 56.1 233.5]
 [190.4 330.5]
 [319.4 153.1]]
IoU vs ground truth: 0.9967
orthogonal fit: 8 vertices, IoU 0.9974
canvas (128, 128) | input->canvas scale 0.333
```

## Results (overfit demo)

We trained the demo checkpoint on 48 shapes: `scene_set(48, seed=7)`. Each
shape is in a 480 × 360 mask, at a random position and size. The shapes go
through the same public `fit()` function that you use.

In the figure, the green line is shapefit. The orange dashed line is
`cv2.approxPolyDP` with ε = 2 % of the perimeter, which is a usual setting.

![results](docs/images/results.png)

Regularisation on one of the 48 shapes:

![regularize](docs/images/regularize.png)

The table compares each method with the true polygon. It uses five metrics:

- **IoU**: the overlap of the output and the true polygon.
- **Vertex count exact**: the percent of shapes with the correct number of corners.
- **Vertex F1**: a score for missed and extra corners. A corner is correct if
  it is within 3 % of the shape size from a true corner.
- **Vertex error**: the median distance from each corner to its true corner,
  as a percent of the shape size.
- **Hausdorff**: the largest distance between the two outlines, as a percent
  of the shape size.

**The 48 memorised shapes:**

| method | IoU ↑ | vertex count exact ↑ | vertex F1 @3% ↑ | vertex err ↓ | Hausdorff ↓ |
|---|---|---|---|---|---|
| shapefit (NN) | 0.994 | 100.0% | 0.997 | 0.19% | 0.29% |
| shapefit (NN) + TTA | 0.942 | 8.3% | 0.559 | 2.19% | 3.45% |
| approxPolyDP 1% | 0.924 | 29.2% | 0.558 | 2.25% | 4.91% |
| approxPolyDP 2% | 0.923 | 64.6% | 0.641 | 2.43% | 4.31% |
| approxPolyDP 3% | 0.903 | 64.6% | 0.648 | 2.43% | 5.59% |
| convex hull + DP | 0.906 | 25.0% | 0.582 | 2.50% | 5.04% |
| min-area rectangle | 0.747 | 43.8% | 0.216 | 11.71% | 20.53% |

The TTA row is worse, and this is expected. TTA shows the network 8 rotated
and flipped copies of each shape. The overfit network learned each shape in
one orientation only. Thus 7 of the 8 copies are new to it. TTA is for a model
after full training. A test (`tests/test_api.py`) confirms that TTA reverses
each rotation and flip correctly.

The overfit run gets IoU 0.994 and 100 % correct corner counts at its first
check (step 500). This takes approximately 2 minutes on an M3 Max. This result
shows that the data, the targets, the framing and the vectoriser agree.

**200 new shapes, not in the training set** (`scene_set(200, seed=999)`):

| method | IoU ↑ | vertex count exact ↑ | vertex F1 @3% ↑ | vertex err ↓ | Hausdorff ↓ |
|---|---|---|---|---|---|
| shapefit (NN) | 0.923 | 0.5% | 0.344 | 3.07% | 5.30% |
| shapefit (NN) + TTA | 0.932 | 3.5% | 0.452 | 2.83% | 4.51% |
| approxPolyDP 1% | 0.911 | 29.0% | 0.552 | 2.66% | 5.52% |
| approxPolyDP 2% | 0.900 | 46.0% | 0.608 | 2.48% | 5.67% |
| approxPolyDP 3% | 0.875 | 47.0% | 0.595 | 2.65% | 7.79% |
| convex hull + DP | 0.888 | 29.0% | 0.583 | 2.82% | 5.78% |
| min-area rectangle | 0.718 | 36.0% | 0.193 | 15.07% | 23.60% |

The overfit model does not work on new shapes. Its IoU is still better than
the other methods. But it does not find the corners correctly: the number of
corners is correct for only 0.5 % of shapes. The full training run uses
millions of new shapes, not 48. It is necessary to correct this.

> [!CAUTION]
> Do not use `overfit_demo.pt` on your own data. Do the full training run first.

## Training data

[docs/DATA.md](docs/DATA.md) gives all the details. In short:

* **6 types of polygon:** regular polygons, rotated rectangles, quads
  (trapezoids, parallelograms, kites), random convex shapes, star shapes, and
  L, T, U and + shapes that look like buildings. Each corner must turn by 12° or
  more. Each edge must be 6 % of the shape size or longer. Very thin shapes are
  rejected.
* **15 types of damage.** Geometric damage moves the outline in or out: jagged
  edges, bumps and dents, wobble, steps and rounded corners. Pixel damage
  changes the mask directly: bites, gaps, blobs, cuts, holes, speckle, loose
  dots, pixelation, blur and morphology. A random value called *severity*
  controls how strong the damage is.

![corruptions](docs/images/corruptions.png)

Training pairs. Row 1 is the input. Row 2 is the clean target with the true
polygon. Row 3 is the target heatmap.

![dataset](docs/images/dataset.png)

## Training

```bash
# the full run, with new data for each step (recommended)
python -m shapefit.train --steps 30000 --batch 64 --out runs/base
cp runs/base/best.pt shapefit/weights/shapefit.pt

# or: write a fixed dataset to disk first (approximately 80 MB for 200,000 shapes)
python -m shapefit.make_dataset --n 200000 --out data/train
python -m shapefit.train --data data/train --steps 30000 --out runs/shards

# make the demo checkpoint again (approximately 10 min on an M3 Max), then make the file smaller (fp16)
python -m shapefit.train --overfit 48 --batch 32 --steps 2500 --eval-every 500 --out runs/overfit
python examples/export_weights.py runs/overfit/best.pt shapefit/weights/overfit_demo.pt

# make all figures and tables in this README again
python examples/make_readme_figures.py --weights shapefit/weights/overfit_demo.pt
```

The training script uses a CUDA GPU if there is one. If not, it uses the
Apple GPU (MPS). If neither is available, it uses the CPU.

Every 2,000 steps, the script measures the five metrics on 500 test shapes. It
writes them to `runs/*/log.jsonl`. It keeps the best checkpoint (`best.pt`) and
the last checkpoint (`last.pt`).

We measured the time of the full run on an Apple M3 Max:

| Item | Time |
|---|---|
| One training step (64 shapes) | 386 ms |
| Of that, the wait for data | 0.4 ms |
| One check on 500 test shapes | 1.6 s |
| **30,000 steps** | **approximately 3.2 h** |

The GPU sets the speed. More data workers do not make training faster. A CUDA
GPU is several times faster.

> [!CAUTION]
> The training script cannot continue a stopped run yet. It saves `last.pt`
> every 2,000 steps, but after a crash you must start again.

## Library layout

| module | contents |
|---|---|
| `shapefit/shapes.py` | the polygon types and the rules for a valid polygon |
| `shapefit/degrade.py` | the damage types (`GEOMETRIC` and `RASTER`) |
| `shapefit/data.py` | drawing, heatmaps, `make_sample`, the `SyntheticShapes` data stream |
| `shapefit/scenes.py` | full-size test images that use the public API |
| `shapefit/make_dataset.py` | writes a fixed dataset to disk; `ShardDataset` reads it |
| `shapefit/frame.py` | moves any input to the 128 × 128 canvas and back |
| `shapefit/model.py` | the `ShapeFitNet` U-Net and its loss functions |
| `shapefit/vectorize.py` | changes network outputs into a polygon; regularisation |
| `shapefit/fit.py` | `ShapeFitter`, `fit_polygon`, `FitResult` |
| `shapefit/metrics.py` | IoU, Hausdorff distance, corner matching |
| `shapefit/baselines.py` | Douglas–Peucker, convex hull, minimum-area rectangle |
| `shapefit/train.py` | the training command (data stream, dataset on disk, or overfit) |

## Limits and next steps

* **No general checkpoint yet.** Do the full training run (see above). The
  overfit run and the vectoriser tests already confirm the data, the losses and
  the vectoriser.
* **One outer edge for each call.** shapefit does not make polygons with
  holes. It ignores holes in the input, as noise. For many shapes in one mask,
  split the mask into connected components. Then call `fit` for each one.
* **Resolution.** The network finds corners on a 128-pixel canvas. The line
  fit then gives an accuracy better than one pixel. Very small details on a
  very large shape can disappear. A 256-pixel version is a small change
  (`ShapeFitNet` and `SIZE`).
* **Curves are not polygons.** A circle or an ellipse gets many corners, or
  corners that are not clear.
* **Possible next work:** an output for the number of corners and for
  confidence, edges that are arcs, polygons with holes, and more training on
  real data (for example, building outlines).

## License

MIT

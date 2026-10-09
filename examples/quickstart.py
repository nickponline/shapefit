"""Fit clean polygons to irregular masks with the demo checkpoint.

    python examples/quickstart.py

The demo checkpoint is *overfit* on 48 scenes (seed 7), so we use those scenes
here. For your own data, train a general model first (see README).
"""

import numpy as np

from shapefit import ShapeFitter, iou
from shapefit.scenes import scene_set

fitter = ShapeFitter("shapefit/weights/overfit_demo.pt")
scenes = scene_set(48, seed=7)  # the exact scenes the demo model memorised

scene = scenes[1]
mask = scene.image  # (360, 480) uint8 binary mask: bumps, bites, speckle, ...

polygon = fitter.fit(mask)  # (N, 2) array of (x, y) vertices, image coordinates
print(f"{len(polygon)} vertices:\n{np.round(polygon, 1)}")
print(f"IoU vs ground truth: {iou(polygon, scene.vertices):.4f}")

# also accepts a noisy vertex list or a shapely Polygon, and can square up edges
building = next(s for s in scenes if s.family == "rectilinear")
square = fitter.fit(building.image, regularize="orthogonal")
print(f"orthogonal fit: {len(square)} vertices, IoU {iou(square, building.vertices):.4f}")

# inspect the intermediate network outputs
r = fitter.fit_detailed(mask)
print("canvas", r.canvas.shape, "| input->canvas scale", round(r.frame.scale, 3))

"""Fit clean polygons to irregular masks with the trained checkpoint.

    python examples/quickstart.py

The scenes below come from a seed that was never used in training.
"""

import numpy as np

from shapefit import ShapeFitter, iou
from shapefit.scenes import scene_set

fitter = ShapeFitter("shapefit/weights/shapefit.pt")
scenes = scene_set(48, seed=7)  # unseen test scenes, 480 x 360 masks

scene = scenes[6]
mask = scene.image  # (360, 480) uint8 binary mask: bumps, bites, speckle, ...

polygon = fitter.fit(mask)  # (N, 2) array of (x, y) vertices, image coordinates
print(f"{len(polygon)} vertices:\n{np.round(polygon, 1)}")
print(f"IoU vs ground truth: {iou(polygon, scene.vertices):.4f}")

# a union of overlapping rectangles, squared up with orthogonal regularisation
union = scene_set(1, seed=5, family="rect_union")[0]
square = fitter.fit(union.image, regularize="orthogonal")
print(f"rectangle union: {len(union.vertices)} true corners, fitted {len(square)}, IoU {iou(square, union.vertices):.4f}")

# also accepts a noisy vertex list or a shapely Polygon; inspect the intermediate outputs
r = fitter.fit_detailed(mask)
print("canvas", r.canvas.shape, "| input->canvas scale", round(r.frame.scale, 3))

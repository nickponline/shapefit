"""shapefit: fit clean polygons to irregular shapes with a neural network."""

from .fit import FitResult, ShapeFitter, fit_polygon
from .metrics import iou, score

__version__ = "0.1.0"
__all__ = ["ShapeFitter", "FitResult", "fit_polygon", "iou", "score"]

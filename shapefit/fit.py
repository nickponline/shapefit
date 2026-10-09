"""High-level inference API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import SIZE
from .frame import Frame, frame_mask, frame_points
from .model import ShapeFitNet
from .vectorize import polygon_from_outputs

DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "shapefit.pt"


@dataclass
class FitResult:
    polygon: np.ndarray  # (N, 2) vertices in input coordinates, CCW on screen
    canvas: np.ndarray  # (S, S) network input
    clean_mask: np.ndarray  # (S, S) predicted clean-mask probability
    vertex_heatmap: np.ndarray  # (S, S) predicted vertex probability
    canvas_polygon: np.ndarray  # polygon in canvas coordinates
    frame: Frame


def _as_input(x):
    """Classify the input as ('mask', array) or ('points', array)."""
    if hasattr(x, "exterior"):  # shapely Polygon
        return "points", np.asarray(x.exterior.coords)[:-1]
    a = np.asarray(x)
    if a.ndim == 2 and a.shape[1] == 2 and a.shape[0] >= 3 and (a.dtype.kind == "f" or a.shape[0] < a.max()):
        return "points", a.astype(np.float64)
    if a.ndim == 3:  # colour / RGBA image: anything non-black is foreground
        a = a[..., :3].max(-1) if a.shape[-1] in (3, 4) else a[..., 0]
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D mask or an (N, 2) point array, got shape {a.shape}")
    return "mask", a


class ShapeFitter:
    """Wraps a trained ``ShapeFitNet`` and turns irregular shapes into clean polygons.

    >>> fitter = ShapeFitter("shapefit/weights/shapefit.pt")
    >>> poly = fitter.fit(mask)           # (N, 2) array of (x, y) vertices
    """

    def __init__(self, weights: str | Path | None = None, device: str = "cpu"):
        path = Path(weights) if weights else DEFAULT_WEIGHTS
        if not path.exists():
            raise FileNotFoundError(
                f"no weights at {path}. Train some with `python -m shapefit.train` "
                "and pass the checkpoint path."
            )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.device = torch.device(device)
        self.model = ShapeFitNet(**ckpt["config"]).to(self.device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.meta = {k: v for k, v in ckpt.items() if k not in ("state_dict",)}

    @torch.no_grad()
    def predict_canvas(self, canvases: np.ndarray, tta: bool = False) -> np.ndarray:
        """Raw logits ``(B, 2, S, S)`` for canonical canvases ``(B, S, S)``."""
        x = torch.from_numpy(np.asarray(canvases, np.float32)[:, None]).to(self.device)
        if not tta:
            return self.model(x).cpu().numpy()
        acc = 0
        for k in range(4):  # 8 dihedral transforms, averaged after undoing them
            for flip in (False, True):
                t = torch.rot90(x, k, (2, 3))
                t = t.flip(3) if flip else t
                y = self.model(t)
                y = y.flip(3) if flip else y
                acc = acc + torch.rot90(y, -k, (2, 3))
        return (acc / 8).cpu().numpy()

    def fit_detailed(self, shape, regularize: str | None = None, tta: bool = False, **kw) -> FitResult:
        kind, arr = _as_input(shape)
        canvas, fr = frame_points(arr) if kind == "points" else frame_mask(arr)
        logits = self.predict_canvas(canvas[None], tta=tta)[0]
        poly_c = polygon_from_outputs(logits[0], logits[1], regularize=regularize, **kw)
        sig = lambda z: 1 / (1 + np.exp(-z))
        return FitResult(
            polygon=fr.from_canvas(poly_c) if len(poly_c) else poly_c,
            canvas=canvas,
            clean_mask=sig(logits[0]),
            vertex_heatmap=sig(logits[1]),
            canvas_polygon=poly_c,
            frame=fr,
        )

    def fit(self, shape, regularize: str | None = None, tta: bool = False, **kw) -> np.ndarray:
        """Fit a clean polygon to ``shape``.

        ``shape`` may be a binary mask (2-D array, any resolution), an ``(N, 2)``
        array of noisy outline points, or a shapely ``Polygon``. Returns an
        ``(N, 2)`` array of vertices in the input's coordinates.

        ``regularize``: ``None``, ``"orthogonal"`` (snap edges to two
        perpendicular directions - buildings, rooms) or ``"parallel"`` (make
        near-parallel edges exactly parallel).
        """
        return self.fit_detailed(shape, regularize=regularize, tta=tta, **kw).polygon


_default: ShapeFitter | None = None


def fit_polygon(shape, weights: str | Path | None = None, **kw) -> np.ndarray:
    """One-shot convenience wrapper around :class:`ShapeFitter` (model is cached)."""
    global _default
    if weights is not None:
        return ShapeFitter(weights).fit(shape, **kw)
    if _default is None:
        _default = ShapeFitter()
    return _default.fit(shape, **kw)


__all__ = ["ShapeFitter", "FitResult", "fit_polygon", "SIZE"]

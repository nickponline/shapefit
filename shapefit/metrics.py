"""Polygon comparison metrics."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from shapely.validation import make_valid


def _poly(v: np.ndarray):
    if len(v) < 3:
        return Polygon()
    p = Polygon(v)
    if p.is_valid:
        return p
    p = make_valid(p)
    if p.geom_type == "GeometryCollection":
        p = p.buffer(0)
    return p


def iou(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = _poly(a), _poly(b)
    if pa.is_empty or pb.is_empty:
        return 0.0
    u = pa.union(pb).area
    return float(pa.intersection(pb).area / u) if u > 0 else 0.0


def hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = _poly(a), _poly(b)
    if pa.is_empty or pb.is_empty:
        return float("inf")
    return float(pa.boundary.hausdorff_distance(pb.boundary))


def vertex_match(pred: np.ndarray, gt: np.ndarray, thresh: float):
    """Hungarian matching of vertices. Returns (precision, recall, mean matched error)."""
    if len(pred) == 0 or len(gt) == 0:
        return 0.0, 0.0, float("inf")
    d = np.linalg.norm(pred[:, None] - gt[None], axis=2)
    r, c = linear_sum_assignment(d)
    md = d[r, c]
    tp = int((md <= thresh).sum())
    return tp / len(pred), tp / len(gt), float(md.mean())


def score(pred: np.ndarray, gt: np.ndarray) -> dict:
    """All metrics; distances normalised by the GT bbox's longest side."""
    size = float((gt.max(0) - gt.min(0)).max())
    p, r, err = vertex_match(pred, gt, 0.03 * size)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {
        "iou": iou(pred, gt),
        "hausdorff": hausdorff(pred, gt) / size,
        "vertex_err": err / size,
        "vertex_f1": f1,
        "count_ok": float(len(pred) == len(gt)),
    }


def summarize(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        v = v[np.isfinite(v)]
        out[k] = float(np.median(v)) if k in ("hausdorff", "vertex_err") else float(v.mean())
    return out

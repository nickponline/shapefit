"""Mapping between user space (any resolution) and the canonical network canvas.

The shape's bounding box is centred in a ``SIZE x SIZE`` canvas and scaled
uniformly so its longest side covers ``FILL`` of the canvas. Pixel centres are
at integer coordinates; points are ``(x, y) = (column, row)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .data import FILL, SIZE, render_polygon


@dataclass
class Frame:
    scale: float
    center: np.ndarray
    size: int = SIZE

    def to_canvas(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts, np.float64) - self.center) * self.scale + (self.size - 1) / 2

    def from_canvas(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts, np.float64) - (self.size - 1) / 2) / self.scale + self.center


def _bbox_frame(lo: np.ndarray, hi: np.ndarray, size: int, fill: float) -> Frame:
    extent = max(float((hi - lo).max()), 1e-6)
    return Frame(scale=fill * size / extent, center=(lo + hi) / 2, size=size)


def foreground(mask: np.ndarray, min_component_frac: float = 0.02) -> np.ndarray:
    """Binarise and drop specks smaller than ``min_component_frac`` of the largest blob."""
    m = (np.asarray(mask) > (0.5 if np.asarray(mask).dtype.kind == "f" else 0)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.nonzero(areas >= min_component_frac * areas.max())[0] + 1
    return np.isin(lab, keep).astype(np.uint8)


def frame_mask(mask: np.ndarray, size: int = SIZE, fill: float = FILL) -> tuple[np.ndarray, Frame]:
    """Return ``(canvas, frame)`` for a binary mask of any resolution."""
    m = foreground(mask)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        raise ValueError("mask is empty")
    # pixel (x, y) covers [x-0.5, x+0.5]: use the true extent of the pixels
    lo = np.array([xs.min() - 0.5, ys.min() - 0.5])
    hi = np.array([xs.max() + 0.5, ys.max() + 0.5])
    fr = _bbox_frame(lo, hi, size, fill)
    s = fr.scale
    src = m.astype(np.float32)
    if s < 0.7:  # anti-alias before downsampling
        src = cv2.GaussianBlur(src, (0, 0), 0.45 / s)
    t = (size - 1) / 2 - s * fr.center
    M = np.array([[s, 0, t[0]], [0, s, t[1]]], np.float32)
    canvas = cv2.warpAffine(src, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)
    return (canvas >= 0.5).astype(np.uint8), fr


def frame_points(pts: np.ndarray, size: int = SIZE, fill: float = FILL) -> tuple[np.ndarray, Frame]:
    """Return ``(canvas, frame)`` for a (noisy) polygon given as an ``(N, 2)`` vertex list."""
    pts = np.asarray(pts, np.float64)
    fr = _bbox_frame(pts.min(0), pts.max(0), size, fill)
    return render_polygon(fr.to_canvas(pts), size), fr

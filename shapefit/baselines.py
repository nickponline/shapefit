"""Classical (non-learned) baselines, for comparison."""

from __future__ import annotations

import cv2
import numpy as np

from .shapes import clean_vertices


def _contour(mask: np.ndarray) -> np.ndarray | None:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    return max(cs, key=cv2.contourArea)


def douglas_peucker(mask: np.ndarray, eps_frac: float = 0.02) -> np.ndarray:
    """``cv2.approxPolyDP`` on the largest contour; eps as a fraction of perimeter."""
    c = _contour(mask)
    if c is None:
        return np.zeros((0, 2))
    eps = eps_frac * cv2.arcLength(c, True)
    v = cv2.approxPolyDP(c, eps, True)[:, 0].astype(np.float64)
    return clean_vertices(v) if len(v) >= 3 else v


def convex_hull(mask: np.ndarray, eps_frac: float = 0.02) -> np.ndarray:
    c = _contour(mask)
    if c is None:
        return np.zeros((0, 2))
    h = cv2.convexHull(c)
    v = cv2.approxPolyDP(h, eps_frac * cv2.arcLength(h, True), True)[:, 0].astype(np.float64)
    return clean_vertices(v) if len(v) >= 3 else v


def min_area_rect(mask: np.ndarray) -> np.ndarray:
    c = _contour(mask)
    if c is None:
        return np.zeros((0, 2))
    return cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float64)

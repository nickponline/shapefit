"""Synthetic training data: (irregular mask) -> (clean mask, vertex heatmap)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .degrade import geometric_corrupt, raster_corrupt
from .shapes import random_polygon

SIZE = 128  # canonical canvas size
FILL = 0.72  # the shape's bbox occupies this fraction of the canvas
HEATMAP_SIGMA = 1.5


def render_polygon(pts: np.ndarray, size: int | tuple[int, int], ss: int = 4) -> np.ndarray:
    """Rasterise a polygon given in pixel coordinates (pixel centres at integers).

    Supersampled ``ss`` x and area-averaged, so a pixel is set iff >= 50 % covered
    (plain ``cv2.fillPoly`` is biased ~0.45 px outward).
    """
    h, w = (size, size) if np.isscalar(size) else size
    big = np.zeros((h * ss, w * ss), np.uint8)
    p = np.round(((np.asarray(pts) + 0.5) * ss - 0.5) * 16).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(big, [p], 255, lineType=cv2.LINE_8, shift=4)
    small = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    return (small >= 128).astype(np.uint8)


def vertex_heatmap(pts: np.ndarray, size: int, sigma: float = HEATMAP_SIGMA) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    h = np.zeros((size, size), np.float32)
    for x, y in pts:
        h = np.maximum(h, np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)))
    # the pixel nearest each vertex is an exact 1.0 "positive" for the focal loss
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < size and 0 <= yi < size:
            h[yi, xi] = 1.0
    return h


@dataclass
class Sample:
    noisy: np.ndarray  # (S, S) uint8 corrupted input
    mask: np.ndarray  # (S, S) uint8 clean target
    heatmap: np.ndarray  # (S, S) float32 vertex heatmap
    vertices: np.ndarray  # (N, 2) clean polygon in canvas pixels
    family: str


def make_sample(
    rng: np.random.Generator,
    size: int = SIZE,
    severity: float | None = None,
    family: str | None = None,
) -> Sample:
    poly, fam = random_polygon(rng, family)
    if severity is None:
        # mostly moderate, sometimes clean, sometimes harsh
        severity = float(np.clip(rng.gamma(4.0, 0.25), 0.05, 2.0))
    dense = geometric_corrupt(poly, rng, severity)

    # frame on the *corrupted* outline, exactly like inference does
    lo, hi = dense.min(0), dense.max(0)
    extent = float((hi - lo).max())
    fill = FILL * rng.uniform(0.92, 1.08)
    scale = fill * size / extent
    centre = (lo + hi) / 2
    jitter = rng.uniform(-2, 2, 2)
    to_px = lambda p: (p - centre) * scale * np.array([1.0, -1.0]) + (size - 1) / 2 + jitter

    verts = to_px(poly)
    noisy = render_polygon(to_px(dense), size)
    noisy = raster_corrupt(noisy, rng, shape_px=fill * size, severity=severity)
    return Sample(
        noisy=noisy,
        mask=render_polygon(verts, size),
        heatmap=vertex_heatmap(verts, size).astype(np.float32),
        vertices=verts,
        family=fam,
    )


class SyntheticShapes(IterableDataset):
    """Infinite stream of freshly generated training pairs."""

    def __init__(self, seed: int = 0, size: int = SIZE):
        self.seed = seed
        self.size = size

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, wid])
        while True:
            s = make_sample(rng, self.size)
            yield (
                torch.from_numpy(s.noisy[None].astype(np.float32)),
                torch.from_numpy(s.mask[None].astype(np.float32)),
                torch.from_numpy(s.heatmap[None]),
            )


def fixed_set(n: int, seed: int, size: int = SIZE, **kw) -> list[Sample]:
    rng = np.random.default_rng(seed)
    return [make_sample(rng, size, **kw) for _ in range(n)]

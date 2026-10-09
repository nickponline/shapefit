"""Full-resolution test scenes: one corrupted shape somewhere in a larger image.

Unlike :func:`shapefit.data.make_sample` (which renders straight into the
canonical canvas), a scene goes through the exact public preprocessing path
(:func:`shapefit.frame.frame_mask`), so it exercises the library end to end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Sample, render_polygon, vertex_heatmap
from .degrade import geometric_corrupt, raster_corrupt
from .frame import frame_mask
from .shapes import random_polygon


@dataclass
class Scene:
    image: np.ndarray  # (H, W) uint8 corrupted mask
    vertices: np.ndarray  # (N, 2) clean polygon in image pixels
    family: str
    severity: float


def make_scene(
    rng: np.random.Generator,
    hw: tuple[int, int] = (360, 480),
    severity: float | None = None,
    family: str | None = None,
) -> Scene:
    H, W = hw
    poly, fam = random_polygon(rng, family)
    if severity is None:
        severity = float(np.clip(rng.gamma(4.0, 0.25), 0.3, 1.6))
    dense = geometric_corrupt(poly, rng, severity)
    size_px = rng.uniform(0.4, 0.8) * min(H, W)
    half = size_px * 0.6
    c = np.array([rng.uniform(half, W - half), rng.uniform(half, H - half)])
    to_px = lambda p: p * size_px * np.array([1.0, -1.0]) + c
    img = render_polygon(to_px(dense), (H, W))
    img = raster_corrupt(img, rng, shape_px=size_px, severity=severity)
    return Scene(img, to_px(poly), fam, severity)


def scene_to_sample(scene: Scene) -> Sample:
    """Training pair in the canonical canvas, produced exactly like inference."""
    canvas, fr = frame_mask(scene.image)
    v = fr.to_canvas(scene.vertices)
    return Sample(
        noisy=canvas,
        mask=render_polygon(v, fr.size),
        heatmap=vertex_heatmap(v, fr.size).astype(np.float32),
        vertices=v,
        family=scene.family,
    )


def scene_set(n: int, seed: int, **kw) -> list[Scene]:
    rng = np.random.default_rng(seed)
    return [make_scene(rng, **kw) for _ in range(n)]

"""Corruptions that turn a clean polygon into a realistic, irregular blob.

Two stages:

1. **Geometric** corruptions act on a densely resampled outline (in the
   polygon's own normalised units, bbox ~1) by displacing it along its normals:
   high-frequency jaggedness, localised bumps and dents, low-frequency wobble,
   staircase steps - plus rounded corners (a morphological buffer).
2. **Raster** corruptions act on the rendered binary mask (pixels): bites and
   gaps in the edge, cuts, attached blobs, holes, speckle, distractor specks,
   pixelation, blur-then-threshold and morphological noise.

Every corruption is a named function in ``GEOMETRIC`` / ``RASTER`` with a base
probability. ``severity`` (0..~2) scales amplitudes and probabilities, and
``only=[...]`` forces an exact set of corruptions (used for the README gallery).
"""

from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import Polygon

# ----------------------------------------------------------------- geometry


def resample_closed(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polyline to ``n`` points equally spaced by arclength."""
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])], 1)


def outward_normals(dense: np.ndarray) -> np.ndarray:
    """Unit outward normals of a dense CCW closed polyline."""
    d = np.roll(dense, -1, axis=0) - np.roll(dense, 1, axis=0)
    nrm = np.stack([d[:, 1], -d[:, 0]], 1)
    return nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)


def _circular_smooth(x: np.ndarray, width: float) -> np.ndarray:
    if width <= 0:
        return x
    k = int(max(1, round(width * 3)))
    ker = np.exp(-0.5 * (np.arange(-k, k + 1) / max(width, 1e-6)) ** 2)
    ker /= ker.sum()
    return np.convolve(np.concatenate([x[-k:], x, x[:k]]), ker, mode="valid")


def round_corners(pts: np.ndarray, rng: np.random.Generator, severity: float) -> np.ndarray:
    """Round convex (or concave) corners with a morphological buffer."""
    r = rng.uniform(0.01, 0.05) * severity
    poly = Polygon(pts)
    if rng.random() < 0.5:
        out = poly.buffer(-r, join_style="round").buffer(r, join_style="round")
    else:
        out = poly.buffer(r, join_style="round").buffer(-r, join_style="round")
    if out.geom_type != "Polygon" or out.is_empty or out.area < 0.5 * poly.area:
        return pts
    q = np.asarray(out.exterior.coords)[:-1]
    return q if out.exterior.is_ccw else q[::-1]


# Each geometric op returns a normal offset for every outline sample.
# s: arclength of each sample, perim: total perimeter.


def _jagged(s, perim, rng, sev):
    amp = rng.uniform(0.002, 0.018) * sev
    noise = _circular_smooth(rng.normal(0, 1, len(s)), rng.uniform(0.0, 2.5))
    return amp * noise / (noise.std() + 1e-9)


def _bumps(s, perim, rng, sev):
    off = np.zeros(len(s))
    for _ in range(int(rng.integers(1, 6))):
        c, sig = rng.uniform(0, perim), rng.uniform(0.015, 0.09)
        amp = rng.uniform(0.015, 0.07) * sev * rng.choice([-1, 1])
        ds = np.abs(s - c)
        ds = np.minimum(ds, perim - ds)
        off += amp * np.exp(-0.5 * (ds / sig) ** 2)
    return off


def _wobble(s, perim, rng, sev):
    off = np.zeros(len(s))
    for _ in range(int(rng.integers(1, 4))):
        f = rng.integers(2, 9)
        amp = rng.uniform(0.003, 0.015) * sev
        off += amp * np.sin(2 * np.pi * f * s / perim + rng.uniform(0, 2 * np.pi))
    return off


def _staircase(s, perim, rng, sev):
    n = len(s)
    return np.repeat(rng.normal(0, 0.008 * sev, n // 20 + 1), 20)[:n]


GEOMETRIC = {  # name: (function, base probability)
    "jagged": (_jagged, 0.75),
    "bumps": (_bumps, 0.65),
    "wobble": (_wobble, 0.4),
    "staircase": (_staircase, 0.2),
}
P_ROUND = 0.35


def geometric_corrupt(
    pts: np.ndarray,
    rng: np.random.Generator,
    severity: float = 1.0,
    n: int = 600,
    only: list[str] | None = None,
) -> np.ndarray:
    """Return a dense, irregular outline derived from the clean CCW polygon ``pts``."""
    pick = lambda name, p: (name in only) if only is not None else rng.random() < p
    base = pts
    if pick("round_corners", P_ROUND * min(severity, 1.0)):
        base = round_corners(base, rng, severity)
    dense = resample_closed(base, n)
    normals = outward_normals(dense)
    perim = float(np.linalg.norm(np.diff(np.vstack([dense, dense[:1]]), axis=0), axis=1).sum())
    s = np.arange(n) * perim / n
    offset = np.zeros(n)
    for name, (fn, p) in GEOMETRIC.items():
        if pick(name, p):
            offset += fn(s, perim, rng, severity)
    return dense + normals * offset[:, None]


# -------------------------------------------------------------------- raster
# Each raster op: f(mask, boundary_pts, rng, sc, sev) -> mask
# sc scales pixel radii, which are specified for a ~100 px shape.


def _rpt(bnd, rng):
    return tuple(int(v) for v in bnd[rng.integers(len(bnd))])


def _bites(m, bnd, rng, sc, sev):
    for _ in range(int(rng.integers(1, 5))):
        r = max(1, int(rng.uniform(1.5, 6.5) * sc * sev))
        cv2.circle(m, _rpt(bnd, rng), r, 0, -1)
    return m


def _gaps(m, bnd, rng, sc, sev):
    L = len(bnd)
    a, length = int(rng.integers(L)), int(L * rng.uniform(0.04, 0.15))
    seg = bnd[np.arange(a, a + length) % L].reshape(-1, 1, 2)
    cv2.polylines(m, [seg], False, 0, max(2, int(rng.uniform(2, 6) * sc * sev)))
    return m


def _blobs(m, bnd, rng, sc, sev):
    for _ in range(int(rng.integers(1, 4))):
        ax = max(1, int(rng.uniform(1.5, 6.0) * sc * sev))
        ay = max(1, int(ax * rng.uniform(0.4, 1.0)))
        cv2.ellipse(m, _rpt(bnd, rng), (ax, ay), rng.uniform(0, 180), 0, 360, 1, -1)
    return m


def _cuts(m, bnd, rng, sc, sev):
    cv2.line(m, _rpt(bnd, rng), _rpt(bnd, rng), 0, max(1, int(rng.uniform(1, 2.5) * sc)))
    return m


def _holes(m, bnd, rng, sc, sev):
    ys, xs = np.nonzero(m)
    for _ in range(int(rng.integers(1, 5))):
        i = rng.integers(len(xs))
        cv2.circle(m, (int(xs[i]), int(ys[i])), max(1, int(rng.uniform(1, 5) * sc * sev)), 0, -1)
    return m


def _speckle(m, bnd, rng, sc, sev):
    k = max(3, int(5 * sc) | 1)
    ker = np.ones((k, k), np.uint8)
    band = cv2.dilate(m, ker) - cv2.erode(m, ker)
    flip = (rng.random(m.shape) < rng.uniform(0.05, 0.3) * sev) & (band > 0)
    return np.where(flip, 1 - m, m).astype(np.uint8)


def _specks(m, bnd, rng, sc, sev):
    H, W = m.shape
    for _ in range(int(rng.integers(1, 6))):
        cv2.circle(m, (int(rng.integers(W)), int(rng.integers(H))), int(rng.integers(0, 3)), 1, -1)
    return m


def _pixelate(m, bnd, rng, sc, sev):
    H, W = m.shape
    f = rng.uniform(2, 2 + 4 * sev)
    small = cv2.resize(m, (max(4, int(W / f)), max(4, int(H / f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)


def _blur(m, bnd, rng, sc, sev):
    b = cv2.GaussianBlur(m.astype(np.float32), (0, 0), rng.uniform(0.7, 2.5) * sc * sev + 0.3)
    return (b > rng.uniform(0.35, 0.65)).astype(np.uint8)


def _morph(m, bnd, rng, sc, sev):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    op = cv2.MORPH_OPEN if rng.random() < 0.5 else cv2.MORPH_CLOSE
    return cv2.morphologyEx(m, op, k, iterations=int(rng.integers(1, 3)))


RASTER = {  # name: (function, base probability, probability scales with severity)
    "bites": (_bites, 0.5, True),
    "gaps": (_gaps, 0.3, True),
    "blobs": (_blobs, 0.4, True),
    "cuts": (_cuts, 0.15, True),
    "holes": (_holes, 0.25, True),
    "speckle": (_speckle, 0.3, True),
    "specks": (_specks, 0.15, True),
    "pixelate": (_pixelate, 0.25, False),
    "blur": (_blur, 0.3, False),
    "morph": (_morph, 0.2, False),
}


def _boundary_points(mask: np.ndarray) -> np.ndarray:
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return np.zeros((0, 2), np.int32)
    return max(cs, key=len)[:, 0, :]


def raster_corrupt(
    mask: np.ndarray,
    rng: np.random.Generator,
    shape_px: float,
    severity: float = 1.0,
    only: list[str] | None = None,
) -> np.ndarray:
    """Corrupt a binary uint8 mask (0/1). ``shape_px`` is the shape's size in pixels."""
    m = np.ascontiguousarray(mask.copy())
    bnd = _boundary_points(m)
    if len(bnd) == 0:
        return m
    sc = shape_px / 100.0
    for name, (fn, p, scaled) in RASTER.items():
        go = (name in only) if only is not None else rng.random() < p * (min(severity, 1.0) if scaled else 1.0)
        if go:
            m = np.ascontiguousarray(fn(m, bnd, rng, sc, severity).astype(np.uint8))
    return m

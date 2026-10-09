"""Random generators for clean, simple polygons.

All generators return an ``(N, 2)`` float array of vertices in counter-clockwise
order, roughly centred on the origin with a bounding box of size ~1. Every
polygon is simple (non self-intersecting), has no near-collinear vertices and
no very short edges, so that "the corners" are a well defined target.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

MIN_TURN_DEG = 12.0  # vertices that turn less than this are considered collinear
MIN_EDGE = 0.06  # minimum edge length relative to the bbox size


def _turn_angles(pts: np.ndarray) -> np.ndarray:
    """Absolute turning angle (degrees) at every vertex of a closed polygon."""
    a = pts - np.roll(pts, 1, axis=0)
    b = np.roll(pts, -1, axis=0) - pts
    ang = np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1))
    return np.degrees(np.abs(ang))


def clean_vertices(pts: np.ndarray, min_turn: float = MIN_TURN_DEG) -> np.ndarray:
    """Drop duplicate and near-collinear vertices, return CCW order."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    changed = True
    while changed and len(pts) > 3:
        changed = False
        d = np.linalg.norm(pts - np.roll(pts, 1, axis=0), axis=1)
        keep = d > 1e-9
        if not keep.all():
            pts = pts[keep]
            changed = True
            continue
        turn = _turn_angles(pts)
        i = int(np.argmin(turn))
        if turn[i] < min_turn:
            pts = np.delete(pts, i, axis=0)
            changed = True
    if signed_area(pts) < 0:
        pts = pts[::-1]
    return pts


def signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def normalize(pts: np.ndarray) -> np.ndarray:
    """Centre on the bbox centre and scale so the bbox's longest side is 1."""
    lo, hi = pts.min(0), pts.max(0)
    return (pts - (lo + hi) / 2) / max(float((hi - lo).max()), 1e-9)


def is_good(pts: np.ndarray) -> bool:
    if len(pts) < 3:
        return False
    poly = Polygon(pts)
    if not poly.is_valid or poly.area < 0.08:  # too thin / sliver-like
        return False
    p = normalize(pts)
    edges = np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1)
    if edges.min() < MIN_EDGE:
        return False
    if _turn_angles(p).min() < MIN_TURN_DEG:
        return False
    # very sharp spikes are ambiguous after rasterisation
    if _turn_angles(p).max() > 165:
        return False
    return True


# --------------------------------------------------------------------- families

def regular_ngon(rng: np.random.Generator) -> np.ndarray:
    n = int(rng.integers(3, 9))
    t = rng.uniform(0, 2 * np.pi) + np.arange(n) * 2 * np.pi / n
    pts = np.stack([np.cos(t), np.sin(t)], 1)
    pts[:, 0] *= rng.uniform(0.6, 1.0)  # mild anisotropy, still a clean polygon
    return pts


def rectangle(rng: np.random.Generator) -> np.ndarray:
    w, h = 1.0, rng.uniform(0.25, 1.0)
    pts = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2
    return _rotate(pts, rng.uniform(0, np.pi))


def quad(rng: np.random.Generator) -> np.ndarray:
    """Trapezoids, parallelograms and kites."""
    kind = rng.integers(3)
    if kind == 0:  # trapezoid
        top = rng.uniform(0.2, 0.9)
        off = rng.uniform(-0.3, 0.3)
        pts = np.array([[-0.5, 0], [0.5, 0], [off + top / 2, 0.7], [off - top / 2, 0.7]])
    elif kind == 1:  # parallelogram
        s = rng.uniform(0.15, 0.5) * rng.choice([-1, 1])
        h = rng.uniform(0.35, 0.8)
        pts = np.array([[0, 0], [1, 0], [1 + s, h], [s, h]])
    else:  # kite
        a, b = rng.uniform(0.2, 0.6), rng.uniform(0.3, 0.6)
        pts = np.array([[0, -b], [a, 0], [0, 1 - b], [-a, 0]])
    return _rotate(pts, rng.uniform(0, 2 * np.pi))


def random_convex(rng: np.random.Generator) -> np.ndarray:
    n = int(rng.integers(5, 12))
    poly = Polygon(rng.uniform(-1, 1, size=(n * 3, 2))).convex_hull
    pts = np.asarray(poly.exterior.coords)[:-1]
    return pts


def star_shaped(rng: np.random.Generator) -> np.ndarray:
    """Concave polygon: sorted angles with random radii."""
    n = int(rng.integers(5, 13))
    t = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = rng.uniform(0.45, 1.0, n)
    if rng.random() < 0.3:  # proper "star": alternate in/out
        t = rng.uniform(0, 2 * np.pi) + np.arange(n) * 2 * np.pi / n
        r = np.where(np.arange(n) % 2 == 0, 1.0, rng.uniform(0.4, 0.7))
    return np.stack([r * np.cos(t), r * np.sin(t)], 1)


def rectilinear(rng: np.random.Generator) -> np.ndarray:
    """Building-like L / T / U / plus shapes: union of axis-aligned rectangles."""
    k = int(rng.integers(2, 5))
    rects = [box(-0.5, -0.5, 0.5, 0.5)]
    for _ in range(k - 1):
        base = rects[int(rng.integers(len(rects)))].bounds
        cx = rng.uniform(base[0], base[2])
        cy = rng.uniform(base[1], base[3])
        w, h = rng.uniform(0.25, 0.9, 2)
        rects.append(box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    u = unary_union(rects)
    if u.geom_type != "Polygon" or len(u.interiors):
        return np.zeros((0, 2))
    pts = np.asarray(u.exterior.coords)[:-1]
    return _rotate(pts, rng.uniform(0, np.pi / 2))


def _rotate(pts: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return pts @ np.array([[c, s], [-s, c]])


FAMILIES = {
    "ngon": (regular_ngon, 0.15),
    "rectangle": (rectangle, 0.15),
    "quad": (quad, 0.12),
    "convex": (random_convex, 0.18),
    "star": (star_shaped, 0.18),
    "rectilinear": (rectilinear, 0.22),
}


def random_polygon(rng: np.random.Generator, family: str | None = None) -> tuple[np.ndarray, str]:
    """Sample a clean polygon. Returns ``(vertices, family_name)``."""
    names = list(FAMILIES)
    probs = np.array([FAMILIES[n][1] for n in names])
    for _ in range(100):
        name = family or names[rng.choice(len(names), p=probs / probs.sum())]
        pts = FAMILIES[name][0](rng)
        if len(pts) < 3:
            continue
        pts = clean_vertices(pts)
        if rng.random() < 0.5:  # random reflection
            pts = pts * np.array([-1.0, 1.0])
            pts = clean_vertices(pts)
        if is_good(pts):
            return normalize(pts), name
    raise RuntimeError("failed to sample a valid polygon")


def polygon_from(pts: np.ndarray) -> Polygon:
    return Polygon(np.asarray(pts))


__all__ = [
    "FAMILIES",
    "random_polygon",
    "clean_vertices",
    "normalize",
    "signed_area",
    "polygon_from",
]

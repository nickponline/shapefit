"""Turn network outputs (clean-mask probability + vertex heatmap) into a polygon.

1. Extract the outer contour of the predicted clean mask (upsampled for
   sub-pixel accuracy). This gives a dense, *ordered* boundary.
2. Find vertex candidates as local maxima of the heatmap and snap each one to
   the contour. Sorting by contour position yields the polygon's vertex order
   for free - no combinatorial ordering problem.
3. For every edge, fit a total-least-squares line to the contour points between
   its two vertices (ignoring the ends near the corners), then intersect
   consecutive lines. Corners become sharp and sub-pixel accurate even though
   the heatmap peaks are blurry.
4. Optionally regularise edge directions (orthogonal / parallel snapping).
"""

from __future__ import annotations

import cv2
import numpy as np

from .shapes import clean_vertices


def _largest_contour(binary: np.ndarray) -> np.ndarray | None:
    cs, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    return c[:, 0, :].astype(np.float64)


def find_peaks(heat: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of ``heat`` above ``threshold`` with sub-pixel centroid refinement."""
    dil = cv2.dilate(heat, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero((heat >= dil) & (heat > threshold))
    pts, scores = [], []
    H, W = heat.shape
    for y, x in zip(ys, xs):
        y0, y1, x0, x1 = max(0, y - 1), min(H, y + 2), max(0, x - 1), min(W, x + 2)
        win = heat[y0:y1, x0:x1]
        gy, gx = np.mgrid[y0:y1, x0:x1]
        w = win.sum() + 1e-9
        pts.append(((gx * win).sum() / w, (gy * win).sum() / w))
        scores.append(heat[y, x])
    return np.array(pts, np.float64).reshape(-1, 2), np.array(scores)


def _fit_line(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total least squares: returns (centroid, unit direction)."""
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt[0]


def _intersect(c1, d1, c2, d2) -> np.ndarray | None:
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, c2 - c1)
    return c1 + t[0] * d1


def _angle_between(d1, d2) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(d1, d2)), 0, 1))))


def regularize_directions(lines: list, mode: str, tol_deg: float | None = None) -> list:
    """Snap edge directions. ``orthogonal``: to a dominant axis or its normal.
    ``parallel``: group near-parallel edges and give them a shared direction."""
    dirs = np.array([d for _, d, _ in lines])
    lens = np.array([w for _, _, w in lines])
    ang = np.arctan2(dirs[:, 1], dirs[:, 0])
    out = []
    if tol_deg is None:
        tol_deg = 25.0 if mode == "orthogonal" else 12.0
    if mode == "orthogonal":
        # length-weighted circular mean of angles modulo 90 degrees
        z = (lens * np.exp(4j * ang)).sum()
        dom = np.angle(z) / 4
        for (c, d, w), a in zip(lines, ang):
            k = np.round((a - dom) / (np.pi / 2))
            snapped = dom + k * np.pi / 2
            if abs(np.degrees(a - snapped)) < tol_deg:
                d = np.array([np.cos(snapped), np.sin(snapped)])
            out.append((c, d, w))
        return out
    if mode == "parallel":
        order = np.argsort(-lens)
        assigned = {}
        groups = []
        for i in order:
            if i in assigned:
                continue
            members = [j for j in order if j not in assigned and _angle_between(dirs[i], dirs[j]) < tol_deg]
            for j in members:
                assigned[j] = len(groups)
            groups.append(members)
        new_dirs = dirs.copy()
        for members in groups:
            ref = dirs[members[0]]
            acc = sum(lens[j] * (dirs[j] if np.dot(dirs[j], ref) >= 0 else -dirs[j]) for j in members)
            acc = acc / (np.linalg.norm(acc) + 1e-12)
            for j in members:
                new_dirs[j] = acc
        return [(c, new_dirs[i], w) for i, (c, _, w) in enumerate(lines)]
    raise ValueError(f"unknown regularize mode {mode!r}")


def _seg_dist(pts: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Distance from each point to the segment ``p``-``q``."""
    d = q - p
    t = np.clip(((pts - p) @ d) / max(float(d @ d), 1e-12), 0, 1)
    return np.linalg.norm(pts - (p + t[:, None] * d), axis=1)


def split_missed_corners(contour: np.ndarray, idx: np.ndarray, tol: float, max_iter: int = 64) -> np.ndarray:
    """Add corners the heatmap missed.

    Wherever the contour between two consecutive corners strays more than ``tol``
    pixels from the straight chord joining them, the farthest contour point
    becomes a new corner (one Douglas-Peucker split, seeded by the network's
    corners). Returns sorted contour indices.
    """
    L = len(contour)
    idx = sorted(int(i) for i in idx)
    for _ in range(max_iter):
        new = []
        for k in range(len(idx)):
            a, b = idx[k], idx[(k + 1) % len(idx)]
            seg = np.arange(a, b if b > a else b + L) % L
            if len(idx) == 1:
                seg = np.arange(a, a + L) % L
            if len(seg) < 3:
                continue
            q = contour[b] if len(idx) > 1 else contour[seg[len(seg) // 2]]
            dist = _seg_dist(contour[seg], contour[a], q)
            j = int(np.argmax(dist))
            if dist[j] > tol:
                new.append(int(seg[j]))
        if not new:
            break
        idx = sorted(set(idx) | set(new))
    return np.asarray(idx)


def polygon_from_outputs(
    mask_logits: np.ndarray,
    heat_logits: np.ndarray,
    peak_threshold: float = 0.25,
    upsample: int = 4,
    regularize: str | None = None,
    min_turn_deg: float = 8.0,
    split_tol: float | None = 3.0,
) -> np.ndarray:
    """Build a polygon (canvas pixel coordinates, (N, 2)) from raw logits ``(S, S)``.

    ``split_tol`` (canvas pixels): add a corner wherever the predicted outline
    strays further than this from the edge between two detected corners. This
    recovers corners whose heatmap peak was too weak. ``None`` disables it.
    """
    S = mask_logits.shape[0]
    big = cv2.resize(mask_logits.astype(np.float32), (S * upsample, S * upsample), interpolation=cv2.INTER_LINEAR)
    contour = _largest_contour(big > 0)
    if contour is None or len(contour) < 8:
        return np.zeros((0, 2))
    contour = (contour + 0.5) / upsample - 0.5  # upsampled pixel centre -> canvas coords
    L = len(contour)

    heat = 1 / (1 + np.exp(-heat_logits.astype(np.float32)))
    peaks, scores = find_peaks(heat, peak_threshold)

    # snap peaks to the contour; discard stray ones far from the boundary
    idx = []
    if len(peaks):
        d = np.linalg.norm(contour[None] - peaks[:, None], axis=2)
        near = d.argmin(1)
        ok = d[np.arange(len(peaks)), near] < 4.0
        peaks, scores, near = peaks[ok], scores[ok], near[ok]
        order = np.argsort(near)
        peaks, scores, idx = peaks[order], scores[order], near[order]

    # merge peaks that are adjacent along the contour
    if len(idx) > 1:
        step = np.median(np.linalg.norm(np.diff(contour, axis=0), axis=1))
        keep = []
        for i in range(len(idx)):
            if keep and (idx[i] - idx[keep[-1]]) * step < 3.0:
                if scores[i] > scores[keep[-1]]:
                    keep[-1] = i
                continue
            keep.append(i)
        if len(keep) > 1 and (idx[keep[0]] + L - idx[keep[-1]]) * step < 3.0:
            keep = keep[1:] if scores[keep[0]] < scores[keep[-1]] else keep[:-1]
        peaks, scores, idx = peaks[keep], scores[keep], np.asarray(idx)[keep]

    if split_tol and len(idx) >= 1:
        found = {int(i): p for i, p in zip(idx, peaks)}
        idx = split_missed_corners(contour, idx, split_tol)
        peaks = np.array([found.get(int(i), contour[int(i)]) for i in idx]).reshape(-1, 2)
    if len(idx) < 3:  # network found no usable corners: fall back to Douglas-Peucker
        approx = cv2.approxPolyDP(contour.astype(np.float32).reshape(-1, 1, 2), 1.5, True)[:, 0]
        return _orient(clean_vertices(approx.astype(np.float64), min_turn_deg))

    # per-edge line fits on the contour between consecutive vertices
    n = len(idx)
    lines = []
    for k in range(n):
        a, b = int(idx[k]), int(idx[(k + 1) % n])
        seg_idx = np.arange(a, b if b > a else b + L) % L
        m = len(seg_idx)
        trim = max(int(round(m * 0.15)), upsample * 2)
        inner = seg_idx[trim : m - trim] if m - 2 * trim >= 4 else np.array([], int)
        if len(inner) >= 4:
            c, d = _fit_line(contour[inner])
        else:
            p, q = peaks[k], peaks[(k + 1) % n]
            c, d = (p + q) / 2, (q - p) / (np.linalg.norm(q - p) + 1e-12)
        lines.append((c, d, float(np.linalg.norm(peaks[(k + 1) % n] - peaks[k]))))

    if regularize:
        lines = regularize_directions(lines, regularize)

    verts = []
    for k in range(n):
        c1, d1, _ = lines[k - 1]
        c2, d2, _ = lines[k]
        v = None
        if _angle_between(d1, d2) > 5.0:
            v = _intersect(c1, d1, c2, d2)
        if v is None:
            if regularize:  # parallel after snapping: the vertex disappears
                continue
            v = peaks[k]
        elif np.linalg.norm(v - peaks[k]) > 4.0 and not regularize:
            v = peaks[k]  # line intersection ran away, trust the heatmap
        verts.append(v)
    verts = np.array(verts)
    if len(verts) < 3:
        return _orient(peaks)
    return _orient(clean_vertices(verts, min_turn_deg))


def _orient(v: np.ndarray) -> np.ndarray:
    """Counter-clockwise as displayed on screen (x right, y down)."""
    if len(v) < 3:
        return v
    x, y = v[:, 0], v[:, 1]
    area = np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)
    return v[::-1] if area > 0 else v

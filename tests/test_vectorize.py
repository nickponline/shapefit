import cv2
import numpy as np

from shapefit import baselines
from shapefit.data import fixed_set
from shapefit.metrics import iou, score, summarize
from shapefit.vectorize import polygon_from_outputs


def _logit(h):
    h = np.clip(h, 1e-4, 1 - 1e-4)
    return np.log(h / (1 - h))


def _oracle(s, **kw):
    m = cv2.GaussianBlur(s.mask.astype(np.float32) * 2 - 1, (0, 0), 0.7) * 8
    return polygon_from_outputs(m, _logit(s.heatmap), **kw)


def test_oracle_outputs_give_exact_polygons():
    """With perfect network outputs the vectorizer recovers the polygon."""
    samples = fixed_set(150, seed=11)
    rows = [score(_oracle(s), s.vertices) for s in samples]
    m = summarize(rows)
    assert m["iou"] > 0.985
    assert m["count_ok"] > 0.97
    assert m["vertex_err"] < 0.01


def test_orthogonal_regularization_gives_right_angles():
    samples = fixed_set(40, seed=12, family="rectilinear")
    for s in samples:
        v = _oracle(s, regularize="orthogonal")
        e = np.roll(v, -1, 0) - v
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        dots = np.abs((e * np.roll(e, -1, 0)).sum(1))
        assert dots.max() < 1e-6
        assert iou(v, s.vertices) > 0.95


def test_parallel_regularization_runs():
    s = fixed_set(1, seed=13, family="quad")[0]
    v = _oracle(s, regularize="parallel")
    assert len(v) >= 3


def test_vectorizer_falls_back_without_peaks():
    s = fixed_set(1, seed=14, family="rectangle")[0]
    v = polygon_from_outputs(s.mask * 16.0 - 8, np.full(s.mask.shape, -10.0))
    assert iou(v, s.vertices) > 0.9


def test_baselines_run():
    s = fixed_set(1, seed=15)[0]
    for f in (baselines.douglas_peucker, baselines.convex_hull, baselines.min_area_rect):
        assert len(f(s.noisy)) >= 3


def test_oracle_on_rectangle_unions():
    """Unions of many overlapping rectangles have many corners; the vectorizer must keep them all."""
    samples = fixed_set(80, seed=16, family="rect_union")
    assert max(len(s.vertices) for s in samples) >= 12
    m = summarize([score(_oracle(s), s.vertices) for s in samples])
    assert m["iou"] > 0.98
    assert m["count_ok"] > 0.9

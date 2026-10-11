import numpy as np
import pytest
from shapely.geometry import Polygon

from shapefit.data import fixed_set, make_sample, render_polygon
from shapefit.degrade import geometric_corrupt, raster_corrupt
from shapefit.frame import frame_mask, frame_points
from shapefit.make_dataset import _pack, _unpack
from shapefit.scenes import scene_set, scene_to_sample
from shapefit.shapes import FAMILIES, _turn_angles, random_polygon, signed_area


@pytest.mark.parametrize("family", list(FAMILIES))
def test_polygons_are_clean(family):
    rng = np.random.default_rng(0)
    for _ in range(50):
        v, fam = random_polygon(rng, family)
        assert fam == family
        assert Polygon(v).is_valid
        assert signed_area(v) > 0  # CCW
        assert _turn_angles(v).min() >= 12
        assert np.isclose((v.max(0) - v.min(0)).max(), 1.0)


def test_render_is_unbiased():
    sq = np.array([[10.0, 10.0], [50.0, 10.0], [50.0, 40.0], [10.0, 40.0]]) + 0.3
    m = render_polygon(sq, 64)
    assert abs(int(m.sum()) - 40 * 30) <= 40  # within ~1/4 px of perimeter


def test_corruptions_change_the_shape_but_keep_it():
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = make_sample(rng, severity=1.0)
        assert s.noisy.dtype == np.uint8 and set(np.unique(s.noisy)) <= {0, 1}
        inter = (s.noisy & s.mask).sum()
        union = (s.noisy | s.mask).sum()
        assert 0.6 < inter / union < 1.0


def test_heatmap_has_one_positive_per_vertex():
    s = make_sample(np.random.default_rng(2))
    assert (s.heatmap == 1.0).sum() == len(s.vertices)


def test_frame_roundtrip_and_fill():
    sc = scene_set(5, seed=3)
    for scene in sc:
        canvas, fr = frame_mask(scene.image)
        pts = np.random.default_rng(0).uniform(0, 300, (10, 2))
        assert np.allclose(fr.from_canvas(fr.to_canvas(pts)), pts)
        ys, xs = np.nonzero(canvas)
        extent = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
        assert abs(extent - 0.72 * 128) < 4
        t = scene_to_sample(scene)
        assert Polygon(t.vertices).is_valid


def test_frame_points():
    pts = np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], float)
    canvas, fr = frame_points(pts)
    assert canvas.shape == (128, 128)
    assert np.allclose(fr.from_canvas(fr.to_canvas(pts)), pts)


def test_pack_roundtrip():
    m = (np.random.default_rng(0).random((128, 128)) > 0.5).astype(np.uint8)
    assert np.array_equal(_unpack(_pack(m), 128), m)


def test_fixed_set_is_deterministic():
    a, b = fixed_set(3, seed=5), fixed_set(3, seed=5)
    for x, y in zip(a, b):
        assert np.array_equal(x.noisy, y.noisy)


def test_rect_union_is_a_union_of_overlapping_rectangles():
    rng = np.random.default_rng(4)
    counts = [len(random_polygon(rng, "rect_union")[0]) for _ in range(200)]
    assert min(counts) >= 4 and max(counts) >= 14
    assert np.mean(counts) > 8  # mostly more complex than a single L or T

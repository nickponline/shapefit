from pathlib import Path

import numpy as np
import pytest
import torch
from shapely.geometry import Polygon

from shapefit import ShapeFitter, fit_polygon
from shapefit.data import fixed_set
from shapefit.metrics import iou
from shapefit.model import ShapeFitNet, focal_heatmap_loss, mask_loss
from shapefit.scenes import scene_set

DEMO = Path(__file__).parents[1] / "shapefit" / "weights" / "overfit_demo.pt"


@pytest.fixture(scope="module")
def random_ckpt(tmp_path_factory):
    torch.manual_seed(0)
    net = ShapeFitNet(base=8, depth=3)
    p = tmp_path_factory.mktemp("w") / "tiny.pt"
    torch.save({"config": net.config, "state_dict": net.state_dict()}, p)
    return p


def test_model_shapes():
    net = ShapeFitNet(base=8, depth=3)
    y = net(torch.zeros(2, 1, 128, 128))
    assert y.shape == (2, 2, 128, 128)


def test_a_few_steps_reduce_loss():
    torch.manual_seed(0)
    s = fixed_set(4, seed=1)
    x = torch.tensor(np.stack([q.noisy for q in s])[:, None], dtype=torch.float32)
    m = torch.tensor(np.stack([q.mask for q in s])[:, None], dtype=torch.float32)
    h = torch.tensor(np.stack([q.heatmap for q in s])[:, None])
    net = ShapeFitNet(base=8, depth=3)
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    losses = []
    for _ in range(30):
        y = net(x)
        loss = mask_loss(y[:, :1], m) + focal_heatmap_loss(y[:, 1:], h)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.7 * losses[0]


def test_api_accepts_all_input_kinds(random_ckpt):
    fitter = ShapeFitter(random_ckpt)
    scene = scene_set(1, seed=0)[0]
    for shape in (scene.image, scene.image.astype(bool), scene.vertices, Polygon(scene.vertices)):
        out = fitter.fit(shape)
        assert out.ndim == 2 and out.shape[1] == 2
    r = fitter.fit_detailed(scene.image, tta=True)
    assert r.clean_mask.shape == (128, 128)


def test_missing_weights_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        ShapeFitter(tmp_path / "nope.pt")


def test_empty_mask_raises(random_ckpt):
    with pytest.raises(ValueError):
        fit_polygon(np.zeros((50, 50), np.uint8), weights=random_ckpt)


@pytest.mark.skipif(not DEMO.exists(), reason="demo weights not built")
def test_overfit_demo_memorised_its_scenes():
    """The demo checkpoint was overfit on these exact scenes via the public API."""
    fitter = ShapeFitter(DEMO)
    meta = fitter.meta
    scenes = scene_set(meta["overfit"], seed=meta["overfit_seed"])
    ious = [iou(fitter.fit(sc.image), sc.vertices) for sc in scenes]
    assert np.mean(ious) > 0.95


def test_tta_inverts_its_transforms(random_ckpt):
    """With a pointwise 'model', TTA must give exactly the plain prediction."""
    fitter = ShapeFitter(random_ckpt)

    class Pointwise(torch.nn.Module):
        def forward(self, x):
            return torch.cat([x * 3 - 1, x ** 2], 1)

    fitter.model = Pointwise()
    canvas = (np.random.default_rng(0).random((2, 128, 128)) > 0.5).astype(np.float32)
    assert np.allclose(fitter.predict_canvas(canvas), fitter.predict_canvas(canvas, tta=True), atol=1e-6)

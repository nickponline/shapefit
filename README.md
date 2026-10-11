# shapefit

**Fit clean polygons to irregular shapes with a neural network.**

shapefit takes a mask with bad edges (bumps, jagged steps, gaps, holes,
rounded corners) and gives back a polygon with the correct corners. It works
on single shapes and on unions of overlapping rectangles, such as building
footprints and room layouts.

![pipeline](docs/images/pipeline.png)

## Installation

```bash
git clone git@github.com:nickponline/shapefit.git && cd shapefit
pip install -e ".[dev]"      # numpy, torch, opencv-python-headless, shapely, scipy (+ pytest, matplotlib)
pytest                       # 29 tests, approximately 15 s
```

The trained checkpoint is included at `shapefit/weights/shapefit.pt` (15 MB).

## Usage

```python
from shapefit import ShapeFitter, fit_polygon

fitter = ShapeFitter("shapefit/weights/shapefit.pt", device="cpu")   # or "cuda" / "mps"

poly = fitter.fit(mask)                             # mask of any size -> (N, 2) array of (x, y) corners
poly = fitter.fit(noisy_points)                     # or an (M, 2) point list, or a shapely Polygon
poly = fitter.fit(mask, regularize="orthogonal")    # all corners 90° (buildings, rooms)
poly = fitter.fit(mask, regularize="parallel")      # almost-parallel edges become parallel

poly = fit_polygon(mask)                            # short form, uses the shipped checkpoint
```

The corners are in input pixel coordinates (x = column, y = row), in
counter-clockwise order on screen. For a complete example, run
`python examples/quickstart.py`.

## Results

The test shapes come from seeds that training did not use. Green: shapefit.
Orange dashes: `cv2.approxPolyDP` (Douglas–Peucker) with ε = 2 % of the
perimeter.

![results](docs/images/results.png)

![unions](docs/images/unions.png)

| | IoU ↑ | correct number of corners ↑ | vertex F1 ↑ |
|---|---|---|---|
| **shapefit**, 500 shapes of all types | **0.960** | **78.8 %** | **0.845** |
| Douglas–Peucker, best ε (1, 2 or 3 %) for each column | 0.910 | 50.6 % | 0.617 |
| **shapefit**, 200 unions of rectangles | **0.947** | **67.0 %** | **0.818** |
| Douglas–Peucker, ε = 1 % (best), unions | 0.898 | 28.0 % | 0.588 |

## More details

The **[illustrated explainer](https://claude.ai/artifact/TQyf1TVKUrR8SDPAA9RXfq)**
(also in [`docs/explainer.html`](docs/explainer.html)) covers the training
data, the network, the vectoriser, full result tables, an interactive gallery
of 56 new shapes, training, and limits.

## License

MIT

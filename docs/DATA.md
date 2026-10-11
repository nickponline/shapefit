# Training data and the full training run

Everything the network sees is synthetic. A sample is a clean polygon (the
target) and a corrupted rasterisation of it (the input). Because the generator
is cheap (~1 ms/sample on one core), the default is to generate data **on the
fly**: the model never sees the same sample twice, which is the most effective
regulariser for this kind of task.

![dataset](images/dataset.png)

## 1. Clean polygons: `shapefit/shapes.py`

| family | weight | what it is |
|---|---|---|
| `ngon` | 0.12 | regular 3–8-gons, mildly stretched |
| `rectangle` | 0.12 | rotated, aspect ratio 1–4 |
| `quad` | 0.10 | trapezoids, parallelograms, kites |
| `convex` | 0.14 | convex hull of random points (5–11 vertices) |
| `star` | 0.14 | concave star-shaped polygons (5–12 vertices) |
| `rectilinear` | 0.16 | building-like L/T/U/+ unions of 2–4 axis-aligned rectangles, rotated |
| `rect_union` | 0.22 | union of 2–6 overlapping rectangles (4–21 vertices, mean ≈ 10); 60 % share one orientation, 40 % have an independent angle each |

All polygons are randomly reflected and pass `is_good()`, so the target is
well defined:

* simple (shapely `is_valid`), counter-clockwise, and normalised so the bbox's long side is 1
* no near-collinear vertices: every vertex turns at least **12°** (otherwise "is that a corner?" is ambiguous)
* no edge shorter than **6 %** of the size, and no spikes sharper than 15°
* area ≥ 8 % of the bbox, so no slivers

## 2. Corruptions: `shapefit/degrade.py`

![corruptions](images/corruptions.png)

Every corruption is a named entry in `GEOMETRIC` or `RASTER` with a base
probability. A per-sample `severity ~ clip(Gamma(4, 0.25), 0.05, 2)` (mean 1)
scales the amplitudes and probabilities, so most samples are moderate, a few are
almost clean, and some are harsh.

**Geometric** (on a 600-point outline, displaced along the outward normal; units = shape size):

| name | p | effect |
|---|---|---|
| `round_corners` | 0.35 | morphological buffer in/out, radius 1–5 % |
| `jagged` | 0.75 | white noise, smoothed 0–2.5 samples, σ 0.2–1.8 % |
| `bumps` | 0.65 | 1–5 Gaussian bumps/dents, amplitude 1.5–7 %, width 1.5–9 % |
| `wobble` | 0.40 | 1–3 sinusoids, 2–8 cycles per perimeter, amplitude 0.3–1.5 % |
| `staircase` | 0.20 | piecewise-constant steps (misregistered tiles) |

**Raster** (on the binary mask; radii given for a 100 px shape and scaled):

| name | p | effect |
|---|---|---|
| `bites` | 0.50 | 1–4 circular notches out of the edge |
| `gaps` | 0.30 | a 4–15 % stretch of edge eroded inward |
| `blobs` | 0.40 | 1–3 ellipses stuck onto the edge |
| `cuts` | 0.15 | a thin background line across the shape |
| `holes` | 0.25 | 1–4 interior holes |
| `speckle` | 0.30 | random pixel flips in a band around the edge |
| `specks` | 0.15 | tiny distractor dots anywhere |
| `pixelate` | 0.25 | down/up-sample by 2–6×: staircase edges |
| `blur` | 0.30 | Gaussian blur, then a random threshold |
| `morph` | 0.20 | 3×3 open or close, 1–2 iterations |

You can force an exact subset with `only=[...]`, e.g.
`raster_corrupt(mask, rng, 92, only=["bites", "gaps"])`.

## 3. Framing: `shapefit/frame.py`

The network works on a fixed **128 × 128** canvas. The shape's bounding box is
centred, and the canvas is scaled uniformly so its longest side covers **72 %**
of the canvas. Training uses the *corrupted* outline's bbox, which is what
inference sees. It also adds ±8 % scale jitter and ±2 px shift jitter, so the
model doesn't depend on the framing being exact. At inference, components
smaller than 2 % of the largest are ignored when computing the bbox, so stray
specks don't shrink the shape.

## 4. Targets

* **clean mask**: the polygon rasterised with 4× supersampling and a 50 %
  coverage rule. Plain `cv2.fillPoly` is biased about 0.45 px outward, which
  would cost about 2 % IoU.
* **vertex heatmap**: a Gaussian with σ = 1.5 px at every vertex. The nearest
  pixel is set to exactly 1.0, as the positive for the CenterNet focal loss.

## 5. Network and loss: `shapefit/model.py`

A U-Net (base 32 channels, depth 4, GroupNorm + SiLU, **7.8 M params**). The
input is the mask plus two CoordConv channels. The output is two logit maps:

* `mask`: BCE + soft Dice
* `vertices`: penalty-reduced focal loss (CenterNet, α = 2, β = 4); the head bias starts at −4

## 6. Running the full training

```bash
# on-the-fly stream feeding the GPU (recommended; this made the shipped checkpoint)
python -m shapefit.train --steps 120000 --batch 64 --workers 28 --eval-every 5000 --out runs/full

# or materialise a fixed dataset first (≈400 B/sample bit-packed: 200k ≈ 80 MB)
python -m shapefit.make_dataset --n 200000 --out data/train --seed 0
python -m shapefit.make_dataset --n 5000   --out data/val   --seed 10000
python -m shapefit.train --data data/train --steps 30000 --out runs/shards
```

* AdamW, 500-step warm-up, then cosine decay; gradients clipped to norm 1.
* On CUDA the step runs under bf16 autocast and `torch.compile` (turn them off
  with `--no-amp` / `--no-compile`). Together they make a step about 1.7× faster.
* Every `--eval-every` steps, the run vectorises a fixed set of 500 validation
  samples (seed 10000) and logs IoU, vertex-count accuracy, vertex F1, vertex
  error and Hausdorff distance to `log.jsonl`. It also scores 200 `rect_union`
  samples (seed 10001) and logs them with a `union_` prefix. It keeps `best.pt`
  (by IoU + count accuracy on the main set) and `last.pt`.
* Throughput on one RTX 4090: about 30 ms/step at batch 64 (bf16 + compile),
  so the shipped 120k-step run took 74 min including its 24 checks. 28 data
  workers produce a batch in about 14 ms, so the GPU sets the speed.
* Copy the result to `shapefit/weights/shapefit.pt` so that `fit_polygon()`
  without arguments picks it up.

### Shard format (`make_dataset`)

`shard_XXXXX.npz` holds `noisy`, `mask` (bit-packed `(n, 2048)` uint8),
`vertices` (`(n, 32, 2)` float32, padded with NaN), `nvert`, `family` and
`size`. Heatmaps are re-rendered from the vertices when a sample is loaded.
`ShardDataset` reads the shards and is a drop-in replacement for the stream.

### Sanity check: overfitting

```bash
python -m shapefit.train --overfit 48 --batch 32 --steps 2500 --eval-every 500 --out runs/overfit
```

This trains and evaluates on the same 48 full-resolution scenes
(`scene_set(48, seed=7)`). The scenes go through the same `frame_mask` path as
inference. If this doesn't reach the vectoriser's oracle ceiling (IoU ≈ 0.994),
something is wrong in the data, the targets or the vectoriser.

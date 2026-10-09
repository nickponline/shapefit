"""Regenerate every figure and table in the README.

    python examples/make_readme_figures.py --weights shapefit/weights/overfit_demo.pt

The demo checkpoint was trained with ``--overfit 48`` (see README): the
"results" figures show scenes it has memorised, which demonstrates that the
pipeline works end to end, not that it generalises.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np

from shapefit import ShapeFitter, baselines
from shapefit.data import SIZE, make_sample, render_polygon
from shapefit.degrade import GEOMETRIC, RASTER, geometric_corrupt, raster_corrupt
from shapefit.metrics import iou, score, summarize
from shapefit.scenes import scene_set
from shapefit.shapes import random_polygon

OUT = Path(__file__).parents[1] / "docs" / "images"
GREEN, ORANGE, RED, BLUE = "#1a9e5c", "#e8833a", "#d62f3a", "#2f6fd6"


def closed(v):
    return np.vstack([v, v[:1]])


def draw_poly(ax, v, color, lw=2.0, ls="-", dots=True, label=None):
    if len(v) < 2:
        return
    c = closed(v)
    ax.plot(c[:, 0], c[:, 1], ls, color=color, lw=lw, label=label)
    if dots:
        ax.plot(v[:, 0], v[:, 1], "o", color=color, ms=3.5)


def show_mask(ax, m, title=None):
    ax.imshow(m, cmap="gray_r", vmin=0, vmax=1.6, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)


def crop(ax, v, img_shape, pad=0.18):
    lo, hi = v.min(0), v.max(0)
    ext = (hi - lo).max() * (1 + 2 * pad)
    c = (lo + hi) / 2
    ax.set_xlim(c[0] - ext / 2, c[0] + ext / 2)
    ax.set_ylim(c[1] + ext / 2, c[1] - ext / 2)


# ------------------------------------------------------------------ figures


def fig_corruptions():
    rng = np.random.default_rng(4)
    poly, _ = random_polygon(rng, "rectilinear")
    to_px = lambda p: p * SIZE * 0.72 * np.array([1, -1]) + (SIZE - 1) / 2
    clean = render_polygon(to_px(poly), SIZE)
    panels = [("clean", clean)]
    for name in ["round_corners", *GEOMETRIC]:
        r = np.random.default_rng(10)
        dense = geometric_corrupt(poly, r, severity=1.4, only=[name])
        panels.append((name, render_polygon(to_px(dense), SIZE)))
    for name in RASTER:
        # illustrative: show the most visible of a few random draws
        outs = [raster_corrupt(clean, np.random.default_rng(k), SIZE * 0.72, 2.0, only=[name]) for k in range(30)]
        panels.append((name, max(outs, key=lambda m: int((m != clean).sum()) if (m & clean).sum() > 0.7 * clean.sum() else -1)))
    r = np.random.default_rng(3)
    for k in range(2):
        dense = geometric_corrupt(poly, r, severity=1.2)
        panels.append((f"random mix {k + 1}", raster_corrupt(render_polygon(to_px(dense), SIZE), r, 92, 1.2)))
    cols = 9
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.05))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (name, m) in zip(axes.flat, panels):
        show_mask(ax, m, name)
        draw_poly(ax, to_px(poly), RED, lw=0.8, dots=False)
        ax.axis("on")
    fig.tight_layout()
    fig.savefig(OUT / "corruptions.png", dpi=110)
    plt.close(fig)


def fig_dataset():
    rng = np.random.default_rng(21)
    samples = [make_sample(rng) for _ in range(8)]
    fig, axes = plt.subplots(3, 8, figsize=(16, 6.3))
    for i, s in enumerate(samples):
        show_mask(axes[0, i], s.noisy, s.family)
        show_mask(axes[1, i], s.mask)
        draw_poly(axes[1, i], s.vertices, RED, lw=1.2)
        axes[2, i].imshow(s.heatmap, cmap="magma", interpolation="nearest")
        axes[2, i].axis("off")
    for r, lab in enumerate(["input (corrupted)", "target: clean mask + polygon", "target: vertex heatmap"]):
        axes[r, 0].set_ylabel(lab, fontsize=10)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "dataset.png", dpi=100)
    plt.close(fig)


def fig_pipeline(fitter, scene):
    r = fitter.fit_detailed(scene.image)
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
    show_mask(axes[0], scene.image, f"1. input mask {scene.image.shape[1]}x{scene.image.shape[0]}")
    show_mask(axes[1], r.canvas, f"2. canonical {SIZE}x{SIZE} canvas")
    axes[2].imshow(r.clean_mask, cmap="gray_r", vmin=0, vmax=1.6)
    axes[2].set_title("3. predicted clean mask", fontsize=10)
    axes[3].imshow(cv2.dilate(r.vertex_heatmap, np.ones((5, 5), np.uint8)), cmap="magma")
    axes[3].set_title("4. vertex heatmap (dilated for display)", fontsize=10)
    show_mask(axes[4], scene.image, "5. fitted polygon (input coords)")
    draw_poly(axes[4], r.polygon, GREEN, lw=2.2)
    crop(axes[4], scene.vertices, scene.image.shape)
    for ax in axes[2:4]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "pipeline.png", dpi=100)
    plt.close(fig)


def fig_results(fitter, scenes, n=12):
    fig, axes = plt.subplots(2, n // 2, figsize=(n // 2 * 2.9, 6.4))
    for ax, sc in zip(axes.flat, scenes[:n]):
        nn = fitter.fit(sc.image)
        dp = baselines.douglas_peucker(sc.image, 0.02)
        show_mask(ax, sc.image)
        draw_poly(ax, dp, ORANGE, lw=1.6, ls="--", dots=False)
        draw_poly(ax, nn, GREEN, lw=2.0)
        crop(ax, sc.vertices, sc.image.shape)
        ax.set_title(f"IoU  NN {iou(nn, sc.vertices):.3f} · DP {iou(dp, sc.vertices):.3f}", fontsize=8.5)
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([], [], color=GREEN, lw=2, marker="o", ms=4, label="shapefit (NN)"),
            Line2D([], [], color=ORANGE, lw=1.6, ls="--", label="cv2.approxPolyDP baseline"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "results.png", dpi=100)
    plt.close(fig)


def fig_regularize(fitter, scenes):
    rect = [s for s in scenes if s.family == "rectilinear"]
    sc = max(rect, key=lambda s: len(s.vertices))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, mode in zip(axes, [None, "orthogonal", "parallel"]):
        v = fitter.fit(sc.image, regularize=mode)
        show_mask(ax, sc.image, f"regularize={mode!r}")
        draw_poly(ax, v, GREEN if mode is None else BLUE, lw=2)
        crop(ax, sc.vertices, sc.image.shape)
    fig.tight_layout()
    fig.savefig(OUT / "regularize.png", dpi=100)
    plt.close(fig)


# ------------------------------------------------------------------- tables


def table(fitter, scenes):
    methods = {
        "shapefit (NN)": lambda im: fitter.fit(im),
        "shapefit (NN) + TTA": lambda im: fitter.fit(im, tta=True),
        "approxPolyDP 1%": lambda im: baselines.douglas_peucker(im, 0.01),
        "approxPolyDP 2%": lambda im: baselines.douglas_peucker(im, 0.02),
        "approxPolyDP 3%": lambda im: baselines.douglas_peucker(im, 0.03),
        "convex hull + DP": lambda im: baselines.convex_hull(im, 0.01),
        "min-area rectangle": lambda im: baselines.min_area_rect(im),
    }
    res = {}
    for name, f in methods.items():
        res[name] = summarize([score(f(sc.image), sc.vertices) for sc in scenes])
    return res


def fmt_table(res):
    lines = [
        "| method | IoU ↑ | vertex count exact ↑ | vertex F1 @3% ↑ | vertex err ↓ | Hausdorff ↓ |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in res.items():
        lines.append(
            f"| {name} | {m['iou']:.3f} | {m['count_ok']:.1%} | {m['vertex_f1']:.3f} | "
            f"{m['vertex_err']:.2%} | {m['hausdorff']:.2%} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="shapefit/weights/overfit_demo.pt")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    fitter = ShapeFitter(args.weights)
    n, seed = fitter.meta.get("overfit"), fitter.meta.get("overfit_seed")
    memorised = scene_set(n, seed=seed)
    unseen = scene_set(200, seed=999)

    fig_corruptions()
    fig_dataset()
    fig_pipeline(fitter, memorised[1])
    fig_results(fitter, memorised)
    fig_regularize(fitter, memorised)

    md = ["## memorised scenes (training set of the overfit run)", fmt_table(table(fitter, memorised))]
    md += ["", "## 200 unseen scenes (overfit model; expected to be poor)", fmt_table(table(fitter, unseen))]
    (OUT.parent / "metrics.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

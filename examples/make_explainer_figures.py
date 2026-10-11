"""Figures used only by docs/explainer.html.

    python examples/make_explainer_figures.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shapefit import ShapeFitter, baselines, iou
from shapefit.data import fixed_set
from shapefit.scenes import scene_set
from shapefit.vectorize import _fit_line, _largest_contour, find_peaks

OUT = Path(__file__).parents[1] / "docs" / "images"
GREEN, ORANGE, RED, BLUE, GREY = "#1a9e5c", "#e8833a", "#d62f3a", "#2f6fd6", "#9aa3ad"


def closed(v):
    return np.vstack([v, v[:1]])


def base(ax, img, title):
    ax.imshow(img, cmap="gray_r", vmin=0, vmax=1.6, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def zoom(ax, pts, pad=0.12):
    lo, hi = pts.min(0), pts.max(0)
    e = (hi - lo).max() * (0.5 + pad)
    c = (lo + hi) / 2
    ax.set_xlim(c[0] - e, c[0] + e)
    ax.set_ylim(c[1] + e, c[1] - e)


def vectorizer_steps(fitter, scene):
    r = fitter.fit_detailed(scene.image)
    S, up = r.canvas.shape[0], 4
    logit = np.log(np.clip(r.clean_mask, 1e-6, 1 - 1e-6) / np.clip(1 - r.clean_mask, 1e-6, 1))
    big = cv2.resize(logit.astype(np.float32), (S * up, S * up), interpolation=cv2.INTER_LINEAR)
    contour = (_largest_contour(big > 0) + 0.5) / up - 0.5
    peaks, _ = find_peaks(r.vertex_heatmap, 0.25)
    d = np.linalg.norm(contour[None] - peaks[:, None], axis=2)
    idx = d.argmin(1)
    order = np.argsort(idx)
    peaks, idx = peaks[order], idx[order]
    poly = r.canvas_polygon
    gt = r.frame.to_canvas(scene.vertices)

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.6))
    base(axes[0], r.canvas, "A. Input on the canvas")
    base(axes[1], r.canvas, "B. Predicted clean outline")
    axes[1].plot(*closed(contour).T, color=BLUE, lw=1.6)
    axes[2].imshow(cv2.dilate(r.vertex_heatmap, np.ones((3, 3), np.uint8)), cmap="magma")
    axes[2].plot(*peaks.T, "o", mfc="none", mec="white", ms=11, mew=1.5)
    axes[2].set_title("C. Corner heatmap and its peaks", fontsize=11)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    base(axes[3], r.canvas, "D. Peaks put on the outline, in order")
    axes[3].plot(*closed(contour).T, color=BLUE, lw=1.2, alpha=0.7)
    for k, (p, i) in enumerate(zip(peaks, idx)):
        q = contour[i]
        axes[3].plot(*q, "o", color=ORANGE, ms=7)
        axes[3].annotate(str(k + 1), q, xytext=(6, -10), textcoords="offset points", fontsize=11,
                         color=ORANGE, weight="bold")
    base(axes[4], r.canvas, "E. Final polygon (green), truth (red dash)")
    axes[4].plot(*closed(gt).T, "--", color=RED, lw=1.4)
    axes[4].plot(*closed(poly).T, "-", color=GREEN, lw=2)
    axes[4].plot(*poly.T, "o", color=GREEN, ms=4)
    for ax in axes:
        zoom(ax, gt)
    fig.tight_layout()
    fig.savefig(OUT / "explainer_vectorize_steps.png", dpi=100)
    plt.close(fig)

    # close-up of the line fit at one corner
    L = len(contour)
    k = int(np.argmax([np.linalg.norm(peaks[(j + 1) % len(peaks)] - peaks[j]) for j in range(len(peaks))]))
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    base(ax, r.canvas, "")
    corner = (k + 1) % len(peaks)
    for e, col in ((k, GREEN), (corner, BLUE)):
        a, b = int(idx[e]), int(idx[(e + 1) % len(idx)])
        seg = np.arange(a, b if b > a else b + L) % L
        m = len(seg)
        t = max(int(round(m * 0.15)), up * 2)
        ax.plot(*contour[seg].T, ".", color=GREY, ms=3)
        inner = contour[seg[t: m - t]]
        ax.plot(*inner.T, ".", color=col, ms=4)
        c, dvec = _fit_line(inner)
        ts = np.linspace(-60, 60, 2)
        ax.plot(c[0] + ts * dvec[0], c[1] + ts * dvec[1], "--", color=col, lw=1.3)
    ax.plot(*peaks[corner], "x", color=ORANGE, ms=13, mew=2.5)
    vi = np.argmin(np.linalg.norm(poly - peaks[corner], axis=1))
    ax.plot(*poly[vi], "o", color=GREEN, ms=10, mfc="none", mew=2.5)
    c0 = peaks[corner]
    ax.set_xlim(c0[0] - 16, c0[0] + 16)
    ax.set_ylim(c0[1] + 14, c0[1] - 14)
    ax.set_title("Edge lines meet at the corner", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "explainer_line_fit.png", dpi=110)
    plt.close(fig)


def dp_failure(fitter, scene):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    runs = [
        ("Douglas–Peucker, ε = 0.5 %", baselines.douglas_peucker(scene.image, 0.005), ORANGE),
        ("Douglas–Peucker, ε = 2 %", baselines.douglas_peucker(scene.image, 0.02), ORANGE),
        ("Douglas–Peucker, ε = 5 %", baselines.douglas_peucker(scene.image, 0.05), ORANGE),
        ("shapefit", fitter.fit(scene.image), GREEN),
    ]
    for ax, (name, v, col) in zip(axes, runs):
        base(ax, scene.image, f"{name}\n{len(v)} corners · IoU {iou(v, scene.vertices):.3f}")
        ax.plot(*closed(scene.vertices).T, "--", color=RED, lw=1.2)
        ax.plot(*closed(v).T, "-", color=col, lw=2)
        ax.plot(*v.T, "o", color=col, ms=4)
        zoom(ax, scene.vertices, pad=0.18)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "explainer_dp_failure.png", dpi=100)
    plt.close(fig)


def heatmap_target():
    s = fixed_set(1, seed=3, family="quad")[0]
    v = s.vertices[0]
    xi, yi = int(round(v[0])), int(round(v[1]))
    r = 5
    crop = s.heatmap[yi - r: yi + r + 1, xi - r: xi + r + 1]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [1, 1.1]})
    axes[0].imshow(s.heatmap, cmap="magma")
    axes[0].plot(*s.vertices.T, "o", mfc="none", mec="white", ms=12)
    axes[0].add_patch(plt.Rectangle((xi - r - 0.5, yi - r - 0.5), 2 * r + 1, 2 * r + 1, fill=False, ec="#4fc3f7", lw=2))
    axes[0].set_title("Target heatmap: one spot per corner", fontsize=11)
    axes[1].imshow(crop, cmap="magma", vmin=0, vmax=1)
    for (y, x), val in np.ndenumerate(crop):
        if val > 0.01:
            axes[1].text(x, y, f"{val:.2f}".lstrip("0") if val < 1 else "1", ha="center", va="center",
                         fontsize=7.5, color="black" if val > 0.6 else "white")
    axes[1].set_title("Close-up: values fall off from the corner", fontsize=11)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "explainer_heatmap_target.png", dpi=100)
    plt.close(fig)


def main():
    fitter = ShapeFitter("shapefit/weights/shapefit.pt")
    scenes = scene_set(500, seed=999)  # unseen during training, same set as the README tables

    # illustrative picks, by a fixed rule: shapefit gets every corner, and the shape
    # shows the point of the figure. The README tables give the averages.
    def clean_fit(s, min_iou):
        v = fitter.fit(s.image)
        return len(v) == len(s.vertices) and iou(v, s.vertices) > min_iou

    star = next(s for s in scenes if s.family == "star" and len(s.vertices) >= 7 and clean_fit(s, 0.97))
    dp = next(
        s for s in scenes
        if s.family in ("quad", "ngon") and 0.9 < s.severity and clean_fit(s, 0.975)
        and len(baselines.douglas_peucker(s.image, 0.02)) != len(s.vertices)
    )
    vectorizer_steps(fitter, star)
    dp_failure(fitter, dp)
    heatmap_target()


if __name__ == "__main__":
    main()

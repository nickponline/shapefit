"""Build the data for the explainer's gallery of unseen examples.

    python examples/make_gallery.py --weights shapefit/weights/shapefit.pt

For each shape family, the first ``--per-family`` scenes of a fresh seed
(2026 + family index) are fitted. Nothing is filtered or re-ordered, so the
gallery shows the good and the bad results alike. The result is written as JSON
into the ``<script id="gallery-data">`` block of ``docs/explainer.html``.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

import cv2
import numpy as np

from shapefit import ShapeFitter, baselines
from shapefit.metrics import score
from shapefit.scenes import scene_set
from shapefit.shapes import FAMILIES

HTML = Path(__file__).parents[1] / "docs" / "explainer.html"
PX = 220  # thumbnail size
PAD = 0.16


def thumb(scene):
    """Square crop around the true polygon, as a PNG data URI, plus the crop transform."""
    v = scene.vertices
    lo, hi = v.min(0), v.max(0)
    ext = (hi - lo).max() * (1 + 2 * PAD)
    c = (lo + hi) / 2
    s = PX / ext
    M = np.array([[s, 0, PX / 2 - s * c[0]], [0, s, PX / 2 - s * c[1]]], np.float32)
    img = cv2.warpAffine(scene.image.astype(np.float32), M, (PX, PX), flags=cv2.INTER_AREA)
    grey = (255 - np.clip(img, 0, 1) * 120).astype(np.uint8)  # shape light grey, background white
    ok, png = cv2.imencode(".png", grey, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    assert ok
    to_crop = lambda p: [[round(float(x * s + M[0, 2]), 1), round(float(y * s + M[1, 2]), 1)] for x, y in p]
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode(), to_crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="shapefit/weights/shapefit.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--per-family", type=int, default=8)
    args = ap.parse_args()

    fitter = ShapeFitter(args.weights, device=args.device)
    items = []
    for k, fam in enumerate(FAMILIES):
        for i, sc in enumerate(scene_set(args.per_family, seed=2026 + k, family=fam)):
            nn = fitter.fit(sc.image)
            dp = baselines.douglas_peucker(sc.image, 0.02)
            a, b = score(nn, sc.vertices), score(dp, sc.vertices)
            src, to_crop = thumb(sc)
            items.append({
                "id": f"{fam}-{i}", "family": fam, "seed": 2026 + k, "index": i,
                "severity": round(sc.severity, 2), "img": src,
                "truth": to_crop(sc.vertices), "nn": to_crop(nn), "dp": to_crop(dp),
                "nn_iou": round(a["iou"], 3), "dp_iou": round(b["iou"], 3),
                "nn_f1": round(a["vertex_f1"], 3), "dp_f1": round(b["vertex_f1"], 3),
            })
            print(f"{fam:12s} {i}  sev {sc.severity:.2f}  corners {len(sc.vertices):2d}/{len(nn):2d}/{len(dp):2d}"
                  f"  IoU {a['iou']:.3f} vs {b['iou']:.3f}")

    blob = json.dumps({"px": PX, "items": items}, separators=(",", ":"))
    html = HTML.read_text()
    pat = re.compile(r'(<script id="gallery-data" type="application/json">).*?(</script>)', re.S)
    assert pat.search(html), "no gallery-data block in explainer.html"
    HTML.write_text(pat.sub(lambda m: m.group(1) + blob + m.group(2), html))
    nn_ok = np.mean([len(it["nn"]) == len(it["truth"]) for it in items])
    dp_ok = np.mean([len(it["dp"]) == len(it["truth"]) for it in items])
    print(f"{len(items)} scenes, {len(blob) / 1e3:.0f} kB | count exact: shapefit {nn_ok:.1%}, DP {dp_ok:.1%}"
          f" | mean IoU: shapefit {np.mean([i['nn_iou'] for i in items]):.3f},"
          f" DP {np.mean([i['dp_iou'] for i in items]):.3f}")


if __name__ == "__main__":
    main()

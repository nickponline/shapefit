"""Shrink a training checkpoint for shipping: fp16 weights, metadata kept.

    python examples/export_weights.py runs/overfit/best.pt shapefit/weights/overfit_demo.pt
"""

import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
ckpt = torch.load(src, map_location="cpu", weights_only=False)
ckpt["state_dict"] = {k: v.half() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()}
torch.save(ckpt, dst)
print(f"wrote {dst} (step {ckpt.get('step')}, metrics {ckpt.get('metrics')})")

"""Train ShapeFitNet.

    # full run, infinite on-the-fly synthetic stream (recommended)
    python -m shapefit.train --steps 30000 --out runs/base
    # full run from pre-generated shards (python -m shapefit.make_dataset ...)
    python -m shapefit.train --data data/train --steps 30000 --out runs/shards
    # memorise a small fixed set (sanity check / README illustrations)
    python -m shapefit.train --overfit 48 --steps 3000 --out runs/overfit
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import SyntheticShapes, fixed_set
from .make_dataset import ShardDataset
from .scenes import scene_set, scene_to_sample
from .metrics import score, summarize
from .model import ShapeFitNet, focal_heatmap_loss, mask_loss
from .vectorize import polygon_from_outputs


class _Repeat(Dataset):
    """A fixed list of samples, repeated to fill an epoch."""

    def __init__(self, samples, length: int = 100_000):
        self.samples, self.length = samples, length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        s = self.samples[i % len(self.samples)]
        return (
            torch.from_numpy(s.noisy[None].astype(np.float32)),
            torch.from_numpy(s.mask[None].astype(np.float32)),
            torch.from_numpy(s.heatmap[None]),
        )


def _cycle(loader):
    while True:
        yield from loader


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, samples, device, batch: int = 128) -> dict:
    model.eval()
    rows = []
    for i in range(0, len(samples), batch):
        chunk = samples[i : i + batch]
        x = torch.from_numpy(np.stack([s.noisy for s in chunk])[:, None].astype(np.float32)).to(device)
        out = model(x).float().cpu().numpy()
        for s, o in zip(chunk, out):
            rows.append(score(polygon_from_outputs(o[0], o[1]), s.vertices))
    model.train()
    return summarize(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--heat-weight", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true", help="disable bf16 autocast on CUDA")
    ap.add_argument("--no-compile", action="store_true", help="disable torch.compile on CUDA")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/base")
    ap.add_argument("--data", default=None, help="shard dir from make_dataset (default: on-the-fly)")
    ap.add_argument("--overfit", type=int, default=0, help="train AND eval on N fixed samples")
    ap.add_argument("--overfit-seed", type=int, default=7)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    model = ShapeFitNet(base=args.base).to(device)
    amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    net = torch.compile(model) if device.type == "cuda" and not args.no_compile else model
    print(f"device={device} amp={amp} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)),
    )
    val = fixed_set(500, seed=10_000)
    val_union = [] if args.overfit else fixed_set(200, seed=10_001, family="rect_union")
    if args.overfit:
        # full-resolution scenes pushed through the public preprocessing, so the
        # memorised examples can be reproduced with ShapeFitter.fit(scene.image)
        val = [scene_to_sample(sc) for sc in scene_set(args.overfit, seed=args.overfit_seed)]
        loader = DataLoader(_Repeat(val), batch_size=args.batch, shuffle=True, num_workers=0)
    elif args.data:
        loader = DataLoader(
            ShardDataset(args.data), batch_size=args.batch, shuffle=True,
            num_workers=args.workers, persistent_workers=args.workers > 0, drop_last=True,
        )
    else:
        loader = DataLoader(
            SyntheticShapes(seed=args.seed),
            batch_size=args.batch,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            prefetch_factor=4 if args.workers > 0 else None,
        )
    log = open(out / "log.jsonl", "a")
    best = -1.0
    t0 = time.time()
    run = {"loss": 0.0, "mask": 0.0, "heat": 0.0}
    for step, (x, m, h) in enumerate(_cycle(loader), 1):
        # NB: no non_blocking copies (silently corrupts data on MPS)
        x, m, h = x.to(device), m.to(device), h.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = net(x)
        logits = logits.float()
        lm = mask_loss(logits[:, :1], m)
        lh = focal_heatmap_loss(logits[:, 1:], h)
        loss = lm + args.heat_weight * lh
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in (("loss", loss), ("mask", lm), ("heat", lh)):
            run[k] += v.item()
        if step % 100 == 0:
            avg = {k: v / 100 for k, v in run.items()}
            run = {k: 0.0 for k in run}
            print(
                f"step {step:6d} loss {avg['loss']:.4f} mask {avg['mask']:.4f} heat {avg['heat']:.4f} "
                f"lr {sched.get_last_lr()[0]:.2e} {(time.time()-t0)/step*1000:.0f}ms/it",
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, val, device)
            if val_union:
                metrics.update({f"union_{k}": v for k, v in evaluate(model, val_union, device).items()})
            print(f"  eval @ {step}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
            log.write(json.dumps({"step": step, **metrics}) + "\n")
            log.flush()
            ckpt = {
                "config": model.config, "state_dict": model.state_dict(), "step": step,
                "metrics": metrics, "overfit": args.overfit, "overfit_seed": args.overfit_seed,
            }
            torch.save(ckpt, out / "last.pt")
            if metrics["iou"] + metrics["count_ok"] > best:
                best = metrics["iou"] + metrics["count_ok"]
                torch.save(ckpt, out / "best.pt")
        if step >= args.steps:
            break


if __name__ == "__main__":
    main()

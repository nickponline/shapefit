"""Pre-generate a fixed synthetic dataset to disk as bit-packed ``.npz`` shards.

    python -m shapefit.make_dataset --n 200000 --out data/train --seed 0
    python -m shapefit.make_dataset --n 5000   --out data/val   --seed 10000

Each shard ``shard_XXXXX.npz`` holds:

* ``noisy``    (n, S*S/8) uint8  - bit-packed corrupted input masks
* ``mask``     (n, S*S/8) uint8  - bit-packed clean target masks
* ``vertices`` (n, 32, 2) float32 - clean polygon vertices (canvas pixels), NaN padded
* ``nvert``    (n,) int16
* ``family``   (n,) <U12

Heatmaps are not stored; they are re-rendered from ``vertices`` on load (cheap).
On-the-fly generation (``SyntheticShapes``) is equally valid and gives infinite
variety; shards exist for reproducibility, sharing and fast non-Python loading.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import SIZE, Sample, make_sample, vertex_heatmap

MAX_VERTS = 32


def _pack(m: np.ndarray) -> np.ndarray:
    return np.packbits(m.astype(bool).reshape(-1))


def _unpack(b: np.ndarray, size: int) -> np.ndarray:
    return np.unpackbits(b)[: size * size].reshape(size, size)


def _write_shard(args) -> str:
    path, n, seed, size = args
    rng = np.random.default_rng(seed)
    noisy, mask, verts, nv, fam = [], [], [], [], []
    for _ in range(n):
        s = make_sample(rng, size)
        noisy.append(_pack(s.noisy))
        mask.append(_pack(s.mask))
        v = np.full((MAX_VERTS, 2), np.nan, np.float32)
        v[: len(s.vertices)] = s.vertices[:MAX_VERTS]
        verts.append(v)
        nv.append(len(s.vertices))
        fam.append(s.family)
    np.savez_compressed(
        path,
        noisy=np.stack(noisy),
        mask=np.stack(mask),
        vertices=np.stack(verts),
        nvert=np.array(nv, np.int16),
        family=np.array(fam),
        size=np.int32(size),
    )
    return str(path)


class ShardDataset(Dataset):
    """Map-style dataset over shards written by this module."""

    def __init__(self, root: str | Path):
        files = sorted(Path(root).glob("shard_*.npz"))
        if not files:
            raise FileNotFoundError(f"no shards in {root}")
        parts = [np.load(f) for f in files]
        self.size = int(parts[0]["size"])
        self.noisy = np.concatenate([p["noisy"] for p in parts])
        self.mask = np.concatenate([p["mask"] for p in parts])
        self.vertices = np.concatenate([p["vertices"] for p in parts])
        self.nvert = np.concatenate([p["nvert"] for p in parts])
        self.family = np.concatenate([p["family"] for p in parts])

    def __len__(self):
        return len(self.noisy)

    def sample(self, i: int) -> Sample:
        v = self.vertices[i, : self.nvert[i]].astype(np.float64)
        return Sample(
            noisy=_unpack(self.noisy[i], self.size),
            mask=_unpack(self.mask[i], self.size),
            heatmap=vertex_heatmap(v, self.size).astype(np.float32),
            vertices=v,
            family=str(self.family[i]),
        )

    def __getitem__(self, i: int):
        s = self.sample(i)
        return (
            torch.from_numpy(s.noisy[None].astype(np.float32)),
            torch.from_numpy(s.mask[None].astype(np.float32)),
            torch.from_numpy(s.heatmap[None]),
        )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="total number of samples")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=10000)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for k, start in enumerate(range(0, args.n, args.shard_size)):
        n = min(args.shard_size, args.n - start)
        jobs.append((out / f"shard_{k:05d}.npz", n, [args.seed, k], args.size))
    with ProcessPoolExecutor(args.workers) as ex:
        for p in ex.map(_write_shard, jobs):
            print("wrote", p, flush=True)
    (out / "meta.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()

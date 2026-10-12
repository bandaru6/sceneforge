#!/usr/bin/env python3
"""Train and compare PointNet against a point Transformer on per-point labels.

The two architectures see the same data, the same split, the same batch size and
the same seed, so the only variable is how points exchange information. Mean IoU
is reported on a held-out test split that is never used for model selection, and
wall time per optimisation step is reported alongside it, because the
Transformer's accuracy is bought with compute and a comparison that hides the
price is not a comparison.

    python scripts/train_segmentation.py --data data/parts --arch pointnet --epochs 50
    python scripts/train_segmentation.py --data data/parts --arch transformer --epochs 50

Data layout: a directory of .npz files, one per point cloud, each containing
  points  (N, 3) or (N, 6) float, XYZ or XYZ+RGB
  labels  (N,) integer class ids in [0, n_classes)

The split is deterministic given the file list: sorted names, then 80/10/10 by
stride, so train, validation and test are fixed across runs and across
architectures without storing a split file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sceneforge.segmentation import (PointNetSeg, PointTransformerSeg,  # noqa: E402
                                     mean_iou)


class PointCloudParts(Dataset):
    """Fixed-size point clouds with per-point labels, read from .npz files.

    Every sample is resampled to exactly `n_points` so clouds can be batched:
    without replacement when the cloud is large enough, with replacement when it
    is not. Resampling happens per epoch rather than once, which acts as mild
    augmentation on the training split; `rng_seed` makes the validation and test
    splits draw the same points every time so their scores are comparable
    between epochs.
    """

    def __init__(self, paths: list[str], n_points: int = 2048,
                 n_classes: int = 5, fixed: bool = False, rng_seed: int = 0):
        self.paths = paths
        self.n_points = n_points
        self.n_classes = n_classes
        self.fixed = fixed
        self.rng_seed = rng_seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with np.load(self.paths[i]) as blob:
            if "points" not in blob or "labels" not in blob:
                raise KeyError(f"{self.paths[i]} needs 'points' and 'labels' arrays")
            points = np.asarray(blob["points"], dtype=np.float32)
            labels = np.asarray(blob["labels"]).astype(np.int64)
        if len(points) != len(labels):
            raise ValueError(f"{self.paths[i]}: {len(points)} points, {len(labels)} labels")
        if points.shape[1] == 3:
            # No colour in the file: pad with mid grey so the input width is the
            # six channels both models expect.
            points = np.hstack([points, np.full((len(points), 3), 0.5, np.float32)])
        if points.shape[1] != 6:
            raise ValueError(f"{self.paths[i]}: expected 3 or 6 columns")

        rng = np.random.default_rng(self.rng_seed + i if self.fixed else None)
        replace = len(points) < self.n_points
        idx = rng.choice(len(points), size=self.n_points, replace=replace)
        points, labels = points[idx], labels[idx]

        # Centre and scale to the unit sphere. Absolute position in the sensor
        # frame carries no information about which part a point belongs to, and
        # leaving it in makes the model memorise where objects happened to sit.
        centre = points[:, :3].mean(axis=0, keepdims=True)
        points[:, :3] -= centre
        scale = np.linalg.norm(points[:, :3], axis=1).max()
        if scale > 0:
            points[:, :3] /= scale
        if labels.max(initial=0) >= self.n_classes:
            raise ValueError(f"{self.paths[i]}: label {labels.max()} exceeds "
                             f"n_classes={self.n_classes}")
        return torch.from_numpy(points), torch.from_numpy(labels)


def split_paths(data_dir: str) -> dict[str, list[str]]:
    """Deterministic 80/10/10 train/val/test split by stride over sorted names."""
    names = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))
    if not names:
        raise FileNotFoundError(f"no .npz files in {data_dir}")
    paths = [os.path.join(data_dir, n) for n in names]
    val = paths[8::10]
    test = paths[9::10]
    train = [p for i, p in enumerate(paths) if i % 10 < 8]
    return {"train": train, "val": val, "test": test}


def build_model(arch: str, n_classes: int, knn_k: int = 16) -> torch.nn.Module:
    if arch == "pointnet":
        return PointNetSeg(in_dim=6, n_classes=n_classes)
    if arch == "transformer":
        return PointTransformerSeg(in_dim=6, n_classes=n_classes, k=knn_k)
    raise ValueError(f"unknown arch {arch!r}")


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device,
             n_classes: int) -> dict[str, float]:
    """Mean IoU and mean cross-entropy over a loader.

    Two IoUs come back. `miou` pools intersections and unions over the whole
    split before dividing, which is the number worth quoting. `miou_per_cloud`
    averages `sceneforge.segmentation.mean_iou` over clouds, which is what part
    segmentation papers usually report. They disagree when classes are unevenly
    spread across clouds, so both are returned and the gap stays visible instead
    of being a silent choice of denominator.
    """
    model.eval()
    inter = np.zeros(n_classes, dtype=np.int64)
    union = np.zeros(n_classes, dtype=np.int64)
    per_cloud: list[float] = []
    total_loss, n_batches = 0.0, 0
    for points, labels in loader:
        points, labels = points.to(device), labels.to(device)
        logits = model(points)
        total_loss += float(torch.nn.functional.cross_entropy(
            logits.reshape(-1, n_classes), labels.reshape(-1)))
        n_batches += 1
        for b in range(logits.shape[0]):
            value = mean_iou(logits[b], labels[b], n_classes)
            if not np.isnan(value):
                per_cloud.append(value)
        pred = logits.argmax(dim=-1)
        for c in range(n_classes):
            p, t = pred == c, labels == c
            inter[c] += int((p & t).sum())
            union[c] += int((p | t).sum())
    present = union > 0
    miou = float((inter[present] / union[present]).mean()) if present.any() else float("nan")
    return {"miou": miou,
            "miou_per_cloud": float(np.mean(per_cloud)) if per_cloud else float("nan"),
            "loss": total_loss / max(n_batches, 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="directory of .npz point clouds")
    ap.add_argument("--arch", choices=["pointnet", "transformer"], required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="keep this equal across architectures when comparing")
    ap.add_argument("--n-points", type=int, default=2048)
    ap.add_argument("--n-classes", type=int, default=5)
    ap.add_argument("--knn-k", type=int, default=16,
                    help="neighbourhood size for the Transformer arm")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0,
                    help="keep this equal across architectures when comparing")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda, mps, cpu; default autodetect")
    ap.add_argument("--out", default=None, help="write the run summary here as JSON")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    splits = split_paths(args.data)
    print(f"split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    loaders = {}
    for name, paths in splits.items():
        if not paths:
            raise ValueError(f"{name} split is empty; need at least 10 clouds")
        ds = PointCloudParts(paths, n_points=args.n_points,
                             n_classes=args.n_classes,
                             fixed=(name != "train"), rng_seed=args.seed)
        loaders[name] = DataLoader(ds, batch_size=args.batch_size,
                                   shuffle=(name == "train"),
                                   num_workers=args.workers, drop_last=False)

    model = build_model(args.arch, args.n_classes, args.knn_k).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"arch={args.arch} params={n_params} device={device} "
          f"batch_size={args.batch_size} seed={args.seed}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    step_times: list[float] = []
    best_val, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_steps = 0.0, 0
        for points, labels in loaders["train"]:
            points, labels = points.to(device), labels.to(device)
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            logits = model(points)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, args.n_classes), labels.reshape(-1))
            loss.backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
            running += float(loss)
            n_steps += 1
        sched.step()

        val = evaluate(model, loaders["val"], device, args.n_classes)
        # Model selection reads validation only. The test split is touched once,
        # after the loop, with the weights this line picked.
        if val["miou"] > best_val:
            best_val = val["miou"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d}  train_loss {running / max(n_steps, 1):.4f}  "
              f"val_loss {val['loss']:.4f}  val_mIoU {val['miou']:.4f}  "
              f"s/step {np.mean(step_times[-n_steps:]):.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, loaders["test"], device, args.n_classes)
    median_step = float(np.median(step_times)) if step_times else float("nan")

    summary = {
        "arch": args.arch,
        "params": n_params,
        "device": str(device),
        "batch_size": args.batch_size,
        "n_points": args.n_points,
        "seed": args.seed,
        "epochs": args.epochs,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "best_val_miou": best_val,
        "test_miou": test["miou"],
        "test_miou_per_cloud": test["miou_per_cloud"],
        "test_loss": test["loss"],
        "median_seconds_per_step": median_step,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

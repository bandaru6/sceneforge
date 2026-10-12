#!/usr/bin/env python3
"""Fit a radiance field to posed images and extract a mesh from it.

This is the stage that turns a video plus camera poses into geometry the
simulator can use. Rays from every training view are pooled and shuffled, so a
batch mixes viewpoints and the field is never optimised against one image at a
time; then density is surfaced on a grid and written as OBJ, which
`scripts/export_to_mujoco.py` takes as its input.

    python scripts/train_nerf.py --images data/frames --poses data/poses.npy \
        --intrinsics data/K.npy --iters 20000 --out-mesh results/scene.obj

Inputs:
  --images      directory of frames, sorted by filename, read with OpenCV
  --poses       .npy of (N, 4, 4) camera-to-world matrices, one per frame
  --intrinsics  .npy of the 3x3 pinhole matrix, or --focal with --downscale

One held-out view is kept back and its PSNR is reported during training, because
training-set PSNR on a radiance field only tells you the network memorised the
rays it was shown.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sceneforge.nerf import (RadianceField, extract_mesh, get_rays,  # noqa: E402
                             load_posed_images, psnr, render_rays,
                             scale_intrinsics, train_step)


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_ray_bank(images: np.ndarray, poses: np.ndarray, K: np.ndarray,
                   device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten every pixel of every view into one bank of (origin, dir, colour).

    Memory here is the real constraint: 100 views at 400x300 is 1.2e7 rays, and
    at 3 float32 each for origin, direction and colour that is about 430 MB. The
    bank is kept on the chosen device when it fits and on the CPU otherwise is
    the caller's problem; `--downscale` is the knob that keeps it small.
    """
    n, h, w = images.shape[:3]
    origins, directions = [], []
    for i in range(n):
        o, d = get_rays(h, w, K, poses[i])
        origins.append(o.reshape(-1, 3))
        directions.append(d.reshape(-1, 3))
    rays_o = torch.cat(origins).to(device)
    rays_d = torch.cat(directions).to(device)
    colours = torch.as_tensor(images.reshape(-1, 3), device=device)
    return rays_o, rays_d, colours


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True, help="directory of frames")
    ap.add_argument("--poses", required=True, help=".npy of (N, 4, 4) poses")
    ap.add_argument("--intrinsics", default=None, help=".npy of the 3x3 matrix")
    ap.add_argument("--focal", type=float, default=None,
                    help="focal length in pixels, if no intrinsics file exists")
    ap.add_argument("--downscale", type=int, default=4,
                    help="shrink frames by this factor before training")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch-rays", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--near", type=float, default=0.1)
    ap.add_argument("--far", type=float, default=10.0)
    ap.add_argument("--n-coarse", type=int, default=64)
    ap.add_argument("--n-fine", type=int, default=64)
    ap.add_argument("--white-background", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-mesh", default=None, help="write the extracted OBJ here")
    ap.add_argument("--mesh-resolution", type=int, default=128)
    ap.add_argument("--mesh-bound", type=float, default=2.0,
                    help="half-side of the cube the density grid covers")
    ap.add_argument("--mesh-level", type=float, default=10.0,
                    help="density isovalue treated as the surface")
    ap.add_argument("--ckpt", default=None, help="write field weights here")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    poses = np.load(args.poses).astype(np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (N, 4, 4), got {poses.shape}")
    images, poses = load_posed_images(args.images, poses, downscale=args.downscale)
    n, h, w = images.shape[:3]
    print(f"{n} views at {w}x{h} on {device}")

    if args.intrinsics:
        K = scale_intrinsics(np.load(args.intrinsics), args.downscale)
    elif args.focal:
        K = np.array([[args.focal / args.downscale, 0.0, w / 2.0],
                      [0.0, args.focal / args.downscale, h / 2.0],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
    else:
        raise SystemExit("need --intrinsics or --focal")

    # Hold back the last view entirely. It never contributes a training ray.
    held_out = n - 1 if n > 1 else 0
    train_idx = [i for i in range(n) if i != held_out] or [0]
    rays_o, rays_d, colours = build_ray_bank(
        images[train_idx], poses[train_idx], K, device)
    print(f"{len(rays_o)} training rays, holding out view {held_out}")

    field = RadianceField().to(device)
    opt = torch.optim.Adam(field.parameters(), lr=args.lr)
    render_kwargs = dict(near=args.near, far=args.far, n_coarse=args.n_coarse,
                         n_fine=args.n_fine, white_background=args.white_background)

    start = time.perf_counter()
    for it in range(1, args.iters + 1):
        sel = torch.randint(0, len(rays_o), (args.batch_rays,), device=device)
        loss = train_step(field, opt, rays_o[sel], rays_d[sel], colours[sel],
                          render_kwargs)
        if it % 500 == 0 or it == args.iters:
            rate = it / (time.perf_counter() - start)
            print(f"iter {it:6d}  loss {loss:.5f}  {rate:.1f} it/s")

    # Held-out view, rendered in chunks with jitter off so the result is
    # deterministic and directly comparable between runs.
    field.eval()
    o, d = get_rays(h, w, K, poses[held_out], device=device)
    o, d = o.reshape(-1, 3), d.reshape(-1, 3)
    target = torch.as_tensor(images[held_out].reshape(-1, 3), device=device)
    chunks = []
    with torch.no_grad():
        for s in range(0, len(o), 4096):
            sl = slice(s, s + 4096)
            chunks.append(render_rays(field, o[sl], d[sl], perturb=False,
                                      **render_kwargs)["rgb"])
    rendered = torch.cat(chunks)
    print(f"held-out view {held_out}: PSNR {psnr(rendered, target):.2f} dB")

    if args.ckpt:
        os.makedirs(os.path.dirname(os.path.abspath(args.ckpt)), exist_ok=True)
        torch.save({"state_dict": field.state_dict(), "args": vars(args)}, args.ckpt)
        print(f"wrote {args.ckpt}")

    if args.out_mesh:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_mesh)), exist_ok=True)
        verts, faces = extract_mesh(
            field, bounds=(-args.mesh_bound, args.mesh_bound),
            resolution=args.mesh_resolution, level=args.mesh_level,
            out_path=args.out_mesh, device=device)
        print(f"wrote {args.out_mesh}: {len(verts)} vertices, {len(faces)} triangles")
        if len(faces) == 0:
            print("the isosurface is empty; --mesh-level is above the density the "
                  "field learned, or --mesh-bound does not cover the scene",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

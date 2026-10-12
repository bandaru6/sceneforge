#!/usr/bin/env python3
"""Frame-to-frame LiDAR odometry over a scan sequence.

Loads a directory of scans, chains ICP to accumulate a trajectory, and if
ground truth poses are supplied reports absolute trajectory error against them.
No loop closure and no pose graph, so drift is expected to grow with sequence
length; that growth is the thing being measured.

    python scripts/run_odometry.py --scans data/scans --out results/trajectory.npy
    python scripts/run_odometry.py --scans data/scans --gt data/poses.txt \
        --max-scans 1024 --voxel 0.2

Scan formats accepted:
  .bin   KITTI Velodyne, flat float32 x y z reflectance (reflectance dropped)
  .npy   (N, 3) or (N, 4) array
  .txt   whitespace-separated, three or four columns

Ground truth is read as KITTI odometry poses: one line per frame, twelve
numbers forming the first three rows of a 4x4 camera-to-world matrix.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sceneforge.registration import (absolute_trajectory_error,  # noqa: E402
                                     accumulate_trajectory)

SCAN_SUFFIXES = (".bin", ".npy", ".txt")


def load_scan(path: str) -> np.ndarray:
    """Read one scan as an (N, 3) float32 array of XYZ in the sensor frame."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        pts = np.fromfile(path, dtype=np.float32)
        if pts.size % 4:
            raise ValueError(f"{path}: {pts.size} floats is not a multiple of 4")
        pts = pts.reshape(-1, 4)
    elif ext == ".npy":
        pts = np.load(path)
    elif ext == ".txt":
        pts = np.loadtxt(path, dtype=np.float32)
    else:
        raise ValueError(f"unsupported scan format {ext!r}")
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, pts.shape[-1])
    if pts.shape[1] < 3:
        raise ValueError(f"{path}: expected at least 3 columns, got {pts.shape[1]}")
    return np.ascontiguousarray(pts[:, :3])


def list_scans(directory: str, limit: int | None = None) -> list[str]:
    """Sorted scan paths. Lexicographic order is frame order for zero-padded names."""
    names = sorted(f for f in os.listdir(directory)
                   if os.path.splitext(f)[1].lower() in SCAN_SUFFIXES)
    if not names:
        raise FileNotFoundError(f"no {SCAN_SUFFIXES} files in {directory}")
    if limit is not None:
        names = names[:limit]
    return [os.path.join(directory, n) for n in names]


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """One representative point per occupied voxel.

    A raw 64-beam sweep is of order 10^5 points and ICP is dominated by the
    nearest-neighbour query, so downsampling is what makes a long sequence
    tractable. Taking the first point in each voxel rather than the centroid
    keeps real measurements instead of inventing new ones.
    """
    if voxel <= 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.ascontiguousarray(points[np.sort(first)])


def load_ground_truth(path: str, n: int | None = None) -> np.ndarray:
    """Read KITTI-style poses into (N, 4, 4) homogeneous matrices."""
    raw = np.loadtxt(path, dtype=np.float64)
    raw = np.atleast_2d(raw)
    if raw.shape[1] != 12:
        raise ValueError(f"{path}: expected 12 columns per pose, got {raw.shape[1]}")
    poses = np.repeat(np.eye(4)[None], len(raw), axis=0)
    poses[:, :3, :4] = raw.reshape(-1, 3, 4)
    return poses if n is None else poses[:n]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scans", required=True, help="directory of scan files")
    ap.add_argument("--out", default=None,
                    help="path to save the (N, 4, 4) trajectory as .npy")
    ap.add_argument("--gt", default=None,
                    help="ground-truth pose file for absolute trajectory error")
    ap.add_argument("--max-scans", type=int, default=None,
                    help="use only the first N scans")
    ap.add_argument("--voxel", type=float, default=0.2,
                    help="voxel size in metres for downsampling, 0 disables")
    ap.add_argument("--max-iter", type=int, default=50, help="ICP iterations per pair")
    ap.add_argument("--max-corr-dist", type=float, default=1.0,
                    help="reject correspondences beyond this distance, metres")
    args = ap.parse_args(argv)

    paths = list_scans(args.scans, args.max_scans)
    print(f"loading {len(paths)} scans from {args.scans}")
    scans = []
    for i, path in enumerate(paths):
        pts = voxel_downsample(load_scan(path), args.voxel)
        scans.append(pts)
        if i == 0:
            print(f"  first scan: {len(pts)} points after downsampling")
    if len(scans) < 2:
        print("need at least two scans to estimate motion", file=sys.stderr)
        return 2

    start = time.perf_counter()
    poses = accumulate_trajectory(scans, max_iter=args.max_iter,
                                  max_corr_dist=args.max_corr_dist)
    elapsed = time.perf_counter() - start
    path_length = float(np.linalg.norm(
        np.diff(poses[:, :3, 3], axis=0), axis=1).sum())
    print(f"registered {len(poses)} poses in {elapsed:.1f}s "
          f"({elapsed / max(len(poses) - 1, 1):.3f}s per pair)")
    print(f"estimated path length: {path_length:.2f} m")

    if args.gt:
        truth = load_ground_truth(args.gt, n=len(poses))
        if len(truth) != len(poses):
            print(f"ground truth has {len(truth)} poses but {len(poses)} were "
                  f"estimated; comparing the common prefix", file=sys.stderr)
            k = min(len(truth), len(poses))
            truth, compared = truth[:k], poses[:k]
        else:
            compared = poses
        # Odometry starts at the identity by construction, so it lives in the
        # frame of its own first scan. Anchoring to the first ground-truth pose
        # puts both trajectories in the same frame; it is a rigid change of
        # basis, not a fit, so it cannot flatter the error.
        compared = truth[0][None] @ compared
        ate = absolute_trajectory_error(compared, truth)
        print(f"absolute trajectory error over {len(compared)} poses: {ate:.3f} m")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.save(args.out, poses)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

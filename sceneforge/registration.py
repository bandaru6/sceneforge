"""Rigid registration: RANSAC-gated SVD fit, ICP, and trajectory accumulation."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def rigid_fit(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rotation and translation taking `src` onto `dst` (Kabsch).

    Reflection is corrected explicitly; without the determinant check a noisy
    correspondence set can produce an improper rotation that silently mirrors
    the cloud.
    """
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


def ransac_rigid_fit(src, dst, iters=200, threshold=0.2, rng=None):
    """Rigid fit robust to wrong correspondences.

    Samples minimal sets of three, scores by inlier count under `threshold`
    metres, then refits on the consensus set.
    """
    rng = rng or np.random.default_rng(0)
    n = len(src)
    best_inliers = np.zeros(n, dtype=bool)
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        R, t = rigid_fit(src[idx], dst[idx])
        resid = np.linalg.norm((src @ R.T + t) - dst, axis=1)
        inliers = resid < threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers.sum() < 3:
        return rigid_fit(src, dst) + (best_inliers,)
    R, t = rigid_fit(src[best_inliers], dst[best_inliers])
    return R, t, best_inliers


def icp(source: np.ndarray, target: np.ndarray, max_iter=50,
        tol=1e-5, max_corr_dist=1.0):
    """Point-to-point ICP.

    Returns (R, t, rmse). Correspondences are nearest neighbours in `target`,
    rejected beyond `max_corr_dist` so that points with no real match do not
    drag the fit.
    """
    tree = cKDTree(target)
    R_total = np.eye(3)
    t_total = np.zeros(3)
    moving = source.copy()
    prev = np.inf
    rmse = np.inf
    for _ in range(max_iter):
        dist, idx = tree.query(moving, k=1)
        keep = dist < max_corr_dist
        if keep.sum() < 3:
            break
        R, t = rigid_fit(moving[keep], target[idx[keep]])
        moving = moving @ R.T + t
        R_total = R @ R_total
        t_total = R @ t_total + t
        rmse = float(np.sqrt((dist[keep] ** 2).mean()))
        if abs(prev - rmse) < tol:
            break
        prev = rmse
    return R_total, t_total, rmse


def accumulate_trajectory(scans: list[np.ndarray], **icp_kwargs) -> np.ndarray:
    """Chain frame-to-frame ICP into a trajectory of 4x4 world poses.

    No loop closure and no pose graph, so error accumulates; that is the point
    of measuring absolute trajectory error against ground truth rather than
    reporting per-frame residuals.
    """
    poses = [np.eye(4)]
    for prev_scan, scan in zip(scans[:-1], scans[1:]):
        R, t, _ = icp(scan, prev_scan, **icp_kwargs)
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = R, t
        poses.append(poses[-1] @ step)
    return np.stack(poses)


def absolute_trajectory_error(estimated: np.ndarray, truth: np.ndarray) -> float:
    """RMSE in metres between estimated and ground-truth translations."""
    de = estimated[:, :3, 3] - truth[:, :3, 3]
    return float(np.sqrt((de ** 2).sum(axis=1).mean()))

"""KITTI calibration chain and projection geometry.

The KITTI chain maps a point in the Velodyne frame to pixels in rectified camera 2:

    x_img = P_rect_2 @ R_rect_0 @ T_velo_to_cam @ x_velo

Each matrix is read from the dataset calibration files. Composing them once and
reusing the 3x4 result is both faster and easier to reason about than applying
them one at a time.
"""
from __future__ import annotations
import numpy as np


def _to_homogeneous(points: np.ndarray) -> np.ndarray:
    """(N, 3) -> (N, 4) by appending a ones column."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected (N, 3), got {points.shape}")
    return np.hstack([points, np.ones((points.shape[0], 1), dtype=points.dtype)])


def compose_projection(P_rect: np.ndarray, R_rect: np.ndarray,
                       T_velo_cam: np.ndarray) -> np.ndarray:
    """Collapse the calibration chain into one 3x4 projection matrix.

    P_rect      (3, 4) rectified projection for the target camera
    R_rect      (3, 3) rectifying rotation of the reference camera
    T_velo_cam  (3, 4) or (4, 4) rigid transform, Velodyne to reference camera
    """
    R4 = np.eye(4)
    R4[:3, :3] = R_rect
    T4 = np.eye(4)
    T4[:3, :4] = T_velo_cam[:3, :4]
    return P_rect @ R4 @ T4


def project_lidar(points_velo: np.ndarray, P: np.ndarray,
                  image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Project Velodyne points into the image plane.

    Returns (uv, depth) for points that land inside the image and sit in front of
    the camera. `uv` is (M, 2) float pixel coordinates, `depth` is (M,) metres.
    """
    h, w = image_shape
    cam = (P @ _to_homogeneous(points_velo).T).T          # (N, 3)
    depth = cam[:, 2]
    in_front = depth > 1e-6
    cam, depth = cam[in_front], depth[in_front]
    uv = cam[:, :2] / depth[:, None]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv[inside], depth[inside]


def sparse_depth_map(points_velo: np.ndarray, P: np.ndarray,
                     image_shape: tuple[int, int]) -> np.ndarray:
    """Rasterise projected LiDAR returns into a sparse depth image.

    A 64-beam sweep covers roughly 5 to 20 percent of pixels, so the result is
    sparse by construction. Zero means "no return", not "zero distance".
    """
    uv, depth = project_lidar(points_velo, P, image_shape)
    out = np.zeros(image_shape, dtype=np.float32)
    cols = uv[:, 0].astype(np.int32)
    rows = uv[:, 1].astype(np.int32)
    # Nearest return wins where several points hit the same pixel.
    order = np.argsort(-depth)
    out[rows[order], cols[order]] = depth[order]
    return out


def metric_depth_for_boxes(depth_map: np.ndarray,
                           boxes: np.ndarray,
                           reducer=np.median) -> np.ndarray:
    """Recover a metric depth per 2D detection box from the sparse depth map.

    boxes is (K, 4) as (x1, y1, x2, y2). Boxes with no LiDAR return inside them
    come back as NaN rather than silently as zero.
    """
    out = np.full(len(boxes), np.nan, dtype=np.float32)
    for k, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
        patch = depth_map[y1:y2, x1:x2]
        hits = patch[patch > 0]
        if hits.size:
            out[k] = reducer(hits)
    return out

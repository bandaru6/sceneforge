"""Projection geometry tests against a synthetic calibration chain.

Everything here is checked against a case where the answer is known in closed
form, so a failure points at the code rather than at a tolerance.
"""
import numpy as np
import pytest

from sceneforge.calib import (compose_projection, metric_depth_for_boxes,
                              project_lidar, sparse_depth_map)

# A pinhole camera with a round focal length and a principal point in the
# middle of a 640x480 image.
K = np.array([[500.0, 0.0, 320.0],
              [0.0, 500.0, 240.0],
              [0.0, 0.0, 1.0]])
IMAGE_SHAPE = (480, 640)


def chain(t_velo_cam=(0.0, 0.0, 0.0), R_rect=None):
    """Build (P_rect, R_rect, T_velo_cam) for a camera at the origin."""
    P_rect = np.hstack([K, np.zeros((3, 1))])
    R = np.eye(3) if R_rect is None else R_rect
    T = np.hstack([np.eye(3), np.asarray(t_velo_cam, dtype=float).reshape(3, 1)])
    return P_rect, R, T


def test_compose_projection_matches_manual_chain():
    """Composing the chain equals applying the three matrices one at a time."""
    rng = np.random.default_rng(0)
    P_rect, R_rect, T = chain(t_velo_cam=(0.1, -0.2, 0.3))
    # A small but non-trivial rectifying rotation, built from a rotation vector
    # so it is guaranteed orthonormal.
    theta = 0.07
    R_rect = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                       [np.sin(theta), np.cos(theta), 0.0],
                       [0.0, 0.0, 1.0]])
    P = compose_projection(P_rect, R_rect, T)

    pts = rng.normal(size=(20, 3)) + np.array([0.0, 0.0, 15.0])
    manual = []
    for p in pts:
        x = R_rect @ (T[:3, :3] @ p + T[:3, 3])
        manual.append(P_rect @ np.append(x, 1.0))
    manual = np.array(manual)
    direct = (P @ np.hstack([pts, np.ones((len(pts), 1))]).T).T
    assert np.allclose(direct, manual, atol=1e-9)


def test_known_point_projects_to_known_pixel():
    """A point on the optical axis lands on the principal point.

    A point 10 m ahead and 1 m to the right must land at
    u = cx + f * X / Z = 320 + 500 * 1 / 10 = 370.
    """
    P = compose_projection(*chain())
    pts = np.array([[0.0, 0.0, 10.0],
                    [1.0, 0.0, 10.0],
                    [0.0, 2.0, 20.0]])
    uv, depth = project_lidar(pts, P, IMAGE_SHAPE)
    assert len(uv) == 3
    assert np.allclose(uv[0], [320.0, 240.0])
    assert np.allclose(uv[1], [370.0, 240.0])
    assert np.allclose(uv[2], [320.0, 240.0 + 500.0 * 2.0 / 20.0])
    assert np.allclose(depth, [10.0, 10.0, 20.0])


def test_translation_shifts_projection_predictably():
    """Translating the sensor 1 m right moves a 10 m point 50 px left."""
    P0 = compose_projection(*chain())
    P1 = compose_projection(*chain(t_velo_cam=(1.0, 0.0, 0.0)))
    pt = np.array([[0.0, 0.0, 10.0]])
    uv0, _ = project_lidar(pt, P0, IMAGE_SHAPE)
    uv1, _ = project_lidar(pt, P1, IMAGE_SHAPE)
    assert np.allclose(uv1[0] - uv0[0], [50.0, 0.0])


def test_points_behind_camera_and_outside_frame_are_dropped():
    P = compose_projection(*chain())
    pts = np.array([[0.0, 0.0, 10.0],      # keep
                    [0.0, 0.0, -10.0],     # behind the camera
                    [0.0, 0.0, 0.0],       # degenerate, on the image plane
                    [100.0, 0.0, 10.0]])   # projects far off the right edge
    uv, depth = project_lidar(pts, P, IMAGE_SHAPE)
    assert len(uv) == 1 and len(depth) == 1
    assert np.allclose(uv[0], [320.0, 240.0])


def test_bad_input_shape_is_rejected():
    P = compose_projection(*chain())
    with pytest.raises(ValueError):
        project_lidar(np.zeros((5, 2)), P, IMAGE_SHAPE)


def test_sparse_depth_map_keeps_the_nearest_return():
    """Two points on the same pixel: the closer one must survive.

    The map is sparse by construction, so the count of non-zero pixels is also
    checked; an implementation that smeared points would fail this.
    """
    P = compose_projection(*chain())
    pts = np.array([[0.0, 0.0, 30.0],
                    [0.0, 0.0, 7.0],
                    [1.0, 0.0, 10.0]])
    dm = sparse_depth_map(pts, P, IMAGE_SHAPE)
    assert dm.shape == IMAGE_SHAPE
    assert np.count_nonzero(dm) == 2
    assert dm[240, 320] == pytest.approx(7.0)
    assert dm[240, 370] == pytest.approx(10.0)


def test_metric_depth_per_box_and_empty_box_is_nan():
    """Median depth inside a box, NaN where the box caught no LiDAR return."""
    dm = np.zeros(IMAGE_SHAPE, dtype=np.float32)
    dm[100, 100] = 5.0
    dm[101, 100] = 7.0
    dm[102, 100] = 9.0
    boxes = np.array([[90, 90, 110, 110],      # contains all three returns
                      [400, 400, 420, 420]])   # contains none
    depths = metric_depth_for_boxes(dm, boxes)
    assert depths[0] == pytest.approx(7.0)
    assert np.isnan(depths[1])


def test_metric_depth_accepts_an_alternative_reducer():
    dm = np.zeros(IMAGE_SHAPE, dtype=np.float32)
    dm[10, 10], dm[11, 11] = 4.0, 8.0
    boxes = np.array([[0, 0, 20, 20]])
    assert metric_depth_for_boxes(dm, boxes, reducer=np.min)[0] == pytest.approx(4.0)
    assert metric_depth_for_boxes(dm, boxes, reducer=np.max)[0] == pytest.approx(8.0)


def test_projection_round_trip_recovers_the_point():
    """Unprojecting pixel plus depth through the inverse intrinsics returns XYZ.

    This is the step that turns a 2D detection into a metric-scale 3D one, so it
    is worth pinning that the forward and backward directions agree.
    """
    P = compose_projection(*chain())
    pts = np.array([[1.5, -0.8, 12.0], [-2.0, 0.4, 25.0]])
    uv, depth = project_lidar(pts, P, IMAGE_SHAPE)
    rays = np.linalg.inv(K) @ np.hstack([uv, np.ones((len(uv), 1))]).T
    recovered = (rays * depth).T
    assert np.allclose(recovered, pts, atol=1e-6)

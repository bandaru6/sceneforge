"""Registration tests: exact recovery, ICP convergence, outlier rejection."""
import numpy as np
import pytest

from sceneforge.registration import (absolute_trajectory_error,
                                     accumulate_trajectory, icp, rigid_fit,
                                     ransac_rigid_fit)


def rotation(axis, angle):
    """Rotation matrix from an axis and angle via Rodrigues' formula."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def cloud(n=400, seed=0, scale=5.0):
    """A random but reproducible point cloud with no symmetry to confuse ICP."""
    rng = np.random.default_rng(seed)
    return rng.normal(scale=scale, size=(n, 3))


def test_rigid_fit_recovers_a_known_transform_exactly():
    """Noise-free correspondences: Kabsch is exact, not approximate."""
    R_true = rotation([0.3, -0.7, 0.6], 0.4)
    t_true = np.array([1.2, -3.4, 0.7])
    src = cloud(seed=1)
    dst = src @ R_true.T + t_true
    R, t = rigid_fit(src, dst)
    assert np.allclose(R, R_true, atol=1e-10)
    assert np.allclose(t, t_true, atol=1e-9)
    assert np.allclose(src @ R.T + t, dst, atol=1e-9)


def test_rigid_fit_returns_a_proper_rotation():
    """det(R) must be +1: a reflection would mirror the cloud silently."""
    src = cloud(seed=2)
    # A reflected target cannot be reached by a rotation. The fit must still
    # return a proper rotation rather than solving the problem by mirroring.
    dst = src * np.array([1.0, 1.0, -1.0])
    R, _ = rigid_fit(src, dst)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_rigid_fit_is_translation_only_when_there_is_no_rotation():
    src = cloud(seed=3)
    t_true = np.array([-2.0, 0.5, 4.25])
    R, t = rigid_fit(src, src + t_true)
    assert np.allclose(R, np.eye(3), atol=1e-10)
    assert np.allclose(t, t_true, atol=1e-9)


def test_ransac_rejects_injected_outliers():
    """A fifth of the correspondences are garbage; the fit must ignore them.

    The plain least-squares fit is checked too, so the test fails if RANSAC is
    not actually doing anything: if the contaminated Kabsch fit were already
    accurate there would be nothing to prove.
    """
    rng = np.random.default_rng(7)
    R_true = rotation([0.0, 0.0, 1.0], 0.25)
    t_true = np.array([0.4, -1.1, 0.05])
    src = cloud(n=300, seed=4)
    dst = src @ R_true.T + t_true

    n_bad = 60
    bad = rng.choice(len(src), size=n_bad, replace=False)
    dst_corrupt = dst.copy()
    dst_corrupt[bad] += rng.normal(scale=20.0, size=(n_bad, 3))

    R_naive, t_naive = rigid_fit(src, dst_corrupt)
    naive_error = np.linalg.norm(t_naive - t_true)

    R, t, inliers = ransac_rigid_fit(src, dst_corrupt, iters=400, threshold=0.1,
                                     rng=np.random.default_rng(0))
    assert np.allclose(R, R_true, atol=1e-6)
    assert np.allclose(t, t_true, atol=1e-6)
    assert inliers.sum() == len(src) - n_bad
    assert not inliers[bad].any()
    assert naive_error > 10 * np.linalg.norm(t - t_true) + 1e-3


def test_ransac_inlier_mask_marks_every_good_correspondence():
    R_true = rotation([1.0, 1.0, 0.0], 0.15)
    t_true = np.array([0.2, 0.2, -0.3])
    src = cloud(n=200, seed=5)
    dst = src @ R_true.T + t_true
    _, _, inliers = ransac_rigid_fit(src, dst, iters=50, threshold=0.05,
                                     rng=np.random.default_rng(1))
    assert inliers.all()


def test_icp_converges_on_a_perturbed_copy():
    """ICP with no correspondences given must recover a small rigid offset.

    The perturbation is deliberately small, a few degrees and a few tens of
    centimetres, because point-to-point ICP has no basin of attraction beyond
    that. A test that asked for more would be testing the wrong claim.
    """
    target = cloud(n=600, seed=6, scale=3.0)
    R_true = rotation([0.1, 0.2, 0.97], np.deg2rad(4.0))
    t_true = np.array([0.25, -0.15, 0.05])
    source = target @ R_true.T + t_true

    R, t, rmse = icp(source, target, max_iter=80, max_corr_dist=2.0)
    aligned = source @ R.T + t
    assert rmse < 1e-3
    assert np.allclose(R, R_true.T, atol=1e-3)
    assert np.allclose(aligned, target, atol=1e-2)


def test_icp_reduces_residual_under_sensor_noise():
    """With noise the fit is not exact, but it must still beat the start."""
    rng = np.random.default_rng(8)
    target = cloud(n=500, seed=9, scale=3.0)
    R_true = rotation([0.0, 0.0, 1.0], np.deg2rad(3.0))
    t_true = np.array([0.2, 0.1, 0.0])
    source = target @ R_true.T + t_true + rng.normal(scale=0.01, size=target.shape)

    before = np.linalg.norm(source - target, axis=1).mean()
    R, t, rmse = icp(source, target, max_iter=60, max_corr_dist=1.0)
    after = np.linalg.norm((source @ R.T + t) - target, axis=1).mean()
    assert after < before / 5.0
    assert rmse < before


def test_icp_identity_on_identical_clouds():
    pts = cloud(n=200, seed=10, scale=2.0)
    R, t, rmse = icp(pts, pts, max_iter=10)
    assert np.allclose(R, np.eye(3), atol=1e-8)
    assert np.allclose(t, np.zeros(3), atol=1e-8)
    assert rmse == pytest.approx(0.0, abs=1e-9)


def test_accumulate_trajectory_recovers_a_straight_drive():
    """Three scans of one cloud translated by a fixed step.

    Poses are returned in the frame of the first scan, so consecutive
    translations must come out as multiples of the step.
    """
    base = cloud(n=500, seed=11, scale=3.0)
    step = np.array([0.5, 0.0, 0.0])
    # Scan k sees the same world from a sensor that has moved by k * step, so
    # the points move by -k * step in the sensor frame.
    scans = [base - k * step for k in range(4)]
    poses = accumulate_trajectory(scans, max_iter=60, max_corr_dist=2.0)
    assert poses.shape == (4, 4, 4)
    assert np.allclose(poses[0], np.eye(4), atol=1e-12)
    for k in range(4):
        assert np.allclose(poses[k][:3, 3], k * step, atol=2e-2)


def test_absolute_trajectory_error_is_zero_for_a_perfect_match():
    poses = np.repeat(np.eye(4)[None], 5, axis=0)
    poses[:, :3, 3] = np.arange(15).reshape(5, 3)
    assert absolute_trajectory_error(poses, poses) == pytest.approx(0.0)


def test_absolute_trajectory_error_equals_the_known_offset():
    """A constant 3-4-0 offset on every pose gives an RMSE of exactly 5 m."""
    truth = np.repeat(np.eye(4)[None], 6, axis=0)
    truth[:, :3, 3] = np.linspace(0.0, 10.0, 18).reshape(6, 3)
    estimated = truth.copy()
    estimated[:, :3, 3] += np.array([3.0, 4.0, 0.0])
    assert absolute_trajectory_error(estimated, truth) == pytest.approx(5.0)

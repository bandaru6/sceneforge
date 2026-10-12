# SceneForge

LiDAR-camera 3D reconstruction, odometry, and mapping, with a path from raw sensor
log to a scene you can simulate.

The pipeline is four stages:

1. **Calibration and projection** (`sceneforge/calib.py`): compose the KITTI
   rectification and Velodyne-to-camera transforms into a single 3x4 projection
   matrix, project LiDAR returns into the image plane to get a sparse depth map,
   and invert the chain to recover metric-scale depth for image detections.
2. **Registration and odometry** (`sceneforge/registration.py`): RANSAC-gated SVD
   rigid fit plus point-to-point ICP, run frame to frame across a scan sequence to
   accumulate a trajectory. Evaluated as absolute trajectory error against ground truth.
3. **Semantic occupancy** (`sceneforge/occupancy.py`): aggregate registered sweeps
   into a sparse voxel grid, carry per-voxel semantic labels, render occupied voxels
   as cubes.
4. **Neural reconstruction and simulation** (`sceneforge/nerf.py`, `sceneforge/sim.py`):
   fit a radiance field to posed images, extract geometry, and export to MJCF so the
   reconstructed scene loads in MuJoCo and behaves under physics. The field is a
   Fourier-encoded MLP returning density and view-dependent colour, composited by
   the volume rendering integral with explicit transmittance; geometry leaves it
   through a marching-cubes pass over a density grid. The export side computes
   mass, centre of mass and the inertia tensor from the mesh triangles under a
   stated material density rather than guessing them, because MuJoCo will
   simulate a physically impossible body without complaint.

Part segmentation (`sceneforge/segmentation.py`) implements PointNet and a
point Transformer against the same data and metrics, so the question "when does
learned point-to-point communication beat one global point-set summary" can be
answered on a fixed split.

## Status

Work in progress. All four stages are implemented and the geometry and physics
paths are covered by tests; what is still missing is evaluation at scale, which
is the active work. Results are not committed yet, see `results/` once the
evaluation runs land.

Known limitations, stated rather than hidden:

- Odometry is frame to frame with no loop closure and no pose graph, so drift
  grows with sequence length. Measuring it against GPS ground truth is the point.
- MuJoCo collides convex geometry, so a concave reconstruction collides as its
  convex hull until it is split into convex parts upstream.
- Material density is an assumption supplied on the command line. Reconstruction
  recovers shape and appearance, not what a thing is made of.

## Install

```bash
pip install -r requirements.txt
```

MuJoCo needs no separate binary install; the `mujoco` wheel ships its own.

## Run

```bash
# LiDAR odometry over a scan sequence, scored against ground-truth poses
python scripts/run_odometry.py --scans data/scans --gt data/poses.txt \
    --max-scans 1024 --out results/trajectory.npy

# the two segmentation arms, at equal batch size and seed so the comparison means
# something; each run reports held-out mean IoU and median seconds per step
python scripts/train_segmentation.py --data data/parts --arch pointnet --epochs 50
python scripts/train_segmentation.py --data data/parts --arch transformer --epochs 50

# posed frames to a radiance field to a mesh
python scripts/train_nerf.py --images data/frames --poses data/poses.npy \
    --intrinsics data/K.npy --iters 20000 --out-mesh results/scene.obj

# mesh to MJCF, compiled and stepped until the scene settles
python scripts/export_to_mujoco.py --mesh results/scene.obj --out results/scene.xml
```

Every script takes `--help`. Tests need no data:

```bash
python -m pytest tests -q
```

## Layout

```
sceneforge/
  calib.py          KITTI calibration chain, projection, metric-scale depth
  registration.py   RANSAC rigid fit, ICP, trajectory accumulation
  occupancy.py      sparse semantic voxel map
  segmentation.py   PointNet and point Transformer
  nerf.py           radiance field, volume rendering, mesh extraction
  sim.py            mesh to MJCF, MuJoCo load and step
scripts/
  run_odometry.py         scan sequence to trajectory, with ATE against ground truth
  train_segmentation.py   PointNet or Transformer on a fixed split
  train_nerf.py           posed frames to radiance field to mesh
  export_to_mujoco.py     mesh to MJCF, compile and settle
tests/
  test_calib.py       projection through a synthetic calibration chain
  test_registration.py  exact rigid fit, ICP convergence, outlier rejection
  test_sim.py         mass properties against closed-form box inertia, MJCF structure
```

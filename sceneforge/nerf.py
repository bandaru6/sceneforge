"""Neural radiance field: encoding, MLP, ray sampling, volume rendering.

The field is a function (x, y, z, theta, phi) -> (density, colour). Density is
view independent, colour is not: a windscreen is bright from one angle and dark
from another, and a field that cannot express that bakes the specular highlight
into the geometry instead.

Rendering a pixel means integrating along its ray,

    C = sum_i T_i (1 - exp(-sigma_i delta_i)) c_i,
    T_i = exp(-sum_{j<i} sigma_j delta_j)

where T_i is the transmittance, the probability that the ray reached sample i
without being absorbed. Everything below is that integral plus the machinery to
get samples and features into it.

Geometry leaves the field through `extract_mesh`, which evaluates density on a
grid and surfaces an isolevel, so the result can be handed to MuJoCo.
"""
from __future__ import annotations

import os
import warnings
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Input encoding
# --------------------------------------------------------------------------

def positional_encoding(x: torch.Tensor, n_freqs: int,
                        include_input: bool = True) -> torch.Tensor:
    """Fourier features at octave-spaced frequencies.

    A coordinate MLP fed raw xyz is biased towards smooth functions and cannot
    fit the high-frequency detail that makes a reconstruction look like the
    scene rather than a blur of it. Mapping x to
    [x, sin(2^0 pi x), cos(2^0 pi x), ..., sin(2^(L-1) pi x), cos(...)]
    moves that detail into the input space where the MLP can reach it.

    x is (..., D); the result is (..., D * (2 * n_freqs + include_input)).
    """
    if n_freqs < 0:
        raise ValueError("n_freqs must be non-negative")
    freqs = 2.0 ** torch.arange(n_freqs, dtype=x.dtype, device=x.device)
    scaled = x[..., None, :] * freqs[:, None] * np.pi        # (..., L, D)
    flat = scaled.reshape(*x.shape[:-1], -1)                 # (..., L*D)
    parts = [x] if include_input else []
    parts += [torch.sin(flat), torch.cos(flat)]
    return torch.cat(parts, dim=-1)


def encoded_dim(in_dim: int, n_freqs: int, include_input: bool = True) -> int:
    """Width `positional_encoding` produces, for sizing the first linear layer."""
    return in_dim * (2 * n_freqs + (1 if include_input else 0))


# --------------------------------------------------------------------------
# The field itself
# --------------------------------------------------------------------------

class RadianceField(nn.Module):
    """Coordinate MLP returning volume density and view-dependent colour.

    Structure follows the original NeRF: a trunk over encoded position with one
    skip connection halfway (so the network does not have to carry the
    coordinate through every layer), a density head reading the trunk directly,
    and a shallow colour head that sees the trunk feature concatenated with the
    encoded view direction. Density is kept independent of direction on purpose;
    letting it depend on view lets the model explain away occlusion errors with
    direction-specific fog and the geometry degrades.
    """

    def __init__(self, pos_freqs: int = 10, dir_freqs: int = 4,
                 width: int = 256, depth: int = 8, skip: int = 4):
        super().__init__()
        self.pos_freqs = pos_freqs
        self.dir_freqs = dir_freqs
        self.skip = skip

        pos_dim = encoded_dim(3, pos_freqs)
        dir_dim = encoded_dim(3, dir_freqs)

        trunk = []
        for i in range(depth):
            fan_in = pos_dim if i == 0 else width + (pos_dim if i == skip else 0)
            trunk.append(nn.Linear(fan_in, width))
        self.trunk = nn.ModuleList(trunk)
        self.sigma_head = nn.Linear(width, 1)
        self.feature = nn.Linear(width, width)
        self.colour = nn.Sequential(
            nn.Linear(width + dir_dim, width // 2), nn.ReLU(inplace=True),
            nn.Linear(width // 2, 3),
        )

    def forward(self, positions: torch.Tensor,
                directions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the field.

        positions   (..., 3) world-space points
        directions  (..., 3) unit view directions, broadcast to match positions

        Returns (sigma, rgb) with shapes (..., ) and (..., 3). Density is passed
        through ReLU so it cannot go negative, colour through a sigmoid so it
        stays in [0, 1] and the renderer never has to clamp.
        """
        if positions.shape[-1] != 3 or directions.shape[-1] != 3:
            raise ValueError("positions and directions must be (..., 3)")
        directions = directions.expand_as(positions)

        pe = positional_encoding(positions, self.pos_freqs)
        de = positional_encoding(directions, self.dir_freqs)

        h = pe
        for i, layer in enumerate(self.trunk):
            if i == self.skip and i != 0:
                h = torch.cat([h, pe], dim=-1)
            h = torch.relu(layer(h))

        sigma = torch.relu(self.sigma_head(h)).squeeze(-1)
        rgb = torch.sigmoid(self.colour(torch.cat([self.feature(h), de], dim=-1)))
        return sigma, rgb

    @torch.no_grad()
    def density_at(self, positions: torch.Tensor,
                   chunk: int = 1 << 16) -> torch.Tensor:
        """Density only, evaluated in chunks. Used by the mesh extractor.

        Colour needs a direction and the mesh does not care about one, so a
        fixed +z direction is passed and the colour output discarded.
        """
        flat = positions.reshape(-1, 3)
        d = torch.zeros_like(flat)
        d[:, 2] = 1.0
        out = []
        for start in range(0, flat.shape[0], chunk):
            sl = slice(start, start + chunk)
            sigma, _ = self(flat[sl], d[sl])
            out.append(sigma)
        return torch.cat(out).reshape(positions.shape[:-1])


# --------------------------------------------------------------------------
# Rays and sampling
# --------------------------------------------------------------------------

def get_rays(height: int, width: int, intrinsics: np.ndarray,
             c2w: np.ndarray, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel ray origins and unit directions for one posed camera.

    intrinsics is the 3x3 pinhole matrix, c2w the 4x4 camera-to-world pose. The
    returned tensors are (H, W, 3); origins are identical across pixels for a
    pinhole camera but are returned per pixel so callers can flatten and shuffle
    rays without tracking which image each came from.
    """
    K = np.asarray(intrinsics, dtype=np.float32)
    pose = np.asarray(c2w, dtype=np.float32)
    jj, ii = np.meshgrid(np.arange(width, dtype=np.float32),
                         np.arange(height, dtype=np.float32), indexing="xy")
    # Pixel centres through the inverse intrinsics, camera looking down -z.
    dirs = np.stack([(jj - K[0, 2]) / K[0, 0],
                     -(ii - K[1, 2]) / K[1, 1],
                     -np.ones_like(jj)], axis=-1)
    world = dirs @ pose[:3, :3].T
    world /= np.linalg.norm(world, axis=-1, keepdims=True)
    origins = np.broadcast_to(pose[:3, 3], world.shape).copy()
    return (torch.as_tensor(origins, device=device),
            torch.as_tensor(world, device=device))


def sample_stratified(rays_o: torch.Tensor, rays_d: torch.Tensor,
                      near: float, far: float, n_samples: int,
                      perturb: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Stratified samples along each ray.

    The interval [near, far] is split into `n_samples` equal bins and one sample
    is drawn per bin. Jittering within the bin (`perturb`) is what keeps the
    network from only ever seeing the same depths and overfitting to a fixed
    comb of planes; it is switched off at evaluation time so renders are
    deterministic.

    Returns (points, t_vals) with shapes (R, n_samples, 3) and (R, n_samples).
    """
    if far <= near:
        raise ValueError("far must exceed near")
    device = rays_o.device
    n_rays = rays_o.shape[0]
    edges = torch.linspace(near, far, n_samples + 1, device=device)
    lower, upper = edges[:-1], edges[1:]
    if perturb:
        u = torch.rand(n_rays, n_samples, device=device)
    else:
        u = torch.full((n_rays, n_samples), 0.5, device=device)
    t_vals = lower + (upper - lower) * u
    points = rays_o[:, None, :] + rays_d[:, None, :] * t_vals[..., None]
    return points, t_vals


def sample_pdf(t_mid: torch.Tensor, weights: torch.Tensor,
               n_samples: int, perturb: bool = True) -> torch.Tensor:
    """Inverse-transform sampling of new depths from the coarse weights.

    The coarse pass tells us where along the ray the mass is; a second pass that
    spends its samples there rather than uniformly is the single cheapest
    quality win in the whole renderer. Returns (R, n_samples) sorted depths.
    """
    weights = weights + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)   # (R, M+1)

    shape = (*cdf.shape[:-1], n_samples)
    u = (torch.rand(shape, device=cdf.device) if perturb
         else torch.linspace(0.0, 1.0, n_samples, device=cdf.device).expand(shape))

    idx = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    lo = (idx - 1).clamp(min=0)
    hi = idx.clamp(max=cdf.shape[-1] - 1)
    cdf_lo = torch.gather(cdf, -1, lo)
    cdf_hi = torch.gather(cdf, -1, hi)
    # t_mid has one fewer entry than cdf; clamp into its range.
    bins_lo = torch.gather(t_mid, -1, lo.clamp(max=t_mid.shape[-1] - 1))
    bins_hi = torch.gather(t_mid, -1, hi.clamp(max=t_mid.shape[-1] - 1))
    span = (cdf_hi - cdf_lo).clamp(min=1e-5)
    frac = (u - cdf_lo) / span
    return torch.sort(bins_lo + frac * (bins_hi - bins_lo), dim=-1).values


# --------------------------------------------------------------------------
# Volume rendering
# --------------------------------------------------------------------------

def volume_render(sigma: torch.Tensor, rgb: torch.Tensor, t_vals: torch.Tensor,
                  rays_d: torch.Tensor | None = None,
                  white_background: bool = False) -> dict[str, torch.Tensor]:
    """Composite samples along each ray into colour, depth and opacity.

    sigma   (R, S) non-negative densities
    rgb     (R, S, 3) colours in [0, 1]
    t_vals  (R, S) sample depths, increasing along the ray
    rays_d  (R, 3) directions; if given, segment lengths are scaled by their
            norm so that density stays per unit of world distance rather than
            per unit of parameter t

    Returns a dict with `rgb` (R, 3), `depth` (R,), `acc` (R,) and the raw
    per-sample `weights` (R, S), which the hierarchical pass needs.
    """
    deltas = t_vals[..., 1:] - t_vals[..., :-1]
    # Last segment runs to infinity: a large finite value keeps alpha -> 1 for
    # anything solid without putting an inf into the exponential.
    last = torch.full_like(deltas[..., :1], 1e10)
    deltas = torch.cat([deltas, last], dim=-1)
    if rays_d is not None:
        deltas = deltas * torch.norm(rays_d[..., None, :], dim=-1)

    alpha = 1.0 - torch.exp(-sigma * deltas)
    # Transmittance is the exclusive cumulative product of (1 - alpha), so
    # sample i is weighted by what is left of the ray before it arrives.
    keep = 1.0 - alpha + 1e-10
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(keep[..., :1]), keep[..., :-1]], dim=-1), dim=-1)
    weights = alpha * transmittance

    colour = (weights[..., None] * rgb).sum(dim=-2)
    depth = (weights * t_vals).sum(dim=-1)
    acc = weights.sum(dim=-1)
    if white_background:
        colour = colour + (1.0 - acc[..., None])
    return {"rgb": colour, "depth": depth, "acc": acc, "weights": weights}


def render_rays(field: RadianceField, rays_o: torch.Tensor, rays_d: torch.Tensor,
                near: float = 0.1, far: float = 80.0, n_coarse: int = 64,
                n_fine: int = 0, perturb: bool = True,
                white_background: bool = False) -> dict[str, torch.Tensor]:
    """Full forward render of a batch of rays, optionally hierarchical.

    With `n_fine` zero this is one stratified pass. With `n_fine` positive the
    coarse weights drive a second set of samples, the two sets are merged and
    sorted, and the field is evaluated once more on the union.
    """
    points, t_vals = sample_stratified(rays_o, rays_d, near, far, n_coarse, perturb)
    sigma, rgb = field(points, rays_d[:, None, :])
    coarse = volume_render(sigma, rgb, t_vals, rays_d, white_background)
    if n_fine <= 0:
        return coarse

    t_mid = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
    extra = sample_pdf(t_mid, coarse["weights"][..., 1:-1].detach(), n_fine, perturb)
    t_all = torch.sort(torch.cat([t_vals, extra], dim=-1), dim=-1).values
    points = rays_o[:, None, :] + rays_d[:, None, :] * t_all[..., None]
    sigma, rgb = field(points, rays_d[:, None, :])
    fine = volume_render(sigma, rgb, t_all, rays_d, white_background)
    fine["rgb_coarse"] = coarse["rgb"]
    return fine


def photometric_loss(rendered: torch.Tensor, target: torch.Tensor,
                     coarse: torch.Tensor | None = None) -> torch.Tensor:
    """Mean squared error on colour, summed over coarse and fine passes.

    Supervising the coarse pass as well as the fine one is what stops the coarse
    network from drifting into a useless proposal distribution.
    """
    loss = torch.mean((rendered - target) ** 2)
    if coarse is not None:
        loss = loss + torch.mean((coarse - target) ** 2)
    return loss


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_posed_images(image_dir: str, poses: np.ndarray, downscale: int = 1,
                      extensions: Iterable[str] = (".png", ".jpg", ".jpeg")
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Read a directory of frames with OpenCV and pair them with their poses.

    Images are decoded BGR by OpenCV and converted to RGB here so that the
    renderer, which emits RGB, never has to think about channel order again.
    `downscale` uses area interpolation, which is the right filter for shrinking
    and avoids the aliasing that nearest or bilinear leave behind.

    Returns (images, poses) with images as (N, H, W, 3) float32 in [0, 1].
    """
    import cv2

    names = sorted(f for f in os.listdir(image_dir)
                   if os.path.splitext(f)[1].lower() in tuple(extensions))
    if not names:
        raise FileNotFoundError(f"no images with {tuple(extensions)} in {image_dir}")
    poses = np.asarray(poses, dtype=np.float32)
    if len(poses) != len(names):
        raise ValueError(f"{len(names)} images but {len(poses)} poses")

    frames = []
    for name in names:
        bgr = cv2.imread(os.path.join(image_dir, name), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"OpenCV could not decode {name}")
        if downscale > 1:
            h, w = bgr.shape[:2]
            bgr = cv2.resize(bgr, (w // downscale, h // downscale),
                             interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames).astype(np.float32) / 255.0, poses


def scale_intrinsics(intrinsics: np.ndarray, downscale: int) -> np.ndarray:
    """Rescale a pinhole matrix to match images shrunk by `downscale`."""
    K = np.asarray(intrinsics, dtype=np.float32).copy()
    K[:2] /= float(downscale)
    return K


# --------------------------------------------------------------------------
# Getting geometry out of the field
# --------------------------------------------------------------------------

@torch.no_grad()
def density_grid(field: RadianceField, bounds: tuple[float, float],
                 resolution: int = 128, device=None) -> np.ndarray:
    """Evaluate density on a regular cube of side `bounds` at `resolution`.

    Returns a (R, R, R) float32 array indexed [x, y, z].
    """
    lo, hi = bounds
    axis = torch.linspace(lo, hi, resolution, device=device)
    xs, ys, zs = torch.meshgrid(axis, axis, axis, indexing="ij")
    pts = torch.stack([xs, ys, zs], dim=-1)
    return field.density_at(pts).cpu().numpy().astype(np.float32)


def _marching_cubes(grid: np.ndarray, level: float, spacing: float):
    """Surface `grid` at `level`, preferring scikit-image.

    scikit-image's marching cubes gives a watertight, correctly oriented
    triangulation. If it cannot be imported we fall back to a blocky but valid
    surface: every occupied voxel whose neighbour across a face is empty
    contributes that face as two triangles. The fallback is deliberately crude,
    it is there so that the export path can be exercised without pulling in
    another dependency, and it is not what you want to hand to a renderer, so it
    warns rather than substituting itself quietly.

    The import is guarded broadly on purpose: an absent scikit-image raises
    ImportError, but one built against a different NumPy raises ValueError from
    its compiled extensions, and both mean the same thing here.
    """
    try:
        from skimage import measure
    except Exception as exc:
        warnings.warn(f"scikit-image unavailable ({exc}); falling back to a "
                      f"blocky voxel surface", RuntimeWarning, stacklevel=2)
        return _voxel_surface(grid, level, spacing)
    verts, faces, _, _ = measure.marching_cubes(
        grid, level=level, spacing=(spacing, spacing, spacing))
    return verts.astype(np.float32), faces.astype(np.int32)


def _voxel_surface(grid: np.ndarray, level: float, spacing: float):
    """Blocky fallback surface: exposed faces of occupied voxels."""
    occ = grid > level
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    index: dict[tuple[int, int, int], int] = {}

    def vertex(c):
        key = (round(c[0], 6), round(c[1], 6), round(c[2], 6))
        if key not in index:
            index[key] = len(verts)
            verts.append(key)
        return index[key]

    # Unit-cube corner offsets per face, in counter-clockwise order seen from
    # outside, so the fallback mesh has consistent winding like the real one.
    face_quads = {
        (1, 0, 0): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
        (-1, 0, 0): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
        (0, 1, 0): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
        (0, -1, 0): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        (0, 0, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
        (0, 0, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    }
    nx, ny, nz = occ.shape
    for i, j, k in zip(*np.nonzero(occ)):
        for (di, dj, dk), quad in face_quads.items():
            ni, nj, nk = i + di, j + dj, k + dk
            inside = 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz
            if inside and occ[ni, nj, nk]:
                continue
            ids = [vertex(((i + a) * spacing, (j + b) * spacing, (k + c) * spacing))
                   for a, b, c in quad]
            faces.append((ids[0], ids[1], ids[2]))
            faces.append((ids[0], ids[2], ids[3]))
    if not faces:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32))
    return (np.asarray(verts, dtype=np.float32),
            np.asarray(faces, dtype=np.int32))


def write_obj(path: str, verts: np.ndarray, faces: np.ndarray) -> str:
    """Write a triangle mesh as Wavefront OBJ (1-based face indices)."""
    with open(path, "w") as fh:
        for v in verts:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces:
            fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")
    return path


def extract_mesh(field: RadianceField, bounds: tuple[float, float] = (-2.0, 2.0),
                 resolution: int = 128, level: float = 10.0,
                 out_path: str | None = None, device=None):
    """Turn the learned density field into a triangle mesh.

    `level` is the density isovalue treated as the surface. There is no
    canonical value for it: density is unnormalised and its scale depends on
    scene units and training length, so the threshold is a knob, and sweeping it
    on a held-out view is the honest way to pick one.

    Vertices come out of the grid in index space and are mapped back to world
    coordinates here. Returns (verts, faces), and also writes OBJ if `out_path`
    is given.
    """
    grid = density_grid(field, bounds, resolution, device=device)
    lo, hi = bounds
    spacing = (hi - lo) / max(resolution - 1, 1)
    verts, faces = _marching_cubes(grid, level, spacing)
    if len(verts):
        verts = verts + np.asarray([lo, lo, lo], dtype=np.float32)
    if out_path is not None:
        write_obj(out_path, verts, faces)
    return verts, faces


def train_step(field: RadianceField, optimiser: torch.optim.Optimizer,
               rays_o: torch.Tensor, rays_d: torch.Tensor, target: torch.Tensor,
               render_kwargs: dict | None = None) -> float:
    """One optimisation step on a batch of rays. Returns the loss value."""
    render_kwargs = render_kwargs or {}
    optimiser.zero_grad(set_to_none=True)
    out = render_rays(field, rays_o, rays_d, **render_kwargs)
    loss = photometric_loss(out["rgb"], target, out.get("rgb_coarse"))
    loss.backward()
    optimiser.step()
    return float(loss.detach())


def psnr(rendered: torch.Tensor, target: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB for images in [0, 1]."""
    mse = torch.mean((rendered - target) ** 2).clamp(min=1e-12)
    return float(-10.0 * torch.log10(mse))

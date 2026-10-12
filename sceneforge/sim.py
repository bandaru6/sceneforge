"""Reconstruction to MuJoCo: physical properties, MJCF generation, stepping.

A mesh out of `sceneforge.nerf.extract_mesh` is geometry and nothing else. To
drop it into a physics engine it needs mass, a centre of mass, an inertia
tensor and a friction coefficient. Mass and inertia follow from the mesh plus an
assumed material density, and are computed here from the triangles rather than
guessed, because MuJoCo will happily simulate a body whose inertia tensor is
physically impossible and the result looks wrong in ways that are hard to trace.

The pipeline is:

    mesh + density -> MassProperties -> MJCF XML -> mujoco.MjModel -> mj_step

`settle_scene` runs the last two stages and reports where everything came to
rest, which is the cheap sanity check on a reconstruction: a scene whose floor
is tilted or whose objects interpenetrate does not settle, it explodes.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# Mesh IO
# --------------------------------------------------------------------------

def load_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a triangle mesh as (verts, faces).

    trimesh is used when available because it handles STL, PLY and OBJ with
    materials; the built-in reader below covers plain triangulated OBJ, which is
    what `extract_mesh` writes, so the export path works without trimesh
    installed.
    """
    try:
        import trimesh
    except ImportError:
        return load_obj(path)
    mesh = trimesh.load(path, force="mesh")
    return (np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int32))


def load_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Minimal Wavefront OBJ reader for triangulated meshes.

    Handles `v` and `f` lines, ignores normals, texture coordinates and
    materials, and fans any polygon with more than three corners into triangles.
    Face indices in OBJ are 1-based and may be negative (relative to the end of
    the vertex list), and both forms are handled.
    """
    verts: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = []
                for token in line.split()[1:]:
                    raw = int(token.split("/")[0])
                    idx.append(raw - 1 if raw > 0 else len(verts) + raw)
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    return (np.asarray(verts, dtype=np.float64),
            np.asarray(faces, dtype=np.int32))


# --------------------------------------------------------------------------
# Physical properties
# --------------------------------------------------------------------------

@dataclass
class MassProperties:
    """Rigid-body properties of a closed triangle mesh.

    volume      m^3, signed; negative means the faces are wound inwards
    mass        kg, volume times material density
    com         (3,) centre of mass in mesh coordinates
    inertia     (3, 3) inertia tensor about the centre of mass, kg m^2
    density     kg/m^3 that was assumed
    friction    sliding friction coefficient written into the MJCF
    """
    volume: float
    mass: float
    com: np.ndarray
    inertia: np.ndarray
    density: float
    friction: float = 0.8

    @property
    def fullinertia(self) -> tuple[float, ...]:
        """MuJoCo `fullinertia` ordering: ixx iyy izz ixy ixz iyz."""
        I = self.inertia
        return (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])

    @property
    def principal(self) -> tuple[np.ndarray, np.ndarray]:
        """Principal moments and axes, from the eigendecomposition of I.

        Useful as a check: all three moments must be positive and must satisfy
        the triangle inequality for a real rigid body.
        """
        vals, vecs = np.linalg.eigh(self.inertia)
        return vals, vecs

    def is_physical(self, tol: float = 1e-9) -> bool:
        """True if the moments are positive and obey the triangle inequality."""
        a, b, c = sorted(self.principal[0])
        return a > tol and (a + b) >= c - tol


def mass_properties(verts: np.ndarray, faces: np.ndarray,
                    density: float = 1000.0,
                    friction: float = 0.8) -> MassProperties:
    """Volume, centre of mass and inertia tensor of a closed triangle mesh.

    The mesh is decomposed into tetrahedra spanning the origin and each
    triangle. For a tetrahedron with vertices (0, a, b, c) the signed volume is
    det[a b c] / 6, and its second-moment (covariance) matrix is

        C = det[a b c] * A C0 A^T,    A = [a b c],
        C0 = (1/120) [[2,1,1],[1,2,1],[1,1,2]]

    Summing over triangles gives the covariance of the whole solid about the
    origin; shifting it to the centre of mass and converting second moments to
    moments of inertia gives

        I = density * (trace(C_com) * Identity - C_com)

    Tetrahedra outside the solid contribute negative volume and cancel, so the
    result is exact for any closed mesh regardless of how it was triangulated.
    It is meaningless for an open one, which is why `volume` is reported and
    worth checking before trusting the mass.
    """
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int32)
    if len(F) == 0:
        raise ValueError("mesh has no faces")
    if density <= 0:
        raise ValueError("density must be positive")

    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))        # det[a b c] per tet
    volume = float(det.sum() / 6.0)
    if abs(volume) < 1e-12:
        raise ValueError("mesh encloses no volume; is it closed and wound consistently?")

    # Centroid: each tet's centroid is the mean of its four vertices, one of
    # which is the origin, so (a + b + c) / 4.
    moment = ((a + b + c) / 4.0 * (det / 6.0)[:, None]).sum(axis=0)
    com = moment / volume

    C0 = np.array([[2.0, 1.0, 1.0],
                   [1.0, 2.0, 1.0],
                   [1.0, 1.0, 2.0]]) / 120.0
    A = np.stack([a, b, c], axis=-1)                      # (T, 3, 3) columns a b c
    cov = np.einsum("t,tij,jk,tlk->il", det, A, C0, A)

    sign = np.sign(volume)
    cov *= sign
    abs_volume = abs(volume)
    cov_com = cov - abs_volume * np.outer(com, com)
    inertia = density * (np.trace(cov_com) * np.eye(3) - cov_com)

    return MassProperties(volume=abs_volume, mass=density * abs_volume, com=com,
                          inertia=inertia, density=density, friction=friction)


def estimate_physical_properties(verts: np.ndarray, faces: np.ndarray,
                                 material: str = "generic") -> MassProperties:
    """Mass properties under a named material assumption.

    Reconstruction recovers shape and appearance, not what a thing is made of,
    so density has to come from somewhere outside the geometry. These are
    textbook bulk densities in kg/m^3, used as a stated assumption rather than a
    measurement; `material="generic"` is water density, the conventional stand-in
    when nothing is known.
    """
    table = {
        "generic": (1000.0, 0.8),
        "wood": (650.0, 0.6),
        "plastic": (950.0, 0.5),
        "concrete": (2400.0, 0.9),
        "steel": (7850.0, 0.4),
        "aluminium": (2700.0, 0.45),
        "rubber": (1200.0, 1.2),
    }
    if material not in table:
        raise KeyError(f"unknown material {material!r}; have {sorted(table)}")
    density, friction = table[material]
    return mass_properties(verts, faces, density=density, friction=friction)


# --------------------------------------------------------------------------
# MJCF generation
# --------------------------------------------------------------------------

@dataclass
class SceneBody:
    """One simulated body: a mesh file plus its properties and initial pose."""
    name: str
    mesh_path: str
    properties: MassProperties
    position: tuple[float, float, float] = (0.0, 0.0, 0.5)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    free: bool = True
    rgba: tuple[float, float, float, float] = (0.75, 0.75, 0.78, 1.0)


def _fmt(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def build_mjcf(bodies: list[SceneBody], model_name: str = "sceneforge",
               timestep: float = 0.002, gravity=(0.0, 0.0, -9.81),
               add_floor: bool = True) -> str:
    """Build the MJCF XML for a reconstructed scene.

    Each body gets an explicit `<inertial>` block so MuJoCo uses the inertia we
    computed instead of inferring one from the convex hull of the mesh, and a
    `<freejoint>` so it can actually move. Mesh assets are referenced by
    basename with `<compiler meshdir>` pointing at the directory, which is how
    MuJoCo expects to resolve them.

    Collision geometry in MuJoCo is convex, so a concave reconstruction collides
    as its hull. That is a real limitation of this export, not something the XML
    can hide; a concave scene has to be split into convex parts upstream.
    """
    if not bodies:
        raise ValueError("need at least one body")
    meshdirs = {os.path.dirname(os.path.abspath(b.mesh_path)) for b in bodies}
    if len(meshdirs) > 1:
        raise ValueError("all meshes must live in one directory for meshdir to work")

    root = ET.Element("mujoco", model=model_name)
    ET.SubElement(root, "compiler", angle="radian", meshdir=meshdirs.pop(),
                  autolimits="true")
    ET.SubElement(root, "option", timestep=f"{timestep:g}", gravity=_fmt(gravity),
                  integrator="implicitfast")

    asset = ET.SubElement(root, "asset")
    for body in bodies:
        ET.SubElement(asset, "mesh", name=f"{body.name}_mesh",
                      file=os.path.basename(body.mesh_path))

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1", directional="true")
    if add_floor:
        ET.SubElement(world, "geom", name="floor", type="plane",
                      size="20 20 0.1", rgba="0.3 0.32 0.35 1",
                      friction=_fmt((1.0, 0.005, 0.0001)))

    for body in bodies:
        node = ET.SubElement(world, "body", name=body.name,
                             pos=_fmt(body.position), quat=_fmt(body.quat))
        if body.free:
            ET.SubElement(node, "freejoint", name=f"{body.name}_free")
        props = body.properties
        ET.SubElement(node, "inertial", pos=_fmt(props.com),
                      mass=f"{props.mass:.6g}", fullinertia=_fmt(props.fullinertia))
        ET.SubElement(node, "geom", name=f"{body.name}_geom", type="mesh",
                      mesh=f"{body.name}_mesh", rgba=_fmt(body.rgba),
                      friction=_fmt((props.friction, 0.005, 0.0001)))

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def write_mjcf(bodies: list[SceneBody], out_path: str, **kwargs) -> str:
    """Write the MJCF for `bodies` to disk and return the path."""
    xml = build_mjcf(bodies, **kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(xml)
    return out_path


def mesh_to_mjcf(mesh_path: str, out_path: str, name: str = "scene",
                 material: str = "generic", drop_height: float = 0.5,
                 **kwargs) -> tuple[str, MassProperties]:
    """One-call export: mesh file in, written MJCF path and properties out."""
    verts, faces = load_mesh(mesh_path)
    props = estimate_physical_properties(verts, faces, material=material)
    body = SceneBody(name=name, mesh_path=mesh_path, properties=props,
                     position=(0.0, 0.0, drop_height))
    return write_mjcf([body], out_path, **kwargs), props


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def load_model(xml_path: str):
    """Compile an MJCF file into a MuJoCo model and matching data buffer.

    Returns (model, data). Import is local so the geometry and export halves of
    this module stay usable in an environment without MuJoCo installed.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml_path)
    return model, mujoco.MjData(model)


def body_poses(model, data) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Current position and quaternion of every named body except the world.

    Body 0 is always the worldbody and is skipped.
    """
    import mujoco

    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        poses[name or f"body_{i}"] = (np.array(data.xpos[i]),
                                      np.array(data.xquat[i]))
    return poses


def step_simulation(model, data, n_steps: int = 2000,
                    record_every: int = 0) -> list[dict]:
    """Advance the simulation `n_steps` times.

    With `record_every` positive a snapshot of all body poses is kept every that
    many steps, which is what you plot to see whether the scene is settling or
    diverging. Returns the list of snapshots, empty if nothing was recorded.
    """
    import mujoco

    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    history: list[dict] = []
    for step in range(n_steps):
        mujoco.mj_step(model, data)
        if record_every and step % record_every == 0:
            history.append({"step": step, "time": float(data.time),
                            "poses": body_poses(model, data)})
    return history


def settle_scene(xml_path: str, n_steps: int = 2000,
                 record_every: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load an MJCF scene, run it for `n_steps`, return final body poses.

    `mj_forward` is called once before stepping so that the returned poses are
    meaningful even when `n_steps` is zero: without it the data buffer still
    holds the uninitialised state rather than the model's keyframe.
    """
    import mujoco

    model, data = load_model(xml_path)
    mujoco.mj_forward(model, data)
    step_simulation(model, data, n_steps=n_steps, record_every=record_every)
    return body_poses(model, data)


def kinetic_energy(model, data) -> float:
    """Total kinetic energy, the scalar that tells you whether motion stopped.

    `mj_energyVel` evaluates 0.5 v^T M v using MuJoCo's own sparse mass matrix,
    so it needs `mj_forward` or `mj_step` to have run at least once on this data
    buffer. A scene that is settling has this falling towards zero; one that is
    exploding has it growing without bound, which is the failure mode a broken
    reconstruction produces.
    """
    import mujoco

    mujoco.mj_energyVel(model, data)
    return float(data.energy[1])


def potential_energy(model, data) -> float:
    """Gravitational plus elastic potential energy, from `mj_energyPos`."""
    import mujoco

    mujoco.mj_energyPos(model, data)
    return float(data.energy[0])

#!/usr/bin/env python3
"""Export a reconstructed mesh to MJCF, then load and step it in MuJoCo.

Mesh in, physics out. The script computes mass properties from the triangles
under a stated material density, writes the MJCF, compiles it with MuJoCo, runs
the scene for a while and prints where every body ended up along with the
remaining kinetic energy. A reconstruction that is geometrically broken shows up
here as a body that never settles.

    python scripts/export_to_mujoco.py --mesh results/scene.obj --out results/scene.xml
    python scripts/export_to_mujoco.py --mesh results/scene.obj --out results/scene.xml \
        --material wood --steps 4000 --drop-height 1.0

Passing --no-simulate writes the XML and stops, which is the mode to use on a
machine without MuJoCo installed.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sceneforge.sim import (SceneBody, estimate_physical_properties,  # noqa: E402
                            kinetic_energy, load_mesh, load_model,
                            step_simulation, write_mjcf, body_poses)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mesh", required=True, help="reconstructed mesh (.obj, .stl, .ply)")
    ap.add_argument("--out", required=True, help="MJCF path to write")
    ap.add_argument("--name", default="scene", help="body name in the MJCF")
    ap.add_argument("--material", default="generic",
                    help="density assumption: generic, wood, plastic, concrete, "
                         "steel, aluminium, rubber")
    ap.add_argument("--drop-height", type=float, default=0.5,
                    help="initial z of the body above the floor, metres")
    ap.add_argument("--steps", type=int, default=2000, help="simulation steps")
    ap.add_argument("--timestep", type=float, default=0.002, help="seconds per step")
    ap.add_argument("--static", action="store_true",
                    help="weld the body to the world instead of giving it a freejoint")
    ap.add_argument("--no-simulate", action="store_true",
                    help="write the MJCF and exit without loading MuJoCo")
    args = ap.parse_args(argv)

    verts, faces = load_mesh(args.mesh)
    print(f"{args.mesh}: {len(verts)} vertices, {len(faces)} triangles")
    if len(faces) == 0:
        print("mesh has no triangles; nothing to export", file=sys.stderr)
        return 2

    props = estimate_physical_properties(verts, faces, material=args.material)
    moments, _ = props.principal
    print(f"material={args.material} density={props.density:g} kg/m^3")
    print(f"volume {props.volume:.6g} m^3  mass {props.mass:.4g} kg")
    print(f"centre of mass {np.round(props.com, 4).tolist()}")
    print(f"principal moments {np.round(moments, 6).tolist()} kg m^2")
    if not props.is_physical():
        print("WARNING: inertia tensor is not physically realisable. The mesh is "
              "probably open or inconsistently wound, so mass and inertia below "
              "are not trustworthy.", file=sys.stderr)

    body = SceneBody(name=args.name, mesh_path=args.mesh, properties=props,
                     position=(0.0, 0.0, args.drop_height), free=not args.static)
    path = write_mjcf([body], args.out, timestep=args.timestep)
    print(f"wrote {path}")

    if args.no_simulate:
        return 0

    try:
        model, data = load_model(path)
    except ImportError:
        print("mujoco is not installed; the MJCF was written but not simulated. "
              "Install it with `pip install mujoco` or rerun with --no-simulate.",
              file=sys.stderr)
        return 1

    import mujoco

    mujoco.mj_forward(model, data)
    print(f"compiled: {model.nbody - 1} bodies, {model.ngeom} geoms, "
          f"{model.nq} position coordinates")
    print(f"energy before stepping: {kinetic_energy(model, data):.6g} J")

    history = step_simulation(model, data, n_steps=args.steps,
                              record_every=max(args.steps // 5, 1))
    for snap in history:
        pos = next(iter(snap["poses"].values()))[0]
        print(f"  t={snap['time']:6.3f}s  {args.name} z={pos[2]:.4f}")

    print(f"energy after {args.steps} steps: {kinetic_energy(model, data):.6g} J")
    for name, (pos, quat) in body_poses(model, data).items():
        print(f"final pose {name}: pos={np.round(pos, 4).tolist()} "
              f"quat={np.round(quat, 4).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

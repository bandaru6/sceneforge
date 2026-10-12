"""Mass properties, OBJ round-tripping and MJCF structure.

The inertia checks are against the closed-form tensor of a box, which is the
only way to tell a correct implementation from one that is merely plausible.
MuJoCo itself is only exercised where it is installed; the geometry and XML
halves are tested unconditionally.
"""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from sceneforge.sim import (SceneBody, build_mjcf, estimate_physical_properties,
                            load_obj, mass_properties, mesh_to_mjcf)


def write_obj(path, verts, faces):
    """Write a triangle mesh as OBJ.

    Deliberately independent of `sceneforge.nerf.write_obj`: the physics half of
    the pipeline does not need PyTorch, and these tests should still run on a
    machine where it is absent.
    """
    with open(path, "w") as fh:
        for v in verts:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces:
            fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")
    return path


def box_mesh(sx=1.0, sy=1.0, sz=1.0, centre=(0.0, 0.0, 0.0)):
    """A closed, outward-wound axis-aligned box of the given side lengths."""
    half = np.array([sx, sy, sz], dtype=float) / 2.0
    signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    verts = signs * half + np.asarray(centre, dtype=float)
    faces = np.array([[0, 2, 1], [0, 3, 2],       # -z
                      [4, 5, 6], [4, 6, 7],       # +z
                      [0, 1, 5], [0, 5, 4],       # -y
                      [1, 2, 6], [1, 6, 5],       # +x
                      [2, 3, 7], [2, 7, 6],       # +y
                      [3, 0, 4], [3, 4, 7]])      # -x
    return verts, faces


def test_unit_cube_volume_mass_and_centroid():
    verts, faces = box_mesh(centre=(0.3, -0.2, 1.5))
    props = mass_properties(verts, faces, density=1000.0)
    assert props.volume == pytest.approx(1.0)
    assert props.mass == pytest.approx(1000.0)
    assert np.allclose(props.com, [0.3, -0.2, 1.5], atol=1e-12)


def test_box_inertia_matches_the_closed_form():
    """For a box, I_xx = m (b^2 + c^2) / 12 and the tensor is diagonal."""
    sx, sy, sz, rho = 2.0, 3.0, 0.5, 800.0
    verts, faces = box_mesh(sx, sy, sz)
    props = mass_properties(verts, faces, density=rho)
    m = rho * sx * sy * sz
    expected = np.diag([m * (sy ** 2 + sz ** 2) / 12.0,
                        m * (sx ** 2 + sz ** 2) / 12.0,
                        m * (sx ** 2 + sy ** 2) / 12.0])
    assert props.mass == pytest.approx(m)
    assert np.allclose(props.inertia, expected, rtol=1e-10, atol=1e-10)
    assert props.is_physical()


def test_inertia_is_about_the_centre_of_mass_not_the_origin():
    """Moving the mesh must not change the inertia tensor.

    This is the mistake the parallel-axis shift in `mass_properties` exists to
    prevent: without it, inertia would grow with distance from the origin.
    """
    at_origin = mass_properties(*box_mesh(1.0, 2.0, 3.0), density=500.0)
    moved = mass_properties(*box_mesh(1.0, 2.0, 3.0, centre=(10.0, -4.0, 7.0)),
                            density=500.0)
    assert np.allclose(at_origin.inertia, moved.inertia, rtol=1e-9, atol=1e-9)


def test_inward_wound_mesh_still_gives_positive_mass():
    """Flipped winding gives negative signed volume; mass must stay positive."""
    verts, faces = box_mesh()
    flipped = faces[:, ::-1]
    props = mass_properties(verts, flipped, density=1000.0)
    assert props.volume == pytest.approx(1.0)
    assert props.mass > 0
    assert props.is_physical()


def test_mass_scales_linearly_with_density():
    verts, faces = box_mesh(1.0, 1.0, 2.0)
    light = mass_properties(verts, faces, density=100.0)
    heavy = mass_properties(verts, faces, density=1000.0)
    assert heavy.mass == pytest.approx(10.0 * light.mass)
    assert np.allclose(heavy.inertia, 10.0 * light.inertia)


def test_degenerate_and_invalid_meshes_are_rejected():
    verts, faces = box_mesh()
    with pytest.raises(ValueError):
        mass_properties(verts, np.zeros((0, 3), dtype=int))
    with pytest.raises(ValueError):
        mass_properties(verts, faces, density=0.0)
    # A flat sheet through the origin encloses no volume, so it has no mass.
    sheet_v = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],
                        [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    sheet_f = np.array([[0, 1, 2], [0, 2, 3]])
    with pytest.raises(ValueError):
        mass_properties(sheet_v, sheet_f)


def test_material_table_is_consulted_and_unknown_names_raise():
    verts, faces = box_mesh()
    steel = estimate_physical_properties(verts, faces, material="steel")
    wood = estimate_physical_properties(verts, faces, material="wood")
    assert steel.mass > wood.mass
    assert steel.friction != wood.friction
    with pytest.raises(KeyError):
        estimate_physical_properties(verts, faces, material="unobtainium")


def test_obj_round_trip_preserves_the_mesh(tmp_path):
    verts, faces = box_mesh(1.0, 2.0, 3.0)
    path = str(tmp_path / "box.obj")
    write_obj(path, verts.astype(np.float32), faces.astype(np.int32))
    back_v, back_f = load_obj(path)
    assert np.allclose(back_v, verts, atol=1e-5)
    assert np.array_equal(back_f, faces)
    # Properties computed from the reloaded mesh must agree with the original.
    a = mass_properties(verts, faces, density=1000.0)
    b = mass_properties(back_v, back_f, density=1000.0)
    assert b.volume == pytest.approx(a.volume, rel=1e-5)


def test_mjcf_carries_the_computed_mass_and_inertia(tmp_path):
    verts, faces = box_mesh(1.0, 1.0, 1.0)
    mesh_path = str(tmp_path / "cube.obj")
    write_obj(mesh_path, verts.astype(np.float32), faces.astype(np.int32))
    props = estimate_physical_properties(verts, faces, material="concrete")
    body = SceneBody(name="cube", mesh_path=mesh_path, properties=props,
                     position=(0.0, 0.0, 1.0))

    root = ET.fromstring(build_mjcf([body]))
    assert root.tag == "mujoco"
    assert root.find("asset/mesh").get("file") == "cube.obj"
    assert root.find("worldbody/geom[@name='floor']") is not None

    node = root.find("worldbody/body[@name='cube']")
    assert node is not None
    assert node.find("freejoint") is not None
    inertial = node.find("inertial")
    assert float(inertial.get("mass")) == pytest.approx(props.mass, rel=1e-5)
    full = [float(x) for x in inertial.get("fullinertia").split()]
    assert len(full) == 6
    assert full[:3] == pytest.approx(list(np.diag(props.inertia)), rel=1e-5)
    # A diagonal tensor must have zero products of inertia.
    assert full[3:] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert node.find("geom").get("type") == "mesh"


def test_static_body_has_no_freejoint(tmp_path):
    verts, faces = box_mesh()
    mesh_path = str(tmp_path / "cube.obj")
    write_obj(mesh_path, verts.astype(np.float32), faces.astype(np.int32))
    props = mass_properties(verts, faces)
    body = SceneBody(name="cube", mesh_path=mesh_path, properties=props, free=False)
    root = ET.fromstring(build_mjcf([body]))
    assert root.find("worldbody/body[@name='cube']/freejoint") is None


def test_build_mjcf_rejects_meshes_in_different_directories(tmp_path):
    verts, faces = box_mesh()
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    props = mass_properties(verts, faces)
    bodies = []
    for name, folder in (("one", a), ("two", b)):
        path = str(folder / f"{name}.obj")
        write_obj(path, verts.astype(np.float32), faces.astype(np.int32))
        bodies.append(SceneBody(name=name, mesh_path=path, properties=props))
    with pytest.raises(ValueError):
        build_mjcf(bodies)
    with pytest.raises(ValueError):
        build_mjcf([])


def test_mujoco_compiles_the_generated_scene_and_it_settles(tmp_path):
    """End to end, where MuJoCo is installed: the box must come to rest.

    A unit cube dropped from 1 m must settle with its centre half a side above
    the floor, and its kinetic energy must be essentially zero. If the MJCF were
    malformed or the inertia impossible, neither would hold.
    """
    mujoco = pytest.importorskip("mujoco")
    from sceneforge.sim import kinetic_energy, settle_scene

    verts, faces = box_mesh(1.0, 1.0, 1.0)
    mesh_path = str(tmp_path / "cube.obj")
    write_obj(mesh_path, verts.astype(np.float32), faces.astype(np.int32))
    xml_path, props = mesh_to_mjcf(mesh_path, str(tmp_path / "cube.xml"),
                                   name="cube", material="wood", drop_height=1.0)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    assert model.body_mass[1] == pytest.approx(props.mass, rel=1e-4)
    assert model.body_inertia[1] == pytest.approx(
        np.diag(props.inertia), rel=1e-3)

    poses = settle_scene(xml_path, n_steps=3000)
    assert np.isclose(poses["cube"][0][2], 0.5, atol=1e-2)

    mujoco.mj_forward(model, data)
    for _ in range(3000):
        mujoco.mj_step(model, data)
    assert kinetic_energy(model, data) < 1e-6

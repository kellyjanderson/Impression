"""Example Impression model that returns simple PyVista meshes."""

from __future__ import annotations

import pyvista as pv


def build():
    """Compose a cube with a chamfered corner to exercise the previewer."""

    cube = pv.Cube(center=(0, 0, 0), x_length=12, y_length=12, z_length=12)
    chamfer = pv.Cylinder(
        center=(6, 6, 0),
        direction=(1, 1, 0),
        radius=4,
        height=14,
        resolution=64,
    ).rotate_z(45, inplace=False)

    return [cube, chamfer]

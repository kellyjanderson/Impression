"""Example Impression model that returns simple meshes."""

from __future__ import annotations

from impression.modeling import make_box, make_cylinder, rotate


def build():
    """Compose a cube and a cylinder to exercise the previewer."""

    cube = make_box(size=(12, 12, 12), center=(0, 0, 0))
    chamfer = make_cylinder(
        center=(6, 6, 0),
        direction=(1, 1, 0),
        radius=4,
        height=14,
        resolution=64,
    )
    rotate(chamfer, axis=(0, 0, 1), angle_deg=45)

    return [cube, chamfer]

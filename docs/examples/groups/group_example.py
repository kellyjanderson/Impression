"""Group example: manipulate meshes together without boolean fusion.

Run with:
  impression preview docs/examples/groups/group_example.py
"""
from __future__ import annotations

from impression.modeling import make_box, make_cylinder, group


def build():
    box = make_box(size=(2.0, 1.0, 0.5), center=(0.0, 0.0, 0.25), color="#5A7BFF")
    cyl = make_cylinder(radius=0.4, height=1.0, center=(0.0, 0.0, 0.5), color="#FF7A18")
    grp = group([box, cyl])
    grp.translate((0.5, 0.0, 0.2))
    grp.rotate(axis=(0, 0, 1), angle_deg=30)
    return grp


if __name__ == "__main__":
    import pyvista as pv

    scene = build()
    pv.plot(scene)

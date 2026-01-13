"""Demonstrate translate() and rotate() helpers.

Run with:
  impression preview docs/examples/transforms/transform_example.py
"""

from __future__ import annotations

from impression.modeling import make_box, rotate, translate


def build():
    base = make_box(size=(0.8, 0.8, 0.4), center=(0.0, 0.0, 0.2), color="#5A7BFF")
    shifted = translate(base.copy(), (1.2, 0.0, 0.0))
    turned = rotate(base.copy(), axis=(0.0, 0.0, 1.0), angle_deg=45.0)
    return [base, shifted, turned]


if __name__ == "__main__":
    import pyvista as pv
    from impression.mesh import mesh_to_pyvista

    meshes = build()
    plotter = pv.Plotter()
    for i, m in enumerate(meshes):
        plotter.add_mesh(mesh_to_pyvista(m), show_edges=True, color=[0.6, 0.8, 1.0], name=f"mesh-{i}")
    plotter.show()

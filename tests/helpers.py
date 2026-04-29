from __future__ import annotations

import pyvista as pv

def is_watertight(mesh: pv.DataSet) -> tuple[bool, int]:
    edges = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, non_manifold_edges=True, manifold_edges=False)
    open_edges = edges.n_cells
    return open_edges == 0, open_edges

def mesh_volume(mesh: pv.DataSet) -> float | None:
    vol = getattr(mesh, 'volume', None)
    try:
        return float(vol) if vol is not None else None
    except Exception:
        return None

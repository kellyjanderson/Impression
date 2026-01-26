from __future__ import annotations

from pathlib import Path

import pytest

from impression.mesh import Mesh, mesh_to_pyvista
from tests.conftest import load_scene_datasets, project_root
from tests.helpers import is_watertight, mesh_volume


@pytest.mark.preview
@pytest.mark.stl
def test_tombstone_union_matches_parts(project_root: Path):
    parts_model = project_root / 'docs/examples/csg/tooth_example.py'
    union_model = project_root / 'docs/examples/csg/tooth_union_example.py'

    parts_datasets = load_scene_datasets(parts_model)
    union_datasets = load_scene_datasets(union_model)

    parts_meshes = [mesh for mesh in parts_datasets if isinstance(mesh, Mesh)]
    union_meshes = [mesh for mesh in union_datasets if isinstance(mesh, Mesh)]

    assert len(parts_meshes) >= 2
    assert len(union_meshes) == 1

    pv_parts = [mesh_to_pyvista(mesh) for mesh in parts_meshes]
    pv_union = mesh_to_pyvista(union_meshes[0])

    # Union should be smaller than the sum of parts and larger than any single part.
    part_volumes = [mesh_volume(mesh) for mesh in pv_parts]
    assert all(volume is not None for volume in part_volumes)
    union_volume = mesh_volume(pv_union)
    assert union_volume is not None
    assert union_volume <= sum(part_volumes)
    assert union_volume >= max(part_volumes)

    # Union should reduce overall triangle counts.
    assert pv_union.n_cells <= sum(mesh.n_cells for mesh in pv_parts)
    assert pv_union.n_points <= sum(mesh.n_points for mesh in pv_parts)

    # Each part and the union should be watertight.
    for mesh in pv_parts:
        wt_part, open_part = is_watertight(mesh)
        assert wt_part and open_part == 0
    wt_union, open_union = is_watertight(pv_union)
    assert wt_union and open_union == 0

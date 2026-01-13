from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import load_scene_mesh, project_root
from tests.helpers import is_watertight, mesh_volume


@pytest.mark.preview
@pytest.mark.stl
def test_tombstone_union_matches_parts(project_root: Path):
    parts_model = project_root / 'docs/examples/csg/tooth_example.py'
    union_model = project_root / 'docs/examples/csg/tooth_union_example.py'

    parts = load_scene_mesh(parts_model)
    union = load_scene_mesh(union_model)

    # Basic counts
    assert union.n_cells == parts.n_cells
    assert union.n_points == parts.n_points

    # Volumes should match closely
    pv_parts = mesh_volume(parts)
    pv_union = mesh_volume(union)
    assert pv_parts is not None and pv_union is not None
    assert abs(pv_parts - pv_union) < 1e-3

    # Watertightness
    wt_parts, open_parts = is_watertight(parts)
    wt_union, open_union = is_watertight(union)
    assert wt_parts and open_parts == 0
    assert wt_union and open_union == 0

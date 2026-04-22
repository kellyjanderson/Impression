from __future__ import annotations

import warnings

import numpy as np

from impression.modeling.csg import union_meshes
from impression.modeling.drafting import make_line
from impression.modeling.heightmap import heightmap
from impression.modeling._legacy_mesh_deprecation import reset_legacy_mesh_deprecation_warnings
from impression.modeling.tessellation import SurfaceMeshAdapter, TessellationRequest, mesh_from_surface_body
from impression.modeling.topology import Loop, Region, Section
from impression.modeling.primitives import make_box


def _simple_section() -> Section:
    return Section(
        regions=(
            Region(
                outer=Loop(
                    np.asarray(
                        [
                            [0.0, 0.0],
                            [1.0, 0.0],
                            [1.0, 1.0],
                            [0.0, 1.0],
                        ],
                        dtype=float,
                    )
                ),
                holes=(),
            ),
        )
    )


def _messages_from(callable_obj) -> list[warnings.WarningMessage]:
    reset_legacy_mesh_deprecation_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        callable_obj()
    return [item for item in caught if issubclass(item.category, DeprecationWarning)]


def test_mesh_only_public_apis_warn() -> None:
    draft_messages = _messages_from(lambda: make_line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    heightmap_messages = _messages_from(
        lambda: heightmap(np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float))
    )
    union_messages = _messages_from(lambda: union_meshes([make_box(size=(1.0, 1.0, 1.0), backend="mesh")]))

    assert any("make_line" in str(item.message) for item in draft_messages)
    assert any("heightmap" in str(item.message) for item in heightmap_messages)
    assert any("union_meshes" in str(item.message) for item in union_messages)


def test_legacy_mesh_bridge_warns() -> None:
    body = make_box(backend="surface")
    adapter_messages = _messages_from(lambda: SurfaceMeshAdapter(request=TessellationRequest()))
    bridge_messages = _messages_from(lambda: mesh_from_surface_body(body))

    assert any("SurfaceMeshAdapter" in str(item.message) for item in adapter_messages)
    assert any("mesh_from_surface_body" in str(item.message) for item in bridge_messages)

from __future__ import annotations

from typing import Iterable

from impression.mesh import Mesh
from impression.modeling.drawing2d import Path2D
from impression.modeling.group import MeshGroup
from impression.modeling.topology import Region, Section

from ._ops_mesh import hull_mesh
from ._ops_planar import hull_planar, is_planar_shape, offset_planar


def offset(
    shape: Section | Region | Path2D,
    r: float | None = None,
    delta: float | None = None,
    chamfer: bool = False,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
) -> list[Section]:
    """Offset a 2D profile/path.

    Topology-native API:
    - accepts ``Section``/``Region`` or closed ``Path2D``
    - returns one or more ``Section`` values

    Legacy shape objects with ``outer``/``holes`` path fields are adapted via
    topology normalization.
    """

    return offset_planar(
        shape=shape,
        r=r,
        delta=delta,
        chamfer=chamfer,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
    )


def hull(shapes: Iterable[Mesh | MeshGroup | Section | Region | Path2D]) -> Mesh | list[Section]:
    """Compute convex hull for either planar or mesh inputs.

    Dispatches to internal domain modules:

    - planar inputs (`Section`, `Region`, `Path2D`) -> `_ops_planar`
    - mesh inputs (`Mesh`, `MeshGroup`) -> `_ops_mesh`

    Mesh hull behavior is retained as an explicit standalone utility. It is not
    canonical surfaced modeling truth.
    """

    items = list(shapes)
    if not items:
        raise ValueError("hull requires at least one shape.")

    if is_planar_shape(items[0]):
        if not all(is_planar_shape(item) for item in items):
            raise TypeError("hull cannot mix planar and mesh inputs.")
        return hull_planar(items)

    if not all(isinstance(item, (Mesh, MeshGroup)) for item in items):
        raise TypeError("hull cannot mix planar and mesh inputs.")
    return hull_mesh(items)


__all__ = ["offset", "hull"]

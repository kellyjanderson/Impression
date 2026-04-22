from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .surface import SurfaceBody
from .tessellation import SurfaceConsumerCollection, SurfaceConsumerRecord


def _normalize_metadata(metadata: dict[str, object] | None) -> dict[str, object]:
    return {} if metadata is None else dict(metadata)


def _as_matrix4(value: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    matrix = np.asarray(value, dtype=float).reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError("transform_matrix must contain only finite values.")
    return matrix


@dataclass(frozen=True)
class SurfaceSceneNode:
    """A deterministic scene node that references one surface body."""

    node_id: str
    body: SurfaceBody
    transform_matrix: np.ndarray = field(default_factory=lambda: np.eye(4))
    metadata: dict[str, object] = field(default_factory=dict)
    visible: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", str(self.node_id))
        object.__setattr__(self, "transform_matrix", _as_matrix4(self.transform_matrix))
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    def resolved_body(self) -> SurfaceBody:
        return self.body.with_transform(self.transform_matrix)


@dataclass(frozen=True)
class SurfaceSceneGroup:
    """A structural grouping of surface scene nodes and nested groups."""

    group_id: str
    children: tuple[SurfaceSceneNode | SurfaceSceneGroup, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", str(self.group_id))
        object.__setattr__(self, "children", tuple(self.children))
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))


def iter_surface_scene_nodes(root: SurfaceSceneNode | SurfaceSceneGroup) -> tuple[SurfaceSceneNode, ...]:
    if isinstance(root, SurfaceSceneNode):
        return (root,)
    ordered: list[SurfaceSceneNode] = []
    for child in root.children:
        ordered.extend(iter_surface_scene_nodes(child))
    return tuple(ordered)


def flatten_surface_scene(
    root: SurfaceSceneNode | SurfaceSceneGroup,
    *,
    collection_metadata: dict[str, object] | None = None,
) -> SurfaceConsumerCollection:
    nodes = [node for node in iter_surface_scene_nodes(root) if node.visible]
    records = tuple(
        SurfaceConsumerRecord(
            body=node.resolved_body(),
            source_id=node.node_id,
            order=index,
            metadata=dict(node.metadata),
        )
        for index, node in enumerate(nodes)
    )
    return SurfaceConsumerCollection(items=records, metadata=_normalize_metadata(collection_metadata))


def handoff_surface_scene(
    root: SurfaceSceneNode | SurfaceSceneGroup,
    *,
    collection_metadata: dict[str, object] | None = None,
) -> SurfaceConsumerCollection:
    """Return a surface-native consumer collection without tessellating."""

    return flatten_surface_scene(root, collection_metadata=collection_metadata)


def make_surface_scene_node(
    node_id: str,
    body: SurfaceBody,
    *,
    transform_matrix: np.ndarray | Iterable[Iterable[float]] | None = None,
    metadata: dict[str, object] | None = None,
    visible: bool = True,
) -> SurfaceSceneNode:
    return SurfaceSceneNode(
        node_id=node_id,
        body=body,
        transform_matrix=np.eye(4) if transform_matrix is None else transform_matrix,
        metadata=_normalize_metadata(metadata),
        visible=visible,
    )


def make_surface_scene_group(
    group_id: str,
    children: Iterable[SurfaceSceneNode | SurfaceSceneGroup],
    *,
    metadata: dict[str, object] | None = None,
) -> SurfaceSceneGroup:
    return SurfaceSceneGroup(group_id=group_id, children=tuple(children), metadata=_normalize_metadata(metadata))


__all__ = [
    "SurfaceSceneGroup",
    "SurfaceSceneNode",
    "flatten_surface_scene",
    "handoff_surface_scene",
    "iter_surface_scene_nodes",
    "make_surface_scene_group",
    "make_surface_scene_node",
]

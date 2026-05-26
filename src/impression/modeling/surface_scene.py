from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np

from impression.mesh import Mesh

from .surface import SurfaceBody, _compose_transform, _stable_hash
from .tessellation import SurfaceConsumerCollection, SurfaceConsumerRecord


def _normalize_metadata(metadata: dict[str, object] | None) -> dict[str, object]:
    return {} if metadata is None else dict(metadata)


def _as_matrix4(value: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    matrix = np.asarray(value, dtype=float).reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError("transform_matrix must contain only finite values.")
    return matrix


@dataclass(frozen=True)
class SurfaceCompositionUnsupportedDiagnostic:
    """Explicit refusal for values that cannot enter authored surface composition."""

    reason: str
    target_type: str
    child_index: int | None = None
    boundary: str = "surface-composition"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "boundary": self.boundary,
            "child_index": self.child_index,
            "reason": self.reason,
            "target_type": self.target_type,
        }


class SurfaceCompositionError(TypeError):
    """Raised when authored surface composition receives unsupported input."""

    def __init__(self, diagnostic: SurfaceCompositionUnsupportedDiagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(
            f"Unsupported {diagnostic.target_type} at surface composition boundary: {diagnostic.reason}."
        )


def surface_composition_unsupported_diagnostic(
    target: object,
    *,
    child_index: int | None = None,
    reason: str | None = None,
) -> SurfaceCompositionUnsupportedDiagnostic:
    target_type = type(target).__name__
    if reason is None:
        if isinstance(target, Mesh):
            reason = "mesh inputs must stay behind an explicit tessellation or compatibility boundary"
        else:
            reason = "expected SurfaceBody or SurfaceComposition"
    return SurfaceCompositionUnsupportedDiagnostic(
        reason=reason,
        target_type=target_type,
        child_index=child_index,
    )


def _normalize_composition_child(child: object, index: int) -> SurfaceBody | "SurfaceComposition":
    if isinstance(child, (SurfaceBody, SurfaceComposition)):
        return child
    raise SurfaceCompositionError(surface_composition_unsupported_diagnostic(child, child_index=index))


def _composition_child_payload(child: SurfaceBody | "SurfaceComposition") -> dict[str, object]:
    if isinstance(child, SurfaceBody):
        return {
            "kind": "surface-body",
            "stable_identity": child.stable_identity,
        }
    return {
        "kind": "surface-composition",
        "stable_identity": child.stable_identity,
    }


@dataclass(frozen=True)
class SurfaceComposition:
    """Public authored grouping type for surface bodies and nested surface compositions."""

    composition_id: str
    children: tuple[SurfaceBody | "SurfaceComposition", ...]
    transform_matrix: np.ndarray = field(default_factory=lambda: np.eye(4))
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_children = tuple(
            _normalize_composition_child(child, index) for index, child in enumerate(self.children)
        )
        if not normalized_children:
            raise SurfaceCompositionError(
                SurfaceCompositionUnsupportedDiagnostic(
                    reason="surface composition requires at least one child",
                    target_type="empty",
                )
            )
        object.__setattr__(self, "composition_id", str(self.composition_id))
        object.__setattr__(self, "children", normalized_children)
        object.__setattr__(self, "transform_matrix", _as_matrix4(self.transform_matrix))
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    @property
    def stable_identity(self) -> str:
        return _stable_hash(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        return {
            "composition_id": self.composition_id,
            "children": [_composition_child_payload(child) for child in self.children],
            "transform_matrix": self.transform_matrix,
            "metadata": self.metadata,
        }

    def with_transform(self, matrix: np.ndarray | Iterable[Iterable[float]]) -> "SurfaceComposition":
        applied = _as_matrix4(matrix)
        return replace(self, transform_matrix=_compose_transform(self.transform_matrix, applied))

    def iter_children(self) -> tuple[SurfaceBody | "SurfaceComposition", ...]:
        return self.children


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


def make_surface_composition(
    children: Iterable[SurfaceBody | SurfaceComposition],
    *,
    composition_id: str = "surface-composition",
    transform_matrix: np.ndarray | Iterable[Iterable[float]] | None = None,
    metadata: dict[str, object] | None = None,
) -> SurfaceComposition:
    """Create an authored surface composition without tessellating child bodies."""

    return SurfaceComposition(
        composition_id=composition_id,
        children=tuple(children),
        transform_matrix=np.eye(4) if transform_matrix is None else transform_matrix,
        metadata=_normalize_metadata(metadata),
    )


def surface_group(
    children: Iterable[SurfaceBody | SurfaceComposition],
    *,
    group_id: str = "surface-group",
    transform_matrix: np.ndarray | Iterable[Iterable[float]] | None = None,
    metadata: dict[str, object] | None = None,
) -> SurfaceComposition:
    """Public surface-native replacement for authored grouping."""

    return make_surface_composition(
        children,
        composition_id=group_id,
        transform_matrix=transform_matrix,
        metadata=metadata,
    )


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
    "SurfaceComposition",
    "SurfaceCompositionError",
    "SurfaceCompositionUnsupportedDiagnostic",
    "SurfaceSceneGroup",
    "SurfaceSceneNode",
    "flatten_surface_scene",
    "handoff_surface_scene",
    "iter_surface_scene_nodes",
    "make_surface_composition",
    "make_surface_scene_group",
    "make_surface_scene_node",
    "surface_composition_unsupported_diagnostic",
    "surface_group",
]

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np

from impression.mesh import Mesh

from .surface import SurfaceBody, _compose_transform, _stable_hash
from .tessellation import (
    SurfaceCollectionTessellationResult,
    SurfaceConsumerCollection,
    SurfaceConsumerRecord,
    TessellationRequest,
    NormalizedTessellationRequest,
    tessellate_surface_consumer_collection,
)


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


@dataclass(frozen=True)
class SurfaceCompositionTraversalRecord:
    """Deterministic traversal record for one body emitted from a composition."""

    source_id: str
    order: int
    depth: int
    body_identity: str
    transform_matrix: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", str(self.source_id))
        object.__setattr__(self, "order", int(self.order))
        object.__setattr__(self, "depth", int(self.depth))
        object.__setattr__(self, "body_identity", str(self.body_identity))
        object.__setattr__(self, "transform_matrix", _as_matrix4(self.transform_matrix))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "order": self.order,
            "depth": self.depth,
            "body_identity": self.body_identity,
            "transform_matrix": self.transform_matrix,
        }


@dataclass(frozen=True)
class SurfaceCompositionTraversalDiagnostic:
    """Summary diagnostic for a completed surface composition traversal."""

    composition_id: str
    emitted_count: int
    traversal_order: tuple[str, ...]
    boundary: str = "surface-composition-traversal"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "boundary": self.boundary,
            "composition_id": self.composition_id,
            "emitted_count": self.emitted_count,
            "traversal_order": self.traversal_order,
        }


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


def traverse_surface_composition(root: SurfaceComposition) -> tuple[SurfaceCompositionTraversalRecord, ...]:
    """Return deterministic pre-order traversal records without tessellating."""

    records, _items = _traverse_surface_composition(root, parent_transform=np.eye(4), path=(root.composition_id,))
    return records


def surface_composition_to_consumer_collection(
    root: SurfaceComposition,
    *,
    collection_metadata: dict[str, object] | None = None,
) -> SurfaceConsumerCollection:
    """Convert authored surface composition into tessellation-ready surface bodies."""

    records, items = _traverse_surface_composition(root, parent_transform=np.eye(4), path=(root.composition_id,))
    diagnostic = SurfaceCompositionTraversalDiagnostic(
        composition_id=root.composition_id,
        emitted_count=len(records),
        traversal_order=tuple(record.source_id for record in records),
    )
    metadata = _normalize_metadata(collection_metadata)
    metadata.setdefault("surface_composition_id", root.composition_id)
    metadata.setdefault("surface_composition_traversal", diagnostic.canonical_payload())
    return SurfaceConsumerCollection(items=tuple(items), metadata=metadata)


def handoff_surface_composition(
    root: SurfaceComposition,
    *,
    collection_metadata: dict[str, object] | None = None,
) -> SurfaceConsumerCollection:
    """Return a surface-native consumer collection from a composition without tessellating."""

    return surface_composition_to_consumer_collection(root, collection_metadata=collection_metadata)


def tessellate_surface_composition(
    root: SurfaceComposition,
    request: TessellationRequest | NormalizedTessellationRequest | None = None,
    *,
    collection_metadata: dict[str, object] | None = None,
) -> SurfaceCollectionTessellationResult:
    """Explicitly tessellate a surface composition at the mesh boundary."""

    collection = handoff_surface_composition(root, collection_metadata=collection_metadata)
    return tessellate_surface_consumer_collection(collection, request)


def _traverse_surface_composition(
    root: SurfaceComposition,
    *,
    parent_transform: np.ndarray,
    path: tuple[str, ...],
) -> tuple[list[SurfaceCompositionTraversalRecord], list[SurfaceConsumerRecord]]:
    composed_transform = _compose_transform(root.transform_matrix, parent_transform)
    records: list[SurfaceCompositionTraversalRecord] = []
    items: list[SurfaceConsumerRecord] = []
    for child_index, child in enumerate(root.children):
        child_path = (*path, str(child_index))
        if isinstance(child, SurfaceBody):
            body = child.with_transform(composed_transform)
            source_id = "/".join(child_path)
            record = SurfaceCompositionTraversalRecord(
                source_id=source_id,
                order=len(items),
                depth=len(path),
                body_identity=body.stable_identity,
                transform_matrix=composed_transform,
            )
            records.append(record)
            items.append(
                SurfaceConsumerRecord(
                    body=body,
                    source_id=source_id,
                    order=record.order,
                    metadata={"traversal": record.canonical_payload(), **body.consumer_metadata()},
                )
            )
            continue
        nested_records, nested_items = _traverse_surface_composition(
            child,
            parent_transform=composed_transform,
            path=(*child_path, child.composition_id),
        )
        offset = len(items)
        for record, item in zip(nested_records, nested_items):
            shifted_record = SurfaceCompositionTraversalRecord(
                source_id=record.source_id,
                order=offset + record.order,
                depth=record.depth,
                body_identity=record.body_identity,
                transform_matrix=record.transform_matrix,
            )
            records.append(shifted_record)
            items.append(
                SurfaceConsumerRecord(
                    body=item.body,
                    source_id=item.source_id,
                    order=shifted_record.order,
                    metadata={**item.metadata, "traversal": shifted_record.canonical_payload()},
                )
            )
    return records, items


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
    "SurfaceCompositionTraversalDiagnostic",
    "SurfaceCompositionTraversalRecord",
    "SurfaceCompositionUnsupportedDiagnostic",
    "SurfaceSceneGroup",
    "SurfaceSceneNode",
    "flatten_surface_scene",
    "handoff_surface_composition",
    "handoff_surface_scene",
    "iter_surface_scene_nodes",
    "make_surface_composition",
    "make_surface_scene_group",
    "make_surface_scene_node",
    "surface_composition_to_consumer_collection",
    "surface_composition_unsupported_diagnostic",
    "surface_group",
    "tessellate_surface_composition",
    "traverse_surface_composition",
]

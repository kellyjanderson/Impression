from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence
import re

import numpy as np

from impression.modeling.drawing2d import Path2D
from impression.modeling.bspline import BSpline2D
from ._color import _normalize_color


def _normalize_loop_points(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if pts.shape[0] == 0:
        return pts
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


def derive_stable_id(name: str) -> str:
    """Return a deterministic id-like token for a user-facing topology name."""

    token = re.sub(r"[^a-zA-Z0-9]+", "-", str(name).strip().lower()).strip("-")
    return token or "unnamed"


def _normalize_provenance(provenance: dict[str, object] | None) -> dict[str, object]:
    return dict(provenance or {})


def _require_finite_vec2(value: Sequence[float], field_name: str) -> np.ndarray:
    point = np.asarray(value, dtype=float).reshape(2)
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{field_name} must contain finite coordinates.")
    return point


def _require_nonempty_builder_token(value: object, field_name: str) -> str:
    token = str(value).strip()
    if not token:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return token


def _optional_builder_token(value: object | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_builder_token(value, field_name)


def _normalize_segment_points(
    points: Iterable[Sequence[float]] | None,
    field_name: str,
) -> tuple[np.ndarray, ...]:
    if points is None:
        return ()
    normalized = tuple(_require_finite_vec2(point, field_name) for point in points)
    if len(normalized) < 2:
        raise ValueError(f"{field_name} must contain at least two points.")
    return normalized


@dataclass(frozen=True)
class TopologyPathSamplingPolicy:
    sample_count: int | str = "auto"
    min_span_samples: int = 1
    preserve_protected_landmarks: bool = True

    def __post_init__(self) -> None:
        sample_count = self.sample_count
        if sample_count != "auto":
            if not isinstance(sample_count, int) or sample_count <= 0:
                raise ValueError("sample_count must be 'auto' or a positive integer.")
        min_span_samples = int(self.min_span_samples)
        if min_span_samples <= 0:
            raise ValueError("min_span_samples must be positive.")
        object.__setattr__(self, "min_span_samples", min_span_samples)
        object.__setattr__(self, "preserve_protected_landmarks", bool(self.preserve_protected_landmarks))


@dataclass(frozen=True)
class TopologyPoint:
    id: str | None
    coordinates: np.ndarray | Sequence[float]
    ordinal: int
    role: str | None = None
    name: str | None = None
    correspondence_id: str | None = None
    protection_policy: str | None = None
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        point_id = self.id
        provenance = _normalize_provenance(self.provenance)
        if point_id is None:
            if self.name is not None:
                point_id = derive_stable_id(self.name)
                provenance.setdefault("id_source", "derived_from_name")
            else:
                point_id = f"point-{int(self.ordinal)}"
                provenance.setdefault("id_source", "derived_from_ordinal")
        if not str(point_id):
            raise ValueError("TopologyPoint id must not be empty.")
        ordinal = int(self.ordinal)
        if ordinal < 0:
            raise ValueError("TopologyPoint ordinal must be non-negative.")
        protection_policy = self.protection_policy
        if protection_policy is None:
            protection_policy = (
                "protected"
                if self.correspondence_id is not None or self.role in {"corner", "seam", "feature", "tangent_transition"}
                else "sample"
            )
        if protection_policy not in {"protected", "sample", "synthetic"}:
            raise ValueError("TopologyPoint protection_policy must be 'protected', 'sample', or 'synthetic'.")
        object.__setattr__(self, "id", str(point_id))
        object.__setattr__(self, "coordinates", _require_finite_vec2(self.coordinates, "TopologyPoint.coordinates"))
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "protection_policy", protection_policy)
        object.__setattr__(self, "provenance", provenance)


@dataclass(frozen=True)
class TopologyLandmark:
    id: str | None = None
    name: str | None = None
    segment_id: str | None = None
    parameter: float | None = None
    point_ordinal: int | None = None
    role: str | None = None
    correspondence_id: str | None = None
    protection_policy: str | None = None
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        landmark_id = self.id
        provenance = _normalize_provenance(self.provenance)
        if landmark_id is None:
            if self.name is not None:
                landmark_id = derive_stable_id(self.name)
                provenance.setdefault("id_source", "derived_from_name")
            elif self.point_ordinal is not None:
                landmark_id = f"point-{int(self.point_ordinal)}-landmark"
                provenance.setdefault("id_source", "derived_from_point_ordinal")
            elif self.segment_id is not None and self.parameter is not None:
                landmark_id = f"{self.segment_id}-u-{float(self.parameter):.6g}"
                provenance.setdefault("id_source", "derived_from_segment_parameter")
            else:
                raise ValueError("TopologyLandmark requires id, name, point_ordinal, or segment parameter.")
        if self.parameter is not None:
            parameter = float(self.parameter)
            if not np.isfinite(parameter):
                raise ValueError("TopologyLandmark parameter must be finite.")
            object.__setattr__(self, "parameter", parameter)
        if self.point_ordinal is not None:
            point_ordinal = int(self.point_ordinal)
            if point_ordinal < 0:
                raise ValueError("TopologyLandmark point_ordinal must be non-negative.")
            object.__setattr__(self, "point_ordinal", point_ordinal)
        protection_policy = self.protection_policy
        if protection_policy is None:
            protection_policy = (
                "protected"
                if self.correspondence_id is not None or self.role in {"corner", "seam", "feature", "tangent_transition"}
                else "sample"
            )
        if protection_policy not in {"protected", "sample", "synthetic"}:
            raise ValueError("TopologyLandmark protection_policy must be 'protected', 'sample', or 'synthetic'.")
        object.__setattr__(self, "id", str(landmark_id))
        object.__setattr__(self, "protection_policy", protection_policy)
        object.__setattr__(self, "provenance", provenance)


@dataclass(frozen=True)
class TopologySegment:
    id: str | None = None
    name: str | None = None
    source_kind: str = "polyline"
    start_ref: str | None = None
    end_ref: str | None = None
    points: tuple[np.ndarray, ...] | Iterable[Sequence[float]] | None = None
    curve: object | None = None
    correspondence_id: str | None = None
    landmarks: tuple[TopologyLandmark, ...] = ()
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        segment_id = self.id
        provenance = _normalize_provenance(self.provenance)
        if segment_id is None:
            if self.name is not None:
                segment_id = derive_stable_id(self.name)
                provenance.setdefault("id_source", "derived_from_name")
            else:
                segment_id = f"segment-{self.source_kind}"
                provenance.setdefault("id_source", "derived_from_source_kind")
        if not str(segment_id):
            raise ValueError("TopologySegment id must not be empty.")
        landmarks = tuple(self.landmarks)
        points = _normalize_segment_points(self.points, "TopologySegment points")
        if points and self.curve is not None:
            raise ValueError("TopologySegment cannot define both points and curve.")
        correspondence_id = _optional_builder_token(self.correspondence_id, "TopologySegment correspondence_id")
        source_kind = str(self.source_kind)
        if points and source_kind == "curve":
            source_kind = "polyline"
        if self.curve is not None and source_kind == "polyline":
            source_kind = "curve"
        object.__setattr__(self, "id", str(segment_id))
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "correspondence_id", correspondence_id)
        object.__setattr__(self, "landmarks", landmarks)
        object.__setattr__(self, "provenance", provenance)


def _normalize_lifecycle_parent(parent: tuple[str, str] | Sequence[str]) -> tuple[str, str]:
    parent_items = tuple(str(item) for item in parent)
    if len(parent_items) != 2 or not all(parent_items):
        raise ValueError("Lifecycle parent must be a two-item point span.")
    if parent_items[0] == parent_items[1]:
        raise ValueError("Lifecycle parent span endpoints must be distinct.")
    return parent_items


def _normalize_lifecycle_points(
    points: Iterable[tuple[str, Sequence[float]]] | None,
) -> tuple[tuple[str, tuple[float, float]], ...]:
    normalized: list[tuple[str, tuple[float, float]]] = []
    for name, coordinates in points or ():
        point_name = str(name)
        if not point_name:
            raise ValueError("Lifecycle point names must not be empty.")
        point = _require_finite_vec2(coordinates, "lifecycle point coordinates")
        normalized.append((point_name, (float(point[0]), float(point[1]))))
    return tuple(normalized)


@dataclass(frozen=True)
class TopologyLifecycleBuilderRequest:
    request_type: str
    parent: tuple[str, str] | Sequence[str]
    points: tuple[tuple[str, Sequence[float]], ...] | Iterable[tuple[str, Sequence[float]]] = ()
    curve: object | None = None
    name: str | None = None
    radius: float | None = None
    names: tuple[str, ...] | Iterable[str] = ()
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        request_type = str(self.request_type)
        if request_type not in {"birth_span", "birth_arc", "death_span"}:
            raise ValueError("TopologyLifecycleBuilderRequest request_type is invalid.")
        parent = _normalize_lifecycle_parent(self.parent)
        points = _normalize_lifecycle_points(self.points)
        names = tuple(str(name) for name in self.names)
        if any(not name for name in names):
            raise ValueError("Lifecycle target names must not be empty.")
        radius = self.radius
        if radius is not None:
            radius = float(radius)
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError("radius must be positive.")
        object.__setattr__(self, "request_type", request_type)
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "provenance", _normalize_provenance(self.provenance))


@dataclass(frozen=True)
class GeneratedRailProvenance:
    shape_kind: str
    name_prefix: str | None
    generated_role: str
    source_parameter: str | None = None
    overridden: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape_kind", str(self.shape_kind))
        object.__setattr__(self, "name_prefix", None if self.name_prefix is None else str(self.name_prefix))
        object.__setattr__(self, "generated_role", str(self.generated_role))
        object.__setattr__(
            self,
            "source_parameter",
            None if self.source_parameter is None else str(self.source_parameter),
        )
        object.__setattr__(self, "overridden", bool(self.overridden))


def _generated_rail_name(base_name: str, name_prefix: str | None) -> str:
    if name_prefix is None:
        return base_name
    return f"{derive_stable_id(name_prefix)}-{base_name}"


def _generated_rail_provenance(
    shape_kind: str,
    name_prefix: str | None,
    generated_role: str,
    source_parameter: str | None = None,
    *,
    overridden: bool = False,
) -> dict[str, object]:
    return {
        "source": "generated_shape_default_rails",
        "generated_rail": GeneratedRailProvenance(
            shape_kind=shape_kind,
            name_prefix=name_prefix,
            generated_role=generated_role,
            source_parameter=source_parameter,
            overridden=overridden,
        ),
    }


def _resolve_generated_anchor(anchor: str | None, valid_base_names: Sequence[str], name_prefix: str | None) -> str:
    base_anchor = anchor or valid_base_names[0]
    generated_names = {_generated_rail_name(name, name_prefix): name for name in valid_base_names}
    if base_anchor in valid_base_names:
        return _generated_rail_name(base_anchor, name_prefix)
    if base_anchor in generated_names:
        return base_anchor
    valid = ", ".join(valid_base_names)
    raise ValueError(f"Invalid generated topology anchor {anchor!r}; valid anchors: {valid}.")


@dataclass(frozen=True)
class TopologyPath:
    id: str = "topology-path"
    closed: bool = True
    anchor_id: str | None = None
    anchor_policy: str = "authored"
    direction: str = "forward"
    sampling_policy: TopologyPathSamplingPolicy | dict[str, object] | None = None
    points: tuple[TopologyPoint, ...] = ()
    segments: tuple[TopologySegment, ...] = ()
    landmarks: tuple[TopologyLandmark, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sampling_policy = self.sampling_policy
        if sampling_policy is None:
            sampling_policy = TopologyPathSamplingPolicy()
        elif isinstance(sampling_policy, dict):
            sampling_policy = TopologyPathSamplingPolicy(**sampling_policy)
        elif not isinstance(sampling_policy, TopologyPathSamplingPolicy):
            raise TypeError("sampling_policy must be TopologyPathSamplingPolicy, dict, or None.")
        object.__setattr__(self, "closed", bool(self.closed))
        object.__setattr__(self, "sampling_policy", sampling_policy)
        object.__setattr__(self, "points", tuple(self.points))
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "landmarks", tuple(self.landmarks))
        object.__setattr__(self, "metadata", dict(self.metadata))
        self.validate()

    @classmethod
    def from_points(
        cls,
        points: Iterable[Sequence[float] | tuple[str, Sequence[float]]],
        *,
        closed: bool = True,
        anchor: str | None = None,
        direction: str = "forward",
        landmarks: Iterable[TopologyLandmark] | None = None,
        sampling_policy: TopologyPathSamplingPolicy | dict[str, object] | None = None,
        id: str = "topology-path",
    ) -> "TopologyPath":
        topology_points: list[TopologyPoint] = []
        for ordinal, raw in enumerate(points):
            name: str | None = None
            coords: Sequence[float]
            if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], str):
                name = raw[0]
                coords = raw[1]
            else:
                coords = raw  # type: ignore[assignment]
            topology_points.append(
                TopologyPoint(
                    id=None,
                    name=name,
                    coordinates=coords,
                    ordinal=ordinal,
                    role="corner" if name is not None else None,
                    correspondence_id=name,
                    provenance={"source": "from_points"},
                )
            )
        anchor_id = anchor
        if anchor_id is None and topology_points:
            anchor_id = topology_points[0].id
        return cls(
            id=id,
            closed=closed,
            anchor_id=anchor_id,
            anchor_policy="authored",
            direction=direction,
            sampling_policy=sampling_policy,
            points=tuple(topology_points),
            landmarks=tuple(landmarks or ()),
            metadata={"source": "from_points"},
        )

    @classmethod
    def from_path2d(
        cls,
        path: Path2D,
        *,
        closed: bool | None = None,
        anchor: str | None = None,
        landmarks: Iterable[TopologyLandmark] | None = None,
        sampling_policy: TopologyPathSamplingPolicy | dict[str, object] | None = None,
        id: str = "topology-path",
        segments_per_circle: int = 64,
        bezier_samples: int = 32,
    ) -> "TopologyPath":
        is_closed = path.closed if closed is None else bool(closed)
        pts = _normalize_loop_points(path.sample(segments_per_circle=segments_per_circle, bezier_samples=bezier_samples))
        result = cls.from_points(
            pts,
            closed=is_closed,
            anchor=anchor,
            landmarks=landmarks,
            sampling_policy=sampling_policy,
            id=id,
        )
        return cls(
            id=result.id,
            closed=result.closed,
            anchor_id=result.anchor_id,
            anchor_policy=result.anchor_policy,
            direction=result.direction,
            sampling_policy=result.sampling_policy,
            points=result.points,
            landmarks=result.landmarks,
            metadata={"source": "from_path2d", "path_closed": path.closed},
        )

    @classmethod
    def from_bspline(
        cls,
        curve: BSpline2D,
        *,
        closed: bool | None = None,
        anchor: str | None = None,
        landmarks: Iterable[TopologyLandmark] | None = None,
        sampling_policy: TopologyPathSamplingPolicy | dict[str, object] | None = None,
        id: str = "topology-path",
    ) -> "TopologyPath":
        is_closed = curve.closure != "open" if closed is None else bool(closed)
        segment_id = f"{id}-bspline"
        normalized_landmarks = tuple(
            TopologyLandmark(
                id=landmark.id,
                name=landmark.name,
                segment_id=landmark.segment_id or segment_id,
                parameter=landmark.parameter,
                point_ordinal=landmark.point_ordinal,
                role=landmark.role,
                correspondence_id=landmark.correspondence_id,
                protection_policy=landmark.protection_policy,
                provenance={**landmark.provenance, "source": "from_bspline"},
            )
            for landmark in (landmarks or ())
        )
        return cls(
            id=id,
            closed=is_closed,
            anchor_id=anchor,
            anchor_policy="authored",
            sampling_policy=sampling_policy,
            segments=(
                TopologySegment(
                    id=segment_id,
                    name="bspline",
                    source_kind="bspline",
                    curve=curve,
                    landmarks=normalized_landmarks,
                    provenance={"source": "from_bspline"},
                ),
            ),
            landmarks=normalized_landmarks,
            metadata={"source": "from_bspline"},
        )

    @classmethod
    def closed(
        cls,
        *,
        anchor: str | None = None,
        direction: str = "forward",
        sampling_policy: TopologyPathSamplingPolicy | dict[str, object] | None = None,
        id: str = "topology-path",
    ) -> "TopologyPathBuilder":
        return TopologyPathBuilder(id=id, closed=True, anchor=anchor, direction=direction, sampling_policy=sampling_policy)

    @classmethod
    def named_rect(
        cls,
        width: float,
        height: float,
        *,
        anchor: str = "bottom-left",
        name_prefix: str | None = None,
    ) -> "TopologyPath":
        width = float(width)
        height = float(height)
        if not np.isfinite(width) or width <= 0 or not np.isfinite(height) or height <= 0:
            raise ValueError("named_rect width and height must be positive.")
        half_w = width / 2.0
        half_h = height / 2.0
        rails = (
            ("bottom-left", (-half_w, -half_h), "corner"),
            ("bottom-mid", (0.0, -half_h), "side_midpoint"),
            ("bottom-right", (half_w, -half_h), "corner"),
            ("right-mid", (half_w, 0.0), "side_midpoint"),
            ("top-right", (half_w, half_h), "corner"),
            ("top-mid", (0.0, half_h), "side_midpoint"),
            ("top-left", (-half_w, half_h), "corner"),
            ("left-mid", (-half_w, 0.0), "side_midpoint"),
        )
        return cls._from_generated_point_rails(
            "rect",
            rails,
            anchor=anchor,
            name_prefix=name_prefix,
            metadata={"width": width, "height": height},
        )

    @classmethod
    def named_circle(
        cls,
        radius: float,
        *,
        anchor: str | None = None,
        name_prefix: str | None = None,
    ) -> "TopologyPath":
        radius = float(radius)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("named_circle radius must be positive.")
        rails = (
            ("start", (radius, 0.0), "start"),
            ("north", (0.0, radius), "quadrant"),
            ("west", (-radius, 0.0), "quadrant"),
            ("south", (0.0, -radius), "quadrant"),
        )
        return cls._from_generated_point_rails(
            "circle",
            rails,
            anchor=anchor or "start",
            name_prefix=name_prefix,
            metadata={"radius": radius},
        )

    @classmethod
    def named_rounded_rect(
        cls,
        width: float,
        height: float,
        radius: float,
        *,
        anchor: str = "bottom-left",
        name_prefix: str | None = None,
    ) -> "TopologyPath":
        width = float(width)
        height = float(height)
        radius = float(radius)
        if not np.isfinite(width) or width <= 0 or not np.isfinite(height) or height <= 0:
            raise ValueError("named_rounded_rect width and height must be positive.")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("named_rounded_rect radius must be positive.")
        if radius > min(width, height) / 2.0:
            raise ValueError("named_rounded_rect radius must fit within width and height.")

        half_w = width / 2.0
        half_h = height / 2.0
        rails = (
            ("bottom-left", (-half_w + radius, -half_h), "tangent_transition"),
            ("bottom-right", (half_w - radius, -half_h), "tangent_transition"),
            ("right-bottom", (half_w, -half_h + radius), "tangent_transition"),
            ("right-top", (half_w, half_h - radius), "tangent_transition"),
            ("top-right", (half_w - radius, half_h), "tangent_transition"),
            ("top-left", (-half_w + radius, half_h), "tangent_transition"),
            ("left-top", (-half_w, half_h - radius), "tangent_transition"),
            ("left-bottom", (-half_w, -half_h + radius), "tangent_transition"),
        )
        points = cls._generated_points("rounded_rect", rails, name_prefix)
        segment_specs = (
            ("bottom", "bottom-left", "bottom-right", "straight"),
            ("bottom-right-arc", "bottom-right", "right-bottom", "corner_arc"),
            ("right", "right-bottom", "right-top", "straight"),
            ("top-right-arc", "right-top", "top-right", "corner_arc"),
            ("top", "top-right", "top-left", "straight"),
            ("top-left-arc", "top-left", "left-top", "corner_arc"),
            ("left", "left-top", "left-bottom", "straight"),
            ("bottom-left-arc", "left-bottom", "bottom-left", "corner_arc"),
        )
        point_by_base = {base_name: point for point, (base_name, _, _) in zip(points, rails, strict=True)}
        segments: list[TopologySegment] = []
        landmarks: list[TopologyLandmark] = []
        for segment_name, start_name, end_name, role in segment_specs:
            generated_name = _generated_rail_name(segment_name, name_prefix)
            segment_id = derive_stable_id(generated_name)
            segment_landmarks = (
                TopologyLandmark(
                    name=f"{generated_name}-start",
                    segment_id=segment_id,
                    parameter=0.0,
                    role="tangent_transition",
                    correspondence_id=point_by_base[start_name].correspondence_id,
                    provenance=_generated_rail_provenance("rounded_rect", name_prefix, "tangent_transition", start_name),
                ),
                TopologyLandmark(
                    name=f"{generated_name}-end",
                    segment_id=segment_id,
                    parameter=1.0,
                    role="tangent_transition",
                    correspondence_id=point_by_base[end_name].correspondence_id,
                    provenance=_generated_rail_provenance("rounded_rect", name_prefix, "tangent_transition", end_name),
                ),
            )
            landmarks.extend(segment_landmarks)
            segments.append(
                TopologySegment(
                    id=segment_id,
                    name=generated_name,
                    source_kind=role,
                    start_ref=point_by_base[start_name].id,
                    end_ref=point_by_base[end_name].id,
                    correspondence_id=segment_id,
                    landmarks=segment_landmarks,
                    provenance=_generated_rail_provenance("rounded_rect", name_prefix, role, segment_name),
                )
            )
        return cls(
            id=_generated_rail_name("rounded-rect", name_prefix),
            closed=True,
            anchor_id=_resolve_generated_anchor(anchor, [name for name, _, _ in rails], name_prefix),
            anchor_policy="generated",
            points=tuple(points),
            segments=tuple(segments),
            landmarks=tuple(landmarks),
            metadata={
                "source": "generated_shape_default_rails",
                "shape_kind": "rounded_rect",
                "width": width,
                "height": height,
                "radius": radius,
                "name_prefix": name_prefix,
            },
        )

    @classmethod
    def _from_generated_point_rails(
        cls,
        shape_kind: str,
        rails: Sequence[tuple[str, tuple[float, float], str]],
        *,
        anchor: str,
        name_prefix: str | None,
        metadata: dict[str, object],
    ) -> "TopologyPath":
        return cls(
            id=_generated_rail_name(shape_kind, name_prefix),
            closed=True,
            anchor_id=_resolve_generated_anchor(anchor, [name for name, _, _ in rails], name_prefix),
            anchor_policy="generated",
            points=tuple(cls._generated_points(shape_kind, rails, name_prefix)),
            metadata={
                "source": "generated_shape_default_rails",
                "shape_kind": shape_kind,
                "name_prefix": name_prefix,
                **metadata,
            },
        )

    @staticmethod
    def _generated_points(
        shape_kind: str,
        rails: Sequence[tuple[str, tuple[float, float], str]],
        name_prefix: str | None,
    ) -> tuple[TopologyPoint, ...]:
        points: list[TopologyPoint] = []
        seen_ids: set[str] = set()
        for ordinal, (base_name, coordinates, role) in enumerate(rails):
            generated_name = _generated_rail_name(base_name, name_prefix)
            point_id = derive_stable_id(generated_name)
            if point_id in seen_ids:
                raise ValueError(f"Generated topology rail collision for {point_id!r}.")
            seen_ids.add(point_id)
            points.append(
                TopologyPoint(
                    id=point_id,
                    name=generated_name,
                    coordinates=coordinates,
                    ordinal=ordinal,
                    role=role,
                    correspondence_id=point_id,
                    provenance=_generated_rail_provenance(shape_kind, name_prefix, role, base_name),
                )
            )
        return tuple(points)

    def validate(self) -> None:
        if not self.id:
            raise ValueError("TopologyPath id must not be empty.")
        if self.anchor_policy not in {"authored", "named", "generated", "inferred"}:
            raise ValueError("anchor_policy must be authored, named, generated, or inferred.")
        if self.direction not in {"forward", "reverse"}:
            raise ValueError("direction must be 'forward' or 'reverse'.")
        if not self.points and not self.segments:
            raise ValueError("TopologyPath requires at least one point or segment.")
        if self.closed and not self.segments and len(self.points) < 3:
            raise ValueError("Closed TopologyPath requires at least three points.")
        point_ids = [point.id for point in self.points]
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("TopologyPath contains duplicate point ids.")
        if self.anchor_id is not None and point_ids and self.anchor_id not in point_ids:
            raise ValueError(f"TopologyPath anchor_id {self.anchor_id!r} does not match a point id.")
        validate_topology_identity_records(self)

    def to_section_loop(self) -> Loop:
        if self.points:
            return Loop(np.vstack([point.coordinates for point in self.points]))
        if len(self.segments) == 1 and self.segments[0].points:
            return Loop(np.vstack(self.segments[0].points))
        if len(self.segments) == 1 and hasattr(self.segments[0].curve, "sample"):
            return Loop(np.asarray(self.segments[0].curve.sample(), dtype=float))
        raise ValueError("TopologyPath cannot be converted to a Loop without point or sampleable curve data.")


@dataclass
class TopologyPathBuilder:
    id: str = "topology-path"
    closed: bool = True
    anchor: str | None = None
    direction: str = "forward"
    sampling_policy: TopologyPathSamplingPolicy | dict[str, object] | None = None
    points: list[TopologyPoint] = field(default_factory=list)
    segments: list[TopologySegment] = field(default_factory=list)
    landmarks: list[TopologyLandmark] = field(default_factory=list)
    lifecycle_requests: list[TopologyLifecycleBuilderRequest] = field(default_factory=list)

    def point(
        self,
        name: str,
        coordinates: Sequence[float],
        *,
        id: str | None = None,
        correspond: str | None = None,
        anchor: bool = False,
        role: str | None = None,
    ) -> "TopologyPathBuilder":
        point_name = _require_nonempty_builder_token(name, "point name")
        point_id = _optional_builder_token(id, "point id")
        correspondence_id = _optional_builder_token(correspond, "point correspond")
        point = TopologyPoint(
            id=point_id,
            name=point_name,
            coordinates=coordinates,
            ordinal=len(self.points),
            role=role or "corner",
            correspondence_id=correspondence_id if correspondence_id is not None else point_name,
            provenance={"source": "builder.point"},
        )
        self._validate_new_point(point)
        if anchor:
            if self.anchor is not None and self.anchor != point.id:
                raise ValueError(f"point anchor conflicts with existing path anchor {self.anchor!r}.")
            self.anchor = point.id
        self.points.append(point)
        return self

    def segment(
        self,
        name: str,
        *,
        id: str | None = None,
        points: Iterable[Sequence[float]] | None = None,
        curve: object | None = None,
        landmarks: Iterable[TopologyLandmark] | None = None,
        correspond: str | None = None,
    ) -> "TopologyPathBuilder":
        segment_name = _require_nonempty_builder_token(name, "segment name")
        segment_id = _optional_builder_token(id, "segment id") or derive_stable_id(segment_name)
        if points is not None and curve is not None:
            raise ValueError("segment cannot define both points and curve.")
        segment_landmarks = tuple(landmarks or ())
        segment = TopologySegment(
            id=segment_id,
            name=segment_name,
            source_kind="curve" if curve is not None else "polyline",
            points=None if points is None else tuple(points),
            curve=curve,
            correspondence_id=_optional_builder_token(correspond, "segment correspond"),
            landmarks=segment_landmarks,
            provenance={"source": "builder.segment"},
        )
        self._validate_new_segment(segment)
        self.segments.append(segment)
        self.landmarks.extend(segment_landmarks)
        return self

    def birth_span(self, parent: tuple[str, str], *, points: Iterable[tuple[str, Sequence[float]]]) -> "TopologyPathBuilder":
        point_items = _normalize_lifecycle_points(points)
        if not point_items:
            raise ValueError("birth_span requires at least one point.")
        self.lifecycle_requests.append(
            TopologyLifecycleBuilderRequest(
                request_type="birth_span",
                parent=parent,
                points=point_items,
                provenance={"source": "builder.birth_span"},
            )
        )
        for name, coordinates in point_items:
            self.point(name, coordinates, correspond=name, role="feature")
        return self

    def birth_arc(
        self,
        name: str,
        *,
        parent: tuple[str, str],
        start: Sequence[float],
        end: Sequence[float],
        radius: float,
        correspond: str | None = None,
    ) -> "TopologyPathBuilder":
        start_point = _require_finite_vec2(start, "birth_arc start")
        end_point = _require_finite_vec2(end, "birth_arc end")
        if np.allclose(start_point, end_point):
            raise ValueError("birth_arc endpoints must be distinct.")
        if not np.isfinite(float(radius)) or radius <= 0:
            raise ValueError("radius must be positive.")
        curve = {
            "kind": "birth_arc",
            "start": (float(start_point[0]), float(start_point[1])),
            "end": (float(end_point[0]), float(end_point[1])),
            "radius": float(radius),
        }
        self.lifecycle_requests.append(
            TopologyLifecycleBuilderRequest(
                request_type="birth_arc",
                parent=parent,
                points=((f"{name}-start", curve["start"]), (f"{name}-end", curve["end"])),
                curve=curve,
                name=name,
                radius=radius,
                provenance={"source": "builder.birth_arc"},
            )
        )
        self.point(f"{name}-start", curve["start"], correspond=correspond or f"{name}-start", role="tangent_transition")
        self.point(f"{name}-end", curve["end"], correspond=correspond or f"{name}-end", role="tangent_transition")
        segment_id = derive_stable_id(name)
        segment_landmarks = (
            TopologyLandmark(
                name=f"{name}-start",
                segment_id=segment_id,
                parameter=0.0,
                role="tangent_transition",
                correspondence_id=correspond or f"{name}-start",
                provenance={"source": "builder.birth_arc"},
            ),
            TopologyLandmark(
                name=f"{name}-end",
                segment_id=segment_id,
                parameter=1.0,
                role="tangent_transition",
                correspondence_id=correspond or f"{name}-end",
                provenance={"source": "builder.birth_arc"},
            ),
        )
        self.segments.append(
            TopologySegment(
                id=segment_id,
                name=name,
                source_kind="curve",
                curve=curve,
                correspondence_id=correspond or name,
                landmarks=segment_landmarks,
                provenance={"source": "builder.birth_arc"},
            )
        )
        self.landmarks.extend(segment_landmarks)
        return self

    def death_span(
        self,
        parent: tuple[str, str],
        *,
        points: Iterable[tuple[str, Sequence[float]]] | None = None,
        names: Iterable[str] | None = None,
    ) -> "TopologyPathBuilder":
        point_items = _normalize_lifecycle_points(points)
        name_items = tuple(str(name) for name in names or ())
        if not point_items and not name_items:
            raise ValueError("death_span requires points or names.")
        self.lifecycle_requests.append(
            TopologyLifecycleBuilderRequest(
                request_type="death_span",
                parent=parent,
                points=point_items,
                names=name_items,
                provenance={"source": "builder.death_span"},
            )
        )
        for name, coordinates in point_items:
            self.point(name, coordinates, correspond=name, role="feature")
        return self

    def build(self) -> TopologyPath:
        metadata: dict[str, object] = {"source": "builder"}
        if self.lifecycle_requests:
            self._validate_lifecycle_requests()
            metadata["lifecycle_requests"] = tuple(self.lifecycle_requests)
        return TopologyPath(
            id=self.id,
            closed=self.closed,
            anchor_id=self.anchor or (self.points[0].id if self.points else None),
            anchor_policy="authored",
            direction=self.direction,
            sampling_policy=self.sampling_policy,
            points=tuple(self.points),
            segments=tuple(self.segments),
            landmarks=tuple(self.landmarks),
            metadata=metadata,
        )

    def _validate_new_point(self, point: TopologyPoint) -> None:
        point_ids = {existing.id for existing in self.points}
        if point.id in point_ids:
            raise ValueError(f"point id {point.id!r} duplicates an existing topology point.")
        if point.name is not None:
            point_names = {existing.name for existing in self.points if existing.name is not None}
            if point.name in point_names:
                raise ValueError(f"point name {point.name!r} duplicates an existing topology point.")
        if point.correspondence_id is not None and not str(point.correspondence_id).strip():
            raise ValueError("point correspond must be a non-empty string.")

    def _validate_new_segment(self, segment: TopologySegment) -> None:
        segment_ids = {existing.id for existing in self.segments}
        if segment.id in segment_ids:
            raise ValueError(f"segment id {segment.id!r} duplicates an existing topology segment.")
        landmark_ids: set[str] = {landmark.id for landmark in self.landmarks}
        for landmark in segment.landmarks:
            if landmark.id in landmark_ids:
                raise ValueError(f"segment landmark id {landmark.id!r} duplicates an existing topology landmark.")
            landmark_ids.add(landmark.id)
        if self.closed and not self.points and segment.points and len(segment.points) < 3:
            raise ValueError("closed segment-only TopologyPath requires at least three authored segment points.")

    def _validate_lifecycle_requests(self) -> None:
        known_refs = {str(point.id) for point in self.points}
        known_refs.update(str(point.name) for point in self.points if point.name is not None)
        for request in self.lifecycle_requests:
            unknown_parent_refs = [ref for ref in request.parent if ref not in known_refs]
            if unknown_parent_refs:
                raise ValueError(f"Lifecycle parent references unknown point names: {unknown_parent_refs!r}.")
            if request.request_type == "death_span":
                unknown_names = [name for name in request.names if name not in known_refs]
                if unknown_names:
                    raise ValueError(f"death_span references unknown point names: {unknown_names!r}.")


def validate_topology_identity_records(path: TopologyPath) -> None:
    point_ids = [point.id for point in path.points]
    if len(set(point_ids)) != len(point_ids):
        raise ValueError("TopologyPath contains duplicate point ids.")
    segment_ids = [segment.id for segment in path.segments]
    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError("TopologyPath contains duplicate segment ids.")
    landmark_ids = [landmark.id for landmark in path.landmarks]
    if len(set(landmark_ids)) != len(landmark_ids):
        raise ValueError("TopologyPath contains duplicate landmark ids.")
    segment_id_set = set(segment_ids)
    point_count = len(path.points)
    correspondence_policies: dict[str, str] = {}
    for record in (*path.points, *path.landmarks):
        correspondence_id = record.correspondence_id
        if correspondence_id is None:
            continue
        policy = record.protection_policy
        prior_policy = correspondence_policies.setdefault(correspondence_id, policy)
        if prior_policy != policy:
            raise ValueError(f"Conflicting protection policies for correspondence id {correspondence_id!r}.")
    for landmark in path.landmarks:
        if landmark.segment_id is not None and segment_id_set and landmark.segment_id not in segment_id_set:
            raise ValueError(f"TopologyLandmark segment_id {landmark.segment_id!r} does not match a segment id.")
        if landmark.point_ordinal is not None and landmark.point_ordinal >= point_count:
            raise ValueError("TopologyLandmark point_ordinal is outside path points.")


@dataclass(frozen=True)
class Loop:
    """A closed 2D loop in local plane coordinates."""

    points: np.ndarray

    def __post_init__(self) -> None:
        pts = _normalize_loop_points(self.points)
        object.__setattr__(self, "points", pts)

    @property
    def area(self) -> float:
        return signed_area(self.points)

    @property
    def is_clockwise(self) -> bool:
        return self.area < 0.0

    @property
    def perimeter(self) -> float:
        if self.points.shape[0] < 2:
            return 0.0
        closed = np.vstack([self.points, self.points[0]])
        return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        if self.points.shape[0] == 0:
            return (0.0, 0.0, 0.0, 0.0)
        mins = self.points.min(axis=0)
        maxs = self.points.max(axis=0)
        return (float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]))

    @property
    def centroid(self) -> np.ndarray:
        if self.points.shape[0] == 0:
            return np.zeros(2, dtype=float)
        return self.points.mean(axis=0)

    def with_winding(self, clockwise: bool) -> "Loop":
        return Loop(ensure_winding(self.points, clockwise=clockwise))

    def resampled(self, count: int) -> "Loop":
        return Loop(resample_loop(self.points, count))

    def anchored(self) -> "Loop":
        return Loop(anchor_loop(self.points))

    def to_path(self) -> Path2D:
        return Path2D.from_points(self.points, closed=True)


@dataclass(frozen=True)
class Region:
    """A planar solid region with one outer loop and zero or more holes."""

    outer: Loop
    holes: tuple[Loop, ...] = ()

    def normalized(self) -> "Region":
        outer = self.outer.with_winding(clockwise=False)
        holes = tuple(hole.with_winding(clockwise=True) for hole in self.holes)
        return Region(outer=outer, holes=holes)

    def is_valid(self) -> bool:
        normalized = self.normalized()
        for hole in normalized.holes:
            if hole.points.shape[0] < 3:
                return False
            if not point_in_polygon(hole.points[0], normalized.outer.points):
                return False
        return normalized.outer.points.shape[0] >= 3

@dataclass(frozen=True)
class Section:
    """A planar collection of one or more disconnected regions."""

    regions: tuple[Region, ...] = field(default_factory=tuple)
    color: tuple[float, float, float, float] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def normalized(self) -> "Section":
        return Section(
            tuple(region.normalized() for region in self.regions),
            color=self.color,
            metadata=dict(self.metadata),
        )

    def with_color(self, color: Sequence[float] | str | None) -> "Section":
        rgba = None if color is None else _normalize_color(color)
        return Section(
            regions=self.regions,
            color=rgba,
            metadata=dict(self.metadata),
        )


def as_section(
    shape: Section | Region | Path2D | object,
    *,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
) -> Section:
    """Normalize a planar input shape into a topology-native Section."""

    if isinstance(shape, Section):
        return shape.normalized()
    if isinstance(shape, Region):
        return Section((shape.normalized(),))
    if isinstance(shape, Path2D):
        if not shape.closed:
            raise ValueError("Path2D must be closed for planar solid operations.")
        pts = _normalize_loop_points(shape.sample(segments_per_circle=segments_per_circle, bezier_samples=bezier_samples))
        if pts.shape[0] < 3:
            return Section(())
        return Section(
            (Region(outer=Loop(ensure_winding(pts, clockwise=False))),),
            color=shape.color,
            metadata=dict(shape.metadata),
        ).normalized()
    if hasattr(shape, "outer") and hasattr(shape, "holes"):
        outer = getattr(shape, "outer")
        holes = getattr(shape, "holes")
        if not isinstance(outer, Path2D):
            raise TypeError("Expected .outer to be a closed Path2D.")
        if not outer.closed:
            raise ValueError("Outer path must be closed for planar solid operations.")
        loops = [loop_points(outer, segments_per_circle, bezier_samples)]
        for hole in holes:
            if not isinstance(hole, Path2D):
                raise TypeError("Expected hole paths to be Path2D values.")
            if not hole.closed:
                raise ValueError("Hole paths must be closed for planar solid operations.")
            loops.append(loop_points(hole, segments_per_circle, bezier_samples))
        if not loops or loops[0].shape[0] < 3:
            return Section(())
        region = Region(
            outer=Loop(ensure_winding(loops[0], clockwise=False)),
            holes=tuple(Loop(ensure_winding(loop, clockwise=True)) for loop in loops[1:]),
        ).normalized()
        color = getattr(shape, "color", None)
        metadata = dict(getattr(shape, "metadata", {}) or {})
        return Section((region,), color=color, metadata=metadata)
    raise TypeError("Expected Section, Region, closed Path2D, or shape with .outer/.holes Path2D loops.")


def as_sections(
    shapes: Iterable[Section | Region | Path2D | object],
    *,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
) -> list[Section]:
    """Normalize multiple planar input shapes into topology-native Sections."""

    return [
        as_section(
            shape,
            segments_per_circle=segments_per_circle,
            bezier_samples=bezier_samples,
        )
        for shape in shapes
    ]


def signed_area(points: np.ndarray) -> float:
    pts = _normalize_loop_points(points)
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ensure_winding(points: np.ndarray, clockwise: bool) -> np.ndarray:
    pts = _normalize_loop_points(points)
    if pts.shape[0] < 3:
        return pts
    area = signed_area(pts)
    is_cw = area < 0.0
    if is_cw != clockwise:
        return pts[::-1].copy()
    return pts


def loop_points(
    path: Path2D,
    segments_per_circle: int,
    bezier_samples: int,
) -> np.ndarray:
    pts = path.sample(segments_per_circle=segments_per_circle, bezier_samples=bezier_samples)
    return _normalize_loop_points(pts)


def profile_loops(
    shape: Section | Region | Path2D | object,
    segments_per_circle: int,
    bezier_samples: int,
    enforce_winding: bool = True,
) -> list[np.ndarray]:
    section = as_section(
        shape,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
    ).normalized()
    if len(section.regions) != 1:
        raise ValueError("profile_loops requires one connected region.")
    region = section.regions[0]
    loops: list[np.ndarray] = []
    outer = _normalize_loop_points(region.outer.points)
    if enforce_winding:
        outer = ensure_winding(outer, clockwise=False)
    loops.append(outer)

    for hole in region.holes:
        pts = _normalize_loop_points(hole.points)
        if enforce_winding:
            pts = ensure_winding(pts, clockwise=True)
        loops.append(pts)
    return loops


def triangulate_profile(
    shape: Section | Region | Path2D | object,
    segments_per_circle: int,
    bezier_samples: int,
    enforce_winding: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    loops = profile_loops(
        shape,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
        enforce_winding=enforce_winding,
    )
    vertices, faces = triangulate_loops(loops)
    return vertices, faces, loops


def triangulate_loops(loops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not loops:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=int)
    try:
        import mapbox_earcut as earcut
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("mapbox_earcut is required for profile triangulation.") from exc

    normalized_loops = [_normalize_loop_points(loop) for loop in loops]
    vertices = np.vstack(normalized_loops).astype(np.float32)
    ring_ends = []
    offset = 0
    for pts in normalized_loops:
        offset += pts.shape[0]
        ring_ends.append(offset)
    ring_end_indices = np.asarray(ring_ends, dtype=np.uint32)
    indices = earcut.triangulate_float32(vertices, ring_end_indices)
    faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    if not _triangulation_covers_loop_boundaries(faces, normalized_loops):
        jittered_vertices = _jitter_vertices_for_triangulation(normalized_loops)
        indices = earcut.triangulate_float32(jittered_vertices, ring_end_indices)
        faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    return vertices.astype(float), faces


def _triangulation_covers_loop_boundaries(
    faces: np.ndarray,
    loops: list[np.ndarray],
) -> bool:
    edge_set: set[tuple[int, int]] = set()
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        edge_set.add((a, b) if a < b else (b, a))
        edge_set.add((b, c) if b < c else (c, b))
        edge_set.add((c, a) if c < a else (a, c))
    cursor = 0
    for loop in loops:
        count = int(loop.shape[0])
        if count < 3:
            return False
        for i in range(count):
            a = cursor + i
            b = cursor + ((i + 1) % count)
            key = (a, b) if a < b else (b, a)
            if key not in edge_set:
                return False
        cursor += count
    return True


def _jitter_vertices_for_triangulation(loops: list[np.ndarray]) -> np.ndarray:
    jittered: list[np.ndarray] = []
    for loop_index, loop in enumerate(loops):
        pts = np.asarray(loop, dtype=float)
        count = int(pts.shape[0])
        if count == 0:
            continue
        span = np.ptp(pts, axis=0)
        scale = max(float(np.hypot(span[0], span[1])), 1.0)
        eps = max(scale * 1e-3, 1e-6)
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False) + loop_index * 0.37
        perturb = np.column_stack((np.cos(angles), np.sin(angles))) * eps
        jittered.append((pts + perturb).astype(np.float32))
    if not jittered:
        return np.zeros((0, 2), dtype=np.float32)
    return np.vstack(jittered)


def resample_loop(points: np.ndarray, count: int) -> np.ndarray:
    if count < 3:
        raise ValueError("resample count must be >= 3.")
    pts = _normalize_loop_points(points)
    if pts.shape[0] < 2:
        raise ValueError("Loop requires at least two points.")
    if pts.shape[0] == count:
        return pts.copy()

    closed = np.vstack([pts, pts[0]])
    seg_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(seg_lengths.sum())
    if total == 0:
        return np.tile(pts[0], (count, 1))

    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    targets = np.linspace(0.0, total, count + 1)[:-1]
    result = []
    seg_index = 0
    for t in targets:
        while seg_index < len(seg_lengths) - 1 and cumulative[seg_index + 1] < t:
            seg_index += 1
        seg_start = cumulative[seg_index]
        seg_end = cumulative[seg_index + 1]
        p0 = closed[seg_index]
        p1 = closed[seg_index + 1]
        if seg_end == seg_start:
            result.append(p0)
        else:
            alpha = (t - seg_start) / (seg_end - seg_start)
            result.append((1 - alpha) * p0 + alpha * p1)
    return np.asarray(result, dtype=float)


def loops_resampled(
    shape: Section | Region | Path2D | object,
    count: int,
    segments_per_circle: int,
    bezier_samples: int,
    enforce_winding: bool = True,
) -> list[np.ndarray]:
    loops = profile_loops(
        shape,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
        enforce_winding=enforce_winding,
    )
    return [resample_loop(loop, count) for loop in loops]


def anchor_loop(loop: np.ndarray) -> np.ndarray:
    pts = _normalize_loop_points(loop)
    if pts.shape[0] == 0:
        return pts

    centroid = pts.mean(axis=0)
    rel = pts - centroid
    angles = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2.0 * np.pi)
    min_angle = np.min(angles)
    candidates = np.where(np.isclose(angles, min_angle))[0]
    if candidates.size == 1:
        idx = int(candidates[0])
    else:
        radii = np.linalg.norm(rel[candidates], axis=1)
        max_radius = np.max(radii)
        radius_candidates = candidates[np.where(np.isclose(radii, max_radius))[0]]
        idx = int(np.min(radius_candidates))
    return np.roll(pts, -idx, axis=0)


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x = float(point[0])
    y = float(point[1])
    poly = _normalize_loop_points(polygon)
    if poly.shape[0] < 3:
        return False

    inside = False
    n = poly.shape[0]
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        if intersect:
            inside = not inside
        j = i
    return inside


def largest_loop(contours: list[np.ndarray]) -> np.ndarray:
    if not contours:
        raise ValueError("largest_loop requires at least one contour.")
    return max(contours, key=lambda c: abs(signed_area(c)))


def classify_loops(loops: list[np.ndarray], expected_holes: int | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    if not loops:
        raise ValueError("classify_loops requires at least one loop.")
    outer = largest_loop(loops)
    holes = [loop for loop in loops if loop is not outer]
    for hole in holes:
        if not point_in_polygon(_normalize_loop_points(hole)[0], outer):
            raise ValueError("Loop set contains disconnected geometry.")
    outer = ensure_winding(outer, clockwise=False)
    holes = [ensure_winding(hole, clockwise=True) for hole in holes]
    if expected_holes is not None and len(holes) != expected_holes:
        raise ValueError("Loop classification changed hole count.")
    return outer, holes


def inset_profile_loops(
    shape: Section | Region | Path2D | object,
    inset: float,
    *,
    join_type: str = "ROUND",
    hole_count: int = 0,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
) -> list[np.ndarray]:
    """Inset a profile's outer/hole loops with topology validation.

    This helper centralizes profile inset behavior used by loft endcap
    generation so loop classification and containment checks stay in topology.
    """

    try:
        import pyclipper
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("pyclipper is required for profile inset operations.") from exc

    loops = profile_loops(
        shape,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
        enforce_winding=True,
    )
    if not loops:
        raise ValueError("Shape has no loops to inset.")
    if inset <= 0:
        return loops

    scale = 1_000_000.0
    join_key = "ROUND" if join_type.upper() in {"ROUND", "COVE"} else join_type.upper()
    jt = pyclipper.JT_ROUND if join_key == "ROUND" else pyclipper.JT_MITER

    def offset_single(path_pts: np.ndarray, delta: float) -> list[np.ndarray]:
        pco = pyclipper.PyclipperOffset(miter_limit=2.0, arc_tolerance=0.25 * scale)
        path = np.round(path_pts * scale).astype(np.int64).tolist()
        pco.AddPath(path, jt, pyclipper.ET_CLOSEDPOLYGON)
        result = pco.Execute(delta * scale)
        return [np.asarray(p, dtype=float) / scale for p in result]

    outer = loops[0]
    holes = loops[1:]

    outer_result = offset_single(outer, -inset)
    if len(outer_result) != 1:
        raise ValueError("endcap_amount too large for profile; inset collapsed.")
    outer = ensure_winding(outer_result[0], clockwise=False)

    inset_holes: list[np.ndarray] = []
    for hole in holes:
        hole_result = offset_single(hole, inset)
        if len(hole_result) != 1:
            raise ValueError("endcap_amount too large for profile; inset collapsed.")
        hole_loop = ensure_winding(hole_result[0], clockwise=True)
        if not point_in_polygon(hole_loop[0], outer):
            raise ValueError("endcap_amount too large for profile; hole collapsed.")
        inset_holes.append(hole_loop)

    if hole_count and len(inset_holes) != hole_count:
        raise ValueError("endcap_amount too large for profile; hole topology changed.")
    return [outer] + inset_holes


def sections_from_paths(paths: Iterable[Path2D]) -> list[Section]:
    """Assemble topology-native sections from a set of closed paths."""
    regions = regions_from_paths(paths)
    return [Section((region,)) for region in regions]


def regions_from_paths(paths: Iterable[Path2D]) -> list[Region]:
    """Assemble topology-native regions from a set of closed paths."""

    info = []
    for path in paths:
        pts = _normalize_loop_points(path.sample())
        if pts.shape[0] < 3:
            continue
        area = abs(signed_area(pts))
        info.append({"path": path, "pts": pts, "area": area, "holes": []})
    info.sort(key=lambda item: item["area"], reverse=True)

    outers = []
    for item in info:
        candidate = None
        for outer in outers:
            if point_in_polygon(item["pts"][0], outer["pts"]):
                if candidate is None or outer["area"] < candidate["area"]:
                    candidate = outer
        if candidate is None:
            outers.append(item)
        else:
            candidate["holes"].append(item["path"])
    regions: list[Region] = []
    for outer in outers:
        outer_path = outer["path"]
        holes_paths = outer.get("holes", [])
        outer_pts = _normalize_loop_points(outer_path.sample())
        outer_loop = Loop(ensure_winding(outer_pts, clockwise=False))
        hole_loops = []
        for hole_path in holes_paths:
            hole_pts = _normalize_loop_points(hole_path.sample())
            hole_loops.append(Loop(ensure_winding(hole_pts, clockwise=True)))
        regions.append(Region(outer=outer_loop, holes=tuple(hole_loops)).normalized())
    return regions


def minimum_cost_loop_assignment(
    source_loops: list[np.ndarray],
    target_loops: list[np.ndarray],
    *,
    area_weight: float = 0.1,
) -> tuple[int, ...]:
    """Return deterministic one-to-one loop correspondence.

    Primary score: centroid distance + weighted area delta.
    Tie-break order is lexicographic:
    1) lower total primary score
    2) lower total centroid distance
    3) lower total area delta
    4) lower target-index tuple in source order
    """

    if len(source_loops) != len(target_loops):
        raise ValueError("minimum_cost_loop_assignment requires equal source/target counts.")
    count = len(source_loops)
    if count == 0:
        return ()
    if count > 12:
        raise ValueError("Too many loops for deterministic assignment.")

    src_centroids = [np.asarray(loop, dtype=float).mean(axis=0) for loop in source_loops]
    dst_centroids = [np.asarray(loop, dtype=float).mean(axis=0) for loop in target_loops]
    src_areas = [abs(signed_area(loop)) for loop in source_loops]
    dst_areas = [abs(signed_area(loop)) for loop in target_loops]

    dist = np.zeros((count, count), dtype=float)
    area = np.zeros((count, count), dtype=float)
    primary = np.zeros((count, count), dtype=float)
    for i in range(count):
        for j in range(count):
            d = float(np.linalg.norm(src_centroids[i] - dst_centroids[j]))
            a = abs(src_areas[i] - dst_areas[j])
            dist[i, j] = d
            area[i, j] = a
            primary[i, j] = d + area_weight * a

    @lru_cache(maxsize=None)
    def solve(i: int, used_mask: int) -> tuple[tuple[float, float, float, tuple[int, ...]], tuple[int, ...]]:
        if i == count:
            zero = (0.0, 0.0, 0.0, ())
            return zero, ()

        best_score: tuple[float, float, float, tuple[int, ...]] | None = None
        best_order: tuple[int, ...] | None = None
        for j in range(count):
            if used_mask & (1 << j):
                continue
            child_score, child_order = solve(i + 1, used_mask | (1 << j))
            score = (
                primary[i, j] + child_score[0],
                dist[i, j] + child_score[1],
                area[i, j] + child_score[2],
                (j, *child_score[3]),
            )
            order = (j, *child_order)
            if best_score is None or score < best_score:
                best_score = score
                best_order = order

        if best_score is None or best_order is None:
            raise ValueError("Failed to compute deterministic loop assignment.")
        return best_score, best_order

    _, assignment = solve(0, 0)
    return assignment


def minimum_cost_subset_assignment(
    source_loops: list[np.ndarray],
    target_loops: list[np.ndarray],
    *,
    area_weight: float = 0.1,
) -> tuple[int, ...]:
    """Return deterministic source->target assignment where len(source) <= len(target)."""

    if len(source_loops) > len(target_loops):
        raise ValueError("subset assignment expects source count <= target count.")
    count = len(source_loops)
    if count == 0:
        return ()
    if len(target_loops) > 12:
        raise ValueError("Too many loops for deterministic subset assignment.")

    src_centroids = [np.asarray(loop, dtype=float).mean(axis=0) for loop in source_loops]
    dst_centroids = [np.asarray(loop, dtype=float).mean(axis=0) for loop in target_loops]
    src_areas = [abs(signed_area(loop)) for loop in source_loops]
    dst_areas = [abs(signed_area(loop)) for loop in target_loops]

    dist = np.zeros((count, len(target_loops)), dtype=float)
    area = np.zeros((count, len(target_loops)), dtype=float)
    primary = np.zeros((count, len(target_loops)), dtype=float)
    for i in range(count):
        for j in range(len(target_loops)):
            d = float(np.linalg.norm(src_centroids[i] - dst_centroids[j]))
            a = abs(src_areas[i] - dst_areas[j])
            dist[i, j] = d
            area[i, j] = a
            primary[i, j] = d + area_weight * a

    @lru_cache(maxsize=None)
    def solve(i: int, used_mask: int) -> tuple[tuple[float, float, float, tuple[int, ...]], tuple[int, ...]]:
        if i == count:
            zero = (0.0, 0.0, 0.0, ())
            return zero, ()

        best_score: tuple[float, float, float, tuple[int, ...]] | None = None
        best_order: tuple[int, ...] | None = None
        for j in range(len(target_loops)):
            if used_mask & (1 << j):
                continue
            child_score, child_order = solve(i + 1, used_mask | (1 << j))
            score = (
                primary[i, j] + child_score[0],
                dist[i, j] + child_score[1],
                area[i, j] + child_score[2],
                (j, *child_score[3]),
            )
            order = (j, *child_order)
            if best_score is None or score < best_score:
                best_score = score
                best_order = order
        if best_score is None or best_order is None:
            raise ValueError("Failed to compute deterministic subset assignment.")
        return best_score, best_order

    _, assignment = solve(0, 0)
    return assignment


def loop_bbox(loop: np.ndarray) -> tuple[float, float, float, float]:
    pts = np.asarray(loop, dtype=float)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])


def bbox_area(box: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_overlap_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = min(ax1, bx1) - max(ax0, bx0)
    dy = min(ay1, by1) - max(ay0, by0)
    if dx <= 0 or dy <= 0:
        return 0.0
    return float(dx * dy)


def loop_span(loop: np.ndarray) -> float:
    x0, y0, x1, y1 = loop_bbox(loop)
    return float(np.hypot(x1 - x0, y1 - y0))


def stable_loop_transition(source_loop: np.ndarray, target_loop: np.ndarray) -> bool:
    source = np.asarray(source_loop, dtype=float)
    target = np.asarray(target_loop, dtype=float)
    src_centroid = source.mean(axis=0)
    dst_centroid = target.mean(axis=0)

    if point_in_polygon(src_centroid, target) or point_in_polygon(dst_centroid, source):
        return True
    if bbox_overlap_area(loop_bbox(source), loop_bbox(target)) > 1e-9:
        return True

    src_span = loop_span(source)
    dst_span = loop_span(target)
    dist = float(np.linalg.norm(src_centroid - dst_centroid))
    scale = max(src_span, dst_span, 1e-9)
    return dist <= (1.5 * scale)


def split_merge_ambiguous(a_loop: np.ndarray, b_loop: np.ndarray) -> bool:
    a = np.asarray(a_loop, dtype=float)
    b = np.asarray(b_loop, dtype=float)
    a_centroid = a.mean(axis=0)
    b_centroid = b.mean(axis=0)
    if point_in_polygon(a_centroid, b) or point_in_polygon(b_centroid, a):
        return True

    a_box = loop_bbox(a)
    b_box = loop_bbox(b)
    overlap = bbox_overlap_area(a_box, b_box)
    if overlap <= 0:
        return False
    min_box_area = max(min(bbox_area(a_box), bbox_area(b_box)), 1e-12)
    overlap_ratio = overlap / min_box_area
    return overlap_ratio >= 0.25


__all__ = [
    "Loop",
    "GeneratedRailProvenance",
    "Region",
    "Section",
    "TopologyLandmark",
    "TopologyLifecycleBuilderRequest",
    "TopologyPath",
    "TopologyPathBuilder",
    "TopologyPathSamplingPolicy",
    "TopologyPoint",
    "TopologySegment",
    "as_section",
    "as_sections",
    "anchor_loop",
    "classify_loops",
    "derive_stable_id",
    "inset_profile_loops",
    "ensure_winding",
    "largest_loop",
    "loop_points",
    "loops_resampled",
    "point_in_polygon",
    "minimum_cost_loop_assignment",
    "minimum_cost_subset_assignment",
    "stable_loop_transition",
    "split_merge_ambiguous",
    "loop_bbox",
    "loop_span",
    "bbox_area",
    "bbox_overlap_area",
    "profile_loops",
    "regions_from_paths",
    "sections_from_paths",
    "resample_loop",
    "signed_area",
    "triangulate_loops",
    "triangulate_profile",
    "validate_topology_identity_records",
]

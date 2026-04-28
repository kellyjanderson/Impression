from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from .path3d import Arc3D, Bezier3D, Line3D, Path3D

ProgressionProvenanceKind = Literal["explicit", "inferred"]


class _StationLike(Protocol):
    @property
    def progression(self) -> float: ...

    @property
    def topology_state(self) -> object | None: ...

    @property
    def placement_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...

    @property
    def directional_correspondence(self) -> tuple[dict[str, frozenset[str]], ...]: ...


def _normalize_progression_provenance_kind(value: str) -> ProgressionProvenanceKind:
    allowed: set[str] = {"explicit", "inferred"}
    kind = str(value)
    if kind not in allowed:
        raise ValueError("kind must be one of: explicit, inferred.")
    return kind  # type: ignore[return-value]


def _segment_signature(segment: Line3D | Arc3D | Bezier3D) -> tuple[object, ...]:
    if isinstance(segment, Line3D):
        return (
            "line",
            tuple(float(v) for v in segment.start),
            tuple(float(v) for v in segment.end),
        )
    if isinstance(segment, Arc3D):
        return (
            "arc",
            tuple(float(v) for v in segment.center),
            float(segment.radius),
            float(segment.start_angle_deg),
            float(segment.end_angle_deg),
            tuple(float(v) for v in segment.normal),
            bool(segment.clockwise),
        )
    return (
        "bezier",
        tuple(float(v) for v in segment.p0),
        tuple(float(v) for v in segment.p1),
        tuple(float(v) for v in segment.p2),
        tuple(float(v) for v in segment.p3),
    )


@dataclass(frozen=True)
class ProgressionProvenanceRecord:
    kind: ProgressionProvenanceKind = "explicit"
    source: str = "authored_path"

    def __init__(
        self,
        kind: ProgressionProvenanceKind = "explicit",
        source: str = "authored_path",
    ) -> None:
        normalized_kind = _normalize_progression_provenance_kind(kind)
        normalized_source = str(source).strip()
        if not normalized_source:
            raise ValueError("source must be non-empty.")
        object.__setattr__(self, "kind", normalized_kind)
        object.__setattr__(self, "source", normalized_source)


@dataclass(frozen=True)
class PathBackedProgression:
    path: Path3D
    domain_start: float = 0.0
    domain_end: float = 1.0
    provenance: ProgressionProvenanceRecord = ProgressionProvenanceRecord()

    def __init__(
        self,
        *,
        path: Path3D,
        domain_start: float = 0.0,
        domain_end: float = 1.0,
        provenance: ProgressionProvenanceRecord | None = None,
    ) -> None:
        if not isinstance(path, Path3D):
            raise ValueError("path must be a Path3D.")
        start = float(domain_start)
        end = float(domain_end)
        if not np.isfinite(start):
            raise ValueError("domain_start must be finite.")
        if not np.isfinite(end):
            raise ValueError("domain_end must be finite.")
        if end <= start:
            raise ValueError("domain_end must be greater than domain_start.")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "domain_start", start)
        object.__setattr__(self, "domain_end", end)
        object.__setattr__(
            self,
            "provenance",
            provenance if provenance is not None else ProgressionProvenanceRecord(),
        )

    @property
    def parameter_domain(self) -> tuple[float, float]:
        return (self.domain_start, self.domain_end)

    @property
    def path_signature(self) -> tuple[object, ...]:
        return (
            "path3d",
            bool(self.path.closed),
            tuple(_segment_signature(segment) for segment in self.path.segments),
        )

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            "path_backed_progression",
            self.path_signature,
            self.parameter_domain,
            self.provenance,
        )

    def attach_stations(
        self,
        stations: list[_StationLike] | tuple[_StationLike, ...],
    ) -> tuple["ProgressionStationAttachment", ...]:
        attachments: list[ProgressionStationAttachment] = []
        last_progression: float | None = None
        for station_index, station in enumerate(stations):
            attachment = ProgressionStationAttachment.from_station(
                progression=self,
                station=station,
                station_index=station_index,
            )
            if last_progression is not None and attachment.progression_value < last_progression:
                raise ValueError("station attachments must remain ordered by progression value.")
            attachments.append(attachment)
            last_progression = attachment.progression_value
        return tuple(attachments)


@dataclass(frozen=True)
class ProgressionStationAttachment:
    progression_identity: tuple[object, ...]
    station_index: int
    progression_value: float
    topology_state: object | None
    placement_frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    directional_correspondence: tuple[dict[str, frozenset[str]], ...]

    @classmethod
    def from_station(
        cls,
        *,
        progression: PathBackedProgression,
        station: _StationLike,
        station_index: int,
    ) -> "ProgressionStationAttachment":
        progression_value = float(station.progression)
        if not np.isfinite(progression_value):
            raise ValueError("station progression value must be finite.")
        if not progression.domain_start <= progression_value <= progression.domain_end:
            raise ValueError("station progression value must lie within the progression domain.")
        placement_frame = tuple(
            np.asarray(component, dtype=float).reshape(3)
            for component in station.placement_frame
        )
        return cls(
            progression_identity=progression.identity,
            station_index=int(station_index),
            progression_value=progression_value,
            topology_state=station.topology_state,
            placement_frame=placement_frame,
            directional_correspondence=station.directional_correspondence,
        )

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            "progression_station_attachment",
            self.progression_identity,
            self.station_index,
            self.progression_value,
        )


__all__ = [
    "PathBackedProgression",
    "ProgressionStationAttachment",
    "ProgressionProvenanceKind",
    "ProgressionProvenanceRecord",
]

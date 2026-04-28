from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .path3d import Arc3D, Bezier3D, Line3D, Path3D

ProgressionProvenanceKind = Literal["explicit", "inferred"]


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


__all__ = [
    "PathBackedProgression",
    "ProgressionProvenanceKind",
    "ProgressionProvenanceRecord",
]

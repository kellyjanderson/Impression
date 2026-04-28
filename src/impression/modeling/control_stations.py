from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

HiddenControlStationSource = Literal["dense_station_fit", "shared_trajectory_fit"]


class _StationLike(Protocol):
    @property
    def topology_state(self) -> object | None: ...

    @property
    def placement_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


def _normalize_hidden_control_station_source(value: str) -> HiddenControlStationSource:
    allowed: set[str] = {"dense_station_fit", "shared_trajectory_fit"}
    source = str(value)
    if source not in allowed:
        raise ValueError("source must be one of: dense_station_fit, shared_trajectory_fit.")
    return source  # type: ignore[return-value]


@dataclass(frozen=True)
class HiddenControlStationProvenanceRecord:
    source: HiddenControlStationSource = "dense_station_fit"
    evidence_reference: str = "fit_candidate"

    def __init__(
        self,
        source: HiddenControlStationSource = "dense_station_fit",
        evidence_reference: str = "fit_candidate",
    ) -> None:
        normalized_source = _normalize_hidden_control_station_source(source)
        normalized_reference = str(evidence_reference).strip()
        if not normalized_reference:
            raise ValueError("evidence_reference must be non-empty.")
        object.__setattr__(self, "source", normalized_source)
        object.__setattr__(self, "evidence_reference", normalized_reference)


@dataclass(frozen=True)
class HiddenControlStationRecord:
    station_id: str
    origin: tuple[float, float, float]
    u: tuple[float, float, float]
    v: tuple[float, float, float]
    n: tuple[float, float, float]
    topology_reference: object | None
    provenance: HiddenControlStationProvenanceRecord

    def __init__(
        self,
        *,
        station_id: str,
        origin: tuple[float, float, float] | np.ndarray,
        u: tuple[float, float, float] | np.ndarray,
        v: tuple[float, float, float] | np.ndarray,
        n: tuple[float, float, float] | np.ndarray,
        topology_reference: object | None,
        provenance: HiddenControlStationProvenanceRecord,
    ) -> None:
        normalized_station_id = str(station_id).strip()
        if not normalized_station_id:
            raise ValueError("station_id must be non-empty.")
        object.__setattr__(self, "station_id", normalized_station_id)
        object.__setattr__(self, "origin", tuple(float(value) for value in np.asarray(origin, dtype=float).reshape(3)))
        object.__setattr__(self, "u", tuple(float(value) for value in np.asarray(u, dtype=float).reshape(3)))
        object.__setattr__(self, "v", tuple(float(value) for value in np.asarray(v, dtype=float).reshape(3)))
        object.__setattr__(self, "n", tuple(float(value) for value in np.asarray(n, dtype=float).reshape(3)))
        object.__setattr__(self, "topology_reference", topology_reference)
        object.__setattr__(self, "provenance", provenance)

    @classmethod
    def from_station(
        cls,
        *,
        station_id: str,
        station: _StationLike,
        provenance: HiddenControlStationProvenanceRecord,
    ) -> "HiddenControlStationRecord":
        origin, u, v, n = station.placement_frame
        return cls(
            station_id=station_id,
            origin=origin,
            u=u,
            v=v,
            n=n,
            topology_reference=station.topology_state,
            provenance=provenance,
        )

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            "hidden_control_station",
            self.station_id,
            self.origin,
            self.provenance,
        )


__all__ = [
    "HiddenControlStationProvenanceRecord",
    "HiddenControlStationRecord",
    "HiddenControlStationSource",
]

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

HiddenControlStationSource = Literal["dense_station_fit", "shared_trajectory_fit"]
HiddenControlStationPlannerStage = Literal["fit_guidance", "trajectory_guidance"]


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


def _normalize_hidden_control_station_planner_stage(value: str) -> HiddenControlStationPlannerStage:
    allowed: set[str] = {"fit_guidance", "trajectory_guidance"}
    stage = str(value)
    if stage not in allowed:
        raise ValueError("planner_stage must be one of: fit_guidance, trajectory_guidance.")
    return stage  # type: ignore[return-value]


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


@dataclass(frozen=True)
class HiddenControlStationPlannerConsumption:
    planner_stage: HiddenControlStationPlannerStage
    topology_station_ids: tuple[str, ...]
    hidden_control_station_ids: tuple[str, ...]
    public_authored_inputs_exposed: bool = False

    def __init__(
        self,
        *,
        planner_stage: HiddenControlStationPlannerStage,
        topology_station_ids: tuple[str, ...] | list[str],
        hidden_control_station_ids: tuple[str, ...] | list[str],
        public_authored_inputs_exposed: bool = False,
    ) -> None:
        normalized_topology = tuple(str(value).strip() for value in topology_station_ids)
        normalized_hidden = tuple(str(value).strip() for value in hidden_control_station_ids)
        if any(not value for value in normalized_topology):
            raise ValueError("topology_station_ids must be non-empty strings.")
        if any(not value for value in normalized_hidden):
            raise ValueError("hidden_control_station_ids must be non-empty strings.")
        object.__setattr__(self, "planner_stage", _normalize_hidden_control_station_planner_stage(planner_stage))
        object.__setattr__(self, "topology_station_ids", normalized_topology)
        object.__setattr__(self, "hidden_control_station_ids", normalized_hidden)
        object.__setattr__(self, "public_authored_inputs_exposed", bool(public_authored_inputs_exposed))
        if self.public_authored_inputs_exposed:
            raise ValueError("hidden control station consumption must remain non-user-facing.")
        overlap = set(self.topology_station_ids) & set(self.hidden_control_station_ids)
        if overlap:
            raise ValueError("hidden control stations must not override topology station identity.")

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            "hidden_control_station_planner_consumption",
            self.planner_stage,
            self.topology_station_ids,
            self.hidden_control_station_ids,
        )


__all__ = [
    "HiddenControlStationPlannerConsumption",
    "HiddenControlStationPlannerStage",
    "HiddenControlStationProvenanceRecord",
    "HiddenControlStationRecord",
    "HiddenControlStationSource",
]

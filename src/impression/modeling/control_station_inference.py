from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .control_stations import HiddenControlStationRecord
from .progression import PathBackedProgression, ProgressionStationAttachment

RetainedStationKind = Literal["topology", "hidden_control"]


def _normalize_retained_station_kind(value: str) -> RetainedStationKind:
    allowed: set[str] = {"topology", "hidden_control"}
    kind = str(value)
    if kind not in allowed:
        raise ValueError("kind must be one of: topology, hidden_control.")
    return kind  # type: ignore[return-value]


@dataclass(frozen=True)
class ReducedProgressionBundle:
    bundle_id: str
    progression_identity: tuple[object, ...]
    retained_progression_values: tuple[float, ...]
    hidden_control_station_ids: tuple[str, ...]
    provenance_source: str
    replay_version: str = "0.1.0.a"

    def __init__(
        self,
        *,
        bundle_id: str,
        progression_identity: tuple[object, ...],
        retained_progression_values: tuple[float, ...] | list[float],
        hidden_control_station_ids: tuple[str, ...] | list[str],
        provenance_source: str = "control_station_inference",
        replay_version: str = "0.1.0.a",
    ) -> None:
        normalized_bundle_id = str(bundle_id).strip()
        normalized_source = str(provenance_source).strip()
        normalized_version = str(replay_version).strip()
        if not normalized_bundle_id:
            raise ValueError("bundle_id must be non-empty.")
        if not normalized_source:
            raise ValueError("provenance_source must be non-empty.")
        if not normalized_version:
            raise ValueError("replay_version must be non-empty.")
        normalized_progression = tuple(float(value) for value in retained_progression_values)
        if len(normalized_progression) < 2:
            raise ValueError("retained_progression_values must contain at least two values.")
        if tuple(sorted(normalized_progression)) != normalized_progression:
            raise ValueError("retained_progression_values must remain ordered.")
        normalized_hidden = tuple(str(value).strip() for value in hidden_control_station_ids)
        if any(not value for value in normalized_hidden):
            raise ValueError("hidden_control_station_ids must be non-empty strings.")
        object.__setattr__(self, "bundle_id", normalized_bundle_id)
        object.__setattr__(self, "progression_identity", progression_identity)
        object.__setattr__(self, "retained_progression_values", normalized_progression)
        object.__setattr__(self, "hidden_control_station_ids", normalized_hidden)
        object.__setattr__(self, "provenance_source", normalized_source)
        object.__setattr__(self, "replay_version", normalized_version)

    @classmethod
    def from_progression(
        cls,
        *,
        bundle_id: str,
        progression: PathBackedProgression,
        retained_progression_values: tuple[float, ...] | list[float],
        hidden_control_station_ids: tuple[str, ...] | list[str],
        provenance_source: str = "control_station_inference",
    ) -> "ReducedProgressionBundle":
        return cls(
            bundle_id=bundle_id,
            progression_identity=progression.identity,
            retained_progression_values=retained_progression_values,
            hidden_control_station_ids=hidden_control_station_ids,
            provenance_source=provenance_source,
        )

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            "reduced_progression_bundle",
            self.bundle_id,
            self.progression_identity,
            self.retained_progression_values,
        )

    @property
    def replay_payload(self) -> tuple[object, ...]:
        return (
            "reduced_progression_bundle",
            self.bundle_id,
            self.replay_version,
            self.progression_identity,
            self.retained_progression_values,
            self.hidden_control_station_ids,
            self.provenance_source,
        )


@dataclass(frozen=True)
class RetainedStationRecord:
    station_id: str
    kind: RetainedStationKind
    progression_value: float
    diagnostic_references: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        station_id: str,
        kind: RetainedStationKind,
        progression_value: float,
        diagnostic_references: tuple[str, ...] | list[str] = (),
    ) -> None:
        normalized_station_id = str(station_id).strip()
        normalized_diagnostics = tuple(str(value).strip() for value in diagnostic_references)
        if not normalized_station_id:
            raise ValueError("station_id must be non-empty.")
        if any(not value for value in normalized_diagnostics):
            raise ValueError("diagnostic_references must be non-empty strings.")
        object.__setattr__(self, "station_id", normalized_station_id)
        object.__setattr__(self, "kind", _normalize_retained_station_kind(kind))
        object.__setattr__(self, "progression_value", float(progression_value))
        object.__setattr__(self, "diagnostic_references", normalized_diagnostics)

    @classmethod
    def from_attachment(
        cls,
        *,
        station_id: str,
        attachment: ProgressionStationAttachment,
        diagnostic_references: tuple[str, ...] | list[str] = (),
    ) -> "RetainedStationRecord":
        return cls(
            station_id=station_id,
            kind="topology",
            progression_value=attachment.progression_value,
            diagnostic_references=diagnostic_references,
        )

    @classmethod
    def from_hidden_control(
        cls,
        *,
        record: HiddenControlStationRecord,
        progression_value: float,
        diagnostic_references: tuple[str, ...] | list[str] = (),
    ) -> "RetainedStationRecord":
        return cls(
            station_id=record.station_id,
            kind="hidden_control",
            progression_value=progression_value,
            diagnostic_references=diagnostic_references,
        )

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            "retained_station_record",
            self.station_id,
            self.kind,
            self.progression_value,
        )


__all__ = [
    "ReducedProgressionBundle",
    "RetainedStationKind",
    "RetainedStationRecord",
]

from __future__ import annotations

from dataclasses import dataclass

from .progression import PathBackedProgression


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


__all__ = [
    "ReducedProgressionBundle",
]

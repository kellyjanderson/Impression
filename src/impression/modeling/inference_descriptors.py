from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class _StationLike(Protocol):
    @property
    def progression(self) -> float: ...

    @property
    def topology_state(self) -> object | None: ...

    @property
    def placement_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...

    @property
    def directional_correspondence(self) -> tuple[dict[str, frozenset[str]], ...]: ...


@dataclass(frozen=True)
class DenseLoftStationDescriptor:
    station_index: int
    progression_value: float
    origin: tuple[float, float, float]
    region_count: int
    directional_correspondence_count: int

    @classmethod
    def from_station(
        cls,
        *,
        station_index: int,
        station: _StationLike,
    ) -> "DenseLoftStationDescriptor":
        origin, _, _, _ = station.placement_frame
        topology_state = station.topology_state
        region_count = 0 if topology_state is None else len(topology_state.regions)
        return cls(
            station_index=int(station_index),
            progression_value=float(station.progression),
            origin=tuple(float(value) for value in np.asarray(origin, dtype=float).reshape(3)),
            region_count=int(region_count),
            directional_correspondence_count=len(station.directional_correspondence),
        )

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            "dense_loft_station_descriptor",
            self.station_index,
            self.progression_value,
            self.origin,
            self.region_count,
            self.directional_correspondence_count,
        )


@dataclass(frozen=True)
class DenseLoftDescriptorBand:
    descriptors: tuple[DenseLoftStationDescriptor, ...]

    @property
    def station_indices(self) -> tuple[int, ...]:
        return tuple(descriptor.station_index for descriptor in self.descriptors)

    @property
    def progression_values(self) -> tuple[float, ...]:
        return tuple(descriptor.progression_value for descriptor in self.descriptors)


@dataclass(frozen=True)
class SectionCurveIntentDescriptor:
    station_index: int
    progression_value: float
    region_count: int


@dataclass(frozen=True)
class LoopCurveIntentDescriptor:
    station_index: int
    loop_count: int


@dataclass(frozen=True)
class CorrespondenceTrackDescriptor:
    station_index: int
    correspondence_track_count: int


@dataclass(frozen=True)
class CurveIntentDescriptorFamilies:
    section_descriptors: tuple[SectionCurveIntentDescriptor, ...]
    loop_descriptors: tuple[LoopCurveIntentDescriptor, ...]
    correspondence_track_descriptors: tuple[CorrespondenceTrackDescriptor, ...]


@dataclass(frozen=True)
class SpanLocalCurveIntentEvidence:
    span_start_station_index: int
    span_end_station_index: int
    section_region_counts: tuple[int, ...]
    loop_counts: tuple[int, ...]
    correspondence_track_counts: tuple[int, ...]


def prepare_dense_loft_fit_descriptors(
    stations: list[_StationLike] | tuple[_StationLike, ...],
) -> DenseLoftDescriptorBand:
    descriptors = tuple(
        DenseLoftStationDescriptor.from_station(
            station_index=station_index,
            station=station,
        )
        for station_index, station in enumerate(stations)
    )
    return DenseLoftDescriptorBand(descriptors=descriptors)


def build_curve_intent_descriptor_families(
    descriptor_band: DenseLoftDescriptorBand,
) -> CurveIntentDescriptorFamilies:
    section_descriptors = tuple(
        SectionCurveIntentDescriptor(
            station_index=descriptor.station_index,
            progression_value=descriptor.progression_value,
            region_count=descriptor.region_count,
        )
        for descriptor in descriptor_band.descriptors
    )
    loop_descriptors = tuple(
        LoopCurveIntentDescriptor(
            station_index=descriptor.station_index,
            loop_count=descriptor.region_count,
        )
        for descriptor in descriptor_band.descriptors
    )
    correspondence_track_descriptors = tuple(
        CorrespondenceTrackDescriptor(
            station_index=descriptor.station_index,
            correspondence_track_count=descriptor.directional_correspondence_count,
        )
        for descriptor in descriptor_band.descriptors
    )
    return CurveIntentDescriptorFamilies(
        section_descriptors=section_descriptors,
        loop_descriptors=loop_descriptors,
        correspondence_track_descriptors=correspondence_track_descriptors,
    )


def assemble_span_local_curve_intent_evidence(
    descriptor_families: CurveIntentDescriptorFamilies,
) -> tuple[SpanLocalCurveIntentEvidence, ...]:
    if not descriptor_families.section_descriptors:
        return ()
    evidence: list[SpanLocalCurveIntentEvidence] = []
    for left, right in zip(
        descriptor_families.section_descriptors,
        descriptor_families.section_descriptors[1:],
        strict=False,
    ):
        start_index = left.station_index
        end_index = right.station_index
        loop_slice = descriptor_families.loop_descriptors[start_index : end_index + 1]
        track_slice = descriptor_families.correspondence_track_descriptors[start_index : end_index + 1]
        evidence.append(
            SpanLocalCurveIntentEvidence(
                span_start_station_index=start_index,
                span_end_station_index=end_index,
                section_region_counts=(left.region_count, right.region_count),
                loop_counts=tuple(descriptor.loop_count for descriptor in loop_slice),
                correspondence_track_counts=tuple(
                    descriptor.correspondence_track_count for descriptor in track_slice
                ),
            )
        )
    return tuple(evidence)


__all__ = [
    "CorrespondenceTrackDescriptor",
    "CurveIntentDescriptorFamilies",
    "DenseLoftDescriptorBand",
    "DenseLoftStationDescriptor",
    "LoopCurveIntentDescriptor",
    "SectionCurveIntentDescriptor",
    "SpanLocalCurveIntentEvidence",
    "assemble_span_local_curve_intent_evidence",
    "build_curve_intent_descriptor_families",
    "prepare_dense_loft_fit_descriptors",
]

from __future__ import annotations

from impression.modeling import (
    HiddenControlStationProvenanceRecord,
    HiddenControlStationRecord,
    Station,
    as_section,
)
from impression.modeling.drawing2d import make_rect


def test_hidden_control_station_records_remain_distinct_from_topology_station_records() -> None:
    station = Station(
        t=0.5,
        section=as_section(make_rect(size=(1.0, 1.0))),
        origin=(0.0, 0.0, 0.5),
        u=(1.0, 0.0, 0.0),
        v=(0.0, 1.0, 0.0),
        n=(0.0, 0.0, 1.0),
    )
    record = HiddenControlStationRecord.from_station(
        station_id="control-0",
        station=station,
        provenance=HiddenControlStationProvenanceRecord(),
    )

    assert record is not station
    assert record.topology_reference is station.topology_state


def test_provenance_metadata_is_carried_with_retained_hidden_control_stations() -> None:
    provenance = HiddenControlStationProvenanceRecord(
        source="shared_trajectory_fit",
        evidence_reference="trajectory_candidate_0",
    )
    station = Station(
        t=0.5,
        section=as_section(make_rect(size=(1.0, 1.0))),
        origin=(0.0, 0.0, 0.5),
        u=(1.0, 0.0, 0.0),
        v=(0.0, 1.0, 0.0),
        n=(0.0, 0.0, 1.0),
    )

    record = HiddenControlStationRecord.from_station(
        station_id="control-1",
        station=station,
        provenance=provenance,
    )

    assert record.provenance == provenance


def test_internal_representation_remains_planner_owned_and_non_user_facing() -> None:
    record = HiddenControlStationRecord(
        station_id="control-2",
        origin=(0.0, 0.0, 0.5),
        u=(1.0, 0.0, 0.0),
        v=(0.0, 1.0, 0.0),
        n=(0.0, 0.0, 1.0),
        topology_reference=None,
        provenance=HiddenControlStationProvenanceRecord(),
    )

    assert record.station_id == "control-2"
    assert not hasattr(record, "t")


def test_provenance_remains_durable_enough_for_later_diagnostics() -> None:
    first = HiddenControlStationProvenanceRecord(
        source="dense_station_fit",
        evidence_reference="fit_candidate_2",
    )
    second = HiddenControlStationProvenanceRecord(
        source="dense_station_fit",
        evidence_reference="fit_candidate_2",
    )

    assert first == second


def test_hidden_control_stations_do_not_collapse_into_stealth_public_authored_inputs() -> None:
    record = HiddenControlStationRecord(
        station_id="control-3",
        origin=(0.0, 0.0, 1.0),
        u=(1.0, 0.0, 0.0),
        v=(0.0, 1.0, 0.0),
        n=(0.0, 0.0, 1.0),
        topology_reference=None,
        provenance=HiddenControlStationProvenanceRecord(),
    )

    assert not hasattr(record, "section")
    assert record.identity[0] == "hidden_control_station"

import dataclasses

import pytest

from impression.modeling.loft import (
    PointLifecycleEvent,
    PointLifecycleState,
    SyntheticRegionLineage,
    SyntheticStationLineage,
    SyntheticSupportReference,
    TopologyPath,
    validate_point_lifecycle_event,
    validate_point_lifecycle_events,
)


def _event(**overrides: object) -> PointLifecycleEvent:
    values = {
        "id": "birth-a",
        "event_type": "point_birth",
        "station_interval": (0, 1),
        "loop_ref": "outer",
        "point_ref": "notch",
        "correspondence_id": "notch",
        "parent_span_ref": ("left", "right"),
        "source": "authored",
        "interpolation_policy": "linear_span",
        "diagnostics": {"source": "test"},
    }
    values.update(overrides)
    return PointLifecycleEvent(**values)


def test_lifecycle_states_are_available_and_birth_event_maps_stations() -> None:
    event = _event()

    assert {state.value for state in PointLifecycleState} == {
        "present",
        "birth",
        "death",
        "synthetic_birth_support",
        "synthetic_death_support",
        "inferred",
    }
    assert event.lifecycle_state_for_station(-1) == PointLifecycleState.INFERRED
    assert event.lifecycle_state_for_station(0) == PointLifecycleState.SYNTHETIC_BIRTH_SUPPORT
    assert event.lifecycle_state_for_station(1) == PointLifecycleState.BIRTH
    assert event.lifecycle_state_for_station(2) == PointLifecycleState.PRESENT


def test_death_event_maps_stations() -> None:
    event = _event(id="death-a", event_type="point_death")

    assert event.lifecycle_state_for_station(-1) == PointLifecycleState.PRESENT
    assert event.lifecycle_state_for_station(0) == PointLifecycleState.DEATH
    assert event.lifecycle_state_for_station(1) == PointLifecycleState.SYNTHETIC_DEATH_SUPPORT
    assert event.lifecycle_state_for_station(2) == PointLifecycleState.INFERRED


def test_invalid_lifecycle_event_type_and_required_fields_fail() -> None:
    with pytest.raises(ValueError, match="event_type"):
        _event(event_type="move_point")

    with pytest.raises(ValueError, match="station_interval"):
        _event(station_interval=(1, 1))

    with pytest.raises(ValueError, match="parent_span_ref"):
        _event(parent_span_ref=("left",))


def test_event_ids_are_unique_inside_lifecycle_event_collections() -> None:
    validate_point_lifecycle_events((_event(id="a"), _event(id="b", event_type="point_death")))

    with pytest.raises(ValueError, match="Duplicate PointLifecycleEvent id"):
        validate_point_lifecycle_events((_event(id="a"), _event(id="a", event_type="point_death")))


def test_diagnostic_provenance_and_serialization_shape_are_preserved() -> None:
    event = _event(diagnostics={"source": "builder.birth_span", "span_parameter": 0.5})

    validate_point_lifecycle_event(event)
    serialized = dataclasses.asdict(event)
    assert serialized["diagnostics"] == {"source": "builder.birth_span", "span_parameter": 0.5}
    assert serialized["parent_span_ref"] == ("left", "right")


def test_synthetic_support_reference_validates_and_normalizes() -> None:
    support = SyntheticSupportReference(
        id="support-a",
        source_event_id="birth-a",
        station_index=0,
        span_ref=("left", "right"),
        span_parameter=0.25,
        coordinates=(0.25, 0.0),
    )

    assert support.coordinates == (0.25, 0.0)
    with pytest.raises(ValueError, match="span_parameter"):
        SyntheticSupportReference(
            id="bad",
            source_event_id="birth-a",
            station_index=0,
            span_ref=("left", "right"),
            span_parameter=1.5,
            coordinates=(0.25, 0.0),
        )


def test_synthetic_station_lineage_is_frozen_complete_and_canonical() -> None:
    path = TopologyPath.from_points(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), id="shell-outer")
    region = SyntheticRegionLineage(
        identity="shell",
        prev_region_ref=("actual", 0),
        curr_region_ref=("actual", 1),
        predecessor_ids=frozenset({"shell"}),
        successor_ids=frozenset({"shell"}),
        loop_identities=("shell-outer",),
        predecessor_loop_ids=("shell-outer",),
        successor_loop_ids=("shell-outer",),
    )
    lineage = SyntheticStationLineage(
        identity="synthetic-station-0-1-1-of-2",
        source_interval=(0.0, 1.0),
        stage_index=0,
        stage_count=2,
        station_t=0.4,
        regions=(region,),
        topology_paths=(path,),
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        lineage.stage_index = 1  # type: ignore[misc]
    assert lineage.canonical_payload()["topology_path_ids"] == ("shell-outer",)
    assert lineage.canonical_payload()["regions"][0]["identity"] == "shell"


def test_synthetic_station_lineage_rejects_missing_and_conflicting_loop_paths() -> None:
    region = SyntheticRegionLineage(
        identity="shell",
        prev_region_ref=("actual", 0),
        curr_region_ref=("actual", 0),
        predecessor_ids=frozenset({"shell"}),
        successor_ids=frozenset({"shell"}),
        loop_identities=("shell-outer",),
        predecessor_loop_ids=("shell-outer",),
        successor_loop_ids=("shell-outer",),
    )

    with pytest.raises(ValueError, match="topology paths do not cover every loop"):
        SyntheticStationLineage(
            identity="missing-path",
            source_interval=(0.0, 1.0),
            stage_index=0,
            stage_count=1,
            station_t=0.5,
            regions=(region,),
            topology_paths=(),
        )

    other_path = TopologyPath.from_points(
        ((2.0, 0.0), (3.0, 0.0), (2.0, 1.0)),
        id="other-outer",
    )
    conflicting_region = dataclasses.replace(
        region,
        loop_identities=("other-outer",),
        predecessor_loop_ids=("other-outer",),
        successor_loop_ids=("other-outer",),
    )
    with pytest.raises(ValueError, match="duplicate region identity"):
        SyntheticStationLineage(
            identity="duplicate-region",
            source_interval=(0.0, 1.0),
            stage_index=0,
            stage_count=1,
            station_t=0.5,
            regions=(region, conflicting_region),
            topology_paths=(
                TopologyPath.from_points(
                    ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
                    id="shell-outer",
                ),
                other_path,
            ),
        )

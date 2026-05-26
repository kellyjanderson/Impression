import dataclasses

import pytest

from impression.modeling.loft import (
    PointLifecycleEvent,
    PointLifecycleState,
    SyntheticSupportReference,
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

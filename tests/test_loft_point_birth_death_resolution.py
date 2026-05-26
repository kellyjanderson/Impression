import pytest

from impression.modeling.loft import (
    PointLifecycleEvent,
    SyntheticSupportReference,
    locate_parent_span,
    resolve_authored_rails,
    resolve_point_birth_death_events,
)
from impression.modeling.topology import TopologyPath


def _path(points: tuple[tuple[str, tuple[float, float], str], ...]) -> TopologyPath:
    builder = TopologyPath.closed()
    for name, coordinates, correspondence_id in points:
        builder.point(name, coordinates, correspond=correspondence_id)
    return builder.build()


def test_rectangle_to_rounded_l_like_extra_points_create_birth_events() -> None:
    source = _path(
        (
            ("bottom-left", (0.0, 0.0), "bottom-left"),
            ("bottom-right", (2.0, 0.0), "bottom-right"),
            ("top-right", (2.0, 1.0), "top-right"),
            ("top-left", (0.0, 1.0), "top-left"),
        )
    )
    target = _path(
        (
            ("bottom-left", (0.0, 0.0), "bottom-left"),
            ("bottom-right", (2.0, 0.0), "bottom-right"),
            ("inner-corner", (2.0, 0.5), "inner-corner"),
            ("top-right", (2.0, 1.0), "top-right"),
            ("top-left", (0.0, 1.0), "top-left"),
        )
    )

    resolution = resolve_point_birth_death_events(
        {"source": source, "target": target, "station_interval": (0, 1), "loop_ref": "outer"},
        resolve_authored_rails(source, target),
        {"collapse_degeneracy": {"min_point_correspondence_span": 1e-6}},
    )

    assert resolution.accepted is True
    assert len(resolution.events) == 1
    assert isinstance(resolution.events[0], PointLifecycleEvent)
    assert resolution.events[0].event_type == "point_birth"
    assert resolution.events[0].parent_span_ref == ("bottom-right", "top-right")
    assert resolution.synthetic_supports[0].span_parameter == pytest.approx(0.5)
    assert resolution.synthetic_supports[0].coordinates == pytest.approx((2.0, 0.5))


def test_source_only_point_creates_death_event_and_target_support() -> None:
    source = _path(
        (
            ("a", (0.0, 0.0), "a"),
            ("dying", (0.5, 0.0), "dying"),
            ("b", (1.0, 0.0), "b"),
            ("c", (0.0, 1.0), "c"),
        )
    )
    target = _path((("a", (0.0, 0.0), "a"), ("b", (1.0, 0.0), "b"), ("c", (0.0, 1.0), "c")))

    resolution = resolve_point_birth_death_events(
        {"source": source, "target": target},
        resolve_authored_rails(source, target),
        None,
    )

    assert resolution.accepted is True
    assert resolution.events[0].event_type == "point_death"
    assert resolution.events[0].parent_span_ref == ("a", "b")
    assert isinstance(resolution.synthetic_supports[0], SyntheticSupportReference)
    assert resolution.synthetic_supports[0].station_index == 1


def test_ambiguous_parent_span_refuses_when_locating_span_directly() -> None:
    with pytest.raises(ValueError, match="ambiguous_parent_span"):
        locate_parent_span(
            "p",
            (
                ("a", "b", (0.0, 0.0), (1.0, 0.0), (0.5, 0.0)),
                ("c", "d", (0.0, 0.0), (1.0, 0.0), (0.5, 0.0)),
            ),
        )


def test_collapsed_span_refuses_through_tolerance_policy() -> None:
    source = _path((("a", (0.0, 0.0), "a"), ("b", (0.0, 0.0), "b"), ("c", (0.0, 1.0), "c")))
    target = _path(
        (
            ("a", (0.0, 0.0), "a"),
            ("born", (0.0, 0.0), "born"),
            ("b", (0.0, 0.0), "b"),
            ("c", (0.0, 1.0), "c"),
        )
    )

    resolution = resolve_point_birth_death_events(
        {"source": source, "target": target},
        resolve_authored_rails(source, target),
        {"collapse_degeneracy": {"min_point_correspondence_span": 0.1}},
    )

    assert resolution.accepted is False
    assert resolution.refusals[0].reason == "collapsed_parent_span"


def test_explicit_correspondence_conflict_refuses_lifecycle_resolution() -> None:
    source = _path((("a", (0.0, 0.0), "a"), ("b", (1.0, 0.0), "b"), ("c", (0.0, 1.0), "c")))
    target = _path(
        (
            ("a", (0.0, 0.0), "a"),
            ("conflict", (0.5, 0.0), "a"),
            ("b", (1.0, 0.0), "b"),
            ("c", (0.0, 1.0), "c"),
        )
    )

    resolution = resolve_point_birth_death_events(
        {"source": source, "target": target},
        resolve_authored_rails(source, target),
        None,
    )

    assert resolution.accepted is False
    assert resolution.refusals[0].reason == "explicit_correspondence_conflict"

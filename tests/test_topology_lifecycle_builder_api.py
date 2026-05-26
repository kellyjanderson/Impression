import pytest

from impression.modeling.topology import TopologyLifecycleBuilderRequest, TopologyPath


def test_birth_span_preserves_authored_point_order() -> None:
    path = (
        TopologyPath.closed()
        .point("left", (0.0, 0.0))
        .point("right", (2.0, 0.0))
        .point("top", (1.0, 1.0))
        .birth_span(("left", "right"), points=[("notch-left", (0.75, 0.0)), ("notch-right", (1.25, 0.0))])
        .build()
    )

    assert [point.name for point in path.points] == ["left", "right", "top", "notch-left", "notch-right"]
    request = path.metadata["lifecycle_requests"][0]
    assert isinstance(request, TopologyLifecycleBuilderRequest)
    assert request.request_type == "birth_span"
    assert request.parent == ("left", "right")
    assert request.points == (("notch-left", (0.75, 0.0)), ("notch-right", (1.25, 0.0)))


def test_birth_arc_records_curve_segment_and_landmarks() -> None:
    path = (
        TopologyPath.closed()
        .point("left", (0.0, 0.0))
        .point("right", (2.0, 0.0))
        .point("top", (1.0, 1.0))
        .birth_arc("rounded-notch", parent=("left", "right"), start=(0.75, 0.0), end=(1.25, 0.0), radius=0.25)
        .build()
    )

    request = path.metadata["lifecycle_requests"][0]
    assert request.request_type == "birth_arc"
    assert request.name == "rounded-notch"
    assert request.radius == 0.25
    assert request.curve["kind"] == "birth_arc"
    assert path.segments[0].id == "rounded-notch"
    assert [landmark.parameter for landmark in path.landmarks] == [0.0, 1.0]
    assert {point.name for point in path.points} >= {"rounded-notch-start", "rounded-notch-end"}


def test_death_span_records_parent_refs_and_target_names() -> None:
    path = (
        TopologyPath.closed()
        .point("left", (0.0, 0.0))
        .point("right", (2.0, 0.0))
        .point("top", (1.0, 1.0))
        .death_span(("left", "right"), names=["top"])
        .build()
    )

    request = path.metadata["lifecycle_requests"][0]
    assert request.request_type == "death_span"
    assert request.parent == ("left", "right")
    assert request.names == ("top",)


def test_unknown_lifecycle_parent_refs_fail_on_build() -> None:
    builder = (
        TopologyPath.closed()
        .point("left", (0.0, 0.0))
        .point("right", (2.0, 0.0))
        .point("top", (1.0, 1.0))
        .birth_span(("left", "missing"), points=[("born", (1.0, 0.0))])
    )

    with pytest.raises(ValueError, match="unknown point names"):
        builder.build()


def test_birth_arc_rejects_invalid_radius_and_endpoints() -> None:
    with pytest.raises(ValueError, match="radius"):
        TopologyPath.closed().birth_arc("bad", parent=("left", "right"), start=(0.0, 0.0), end=(1.0, 0.0), radius=0.0)

    with pytest.raises(ValueError, match="endpoints"):
        TopologyPath.closed().birth_arc("bad", parent=("left", "right"), start=(0.0, 0.0), end=(0.0, 0.0), radius=1.0)
